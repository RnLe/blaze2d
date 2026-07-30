//! Post-solve Rayleigh–Ritz refinement and eigenpair certification (B3).
//!
//! LOBPCG converges on eigenvalue stagnation and gives soft-locked bands zero
//! residual placeholders, so the `converged` flag is NOT an accuracy
//! certificate. This module performs, after LOBPCG convergence:
//!
//! 1. a final dense Rayleigh–Ritz rotation in the full returned block,
//! 2. FRESH operator applications A·u and B·u for every rotated band,
//! 3. normalized generalized residuals ‖Au − λBu‖ / (‖Au‖ + |λ|·‖Bu‖),
//! 4. the B-orthogonality defect max |⟨uᵢ|B|uⱼ⟩ − δᵢⱼ| of the returned block.
//!
//! Nothing here is ever silently zeroed: every returned band (including
//! soft-locked ones) gets a residual computed from fresh A/B applications.

use num_complex::{Complex, Complex64};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::{Field2D, Real};
use crate::operators::LinearOperator;

use super::dense;

/// Certification data for a returned eigenblock.
#[derive(Debug, Clone)]
pub struct BlockCertification {
    /// Fresh normalized generalized residual ‖Au − λBu‖/(‖Au‖+|λ|‖Bu‖) per
    /// band, in the (ascending) band order of the refined block.
    pub residuals: Vec<f64>,
    /// max_{ij} |⟨uᵢ|B|uⱼ⟩ − δᵢⱼ| over the returned block.
    pub b_orthogonality_defect: f64,
}

impl BlockCertification {
    pub fn empty() -> Self {
        Self {
            residuals: Vec::new(),
            b_orthogonality_defect: 0.0,
        }
    }
}

/// Perform the final dense Rayleigh–Ritz rotation on the returned eigenblock
/// (in place) and certify it with fresh residuals.
///
/// The block is B-orthonormalized via S^{-1/2} (S = UᴴBU), the projected
/// operator Hᵗ = S^{-1/2} UᴴAU S^{-1/2} is diagonalized densely, and the
/// rotated Ritz vectors replace `eigenvectors` with `eigenvalues` set to the
/// Ritz values (ascending — absolute band order). Residuals and the
/// B-orthogonality defect are then computed from FRESH A/B applications on
/// the rotated vectors.
pub fn rayleigh_ritz_certify<O, B>(
    operator: &mut O,
    eigenvalues: &mut [f64],
    eigenvectors: &mut [Field2D],
) -> BlockCertification
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let n = eigenvalues.len().min(eigenvectors.len());
    if n == 0 {
        return BlockCertification::empty();
    }

    // -- Load the block into backend buffers --
    let u: Vec<B::Buffer> = eigenvectors[..n]
        .iter()
        .map(|field| field_to_buffer::<O, B>(operator, field))
        .collect();

    // -- A·u and B·u for the projected matrices --
    let mut au: Vec<B::Buffer> = Vec::with_capacity(n);
    let mut bu: Vec<B::Buffer> = Vec::with_capacity(n);
    for x in &u {
        let mut a = operator.alloc_field();
        operator.apply(x, &mut a);
        au.push(a);
        let mut b = operator.alloc_field();
        operator.apply_mass(x, &mut b);
        bu.push(b);
    }

    // -- Projected matrices S = UᴴBU, H = UᴴAU (column-major) --
    let mut s_mat = vec![Complex64::ZERO; n * n];
    let mut h_mat = vec![Complex64::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            s_mat[i + j * n] = operator.backend().dot(&u[i], &bu[j]);
            h_mat[i + j * n] = operator.backend().dot(&u[i], &au[j]);
        }
    }
    hermitize_in_place(&mut s_mat, n);
    hermitize_in_place(&mut h_mat, n);

    // -- W = S^{-1/2} via the dense Hermitian eigendecomposition of S --
    let s_eig = dense::solve_hermitian_eigen(&s_mat, n);
    let mut w = vec![Complex64::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut acc = Complex64::ZERO;
            for k in 0..n {
                // Converged blocks have S ≈ I; the floor only guards a
                // pathologically rank-deficient block from producing NaNs.
                let s_k = s_eig.eigenvalues[k].max(1e-300);
                let inv_sqrt = 1.0 / s_k.sqrt();
                acc += s_eig.eigenvectors[i + k * n]
                    * inv_sqrt
                    * s_eig.eigenvectors[j + k * n].conj();
            }
            w[i + j * n] = acc;
        }
    }

    // -- H~ = Wᴴ H W = W H W (W Hermitian), then dense Rayleigh–Ritz --
    let hw = matmul_col_major(&h_mat, &w, n);
    let mut h_tilde = matmul_col_major(&w, &hw, n);
    hermitize_in_place(&mut h_tilde, n);
    let h_eig = dense::solve_hermitian_eigen(&h_tilde, n);

    // Full rotation Z = W · Y: u'_j = Σ_i Z[i,j] u_i.
    let z = matmul_col_major(&w, &h_eig.eigenvectors, n);

    // -- Rotate the block --
    let mut u_new: Vec<B::Buffer> = Vec::with_capacity(n);
    for j in 0..n {
        let mut acc = operator.alloc_field();
        zero_buffer::<B>(&mut acc);
        for i in 0..n {
            operator.backend().axpy(z[i + j * n], &u[i], &mut acc);
        }
        u_new.push(acc);
    }

    // -- FRESH applications on the rotated vectors --
    let mut au_new: Vec<B::Buffer> = Vec::with_capacity(n);
    let mut bu_new: Vec<B::Buffer> = Vec::with_capacity(n);
    for x in &u_new {
        let mut a = operator.alloc_field();
        operator.apply(x, &mut a);
        au_new.push(a);
        let mut b = operator.alloc_field();
        operator.apply_mass(x, &mut b);
        bu_new.push(b);
    }

    // -- Fresh normalized generalized residuals --
    let mut residuals = Vec::with_capacity(n);
    for j in 0..n {
        let lambda = h_eig.eigenvalues[j];
        let mut r = au_new[j].clone();
        operator
            .backend()
            .axpy(Complex64::new(-lambda, 0.0), &bu_new[j], &mut r);
        let num = operator.backend().dot(&r, &r).re.max(0.0).sqrt();
        let den = operator.backend().dot(&au_new[j], &au_new[j]).re.max(0.0).sqrt()
            + lambda.abs() * operator.backend().dot(&bu_new[j], &bu_new[j]).re.max(0.0).sqrt();
        residuals.push(if den > 0.0 { num / den } else { num });
    }

    // -- B-orthogonality defect of the rotated block --
    let mut defect = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let s_ij = operator.backend().dot(&u_new[i], &bu_new[j]);
            let target = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::ZERO
            };
            defect = defect.max((s_ij - target).norm());
        }
    }

    // -- Write the refined eigenpairs back --
    let grid = operator.grid();
    for j in 0..n {
        eigenvalues[j] = h_eig.eigenvalues[j];
        eigenvectors[j] = Field2D::from_vec(grid, buffer_to_f64_vec::<B>(&u_new[j]));
    }

    BlockCertification {
        residuals,
        b_orthogonality_defect: defect,
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn field_to_buffer<O, B>(operator: &O, field: &Field2D) -> B::Buffer
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut buffer = operator.alloc_field();
    for (dst, src) in buffer.as_mut_slice().iter_mut().zip(field.as_slice()) {
        *dst = Complex::new(
            <B::Real as Real>::from_accum(src.re),
            <B::Real as Real>::from_accum(src.im),
        );
    }
    buffer
}

fn buffer_to_f64_vec<B: SpectralBackend>(buffer: &B::Buffer) -> Vec<Complex64> {
    buffer
        .as_slice()
        .iter()
        .map(|c| Complex64::new(c.re.to_accum(), c.im.to_accum()))
        .collect()
}

fn zero_buffer<B: SpectralBackend>(buffer: &mut B::Buffer) {
    for value in buffer.as_mut_slice() {
        *value = Complex::new(
            <B::Real as Real>::from_accum(0.0),
            <B::Real as Real>::from_accum(0.0),
        );
    }
}

fn hermitize_in_place(matrix: &mut [Complex64], n: usize) {
    for j in 0..n {
        for i in 0..=j {
            let avg = 0.5 * (matrix[i + j * n] + matrix[j + i * n].conj());
            matrix[i + j * n] = avg;
            matrix[j + i * n] = avg.conj();
        }
    }
}

/// C = A·B for small column-major complex matrices.
fn matmul_col_major(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::ZERO; n * n];
    for j in 0..n {
        for k in 0..n {
            let b_kj = b[k + j * n];
            if b_kj == Complex64::ZERO {
                continue;
            }
            for i in 0..n {
                c[i + j * n] += a[i + k * n] * b_kj;
            }
        }
    }
    c
}
