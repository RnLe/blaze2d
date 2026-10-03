//! Eigenpair refinement and certification after solving.
//!
//! Convergence uses eigenvalue changes. Certification separately measures
//! normalized and absolute residuals with fresh operator applications, and
//! the B-orthogonality defect of the returned block.
//!
//! `rayleigh_ritz_certify` refines the returned eigenpairs through a dense
//! Rayleigh-Ritz projection. `certify_without_refinement` preserves the fields,
//! eigenvalues, and ordering supplied by the caller.

use num_complex::{Complex, Complex64};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::{Field2D, Real};
use crate::operators::LinearOperator;


/// Certification data for a returned eigenblock.
#[derive(Debug, Clone)]
pub struct BlockCertification {
    /// Fresh normalized generalized residual ‖Au − λBu‖/(‖Au‖+|λ|‖Bu‖) per
    /// band, in the (ascending) band order of the refined block.
    pub residuals: Vec<f64>,
    /// Fresh absolute eigenpair residual norm(Au - lambda Bu) / norm(u).
    pub absolute_residuals: Vec<f64>,
    /// max_{ij} |⟨uᵢ|B|uⱼ⟩ − δᵢⱼ| over the returned block.
    pub b_orthogonality_defect: f64,
}

impl BlockCertification {
    pub fn empty() -> Self {
        Self {
            residuals: Vec::new(),
            absolute_residuals: Vec::new(),
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
/// Ritz values (ascending; absolute band order). Residuals and the
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
    let grams = operator.backend().hermitian_gram_matrices(&u, &[bu.as_slice(), au.as_slice()]);
    let column_major = |g: &[Complex64]| (0..n * n).map(|k| g[(k % n) * n + k / n]).collect::<Vec<_>>();
    let mut s_mat = column_major(&grams[0]);
    let mut h_mat = column_major(&grams[1]);
    hermitize_in_place(&mut s_mat, n);
    hermitize_in_place(&mut h_mat, n);

    // -- W = S^{-1/2} via the dense Hermitian eigendecomposition of S --
    let s_eig = operator.backend().dense_eigen(&s_mat, n);
    let mut w = vec![Complex64::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut acc = Complex64::ZERO;
            for k in 0..n {
                // Converged blocks have S ≈ I; the floor only guards a
                // pathologically rank-deficient block from producing NaNs.
                let s_k = s_eig.eigenvalues[k].max(1e-300);
                let inv_sqrt = 1.0 / s_k.sqrt();
                acc +=
                    s_eig.eigenvectors[i + k * n] * inv_sqrt * s_eig.eigenvectors[j + k * n].conj();
            }
            w[i + j * n] = acc;
        }
    }

    // -- H~ = Wᴴ H W = W H W (W Hermitian), then dense Rayleigh–Ritz --
    let hw = matmul_col_major(&h_mat, &w, n);
    let mut h_tilde = matmul_col_major(&w, &hw, n);
    hermitize_in_place(&mut h_tilde, n);
    let h_eig = operator.backend().dense_eigen(&h_tilde, n);

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
    let mut absolute_residuals = Vec::with_capacity(n);
    for j in 0..n {
        let lambda = h_eig.eigenvalues[j];
        let mut r = au_new[j].clone();
        operator
            .backend()
            .axpy(Complex64::new(-lambda, 0.0), &bu_new[j], &mut r);
        let num = operator.backend().dot(&r, &r).re.max(0.0).sqrt();
        let den = operator
            .backend()
            .dot(&au_new[j], &au_new[j])
            .re
            .max(0.0)
            .sqrt()
            + lambda.abs()
                * operator
                    .backend()
                    .dot(&bu_new[j], &bu_new[j])
                    .re
                    .max(0.0)
                    .sqrt();
        residuals.push(if den > 0.0 { num / den } else { num });
        let norm = operator.backend().dot(&u_new[j], &u_new[j]).re.max(0.0).sqrt();
        absolute_residuals.push(if norm > 0.0 { num / norm } else { num });
    }

    // -- B-orthogonality defect of the rotated block --
    let defect = orthogonality_defect(&operator.backend().gram_matrix(&u_new, &bu_new), n);

    // -- Write the refined eigenpairs back --
    let grid = operator.grid();
    for j in 0..n {
        eigenvalues[j] = h_eig.eigenvalues[j];
        eigenvectors[j] = Field2D::from_vec(grid, buffer_to_f64_vec::<B>(&u_new[j]));
    }

    BlockCertification {
        residuals,
        absolute_residuals,
        b_orthogonality_defect: defect,
    }
}

/// Measure the supplied eigenpairs without rotating, sorting, or changing them.
/// This preserves band tracking while replacing cached iteration residuals with
/// fresh operator applications. The residual convention matches refinement.
pub fn certify_without_refinement<O, B>(
    operator: &mut O,
    eigenvalues: &[f64],
    eigenvectors: &[Field2D],
) -> BlockCertification
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    crate::profile_scope!("final_certification");
    let n = eigenvalues.len().min(eigenvectors.len());
    let fields: Vec<B::Buffer> = eigenvectors[..n]
        .iter()
        .map(|field| field_to_buffer::<O, B>(operator, field))
        .collect();
    let mut residuals = Vec::with_capacity(n);
    let mut absolute_residuals = Vec::with_capacity(n);
    let mut masses = Vec::with_capacity(n);
    for (j, field) in fields.iter().enumerate() {
        let mut a = operator.alloc_field();
        let mut b = operator.alloc_field();
        operator.apply(field, &mut a);
        operator.apply_mass(field, &mut b);
        let lambda = eigenvalues[j];
        let (_, [rr, aa, bb]) = operator.backend().residual(&a, &b, lambda);
        let denominator = aa.max(0.0).sqrt() + lambda.abs() * bb.max(0.0).sqrt();
        let numerator = rr.max(0.0).sqrt();
        let norm = operator.backend().dot(field, field).re.max(0.0).sqrt();
        absolute_residuals.push(if norm > 0.0 { numerator / norm } else { numerator });
        residuals.push(if denominator > 0.0 {
            numerator / denominator
        } else {
            numerator
        });
        masses.push(b);
    }
    let defect = orthogonality_defect(&operator.backend().gram_matrix(&fields, &masses), n);
    BlockCertification {
        residuals,
        absolute_residuals,
        b_orthogonality_defect: defect,
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Largest entry of |UᴴBU − I| from the row-major n×n Gram matrix.
fn orthogonality_defect(gram: &[Complex64], n: usize) -> f64 {
    gram.iter().enumerate().fold(0.0f64, |defect, (k, &s)| {
        let target = if k / n == k % n { Complex64::new(1.0, 0.0) } else { Complex64::ZERO };
        defect.max((s - target).norm())
    })
}

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
