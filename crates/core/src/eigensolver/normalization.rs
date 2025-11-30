//! Normalization and B-orthogonalization routines for the LOBPCG eigensolver.
//!
//! This module provides functions for B-orthonormalization of vectors and blocks,
//! including the robust SVQB (SVD-based Quasi-B-orthonormalization) algorithm
//! that handles near-degenerate eigenspaces gracefully.
//!
//! # Overview
//!
//! In the LOBPCG algorithm, we need to maintain B-orthonormal vectors at several points:
//! - Initial block X_0 creation
//! - After Rayleigh-Ritz projection
//! - When combining [X, W, P] subspaces
//!
//! The naive Gram-Schmidt approach fails near degeneracies because nearly parallel
//! vectors produce catastrophic cancellation. SVQB solves this by:
//! 1. Computing the Gram matrix G = X^H B X
//! 2. Performing eigendecomposition G = Q Σ Q^H
//! 3. Dropping directions with small singular values (rank revelation)
//! 4. Rescaling to produce B-orthonormal output
//!
//! # References
//!
//! - Duersch & Shao, "A Robust and Efficient Implementation of LOBPCG" (2018)
//! - Hetmaniuk & Lehoucq, "Basis selection in LOBPCG" (2006)

use num_complex::Complex64;

use crate::backend::{SpectralBackend, SpectralBuffer};

// ============================================================================
// Single-Vector Normalization
// ============================================================================

/// Compute the B-norm of a vector: ||x||_B = sqrt(<x, Bx>).
///
/// # Arguments
/// * `backend` - The spectral backend for linear algebra operations
/// * `vector` - The vector x
/// * `mass_vector` - The precomputed mass vector Bx
///
/// # Returns
/// The B-norm, guaranteed to be non-negative.
pub fn b_norm<B: SpectralBackend>(backend: &B, vector: &B::Buffer, mass_vector: &B::Buffer) -> f64 {
    backend.dot(vector, mass_vector).re.max(0.0).sqrt()
}

/// Normalize a single vector to unit B-norm in-place.
///
/// This scales both the vector and its precomputed mass vector by 1/||x||_B.
/// If the B-norm is below a threshold, the vector is left unchanged.
///
/// # Arguments
/// * `backend` - The spectral backend for linear algebra operations
/// * `vector` - The vector to normalize (modified in-place)
/// * `mass_vector` - The mass vector Bx (modified in-place)
///
/// # Returns
/// The original B-norm before normalization. Returns 0.0 if the norm was too small.
pub fn normalize_to_unit_b_norm<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass_vector: &mut B::Buffer,
) -> f64 {
    normalize_to_unit_b_norm_with_tol(backend, vector, mass_vector, 1e-15)
}

/// Normalize a single vector to unit B-norm with a specified tolerance.
///
/// # Arguments
/// * `backend` - The spectral backend for linear algebra operations
/// * `vector` - The vector to normalize (modified in-place)
/// * `mass_vector` - The mass vector Bx (modified in-place)
/// * `tol` - Minimum norm threshold; vectors with smaller norms are unchanged
///
/// # Returns
/// The original B-norm before normalization. Returns 0.0 if the norm was below tolerance.
pub fn normalize_to_unit_b_norm_with_tol<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass_vector: &mut B::Buffer,
    tol: f64,
) -> f64 {
    let norm = b_norm(backend, vector, mass_vector);
    if norm > tol {
        let scale = Complex64::new(1.0 / norm, 0.0);
        backend.scale(scale, vector);
        backend.scale(scale, mass_vector);
        norm
    } else {
        0.0
    }
}

// ============================================================================
// B-Inner Product and Projection
// ============================================================================

/// Compute the B-inner product <x, y>_B = x^H B y.
///
/// Note: This uses the sesquilinear (physics) convention where the first
/// argument is conjugated.
///
/// # Arguments
/// * `backend` - The spectral backend
/// * `x` - First vector
/// * `by` - Precomputed mass vector By
///
/// # Returns
/// The complex B-inner product.
pub fn b_inner_product<B: SpectralBackend>(
    backend: &B,
    x: &B::Buffer,
    by: &B::Buffer,
) -> Complex64 {
    backend.dot(x, by)
}

/// Project out the component of a vector along a B-normalized basis vector.
///
/// Computes: v := v - <v, u>_B * u
///
/// where u is assumed to be B-normalized (||u||_B = 1).
///
/// # Arguments
/// * `backend` - The spectral backend
/// * `vector` - Vector to modify (modified in-place)
/// * `mass_vector` - Mass vector Bv (modified in-place)
/// * `basis` - B-normalized basis vector u
/// * `basis_mass` - Mass vector Bu
pub fn project_out<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass_vector: &mut B::Buffer,
    basis: &B::Buffer,
    basis_mass: &B::Buffer,
) {
    // For sesquilinear B-inner product <x,y>_B = x^H B y:
    // The coefficient α = <v, u>_B needs to be conjugated to project properly
    // because we want v' = v - α*u such that <v', u>_B = 0
    let coeff = backend.dot(vector, basis_mass).conj();
    backend.axpy(-coeff, basis, vector);
    backend.axpy(-coeff, basis_mass, mass_vector);
}

// ============================================================================
// SVQB: SVD-based B-orthonormalization
// ============================================================================

/// Result of SVQB orthonormalization.
#[derive(Debug, Clone)]
pub struct SvqbResult {
    /// Number of vectors in the output (numerical rank).
    pub output_rank: usize,
    /// Number of vectors in the input.
    pub input_count: usize,
    /// Number of vectors dropped due to small singular values.
    pub dropped_count: usize,
    /// The singular values (before dropping).
    pub singular_values: Vec<f64>,
    /// Indices of kept vectors in the original ordering.
    pub kept_indices: Vec<usize>,
    /// Per-block drop counts for serial orthonormalization: (X_dropped, P_dropped, W_dropped).
    /// For batch mode, all drops are in W (third component).
    /// None if not tracked (legacy code paths).
    pub block_drops: Option<(usize, usize, usize)>,
}

impl SvqbResult {
    /// Check if any vectors were dropped.
    pub fn had_rank_deficiency(&self) -> bool {
        self.dropped_count > 0
    }

    /// Get the condition number estimate (ratio of largest to smallest kept singular value).
    pub fn condition_number(&self) -> f64 {
        if self.output_rank == 0 {
            return f64::INFINITY;
        }
        let max_sv = self
            .singular_values
            .iter()
            .take(self.output_rank)
            .cloned()
            .fold(0.0, f64::max);
        let min_sv = self
            .singular_values
            .iter()
            .take(self.output_rank)
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if min_sv > 0.0 {
            max_sv / min_sv
        } else {
            f64::INFINITY
        }
    }

    /// Compute per-block drop counts from kept_indices for batch [X, P, W] mode.
    ///
    /// Given the block sizes, analyzes which original vectors were dropped.
    /// Returns (x_dropped, p_dropped, w_dropped).
    pub fn compute_block_drops(
        &self,
        x_size: usize,
        p_size: usize,
        w_size: usize,
    ) -> (usize, usize, usize) {
        let p_start = x_size;
        let w_start = x_size + p_size;

        // Count how many from each block were kept
        let x_kept = self.kept_indices.iter().filter(|&&i| i < p_start).count();
        let p_kept = self
            .kept_indices
            .iter()
            .filter(|&&i| i >= p_start && i < w_start)
            .count();
        let w_kept = self.kept_indices.iter().filter(|&&i| i >= w_start).count();

        // Drops = original size - kept
        let x_dropped = x_size.saturating_sub(x_kept);
        let p_dropped = p_size.saturating_sub(p_kept);
        let w_dropped = w_size.saturating_sub(w_kept);

        (x_dropped, p_dropped, w_dropped)
    }
}

/// Configuration for SVQB orthonormalization.
#[derive(Debug, Clone)]
pub struct SvqbConfig {
    /// Relative tolerance for singular value truncation.
    /// Singular values with σ_i / σ_max < drop_tol are dropped.
    pub drop_tol: f64,
}

impl Default for SvqbConfig {
    fn default() -> Self {
        Self { drop_tol: 1e-12 }
    }
}

/// Perform SVQB B-orthonormalization on a block of vectors.
///
/// Given a set of vectors that may be nearly linearly dependent in the B-inner product,
/// this function computes a well-conditioned B-orthonormal basis for their span.
///
/// # Algorithm
///
/// 1. Form the Gram matrix G = X^H B X (p×p for p vectors)
/// 2. Compute eigendecomposition G = Q Λ Q^H (Λ = diag(λ_1, ..., λ_p))
/// 3. Drop eigenpairs with λ_i / λ_max < drop_tol
/// 4. Form transformation T = Q_kept Λ_kept^{-1/2}
/// 5. Compute X_new = X T (now B-orthonormal)
///
/// # Arguments
/// * `backend` - The spectral backend
/// * `vectors` - Input vectors (modified in-place to B-orthonormal output)
/// * `mass_vectors` - Corresponding Bx vectors (modified in-place)
/// * `config` - SVQB configuration
///
/// # Returns
/// The SVQB result containing rank information.
///
/// # Notes
/// - The output vectors are a subset of linear combinations of the input
/// - If rank < input_count, the excess slots in vectors/mass_vectors are zeroed
/// - This is O(p³) in the number of vectors, plus O(p) operator applications
pub fn svqb_orthonormalize<B: SpectralBackend>(
    backend: &B,
    vectors: &mut [B::Buffer],
    mass_vectors: &mut [B::Buffer],
    config: &SvqbConfig,
) -> SvqbResult {
    let p = vectors.len();

    if p == 0 {
        return SvqbResult {
            output_rank: 0,
            input_count: 0,
            dropped_count: 0,
            singular_values: vec![],
            kept_indices: vec![],
            block_drops: None,
        };
    }

    // Normalize each column to unit B-norm before forming the Gram matrix.
    // This keeps the Gram spectrum focused on linear independence rather than
    // raw vector magnitudes, so tiny but essential residual directions are
    // not discarded purely because their norms are small.
    for (vector, mass_vector) in vectors.iter_mut().zip(mass_vectors.iter_mut()) {
        normalize_to_unit_b_norm_with_tol(backend, vector, mass_vector, 1e-30);
    }

    // Step 1: Form the Gram matrix G = X^H B X
    #[cfg(feature = "cuda")]
    let gram = {
        // GPU path: use batched gram_matrix (single ZGEMM call)
        backend.gram_matrix(vectors, mass_vectors)
    };

    #[cfg(not(feature = "cuda"))]
    let gram = {
        // CPU path: use individual dot products (better compiler optimization)
        let mut gram = vec![Complex64::ZERO; p * p];
        for i in 0..p {
            for j in 0..p {
                gram[i * p + j] = backend.dot(&vectors[i], &mass_vectors[j]);
            }
        }
        gram
    };

    // Step 2: Eigendecomposition of the Hermitian Gram matrix
    // G = Q Λ Q^H where Λ = diag(λ_1, ..., λ_p), sorted descending
    let (eigenvalues, eigenvectors) = hermitian_eigendecomposition(&gram, p);

    // Step 3: Determine numerical rank
    let max_eigenvalue = eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let threshold = config.drop_tol * max_eigenvalue;

    let mut kept_indices = Vec::new();
    let mut singular_values = Vec::new();

    for (i, &ev) in eigenvalues.iter().enumerate() {
        singular_values.push(ev.max(0.0).sqrt());
        if ev >= threshold && ev > 0.0 {
            kept_indices.push(i);
        }
    }

    let rank = kept_indices.len();
    let dropped = p - rank;

    if rank == 0 {
        // All vectors are numerically zero - zero them out
        for v in vectors.iter_mut() {
            zero_buffer(v.as_mut_slice());
        }
        for m in mass_vectors.iter_mut() {
            zero_buffer(m.as_mut_slice());
        }
        return SvqbResult {
            output_rank: 0,
            input_count: p,
            dropped_count: p,
            singular_values,
            kept_indices: vec![],
            block_drops: None,
        };
    }

    // Step 4: Build transformation matrix T = Q_kept * Λ_kept^{-1/2}
    // T is p×rank matrix
    let mut transform = vec![Complex64::ZERO; p * rank];
    for (out_col, &in_col) in kept_indices.iter().enumerate() {
        let scale = 1.0 / eigenvalues[in_col].max(1e-30).sqrt();
        for row in 0..p {
            transform[row * rank + out_col] = eigenvectors[row * p + in_col] * scale;
        }
    }

    // Step 5: Apply transformation X_new = X_old * T
    // We need temporary storage to avoid overwriting while reading
    let n = vectors[0].as_slice().len();
    let mut new_vectors: Vec<Vec<Complex64>> = vec![vec![Complex64::ZERO; n]; rank];
    let mut new_mass: Vec<Vec<Complex64>> = vec![vec![Complex64::ZERO; n]; rank];

    for out_idx in 0..rank {
        for in_idx in 0..p {
            let coeff = transform[in_idx * rank + out_idx];
            if coeff.norm() > 1e-30 {
                for k in 0..n {
                    new_vectors[out_idx][k] += coeff * vectors[in_idx].as_slice()[k];
                    new_mass[out_idx][k] += coeff * mass_vectors[in_idx].as_slice()[k];
                }
            }
        }
    }

    // Copy results back and zero out unused slots
    for i in 0..p {
        if i < rank {
            vectors[i].as_mut_slice().copy_from_slice(&new_vectors[i]);
            mass_vectors[i].as_mut_slice().copy_from_slice(&new_mass[i]);
        } else {
            zero_buffer(vectors[i].as_mut_slice());
            zero_buffer(mass_vectors[i].as_mut_slice());
        }
    }

    SvqbResult {
        output_rank: rank,
        input_count: p,
        dropped_count: dropped,
        singular_values,
        kept_indices,
        block_drops: None,
    }
}

// ============================================================================
// Simple Sequential B-Gram-Schmidt (for initialization)
// ============================================================================

/// Orthogonalize a vector against existing B-orthonormal vectors using Gram-Schmidt.
///
/// This is a simpler but less robust alternative to SVQB, suitable for initialization
/// where vectors are unlikely to be nearly dependent.
///
/// # Arguments
/// * `backend` - The spectral backend
/// * `vector` - Vector to orthogonalize (modified in-place)
/// * `mass_vector` - Mass vector Bx (modified in-place)
/// * `basis_vectors` - Existing B-orthonormal basis
/// * `basis_mass` - Mass vectors for basis
///
/// # Returns
/// The B-norm of the vector after orthogonalization (before final normalization).
pub fn orthogonalize_against_basis<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass_vector: &mut B::Buffer,
    basis_vectors: &[B::Buffer],
    basis_mass: &[B::Buffer],
) -> f64 {
    // Project out all basis components
    for (basis, basis_m) in basis_vectors.iter().zip(basis_mass.iter()) {
        project_out(backend, vector, mass_vector, basis, basis_m);
    }

    // Return the remaining norm
    b_norm(backend, vector, mass_vector)
}

/// Orthogonalize and normalize a vector against existing basis, returning success.
///
/// # Arguments
/// * `backend` - The spectral backend
/// * `vector` - Vector to orthogonalize (modified in-place)
/// * `mass_vector` - Mass vector Bx (modified in-place)
/// * `basis_vectors` - Existing B-orthonormal basis
/// * `basis_mass` - Mass vectors for basis
/// * `tol` - Minimum norm for acceptance
///
/// # Returns
/// `true` if the vector was successfully orthogonalized and normalized,
/// `false` if it was in the span of the basis (norm < tol).
pub fn orthonormalize_against_basis<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass_vector: &mut B::Buffer,
    basis_vectors: &[B::Buffer],
    basis_mass: &[B::Buffer],
    tol: f64,
) -> bool {
    let norm = orthogonalize_against_basis(backend, vector, mass_vector, basis_vectors, basis_mass);

    if norm > tol {
        let scale = Complex64::new(1.0 / norm, 0.0);
        backend.scale(scale, vector);
        backend.scale(scale, mass_vector);
        true
    } else {
        false
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Fill a buffer with zeros.
pub fn zero_buffer(data: &mut [Complex64]) {
    for value in data.iter_mut() {
        *value = Complex64::ZERO;
    }
}

/// Compute eigendecomposition of a Hermitian matrix.
///
/// Returns (eigenvalues, eigenvectors) where eigenvalues are sorted in descending order
/// and eigenvectors[i*n + j] is the j-th component of the i-th eigenvector.
///
/// This uses the Jacobi algorithm for small matrices, which is simple and robust.
fn hermitian_eigendecomposition(matrix: &[Complex64], n: usize) -> (Vec<f64>, Vec<Complex64>) {
    if n == 0 {
        return (vec![], vec![]);
    }

    if n == 1 {
        return (vec![matrix[0].re], vec![Complex64::ONE]);
    }

    // Copy matrix for working
    let mut work = matrix.to_vec();
    let mut eigenvectors = vec![Complex64::ZERO; n * n];

    // Initialize eigenvectors to identity
    for i in 0..n {
        eigenvectors[i * n + i] = Complex64::ONE;
    }

    // Jacobi iteration for Hermitian eigendecomposition
    const MAX_SWEEPS: usize = 50;
    const TOL: f64 = 1e-15;

    for _sweep in 0..MAX_SWEEPS {
        // One sweep = process all off-diagonal pairs
        let mut max_off = 0.0f64;

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[p * n + q];
                let apq_norm = apq.norm();

                if apq_norm > max_off {
                    max_off = apq_norm;
                }

                if apq_norm < TOL {
                    continue;
                }

                let app = work[p * n + p].re;
                let aqq = work[q * n + q].re;

                // For Hermitian matrices with complex off-diagonal:
                // First, factor out the phase: a_pq = |a_pq| * e^{i*phi}
                let phase = if apq_norm > 1e-30 {
                    apq / apq_norm
                } else {
                    Complex64::ONE
                };

                // Compute rotation angle for the real Jacobi problem with |a_pq|
                let tau = (aqq - app) / (2.0 * apq_norm);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // The complex rotation incorporates the phase
                let s_phase = Complex64::new(s, 0.0) * phase;
                let c_val = Complex64::new(c, 0.0);

                // Update diagonal elements
                let new_app = app - t * apq_norm;
                let new_aqq = aqq + t * apq_norm;
                work[p * n + p] = Complex64::new(new_app, 0.0);
                work[q * n + q] = Complex64::new(new_aqq, 0.0);
                work[p * n + q] = Complex64::ZERO;
                work[q * n + p] = Complex64::ZERO;

                // Update off-diagonal elements in rows/columns p and q
                for k in 0..n {
                    if k != p && k != q {
                        let akp = work[k * n + p];
                        let akq = work[k * n + q];

                        // Apply Givens rotation: [c, -s*conj(phase); s*phase, c]
                        let new_akp = c_val * akp - s_phase.conj() * akq;
                        let new_akq = s_phase * akp + c_val * akq;

                        work[k * n + p] = new_akp;
                        work[p * n + k] = new_akp.conj();
                        work[k * n + q] = new_akq;
                        work[q * n + k] = new_akq.conj();
                    }
                }

                // Update eigenvector matrix
                for k in 0..n {
                    let vkp = eigenvectors[k * n + p];
                    let vkq = eigenvectors[k * n + q];
                    eigenvectors[k * n + p] = c_val * vkp - s_phase.conj() * vkq;
                    eigenvectors[k * n + q] = s_phase * vkp + c_val * vkq;
                }
            }
        }

        if max_off < TOL {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| work[i * n + i].re).collect();

    // Sort by descending eigenvalue
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_eigenvectors = vec![Complex64::ZERO; n * n];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            sorted_eigenvectors[row * n + new_col] = eigenvectors[row * n + old_col];
        }
    }

    (sorted_eigenvalues, sorted_eigenvectors)
}
