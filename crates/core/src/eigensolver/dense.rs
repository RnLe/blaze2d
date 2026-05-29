//! Dense Hermitian eigensolver with backend abstraction.
//!
//! This module provides efficient eigendecomposition for the small dense
//! projected matrices that arise in the Rayleigh-Ritz step of LOBPCG.
//!
//! For a Hermitian matrix A_s of size r×r (typically r ≤ 30), we solve:
//!
//! ```text
//! A_s Y = Y Θ
//! ```
//!
//! where:
//! - Θ = diag(θ₁, ..., θᵣ) contains the eigenvalues in ascending order
//! - Y contains the corresponding eigenvectors as columns
//!
//! Since B_s = I (ensured by SVQB B-orthonormalization), this is a standard
//! eigenvalue problem, not generalized.
//!
//! # Backend Selection
//!
//! - `native-linalg` feature: Uses faer's optimized self-adjoint eigensolver
//! - `wasm-linalg` feature: Uses nalgebra's symmetric eigensolver (WASM-compatible)

#[cfg(feature = "native-linalg")]
use faer::{Mat, Side};

#[cfg(feature = "wasm-linalg")]
use nalgebra::{Complex as NaComplex, DMatrix};

use num_complex::Complex64;

/// Result of dense Hermitian eigendecomposition.
#[derive(Debug, Clone)]
pub struct DenseEigenResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix Y where column j is the eigenvector for eigenvalue j.
    /// Stored in column-major order as a flat Vec: Y[i,j] at index i + j*dim.
    pub eigenvectors: Vec<Complex64>,
    /// Dimension of the matrix (number of eigenvalues/eigenvectors).
    pub dim: usize,
}

impl DenseEigenResult {
    /// Get the j-th eigenvector as a slice.
    #[inline]
    pub fn eigenvector(&self, j: usize) -> &[Complex64] {
        let start = j * self.dim;
        let end = start + self.dim;
        &self.eigenvectors[start..end]
    }

    /// Get the j-th eigenvalue.
    #[inline]
    pub fn eigenvalue(&self, j: usize) -> f64 {
        self.eigenvalues[j]
    }
}

/// Solve the dense Hermitian eigenvalue problem A_s Y = Y Θ.
///
/// # Arguments
/// * `a_matrix` - The Hermitian matrix A_s in column-major order (size dim×dim)
/// * `dim` - The dimension of the matrix
///
/// # Returns
/// A `DenseEigenResult` with eigenvalues in ascending order and corresponding eigenvectors.
///
/// # Panics
/// Panics if `a_matrix.len() != dim * dim`.
pub fn solve_hermitian_eigen(a_matrix: &[Complex64], dim: usize) -> DenseEigenResult {
    assert_eq!(a_matrix.len(), dim * dim, "Matrix size mismatch");

    if dim == 0 {
        return DenseEigenResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            dim: 0,
        };
    }

    #[cfg(feature = "native-linalg")]
    {
        solve_hermitian_eigen_faer(a_matrix, dim)
    }

    #[cfg(feature = "wasm-linalg")]
    {
        solve_hermitian_eigen_nalgebra(a_matrix, dim)
    }
}

// ============================================================================
// faer backend (native-linalg)
// ============================================================================

#[cfg(feature = "native-linalg")]
fn solve_hermitian_eigen_faer(a_matrix: &[Complex64], dim: usize) -> DenseEigenResult {
    // Convert to faer Mat<c64>
    // faer uses c64 which is compatible with num_complex::Complex64
    let a_faer = Mat::<faer::c64>::from_fn(dim, dim, |i, j| {
        let c = a_matrix[i + j * dim];
        faer::c64::new(c.re, c.im)
    });

    // Compute eigendecomposition of self-adjoint matrix
    // faer::Side::Lower means we use the lower triangular part (standard for Hermitian)
    // Returns eigenvalues in ascending order
    let eigen = a_faer
        .self_adjoint_eigen(Side::Lower)
        .expect("Eigendecomposition should succeed for Hermitian matrix");

    // Extract eigenvalues (real for Hermitian matrices)
    // S() returns DiagRef<c64>, use column_vector() to access elements
    let s_diag = eigen.S().column_vector();
    let eigenvalues: Vec<f64> = (0..dim).map(|i| s_diag.get(i).re).collect();

    // Extract eigenvectors in column-major order
    // U() returns MatRef<c64>
    let u = eigen.U();
    let mut eigenvectors = vec![Complex64::new(0.0, 0.0); dim * dim];
    for j in 0..dim {
        for i in 0..dim {
            let c = u.get(i, j);
            eigenvectors[i + j * dim] = Complex64::new(c.re, c.im);
        }
    }

    DenseEigenResult {
        eigenvalues,
        eigenvectors,
        dim,
    }
}

// ============================================================================
// nalgebra backend (wasm-linalg)
// ============================================================================

#[cfg(feature = "wasm-linalg")]
fn solve_hermitian_eigen_nalgebra(a_matrix: &[Complex64], dim: usize) -> DenseEigenResult {
    // Convert to nalgebra DMatrix<Complex<f64>>
    // Input is column-major, nalgebra also uses column-major
    let a_na = DMatrix::<NaComplex<f64>>::from_fn(dim, dim, |i, j| {
        let c = a_matrix[i + j * dim];
        NaComplex::new(c.re, c.im)
    });

    // Compute eigendecomposition using nalgebra's symmetric_eigen
    // This works for Hermitian matrices (complex symmetric)
    let eigen = a_na.clone().symmetric_eigen();

    // eigenvalues are in eigen.eigenvalues (ascending order in nalgebra)
    // eigenvectors are columns of eigen.eigenvectors

    // nalgebra returns eigenvalues as RealField (f64) for complex matrices
    let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();

    // Sort eigenvalues and get the sorting permutation
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues[i]
            .partial_cmp(&eigenvalues[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply sorting to eigenvalues
    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

    // Extract eigenvectors in sorted order (column-major)
    let mut eigenvectors = vec![Complex64::new(0.0, 0.0); dim * dim];
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..dim {
            let c = eigen.eigenvectors[(i, old_j)];
            eigenvectors[i + new_j * dim] = Complex64::new(c.re, c.im);
        }
    }

    DenseEigenResult {
        eigenvalues: sorted_eigenvalues,
        eigenvectors,
        dim,
    }
}

#[cfg(test)]
#[path = "_tests_dense.rs"]
mod tests;
