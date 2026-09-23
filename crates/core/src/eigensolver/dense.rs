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
    crate::profile_scope!("dense_ritz");
    crate::profiler::count("dense_ritz.calls", 1);
    crate::profiler::count("dense_ritz.dimension_sum", dim);
    crate::profiler::count("dense_ritz.dimension_cubes", dim * dim * dim);
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

/// Diagonalize a real symmetric matrix through the complex reference kernel,
/// then recover a real basis within every degenerate eigenspace. Complex
/// eigenvectors may have arbitrary phases, so discarding imaginary parts is
/// not a valid conversion.
pub fn solve_real_via_complex(matrix:&[Complex64],n:usize)->DenseEigenResult {
    assert!(matrix.iter().all(|z|z.im==0.0));
    let source=solve_hermitian_eigen(matrix,n);let mut vectors=vec![Complex64::ZERO;n*n];let mut start=0;
    while start<n {
        let mut end=start+1;let scale=source.eigenvalues.iter().map(|x|x.abs()).fold(1.0,f64::max);
        while end<n && (source.eigenvalues[end]-source.eigenvalues[start]).abs()<=128.0*f64::EPSILON*n as f64*scale {end+=1;}
        let mut basis:Vec<Vec<f64>>=Vec::new();
        for column in start..end {for imaginary in [false,true] {
            let mut v:Vec<_>=(0..n).map(|i|if imaginary {source.eigenvectors[i+n*column].im}else{source.eigenvectors[i+n*column].re}).collect();
            for _ in 0..2 {for q in &basis {let dot:f64=q.iter().zip(&v).map(|(a,b)|a*b).sum();for (dst,&a) in v.iter_mut().zip(q) {*dst-=dot*a;}}}
            let norm=v.iter().map(|x|x*x).sum::<f64>().sqrt();if norm>1e-8 {for x in &mut v {*x/=norm;}basis.push(v);}
            if basis.len()==end-start {break;}
        }if basis.len()==end-start {break;}}
        assert_eq!(basis.len(),end-start,"complex eigenspace did not span the expected real eigenspace");
        for (j,v) in basis.iter().enumerate() {for (i,&x) in v.iter().enumerate() {vectors[i+n*(start+j)]=Complex64::new(x,0.0);}}
        start=end;
    }
    DenseEigenResult{eigenvalues:source.eigenvalues,eigenvectors:vectors,dim:n}
}

/// Real symmetric dense kernel at f64 precision. Input/output layout matches
/// the complex reference so the vector-space change is independent of storage.
pub fn solve_real_symmetric_eigen(matrix:&[Complex64],n:usize)->DenseEigenResult {
    crate::profile_scope!("real_dense_ritz");
    crate::profiler::count("real_dense_ritz.calls",1);
    crate::profiler::count("real_dense_ritz.dimension_sum",n);
    crate::profiler::count("real_dense_ritz.dimension_cubes",n*n*n);
    assert_eq!(matrix.len(),n*n);assert!(matrix.iter().all(|z|z.im==0.0));
    if n==0 {return DenseEigenResult{eigenvalues:vec![],eigenvectors:vec![],dim:0};}
    #[cfg(feature="native-linalg")]
    {
        let a=Mat::<f64>::from_fn(n,n,|i,j|matrix[i+n*j].re);
        let result=a.self_adjoint_eigen(Side::Lower).expect("real symmetric eigendecomposition");
        let diagonal=result.S().column_vector();let mut vectors=vec![Complex64::ZERO;n*n];
        for j in 0..n {for i in 0..n {vectors[i+n*j]=Complex64::new(*result.U().get(i,j),0.0);}}
        DenseEigenResult{eigenvalues:(0..n).map(|i|*diagonal.get(i)).collect(),eigenvectors:vectors,dim:n}
    }
    #[cfg(feature="wasm-linalg")]
    {
        let a=DMatrix::<f64>::from_fn(n,n,|i,j|matrix[i+n*j].re);let result=a.symmetric_eigen();
        let mut order:Vec<_>=(0..n).collect();order.sort_by(|&i,&j|result.eigenvalues[i].total_cmp(&result.eigenvalues[j]));
        let mut vectors=vec![Complex64::ZERO;n*n];for j in 0..n {for i in 0..n {vectors[i+n*j]=Complex64::new(result.eigenvectors[(i,order[j])],0.0);}}
        DenseEigenResult{eigenvalues:order.iter().map(|&i|result.eigenvalues[i]).collect(),eigenvectors:vectors,dim:n}
    }
}
