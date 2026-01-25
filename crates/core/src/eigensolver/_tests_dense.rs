//! Comprehensive tests for the dense Hermitian eigensolver.
//!
//! These tests verify correctness and numerical robustness of the faer-based
//! eigendecomposition used in the Rayleigh-Ritz step of LOBPCG.
//!
//! Test categories:
//! 1. Basic functionality (empty, 1×1, small matrices)
//! 2. Real symmetric matrices (diagonal, tridiagonal)
//! 3. Complex Hermitian matrices
//! 4. Eigenvalue ordering verification
//! 5. Eigenvector orthonormality
//! 6. Eigenvalue residual A*v = λ*v
//! 7. Degenerate (repeated) eigenvalues
//! 8. Edge cases for LOBPCG context:
//!    - Matrices with clustered eigenvalues
//!    - Matrices with widely separated eigenvalues
//!    - Near-singular matrices
//!    - Matrices with negative eigenvalues (relevant for photonic band gaps)
//!    - Typical LOBPCG subspace sizes (up to ~30×30)

use super::{DenseEigenResult, solve_hermitian_eigen};
use num_complex::Complex64;

// ============================================================================
// Helper Functions
// ============================================================================

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn r(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

/// Create a diagonal matrix from eigenvalues.
fn diagonal_matrix(eigenvalues: &[f64]) -> Vec<Complex64> {
    let n = eigenvalues.len();
    let mut a = vec![c(0.0, 0.0); n * n];
    for i in 0..n {
        a[i + i * n] = r(eigenvalues[i]);
    }
    a
}

/// Create a real symmetric tridiagonal matrix with given diagonal and off-diagonal.
fn tridiagonal_matrix(diag: &[f64], offdiag: &[f64]) -> Vec<Complex64> {
    let n = diag.len();
    assert_eq!(offdiag.len(), n - 1);
    let mut a = vec![c(0.0, 0.0); n * n];
    for i in 0..n {
        a[i + i * n] = r(diag[i]);
    }
    for i in 0..n - 1 {
        a[i + (i + 1) * n] = r(offdiag[i]); // upper off-diagonal
        a[(i + 1) + i * n] = r(offdiag[i]); // lower off-diagonal (symmetric)
    }
    a
}

/// Verify that eigenvectors are orthonormal.
fn check_orthonormality(result: &DenseEigenResult, tol: f64) {
    let n = result.dim;
    for i in 0..n {
        for j in 0..n {
            let vi = result.eigenvector(i);
            let vj = result.eigenvector(j);
            let inner: Complex64 = vi
                .iter()
                .zip(vj.iter())
                .map(|(a, b): (&Complex64, &Complex64)| a.conj() * b)
                .sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (inner.re - expected).abs() < tol && inner.im.abs() < tol,
                "Orthonormality violated at ({}, {}): got ({:.2e}, {:.2e}), expected ({}, 0)",
                i,
                j,
                inner.re,
                inner.im,
                expected
            );
        }
    }
}

/// Verify A*v = λ*v for each eigenpair.
fn check_eigenpair_residuals(a: &[Complex64], result: &DenseEigenResult, tol: f64) {
    let n = result.dim;
    for j in 0..n {
        let lambda = result.eigenvalues[j];
        let v = result.eigenvector(j);

        // Compute A*v
        let mut av = vec![c(0.0, 0.0); n];
        for row in 0..n {
            for col in 0..n {
                av[row] += a[row + col * n] * v[col];
            }
        }

        // Compute residual ||A*v - λ*v|| / |λ| (relative if λ ≠ 0)
        let residual_norm: f64 = av
            .iter()
            .zip(v.iter())
            .map(|(av_i, v_i)| (av_i - v_i * lambda).norm_sqr())
            .sum::<f64>()
            .sqrt();

        let scale = if lambda.abs() > 1e-10 {
            lambda.abs()
        } else {
            1.0
        };
        let relative_residual = residual_norm / scale;

        assert!(
            relative_residual < tol,
            "Eigenpair {} residual too large: ||A*v - λ*v||/|λ| = {:.2e} > {:.2e}",
            j,
            relative_residual,
            tol
        );
    }
}

/// Verify eigenvalues are sorted in ascending order.
fn check_eigenvalue_ordering(result: &DenseEigenResult) {
    for i in 1..result.dim {
        assert!(
            result.eigenvalues[i] >= result.eigenvalues[i - 1] - 1e-14,
            "Eigenvalues not sorted: λ[{}] = {} < λ[{}] = {}",
            i,
            result.eigenvalues[i],
            i - 1,
            result.eigenvalues[i - 1]
        );
    }
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_empty_matrix() {
    let result = solve_hermitian_eigen(&[], 0);
    assert_eq!(result.dim, 0);
    assert!(result.eigenvalues.is_empty());
    assert!(result.eigenvectors.is_empty());
}

#[test]
fn test_1x1_real() {
    let a = vec![r(5.0)];
    let result = solve_hermitian_eigen(&a, 1);

    assert_eq!(result.dim, 1);
    assert!((result.eigenvalue(0) - 5.0).abs() < 1e-14);

    // Eigenvector should be unit
    let v = result.eigenvector(0);
    let norm: f64 = v
        .iter()
        .map(|x: &Complex64| x.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-14);
}

#[test]
fn test_1x1_negative() {
    let a = vec![r(-3.5)];
    let result = solve_hermitian_eigen(&a, 1);
    assert!((result.eigenvalue(0) - (-3.5)).abs() < 1e-14);
}

#[test]
fn test_1x1_zero() {
    let a = vec![r(0.0)];
    let result = solve_hermitian_eigen(&a, 1);
    assert!(result.eigenvalue(0).abs() < 1e-14);
}

// ============================================================================
// Real Symmetric Matrices
// ============================================================================

#[test]
fn test_2x2_diagonal_ordered() {
    // Already ordered: λ₁ = 2 < λ₂ = 5
    let a = diagonal_matrix(&[2.0, 5.0]);
    let result = solve_hermitian_eigen(&a, 2);

    assert!((result.eigenvalue(0) - 2.0).abs() < 1e-14);
    assert!((result.eigenvalue(1) - 5.0).abs() < 1e-14);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_2x2_diagonal_reversed() {
    // Reversed order on diagonal: should still give sorted output
    let a = diagonal_matrix(&[7.0, 3.0]);
    let result = solve_hermitian_eigen(&a, 2);

    assert!((result.eigenvalue(0) - 3.0).abs() < 1e-14);
    assert!((result.eigenvalue(1) - 7.0).abs() < 1e-14);
    check_eigenvalue_ordering(&result);
}

#[test]
fn test_2x2_symmetric_offdiagonal() {
    // A = [[2, 1], [1, 2]] → eigenvalues 1, 3
    let a = vec![
        r(2.0),
        r(1.0), // column 0
        r(1.0),
        r(2.0), // column 1
    ];
    let result = solve_hermitian_eigen(&a, 2);

    assert!((result.eigenvalue(0) - 1.0).abs() < 1e-14);
    assert!((result.eigenvalue(1) - 3.0).abs() < 1e-14);
    check_eigenpair_residuals(&a, &result, 1e-12);
}

#[test]
fn test_3x3_tridiagonal() {
    // Classic tridiagonal: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    // Eigenvalues: 2 - √2, 2, 2 + √2
    let a = tridiagonal_matrix(&[2.0, 2.0, 2.0], &[-1.0, -1.0]);
    let result = solve_hermitian_eigen(&a, 3);

    let sqrt2 = 2.0_f64.sqrt();
    assert!((result.eigenvalue(0) - (2.0 - sqrt2)).abs() < 1e-12);
    assert!((result.eigenvalue(1) - 2.0).abs() < 1e-12);
    assert!((result.eigenvalue(2) - (2.0 + sqrt2)).abs() < 1e-12);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_4x4_diagonal_mixed_signs() {
    // Mixed positive and negative eigenvalues
    let a = diagonal_matrix(&[3.0, -2.0, 1.0, -4.0]);
    let result = solve_hermitian_eigen(&a, 4);

    // Should be sorted: -4, -2, 1, 3
    assert!((result.eigenvalue(0) - (-4.0)).abs() < 1e-14);
    assert!((result.eigenvalue(1) - (-2.0)).abs() < 1e-14);
    assert!((result.eigenvalue(2) - 1.0).abs() < 1e-14);
    assert!((result.eigenvalue(3) - 3.0).abs() < 1e-14);
}

// ============================================================================
// Complex Hermitian Matrices
// ============================================================================

#[test]
fn test_2x2_hermitian_imaginary_offdiag() {
    // A = [[3, i], [-i, 3]] → eigenvalues 2, 4
    let a = vec![
        c(3.0, 0.0),
        c(0.0, -1.0), // column 0
        c(0.0, 1.0),
        c(3.0, 0.0), // column 1
    ];
    let result = solve_hermitian_eigen(&a, 2);

    assert!((result.eigenvalue(0) - 2.0).abs() < 1e-12);
    assert!((result.eigenvalue(1) - 4.0).abs() < 1e-12);
    check_eigenpair_residuals(&a, &result, 1e-12);
}

#[test]
fn test_2x2_hermitian_complex_offdiag() {
    // A = [[4, 1-i], [1+i, 3]]
    // Eigenvalues can be computed analytically
    let a = vec![
        c(4.0, 0.0),
        c(1.0, 1.0), // column 0
        c(1.0, -1.0),
        c(3.0, 0.0), // column 1
    ];
    let result = solve_hermitian_eigen(&a, 2);

    check_eigenvalue_ordering(&result);
    check_orthonormality(&result, 1e-12);
    check_eigenpair_residuals(&a, &result, 1e-12);

    // Trace = sum of eigenvalues = 4 + 3 = 7
    let trace: f64 = result.eigenvalues.iter().sum();
    assert!((trace - 7.0).abs() < 1e-12);
}

#[test]
fn test_3x3_hermitian_tridiagonal_complex() {
    // Hermitian tridiagonal with complex off-diagonals
    // A = [[2,  i, 0],
    //      [-i, 3, 2i],
    //      [0, -2i, 2]]
    let a = vec![
        c(2.0, 0.0),
        c(0.0, -1.0),
        c(0.0, 0.0), // column 0
        c(0.0, 1.0),
        c(3.0, 0.0),
        c(0.0, -2.0), // column 1
        c(0.0, 0.0),
        c(0.0, 2.0),
        c(2.0, 0.0), // column 2
    ];
    let result = solve_hermitian_eigen(&a, 3);

    check_eigenvalue_ordering(&result);
    check_orthonormality(&result, 1e-11);
    check_eigenpair_residuals(&a, &result, 1e-11);

    // Trace check
    let trace: f64 = result.eigenvalues.iter().sum();
    assert!((trace - 7.0).abs() < 1e-11); // 2 + 3 + 2 = 7
}

// ============================================================================
// Degenerate (Repeated) Eigenvalues
// ============================================================================

#[test]
fn test_identity_2x2() {
    let a = diagonal_matrix(&[1.0, 1.0]);
    let result = solve_hermitian_eigen(&a, 2);

    assert!((result.eigenvalue(0) - 1.0).abs() < 1e-14);
    assert!((result.eigenvalue(1) - 1.0).abs() < 1e-14);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_identity_3x3() {
    let a = diagonal_matrix(&[1.0, 1.0, 1.0]);
    let result = solve_hermitian_eigen(&a, 3);

    for i in 0..3 {
        assert!((result.eigenvalue(i) - 1.0).abs() < 1e-14);
    }
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_twofold_degeneracy() {
    // Two eigenvalues are the same
    let a = diagonal_matrix(&[2.0, 5.0, 5.0]);
    let result = solve_hermitian_eigen(&a, 3);

    assert!((result.eigenvalue(0) - 2.0).abs() < 1e-14);
    assert!((result.eigenvalue(1) - 5.0).abs() < 1e-14);
    assert!((result.eigenvalue(2) - 5.0).abs() < 1e-14);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_threefold_degeneracy_with_distinct() {
    // Three of the same, one different
    let a = diagonal_matrix(&[3.0, 3.0, 3.0, 7.0]);
    let result = solve_hermitian_eigen(&a, 4);

    for i in 0..3 {
        assert!((result.eigenvalue(i) - 3.0).abs() < 1e-14);
    }
    assert!((result.eigenvalue(3) - 7.0).abs() < 1e-14);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_zero_matrix() {
    // All eigenvalues zero (maximum degeneracy)
    let a = vec![c(0.0, 0.0); 9]; // 3×3 zero matrix
    let result = solve_hermitian_eigen(&a, 3);

    for i in 0..3 {
        assert!(result.eigenvalue(i).abs() < 1e-14);
    }
    check_orthonormality(&result, 1e-12);
}

// ============================================================================
// Clustered Eigenvalues (LOBPCG Edge Case)
// ============================================================================

#[test]
fn test_clustered_eigenvalues_tight() {
    // Eigenvalues very close together: 1.0, 1.0 + 1e-6, 1.0 + 2e-6
    let eps = 1e-6;
    let a = diagonal_matrix(&[1.0, 1.0 + eps, 1.0 + 2.0 * eps]);
    let result = solve_hermitian_eigen(&a, 3);

    assert!((result.eigenvalue(0) - 1.0).abs() < 1e-12);
    assert!((result.eigenvalue(1) - (1.0 + eps)).abs() < 1e-12);
    assert!((result.eigenvalue(2) - (1.0 + 2.0 * eps)).abs() < 1e-12);
    check_eigenvalue_ordering(&result);
    check_orthonormality(&result, 1e-10);
}

#[test]
fn test_clustered_eigenvalues_two_groups() {
    // Two clusters: {1.0, 1.001} and {5.0, 5.001}
    let a = diagonal_matrix(&[1.0, 1.001, 5.0, 5.001]);
    let result = solve_hermitian_eigen(&a, 4);

    check_eigenvalue_ordering(&result);
    check_orthonormality(&result, 1e-12);

    // Check clustering preserved
    assert!((result.eigenvalue(1) - result.eigenvalue(0)) < 0.01);
    assert!((result.eigenvalue(3) - result.eigenvalue(2)) < 0.01);
    assert!((result.eigenvalue(2) - result.eigenvalue(1)) > 3.9);
}

// ============================================================================
// Widely Separated Eigenvalues (LOBPCG Edge Case)
// ============================================================================

#[test]
fn test_widely_separated_eigenvalues() {
    // Eigenvalues spanning several orders of magnitude
    let a = diagonal_matrix(&[1e-3, 1.0, 1e3]);
    let result = solve_hermitian_eigen(&a, 3);

    assert!((result.eigenvalue(0) - 1e-3).abs() < 1e-15);
    assert!((result.eigenvalue(1) - 1.0).abs() < 1e-12);
    assert!((result.eigenvalue(2) - 1e3).abs() < 1e-9);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_extreme_eigenvalue_ratio() {
    // Very large ratio between smallest and largest
    let a = diagonal_matrix(&[1e-6, 1.0, 1e6]);
    let result = solve_hermitian_eigen(&a, 3);

    check_eigenvalue_ordering(&result);
    check_eigenpair_residuals(&a, &result, 1e-6); // Looser tolerance for extreme case
}

// ============================================================================
// Negative Eigenvalues (Photonic Crystal Context)
// ============================================================================

#[test]
fn test_all_negative_eigenvalues() {
    let a = diagonal_matrix(&[-5.0, -3.0, -1.0]);
    let result = solve_hermitian_eigen(&a, 3);

    // Should be sorted: -5, -3, -1
    assert!((result.eigenvalue(0) - (-5.0)).abs() < 1e-14);
    assert!((result.eigenvalue(1) - (-3.0)).abs() < 1e-14);
    assert!((result.eigenvalue(2) - (-1.0)).abs() < 1e-14);
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_mixed_sign_eigenvalues() {
    // Common in indefinite projected matrices
    let a = diagonal_matrix(&[-2.0, -0.5, 0.5, 2.0]);
    let result = solve_hermitian_eigen(&a, 4);

    assert!((result.eigenvalue(0) - (-2.0)).abs() < 1e-14);
    assert!((result.eigenvalue(1) - (-0.5)).abs() < 1e-14);
    assert!((result.eigenvalue(2) - 0.5).abs() < 1e-14);
    assert!((result.eigenvalue(3) - 2.0).abs() < 1e-14);
}

#[test]
fn test_eigenvalue_near_zero() {
    // One eigenvalue very close to zero
    let a = diagonal_matrix(&[-1.0, 1e-12, 1.0]);
    let result = solve_hermitian_eigen(&a, 3);

    assert!((result.eigenvalue(0) - (-1.0)).abs() < 1e-12);
    assert!(result.eigenvalue(1).abs() < 1e-10);
    assert!((result.eigenvalue(2) - 1.0).abs() < 1e-12);
}

// ============================================================================
// Typical LOBPCG Subspace Sizes
// ============================================================================

#[test]
fn test_10x10_random_hermitian() {
    // Create a random-ish Hermitian matrix with known properties
    // Use a simple construction: A = D + U + U^* where D is diagonal
    let n = 10;
    let mut a = vec![c(0.0, 0.0); n * n];

    // Diagonal entries
    for i in 0..n {
        a[i + i * n] = r((i + 1) as f64); // 1, 2, 3, ..., 10
    }

    // Small off-diagonal perturbations (Hermitian: A[i,j] = conj(A[j,i]))
    for i in 0..n {
        for j in (i + 1)..n {
            let val = c(0.1 / ((j - i) as f64), 0.05 / ((j - i + 1) as f64));
            a[i + j * n] = val;
            a[j + i * n] = val.conj();
        }
    }

    let result = solve_hermitian_eigen(&a, n);

    check_eigenvalue_ordering(&result);
    check_orthonormality(&result, 1e-10);
    check_eigenpair_residuals(&a, &result, 1e-10);

    // Trace should equal sum of diagonal: 1 + 2 + ... + 10 = 55
    let trace: f64 = result.eigenvalues.iter().sum();
    assert!((trace - 55.0).abs() < 1e-8);
}

#[test]
fn test_20x20_diagonal() {
    // Larger diagonal matrix
    let eigenvalues: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
    let a = diagonal_matrix(&eigenvalues);
    let result = solve_hermitian_eigen(&a, 20);

    for (i, &expected) in eigenvalues.iter().enumerate() {
        assert!(
            (result.eigenvalue(i) - expected).abs() < 1e-12,
            "Mismatch at eigenvalue {}: expected {}, got {}",
            i,
            expected,
            result.eigenvalue(i)
        );
    }
    check_orthonormality(&result, 1e-12);
}

#[test]
fn test_30x30_tridiagonal() {
    // Typical maximum LOBPCG subspace size
    let n = 30;
    let diag: Vec<f64> = vec![2.0; n];
    let offdiag: Vec<f64> = vec![-1.0; n - 1];
    let a = tridiagonal_matrix(&diag, &offdiag);

    let result = solve_hermitian_eigen(&a, n);

    check_eigenvalue_ordering(&result);
    check_orthonormality(&result, 1e-10);
    check_eigenpair_residuals(&a, &result, 1e-10);

    // All eigenvalues should be positive for this matrix
    for i in 0..n {
        assert!(
            result.eigenvalue(i) > 0.0,
            "Eigenvalue {} should be positive",
            i
        );
    }

    // Eigenvalues of this tridiagonal are known: 2 - 2*cos(k*π/(n+1)) for k=1..n
    let expected_min = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
    let expected_max = 2.0 - 2.0 * (n as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
    assert!((result.eigenvalue(0) - expected_min).abs() < 1e-10);
    assert!((result.eigenvalue(n - 1) - expected_max).abs() < 1e-10);
}

// ============================================================================
// Near-Singular / Ill-Conditioned Matrices
// ============================================================================

#[test]
fn test_near_singular_matrix() {
    // Matrix with one very small eigenvalue
    let a = diagonal_matrix(&[1e-10, 1.0, 2.0]);
    let result = solve_hermitian_eigen(&a, 3);

    assert!(result.eigenvalue(0).abs() < 1e-8);
    assert!((result.eigenvalue(1) - 1.0).abs() < 1e-12);
    assert!((result.eigenvalue(2) - 2.0).abs() < 1e-12);
    check_orthonormality(&result, 1e-10);
}

#[test]
fn test_high_condition_number() {
    // Condition number ~1e10
    let a = diagonal_matrix(&[1e-5, 1.0, 1e5]);
    let result = solve_hermitian_eigen(&a, 3);

    check_eigenvalue_ordering(&result);
    // Orthonormality may be slightly degraded for very ill-conditioned
    check_orthonormality(&result, 1e-8);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_repeated_solve_consistency() {
    // Verify deterministic results
    let a = vec![c(3.0, 0.0), c(1.0, 1.0), c(1.0, -1.0), c(2.0, 0.0)];

    let result1 = solve_hermitian_eigen(&a, 2);
    let result2 = solve_hermitian_eigen(&a, 2);

    for i in 0..2 {
        assert!((result1.eigenvalue(i) - result2.eigenvalue(i)).abs() < 1e-15);
    }
}

#[test]
fn test_scaled_matrix() {
    // Scaling the matrix should scale eigenvalues proportionally
    let a_base = vec![r(2.0), r(1.0), r(1.0), r(2.0)];

    let scale = 100.0;
    let a_scaled: Vec<Complex64> = a_base.iter().map(|x| x * scale).collect();

    let result_base = solve_hermitian_eigen(&a_base, 2);
    let result_scaled = solve_hermitian_eigen(&a_scaled, 2);

    for i in 0..2 {
        assert!(
            (result_scaled.eigenvalue(i) - scale * result_base.eigenvalue(i)).abs() < 1e-10,
            "Scaled eigenvalue mismatch at {}",
            i
        );
    }
}

// ============================================================================
// Accessor Method Tests
// ============================================================================

#[test]
fn test_eigenvector_accessor() {
    let a = diagonal_matrix(&[1.0, 2.0, 3.0]);
    let result = solve_hermitian_eigen(&a, 3);

    for j in 0..3 {
        let v = result.eigenvector(j);
        assert_eq!(v.len(), 3);

        // For diagonal matrix, eigenvectors are standard basis vectors
        // The j-th smallest eigenvalue (j after sorting) corresponds to some basis vector
    }
}

#[test]
fn test_eigenvalue_accessor() {
    let a = diagonal_matrix(&[5.0, 3.0, 7.0]);
    let result = solve_hermitian_eigen(&a, 3);

    // After sorting: 3, 5, 7
    assert!((result.eigenvalue(0) - 3.0).abs() < 1e-14);
    assert!((result.eigenvalue(1) - 5.0).abs() < 1e-14);
    assert!((result.eigenvalue(2) - 7.0).abs() < 1e-14);
}

// ============================================================================
// DenseEigenResult Construction Test
// ============================================================================

#[test]
fn test_result_structure() {
    let a = diagonal_matrix(&[1.0, 2.0]);
    let result = solve_hermitian_eigen(&a, 2);

    assert_eq!(result.dim, 2);
    assert_eq!(result.eigenvalues.len(), 2);
    assert_eq!(result.eigenvectors.len(), 4); // 2×2 = 4
}
