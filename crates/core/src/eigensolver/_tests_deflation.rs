//! Tests for the deflation module.
//!
//! These tests cover:
//! - Basic locking logic
//! - DeflationSubspace operations with a mock backend
//! - Physics-relevant edge cases for photonic crystal band calculations
//!
//! # Physics Context
//!
//! In photonic crystal band structure calculations:
//! - Bands converge at different rates (lower bands faster)
//! - Near-degenerate bands (band crossings) require careful handling
//! - The Γ point (k=0) has special symmetry properties
//! - Band gaps create natural separation between band groups

use super::deflation::{LockingResult, check_for_locking};

// ============================================================================
// check_for_locking Tests
// ============================================================================

#[test]
fn test_check_for_locking_partial() {
    let eigenvalue_changes = vec![1e-8, 1e-4, 1e-9, 1e-3];
    let tol = 1e-6;
    let result = check_for_locking(&eigenvalue_changes, tol);

    // Bands 0 and 2 should lock (below 1e-6)
    assert_eq!(result.bands_to_lock, vec![0, 2]);
    // Bands 1 and 3 should remain active
    assert_eq!(result.bands_to_keep, vec![1, 3]);
}

#[test]
fn test_check_for_locking_empty() {
    let eigenvalue_changes: Vec<f64> = vec![];
    let tol = 1e-6;
    let result = check_for_locking(&eigenvalue_changes, tol);

    assert!(result.bands_to_lock.is_empty());
    assert!(result.bands_to_keep.is_empty());
}

#[test]
fn test_check_for_locking_single_band() {
    let tol = 1e-6;

    // Single band converged
    let result = check_for_locking(&[1e-8], tol);
    assert_eq!(result.bands_to_lock, vec![0]);
    assert!(result.bands_to_keep.is_empty());

    // Single band not converged
    let result = check_for_locking(&[1e-4], tol);
    assert!(result.bands_to_lock.is_empty());
    assert_eq!(result.bands_to_keep, vec![0]);
}

#[test]
fn test_locking_result_has_locks() {
    let with_locks = LockingResult {
        bands_to_lock: vec![0, 2],
        bands_to_keep: vec![1, 3],
    };
    assert!(with_locks.has_locks());

    let without_locks = LockingResult {
        bands_to_lock: vec![],
        bands_to_keep: vec![0, 1, 2, 3],
    };
    assert!(!without_locks.has_locks());
}

// ============================================================================
// Physics-Relevant Edge Cases
// ============================================================================

/// Test typical photonic crystal convergence pattern:
/// Lower bands converge first, higher bands converge later.
#[test]
fn test_photonic_crystal_typical_convergence_order() {
    let tol = 1e-6;

    // Typical pattern: band 1 converges first, then 2, 3, 4...
    // Only bands 0-1 have converged (eigenvalue changes below tol)
    let eigenvalue_changes = vec![1e-8, 1e-7, 1e-4, 1e-3, 1e-2, 1e-1];
    let result = check_for_locking(&eigenvalue_changes, tol);
    assert_eq!(result.bands_to_lock, vec![0, 1]);
    assert_eq!(result.bands_to_keep, vec![2, 3, 4, 5]);
}

/// Test near-degenerate bands (band crossing/anti-crossing).
/// In photonic crystals, bands can be very close in eigenvalue at certain k-points.
#[test]
fn test_near_degenerate_bands() {
    let tol = 1e-6;

    // At a band crossing, two bands might have very similar convergence rates
    let eigenvalue_changes = vec![1e-8, 1e-8, 1e-3, 1e-3];
    let result = check_for_locking(&eigenvalue_changes, tol);

    // Both degenerate bands should lock together
    assert_eq!(result.bands_to_lock, vec![0, 1]);
    assert_eq!(result.bands_to_keep, vec![2, 3]);
}

/// Test bands across a photonic band gap.
/// Bands below the gap converge much faster than bands above.
#[test]
fn test_convergence_across_band_gap() {
    let tol = 1e-6;

    // Bands 0-2 are below the gap (well converged)
    // Bands 3-5 are above the gap (still converging)
    let eigenvalue_changes = vec![
        1e-10, 1e-9, 1e-8, // Below gap: very converged
        1e-3, 1e-2, 1e-1, // Above gap: still far
    ];

    let result = check_for_locking(&eigenvalue_changes, tol);
    assert_eq!(result.bands_to_lock, vec![0, 1, 2]);
    assert_eq!(result.bands_to_keep, vec![3, 4, 5]);
}

/// Test stalled convergence (eigenvalue changes plateau).
/// This can happen when the subspace becomes rank-deficient.
#[test]
fn test_stalled_convergence() {
    let tol = 1e-6;

    // All bands stalled just above tolerance
    let stalled_changes = vec![1e-5, 1e-5, 1e-5, 1e-5];
    let result = check_for_locking(&stalled_changes, tol);

    // Nothing should lock (all above tolerance)
    assert!(result.bands_to_lock.is_empty());
    assert_eq!(result.bands_to_keep.len(), 4);
}

/// Test tolerance exactly at boundary.
#[test]
fn test_tolerance_boundary() {
    let tol = 1e-6;

    // Exactly at tolerance should NOT lock (strictly less than)
    let eigenvalue_changes = vec![1e-6, 0.999999e-6, 1.000001e-6];
    let result = check_for_locking(&eigenvalue_changes, tol);

    // Only the one strictly below tolerance should lock
    assert_eq!(result.bands_to_lock, vec![1]); // 0.999999e-6 < 1e-6
    assert_eq!(result.bands_to_keep, vec![0, 2]);
}

/// Test very tight tolerance (for high-precision calculations).
#[test]
fn test_tight_tolerance() {
    let tol = 1e-12; // Very tight

    let eigenvalue_changes = vec![1e-10, 1e-11, 1e-13, 1e-14];
    let result = check_for_locking(&eigenvalue_changes, tol);

    // Only bands 2, 3 are below 1e-12
    assert_eq!(result.bands_to_lock, vec![2, 3]);
    assert_eq!(result.bands_to_keep, vec![0, 1]);
}

/// Test loose tolerance (for quick surveys).
#[test]
fn test_loose_tolerance() {
    let tol = 1e-3; // Very loose

    let eigenvalue_changes = vec![1e-4, 1e-3, 5e-4, 2e-3];
    let result = check_for_locking(&eigenvalue_changes, tol);

    // Bands 0, 2 are below 1e-3
    assert_eq!(result.bands_to_lock, vec![0, 2]);
    assert_eq!(result.bands_to_keep, vec![1, 3]);
}

/// Test with NaN/Inf eigenvalue changes (should not lock these).
#[test]
fn test_nan_inf_eigenvalue_changes() {
    let tol = 1e-6;

    let eigenvalue_changes = vec![1e-8, f64::NAN, f64::INFINITY, 1e-10];
    let result = check_for_locking(&eigenvalue_changes, tol);

    // NaN and Inf should NOT lock (NaN < 1e-6 is false, Inf < 1e-6 is false)
    assert_eq!(result.bands_to_lock, vec![0, 3]);
    assert_eq!(result.bands_to_keep, vec![1, 2]);
}

/// Test negative eigenvalue changes (should not happen, but handle gracefully).
#[test]
fn test_negative_eigenvalue_changes() {
    let tol = 1e-6;

    // Negative changes shouldn't happen, but they're < tol, so would lock
    let eigenvalue_changes = vec![-1e-8, 1e-8, -1.0, 1e-4];
    let result = check_for_locking(&eigenvalue_changes, tol);

    // Bands 0, 1, 2 are below 1e-6 (including negatives)
    assert_eq!(result.bands_to_lock, vec![0, 1, 2]);
    assert_eq!(result.bands_to_keep, vec![3]);
}

/// Test many bands (typical for large supercell calculations).
#[test]
fn test_many_bands() {
    let tol = 1e-6;

    // 20 bands with varying convergence
    let mut eigenvalue_changes = Vec::with_capacity(20);
    for i in 0..20 {
        // Lower bands converge faster (exponential decay with band index)
        eigenvalue_changes.push(1e-8 * (10.0_f64).powi(i as i32 / 3));
    }

    let result = check_for_locking(&eigenvalue_changes, tol);

    // First ~6 bands should be below 1e-6
    assert!(!result.bands_to_lock.is_empty());
    assert!(!result.bands_to_keep.is_empty());

    // Verify locked bands are the lowest-indexed ones
    for &band in &result.bands_to_lock {
        assert!(eigenvalue_changes[band] < tol);
    }
    for &band in &result.bands_to_keep {
        assert!(eigenvalue_changes[band] >= tol);
    }
}

/// Test alternating convergence pattern (can happen with symmetry).
#[test]
fn test_alternating_convergence() {
    let tol = 1e-6;

    // Alternating pattern: even bands converged, odd bands not
    // This can happen when certain symmetry modes converge faster
    let eigenvalue_changes = vec![1e-8, 1e-3, 1e-9, 1e-2, 1e-7, 1e-4];
    let result = check_for_locking(&eigenvalue_changes, tol);

    assert_eq!(result.bands_to_lock, vec![0, 2, 4]);
    assert_eq!(result.bands_to_keep, vec![1, 3, 5]);
}

// ============================================================================
// B-Orthogonal Projection Tests
// ============================================================================
//
// These tests verify that project_single and project_block_no_mass correctly
// implement B-orthogonal projection:
//   v' = v - <y, v>_B * y
// where <x, y>_B = x^H B y is the B-inner product.
//
// After projection, we should have <y, v'>_B = 0 (within numerical tolerance).

use super::deflation::DeflationSubspace;
use crate::backend::SpectralBackend;
use crate::field::Field2D;
use crate::grid::Grid2D;
use num_complex::Complex64;

/// Helper: compute B-inner product <x, y>_B = x^H B y
fn b_inner_product(x: &[Complex64], by: &[Complex64]) -> Complex64 {
    x.iter()
        .zip(by.iter())
        .map(|(xi, byi)| xi.conj() * byi)
        .sum()
}

/// Helper: compute standard inner product <x, y> = x^H y
fn inner_product(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| xi.conj() * yi).sum()
}

/// Mock backend for testing projection
#[derive(Clone)]
struct MockProjectionBackend;

impl SpectralBackend for MockProjectionBackend {
    type Buffer = Field2D;

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}
    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn scale(&self, alpha: Complex64, x: &mut Self::Buffer) {
        for v in x.as_mut_slice() {
            *v *= alpha;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (yi, xi) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *yi += alpha * xi;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        inner_product(x.as_slice(), y.as_slice())
    }
}

/// Helper to create a 2x2 grid with physical size 1x1
fn test_grid() -> Grid2D {
    Grid2D::new(2, 2, 1.0, 1.0)
}

/// Test project_single with a real-valued basis vector.
///
/// Setup: y = [1, 0, 0, 0] / ||y||_B, B = I (identity)
/// Then <y, y>_B = 1 after normalization.
///
/// For v = [1, 1, 1, 1], after projection:
/// v' = v - <y, v>_B * y should satisfy <y, v'>_B = 0
#[test]
fn test_project_single_real_basis() {
    let backend = MockProjectionBackend;
    let grid = test_grid();

    // Create deflation subspace with one B-normalized vector
    // y = [1, 0, 0, 0], By = [1, 0, 0, 0] (B = I)
    let mut y = Field2D::zeros(grid);
    y.as_mut_slice()[0] = Complex64::new(1.0, 0.0);

    let by = y.clone(); // B = I, so By = y

    let mut deflation: DeflationSubspace<MockProjectionBackend> = DeflationSubspace::new();
    deflation.add_vector(&backend, &y, &by, 1.0, 0);

    // Create test vector v = [1, 1, 1, 1]
    let mut v = Field2D::zeros(grid);
    for val in v.as_mut_slice() {
        *val = Complex64::new(1.0, 0.0);
    }
    let mut bv = v.clone(); // B = I

    // Compute <y, v>_B before projection
    let overlap_before = b_inner_product(y.as_slice(), bv.as_slice());
    assert!(
        (overlap_before.re - 1.0).abs() < 1e-10,
        "overlap before should be 1.0"
    );

    // Project
    deflation.project_single(&backend, &mut v, &mut bv);

    // Compute <y, v'>_B after projection - should be zero
    let overlap_after = b_inner_product(y.as_slice(), bv.as_slice());
    assert!(
        overlap_after.norm() < 1e-10,
        "overlap after projection should be ~0, got {:?}",
        overlap_after
    );

    // Verify v' = [0, 1, 1, 1] (the y component was removed)
    assert!((v.as_slice()[0].re - 0.0).abs() < 1e-10, "v[0] should be 0");
    assert!(
        (v.as_slice()[1].re - 1.0).abs() < 1e-10,
        "v[1] should still be 1"
    );
}

/// Test project_single with a complex-valued basis vector.
///
/// This is the critical test for the conjugation question.
/// If the conjugation is wrong, complex phases won't cancel properly.
#[test]
fn test_project_single_complex_basis() {
    let backend = MockProjectionBackend;
    let grid = test_grid();

    // Create a complex B-normalized basis vector
    // y = [1+i, 0, 0, 0] / ||y||_B, with B = I
    // ||y||_B = sqrt((1+i)^* (1+i)) = sqrt(2)
    let norm_factor = 1.0 / 2.0_f64.sqrt();
    let mut y = Field2D::zeros(grid);
    y.as_mut_slice()[0] = Complex64::new(norm_factor, norm_factor); // (1+i)/sqrt(2)

    let by = y.clone(); // B = I

    // Verify B-normalization: <y, y>_B = 1
    let y_norm_sq = b_inner_product(y.as_slice(), by.as_slice());
    assert!(
        (y_norm_sq.re - 1.0).abs() < 1e-10 && y_norm_sq.im.abs() < 1e-10,
        "y should be B-normalized, got <y,y>_B = {:?}",
        y_norm_sq
    );

    let mut deflation: DeflationSubspace<MockProjectionBackend> = DeflationSubspace::new();
    deflation.add_vector(&backend, &y, &by, 1.0, 0);

    // Create test vector v = [1, 0, 0, 0] (purely real)
    let mut v = Field2D::zeros(grid);
    v.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
    let mut bv = v.clone(); // B = I

    // Compute <y, v>_B before projection
    // <y, v>_B = y^H v = conj((1+i)/sqrt(2)) * 1 = (1-i)/sqrt(2)
    let overlap_before = b_inner_product(y.as_slice(), bv.as_slice());
    let expected = Complex64::new(norm_factor, -norm_factor);
    assert!(
        (overlap_before - expected).norm() < 1e-10,
        "overlap before should be (1-i)/sqrt(2), got {:?}",
        overlap_before
    );

    // Project
    deflation.project_single(&backend, &mut v, &mut bv);

    // Compute <y, v'>_B after projection - should be zero
    let overlap_after = b_inner_product(y.as_slice(), bv.as_slice());
    assert!(
        overlap_after.norm() < 1e-10,
        "overlap after projection should be ~0, got {:?} (norm={})",
        overlap_after,
        overlap_after.norm()
    );
}

/// Test project_single with purely imaginary basis vector.
#[test]
fn test_project_single_imaginary_basis() {
    let backend = MockProjectionBackend;
    let grid = test_grid();

    // y = [i, 0, 0, 0] (already normalized since |i|=1)
    let mut y = Field2D::zeros(grid);
    y.as_mut_slice()[0] = Complex64::new(0.0, 1.0);
    let by = y.clone();

    let mut deflation: DeflationSubspace<MockProjectionBackend> = DeflationSubspace::new();
    deflation.add_vector(&backend, &y, &by, 1.0, 0);

    // v = [1, 1, 0, 0]
    let mut v = Field2D::zeros(grid);
    v.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
    v.as_mut_slice()[1] = Complex64::new(1.0, 0.0);
    let mut bv = v.clone();

    // <y, v>_B = conj(i) * 1 = -i
    let overlap_before = b_inner_product(y.as_slice(), bv.as_slice());
    assert!(
        (overlap_before - Complex64::new(0.0, -1.0)).norm() < 1e-10,
        "overlap before should be -i, got {:?}",
        overlap_before
    );

    deflation.project_single(&backend, &mut v, &mut bv);

    let overlap_after = b_inner_product(y.as_slice(), bv.as_slice());
    assert!(
        overlap_after.norm() < 1e-10,
        "overlap after should be ~0, got {:?}",
        overlap_after
    );

    // v' should be [1 - (-i)*i, 1, 0, 0] = [1 - 1, 1, 0, 0] = [0, 1, 0, 0]
    assert!(
        v.as_slice()[0].norm() < 1e-10,
        "v[0] should be 0, got {:?}",
        v.as_slice()[0]
    );
    assert!(
        (v.as_slice()[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10,
        "v[1] should be 1, got {:?}",
        v.as_slice()[1]
    );
}

/// Test project_block_no_mass with complex basis.
///
/// This function uses the alternative formula:
///   c = (Bv)^H y  (then conjugated)
/// vs project_single which uses:
///   c = y^H (Bv) (then conjugated)
///
/// Both should give the same result.
#[test]
fn test_project_block_no_mass_complex() {
    let backend = MockProjectionBackend;
    let grid = test_grid();

    // Same complex basis as before
    let norm_factor = 1.0 / 2.0_f64.sqrt();
    let mut y = Field2D::zeros(grid);
    y.as_mut_slice()[0] = Complex64::new(norm_factor, norm_factor);
    let by = y.clone();

    let mut deflation: DeflationSubspace<MockProjectionBackend> = DeflationSubspace::new();
    deflation.add_vector(&backend, &y, &by, 1.0, 0);

    // v = [1, 0, 0, 0]
    let mut v = Field2D::zeros(grid);
    v.as_mut_slice()[0] = Complex64::new(1.0, 0.0);

    // Project using project_block_no_mass
    deflation.project_block_no_mass(&backend, std::slice::from_mut(&mut v));

    // Compute <y, v'>_B after projection
    // Since B = I, we just need y^H v'
    let overlap_after = b_inner_product(y.as_slice(), v.as_slice());
    assert!(
        overlap_after.norm() < 1e-10,
        "project_block_no_mass: overlap after should be ~0, got {:?}",
        overlap_after
    );
}

/// Test that project_single and project_block_no_mass give same results.
#[test]
fn test_projection_methods_equivalent() {
    let backend = MockProjectionBackend;
    let grid = test_grid();

    // Complex basis
    let mut y = Field2D::zeros(grid);
    y.as_mut_slice()[0] = Complex64::new(0.6, 0.8); // Not normalized
    let by = y.clone();

    // Normalize
    let norm_sq = b_inner_product(y.as_slice(), by.as_slice()).re;
    let norm = norm_sq.sqrt();
    for val in y.as_mut_slice() {
        *val /= norm;
    }
    let by = y.clone();

    let mut deflation: DeflationSubspace<MockProjectionBackend> = DeflationSubspace::new();
    deflation.add_vector(&backend, &y, &by, 1.0, 0);

    // Create identical test vectors
    let mut v1 = Field2D::zeros(grid);
    let mut v2 = Field2D::zeros(grid);
    for i in 0..4 {
        let val = Complex64::new(i as f64, (i as f64) * 0.5);
        v1.as_mut_slice()[i] = val;
        v2.as_mut_slice()[i] = val;
    }
    let mut bv1 = v1.clone();

    // Project with different methods
    deflation.project_single(&backend, &mut v1, &mut bv1);
    deflation.project_block_no_mass(&backend, std::slice::from_mut(&mut v2));

    // Results should be identical
    for i in 0..4 {
        let diff = (v1.as_slice()[i] - v2.as_slice()[i]).norm();
        assert!(
            diff < 1e-10,
            "Methods differ at index {}: {:?} vs {:?}",
            i,
            v1.as_slice()[i],
            v2.as_slice()[i]
        );
    }
}

/// Test projection with multiple basis vectors.
#[test]
fn test_project_multiple_basis_vectors() {
    let backend = MockProjectionBackend;
    let grid = test_grid();

    // Two orthonormal basis vectors
    let mut y1 = Field2D::zeros(grid);
    y1.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
    let by1 = y1.clone();

    let mut y2 = Field2D::zeros(grid);
    y2.as_mut_slice()[1] = Complex64::new(1.0, 0.0);
    let by2 = y2.clone();

    let mut deflation: DeflationSubspace<MockProjectionBackend> = DeflationSubspace::new();
    deflation.add_vector(&backend, &y1, &by1, 1.0, 0);
    deflation.add_vector(&backend, &y2, &by2, 2.0, 1);

    // v = [1, 1, 1, 1]
    let mut v = Field2D::zeros(grid);
    for val in v.as_mut_slice() {
        *val = Complex64::new(1.0, 0.0);
    }
    let mut bv = v.clone();

    deflation.project_single(&backend, &mut v, &mut bv);

    // Should be orthogonal to both y1 and y2
    let overlap1 = b_inner_product(y1.as_slice(), bv.as_slice());
    let overlap2 = b_inner_product(y2.as_slice(), bv.as_slice());

    assert!(overlap1.norm() < 1e-10, "overlap with y1 should be ~0");
    assert!(overlap2.norm() < 1e-10, "overlap with y2 should be ~0");

    // v' should be [0, 0, 1, 1]
    assert!(v.as_slice()[0].norm() < 1e-10);
    assert!(v.as_slice()[1].norm() < 1e-10);
    assert!((v.as_slice()[2].re - 1.0).abs() < 1e-10);
    assert!((v.as_slice()[3].re - 1.0).abs() < 1e-10);
}
