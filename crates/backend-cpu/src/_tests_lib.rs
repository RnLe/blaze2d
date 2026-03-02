//! Tests for the CPU backend.
//!
//! These tests verify that the CPU backend correctly implements the
//! `SpectralBackend` trait, including FFT operations and BLAS-like
//! linear algebra primitives.

#![cfg(test)]

use crate::CpuBackend;
use mpb2d_core::backend::SpectralBackend;
use mpb2d_core::field::Field2D;
use mpb2d_core::grid::Grid2D;
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================================
// FFT Tests
// ============================================================================

#[test]
fn fft_roundtrip_recovers_signal() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    // Initialize with a simple pattern
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(idx as f64, -(idx as f64));
    }
    let original = field.clone();

    // Forward then inverse should recover the original
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);

    for (rec, expect) in field.as_slice().iter().zip(original.as_slice()) {
        let diff = (*rec - *expect).norm();
        assert!(diff < 1e-9, "FFT roundtrip diverged: diff={diff}");
    }
}

#[test]
fn fft_roundtrip_preserves_energy_norm() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(6, 2, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new((idx as f64).sin(), (idx as f64).cos());
    }

    let before = field.as_slice().iter().map(|v| v.norm_sqr()).sum::<f64>();
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);
    let after = field.as_slice().iter().map(|v| v.norm_sqr()).sum::<f64>();

    assert!(
        (before - after).abs() < 1e-9,
        "energy drifted by {}",
        after - before
    );
}

#[test]
fn fft_forward_of_constant_is_dc_component() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let n = (grid.nx * grid.ny) as f64;
    let mut field = Field2D::zeros(grid);

    // Constant field of value 1.0
    for value in field.as_mut_slice().iter_mut() {
        *value = Complex64::new(1.0, 0.0);
    }

    backend.forward_fft_2d(&mut field);

    // DC component should be n (sum of all 1s)
    let dc = field.as_slice()[0];
    assert!(
        (dc - Complex64::new(n, 0.0)).norm() < 1e-9,
        "DC component should be {n}, got {dc}"
    );

    // All other components should be zero
    for (idx, &value) in field.as_slice().iter().enumerate().skip(1) {
        assert!(
            value.norm() < 1e-9,
            "Non-DC component at index {idx} should be zero, got {value}"
        );
    }
}

#[test]
fn fft_of_plane_wave_is_single_peak() {
    let backend = CpuBackend::new();
    let nx = 8;
    let ny = 8;
    let grid = Grid2D::new(nx, ny, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    // Create a plane wave with k_x = 1, k_y = 0 (one cycle across x)
    for iy in 0..ny {
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let x = ix as f64 / nx as f64;
            field.as_mut_slice()[idx] = Complex64::from_polar(1.0, 2.0 * PI * x);
        }
    }

    backend.forward_fft_2d(&mut field);

    // The peak should be at index (1, 0) = index 1
    let peak_idx = 1;
    let peak = field.as_slice()[peak_idx].norm();
    let n = (nx * ny) as f64;

    assert!(
        (peak - n).abs() < 1e-6,
        "Peak amplitude should be {n}, got {peak}"
    );
}

// ============================================================================
// BLAS-like Operation Tests
// ============================================================================

#[test]
fn scale_multiplies_all_elements() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(2, 3, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(idx as f64, 0.0);
    }

    let alpha = Complex64::new(2.0, 1.0);
    backend.scale(alpha, &mut field);

    for (idx, &value) in field.as_slice().iter().enumerate() {
        let expected = alpha * Complex64::new(idx as f64, 0.0);
        assert!(
            (value - expected).norm() < 1e-12,
            "index {idx}: expected {expected}, got {value}"
        );
    }
}

#[test]
fn axpy_computes_y_plus_alpha_x() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);
    let mut y = Field2D::zeros(grid);

    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(i as f64 + 1.0, 0.0);
    }
    for (i, value) in y.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(0.0, i as f64);
    }

    let alpha = Complex64::new(2.0, 0.0);
    backend.axpy(alpha, &x, &mut y);

    // y should now be original_y + alpha * x
    for (idx, &value) in y.as_slice().iter().enumerate() {
        let expected = Complex64::new(2.0 * (idx as f64 + 1.0), idx as f64);
        assert!(
            (value - expected).norm() < 1e-12,
            "index {idx}: expected {expected}, got {value}"
        );
    }
}

#[test]
fn dot_computes_conjugate_inner_product() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);
    let mut y = Field2D::zeros(grid);

    // x = [1, 2, 3, 4]
    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(i as f64 + 1.0, 0.0);
    }

    // y = [1, i, -1, -i]
    y.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
    y.as_mut_slice()[1] = Complex64::new(0.0, 1.0);
    y.as_mut_slice()[2] = Complex64::new(-1.0, 0.0);
    y.as_mut_slice()[3] = Complex64::new(0.0, -1.0);

    // dot(x, y) = conj(x[0])*y[0] + conj(x[1])*y[1] + conj(x[2])*y[2] + conj(x[3])*y[3]
    //           = 1*1 + 2*i + 3*(-1) + 4*(-i)
    //           = 1 + 2i - 3 - 4i = -2 - 2i
    let result = backend.dot(&x, &y);
    let expected = Complex64::new(-2.0, -2.0);

    assert!(
        (result - expected).norm() < 1e-12,
        "expected {expected}, got {result}"
    );
}

#[test]
fn dot_of_vector_with_itself_is_real() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(3, 3, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);

    // Complex vector
    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new((i as f64).sin(), (i as f64).cos());
    }

    let result = backend.dot(&x, &x);

    // <x, x> should be real and equal to ||x||²
    assert!(
        result.im.abs() < 1e-12,
        "self-dot should be real, got {result}"
    );

    let expected_norm_sq: f64 = x.as_slice().iter().map(|v| v.norm_sqr()).sum();
    assert!(
        (result.re - expected_norm_sq).abs() < 1e-12,
        "expected {expected_norm_sq}, got {result}"
    );
}

// ============================================================================
// Field Allocation Tests
// ============================================================================

#[test]
fn alloc_field_creates_correct_grid() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(5, 7, 2.0, 3.0);
    let field = backend.alloc_field(grid);

    assert_eq!(field.grid().nx, 5);
    assert_eq!(field.grid().ny, 7);
    assert_eq!(field.as_slice().len(), 35);
}

#[test]
fn alloc_field_initializes_to_zero() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let field = backend.alloc_field(grid);

    for &value in field.as_slice() {
        assert_eq!(value, Complex64::ZERO);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn combined_operations_maintain_consistency() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);

    // Create two fields
    let mut a = backend.alloc_field(grid);
    let mut b = backend.alloc_field(grid);

    for (idx, value) in a.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new((idx as f64).cos(), (idx as f64).sin());
    }
    for value in b.as_mut_slice().iter_mut() {
        *value = Complex64::new(1.0, 0.0);
    }

    // Compute <a, a> before operations
    let norm_sq_before = backend.dot(&a, &a).re;

    // Do FFT roundtrip
    backend.forward_fft_2d(&mut a);
    backend.inverse_fft_2d(&mut a);

    // Compute <a, a> after operations
    let norm_sq_after = backend.dot(&a, &a).re;

    assert!(
        (norm_sq_before - norm_sq_after).abs() < 1e-9,
        "norm changed: {norm_sq_before} -> {norm_sq_after}"
    );
}

// ============================================================================
// EAOperator + single_solve Integration Test
// ============================================================================

/// Test that EAOperator works with the single_solve driver.
///
/// This verifies the full integration path:
/// - EAOperator implements LinearOperator correctly
/// - single_solve driver orchestrates the eigensolver
/// - CpuBackend provides working FFT and linear algebra
#[test]
fn ea_operator_single_solve_integration() {
    use mpb2d_core::drivers::single_solve::{solve, SingleSolveJob};
    use mpb2d_core::operators::EAOperator;

    let backend = CpuBackend::new();
    let nx = 16;
    let ny = 16;
    let n = nx * ny;
    let dx = 1.0;
    let dy = 1.0;
    let eta = 0.1;

    // Simple harmonic potential well centered in the domain
    let lx = nx as f64 * dx;
    let ly = ny as f64 * dy;
    let cx = lx / 2.0;
    let cy = ly / 2.0;
    let l_scale = (lx * ly).sqrt();

    let potential: Vec<f64> = (0..n)
        .map(|idx| {
            let i = idx / ny;
            let j = idx % ny;
            let x = i as f64 * dx;
            let y = j as f64 * dy;
            let r_sq = (x - cx).powi(2) + (y - cy).powi(2);
            // Positive definite potential
            1.0 + 0.1 * r_sq / l_scale.powi(2)
        })
        .collect();

    // Constant identity inverse mass tensor
    let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

    let mut op = EAOperator::new(
        backend,
        nx,
        ny,
        dx,
        dy,
        eta,
        potential,
        mass_inv,
        None, // no drift term
        0.0,  // omega_ref
    );

    // Solve for the lowest 4 eigenvalues
    let job = SingleSolveJob::new(4)
        .with_tolerance(1e-6)
        .with_max_iterations(200);

    let result = solve(&mut op, None, &job);

    // Assertions:
    // 1. We got the requested number of eigenvalues
    assert_eq!(
        result.eigenvalues.len(),
        4,
        "Should return exactly 4 eigenvalues"
    );

    // 2. All eigenvalues should be positive (positive definite operator)
    for (i, &ev) in result.eigenvalues.iter().enumerate() {
        assert!(
            ev > 0.0,
            "Eigenvalue {} should be positive, got {}",
            i,
            ev
        );
    }

    // 3. Eigenvalues should be in ascending order
    for i in 0..result.eigenvalues.len() - 1 {
        assert!(
            result.eigenvalues[i] <= result.eigenvalues[i + 1] + 1e-10,
            "Eigenvalues not sorted: {} > {}",
            result.eigenvalues[i],
            result.eigenvalues[i + 1]
        );
    }

    // 4. Ground state should be close to the minimum potential (~1.0)
    // The kinetic term adds a small positive contribution
    assert!(
        result.eigenvalues[0] > 0.9 && result.eigenvalues[0] < 2.0,
        "Ground state eigenvalue should be near the potential minimum, got {}",
        result.eigenvalues[0]
    );

    // 5. Should converge reasonably quickly for this simple problem
    assert!(
        result.iterations < 150,
        "Should converge in < 150 iterations, took {}",
        result.iterations
    );

    // 6. Verify eigenvectors are orthonormal under B (identity mass)
    let backend = CpuBackend::new();
    for i in 0..result.eigenvectors.len() {
        // Check normalization
        let norm_sq = backend.dot(&result.eigenvectors[i], &result.eigenvectors[i]).re;
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "Eigenvector {} not normalized: ||v||² = {}",
            i,
            norm_sq
        );

        // Check orthogonality
        for j in (i + 1)..result.eigenvectors.len() {
            let overlap = backend.dot(&result.eigenvectors[i], &result.eigenvectors[j]).norm();
            assert!(
                overlap < 1e-4,
                "Eigenvectors {} and {} not orthogonal: <v_i, v_j> = {}",
                i,
                j,
                overlap
            );
        }
    }
}
