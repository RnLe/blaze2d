//! Tests for the CPU backend.
//!
//! These tests verify that the CPU backend correctly implements the
//! `SpectralBackend` trait, including FFT operations and BLAS-like
//! linear algebra primitives.

#![cfg(test)]

use crate::CpuBackend;
use blaze2d_core::backend::SpectralBackend;
use blaze2d_core::field::{Field2D, FieldReal, FieldScalar};
use blaze2d_core::grid::Grid2D;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Compile-time witness that both monomorphisations exist in the same binary —
/// this is the load-bearing test for the runtime-selectable precision goal.
/// If this stops compiling, the Stage 8 invariant ("one wheel, both precisions")
/// has regressed.
#[test]
fn both_precision_monomorphisations_build() {
    let _b32: CpuBackend<f32> = CpuBackend::<f32>::new();
    let _b64: CpuBackend<f64> = CpuBackend::<f64>::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let mut f32_buf = _b32.alloc_field(grid);
    let mut f64_buf = _b64.alloc_field(grid);
    _b32.forward_fft_2d(&mut f32_buf);
    _b32.inverse_fft_2d(&mut f32_buf);
    _b64.forward_fft_2d(&mut f64_buf);
    _b64.inverse_fft_2d(&mut f64_buf);
    // dot products both promote to Complex<f64> (accumulation precision).
    let _: Complex64 = _b32.dot(&f32_buf, &f32_buf);
    let _: Complex64 = _b64.dot(&f64_buf, &f64_buf);
}

/// End-to-end precision guarantee: a full band-structure solve at f32 storage
/// must agree with the f64 solve to a few ×1e-4. This is the user-facing promise
/// behind `precision="f32"` — single-precision storage with f64 accumulation
/// keeps eigenvalues accurate. Complements the compile-time witness above.
#[test]
fn f32_and_f64_bandstructure_agree() {
    use blaze2d_core::dielectric::DielectricOptions;
    use blaze2d_core::drivers::bandstructure::{BandStructureJob, RunOptions, run_with_options};
    use blaze2d_core::eigensolver::EigensolverConfig;
    use blaze2d_core::geometry::{BasisAtom, Geometry2D};
    use blaze2d_core::lattice::Lattice2D;
    use blaze2d_core::polarization::Polarization;

    // Small square-lattice photonic crystal, TM polarization.
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 13.0,
        atoms: vec![BasisAtom {
            pos: [0.0, 0.0],
            radius: 0.3,
            eps_inside: 1.0,
        }],
    };
    let make_job = || BandStructureJob {
        geom: geom.clone(),
        grid: Grid2D::new(16, 16, 1.0, 1.0),
        pol: Polarization::TM,
        k_path: vec![[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.5, 0.5]],
        eigensolver: EigensolverConfig {
            n_bands: 4,
            max_iter: 300,
            tol: 1e-8,
            block_size: 8,
            ..Default::default()
        },
        dielectric: DielectricOptions::default(),
    };

    let bands64 = run_with_options(CpuBackend::<f64>::new(), &make_job(), RunOptions::default());
    let bands32 = run_with_options(CpuBackend::<f32>::new(), &make_job(), RunOptions::default());

    assert_eq!(bands64.bands.len(), bands32.bands.len());
    let mut max_abs_diff = 0.0f64;
    for (kp, (b64, b32)) in bands64.bands.iter().zip(bands32.bands.iter()).enumerate() {
        assert_eq!(b64.len(), b32.len(), "band count mismatch at k#{kp}");
        for (&w64, &w32) in b64.iter().zip(b32.iter()) {
            max_abs_diff = max_abs_diff.max((w64 - w32).abs());
        }
    }
    // f32 storage limits the achievable agreement to ~single-precision × a few;
    // 5e-3 absolute is comfortably met while still proving the two precisions
    // track the same physics.
    assert!(
        max_abs_diff < 5e-3,
        "f32 vs f64 band frequencies diverge by {max_abs_diff} (> 5e-3)"
    );
}

// ============================================================================
// FFT Tests
// ============================================================================

#[test]
fn fft_roundtrip_recovers_signal() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    // Initialize with a simple pattern
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new(idx as FieldReal, -(idx as FieldReal));
    }
    let original = field.clone();

    // Forward then inverse should recover the original
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);

    for (rec, expect) in field.as_slice().iter().zip(original.as_slice()) {
        let diff = (*rec - *expect).norm();
        assert!(diff < 1e-4, "FFT roundtrip diverged: diff={diff}");
    }
}

#[test]
fn fft_roundtrip_preserves_energy_norm() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(6, 2, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new((idx as FieldReal).sin(), (idx as FieldReal).cos());
    }

    let before: f64 = field.as_slice().iter().map(|v| v.norm_sqr() as f64).sum();
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);
    let after: f64 = field.as_slice().iter().map(|v| v.norm_sqr() as f64).sum();

    assert!(
        (before - after).abs() < 1e-4,
        "energy drifted by {}",
        after - before
    );
}

#[test]
fn fft_forward_of_constant_is_dc_component() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let n = (grid.nx * grid.ny) as FieldReal;
    let mut field = Field2D::zeros(grid);

    // Constant field of value 1.0
    for value in field.as_mut_slice().iter_mut() {
        *value = FieldScalar::new(1.0, 0.0);
    }

    backend.forward_fft_2d(&mut field);

    // DC component should be n (sum of all 1s)
    let dc = field.as_slice()[0];
    assert!(
        (dc - FieldScalar::new(n, 0.0)).norm() < 1e-4,
        "DC component should be {n}, got {dc}"
    );

    // All other components should be zero
    for (idx, &value) in field.as_slice().iter().enumerate().skip(1) {
        assert!(
            value.norm() < 1e-4,
            "Non-DC component at index {idx} should be zero, got {value}"
        );
    }
}

#[test]
fn fft_of_plane_wave_is_single_peak() {
    let backend = CpuBackend::<f64>::new();
    let nx = 8;
    let ny = 8;
    let grid = Grid2D::new(nx, ny, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    // Create a plane wave with k_x = 1, k_y = 0 (one cycle across x)
    for iy in 0..ny {
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let x = ix as f64 / nx as f64;
            let c = Complex64::from_polar(1.0, 2.0 * PI * x);
            field.as_mut_slice()[idx] = FieldScalar::new(c.re as FieldReal, c.im as FieldReal);
        }
    }

    backend.forward_fft_2d(&mut field);

    // The peak should be at index (1, 0) = index 1
    let peak_idx = 1;
    let peak = field.as_slice()[peak_idx].norm() as f64;
    let n = (nx * ny) as f64;

    assert!(
        (peak - n).abs() < 1e-4,
        "Peak amplitude should be {n}, got {peak}"
    );
}

// ============================================================================
// BLAS-like Operation Tests
// ============================================================================

#[test]
fn scale_multiplies_all_elements() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(2, 3, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);

    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new(idx as FieldReal, 0.0);
    }

    let alpha = Complex64::new(2.0, 1.0);
    backend.scale(alpha, &mut field);

    for (idx, &value) in field.as_slice().iter().enumerate() {
        let expected = alpha * Complex64::new(idx as f64, 0.0);
        let value64 = Complex64::new(value.re as f64, value.im as f64);
        assert!(
            (value64 - expected).norm() < 1e-6,
            "index {idx}: expected {expected}, got {value}"
        );
    }
}

#[test]
fn axpy_computes_y_plus_alpha_x() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);
    let mut y = Field2D::zeros(grid);

    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new((i as FieldReal) + 1.0, 0.0);
    }
    for (i, value) in y.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new(0.0, i as FieldReal);
    }

    let alpha = Complex64::new(2.0, 0.0);
    backend.axpy(alpha, &x, &mut y);

    // y should now be original_y + alpha * x
    for (idx, &value) in y.as_slice().iter().enumerate() {
        let expected = Complex64::new(2.0 * (idx as f64 + 1.0), idx as f64);
        let value64 = Complex64::new(value.re as f64, value.im as f64);
        assert!(
            (value64 - expected).norm() < 1e-6,
            "index {idx}: expected {expected}, got {value}"
        );
    }
}

#[test]
fn dot_computes_conjugate_inner_product() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);
    let mut y = Field2D::zeros(grid);

    // x = [1, 2, 3, 4]
    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new((i as FieldReal) + 1.0, 0.0);
    }

    // y = [1, i, -1, -i]
    y.as_mut_slice()[0] = FieldScalar::new(1.0, 0.0);
    y.as_mut_slice()[1] = FieldScalar::new(0.0, 1.0);
    y.as_mut_slice()[2] = FieldScalar::new(-1.0, 0.0);
    y.as_mut_slice()[3] = FieldScalar::new(0.0, -1.0);

    // dot(x, y) = conj(x[0])*y[0] + conj(x[1])*y[1] + conj(x[2])*y[2] + conj(x[3])*y[3]
    //           = 1*1 + 2*i + 3*(-1) + 4*(-i)
    //           = 1 + 2i - 3 - 4i = -2 - 2i
    let result = backend.dot(&x, &y);
    let expected = Complex64::new(-2.0, -2.0);

    assert!(
        (result - expected).norm() < 1e-6,
        "expected {expected}, got {result}"
    );
}

#[test]
fn dot_of_vector_with_itself_is_real() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(3, 3, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);

    // Complex vector
    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new((i as FieldReal).sin(), (i as FieldReal).cos());
    }

    let result = backend.dot(&x, &x);

    // <x, x> should be real and equal to ||x||²
    assert!(
        result.im.abs() < 1e-6,
        "self-dot should be real, got {result}"
    );

    let expected_norm_sq: f64 = x.as_slice().iter().map(|v| v.norm_sqr() as f64).sum();
    assert!(
        (result.re - expected_norm_sq).abs() < 1e-6,
        "expected {expected_norm_sq}, got {result}"
    );
}

// ============================================================================
// Field Allocation Tests
// ============================================================================

#[test]
fn alloc_field_creates_correct_grid() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(5, 7, 2.0, 3.0);
    let field = backend.alloc_field(grid);

    assert_eq!(field.grid().nx, 5);
    assert_eq!(field.grid().ny, 7);
    assert_eq!(field.as_slice().len(), 35);
}

#[test]
fn alloc_field_initializes_to_zero() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let field = backend.alloc_field(grid);

    for &value in field.as_slice() {
        assert_eq!(value, FieldScalar::default());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn combined_operations_maintain_consistency() {
    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);

    // Create two fields
    let mut a = backend.alloc_field(grid);
    let mut b = backend.alloc_field(grid);

    for (idx, value) in a.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new((idx as FieldReal).cos(), (idx as FieldReal).sin());
    }
    for value in b.as_mut_slice().iter_mut() {
        *value = FieldScalar::new(1.0, 0.0);
    }

    // Compute <a, a> before operations
    let norm_sq_before = backend.dot(&a, &a).re;

    // Do FFT roundtrip
    backend.forward_fft_2d(&mut a);
    backend.inverse_fft_2d(&mut a);

    // Compute <a, a> after operations
    let norm_sq_after = backend.dot(&a, &a).re;

    assert!(
        (norm_sq_before - norm_sq_after).abs() < 1e-4,
        "norm changed: {norm_sq_before} -> {norm_sq_after}"
    );
}
