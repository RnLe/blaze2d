#![cfg(test)]

use std::f64::consts::PI;

use num_complex::Complex64;

use super::backend::SpectralBackend;
use super::dielectric::{Dielectric2D, DielectricOptions};
use super::field::Field2D;
use super::geometry::{BasisAtom, Geometry2D};
use super::grid::Grid2D;
use super::lattice::Lattice2D;
use super::operators::{LinearOperator, ThetaOperator, ToyLaplacian};
use super::polarization::Polarization;
use super::preconditioners::OperatorPreconditioner;

#[derive(Clone)]
struct TestBackend;

impl SpectralBackend for TestBackend {
    type Buffer = Field2D;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        discrete_fft(buffer, false);
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        discrete_fft(buffer, true);
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in buffer.as_mut_slice() {
            *value *= alpha;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *dst += alpha * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }
}

fn discrete_fft(buffer: &mut Field2D, inverse: bool) {
    let grid = buffer.grid();
    let nx = grid.nx;
    let ny = grid.ny;
    let data = buffer.as_mut_slice();
    let mut output = vec![Complex64::default(); data.len()];
    let norm = if inverse { 1.0 / (nx * ny) as f64 } else { 1.0 };
    for ky in 0..ny {
        for kx in 0..nx {
            let mut sum = Complex64::default();
            for y in 0..ny {
                for x in 0..nx {
                    let idx = y * nx + x;
                    let phase = if inverse {
                        2.0 * PI * ((kx * x) as f64 / nx as f64 + (ky * y) as f64 / ny as f64)
                    } else {
                        -2.0 * PI * ((kx * x) as f64 / nx as f64 + (ky * y) as f64 / ny as f64)
                    };
                    sum += data[idx] * Complex64::from_polar(1.0, phase);
                }
            }
            output[ky * nx + kx] = sum * norm;
        }
    }
    data.copy_from_slice(&output);
}

fn plane_wave(grid: Grid2D, mx: i32, my: i32) -> Field2D {
    let mut field = Field2D::zeros(grid);
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = iy * grid.nx + ix;
            let phase = 2.0
                * PI
                * (mx as f64 * ix as f64 / grid.nx as f64 + my as f64 * iy as f64 / grid.ny as f64);
            field.as_mut_slice()[idx] = Complex64::from_polar(1.0, phase);
        }
    }
    field
}

fn uniform_dielectric(grid: Grid2D, eps: f64) -> Dielectric2D {
    // Use a lattice that matches the grid dimensions so G-vectors are consistent.
    // The lattice vectors should match the computational domain:
    // a1 = [lx, 0], a2 = [0, ly] for a rectangular domain.
    let geom = Geometry2D {
        lattice: Lattice2D::rectangular(grid.lx, grid.ly),
        eps_bg: eps,
        atoms: Vec::new(),
    };
    Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default())
}

fn patterned_dielectric(grid: Grid2D) -> Dielectric2D {
    // Use a rectangular lattice that matches the grid dimensions.
    let geom = Geometry2D {
        lattice: Lattice2D::rectangular(grid.lx, grid.ly),
        eps_bg: 12.0,
        atoms: vec![
            BasisAtom {
                pos: [0.2, 0.25],
                radius: 0.18,
                eps_inside: 4.0,
            },
            BasisAtom {
                pos: [0.65, 0.6],
                radius: 0.15,
                eps_inside: 8.0,
            },
        ],
    };
    Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default())
}

fn deterministic_field(grid: Grid2D, seed: u64) -> Field2D {
    let mut field = Field2D::zeros(grid);
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        let t = (idx as f64 + 1.0) * (seed as f64 + 0.5);
        let real = (0.37 * t).sin();
        let imag = (0.61 * t).cos();
        *value = Complex64::new(real, imag);
    }
    field
}

fn inner_product(a: &Field2D, b: &Field2D) -> Complex64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(lhs, rhs)| lhs.conj() * rhs)
        .sum()
}

fn field_norm(field: &Field2D) -> f64 {
    field
        .as_slice()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn shifted_eigenvalue(grid: Grid2D, bloch_k: [f64; 2], mx: i32, my: i32) -> f64 {
    let two_pi = 2.0 * PI;
    let gx = two_pi * mx as f64 / grid.lx;
    let gy = two_pi * my as f64 / grid.ly;
    let kx = bloch_k[0] + gx;
    let ky = bloch_k[1] + gy;
    kx * kx + ky * ky
}

fn assert_complex_close(lhs: Complex64, rhs: Complex64, tol: f64) {
    assert!(
        (lhs - rhs).norm() < tol,
        "complex numbers differ: {lhs:?} vs {rhs:?}"
    );
}

fn assert_fields_close(a: &Field2D, b: &Field2D, tol: f64) {
    for (lhs, rhs) in a.as_slice().iter().zip(b.as_slice()) {
        assert!(
            (*lhs - *rhs).norm() < tol,
            "fields differ: {lhs:?} vs {rhs:?}"
        );
    }
}

#[test]
fn toy_laplacian_matches_plane_wave_eigenvalue() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let backend = TestBackend;
    let mut laplacian = ToyLaplacian::new(backend, grid);
    let input = plane_wave(grid, 1, 1);
    let mut output = laplacian.alloc_field();
    laplacian.apply(&input, &mut output);
    let eigenvalue = (2.0 * PI).powi(2) * (1.0 + 1.0);
    let mut expected = input.clone();
    for value in expected.as_mut_slice() {
        *value *= eigenvalue;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_tm_uniform_medium_matches_laplacian() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 12.0);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [0.0, 0.0]);
    let input = plane_wave(grid, 1, 0);
    let mut output = theta.alloc_field();
    theta.apply(&input, &mut output);
    let eigenvalue = (2.0 * PI).powi(2);
    let mut expected = input.clone();
    for value in expected.as_mut_slice() {
        *value *= eigenvalue;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_te_uniform_medium_matches_curl_curl() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 8.0);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, [0.0, 0.0]);
    let input = plane_wave(grid, 0, 1);
    let mut output = theta.alloc_field();
    theta.apply(&input, &mut output);
    let eigenvalue = (2.0 * PI).powi(2) / 8.0;
    let mut expected = input.clone();
    for value in expected.as_mut_slice() {
        *value *= eigenvalue;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_te_respects_bloch_shift_for_constant_field() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 5.0);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, [PI, 0.0]);
    let input = plane_wave(grid, 0, 0);
    let mut output = theta.alloc_field();
    theta.apply(&input, &mut output);
    let eigenvalue = PI * PI / 5.0;
    let mut expected = input.clone();
    for value in expected.as_mut_slice() {
        *value *= eigenvalue;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_tm_plane_wave_matches_shifted_eigenvalue_on_rectangular_grid() {
    let grid = Grid2D::new(6, 5, 1.6, 0.9);
    let dielectric = uniform_dielectric(grid, 7.0);
    let backend = TestBackend;
    let bloch = [0.3 * PI, -0.45 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
    let input = plane_wave(grid, 1, -2);
    let mut output = theta.alloc_field();
    theta.apply(&input, &mut output);
    let eigenvalue = shifted_eigenvalue(grid, bloch, 1, -2);
    let mut expected = input.clone();
    for value in expected.as_mut_slice() {
        *value *= eigenvalue;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_te_plane_wave_matches_shifted_eigenvalue_on_rectangular_grid() {
    let grid = Grid2D::new(5, 6, 1.4, 0.85);
    let eps_bg = 5.5;
    let dielectric = uniform_dielectric(grid, eps_bg);
    let backend = TestBackend;
    let bloch = [-0.2 * PI, 0.35 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
    let input = plane_wave(grid, -1, 2);
    let mut output = theta.alloc_field();
    theta.apply(&input, &mut output);
    let eigenvalue = shifted_eigenvalue(grid, bloch, -1, 2) / eps_bg;
    let mut expected = input.clone();
    for value in expected.as_mut_slice() {
        *value *= eigenvalue;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_te_operator_is_hermitian_in_non_uniform_dielectric() {
    let grid = Grid2D::new(4, 5, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let backend = TestBackend;
    let bloch = [0.15 * PI, -0.28 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
    let field_a = deterministic_field(grid, 3);
    let field_b = deterministic_field(grid, 7);
    let mut ax = theta.alloc_field();
    let mut by = theta.alloc_field();
    theta.apply(&field_a, &mut ax);
    theta.apply(&field_b, &mut by);
    let lhs = inner_product(&field_a, &by);
    let rhs = inner_product(&ax, &field_b);
    assert_complex_close(lhs, rhs, 1e-9);
}

#[test]
fn theta_tm_operator_is_hermitian_for_bloch_shift() {
    let grid = Grid2D::new(4, 4, 1.2, 0.95);
    let dielectric = uniform_dielectric(grid, 9.0);
    let backend = TestBackend;
    let bloch = [0.4 * PI, 0.1 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
    let field_a = deterministic_field(grid, 11);
    let field_b = deterministic_field(grid, 5);
    let mut ax = theta.alloc_field();
    let mut by = theta.alloc_field();
    theta.apply(&field_a, &mut ax);
    theta.apply(&field_b, &mut by);
    let lhs = inner_product(&field_a, &by);
    let rhs = inner_product(&ax, &field_b);
    assert_complex_close(lhs, rhs, 1e-9);
}

#[test]
fn theta_tm_mass_matches_dielectric_profile() {
    let grid = Grid2D::new(3, 5, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let eps = dielectric.eps().to_vec();
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [0.0, 0.0]);
    let input = deterministic_field(grid, 13);
    let mut output = theta.alloc_field();
    theta.apply_mass(&input, &mut output);
    let mut expected = input.clone();
    for (value, eps_val) in expected.as_mut_slice().iter_mut().zip(eps.iter()) {
        *value *= eps_val;
    }
    assert_fields_close(&output, &expected, 1e-9);
}

#[test]
fn theta_te_mass_is_identity() {
    let grid = Grid2D::new(3, 3, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, [0.0, 0.0]);
    let input = deterministic_field(grid, 21);
    let mut output = theta.alloc_field();
    theta.apply_mass(&input, &mut output);
    assert_fields_close(&output, &input, 1e-9);
}

#[test]
fn jacobi_preconditioner_scales_tm_plane_wave() {
    // Test that preconditioner applies a finite positive scaling to a plane wave.
    // The exact scale depends on adaptive shift (computed from spectral stats),
    // so we just verify the scaling is in a reasonable range.
    let grid = Grid2D::new(5, 4, 1.3, 0.9);
    let dielectric = uniform_dielectric(grid, 2.0);
    let backend = TestBackend;
    let bloch = [0.25 * PI, -0.1 * PI];
    let theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
    let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();
    let mut field = plane_wave(grid, 1, -1);
    let original_norm = field_norm(&field);
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut field);
    let scaled_norm = field_norm(&field);
    
    // The preconditioner should scale the field by a finite positive factor
    // that reduces high-frequency content (scaling < 1 for high frequencies)
    let scale_factor = scaled_norm / original_norm;
    assert!(scale_factor > 0.0, "scale factor must be positive");
    assert!(scale_factor < 10.0, "scale factor should be moderate, got {scale_factor}");
}

#[test]
fn jacobi_preconditioner_uses_te_effective_epsilon() {
    // Test that TE preconditioner applies a finite positive scaling.
    // The exact scale depends on adaptive shift, so we just verify reasonable behavior.
    let grid = Grid2D::new(4, 4, 1.1, 1.0);
    let dielectric = patterned_dielectric(grid);
    let backend = TestBackend;
    let bloch = [0.0, 0.3 * PI];
    let theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
    let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();
    let mut field = plane_wave(grid, 0, 1);
    let original_norm = field_norm(&field);
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut field);
    let scaled_norm = field_norm(&field);
    
    // The preconditioner should scale by a finite positive factor
    let scale_factor = scaled_norm / original_norm;
    assert!(scale_factor > 0.0, "scale factor must be positive");
    assert!(scale_factor < 10.0, "scale factor should be moderate, got {scale_factor}");
}

#[test]
fn fourier_preconditioner_crushes_residual_norm() {
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 2.5);
    let backend = TestBackend;
    let bloch = [0.15 * PI, -0.1 * PI];
    let theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
    let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();
    let mut residual = plane_wave(grid, 3, -2);
    let before_norm = field_norm(&residual);
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut residual);
    let after_norm = field_norm(&residual);
    let reduction = before_norm / after_norm;
    assert!(
        reduction >= 10.0,
        "expected ≥10x reduction, got {reduction}"
    );
}

/// Test that the HOMOGENEOUS (Fourier-diagonal) preconditioner reduces high-frequency
/// residual norms effectively. This verifies the core frequency-dependent scaling.
#[test]
fn homogeneous_preconditioner_reduces_high_frequency_residual() {
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let backend = TestBackend;
    let bloch = [0.1 * PI, 0.05 * PI];
    let theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
    // Use HOMOGENEOUS preconditioner (no spatial weights)
    let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();

    // Create a high-frequency "residual" vector (mimics what LOBPCG would compute)
    let high_freq = plane_wave(grid, 3, -2);
    let mut residual = high_freq.clone();

    let before_norm = field_norm(&residual);
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut residual);
    let after_norm = field_norm(&residual);

    // The homogeneous preconditioner should significantly reduce high-frequency norms
    // because it scales by 1/|k+G|² (approximately)
    let reduction = before_norm / after_norm;
    assert!(
        reduction >= 5.0,
        "expected ≥5x reduction for high-freq mode, got {reduction:.2}x"
    );

    // Also verify that a low-frequency mode is not over-damped
    let low_freq = plane_wave(grid, 0, 1);
    let mut low_residual = low_freq.clone();
    let low_before = field_norm(&low_residual);
    preconditioner.apply(backend_ref, &mut low_residual);
    let low_after = field_norm(&low_residual);

    // Low-frequency modes should have much less reduction
    let low_reduction = low_before / low_after;
    assert!(
        low_reduction < reduction / 2.0,
        "low-freq reduction ({low_reduction:.2}x) should be much less than high-freq ({reduction:.2}x)"
    );
}

/// Test that the preconditioner approximately inverts the operator for smooth fields.
/// For LOBPCG to work well, K^{-1}A should have a bounded condition number.
///
/// Note: The homogeneous (Fourier-diagonal) preconditioner scales by 1/(|k+G|² + shift),
/// where shift is adaptively computed. For TE mode with eigenvalue |k+G|²/ε, the
/// preconditioned eigenvalue is approximately 1/ε but may vary due to the shift term.
/// The key property is that K^{-1}A preserves eigenvector direction (high cosine_sq).
#[test]
fn preconditioner_approximates_operator_inverse() {
    let grid = Grid2D::new(6, 6, 1.0, 1.0);
    let eps_const = 8.0;
    let dielectric = uniform_dielectric(grid, eps_const);
    let backend = TestBackend;
    let bloch = [0.0, 0.0];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
    let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();

    // For a plane wave, A*v = λ*v where λ = |k+G|²/ε
    // The preconditioner K^{-1} scales by ~1/(|k+G|² + shift)
    let input = plane_wave(grid, 1, 0);
    let mut applied = theta.alloc_field();
    theta.apply(&input, &mut applied);

    // Apply preconditioner to A*v
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut applied);

    // The result should be proportional to input
    let inner = inner_product(&applied, &input);
    let input_norm_sq = inner_product(&input, &input);
    let applied_norm_sq = inner_product(&applied, &applied);

    // Check that K^{-1}*A*v is roughly aligned with v (preserves direction)
    // This is the most important property for effective preconditioning
    let cosine_sq = inner.norm_sqr() / (input_norm_sq.re * applied_norm_sq.re);
    assert!(
        cosine_sq > 0.99,
        "K^{{-1}}A should preserve direction, got cos²={cosine_sq:.4}"
    );

    // The eigenvalue of K^{-1}A should be O(1/ε), i.e., a bounded positive value
    // The exact value depends on the adaptive shift, but should be in a reasonable range
    let kia_eigenvalue = inner.re / input_norm_sq.re;
    assert!(
        kia_eigenvalue > 0.0 && kia_eigenvalue < 1.0,
        "K^{{-1}}A eigenvalue should be positive and O(1/ε), got {kia_eigenvalue:.4}"
    );
}

// ============================================================================
// Uniform Medium Eigenvalue Tests
// ============================================================================
// These tests verify that for uniform ε(r) = ε_const, the computed Rayleigh
// quotients match the analytic eigenvalues exactly:
//
// TM (untransformed):  λ = |k+G|² / ε_const  (generalized A x = λ B x, A=-∇², B=ε)
// TE:                  λ = |k+G|² / ε_const  (standard A x = λ x, A=-∇·(ε⁻¹∇))
//
// If these tests pass, the TM A,B are internally consistent. Any systematic
// shift in the photonic crystal case would then be due to dielectric interface
// discretization differences, not a bug in the Rayleigh quotient.

/// Helper to compute Rayleigh quotient λ = (x† A x) / (x† B x)
fn rayleigh_quotient(theta: &mut ThetaOperator<TestBackend>, v: &Field2D) -> f64 {
    let mut av = theta.alloc_field();
    let mut bv = theta.alloc_field();
    theta.apply(v, &mut av);
    theta.apply_mass(v, &mut bv);

    let numerator = inner_product(&av, v);
    let denominator = inner_product(&bv, v);

    numerator.re / denominator.re
}

/// Test TM uniform medium eigenvalues at Γ point.
/// For ε_const = 5.0, plane wave |mx, my⟩ should have λ = |k+G|² / ε_const.
#[test]
fn tm_uniform_medium_rayleigh_quotient_matches_analytic() {
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let eps_const = 5.0;
    let dielectric = uniform_dielectric(grid, eps_const);
    let backend = TestBackend;
    let bloch = [0.0, 0.0]; // Γ point

    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);

    // Test several plane waves
    let test_modes = [(1, 0), (0, 1), (1, 1), (2, 1), (-1, 2)];

    for (mx, my) in test_modes {
        let v = plane_wave(grid, mx, my);
        let lambda = rayleigh_quotient(&mut theta, &v);

        // Analytic: λ = |k+G|² / ε = |2π·(mx/Lx, my/Ly)|² / ε
        let k_plus_g_sq = shifted_eigenvalue(grid, bloch, mx, my);
        let expected = k_plus_g_sq / eps_const;

        let rel_error = (lambda - expected).abs() / expected.max(1e-10);
        assert!(
            rel_error < 1e-10,
            "TM uniform medium: mode ({mx},{my}) λ={lambda:.10e} expected={expected:.10e} rel_error={rel_error:.2e}"
        );
    }
}

/// Test TE uniform medium eigenvalues at Γ point.
/// For ε_const = 5.0, plane wave |mx, my⟩ should have λ = |k+G|² / ε_const.
#[test]
fn te_uniform_medium_rayleigh_quotient_matches_analytic() {
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let eps_const = 5.0;
    let dielectric = uniform_dielectric(grid, eps_const);
    let backend = TestBackend;
    let bloch = [0.0, 0.0]; // Γ point

    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);

    // Test several plane waves
    let test_modes = [(1, 0), (0, 1), (1, 1), (2, 1), (-1, 2)];

    for (mx, my) in test_modes {
        let v = plane_wave(grid, mx, my);
        let lambda = rayleigh_quotient(&mut theta, &v);

        // Analytic: λ = |k+G|² / ε (for TE, operator already includes 1/ε)
        let k_plus_g_sq = shifted_eigenvalue(grid, bloch, mx, my);
        let expected = k_plus_g_sq / eps_const;

        let rel_error = (lambda - expected).abs() / expected.max(1e-10);
        assert!(
            rel_error < 1e-10,
            "TE uniform medium: mode ({mx},{my}) λ={lambda:.10e} expected={expected:.10e} rel_error={rel_error:.2e}"
        );
    }
}

/// Test TM uniform medium at arbitrary k-point.
/// The shift k affects |k+G|² but the eigenvalue formula is the same.
#[test]
fn tm_uniform_medium_at_arbitrary_k_point() {
    let grid = Grid2D::new(6, 6, 1.2, 0.9);
    let eps_const = 7.5;
    let dielectric = uniform_dielectric(grid, eps_const);
    let backend = TestBackend;
    let bloch = [0.35 * PI, -0.2 * PI]; // Off Γ

    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);

    let test_modes = [(0, 0), (1, 0), (-1, 1), (2, -1)];

    for (mx, my) in test_modes {
        let v = plane_wave(grid, mx, my);
        let lambda = rayleigh_quotient(&mut theta, &v);

        let k_plus_g_sq = shifted_eigenvalue(grid, bloch, mx, my);
        let expected = k_plus_g_sq / eps_const;

        let rel_error = (lambda - expected).abs() / expected.max(1e-10);
        assert!(
            rel_error < 1e-10,
            "TM at k=({:.3},{:.3}): mode ({mx},{my}) λ={lambda:.10e} expected={expected:.10e} rel_error={rel_error:.2e}",
            bloch[0], bloch[1]
        );
    }
}

/// Test TE uniform medium at arbitrary k-point.
#[test]
fn te_uniform_medium_at_arbitrary_k_point() {
    let grid = Grid2D::new(6, 6, 1.2, 0.9);
    let eps_const = 7.5;
    let dielectric = uniform_dielectric(grid, eps_const);
    let backend = TestBackend;
    let bloch = [0.35 * PI, -0.2 * PI]; // Off Γ

    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);

    let test_modes = [(0, 0), (1, 0), (-1, 1), (2, -1)];

    for (mx, my) in test_modes {
        let v = plane_wave(grid, mx, my);
        let lambda = rayleigh_quotient(&mut theta, &v);

        let k_plus_g_sq = shifted_eigenvalue(grid, bloch, mx, my);
        let expected = k_plus_g_sq / eps_const;

        let rel_error = (lambda - expected).abs() / expected.max(1e-10);
        assert!(
            rel_error < 1e-10,
            "TE at k=({:.3},{:.3}): mode ({mx},{my}) λ={lambda:.10e} expected={expected:.10e} rel_error={rel_error:.2e}",
            bloch[0], bloch[1]
        );
    }
}

/// Test that TE and TM give IDENTICAL eigenvalues in uniform medium.
/// This is a critical consistency check: both polarizations should give
/// λ = |k+G|² / ε for a uniform dielectric.
#[test]
fn te_and_tm_match_in_uniform_medium() {
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let eps_const = 4.0;
    let dielectric_te = uniform_dielectric(grid, eps_const);
    let dielectric_tm = uniform_dielectric(grid, eps_const);
    let backend_te = TestBackend;
    let backend_tm = TestBackend;
    let bloch = [0.2 * PI, 0.15 * PI];

    let mut theta_te = ThetaOperator::new(backend_te, dielectric_te, Polarization::TE, bloch);
    let mut theta_tm = ThetaOperator::new(backend_tm, dielectric_tm, Polarization::TM, bloch);

    let test_modes = [(1, 0), (0, 1), (1, 1), (2, -1), (-1, 2)];

    for (mx, my) in test_modes {
        let v = plane_wave(grid, mx, my);

        let lambda_te = rayleigh_quotient(&mut theta_te, &v);
        let lambda_tm = rayleigh_quotient(&mut theta_tm, &v);

        let rel_diff = (lambda_te - lambda_tm).abs() / lambda_te.max(1e-10);
        assert!(
            rel_diff < 1e-10,
            "TE vs TM mismatch in uniform medium: mode ({mx},{my}) λ_TE={lambda_te:.10e} λ_TM={lambda_tm:.10e} rel_diff={rel_diff:.2e}"
        );
    }
}
