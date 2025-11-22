#![cfg(test)]

use std::f64::consts::PI;

use num_complex::Complex64;

use super::backend::SpectralBackend;
use super::dielectric::Dielectric2D;
use super::field::Field2D;
use super::geometry::{BasisAtom, Geometry2D};
use super::grid::Grid2D;
use super::lattice::Lattice2D;
use super::operator::{LinearOperator, ThetaOperator, ToyLaplacian};
use super::polarization::Polarization;
use super::preconditioner::{FOURIER_DIAGONAL_SHIFT, OperatorPreconditioner};

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
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: eps,
        atoms: Vec::new(),
    };
    Dielectric2D::from_geometry(&geom, grid)
}

fn patterned_dielectric(grid: Grid2D) -> Dielectric2D {
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
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
    Dielectric2D::from_geometry(&geom, grid)
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
fn theta_te_uniform_medium_matches_laplacian() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 12.0);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, [0.0, 0.0]);
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
fn theta_tm_uniform_medium_matches_laplacian() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 8.0);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [0.0, 0.0]);
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
fn theta_tm_respects_bloch_shift_for_constant_field() {
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let dielectric = uniform_dielectric(grid, 5.0);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [PI, 0.0]);
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
fn theta_te_plane_wave_matches_shifted_eigenvalue_on_rectangular_grid() {
    let grid = Grid2D::new(6, 5, 1.6, 0.9);
    let dielectric = uniform_dielectric(grid, 7.0);
    let backend = TestBackend;
    let bloch = [0.3 * PI, -0.45 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
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
fn theta_tm_plane_wave_matches_shifted_eigenvalue_on_rectangular_grid() {
    let grid = Grid2D::new(5, 6, 1.4, 0.85);
    let eps_bg = 5.5;
    let dielectric = uniform_dielectric(grid, eps_bg);
    let backend = TestBackend;
    let bloch = [-0.2 * PI, 0.35 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
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
fn theta_tm_operator_is_hermitian_in_non_uniform_dielectric() {
    let grid = Grid2D::new(4, 5, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let backend = TestBackend;
    let bloch = [0.15 * PI, -0.28 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
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
fn theta_te_operator_is_hermitian_for_bloch_shift() {
    let grid = Grid2D::new(4, 4, 1.2, 0.95);
    let dielectric = uniform_dielectric(grid, 9.0);
    let backend = TestBackend;
    let bloch = [0.4 * PI, 0.1 * PI];
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
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
fn theta_te_mass_matches_dielectric_profile() {
    let grid = Grid2D::new(3, 5, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let eps = dielectric.eps().to_vec();
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TE, [0.0, 0.0]);
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
fn theta_tm_mass_is_identity() {
    let grid = Grid2D::new(3, 3, 1.0, 1.0);
    let dielectric = patterned_dielectric(grid);
    let backend = TestBackend;
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [0.0, 0.0]);
    let input = deterministic_field(grid, 21);
    let mut output = theta.alloc_field();
    theta.apply_mass(&input, &mut output);
    assert_fields_close(&output, &input, 1e-9);
}

#[test]
fn fourier_preconditioner_scales_te_plane_wave() {
    let grid = Grid2D::new(5, 4, 1.3, 0.9);
    let dielectric = uniform_dielectric(grid, 2.0);
    let backend = TestBackend;
    let bloch = [0.25 * PI, -0.1 * PI];
    let theta = ThetaOperator::new(backend, dielectric, Polarization::TE, bloch);
    let mut preconditioner = theta.build_fourier_diagonal_preconditioner();
    let mut field = plane_wave(grid, 1, -1);
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut field);
    let expected_scale = 1.0 / (shifted_eigenvalue(grid, bloch, 1, -1) + FOURIER_DIAGONAL_SHIFT);
    let mut expected = plane_wave(grid, 1, -1);
    for value in expected.as_mut_slice() {
        *value *= expected_scale;
    }
    assert_fields_close(&field, &expected, 1e-9);
}

#[test]
fn fourier_preconditioner_uses_tm_effective_epsilon() {
    let grid = Grid2D::new(4, 4, 1.1, 1.0);
    let dielectric = patterned_dielectric(grid);
    let dielectric_clone = dielectric.clone();
    let backend = TestBackend;
    let bloch = [0.0, 0.3 * PI];
    let theta = ThetaOperator::new(backend, dielectric, Polarization::TM, bloch);
    let mut preconditioner = theta.build_fourier_diagonal_preconditioner();
    let mut field = plane_wave(grid, 0, 1);
    let backend_ref = theta.backend();
    preconditioner.apply(backend_ref, &mut field);
    let inv_eps = dielectric_clone.inv_eps();
    let avg_inv = inv_eps.iter().copied().sum::<f64>() / inv_eps.len() as f64;
    let eps_eff = if avg_inv <= 0.0 { 1.0 } else { 1.0 / avg_inv };
    let expected_scale =
        1.0 / (shifted_eigenvalue(grid, bloch, 0, 1) / eps_eff + FOURIER_DIAGONAL_SHIFT);
    let mut expected = plane_wave(grid, 0, 1);
    for value in expected.as_mut_slice() {
        *value *= expected_scale;
    }
    assert_fields_close(&field, &expected, 1e-9);
}
