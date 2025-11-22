#![cfg(test)]

use std::f64::consts::PI;

use num_complex::Complex64;

use super::backend::SpectralBackend;
use super::dielectric::Dielectric2D;
use super::field::Field2D;
use super::geometry::Geometry2D;
use super::grid::Grid2D;
use super::lattice::Lattice2D;
use super::operator::{LinearOperator, ThetaOperator, ToyLaplacian};
use super::polarization::Polarization;

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
