#![cfg(test)]

use num_complex::Complex64;

use super::backend::{SpectralBackend, SpectralBuffer};
use super::field::{AccumScalar, Field2D, FieldReal, FieldScalar};
use super::grid::Grid2D;

#[derive(Clone)]
struct DummyBackend;

impl DummyBackend {
    fn phase(grid: Grid2D, ix: usize, iy: usize) -> FieldScalar {
        let argument =
            -2.0 * std::f64::consts::PI * (ix as f64 / grid.nx as f64 + iy as f64 / grid.ny as f64);
        let c = Complex64::from_polar(1.0, argument);
        FieldScalar::new(c.re as FieldReal, c.im as FieldReal)
    }
}

impl SpectralBackend for DummyBackend {
    type Buffer = Field2D;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        let grid = buffer.grid();
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let idx = buffer.idx(ix, iy);
                buffer.as_mut_slice()[idx] *= Self::phase(grid, ix, iy);
            }
        }
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        let grid = buffer.grid();
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let idx = buffer.idx(ix, iy);
                buffer.as_mut_slice()[idx] *= Self::phase(grid, ix, iy).conj();
            }
        }
    }

    fn scale(&self, alpha: AccumScalar, buffer: &mut Self::Buffer) {
        let alpha_field = FieldScalar::new(alpha.re as FieldReal, alpha.im as FieldReal);
        for value in buffer.as_mut_slice() {
            *value *= alpha_field;
        }
    }

    fn axpy(&self, alpha: AccumScalar, x: &Self::Buffer, y: &mut Self::Buffer) {
        let alpha_field = FieldScalar::new(alpha.re as FieldReal, alpha.im as FieldReal);
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *dst += alpha_field * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> AccumScalar {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| {
                let a64 = Complex64::new(a.re as f64, a.im as f64);
                let b64 = Complex64::new(b.re as f64, b.im as f64);
                a64.conj() * b64
            })
            .sum()
    }
}

#[test]
fn field2d_as_spectral_buffer_exposes_shared_storage() {
    let grid = Grid2D::new(3, 2, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new(idx as FieldReal, -(idx as FieldReal));
    }
    assert_eq!(SpectralBuffer::len(&field), grid.len());
    let view = SpectralBuffer::grid(&field);
    assert_eq!(view.nx, grid.nx);
    assert_eq!(view.ny, grid.ny);
    assert!((view.lx - grid.lx).abs() < f64::EPSILON);
    assert!((view.ly - grid.ly).abs() < f64::EPSILON);
    assert_eq!(SpectralBuffer::as_slice(&field), field.as_slice());

    SpectralBuffer::as_mut_slice(&mut field)[1] = FieldScalar::new(42.0, 1.0);
    assert_eq!(field.as_slice()[1], FieldScalar::new(42.0, 1.0));
}

#[test]
fn dummy_backend_allocates_zeroed_fields() {
    let backend = DummyBackend;
    let grid = Grid2D::new(2, 3, 1.0, 1.0);
    let field = backend.alloc_field(grid);
    let view = field.grid();
    assert_eq!(view.nx, grid.nx);
    assert_eq!(view.ny, grid.ny);
    assert!((view.lx - grid.lx).abs() < f64::EPSILON);
    assert!((view.ly - grid.ly).abs() < f64::EPSILON);
    assert!(
        field
            .as_slice()
            .iter()
            .all(|v| *v == FieldScalar::default())
    );
}

#[test]
fn dummy_backend_scale_axpy_and_dot_follow_linear_algebra_rules() {
    let backend = DummyBackend;
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let mut x = backend.alloc_field(grid);
    let mut y = backend.alloc_field(grid);
    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new(i as FieldReal + 1.0, -(i as FieldReal));
    }
    for (i, value) in y.as_mut_slice().iter_mut().enumerate() {
        *value = FieldScalar::new(0.5 * i as FieldReal, i as FieldReal);
    }
    let y_before_scale = y.as_slice().to_vec();
    backend.scale(Complex64::new(0.0, 1.0), &mut y);
    let scaled_manual: Vec<_> = y_before_scale
        .iter()
        .map(|v| FieldScalar::new(-v.im, v.re)) // multiplying by i
        .collect();
    for (actual, expect) in y.as_slice().iter().zip(scaled_manual.iter()) {
        assert!((*actual - *expect).norm() < 1e-5);
    }
    let after_scale = y.as_slice().to_vec();
    backend.axpy(Complex64::new(2.0, 0.0), &x, &mut y);
    for ((actual, x_val), y_scaled) in y
        .as_slice()
        .iter()
        .zip(x.as_slice())
        .zip(after_scale.iter())
    {
        let expect = *y_scaled + FieldScalar::new(2.0, 0.0) * x_val;
        assert!((*actual - expect).norm() < 1e-5);
    }
    let dot = backend.dot(&x, &y);
    let manual: Complex64 = x
        .as_slice()
        .iter()
        .zip(y.as_slice())
        .map(|(a, b)| {
            let a64 = Complex64::new(a.re as f64, a.im as f64);
            let b64 = Complex64::new(b.re as f64, b.im as f64);
            a64.conj() * b64
        })
        .sum();
    assert!((dot - manual).norm() < 1e-5);
}

#[test]
fn dummy_backend_forward_then_inverse_is_identity_up_to_roundoff() {
    let backend = DummyBackend;
    let grid = Grid2D::new(4, 2, 1.0, 1.0);
    let mut field = backend.alloc_field(grid);
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        let c = Complex64::from_polar(1.0 + idx as f64 * 0.1, idx as f64 * 0.2);
        *value = FieldScalar::new(c.re as FieldReal, c.im as FieldReal);
    }
    let before = field.as_slice().to_vec();
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);
    for (orig, after) in before.iter().zip(field.as_slice()) {
        assert!((*orig - *after).norm() < 1e-5);
    }
}
