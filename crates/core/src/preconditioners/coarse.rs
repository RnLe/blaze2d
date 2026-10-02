//! Additive two-level correction on the lowest plane waves.
//!
//! T₂ = T + s Z (Zᴴ K Z)⁻¹ Zᴴ with K = A + σ_c B (+ the smoother's mass floor), a
//! scale s matching the smoother's normalization, and Z the orthonormal plane waves
//! of the smallest |k+G| (completed to whole shells, kernel modes excluded). The
//! operator assembles the Galerkin matrix from Fourier coefficients of the material
//! (see `ThetaOperator::plane_wave_galerkin`), so no operator is applied. In
//! Fourier coordinates the correction adds K_c⁻¹ r̂ on the coarse indices, which
//! does not depend on the FFT normalization and fuses into the smoother's Fourier
//! path. Both terms are Hermitian positive semidefinite, and the correction
//! commutes with PT whenever the material is even under inversion.

use num_complex::{Complex, Complex64};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::Real;
use crate::preconditioners::OperatorPreconditioner;

pub struct PlaneWaveCorrection<'a, B: SpectralBackend> {
    smoother: Box<dyn OperatorPreconditioner<B> + 'a>,
    indices: Vec<usize>,
    /// Lower Cholesky factor of the Galerkin matrix, row-major m × m.
    factor: Vec<Complex64>,
    scale: f64,
    coarse: Vec<Complex64>,
}

impl<'a, B: SpectralBackend> PlaneWaveCorrection<'a, B> {
    /// `galerkin` is the row-major Hermitian matrix on `indices`, `scale` > 0 the
    /// coarse term's weight. Fails when the matrix is not numerically positive definite.
    pub fn new(smoother: Box<dyn OperatorPreconditioner<B> + 'a>, indices: Vec<usize>, galerkin: Vec<Complex64>, scale: f64)
        -> Result<Self, (Box<dyn OperatorPreconditioner<B> + 'a>, String)> {
        let m = indices.len();
        assert_eq!(galerkin.len(), m * m);
        assert!(scale > 0.0 && scale.is_finite());
        match cholesky(galerkin, m) {
            Ok(factor) => Ok(Self { smoother, indices, factor, scale, coarse: vec![Complex64::new(0.0, 0.0); m] }),
            Err(reason) => Err((smoother, reason)),
        }
    }

    pub fn modes(&self) -> usize { self.indices.len() }
}

/// L with A = L Lᴴ for a Hermitian positive definite row-major matrix.
fn cholesky(mut a: Vec<Complex64>, m: usize) -> Result<Vec<Complex64>, String> {
    let scale = (0..m).map(|i| a[i * m + i].re.abs()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    for j in 0..m {
        let mut d = a[j * m + j].re;
        for k in 0..j { d -= a[j * m + k].norm_sqr(); }
        if !(d > 1e-13 * scale) {
            return Err(format!("coarse Galerkin matrix is not positive definite (pivot {d:e} at {j})"));
        }
        let d = d.sqrt();
        a[j * m + j] = Complex64::new(d, 0.0);
        for i in j + 1..m {
            let mut s = a[i * m + j];
            for k in 0..j { s -= a[i * m + k] * a[j * m + k].conj(); }
            a[i * m + j] = s / d;
        }
        for k in j + 1..m { a[j * m + k] = Complex64::new(0.0, 0.0); }
    }
    Ok(a)
}

/// Solve L Lᴴ x = b in place.
fn cholesky_solve(factor: &[Complex64], m: usize, x: &mut [Complex64]) {
    for i in 0..m {
        let mut s = x[i];
        for k in 0..i { s -= factor[i * m + k] * x[k]; }
        x[i] = s / factor[i * m + i].re;
    }
    for i in (0..m).rev() {
        let mut s = x[i];
        for k in i + 1..m { s -= factor[k * m + i].conj() * x[k]; }
        x[i] = s / factor[i * m + i].re;
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for PlaneWaveCorrection<'_, B> {
    fn apply_fourier(&mut self, backend: &B, buffer: &mut B::Buffer) -> bool {
        for (c, &i) in self.coarse.iter_mut().zip(&self.indices) {
            let z = buffer.as_slice()[i];
            *c = Complex64::new(z.re.to_accum(), z.im.to_accum());
        }
        if !self.smoother.apply_fourier(backend, buffer) {
            return false;
        }
        cholesky_solve(&self.factor, self.indices.len(), &mut self.coarse);
        let data = buffer.as_mut_slice();
        for (c, &i) in self.coarse.iter().zip(&self.indices) {
            let c = c * self.scale;
            data[i] += Complex::new(<B::Real as Real>::from_accum(c.re), <B::Real as Real>::from_accum(c.im));
        }
        crate::profiler::count("preconditioner.coarse_columns", 1);
        true
    }

    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        backend.forward_fft_2d(buffer);
        let applied = self.apply_fourier(backend, buffer);
        assert!(applied, "the coarse correction needs a smoother with a Fourier path");
        backend.inverse_fft_2d(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_hermitian_systems() {
        let m = 3;
        let a = vec![
            Complex64::new(4.0, 0.0), Complex64::new(1.0, 1.0), Complex64::new(0.0, -0.5),
            Complex64::new(1.0, -1.0), Complex64::new(5.0, 0.0), Complex64::new(0.25, 0.0),
            Complex64::new(0.0, 0.5), Complex64::new(0.25, 0.0), Complex64::new(3.0, 0.0),
        ];
        let b = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0), Complex64::new(-1.0, 1.0)];
        let factor = cholesky(a.clone(), m).unwrap();
        let mut x = b;
        cholesky_solve(&factor, m, &mut x);
        for i in 0..m {
            let row: Complex64 = (0..m).map(|j| a[i * m + j] * x[j]).sum();
            assert!((row - b[i]).norm() < 1e-13);
        }
        assert!(cholesky(vec![Complex64::new(-1.0, 0.0)], 1).is_err());
    }
}
