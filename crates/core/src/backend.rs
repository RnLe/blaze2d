//! Backend traits for spectral operations.
//!
//! # Precision strategy (runtime-selectable f32 / f64)
//!
//! Field storage precision is parameterised by the [`Real`] trait. Each
//! backend chooses its storage type via the [`SpectralBackend::Real`]
//! associated type; the buffer type is constrained to
//! `SpectralBuffer<Real = Self::Real>` so the link between backend and
//! storage is single-sourced. Downstream code (operators, eigensolver,
//! drivers) takes a single `B: SpectralBackend` parameter and reads
//! `B::Real` / `B::Buffer` internally — no separate `R` generic on every
//! signature.
//!
//! **The f32-storage / f64-accumulation invariant** is the single most
//! important rule for any backend implementing this trait at `R = f32`:
//!
//! - **Storage**: `Complex<R>` (saves bandwidth at `R = f32`).
//! - **FFTs**: performed in `R` (rustfft supports both).
//! - **Dot products / Gram matrices**: returned as `Complex<f64>` and
//!   MUST be accumulated in `Complex<f64>` even when storage is `f32`.
//!   Use the [`Real::to_accum`] / [`Real::from_accum`] boundary helpers
//!   in [`crate::field`]. Accumulating in `f32` causes catastrophic
//!   cancellation during orthogonalisation.
//! - **`scale` / `axpy`**: take `alpha: Complex<f64>` and downcast to
//!   storage precision internally. Callers do not need to know `B::Real`
//!   to construct the coefficient.
//! - **Rayleigh–Ritz / SVQB dense eigen-decomp**: always runs in
//!   `Complex<f64>` regardless of storage; the small N_band × N_band
//!   Gram matrix is already returned at accumulation precision.
//! - **Eigenvalues**: stored as `Vec<f64>` regardless of `R` (variational
//!   principle — error is O(ε_storage²) in the eigenvalues even when
//!   storage is single-precision).
//!
//! See [crates/core/src/field.rs](crate::field) and
//! [docs/state_report.md](../../docs/state_report.md) §1.3.

use num_complex::{Complex, Complex64};

use crate::field::{Field2D, Real};
use crate::grid::Grid2D;

/// Storage-precision-aware view of a complex 2D field buffer.
pub trait SpectralBuffer {
    /// The real scalar type backing this buffer (`f32` or `f64`).
    type Real: Real;

    fn len(&self) -> usize;
    fn grid(&self) -> Grid2D;
    fn as_slice(&self) -> &[Complex<Self::Real>];
    fn as_mut_slice(&mut self) -> &mut [Complex<Self::Real>];
}

impl<R: Real> SpectralBuffer for Field2D<R> {
    type Real = R;

    fn len(&self) -> usize {
        Field2D::<R>::len(self)
    }

    fn grid(&self) -> Grid2D {
        Field2D::<R>::grid(self)
    }

    fn as_slice(&self) -> &[Complex<R>] {
        Field2D::<R>::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [Complex<R>] {
        Field2D::<R>::as_mut_slice(self)
    }
}

pub trait SpectralBackend {
    /// The real scalar type used for field storage (`f32` or `f64`).
    type Real: Real;

    /// Buffer type whose storage scalar matches `Self::Real`.
    type Buffer: SpectralBuffer<Real = Self::Real> + Clone;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer;
    fn forward_fft_2d(&self, buffer: &mut Self::Buffer);
    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer);

    /// Scale buffer by a complex scalar. `alpha` is always taken at f64
    /// accumulation precision; backends downcast to storage precision
    /// internally if `Self::Real = f32`.
    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer);

    /// Compute y += alpha * x. `alpha` is always f64-precision.
    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer);

    /// Compute conjugate dot product ⟨x, y⟩ = x^H · y.
    ///
    /// **MUST accumulate in `Complex<f64>`** even when storage is `f32`
    /// (see module-level doc). Backends at `R = f64` get a fast straight
    /// loop; backends at `R = f32` should promote each summand via
    /// [`Real::to_accum`] before adding.
    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64;

    /// Compute the Gram matrix G_ij = ⟨x_i, y_j⟩ for batches of vectors.
    ///
    /// Result is row-major p×q where p = x.len(), q = y.len(). Always
    /// returned as `Complex<f64>` (accumulation precision); the eigensolver
    /// passes this directly to faer / nalgebra.
    fn gram_matrix(&self, x: &[Self::Buffer], y: &[Self::Buffer]) -> Vec<Complex64> {
        let p = x.len();
        let q = y.len();
        let mut result = vec![Complex64::ZERO; p * q];
        for i in 0..p {
            for j in 0..q {
                result[i * q + j] = self.dot(&x[i], &y[j]);
            }
        }
        result
    }

    /// Compute batched linear combinations: Y = Q × C
    ///
    /// Given a basis Q of r vectors (each of length n) and a coefficient
    /// matrix C of size r×m (column-major), compute m output vectors where
    /// `y_j = Σ_i q_i * C[i, j]`. Coefficients are at accumulation precision.
    fn linear_combinations(
        &self,
        q: &[Self::Buffer],
        coeffs: &[Complex64],
        m: usize,
    ) -> Vec<Self::Buffer> {
        let r = q.len();
        if r == 0 || m == 0 {
            return vec![];
        }

        assert_eq!(coeffs.len(), r * m, "coeffs must have r×m elements");

        let mut outputs: Vec<Self::Buffer> = Vec::with_capacity(m);

        for j in 0..m {
            let mut y = q[0].clone();
            self.scale(coeffs[j * r], &mut y);

            for i in 1..r {
                let coeff = coeffs[i + j * r];
                self.axpy(coeff, &q[i], &mut y);
            }

            outputs.push(y);
        }

        outputs
    }

    /// Batched forward FFT on multiple buffers.
    fn batch_forward_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        for buffer in buffers.iter_mut() {
            self.forward_fft_2d(buffer);
        }
    }

    /// Batched inverse FFT on multiple buffers.
    fn batch_inverse_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        for buffer in buffers.iter_mut() {
            self.inverse_fft_2d(buffer);
        }
    }
}
