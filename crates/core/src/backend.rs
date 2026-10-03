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

/// `values` (row length `nx`, `ny` rows) in transposed order: `out[ix * ny + iy] =
/// values[iy * nx + ix]`, the spatial layout of `inverse_fft_2d_transposed`.
pub fn transposed_layout<T: Copy>(values: &[T], nx: usize, ny: usize) -> Vec<T> {
    assert_eq!(values.len(), nx * ny);
    (0..nx * ny).map(|k| values[(k % ny) * nx + k / ny]).collect()
}

pub trait SpectralBackend {
    /// The real scalar type used for field storage (`f32` or `f64`).
    type Real: Real;

    /// Buffer type whose storage scalar matches `Self::Real`.
    type Buffer: SpectralBuffer<Real = Self::Real> + Clone;

    /// Coordinate backends can encode a real vector space in paired storage.
    /// Complex-GEMM shortcuts must use this backend's Gram implementation when
    /// this is true; an imaginary storage lane is then a second real coordinate.
    fn real_coefficients(&self)->bool {false}
    fn prefer_backend_linear_combinations(&self)->bool {false}
    fn gram_rows(&self,buffer:&Self::Buffer)->usize {buffer.len()}
    fn dense_eigen(&self,matrix:&[Complex64],dimension:usize)->crate::eigensolver::dense::DenseEigenResult {
        crate::eigensolver::dense::solve_hermitian_eigen(matrix,dimension)
    }

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer;
    fn forward_fft_2d(&self, buffer: &mut Self::Buffer);
    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer);

    /// Inverse 2D FFT that leaves the spatial field transposed (node (ix, iy) at
    /// `ix * ny + iy`) and returns the normalization it did not apply. Pointwise work on
    /// the transposed field uses transposed coefficients ([`transposed_layout`]) and applies
    /// the factor; [`forward_fft_2d_transposed`](Self::forward_fft_2d_transposed) returns
    /// to Fourier coefficients in the standard layout. A backend can skip the transposes
    /// that restore the spatial layout in between.
    fn inverse_fft_2d_transposed(&self, buffer: &mut Self::Buffer) -> f64 {
        self.inverse_fft_2d(buffer);
        let grid = buffer.grid();
        let transposed = transposed_layout(buffer.as_slice(), grid.nx, grid.ny);
        buffer.as_mut_slice().copy_from_slice(&transposed);
        1.0
    }

    /// Forward 2D FFT of a transposed spatial field (see `inverse_fft_2d_transposed`).
    fn forward_fft_2d_transposed(&self, buffer: &mut Self::Buffer) {
        let grid = buffer.grid();
        let standard = transposed_layout(buffer.as_slice(), grid.ny, grid.nx);
        buffer.as_mut_slice().copy_from_slice(&standard);
        self.forward_fft_2d(buffer);
    }

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

    /// Residual r = a − λ·b with the squared Euclidean norms [‖r‖², ‖a‖², ‖b‖²], so a
    /// backend can produce all of them from one pass over `a` and `b`.
    fn residual(&self, a: &Self::Buffer, b: &Self::Buffer, lambda: f64) -> (Self::Buffer, [f64; 3]) {
        let mut r = a.clone();
        self.axpy(Complex64::new(-lambda, 0.0), b, &mut r);
        let norms = [self.dot(&r, &r).re, self.dot(a, a).re, self.dot(b, b).re];
        (r, norms)
    }

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

    /// Gram matrices of `x` with each block of `ys` (each row-major), so a backend can
    /// read `x` once for all of them; a block equal to `x` itself may reuse its copy.
    fn gram_matrices(&self, x: &[Self::Buffer], ys: &[&[Self::Buffer]]) -> Vec<Vec<Complex64>> {
        ys.iter().map(|y| self.gram_matrix(x, y)).collect()
    }

    /// [`gram_matrices`](Self::gram_matrices) whose square blocks are Hermitian by
    /// construction (Xᴴ·B·X, Xᴴ·A·X for self-adjoint A, B), so a backend may compute one
    /// triangle and mirror it.
    fn hermitian_gram_matrices(&self, x: &[Self::Buffer], ys: &[&[Self::Buffer]]) -> Vec<Vec<Complex64>> {
        self.gram_matrices(x, ys)
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
