//! Backend traits for spectral operations.
//!
//! # Mixed Precision Strategy
//!
//! When the `mixed-precision` feature is enabled:
//! - **Storage**: Fields use `Complex<f32>` for 2x bandwidth efficiency
//! - **FFTs**: Performed in f32 (sufficient for iterative refinement)
//! - **Dot products**: MUST accumulate in f64 to prevent catastrophic cancellation
//! - **Rayleigh-Ritz**: Dense eigenproblem always in f64
//!
//! This follows the standard HPC "f32 storage, f64 accumulation" pattern.

use num_complex::Complex64;

use crate::field::{Field2D, FieldScalar};
use crate::grid::Grid2D;

pub trait SpectralBuffer {
    fn len(&self) -> usize;
    fn grid(&self) -> Grid2D;
    fn as_slice(&self) -> &[FieldScalar];
    fn as_mut_slice(&mut self) -> &mut [FieldScalar];
}

impl SpectralBuffer for Field2D {
    fn len(&self) -> usize {
        self.len()
    }

    fn grid(&self) -> Grid2D {
        self.grid()
    }

    fn as_slice(&self) -> &[FieldScalar] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [FieldScalar] {
        self.as_mut_slice()
    }
}

pub trait SpectralBackend {
    type Buffer: SpectralBuffer + Clone;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer;
    fn forward_fft_2d(&self, buffer: &mut Self::Buffer);
    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer);
    
    /// Scale buffer by a complex scalar.
    /// Note: In mixed-precision mode, alpha is converted to storage precision.
    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer);
    
    /// Compute y += alpha * x (axpy operation).
    /// Note: In mixed-precision mode, alpha is converted to storage precision.
    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer);
    
    /// Compute conjugate dot product ⟨x, y⟩ = x^H · y.
    /// 
    /// **CRITICAL for mixed precision**: This MUST accumulate in f64 even when
    /// storage is f32. Accumulating in f32 causes catastrophic cancellation
    /// during orthogonalization, leading to loss of basis independence.
    /// 
    /// Implementation pattern for f32 storage:
    /// ```ignore
    /// let dot: f64 = x.iter().zip(y.iter())
    ///     .map(|(a, b)| {
    ///         let a64 = Complex64::new(a.re as f64, a.im as f64);
    ///         let b64 = Complex64::new(b.re as f64, b.im as f64);
    ///         a64.conj() * b64
    ///     })
    ///     .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x);
    /// ```
    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64;

    /// Compute the Gram matrix G_ij = ⟨x_i, y_j⟩ for batches of vectors.
    ///
    /// This computes the conjugate dot product between all pairs of vectors,
    /// returning a row-major p×q matrix where p = x.len() and q = y.len().
    ///
    /// # Arguments
    /// * `x` - First set of vectors (will be conjugated)
    /// * `y` - Second set of vectors
    ///
    /// # Returns
    /// A vector of length p×q containing G in row-major order:
    /// `result[i * q + j] = ⟨x[i], y[j]⟩ = x[i]^H · y[j]`
    ///
    /// # Performance
    /// GPU backends can implement this as a single ZGEMM call: G = X^H × Y
    /// where X and Y are (n × p) and (n × q) matrices respectively.
    fn gram_matrix(&self, x: &[Self::Buffer], y: &[Self::Buffer]) -> Vec<Complex64> {
        // Default implementation using individual dot products
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
    /// Given a basis Q of r vectors (each of length n) and a coefficient matrix C
    /// of size r×m (column-major), compute m output vectors where:
    ///   y_j = Σ_i q_i * C[i,j]
    ///
    /// # Arguments
    /// * `q` - Basis vectors, slice of r buffers each of length n
    /// * `coeffs` - Coefficient matrix C in column-major order, length r×m
    /// * `m` - Number of output vectors (columns of C)
    ///
    /// # Returns
    /// Vector of m output buffers, where output[j] = Σ_i q[i] * coeffs[i + j*r]
    ///
    /// # Performance
    /// GPU backends can implement this as a single ZGEMM call: Y = Q × C
    /// where Q is treated as (n × r) and C is (r × m), yielding Y as (n × m).
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
            // y_j = Σ_i q_i * C[i,j] where C is column-major: C[i,j] = coeffs[i + j*r]
            let mut y = q[0].clone();
            self.scale(coeffs[j * r], &mut y); // First term: q_0 * C[0,j]

            for i in 1..r {
                let coeff = coeffs[i + j * r]; // C[i,j] in column-major
                self.axpy(coeff, &q[i], &mut y);
            }

            outputs.push(y);
        }

        outputs
    }

    /// Batched forward FFT on multiple buffers.
    ///
    /// This applies forward 2D FFT to all buffers in the slice.
    /// Backends can override this for better cache locality by processing
    /// all rows first across all buffers, then all columns.
    ///
    /// # Performance
    /// CPU backends can gain significant speedup by:
    /// 1. Processing all rows of all buffers together (better cache use of FFT plans)
    /// 2. Then processing all columns of all buffers together
    /// 3. Amortizing the plan lookup cost across all buffers
    fn batch_forward_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        // Default: process each buffer individually
        for buffer in buffers.iter_mut() {
            self.forward_fft_2d(buffer);
        }
    }

    /// Batched inverse FFT on multiple buffers.
    ///
    /// This applies inverse 2D FFT to all buffers in the slice.
    /// See `batch_forward_fft_2d` for performance notes.
    fn batch_inverse_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        // Default: process each buffer individually
        for buffer in buffers.iter_mut() {
            self.inverse_fft_2d(buffer);
        }
    }
}
