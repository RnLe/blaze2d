//! FFT-based preconditioner for envelope approximation operators.
//!
//! This preconditioner approximates the inverse of the EA Hamiltonian using
//! a constant-coefficient Laplacian, which is diagonal in Fourier space.
//!
//! # Preconditioner Design
//!
//! The ideal preconditioner P approximates H^{-1} cheaply. For variable-coefficient
//! Laplacians, we use the constant-coefficient approximation:
//!
//! ```text
//! P^{-1} ≈ V̄ - (η² m̄ / 2) Δ
//! ```
//!
//! where V̄ and m̄ are spatial averages. This is diagonal in Fourier space:
//!
//! ```text
//! P̂^{-1}(kx, ky) = V̄ + (η² m̄ / 2) (kx² + ky²)
//! ```
//!
//! So:
//!
//! ```text
//! P̂(kx, ky) = 1 / (V̄ + (η² m̄ / 2) (kx² + ky²))
//! ```
//!
//! # Performance
//!
//! - FFT is O(N log N) vs O(N²) for dense preconditioner
//! - 2 FFT operations per application (forward + inverse)
//! - Preconditioner construction is one-time cost per solve

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::preconditioners::OperatorPreconditioner;

// ============================================================================
// FFT Preconditioner for EA Operator
// ============================================================================

/// FFT-based preconditioner for the envelope approximation Hamiltonian.
///
/// Approximates H^{-1} using a constant-coefficient Laplacian that is
/// diagonal in Fourier space.
pub struct FFTPreconditioner<B: SpectralBackend> {
    /// Grid dimensions (stored for potential future use)
    #[allow(dead_code)]
    nx: usize,
    #[allow(dead_code)]
    ny: usize,
    /// Precomputed: 1 / (V_mean + η²/2 * m_mean * |k|²)
    inv_spectrum: Vec<f64>,
    /// Marker for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: SpectralBackend> FFTPreconditioner<B> {
    /// Create a new FFT preconditioner for the EA operator.
    ///
    /// # Arguments
    ///
    /// * `nx`, `ny` - Grid dimensions
    /// * `dx`, `dy` - Grid spacings
    /// * `eta` - Twist parameter
    /// * `v_mean` - Mean potential
    /// * `m_mean` - Mean inverse mass (average of trace/2)
    pub fn new(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        eta: f64,
        v_mean: f64,
        m_mean: f64,
    ) -> Self {
        let prefactor = 0.5 * eta * eta * m_mean;

        let mut inv_spectrum = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let kx = fft_freq(i, nx, dx);
                let ky = fft_freq(j, ny, dy);
                let k_sq = kx * kx + ky * ky;
                let denom = v_mean + prefactor * k_sq;
                inv_spectrum[i * ny + j] = 1.0 / denom.max(1e-12);
            }
        }

        Self {
            nx,
            ny,
            inv_spectrum,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the precomputed inverse spectrum.
    pub fn inv_spectrum(&self) -> &[f64] {
        &self.inv_spectrum
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for FFTPreconditioner<B> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        // 1. Forward FFT: x → x̂
        backend.forward_fft_2d(buffer);

        // 2. Pointwise multiply: x̂ * P̂
        for (xh, &p) in buffer.as_mut_slice().iter_mut().zip(&self.inv_spectrum) {
            *xh *= p;
        }

        // 3. Inverse FFT: x̂ → y
        backend.inverse_fft_2d(buffer);
    }
}

/// Compute FFT frequency for index i in a grid of size n with spacing d.
fn fft_freq(i: usize, n: usize, d: f64) -> f64 {
    let freq = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
    2.0 * std::f64::consts::PI * freq / (n as f64 * d)
}
