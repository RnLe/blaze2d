//! MPB-style transverse-projection preconditioner.
//!
//! This preconditioner implements the approximate inverse of the Maxwell operator
//! as described in Johnson & Joannopoulos, Optics Express 8, 173 (2001).
//!
//! # Key Insight: Same Algorithm for Both Polarizations
//!
//! MPB uses the **same** FFT-based transverse-projection preconditioner for both
//! TE and TM modes. The algorithm inverts the curl–(1/ε)–curl structure that
//! appears in both formulations.
//!
//! # Algorithm (6 FFT operations)
//!
//! 1. FFT residual r → r̂
//! 2. **Invert first curl**: X̂ = -i(k+G) r̂ / |k+G|² (2-component vector)
//! 3. IFFT both components to real space
//! 4. **Multiply by ε(r)** (not ε⁻¹!) - this inverts the 1/ε in the operator
//! 5. FFT both components back to k-space
//! 6. **Invert second curl**: ĥ = -i(k+G)·Ŷ / |k+G|² (scalar)
//! 7. IFFT to get final result
//!
//! # Hermiticity
//!
//! The preconditioner is Hermitian (self-adjoint), which is essential for
//! conjugate-gradient convergence.

use num_complex::Complex64;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::dielectric::Dielectric2D;
use crate::grid::Grid2D;
use crate::polarization::Polarization;
use crate::preconditioners::OperatorPreconditioner;

// ============================================================================
// Transverse-Projection Preconditioner
// ============================================================================

/// MPB-style transverse-projection preconditioner.
pub struct TransverseProjectionPreconditioner<B: SpectralBackend> {
    /// Grid dimensions
    grid: Grid2D,
    /// Polarization mode (stored for debugging/logging)
    #[allow(dead_code)]
    polarization: Polarization,
    /// (k+G)_x components for each Fourier mode
    k_plus_g_x: Vec<f64>,
    /// (k+G)_y components for each Fourier mode
    k_plus_g_y: Vec<f64>,
    /// |k+G|² values for each Fourier mode
    #[allow(dead_code)]
    k_plus_g_sq: Vec<f64>,
    /// Precomputed 1/(|k+G|² + σ²) for inversion with regularization
    inverse_k_sq: Vec<f64>,
    /// Mask for near-zero modes (true = should be zeroed)
    #[allow(dead_code)]
    near_zero_mask: Vec<bool>,
    /// Dielectric function ε(r)
    eps: Vec<f64>,
    /// Regularization shift σ²
    #[allow(dead_code)]
    shift: f64,
    /// Scratch buffer for gradient x-component
    grad_x: B::Buffer,
    /// Scratch buffer for gradient y-component
    grad_y: B::Buffer,
}

impl<B: SpectralBackend> TransverseProjectionPreconditioner<B> {
    /// Create a new TransverseProjectionPreconditioner.
    ///
    /// # Arguments
    ///
    /// * `backend` - The spectral backend for FFT operations
    /// * `dielectric` - The dielectric function ε(r)
    /// * `polarization` - TE or TM mode
    /// * `k_plus_g_x` - (k+G)_x components for each Fourier mode
    /// * `k_plus_g_y` - (k+G)_y components for each Fourier mode
    /// * `k_plus_g_sq` - |k+G|² values for each Fourier mode
    /// * `near_zero_mask` - Mask indicating which modes have |k+G|² ≈ 0
    /// * `shift` - Regularization shift σ²
    pub fn new(
        backend: &B,
        dielectric: &Dielectric2D,
        polarization: Polarization,
        k_plus_g_x: Vec<f64>,
        k_plus_g_y: Vec<f64>,
        k_plus_g_sq: Vec<f64>,
        near_zero_mask: Vec<bool>,
        shift: f64,
    ) -> Self {
        let grid = dielectric.grid;

        // Precompute 1/(|k+G|² + σ²)
        let inverse_k_sq: Vec<f64> = k_plus_g_sq
            .iter()
            .zip(near_zero_mask.iter())
            .map(|(&k_sq, &is_near_zero)| {
                if is_near_zero {
                    0.0
                } else {
                    let denom = k_sq + shift;
                    if denom > 1e-15 { 1.0 / denom } else { 0.0 }
                }
            })
            .collect();

        let eps = dielectric.eps().to_vec();
        let grad_x = backend.alloc_field(grid);
        let grad_y = backend.alloc_field(grid);

        Self {
            grid,
            polarization,
            k_plus_g_x,
            k_plus_g_y,
            k_plus_g_sq,
            inverse_k_sq,
            near_zero_mask,
            eps,
            shift,
            grad_x,
            grad_y,
        }
    }

    /// Update the regularization shift σ² in-place.
    ///
    /// This recomputes the `inverse_k_sq` diagonal without reallocating
    /// the gradient buffers. Useful for band-window-based shift refinement
    /// during LOBPCG iteration.
    ///
    /// # Arguments
    ///
    /// * `new_shift` - The new regularization shift σ²
    pub fn update_shift(&mut self, new_shift: f64) {
        self.shift = new_shift;

        // Recompute 1/(|k+G|² + σ²)
        for ((inv_k_sq, &k_sq), &is_near_zero) in self
            .inverse_k_sq
            .iter_mut()
            .zip(self.k_plus_g_sq.iter())
            .zip(self.near_zero_mask.iter())
        {
            if is_near_zero {
                *inv_k_sq = 0.0;
            } else {
                let denom = k_sq + new_shift;
                *inv_k_sq = if denom > 1e-15 { 1.0 / denom } else { 0.0 };
            }
        }
    }

    /// Get the current regularization shift σ².
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// Get a reference to the inverse k-squared values.
    pub fn inverse_k_sq(&self) -> &[f64] {
        &self.inverse_k_sq
    }

    /// Apply the full transverse-projection preconditioner.
    fn apply_transverse_projection(&mut self, backend: &B, buffer: &mut B::Buffer) {
        // Step 1: FFT the input residual
        backend.forward_fft_2d(buffer);

        // Step 2: Invert gradient
        {
            let input_fourier = buffer.as_slice();
            let grad_x_data = self.grad_x.as_mut_slice();
            let grad_y_data = self.grad_y.as_mut_slice();

            for idx in 0..self.grid.len() {
                let r_hat = input_fourier[idx];
                let inv_k_sq = self.inverse_k_sq[idx];
                let kx = self.k_plus_g_x[idx];
                let ky = self.k_plus_g_y[idx];

                let factor_x = Complex64::new(0.0, -kx) * inv_k_sq;
                let factor_y = Complex64::new(0.0, -ky) * inv_k_sq;

                grad_x_data[idx] = r_hat * factor_x;
                grad_y_data[idx] = r_hat * factor_y;
            }
        }

        // Step 3: IFFT both gradient components to real space
        backend.inverse_fft_2d(&mut self.grad_x);
        backend.inverse_fft_2d(&mut self.grad_y);

        // Step 4: Multiply by ε(r) in real space
        {
            let grad_x_data = self.grad_x.as_mut_slice();
            let grad_y_data = self.grad_y.as_mut_slice();
            for idx in 0..self.grid.len() {
                let eps_val = self.eps[idx];
                grad_x_data[idx] *= eps_val;
                grad_y_data[idx] *= eps_val;
            }
        }

        // Step 5: FFT both components back to k-space
        backend.forward_fft_2d(&mut self.grad_x);
        backend.forward_fft_2d(&mut self.grad_y);

        // Step 6: Assemble divergence and apply second inverse Laplacian
        {
            let output_fourier = buffer.as_mut_slice();
            let grad_x_fourier = self.grad_x.as_slice();
            let grad_y_fourier = self.grad_y.as_slice();

            for idx in 0..self.grid.len() {
                let g_x = grad_x_fourier[idx];
                let g_y = grad_y_fourier[idx];
                let inv_k_sq = self.inverse_k_sq[idx];
                let kx = self.k_plus_g_x[idx];
                let ky = self.k_plus_g_y[idx];

                let div = Complex64::new(0.0, kx) * g_x + Complex64::new(0.0, ky) * g_y;
                output_fourier[idx] = -div * inv_k_sq;
            }
        }

        // Step 7: IFFT to get final result
        backend.inverse_fft_2d(buffer);
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for TransverseProjectionPreconditioner<B> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        self.apply_transverse_projection(backend, buffer);
    }
}
