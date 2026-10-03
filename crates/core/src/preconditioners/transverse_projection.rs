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
//! 4. **Multiply by ε(r)** (not ε⁻¹!) - this inverts the 1/ε in the operator; with
//!    a tensor weight, by the inverse of the smoothed ε⁻¹ tensor instead
//! 5. FFT both components back to k-space
//! 6. **Invert second curl**: ĥ = i(k+G)·Ŷ / |k+G|² (scalar)
//! 7. IFFT to get final result
//!
//! # Hermiticity
//!
//! The result D Σ_c k_c (F ε F⁻¹) k_c D, with D = 1/(|k+G|²+σ), is Hermitian
//! positive semidefinite like the operator it approximates.

use num_complex::Complex;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::dielectric::Dielectric2D;
use crate::field::Real;
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
    /// Per-pixel weight E = (ε⁻¹ tensor)⁻¹, row-major [xx, xy, yx, yy]; replaces ε when set.
    tensor_weights: Option<Vec<[f64; 4]>>,
    /// Spectral profile factor W^½ per mode, folded into `inverse_k_sq`.
    profile_root: Option<Vec<f64>>,
    /// Regularization shift σ²
    #[allow(dead_code)]
    shift: f64,
    /// Scratch buffer for gradient x-component
    grad_x: B::Buffer,
    /// Scratch buffer for gradient y-component
    grad_y: B::Buffer,
    /// The spatial weight in the transposed node order of `inverse_fft_2d_transposed`,
    /// built on first use.
    transposed_weights: std::sync::OnceLock<TransposedWeights>,
}

/// Tensor or scalar spatial weight in transposed node order.
enum TransposedWeights {
    Tensors(Vec<[f64; 4]>),
    Scalars(Vec<f64>),
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
            tensor_weights: None,
            profile_root: None,
            shift,
            grad_x,
            grad_y,
            transposed_weights: std::sync::OnceLock::new(),
        }
    }

    /// Weight the gradient by the inverse of the smoothed ε⁻¹ tensors the operator
    /// uses instead of the scalar ε. Singular tensors keep the scalar weight.
    pub fn with_tensor_weight(mut self, inverse_eps: &[[f64; 4]]) -> Self {
        assert_eq!(inverse_eps.len(), self.eps.len());
        self.tensor_weights = Some(inverse_eps.iter().zip(&self.eps).map(|(t, &eps)| {
            let det = t[0] * t[3] - t[1] * t[2];
            if det.is_finite() && det.abs() > 1e-300 {
                [t[3] / det, -t[1] / det, -t[2] / det, t[0] / det]
            } else {
                [eps, 0.0, 0.0, eps]
            }
        }).collect());
        self.transposed_weights = std::sync::OnceLock::new();
        self
    }

    /// Fold a spectral profile into both 1/(|k+G|² + σ) factors, giving W^½ T W^½ for
    /// the per-mode factor `root` = W^½ (see `profile_root`).
    pub fn with_profile_root(mut self, root: &[f64]) -> Self {
        assert_eq!(root.len(), self.inverse_k_sq.len());
        self.inverse_k_sq.iter_mut().zip(root).for_each(|(d, r)| *d *= r);
        self.profile_root = Some(root.to_vec());
        self
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
        if let Some(root) = &self.profile_root {
            self.inverse_k_sq.iter_mut().zip(root).for_each(|(d, r)| *d *= r);
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
        self.apply_transverse_projection_fourier(backend,buffer);
        backend.inverse_fft_2d(buffer);
    }

    fn apply_transverse_projection_fourier(&mut self, backend: &B, buffer: &mut B::Buffer) {
        // Step 2: Invert gradient
        {
            let input_fourier = buffer.as_slice();
            let grad_x_data = self.grad_x.as_mut_slice();
            let grad_y_data = self.grad_y.as_mut_slice();

            for idx in 0..self.grid.len() {
                let r_hat = input_fourier[idx];
                let inv_k_sq = <B::Real as Real>::from_accum(self.inverse_k_sq[idx]);
                let kx = <B::Real as Real>::from_accum(self.k_plus_g_x[idx]);
                let ky = <B::Real as Real>::from_accum(self.k_plus_g_y[idx]);

                let factor_x = Complex::new(<B::Real as num_traits::Zero>::zero(), -kx) * inv_k_sq;
                let factor_y = Complex::new(<B::Real as num_traits::Zero>::zero(), -ky) * inv_k_sq;

                grad_x_data[idx] = r_hat * factor_x;
                grad_y_data[idx] = r_hat * factor_y;
            }
        }

        // Step 3: IFFT both gradient components to real space, left in transposed node
        // order (the weight is pointwise) with the normalization still to apply
        let scale = backend.inverse_fft_2d_transposed(&mut self.grad_x);
        backend.inverse_fft_2d_transposed(&mut self.grad_y);

        // Step 4: Multiply by ε(r), or by the tensor weight, in real space
        {
            let s = <B::Real as Real>::from_accum(scale);
            let (nx, ny) = (self.grid.nx, self.grid.ny);
            let weights = self.transposed_weights.get_or_init(|| match &self.tensor_weights {
                Some(tensors) => TransposedWeights::Tensors(crate::backend::transposed_layout(tensors, nx, ny)),
                None => TransposedWeights::Scalars(crate::backend::transposed_layout(&self.eps, nx, ny)),
            });
            let grad_x_data = self.grad_x.as_mut_slice();
            let grad_y_data = self.grad_y.as_mut_slice();
            match weights {
                TransposedWeights::Tensors(tensors) => {
                    for ((gx, gy), w) in grad_x_data.iter_mut().zip(grad_y_data.iter_mut()).zip(tensors) {
                        let (x, y) = (*gx * s, *gy * s);
                        let w = w.map(<B::Real as Real>::from_accum);
                        *gx = x * w[0] + y * w[1];
                        *gy = x * w[2] + y * w[3];
                    }
                }
                TransposedWeights::Scalars(eps) => {
                    for ((gx, gy), &eps_val) in grad_x_data.iter_mut().zip(grad_y_data.iter_mut()).zip(eps) {
                        let eps_val = <B::Real as Real>::from_accum(eps_val);
                        *gx = *gx * s * eps_val;
                        *gy = *gy * s * eps_val;
                    }
                }
            }
        }

        // Step 5: FFT both components back to k-space
        backend.forward_fft_2d_transposed(&mut self.grad_x);
        backend.forward_fft_2d_transposed(&mut self.grad_y);

        // Step 6: Assemble divergence and apply second inverse Laplacian
        {
            let output_fourier = buffer.as_mut_slice();
            let grad_x_fourier = self.grad_x.as_slice();
            let grad_y_fourier = self.grad_y.as_slice();

            for idx in 0..self.grid.len() {
                let g_x = grad_x_fourier[idx];
                let g_y = grad_y_fourier[idx];
                let inv_k_sq = <B::Real as Real>::from_accum(self.inverse_k_sq[idx]);
                let kx = <B::Real as Real>::from_accum(self.k_plus_g_x[idx]);
                let ky = <B::Real as Real>::from_accum(self.k_plus_g_y[idx]);

                let div = Complex::new(<B::Real as num_traits::Zero>::zero(), kx) * g_x
                    + Complex::new(<B::Real as num_traits::Zero>::zero(), ky) * g_y;
                output_fourier[idx] = div * inv_k_sq;
            }
        }

    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for TransverseProjectionPreconditioner<B> {
    fn apply_fourier(&mut self, backend: &B, buffer: &mut B::Buffer) -> bool {
        self.apply_transverse_projection_fourier(backend,buffer);
        true
    }
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        self.apply_transverse_projection(backend, buffer);
    }
}
