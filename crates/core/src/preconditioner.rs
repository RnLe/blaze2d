//! Operator preconditioners shared by eigensolvers.
//!
//! # Polarization Convention
//!
//! **Important**: This crate uses the opposite TE/TM convention from MPB!
//! - **Our TM** (H_z scalar) = **MPB's TE**: Operator is Θ = -∇·(ε⁻¹∇)
//! - **Our TE** (E_z scalar) = **MPB's TM**: Operator is Θ = -∇² with mass B = ε
//!
//! # Preconditioner Hierarchy
//!
//! This module provides two preconditioner variants:
//!
//! 1. **[`FourierDiagonalPreconditioner`]**: Simple Fourier-space scaling by `1/|k+G|²`.
//!    - O(N log N) cost per application (2 FFTs)
//!    - Treats ε as uniform - limited effectiveness for high contrast
//!    - Includes kernel compensation: zeros DC mode at Γ-point
//!    - Default for TE mode
//!
//! 2. **[`TransverseProjectionPreconditioner`]**: MPB-style physics-informed preconditioner.
//!    - O(N log N) cost per application (6 FFTs for both TE and TM)
//!    - Accounts for spatial ε(r) variation via approximate inverse
//!    - Achieves dramatic condition number reduction (10-100× better than diagonal)
//!    - Default for TM mode
//!    - Based on Johnson & Joannopoulos, Optics Express 8, 173 (2001)
//!
//! # MPB Transverse-Projection Algorithm
//!
//! The algorithm approximates the inverse of the curl–(1/ε)–curl operator:
//!
//! 1. FFT residual r to k-space: r̂
//! 2. **Invert first curl**: X̂ = -i(k+G) r̂ / |k+G|² (2-component vector field)
//! 3. IFFT both components to real space: X(r)
//! 4. **Apply ε** (not ε⁻¹!): Y(r) = ε(r) · X(r)
//! 5. FFT both components back to k-space: Ŷ
//! 6. **Invert second curl**: ĥ = -i(k+G) · Ŷ / |k+G|² (scalar)
//! 7. IFFT to get final result h(r)
//!
//! # When to Use Each Preconditioner
//!
//! | Scenario | Recommended Preconditioner |
//! |----------|---------------------------|
//! | TE mode (any contrast) | FourierDiagonalKernelCompensated (default) |
//! | TM mode (any contrast) | TransverseProjection (default) |

use num_complex::Complex64;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::dielectric::Dielectric2D;
use crate::grid::Grid2D;
use crate::polarization::Polarization;

/// Fraction of s_min to use for adaptive shift: σ(k) = α * s_min(k).
/// At Γ-point, s_min is the squared magnitude of the first reciprocal shell.
/// α = 0.5 means the zero mode gets regularized with σ ≈ s_min/2.
pub(crate) const SHIFT_SMIN_FRACTION: f64 = 0.5;

/// Spectral statistics for a given k-point's |k+G|² values.
///
/// Used to compute k-dependent regularization shifts that scale with
/// the local spectral range rather than using a fixed global constant.
#[derive(Debug, Clone, Copy)]
pub struct SpectralStats {
    /// Minimum nonzero |k+G|² (excludes the DC mode at Γ)
    pub s_min: f64,
    /// Median |k+G|²
    pub s_median: f64,
    /// Maximum |k+G|²
    pub s_max: f64,
}

impl SpectralStats {
    /// Compute spectral statistics from |k+G|² values.
    ///
    /// Excludes exact zeros and near-zero floored values (DC mode at Γ-point)
    /// from s_min calculation. The threshold 1e-6 is chosen to be above the
    /// operator's K_PLUS_G_NEAR_ZERO_FLOOR (1e-9) so floored values are excluded.
    pub fn compute(k_plus_g_sq: &[f64]) -> Self {
        // Use threshold above K_PLUS_G_NEAR_ZERO_FLOOR to exclude floored values
        const NEAR_ZERO_THRESHOLD: f64 = 1e-6;

        let mut nonzero_values: Vec<f64> = k_plus_g_sq
            .iter()
            .copied()
            .filter(|&v| v > NEAR_ZERO_THRESHOLD && v.is_finite())
            .collect();

        let s_min = nonzero_values.iter().copied().fold(f64::INFINITY, f64::min);
        let s_max = nonzero_values.iter().copied().fold(0.0, f64::max);

        // Compute median
        let s_median = if nonzero_values.is_empty() {
            1.0 // Fallback for degenerate case
        } else {
            nonzero_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = nonzero_values.len() / 2;
            if nonzero_values.len() % 2 == 0 {
                (nonzero_values[mid - 1] + nonzero_values[mid]) / 2.0
            } else {
                nonzero_values[mid]
            }
        };

        Self {
            s_min: if s_min.is_finite() { s_min } else { 1.0 },
            s_median,
            s_max: if s_max > 0.0 { s_max } else { 1.0 },
        }
    }

    /// Compute k-dependent shift using s_min-based scaling (recommended).
    ///
    /// σ(k) = α * s_min(k), where s_min is the smallest nonzero |k+G|².
    /// At Γ-point, s_min is the squared magnitude of the first reciprocal shell.
    ///
    /// This ensures:
    /// - For the zero mode at Γ: denominator = σ(k) (finite, not tiny)
    /// - For the first shells: denominator ≈ (1+α) * s_min
    /// - For large |k+G|: denominator ≈ |k+G|² as usual
    pub fn adaptive_shift(&self) -> f64 {
        SHIFT_SMIN_FRACTION * self.s_min
    }
}

pub trait OperatorPreconditioner<B: SpectralBackend> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer);
}

/// Fourier-diagonal preconditioner with kernel compensation.
///
/// This preconditioner applies a diagonal scaling in Fourier space:
/// M⁻¹(q) = ε_eff / (|q|² + σ²)
///
/// # Kernel Compensation
///
/// At the Γ-point (k=0), the DC mode (|k+G|²=0) is in the null space of the
/// Laplacian-type operator. This preconditioner explicitly zeros that mode,
/// relying on deflation to handle the null space properly. Away from Γ,
/// the shift σ² provides natural regularization since |k|² > 0.
///
/// # Adaptive Shift
///
/// The shift σ² is computed adaptively based on spectral statistics:
/// σ(k) = α × s_min(k), where s_min is the smallest nonzero |k+G|².
/// This ensures the preconditioner scales properly at each k-point.
#[derive(Debug, Clone)]
pub struct FourierDiagonalPreconditioner {
    inverse_diagonal: Vec<f64>,
}

impl FourierDiagonalPreconditioner {
    pub(crate) fn new(inverse_diagonal: Vec<f64>) -> Self {
        Self { inverse_diagonal }
    }

    pub fn inverse_diagonal(&self) -> &[f64] {
        &self.inverse_diagonal
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for FourierDiagonalPreconditioner {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        // Apply F⁻¹ · D · F (Fourier-diagonal operation)
        backend.forward_fft_2d(buffer);
        for (value, scale) in buffer
            .as_mut_slice()
            .iter_mut()
            .zip(self.inverse_diagonal.iter())
        {
            *value *= *scale;
        }
        backend.inverse_fft_2d(buffer);
    }
}

// ============================================================================
// MPB-Style Transverse-Projection Preconditioner
// ============================================================================

/// MPB-style "transverse-projection" preconditioner.
///
/// This preconditioner implements the approximate inverse of the Maxwell operator
/// as described in Johnson & Joannopoulos, Optics Express 8, 173 (2001).
///
/// # Key Insight: Same Algorithm for Both Polarizations
///
/// MPB uses the **same** FFT-based transverse-projection preconditioner for both
/// TE and TM modes. The algorithm inverts the curl–(1/ε)–curl structure that
/// appears in both formulations (MPB internally converts TM to this form).
///
/// # Algorithm (6 FFT operations)
///
/// 1. FFT residual r → r̂
/// 2. **Invert first curl**: X̂ = -i(k+G) r̂ / |k+G|² (2-component vector)
/// 3. IFFT both components to real space
/// 4. **Multiply by ε(r)** (not ε⁻¹!) - this inverts the 1/ε in the operator
/// 5. FFT both components back to k-space
/// 6. **Invert second curl**: ĥ = -i(k+G)·Ŷ / |k+G|² (scalar)
/// 7. IFFT to get final result
///
/// # Hermiticity
///
/// The preconditioner is Hermitian (self-adjoint), which is essential for
/// conjugate-gradient convergence. This follows from the symmetric structure
/// of the inverse operator.
///
/// # Regularization
///
/// Near-zero |k+G|² modes (especially at Γ-point) are regularized using a shift
/// σ² to prevent division by zero. The shift is chosen adaptively based on the
/// spectral properties of the k-point.
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
                    // Zero out near-zero modes (handled by deflation)
                    0.0
                } else {
                    let denom = k_sq + shift;
                    if denom > 1e-15 {
                        1.0 / denom
                    } else {
                        0.0
                    }
                }
            })
            .collect();

        // Store ε(r) for TM mode
        let eps = dielectric.eps().to_vec();

        // Allocate scratch buffers
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

    /// Apply the full transverse-projection preconditioner for TM mode.
    ///
    /// This is the MPB-style preconditioner for the operator Θ = -∇·(ε⁻¹∇).
    /// It approximates Θ⁻¹ by inverting the gradient/divergence operators
    /// and applying ε (not ε⁻¹!) in real space.
    ///
    /// Algorithm (following Johnson & Joannopoulos 2001):
    /// 1. FFT the input residual r → r̂
    /// 2. Invert divergence: X̂ = -i(k+G) r̂ / |k+G|²  (2-component vector)
    /// 3. IFFT both components to real space: X(r)
    /// 4. Multiply by ε(r): Y(r) = ε(r) · X(r)
    /// 5. FFT both components back: Ŷ
    /// 6. Invert gradient: ĥ = -i(k+G) · Ŷ / |k+G|²  (scalar)
    /// 7. IFFT to get final result h(r)
    ///
    /// This is exactly 6 FFT operations (1 + 2 + 2 + 1).
    fn apply_transverse_projection(&mut self, backend: &B, buffer: &mut B::Buffer) {
        // Step 1: FFT the input residual
        backend.forward_fft_2d(buffer);

        // Step 2: Invert gradient - compute G = -i(k+G)/|k+G|² · r̂
        // G_x = -i k_x / |k|² · r̂ = (-i k_x · inv_k_sq) · r̂
        // G_y = -i k_y / |k|² · r̂ = (-i k_y · inv_k_sq) · r̂
        {
            let input_fourier = buffer.as_slice();
            let grad_x_data = self.grad_x.as_mut_slice();
            let grad_y_data = self.grad_y.as_mut_slice();

            for idx in 0..self.grid.len() {
                let r_hat = input_fourier[idx];
                let inv_k_sq = self.inverse_k_sq[idx];
                let kx = self.k_plus_g_x[idx];
                let ky = self.k_plus_g_y[idx];

                // -i * k_x / |k|² = Complex64::new(0, -k_x) * inv_k_sq
                // But since we want the gradient (not curl), use +i for consistency
                // Actually for gradient: ∇ = i(k+G), so ∇⁻¹ = -i(k+G)/|k+G|²
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
        // result = Σ_j i(k+G)_j · Ĝ_j / |k+G|²
        // Divergence: i k_x · Ĝ_x + i k_y · Ĝ_y
        // Then multiply by 1/|k+G|² again (but we combine into one step)
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

                // Divergence: i k_x · G_x + i k_y · G_y
                // Note: we need the NEGATIVE divergence to match the operator sign
                let div = Complex64::new(0.0, kx) * g_x + Complex64::new(0.0, ky) * g_y;

                // Apply second 1/|k+G|² and negate (since Θ = -∇·(ε⁻¹∇))
                // The overall sign: Θ⁻¹ should give positive eigenvalues
                output_fourier[idx] = -div * inv_k_sq;
            }
        }

        // Step 7: IFFT to get final result in real space
        backend.inverse_fft_2d(buffer);
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for TransverseProjectionPreconditioner<B> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        // MPB uses the SAME transverse-projection algorithm for both TE and TM.
        // The algorithm inverts the curl–(1/ε)–curl structure common to both.
        self.apply_transverse_projection(backend, buffer);
    }
}
