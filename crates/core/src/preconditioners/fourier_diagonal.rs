//! Fourier-diagonal preconditioner for Maxwell operators.
//!
//! This preconditioner applies a diagonal scaling in Fourier space:
//! M⁻¹(q) = ε_eff / (|q|² + σ²)
//!
//! # Kernel Compensation
//!
//! At the Γ-point (k=0), the DC mode (|k+G|²=0) is in the null space of the
//! Laplacian-type operator. This preconditioner explicitly zeros that mode,
//! relying on deflation to handle the null space properly. Away from Γ,
//! the shift σ² provides natural regularization since |k|² > 0.
//!
//! # Adaptive Shift
//!
//! The shift σ² is computed adaptively based on spectral statistics:
//! σ(k) = α × s_min(k), where s_min is the smallest nonzero |k+G|².
//! This ensures the preconditioner scales properly at each k-point.

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::preconditioners::OperatorPreconditioner;

/// Fraction of s_min to use for adaptive shift: σ(k) = α * s_min(k).
pub const SHIFT_SMIN_FRACTION: f64 = 0.5;

// ============================================================================
// Spectral Statistics
// ============================================================================

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
    /// from s_min calculation.
    pub fn compute(k_plus_g_sq: &[f64]) -> Self {
        const NEAR_ZERO_THRESHOLD: f64 = 1e-6;

        let mut nonzero_values: Vec<f64> = k_plus_g_sq
            .iter()
            .copied()
            .filter(|&v| v > NEAR_ZERO_THRESHOLD && v.is_finite())
            .collect();

        let s_min = nonzero_values.iter().copied().fold(f64::INFINITY, f64::min);
        let s_max = nonzero_values.iter().copied().fold(0.0, f64::max);

        let s_median = if nonzero_values.is_empty() {
            1.0
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

    /// Compute k-dependent shift using s_min-based scaling.
    ///
    /// σ(k) = α * s_min(k), where s_min is the smallest nonzero |k+G|².
    pub fn adaptive_shift(&self) -> f64 {
        SHIFT_SMIN_FRACTION * self.s_min
    }
}

// ============================================================================
// Fourier-Diagonal Preconditioner
// ============================================================================

/// Fourier-diagonal preconditioner with kernel compensation.
///
/// This preconditioner applies a diagonal scaling in Fourier space:
/// M⁻¹(q) = ε_eff / (|q|² + σ²)
#[derive(Debug, Clone)]
pub struct FourierDiagonalPreconditioner {
    inverse_diagonal: Vec<f64>,
}

impl FourierDiagonalPreconditioner {
    /// Create a new Fourier-diagonal preconditioner.
    ///
    /// # Arguments
    ///
    /// * `inverse_diagonal` - Precomputed 1/(|k+G|² + σ²) values
    pub fn new(inverse_diagonal: Vec<f64>) -> Self {
        Self { inverse_diagonal }
    }

    /// Get the inverse diagonal values.
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
