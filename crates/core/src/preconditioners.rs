//! Preconditioners for iterative eigensolvers.
//!
//! This module provides preconditioners that accelerate convergence of the
//! LOBPCG eigensolver by approximating the inverse of the operator.
//!
//! # Available Preconditioners
//!
//! ## Fourier-Space Preconditioners
//!
//! - [`FourierDiagonalPreconditioner`]: Simple Fourier-space scaling by `1/|k+G|²`.
//!   O(N log N) cost per application (2 FFTs). Default for TM mode.
//!
//! - [`TransverseProjectionPreconditioner`]: MPB-style physics-informed preconditioner.
//!   O(N log N) cost per application (6 FFTs). Default for TE mode.
//!   Based on Johnson & Joannopoulos, Optics Express 8, 173 (2001).
//!
//! ## Future Preconditioners
//!
//! - `FFTPreconditioner`: For envelope approximation operator (EA)
//!
//! # Preconditioner Trait
//!
//! All preconditioners implement [`OperatorPreconditioner<B>`], which requires:
//!
//! - `apply(&mut self, backend, buffer)`: Apply M^{-1} to the buffer in-place
//!
//! # Adaptive Shift Tuning
//!
//! For difficult k-points (especially near Γ), the preconditioner shift σ² can
//! be tuned based on eigenvalue estimates from LOBPCG iterations. This is
//! provided through:
//!
//! - [`BandWindow`]: Eigenvalue range from current iteration
//! - [`SpectralStats::adaptive_shift_blended`]: Combines geometric and band-window shifts
//! - [`ShiftCalibrationConfig`]: Configuration for RQ-based shift tuning
//!
//! # Example
//!
//! ```ignore
//! use mpb2d_core::preconditioners::{OperatorPreconditioner, FourierDiagonalPreconditioner};
//!
//! // Build preconditioner
//! let mut precond = operator.build_homogeneous_preconditioner_adaptive();
//!
//! // Apply to residual
//! precond.apply(&backend, &mut residual);
//!
//! // After some iterations, refine with band-window shift:
//! let eigenvalues = solver.eigenvalues();
//! let mut refined_precond = operator.build_homogeneous_preconditioner_band_window(
//!     eigenvalues,
//!     Some(0.3),  // Favor band window
//!     None,       // Default scale
//! );
//! ```

use crate::backend::SpectralBackend;

pub mod fft_preconditioner;
pub mod fourier_diagonal;
pub mod transverse_projection;

// Re-export commonly used types
pub use fft_preconditioner::{
    EAPreconditionerConfig,
    FFTPreconditioner,
    ShiftDiagnostics,
    ShiftStrategy,
};
pub use fourier_diagonal::{
    BandWindow,
    DEFAULT_BAND_WINDOW_BLEND,
    DEFAULT_BAND_WINDOW_SCALE,
    FourierDiagonalPreconditioner,
    // Shift calibration types
    PreconditionedRQStats,
    SHIFT_SMIN_FRACTION,
    ShiftCalibrationConfig,
    ShiftCalibrationResult,
    SpectralStats,
    build_inverse_diagonal_standalone,
    compute_shift_quality_score,
};
pub use transverse_projection::TransverseProjectionPreconditioner;

// ============================================================================
// Core Preconditioner Trait
// ============================================================================

/// A preconditioner for iterative eigensolvers.
///
/// Preconditioners approximate the inverse of the operator to accelerate
/// convergence. They transform the residual r → M^{-1} r where M ≈ A.
///
/// # Requirements
///
/// For optimal LOBPCG convergence, preconditioners should:
/// - Be symmetric positive definite (SPD)
/// - Approximate the inverse of the operator
/// - Be cheap to apply (ideally O(N log N) or O(N))
///
/// # Type Parameters
///
/// - `B`: The spectral backend type
pub trait OperatorPreconditioner<B: SpectralBackend> {
    /// Apply the preconditioner to a buffer in-place: buffer ← M^{-1} buffer
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer);
}

// ============================================================================
// Adaptive Shift Configuration
// ============================================================================

/// Configuration for adaptive shift tuning in preconditioners.
///
/// This allows fine-tuning the preconditioner shift based on:
/// - Current eigenvalue estimates (band window)
/// - Rayleigh quotient statistics
///
/// # Example
///
/// ```ignore
/// let config = AdaptiveShiftConfig::new()
///     .with_blend(0.3)      // Favor band window over s_min
///     .with_band_scale(0.5) // Scale factor for λ_median
///     .with_auto_calibrate(true);  // Enable RQ-based calibration
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveShiftConfig {
    /// Blending factor β ∈ [0, 1] for combining s_min and band-window shifts.
    /// β = 1.0 uses only s_min (default), β = 0.0 uses only band window.
    pub blend: f64,
    /// Scaling factor c for band-window shift: σ²_band = c × λ_median.
    pub band_scale: f64,
    /// Whether to automatically calibrate shift using RQ statistics.
    /// If true, performs a calibration phase at difficult k-points.
    pub auto_calibrate: bool,
    /// Configuration for RQ-based calibration (when auto_calibrate is true).
    pub calibration_config: ShiftCalibrationConfig,
    /// Iteration threshold: only apply band-window shift after this many iterations.
    /// This allows eigenvalue estimates to stabilize before using them.
    pub min_iterations: usize,
    /// Whether this k-point is considered "difficult" (e.g., near Γ).
    /// Difficult k-points may trigger additional calibration.
    pub is_difficult_k: bool,
}

impl Default for AdaptiveShiftConfig {
    fn default() -> Self {
        Self {
            blend: DEFAULT_BAND_WINDOW_BLEND,
            band_scale: DEFAULT_BAND_WINDOW_SCALE,
            auto_calibrate: false,
            calibration_config: ShiftCalibrationConfig::default(),
            min_iterations: 2,
            is_difficult_k: false,
        }
    }
}

impl AdaptiveShiftConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the blending factor β.
    pub fn with_blend(mut self, blend: f64) -> Self {
        self.blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Set the band-window scaling factor c.
    pub fn with_band_scale(mut self, scale: f64) -> Self {
        self.band_scale = scale.max(0.01);
        self
    }

    /// Enable or disable automatic RQ-based calibration.
    pub fn with_auto_calibrate(mut self, enabled: bool) -> Self {
        self.auto_calibrate = enabled;
        self
    }

    /// Set the calibration configuration.
    pub fn with_calibration_config(mut self, config: ShiftCalibrationConfig) -> Self {
        self.calibration_config = config;
        self
    }

    /// Set the minimum iterations before using band-window shift.
    pub fn with_min_iterations(mut self, iters: usize) -> Self {
        self.min_iterations = iters;
        self
    }

    /// Mark this k-point as difficult.
    pub fn mark_as_difficult(mut self) -> Self {
        self.is_difficult_k = true;
        self
    }

    /// Create a configuration optimized for Γ-point and near-Γ k-points.
    ///
    /// This uses settings that favor band-window information and enables
    /// automatic calibration for the low-frequency cluster.
    pub fn for_gamma() -> Self {
        Self {
            blend: 0.3, // Favor band window
            band_scale: 0.5,
            auto_calibrate: true, // Enable calibration for difficult point
            calibration_config: ShiftCalibrationConfig::default(),
            min_iterations: 1, // Start using band info early
            is_difficult_k: true,
        }
    }

    /// Check if the configuration should use band-window information.
    pub fn should_use_band_window(&self, current_iteration: usize) -> bool {
        current_iteration >= self.min_iterations && self.blend < 1.0
    }
}
