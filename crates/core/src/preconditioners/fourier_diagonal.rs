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
//!
//! # Band-Window-Based Shift (Advanced)
//!
//! For improved convergence near Γ, the shift can be computed using eigenvalue
//! estimates from the current LOBPCG iteration. This targets the actual band
//! window [λ_min, λ_max] rather than the geometric spectral range.
//!
//! The blended shift is: σ²(k) = β·σ²_smin + (1-β)·σ²_band
//! where σ²_band ≈ c·λ_med for some constant c ∈ [0.1, 1.0].

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::preconditioners::OperatorPreconditioner;

/// Fraction of s_min to use for adaptive shift: σ(k) = α * s_min(k).
pub const SHIFT_SMIN_FRACTION: f64 = 0.5;

/// Default blending factor β for combining s_min-based and band-window-based shifts.
/// β = 1.0 uses only s_min (original behavior), β = 0.0 uses only band window.
pub const DEFAULT_BAND_WINDOW_BLEND: f64 = 0.5;

/// Default scaling factor c for band-window shift: σ²_band = c × λ_med.
/// Values in [0.1, 1.0] are reasonable; smaller values give less regularization.
pub const DEFAULT_BAND_WINDOW_SCALE: f64 = 0.5;

// ============================================================================
// Spectral Statistics
// ============================================================================

/// Spectral statistics for a given k-point's |k+G|² values.
///
/// Used to compute k-dependent regularization shifts that scale with
/// the local spectral range rather than using a fixed global constant.
///
/// # Band Window Extension
///
/// The `band_window` field allows incorporating eigenvalue estimates from
/// LOBPCG iterations to compute a shift targeting the actual band region
/// rather than the full geometric spectrum.
#[derive(Debug, Clone)]
pub struct SpectralStats {
    /// Minimum nonzero |k+G|² (excludes the DC mode at Γ)
    pub s_min: f64,
    /// Median |k+G|²
    pub s_median: f64,
    /// Maximum |k+G|²
    pub s_max: f64,
    /// Optional band window from current eigenvalue estimates.
    /// Contains (λ_min, λ_max, λ_median) for the bands being computed.
    pub band_window: Option<BandWindow>,
}

/// Band window information from eigenvalue estimates.
///
/// This contains the eigenvalue range from the current LOBPCG iteration,
/// allowing the preconditioner shift to target the actual spectral region
/// of interest rather than the full |k+G|² range.
#[derive(Debug, Clone, Copy)]
pub struct BandWindow {
    /// Minimum eigenvalue among tracked bands.
    pub lambda_min: f64,
    /// Maximum eigenvalue among tracked bands.
    pub lambda_max: f64,
    /// Median eigenvalue among tracked bands.
    pub lambda_median: f64,
}

impl BandWindow {
    /// Create a new band window from eigenvalue estimates.
    ///
    /// Returns `None` if the slice is empty or contains only non-positive values.
    pub fn from_eigenvalues(eigenvalues: &[f64]) -> Option<Self> {
        if eigenvalues.is_empty() {
            return None;
        }

        // Filter to positive eigenvalues only (skip any spurious zeros)
        let mut positive: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&ev| ev > 1e-15)
            .collect();

        if positive.is_empty() {
            return None;
        }

        positive.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lambda_min = positive.first().copied().unwrap_or(1e-10);
        let lambda_max = positive.last().copied().unwrap_or(1.0);

        let lambda_median = if positive.len() % 2 == 0 {
            let mid = positive.len() / 2;
            (positive[mid - 1] + positive[mid]) / 2.0
        } else {
            positive[positive.len() / 2]
        };

        Some(Self {
            lambda_min,
            lambda_max,
            lambda_median,
        })
    }

    /// Compute the band-window-based shift σ².
    ///
    /// Uses the median eigenvalue scaled by a factor c ∈ [0.1, 1.0].
    /// This targets the center of the band window for optimal preconditioning.
    pub fn compute_shift(&self, scale: f64) -> f64 {
        scale * self.lambda_median
    }
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
            band_window: None,
        }
    }

    /// Update the band window from eigenvalue estimates.
    ///
    /// Call this after one or more LOBPCG iterations to incorporate
    /// eigenvalue information into the adaptive shift computation.
    pub fn with_band_window(mut self, eigenvalues: &[f64]) -> Self {
        self.band_window = BandWindow::from_eigenvalues(eigenvalues);
        self
    }

    /// Set the band window directly.
    pub fn set_band_window(&mut self, eigenvalues: &[f64]) {
        self.band_window = BandWindow::from_eigenvalues(eigenvalues);
    }

    /// Compute k-dependent shift using s_min-based scaling (original method).
    ///
    /// σ(k) = α * s_min(k), where s_min is the smallest nonzero |k+G|².
    pub fn adaptive_shift(&self) -> f64 {
        SHIFT_SMIN_FRACTION * self.s_min
    }

    /// Compute k-dependent shift using band-window-aware blending.
    ///
    /// This combines the geometric s_min-based shift with a band-window-based
    /// shift derived from current eigenvalue estimates:
    ///
    /// σ²(k) = β·σ²_smin + (1-β)·σ²_band
    ///
    /// where:
    /// - σ²_smin = α × s_min (geometric, always available)
    /// - σ²_band = c × λ_median (from eigenvalue estimates, when available)
    ///
    /// # Arguments
    ///
    /// - `blend`: Blending factor β ∈ [0, 1]. β=1 uses only s_min, β=0 uses only band window.
    /// - `band_scale`: Scaling factor c for band-window shift. Typically 0.1-1.0.
    ///
    /// Falls back to pure s_min-based shift if no band window is available.
    pub fn adaptive_shift_blended(&self, blend: f64, band_scale: f64) -> f64 {
        let shift_smin = SHIFT_SMIN_FRACTION * self.s_min;

        match &self.band_window {
            Some(window) => {
                let shift_band = window.compute_shift(band_scale);
                // Blend: β·σ²_smin + (1-β)·σ²_band
                let beta = blend.clamp(0.0, 1.0);
                beta * shift_smin + (1.0 - beta) * shift_band
            }
            None => shift_smin, // Fallback to original method
        }
    }

    /// Compute the recommended shift based on available information.
    ///
    /// - If band window is available, uses blended shift with default parameters.
    /// - Otherwise, falls back to pure s_min-based shift.
    ///
    /// This is the recommended method for general use.
    pub fn adaptive_shift_auto(&self) -> f64 {
        if self.band_window.is_some() {
            self.adaptive_shift_blended(DEFAULT_BAND_WINDOW_BLEND, DEFAULT_BAND_WINDOW_SCALE)
        } else {
            self.adaptive_shift()
        }
    }
}

// ============================================================================
// Fourier-Diagonal Preconditioner
// ============================================================================

/// Fourier-diagonal preconditioner with kernel compensation.
///
/// This preconditioner applies a diagonal scaling in Fourier space:
/// M⁻¹(q) = ε_eff / (|q|² + σ²)
///
/// # In-Place Shift Updates
///
/// For band-window-based shift refinement, the preconditioner supports
/// in-place updates via [`update_diagonal`](Self::update_diagonal). This
/// avoids reallocating the diagonal vector when only the shift changes.
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

    /// Get mutable access to the inverse diagonal values.
    ///
    /// This is useful for in-place updates when the shift changes.
    pub fn inverse_diagonal_mut(&mut self) -> &mut [f64] {
        &mut self.inverse_diagonal
    }

    /// Update the diagonal with new values.
    ///
    /// This is more efficient than creating a new preconditioner when only
    /// the shift changes, as it reuses the existing allocation.
    ///
    /// # Panics
    ///
    /// Panics if `new_diagonal.len() != self.inverse_diagonal.len()`.
    pub fn update_diagonal(&mut self, new_diagonal: Vec<f64>) {
        assert_eq!(
            new_diagonal.len(),
            self.inverse_diagonal.len(),
            "New diagonal must have same length as existing"
        );
        self.inverse_diagonal = new_diagonal;
    }

    /// Update the diagonal in-place from a slice.
    ///
    /// # Panics
    ///
    /// Panics if `new_values.len() != self.inverse_diagonal.len()`.
    pub fn update_diagonal_from_slice(&mut self, new_values: &[f64]) {
        assert_eq!(
            new_values.len(),
            self.inverse_diagonal.len(),
            "New values must have same length as existing diagonal"
        );
        self.inverse_diagonal.copy_from_slice(new_values);
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
// Helper Functions for Building Inverse Diagonals
// ============================================================================

/// Build the inverse diagonal for a given shift and epsilon.
///
/// This is a standalone helper function that can be used to recompute the
/// diagonal when the shift changes during iteration.
///
/// # Arguments
///
/// * `k_plus_g_sq` - The |k+G|² values for all Fourier modes
/// * `shift` - The regularization shift σ²
/// * `eps_eff` - The effective dielectric constant
/// * `mass_floor` - Additional mass term for TM mode (0 for TE)
/// * `near_zero_mask` - Optional mask for near-zero modes to be zeroed (Γ-point DC)
///
/// # Returns
///
/// Vector of inverse diagonal values: ε_eff / (|k+G|² + mass_floor + shift·ε_eff)
pub fn build_inverse_diagonal_standalone(
    k_plus_g_sq: &[f64],
    shift: f64,
    eps_eff: f64,
    mass_floor: f64,
    near_zero_mask: Option<&[bool]>,
) -> Vec<f64> {
    let safe_eps = eps_eff.max(1e-12);
    let shift_scaled = shift * safe_eps;
    let safe_mass = if mass_floor.is_finite() && mass_floor > 0.0 {
        mass_floor
    } else {
        0.0
    };

    let mut result: Vec<f64> = k_plus_g_sq
        .iter()
        .map(|&k_sq| {
            if !k_sq.is_finite() || !eps_eff.is_finite() || eps_eff <= 0.0 {
                return 0.0;
            }
            let safe_k_sq = k_sq.max(0.0);
            let denominator = safe_k_sq + safe_mass + shift_scaled;
            safe_eps / denominator
        })
        .collect();

    // Zero out near-zero modes (DC at Γ-point)
    if let Some(mask) = near_zero_mask {
        for (scale, &is_near_zero) in result.iter_mut().zip(mask.iter()) {
            if is_near_zero {
                *scale = 0.0;
            }
        }
    }

    result
}

// ============================================================================
// Shift Calibration via Rayleigh Quotient Statistics
// ============================================================================

/// Statistics from preconditioned Rayleigh quotient analysis.
///
/// These metrics measure how well the preconditioner approximates the
/// inverse of the operator on the current residual directions.
#[derive(Debug, Clone)]
pub struct PreconditionedRQStats {
    /// Original Rayleigh quotients ρ_j = ⟨r_j, A r_j⟩ / ⟨r_j, B r_j⟩
    pub rq_original: Vec<f64>,
    /// Preconditioned Rayleigh quotients ρ'_j = ⟨M⁻¹r_j, A M⁻¹r_j⟩ / ⟨M⁻¹r_j, B M⁻¹r_j⟩
    pub rq_preconditioned: Vec<f64>,
    /// Minimum preconditioned RQ
    pub rq_min: f64,
    /// Maximum preconditioned RQ
    pub rq_max: f64,
    /// Mean preconditioned RQ
    pub rq_mean: f64,
    /// Variance of preconditioned RQs
    pub rq_variance: f64,
    /// Spread measure: (max - min) / mean
    pub rq_spread: f64,
}

impl PreconditionedRQStats {
    /// Compute statistics from original and preconditioned Rayleigh quotients.
    pub fn compute(rq_original: Vec<f64>, rq_preconditioned: Vec<f64>) -> Self {
        let n = rq_preconditioned.len();
        if n == 0 {
            return Self {
                rq_original,
                rq_preconditioned,
                rq_min: 0.0,
                rq_max: 0.0,
                rq_mean: 0.0,
                rq_variance: 0.0,
                rq_spread: 0.0,
            };
        }

        let rq_min = rq_preconditioned
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let rq_max = rq_preconditioned
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let rq_mean = rq_preconditioned.iter().sum::<f64>() / n as f64;

        let rq_variance = if n > 1 {
            rq_preconditioned
                .iter()
                .map(|&rq| (rq - rq_mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            0.0
        };

        let rq_spread = if rq_mean.abs() > 1e-15 {
            (rq_max - rq_min) / rq_mean
        } else {
            0.0
        };

        Self {
            rq_original,
            rq_preconditioned,
            rq_min,
            rq_max,
            rq_mean,
            rq_variance,
            rq_spread,
        }
    }

    /// Check if the preconditioner is well-tuned.
    ///
    /// A well-tuned preconditioner should have:
    /// - Mean RQ close to 1 (ideally M⁻¹ ≈ A⁻¹)
    /// - Small spread (all directions preconditioned similarly)
    ///
    /// Returns true if spread < threshold and mean is in [0.5, 2.0].
    pub fn is_well_tuned(&self, spread_threshold: f64) -> bool {
        self.rq_spread < spread_threshold && self.rq_mean > 0.5 && self.rq_mean < 2.0
    }
}

/// Result of a shift calibration attempt.
#[derive(Debug, Clone)]
pub struct ShiftCalibrationResult {
    /// The original shift used
    pub original_shift: f64,
    /// The calibrated (optimal) shift
    pub calibrated_shift: f64,
    /// Statistics at original shift
    pub stats_original: PreconditionedRQStats,
    /// Statistics at calibrated shift (if different from original)
    pub stats_calibrated: Option<PreconditionedRQStats>,
    /// Whether calibration changed the shift
    pub shift_changed: bool,
    /// Number of trial shifts evaluated
    pub trials_evaluated: usize,
}

/// Configuration for shift calibration.
#[derive(Debug, Clone)]
pub struct ShiftCalibrationConfig {
    /// Scaling factors γ to try: shift' = γ × shift
    /// Default: [0.5, 0.75, 1.0, 1.5, 2.0]
    pub gamma_factors: Vec<f64>,
    /// Spread threshold for considering a shift "good"
    /// Default: 0.5
    pub spread_threshold: f64,
    /// Whether to prefer shifts that give RQ mean closer to 1
    /// Default: true
    pub prefer_unit_mean: bool,
}

impl Default for ShiftCalibrationConfig {
    fn default() -> Self {
        Self {
            gamma_factors: vec![0.5, 0.75, 1.0, 1.5, 2.0],
            spread_threshold: 0.5,
            prefer_unit_mean: true,
        }
    }
}

impl ShiftCalibrationConfig {
    /// Create a new calibration config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the gamma factors for trial shifts.
    pub fn with_gamma_factors(mut self, factors: Vec<f64>) -> Self {
        self.gamma_factors = factors;
        self
    }

    /// Set the spread threshold.
    pub fn with_spread_threshold(mut self, threshold: f64) -> Self {
        self.spread_threshold = threshold;
        self
    }

    /// Set whether to prefer unit mean.
    pub fn with_prefer_unit_mean(mut self, prefer: bool) -> Self {
        self.prefer_unit_mean = prefer;
        self
    }
}

/// Compute a quality score for a set of RQ statistics.
///
/// Lower score is better. The score combines:
/// - Spread penalty (larger spread = worse)
/// - Mean deviation from 1 (if `prefer_unit_mean` is true)
pub fn compute_shift_quality_score(stats: &PreconditionedRQStats, prefer_unit_mean: bool) -> f64 {
    let spread_penalty = stats.rq_spread;
    let mean_penalty = if prefer_unit_mean {
        (stats.rq_mean - 1.0).abs()
    } else {
        0.0
    };
    // Combined score: spread is primary, mean deviation is secondary
    spread_penalty + 0.5 * mean_penalty
}
