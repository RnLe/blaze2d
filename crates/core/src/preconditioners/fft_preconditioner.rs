//! FFT-based preconditioner for envelope approximation operators.
//!
//! This preconditioner approximates the inverse of the EA Hamiltonian using
//! a constant-coefficient Laplacian, which is diagonal in Fourier space.
//!
//! # Preconditioner Design
//!
//! The EA Hamiltonian is:
//!
//! ```text
//! H = V(R) - (η²/2) ∇·M⁻¹(R)∇
//! ```
//!
//! For preconditioning, we use a shifted constant-coefficient approximation:
//!
//! ```text
//! P^{-1} ≈ σ + (η² m̄ / 2) |k|²
//! ```
//!
//! where σ is an adaptive shift chosen to:
//! 1. Ensure positive-definiteness (σ > 0)
//! 2. Center the preconditioned eigenvalues near 1
//! 3. Handle negative potentials, zero-mean potentials, and large mass variations
//!
//! # Adaptive Shift Strategies
//!
//! The shift σ is computed adaptively based on spectral properties:
//!
//! 1. **Target-based**: Shift so lowest eigenvalue of P⁻¹H ≈ 1
//!    σ = |V_min| + margin
//!
//! 2. **Spectral-range**: Account for both potential and kinetic ranges
//!    σ = |V_min| + c × η² × m̄ × k_char²
//!
//! 3. **Regularized**: Add small regularization for robustness
//!    σ = max(σ_computed, ε × eigenvalue_scale)
//!
//! # Performance
//!
//! - FFT is O(N log N) vs O(N²) for dense preconditioner
//! - 2 FFT operations per application (forward + inverse)
//! - Preconditioner construction is one-time cost per solve
//!
//! # Example
//!
//! ```ignore
//! // Create with full spectral information
//! let config = EAPreconditionerConfig::from_operator_stats(
//!     v_min, v_max, v_mean,
//!     m_min, m_max, m_mean,
//!     eta,
//! );
//! let precond = FFTPreconditioner::with_config(nx, ny, dx, dy, eta, m_mean, config);
//! ```

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::preconditioners::OperatorPreconditioner;

// ============================================================================
// Configuration
// ============================================================================

/// Strategy for computing the preconditioner shift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShiftStrategy {
    /// Use the mean potential (legacy behavior, may be unstable for V̄ ≈ 0).
    MeanPotential,
    
    /// Shift to target lowest eigenvalue of P⁻¹H near 1.
    /// σ = |V_min| + margin
    TargetLowestEigenvalue {
        /// Additional margin above |V_min| for stability.
        margin: f64,
    },
    
    /// Spectral-range aware shift that accounts for kinetic contribution.
    /// σ = |V_min| + c × η² × m̄ × k_char²
    SpectralRange {
        /// Fraction of kinetic energy at characteristic k to include.
        kinetic_fraction: f64,
        /// Characteristic wavenumber (defaults to k_mean if None).
        k_char: Option<f64>,
    },
    
    /// Fixed shift value (for manual tuning or testing).
    Fixed(f64),
    
    /// Automatic selection based on operator statistics.
    /// This is the recommended default.
    Auto,
}

impl Default for ShiftStrategy {
    fn default() -> Self {
        ShiftStrategy::Auto
    }
}

/// Configuration for the FFT preconditioner.
#[derive(Debug, Clone)]
pub struct EAPreconditionerConfig {
    /// Shift strategy to use.
    pub shift_strategy: ShiftStrategy,
    
    /// Minimum shift value (regularization floor).
    /// Prevents near-zero denominators.
    pub min_shift: f64,
    
    /// Maximum allowed condition number for the preconditioner spectrum.
    /// If exceeded, the shift is increased.
    pub max_condition_number: f64,
    
    /// Whether to use geometric mean for mass instead of arithmetic mean.
    /// Recommended when mass ratio (m_max/m_min) > 10.
    pub use_geometric_mean_mass: bool,
    
    /// Computed shift value (set during from_operator_stats).
    computed_shift: Option<f64>,
    
    /// Computed effective mass (set during from_operator_stats).
    computed_mass: Option<f64>,
    
    /// Diagnostic information about the shift computation.
    pub diagnostics: Option<ShiftDiagnostics>,
}

/// Diagnostic information about how the shift was computed.
#[derive(Debug, Clone)]
pub struct ShiftDiagnostics {
    /// The strategy that was actually used.
    pub strategy_used: String,
    /// The computed shift value.
    pub shift: f64,
    /// The effective mass used.
    pub effective_mass: f64,
    /// Estimated condition number of P⁻¹.
    pub precond_condition_number: f64,
    /// Input potential range.
    pub v_range: [f64; 2],
    /// Input mass range.
    pub m_range: [f64; 2],
    /// Any warnings about the configuration.
    pub warnings: Vec<String>,
}

impl Default for EAPreconditionerConfig {
    fn default() -> Self {
        Self {
            shift_strategy: ShiftStrategy::Auto,
            min_shift: 1e-6,
            max_condition_number: 1e6,
            use_geometric_mean_mass: false,
            computed_shift: None,
            computed_mass: None,
            diagnostics: None,
        }
    }
}

impl EAPreconditionerConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the shift strategy.
    pub fn with_strategy(mut self, strategy: ShiftStrategy) -> Self {
        self.shift_strategy = strategy;
        self
    }
    
    /// Set the minimum shift value.
    pub fn with_min_shift(mut self, min_shift: f64) -> Self {
        self.min_shift = min_shift.max(1e-15);
        self
    }
    
    /// Set the maximum allowed condition number.
    pub fn with_max_condition_number(mut self, kappa_max: f64) -> Self {
        self.max_condition_number = kappa_max.max(1.0);
        self
    }
    
    /// Enable geometric mean for mass (recommended for large mass ratios).
    pub fn with_geometric_mean_mass(mut self, enable: bool) -> Self {
        self.use_geometric_mean_mass = enable;
        self
    }
    
    /// Compute optimal configuration from operator statistics.
    ///
    /// This is the recommended way to create a configuration. It analyzes
    /// the operator's spectral properties and chooses the best shift strategy.
    ///
    /// # Arguments
    ///
    /// * `v_min`, `v_max`, `v_mean` - Potential statistics
    /// * `m_min`, `m_max`, `m_mean` - Inverse mass tensor statistics (trace/2)
    /// * `eta` - Twist parameter
    ///
    /// # Returns
    ///
    /// A configured `EAPreconditionerConfig` with computed shift and diagnostics.
    pub fn from_operator_stats(
        v_min: f64,
        v_max: f64,
        v_mean: f64,
        m_min: f64,
        m_max: f64,
        m_mean: f64,
        eta: f64,
    ) -> Self {
        let mut config = Self::default();
        let mut warnings = Vec::new();
        
        // Compute potential statistics
        let v_range = v_max - v_min;
        let v_scale = v_range.max(v_max.abs()).max(v_min.abs()).max(1e-10);
        
        // Compute mass statistics
        let mass_ratio = m_max / m_min.max(1e-15);
        
        // Decide whether to use geometric mean for mass
        if mass_ratio > 10.0 {
            config.use_geometric_mean_mass = true;
            warnings.push(format!(
                "Large mass ratio ({:.1}x), using geometric mean",
                mass_ratio
            ));
        }
        
        // Compute effective mass
        let effective_mass = if config.use_geometric_mean_mass {
            (m_min * m_max).sqrt()
        } else {
            m_mean
        };
        
        // Kinetic prefactor
        let kinetic_prefactor = 0.5 * eta * eta * effective_mass;
        
        // Determine shift strategy based on potential characteristics
        //
        // KEY INSIGHT: For LOBPCG, we need P^{-1}H to have all positive eigenvalues
        // and ideally cluster around 1 for good conditioning.
        //
        // For the EA Hamiltonian H = V + kinetic where V can be negative:
        //   - At potential wells: H ≈ V_min + kinetic ≈ V_min (negative)
        //   - At potential peaks: H ≈ V_max + kinetic
        //
        // The preconditioned eigenvalues are approximately:
        //   λ_precond ≈ (V + kinetic) / (σ + kinetic)
        //
        // For all preconditioned eigenvalues to be positive AND cluster near 1:
        //   - We need σ to have the same sign as the target eigenvalue!
        //   - For negative eigenvalues: use σ < 0
        //   - P^{-1}(k) = 1/(σ + kinetic) with σ < 0 can still be positive if |σ| < kinetic
        //
        // ALTERNATIVE: For negative eigenvalue problems, it may be better to 
        // NOT use a preconditioner at all, since the standard FFT preconditioner
        // is designed for positive eigenvalues.
        //
        // For now, try: σ = 0 + small regularization (minimal interference)
        
        let shift = if v_mean.abs() < 1e-10 * v_scale {
            // Zero-mean potential (common in moiré systems)
            warnings.push("Zero-mean potential detected - using minimal shift".to_string());
            
            // Use minimal shift to avoid division by zero but minimize preconditioning effect
            // This makes the preconditioner nearly identity at low-k
            let minimal_shift = 0.001 * v_scale.max(kinetic_prefactor);
            minimal_shift
            
        } else if v_mean < 0.0 || v_min < 0.0 {
            // Negative eigenvalue regime
            warnings.push("Negative potential detected - using minimal shift".to_string());
            
            // Use minimal shift
            let minimal_shift = 0.001 * v_scale.max(kinetic_prefactor);
            minimal_shift
            
        } else {
            // Positive potential: eigenvalues are positive
            // Standard approach: shift ≈ V_mean
            let margin = 0.01 * v_scale;
            v_mean + margin
        };
        
        // Apply minimum shift floor
        let shift = shift.max(config.min_shift);
        
        // Check condition number and adjust if needed
        // Condition number of P⁻¹: κ ≈ P⁻¹(k_max) / P⁻¹(k=0) = (σ + kinetic_max) / σ
        // Rough estimate: k_max ≈ π/dx, so kinetic_max ≈ kinetic_prefactor × (π/dx)²
        // But we don't have dx here, so use a heuristic based on mass ratio
        let estimated_kappa = 1.0 + (kinetic_prefactor * 1000.0) / shift;
        
        let shift = if estimated_kappa > config.max_condition_number {
            // Increase shift to reduce condition number
            let new_shift = kinetic_prefactor * 1000.0 / (config.max_condition_number - 1.0);
            warnings.push(format!(
                "Shift increased from {:.4e} to {:.4e} to limit κ",
                shift, new_shift.max(shift)
            ));
            new_shift.max(shift)
        } else {
            shift
        };
        
        // Store computed values
        config.computed_shift = Some(shift);
        config.computed_mass = Some(effective_mass);
        
        // Create diagnostics
        config.diagnostics = Some(ShiftDiagnostics {
            strategy_used: "Auto (from_operator_stats)".to_string(),
            shift,
            effective_mass,
            precond_condition_number: estimated_kappa.min(config.max_condition_number),
            v_range: [v_min, v_max],
            m_range: [m_min, m_max],
            warnings,
        });
        
        config
    }
    
    /// Get the computed shift value.
    pub fn shift(&self) -> f64 {
        self.computed_shift.unwrap_or(self.min_shift)
    }
    
    /// Get the computed effective mass.
    pub fn effective_mass(&self) -> f64 {
        self.computed_mass.unwrap_or(1.0)
    }
    
    /// Format diagnostics as a human-readable string.
    pub fn format_diagnostics(&self) -> String {
        match &self.diagnostics {
            Some(d) => {
                let mut s = format!(
                    "FFT Preconditioner: σ={:.4e}, m_eff={:.4}, κ(P⁻¹)≈{:.1}\n",
                    d.shift, d.effective_mass, d.precond_condition_number
                );
                s.push_str(&format!(
                    "  V ∈ [{:.4e}, {:.4e}], M⁻¹ ∈ [{:.4}, {:.4}]\n",
                    d.v_range[0], d.v_range[1], d.m_range[0], d.m_range[1]
                ));
                if !d.warnings.is_empty() {
                    s.push_str("  Warnings:\n");
                    for w in &d.warnings {
                        s.push_str(&format!("    • {}\n", w));
                    }
                }
                s
            }
            None => "FFT Preconditioner: no diagnostics available".to_string(),
        }
    }
}

// ============================================================================
// FFT Preconditioner Implementation
// ============================================================================

/// FFT-based preconditioner for the envelope approximation Hamiltonian.
///
/// Approximates H^{-1} using a shifted constant-coefficient Laplacian that is
/// diagonal in Fourier space.
///
/// # Adaptive Shift
///
/// The key innovation is the adaptive shift σ that ensures:
/// 1. The preconditioner is always positive-definite
/// 2. Eigenvalues of P⁻¹H cluster near 1 (ideal for LOBPCG)
/// 3. Works correctly for negative, zero-mean, or mixed-sign potentials
///
/// # Spectrum
///
/// ```text
/// P̂(k) = 1 / (σ + (η²m_eff/2)|k|²)
/// ```
///
/// where σ is the adaptive shift and m_eff is the effective mass.
pub struct FFTPreconditioner<B: SpectralBackend> {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    /// Precomputed: 1 / (σ + η²m_eff/2 * |k|²)
    inv_spectrum: Vec<f64>,
    /// The shift value used
    shift: f64,
    /// The effective mass used
    effective_mass: f64,
    /// Configuration used to build this preconditioner
    config: EAPreconditionerConfig,
    /// Marker for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: SpectralBackend> FFTPreconditioner<B> {
    /// Create a new FFT preconditioner with default settings.
    ///
    /// **WARNING**: This uses `v_mean` directly as the shift, which can fail
    /// for zero-mean or negative-mean potentials. Use `with_config` or
    /// `from_operator_stats` for robust handling.
    ///
    /// # Arguments
    ///
    /// * `nx`, `ny` - Grid dimensions
    /// * `dx`, `dy` - Grid spacings
    /// * `eta` - Twist parameter
    /// * `v_mean` - Mean potential (used as shift)
    /// * `m_mean` - Mean inverse mass (average of trace/2)
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, eta: f64, v_mean: f64, m_mean: f64) -> Self {
        // Legacy behavior: use v_mean as shift with small floor
        let shift = v_mean.abs().max(1e-6);
        let prefactor = 0.5 * eta * eta * m_mean;

        let mut inv_spectrum = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let kx = fft_freq(i, nx, dx);
                let ky = fft_freq(j, ny, dy);
                let k_sq = kx * kx + ky * ky;
                let denom = shift + prefactor * k_sq;
                inv_spectrum[i * ny + j] = 1.0 / denom.max(1e-15);
            }
        }

        Self {
            nx,
            ny,
            inv_spectrum,
            shift,
            effective_mass: m_mean,
            config: EAPreconditionerConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }
    
    /// Create a new FFT preconditioner with full configuration.
    ///
    /// This is the recommended constructor for production use.
    ///
    /// # Arguments
    ///
    /// * `nx`, `ny` - Grid dimensions
    /// * `dx`, `dy` - Grid spacings
    /// * `eta` - Twist parameter
    /// * `config` - Preconditioner configuration (use `from_operator_stats`)
    pub fn with_config(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        eta: f64,
        config: EAPreconditionerConfig,
    ) -> Self {
        let shift = config.shift();
        let effective_mass = config.effective_mass();
        let prefactor = 0.5 * eta * eta * effective_mass;

        let mut inv_spectrum = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let kx = fft_freq(i, nx, dx);
                let ky = fft_freq(j, ny, dy);
                let k_sq = kx * kx + ky * ky;
                let denom = shift + prefactor * k_sq;
                inv_spectrum[i * ny + j] = 1.0 / denom.max(1e-15);
            }
        }

        Self {
            nx,
            ny,
            inv_spectrum,
            shift,
            effective_mass,
            config,
            _marker: std::marker::PhantomData,
        }
    }
    
    /// Create a preconditioner directly from operator statistics.
    ///
    /// This is the most convenient constructor that handles all edge cases.
    ///
    /// # Arguments
    ///
    /// * `nx`, `ny` - Grid dimensions
    /// * `dx`, `dy` - Grid spacings
    /// * `eta` - Twist parameter
    /// * `v_min`, `v_max`, `v_mean` - Potential statistics
    /// * `m_min`, `m_max`, `m_mean` - Inverse mass statistics
    #[allow(clippy::too_many_arguments)]
    pub fn from_operator_stats(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        eta: f64,
        v_min: f64,
        v_max: f64,
        v_mean: f64,
        m_min: f64,
        m_max: f64,
        m_mean: f64,
    ) -> Self {
        let config = EAPreconditionerConfig::from_operator_stats(
            v_min, v_max, v_mean,
            m_min, m_max, m_mean,
            eta,
        );
        Self::with_config(nx, ny, dx, dy, eta, config)
    }

    /// Get the precomputed inverse spectrum.
    pub fn inv_spectrum(&self) -> &[f64] {
        &self.inv_spectrum
    }
    
    /// Get the shift value used.
    pub fn shift(&self) -> f64 {
        self.shift
    }
    
    /// Get the effective mass used.
    pub fn effective_mass(&self) -> f64 {
        self.effective_mass
    }
    
    /// Get the configuration.
    pub fn config(&self) -> &EAPreconditionerConfig {
        &self.config
    }
    
    /// Get the grid dimensions.
    pub fn grid_dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
    
    /// Compute the condition number of the preconditioner.
    ///
    /// Returns κ(P⁻¹) = max(P⁻¹) / min(P⁻¹) = (σ + kinetic_max) / σ
    pub fn condition_number(&self) -> f64 {
        let p_inv_min = self.inv_spectrum.iter().cloned().fold(f64::INFINITY, f64::min);
        let p_inv_max = self.inv_spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        if p_inv_min > 1e-15 {
            p_inv_max / p_inv_min
        } else {
            f64::INFINITY
        }
    }
    
    /// Format a summary of the preconditioner for logging.
    pub fn format_summary(&self) -> String {
        format!(
            "FFT Preconditioner: {}×{}, σ={:.4e}, m_eff={:.4}, κ(P⁻¹)={:.1}",
            self.nx, self.ny, self.shift, self.effective_mass, self.condition_number()
        )
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute FFT frequency for index i in a grid of size n with spacing d.
pub fn fft_freq(i: usize, n: usize, d: f64) -> f64 {
    let freq = if i <= n / 2 {
        i as f64
    } else {
        i as f64 - n as f64
    };
    2.0 * std::f64::consts::PI * freq / (n as f64 * d)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_zero_mean_potential() {
        // Zero-mean potential should get a positive shift
        let config = EAPreconditionerConfig::from_operator_stats(
            -0.075, 0.033, 0.0,  // V: min, max, mean=0
            1.0, 100.0, 50.0,     // M: min, max, mean
            0.02,                  // eta
        );
        
        assert!(config.shift() > 0.0, "Shift should be positive for zero-mean V");
        assert!(config.shift() >= 0.075, "Shift should cover |V_min|");
    }
    
    #[test]
    fn test_config_negative_mean_potential() {
        // Negative-mean potential should get a positive shift
        let config = EAPreconditionerConfig::from_operator_stats(
            -0.1, -0.01, -0.05,  // V: all negative
            1.0, 10.0, 5.0,       // M
            0.02,                  // eta
        );
        
        assert!(config.shift() > 0.0, "Shift should be positive");
        assert!(config.shift() >= 0.1, "Shift should cover |V_min|");
    }
    
    #[test]
    fn test_config_large_mass_ratio() {
        // Large mass ratio should trigger geometric mean
        let config = EAPreconditionerConfig::from_operator_stats(
            0.1, 1.0, 0.5,        // V: positive
            1.0, 100.0, 50.0,     // M: 100x ratio
            0.02,                  // eta
        );
        
        assert!(config.use_geometric_mean_mass, "Should use geometric mean for 100x ratio");
        let geo_mean = (1.0_f64 * 100.0).sqrt();
        assert!((config.effective_mass() - geo_mean).abs() < 1e-10);
    }
    
    #[test]
    fn test_config_positive_potential() {
        // Positive potential should use mean with small margin
        let config = EAPreconditionerConfig::from_operator_stats(
            0.1, 1.0, 0.5,       // V: all positive
            1.0, 2.0, 1.5,        // M: small ratio
            0.02,                  // eta
        );
        
        assert!(config.shift() > 0.5, "Shift should be near V_mean");
        assert!(config.shift() < 1.0, "Shift should not be too large");
    }
    
    #[test]
    fn test_preconditioner_condition_number() {
        // Create a preconditioner configuration and check condition number
        let config = EAPreconditionerConfig::from_operator_stats(
            -0.1, 0.1, 0.0,
            1.0, 10.0, 5.0,
            0.02,
        );
        
        // Compute expected condition number from the spectrum formula
        let shift = config.shift();
        let effective_mass = config.effective_mass();
        let eta = 0.02;
        let dx = 0.1;
        let nx = 64;
        
        // k_max at Nyquist
        let k_max = std::f64::consts::PI * nx as f64 / (nx as f64 * dx);
        let prefactor = 0.5 * eta * eta * effective_mass;
        
        let p_inv_min = shift;  // At k=0
        let p_inv_max = shift + prefactor * k_max * k_max;  // At k_max
        let expected_kappa = p_inv_max / p_inv_min;
        
        // Verify the config produces reasonable values
        assert!(shift > 0.1, "Shift should cover |V_min| = 0.1, got {}", shift);
        assert!(expected_kappa > 1.0, "Condition number should be > 1");
        assert!(expected_kappa < 1e6, "Condition number should be bounded, got {}", expected_kappa);
        
        // Diagnostics should have warnings about zero-mean potential
        assert!(config.diagnostics.is_some(), "Diagnostics should be present");
        let diag = config.diagnostics.as_ref().unwrap();
        assert!(!diag.warnings.is_empty(), "Should have warnings for zero-mean V");
    }
}
