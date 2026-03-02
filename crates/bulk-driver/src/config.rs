//! Configuration types for bulk parameter sweeps.
//!
//! This module defines the TOML structure for bulk jobs, including parameter ranges
//! and output configuration.
//!
//! # Solver Types
//!
//! The bulk driver supports two solver types:
//!
//! - **Maxwell** (default): Photonic crystal band structure calculations using the
//!   Maxwell eigenproblem. Requires geometry, k-path, and polarization.
//!
//! - **EA (Envelope Approximation)**: Moiré lattice eigenproblems using the effective
//!   Hamiltonian H = V(R) - (η²/2)∇·M⁻¹(R)∇. Requires input data files for potential,
//!   mass tensor, and optionally group velocity for drift terms.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use mpb2d_core::{
    dielectric::DielectricOptions,
    eigensolver::EigensolverConfig,
    grid::Grid2D,
    io::PathSpec,
    polarization::Polarization,
};

// ============================================================================
// Solver Type Selection
// ============================================================================

/// Type of eigensolver to use.
///
/// The bulk driver can operate in different modes depending on the physics problem:
///
/// - `Maxwell`: Traditional photonic crystal band structure (requires geometry, k-path)
/// - `EA`: Envelope approximation for moiré lattices (requires input data files)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SolverType {
    /// Maxwell eigenproblem for photonic crystals.
    ///
    /// Solves: ∇ × (1/ε)∇ × H = (ω/c)² H
    /// Requires: geometry, k-path, polarization
    #[default]
    Maxwell,

    /// Envelope approximation for moiré lattices.
    ///
    /// Solves: H ψ = E ψ where H = V(R) - (η²/2)∇·M⁻¹(R)∇ - iη v_g·∇
    /// Requires: input data files for V, M⁻¹, and optionally v_g
    #[serde(rename = "ea")]
    EA,
}

impl std::fmt::Display for SolverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverType::Maxwell => write!(f, "Maxwell"),
            SolverType::EA => write!(f, "EA (Envelope Approximation)"),
        }
    }
}

// ============================================================================
// EA (Envelope Approximation) Configuration
// ============================================================================

/// Configuration for Envelope Approximation (EA) solver.
///
/// The EA solver reads pre-computed spatial data from files:
/// - `potential`: V(R) - the effective potential on the moiré superlattice
/// - `mass_inv`: M⁻¹(R) - the inverse mass tensor (spatially varying)
/// - `vg`: Optional group velocity for drift terms
///
/// # File Format
///
/// All input files should be binary files containing f64 values in **row-major**
/// (C-order) layout. The grid dimensions are inferred from the file size and
/// the configured resolution.
///
/// ## Data Layout
///
/// - `potential`: `[Nx * Ny]` f64 values, V(x, y) at each grid point
/// - `mass_inv`: `[Nx * Ny * 4]` f64 values, [m_xx, m_xy, m_yx, m_yy] at each point
/// - `vg`: `[Nx * Ny * 2]` f64 values, [vg_x, vg_y] at each point (optional)
///
/// # Example TOML
///
/// ```toml
/// [solver]
/// type = "ea"
///
/// [ea]
/// potential = "data/potential.bin"
/// mass_inv = "data/mass_inv.bin"
/// vg = "data/group_velocity.bin"  # optional
/// eta = 1.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EAConfig {
    /// Path to potential data file V(R).
    ///
    /// Binary file with Nx*Ny f64 values in row-major order.
    #[serde(default)]
    pub potential: Option<PathBuf>,

    /// Path to inverse mass tensor data file M⁻¹(R).
    ///
    /// Binary file with Nx*Ny*4 f64 values in row-major order.
    /// Components ordered as [m_xx, m_xy, m_yx, m_yy] at each grid point.
    #[serde(default)]
    pub mass_inv: Option<PathBuf>,

    /// Path to group velocity data file v_g(R) for drift term.
    ///
    /// Binary file with Nx*Ny*2 f64 values in row-major order.
    /// Components ordered as [vg_x, vg_y] at each grid point.
    /// If not provided, drift term is disabled.
    #[serde(default)]
    pub vg: Option<PathBuf>,

    /// Small parameter η in the envelope equation.
    ///
    /// This appears in both kinetic (-η²/2 ∇·M⁻¹∇) and drift (-iη v_g·∇) terms.
    #[serde(default = "default_eta")]
    pub eta: f64,

    /// Physical dimensions of the simulation domain [Lx, Ly].
    ///
    /// If not specified, defaults to [1.0, 1.0] (unit cell).
    #[serde(default = "default_domain_size")]
    pub domain_size: [f64; 2],

    /// Whether to use periodic boundary conditions.
    ///
    /// Currently only periodic BCs are supported.
    #[serde(default = "default_periodic")]
    pub periodic: bool,
}

fn default_eta() -> f64 {
    1.0
}

fn default_domain_size() -> [f64; 2] {
    [1.0, 1.0]
}

fn default_periodic() -> bool {
    true
}

impl EAConfig {
    /// Validate the EA configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Potential file is required
        if self.potential.is_none() {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver requires 'potential' file path".into(),
            ));
        }

        // Mass inverse file is required
        if self.mass_inv.is_none() {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver requires 'mass_inv' file path".into(),
            ));
        }

        // Eta must be positive
        if self.eta <= 0.0 {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver 'eta' must be positive".into(),
            ));
        }

        // Domain size must be positive
        if self.domain_size[0] <= 0.0 || self.domain_size[1] <= 0.0 {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver 'domain_size' components must be positive".into(),
            ));
        }

        Ok(())
    }

    /// Check if input files exist and are readable.
    pub fn check_files(&self) -> Result<(), ConfigError> {
        if let Some(ref path) = self.potential {
            if !path.exists() {
                return Err(ConfigError::InvalidEAConfig(format!(
                    "potential file not found: {}",
                    path.display()
                )));
            }
        }

        if let Some(ref path) = self.mass_inv {
            if !path.exists() {
                return Err(ConfigError::InvalidEAConfig(format!(
                    "mass_inv file not found: {}",
                    path.display()
                )));
            }
        }

        if let Some(ref path) = self.vg {
            if !path.exists() {
                return Err(ConfigError::InvalidEAConfig(format!(
                    "vg file not found: {}",
                    path.display()
                )));
            }
        }

        Ok(())
    }
}

/// Solver selection section in the TOML configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SolverSection {
    /// Type of solver to use: "maxwell" or "ea"
    #[serde(default, rename = "type")]
    pub solver_type: SolverType,
}

// ============================================================================
// Range Specification
// ============================================================================

/// Specification for a numeric parameter range.
///
/// Defines min, max, and step for generating a sequence of values.
/// The range is inclusive: values are `min, min+step, min+2*step, ..., max`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeSpec {
    /// Minimum value (inclusive)
    pub min: f64,
    /// Maximum value (inclusive)
    pub max: f64,
    /// Step size between values
    pub step: f64,
}

impl RangeSpec {
    /// Generate all values in this range.
    pub fn values(&self) -> Vec<f64> {
        let mut result = Vec::new();
        let mut v = self.min;
        // Use epsilon tolerance for floating point comparison
        while v <= self.max + self.step * 1e-9 {
            result.push(v);
            v += self.step;
        }
        result
    }

    /// Count how many values are in this range.
    pub fn count(&self) -> usize {
        if self.step <= 0.0 || self.max < self.min {
            return 0;
        }
        ((self.max - self.min) / self.step).floor() as usize + 1
    }
}

/// List of discrete values for a parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueList<T> {
    pub values: Vec<T>,
}

impl<T: Clone> ValueList<T> {
    pub fn values(&self) -> Vec<T> {
        self.values.clone()
    }

    pub fn count(&self) -> usize {
        self.values.len()
    }
}

// ============================================================================
// Parameter Ranges
// ============================================================================

/// Range specification for atom parameters.
///
/// Allows sweeping atom radius and position.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AtomRanges {
    /// Range for atom radius (in units of lattice constant)
    #[serde(default)]
    pub radius: Option<RangeSpec>,

    /// Range for x-position (fractional coordinates)
    #[serde(default)]
    pub pos_x: Option<RangeSpec>,

    /// Range for y-position (fractional coordinates)
    #[serde(default)]
    pub pos_y: Option<RangeSpec>,

    /// Range for epsilon inside the atom
    #[serde(default)]
    pub eps_inside: Option<RangeSpec>,
}

/// Range specifications for all parameters.
///
/// Any parameter specified here will be swept. Parameters NOT in ranges
/// must be specified in the base configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParameterRange {
    /// Range for background epsilon
    #[serde(default)]
    pub eps_bg: Option<RangeSpec>,

    /// Range for resolution (integer values)
    /// Specified as min, max, step but values are rounded to integers
    #[serde(default)]
    pub resolution: Option<RangeSpec>,

    /// List of polarizations to compute (e.g., ["TM", "TE"])
    #[serde(default)]
    pub polarization: Option<Vec<Polarization>>,

    /// List of lattice types to compute (e.g., ["square", "hexagonal"])
    #[serde(default)]
    pub lattice_type: Option<Vec<LatticeTypeSpec>>,

    /// Per-atom parameter ranges (indexed by atom number, 0-based)
    #[serde(default)]
    pub atoms: Vec<AtomRanges>,
}

impl ParameterRange {
    /// Calculate the total number of configurations from all ranges.
    pub fn total_configurations(&self) -> usize {
        let mut count = 1usize;

        if let Some(range) = &self.eps_bg {
            count *= range.count();
        }
        if let Some(range) = &self.resolution {
            count *= range.count();
        }
        if let Some(pols) = &self.polarization {
            count *= pols.len().max(1);
        }
        if let Some(types) = &self.lattice_type {
            count *= types.len().max(1);
        }

        for atom_range in &self.atoms {
            if let Some(r) = &atom_range.radius {
                count *= r.count();
            }
            if let Some(r) = &atom_range.pos_x {
                count *= r.count();
            }
            if let Some(r) = &atom_range.pos_y {
                count *= r.count();
            }
            if let Some(r) = &atom_range.eps_inside {
                count *= r.count();
            }
        }

        count
    }
}

// ============================================================================
// Lattice Type Specification
// ============================================================================

/// Lattice type for parameter sweeps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LatticeTypeSpec {
    Square,
    Rectangular,
    Triangular,
    Hexagonal,
}

impl std::fmt::Display for LatticeTypeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatticeTypeSpec::Square => write!(f, "square"),
            LatticeTypeSpec::Rectangular => write!(f, "rectangular"),
            LatticeTypeSpec::Triangular => write!(f, "triangular"),
            LatticeTypeSpec::Hexagonal => write!(f, "hexagonal"),
        }
    }
}

// ============================================================================
// Base Configuration
// ============================================================================

/// Base geometry configuration (non-swept parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseGeometry {
    /// Background epsilon (used if not in ranges)
    #[serde(default = "default_eps_bg")]
    pub eps_bg: f64,

    /// Lattice specification
    pub lattice: BaseLattice,

    /// Base atom definitions
    #[serde(default)]
    pub atoms: Vec<BaseAtom>,
}

fn default_eps_bg() -> f64 {
    12.0
}

/// Base lattice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseLattice {
    /// Explicit lattice vectors (for oblique/custom)
    #[serde(default)]
    pub a1: Option<[f64; 2]>,
    #[serde(default)]
    pub a2: Option<[f64; 2]>,

    /// Typed lattice specification
    #[serde(default, rename = "type")]
    pub lattice_type: Option<LatticeTypeSpec>,

    /// Lattice constant
    #[serde(default = "default_lattice_constant")]
    pub a: f64,

    /// Second lattice constant (for rectangular)
    #[serde(default)]
    pub b: Option<f64>,

    /// Angle in radians (for oblique)
    #[serde(default)]
    pub alpha: Option<f64>,
}

fn default_lattice_constant() -> f64 {
    1.0
}

/// Base atom configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseAtom {
    /// Position in fractional coordinates
    pub pos: [f64; 2],

    /// Radius in units of lattice constant
    pub radius: f64,

    /// Epsilon inside the atom
    #[serde(default = "default_eps_inside")]
    pub eps_inside: f64,
}

fn default_eps_inside() -> f64 {
    1.0
}

// ============================================================================
// Output Configuration
// ============================================================================

/// Output mode selection.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// One CSV file per solver run with complete band structure
    #[default]
    Full,
    /// Single merged CSV with selected k-points and bands
    Selective,
}

/// Specification for selective output.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SelectiveSpec {
    /// K-point indices to include (0-based)
    #[serde(default)]
    pub k_indices: Vec<usize>,

    /// K-point labels to include (e.g., "Gamma", "X", "M")
    #[serde(default)]
    pub k_labels: Vec<String>,

    /// Band indices to include (1-based, matching band1, band2, ...)
    #[serde(default)]
    pub bands: Vec<usize>,
}

/// I/O mode for output handling.
///
/// This controls how results are written during computation:
/// - **Sync**: Traditional synchronous writes (current behavior)
/// - **Batch**: Buffer results in memory and write in large chunks
/// - **Stream**: Emit results in real-time for live consumers
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IoMode {
    /// Traditional synchronous I/O (write each result immediately)
    #[default]
    Sync,
    /// Batched I/O with background writer (buffer results, write in chunks)
    Batch,
    /// Streaming mode for real-time consumers (Python, WASM)
    Stream,
}

/// Settings for batch mode I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSettings {
    /// Buffer size in bytes before triggering a flush (default: 10 MB)
    ///
    /// For typical band structure results (10 bands × 100 k-points):
    /// - Each result is ~10 KB
    /// - 10 MB buffer holds ~1000 results
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,

    /// Maximum time between flushes in seconds (optional)
    ///
    /// If set, the buffer will be flushed at this interval even if not full.
    /// Useful for long-running jobs where you want periodic checkpoints.
    #[serde(default)]
    pub flush_interval_secs: Option<f64>,
}

fn default_buffer_size() -> usize {
    10 * 1024 * 1024 // 10 MB
}

impl Default for BatchSettings {
    fn default() -> Self {
        Self {
            buffer_size: default_buffer_size(),
            flush_interval_secs: None,
        }
    }
}

/// Output configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output mode: "full" or "selective"
    #[serde(default)]
    pub mode: OutputMode,

    /// I/O mode: "sync", "batch", or "stream"
    #[serde(default)]
    pub io_mode: IoMode,

    /// Output directory for full mode (files named by job index)
    #[serde(default = "default_output_dir")]
    pub directory: PathBuf,

    /// Output filename for selective mode (single merged file)
    #[serde(default = "default_output_file")]
    pub filename: PathBuf,

    /// Prefix for full mode filenames
    #[serde(default = "default_prefix")]
    pub prefix: String,

    /// Selective output specification
    #[serde(default)]
    pub selective: SelectiveSpec,

    /// Write output in batches (number of jobs before flushing) - legacy
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Batch mode settings
    #[serde(default)]
    pub batch: BatchSettings,
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("./bulk_output")
}

fn default_output_file() -> PathBuf {
    PathBuf::from("./bulk_results.csv")
}

fn default_prefix() -> String {
    "job".to_string()
}

fn default_batch_size() -> usize {
    100
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            mode: OutputMode::Full,
            io_mode: IoMode::Sync,
            directory: default_output_dir(),
            filename: default_output_file(),
            prefix: default_prefix(),
            selective: SelectiveSpec::default(),
            batch_size: default_batch_size(),
            batch: BatchSettings::default(),
        }
    }
}

// ============================================================================
// Bulk Configuration Section
// ============================================================================

/// The `[bulk]` section that marks a TOML as a bulk request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkSection {
    /// Number of threads to use (default: all available cores)
    #[serde(default)]
    pub threads: Option<usize>,

    /// Enable verbose progress logging
    #[serde(default)]
    pub verbose: bool,

    /// Dry run: count jobs without executing
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for BulkSection {
    fn default() -> Self {
        Self {
            threads: None,
            verbose: false,
            dry_run: false,
        }
    }
}

// ============================================================================
// Complete Bulk Configuration
// ============================================================================

/// Complete configuration for a bulk parameter sweep.
///
/// A TOML file is recognized as a bulk request if it contains the `[bulk]` section.
///
/// # Solver Types
///
/// The configuration supports two solver types:
///
/// - **Maxwell** (default): Requires `geometry`, `k_path`/`path`, and `polarization`.
/// - **EA**: Requires `ea` section with input file paths. No k-path or geometry needed.
///
/// # Example (Maxwell)
///
/// ```toml
/// [bulk]
/// threads = 4
///
/// [solver]
/// type = "maxwell"  # default, can be omitted
///
/// [geometry]
/// eps_bg = 12.0
/// # ... atoms, lattice ...
///
/// [grid]
/// nx = 32
/// ny = 32
///
/// # ... rest of Maxwell config
/// ```
///
/// # Example (EA)
///
/// ```toml
/// [bulk]
/// threads = 4
///
/// [solver]
/// type = "ea"
///
/// [ea]
/// potential = "data/V.bin"
/// mass_inv = "data/M_inv.bin"
/// eta = 0.1
///
/// [grid]
/// nx = 64
/// ny = 64
///
/// [eigensolver]
/// n_bands = 10
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkConfig {
    /// Bulk execution settings (presence marks this as a bulk request)
    pub bulk: BulkSection,

    /// Solver type selection (maxwell or ea)
    #[serde(default)]
    pub solver: SolverSection,

    /// EA (Envelope Approximation) solver configuration.
    /// Required when solver.type = "ea".
    #[serde(default)]
    pub ea: EAConfig,

    /// Base geometry (non-swept parameters).
    /// Required for Maxwell solver, ignored for EA solver.
    #[serde(default)]
    pub geometry: Option<BaseGeometry>,

    /// Computational grid
    pub grid: Grid2D,

    /// Base polarization (used if not in ranges).
    /// Required for Maxwell solver, ignored for EA solver.
    #[serde(default = "default_polarization")]
    pub polarization: Polarization,

    /// K-path specification.
    /// Required for Maxwell solver, ignored for EA solver.
    #[serde(default)]
    pub path: Option<PathSpec>,

    /// Explicit k-path (overrides path preset).
    /// Required for Maxwell solver, ignored for EA solver.
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,

    /// Eigensolver configuration
    #[serde(default)]
    pub eigensolver: EigensolverConfig,

    /// Dielectric options.
    /// Only used by Maxwell solver.
    #[serde(default)]
    pub dielectric: DielectricOptions,

    /// Parameter ranges to sweep.
    /// For Maxwell: eps_bg, resolution, polarization, lattice_type, atoms.
    /// For EA: Currently no parameter sweeps (future: eta, domain_size).
    #[serde(default)]
    pub ranges: ParameterRange,

    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
}

fn default_polarization() -> Polarization {
    Polarization::TM
}

impl BulkConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        // First check if it contains [bulk] section
        if !s.contains("[bulk]") {
            return Err(ConfigError::NotBulkConfig);
        }

        let config: BulkConfig = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    /// Get the solver type.
    pub fn solver_type(&self) -> SolverType {
        self.solver.solver_type
    }

    /// Check if this is an EA solver configuration.
    pub fn is_ea(&self) -> bool {
        self.solver.solver_type == SolverType::EA
    }

    /// Check if this is a Maxwell solver configuration.
    pub fn is_maxwell(&self) -> bool {
        self.solver.solver_type == SolverType::Maxwell
    }

    /// Get geometry for Maxwell solver.
    /// Panics if geometry is not set (EA solver).
    pub fn geometry(&self) -> &BaseGeometry {
        self.geometry.as_ref().expect("geometry required for Maxwell solver")
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.solver.solver_type {
            SolverType::Maxwell => self.validate_maxwell(),
            SolverType::EA => self.validate_ea(),
        }
    }

    /// Validate Maxwell-specific configuration.
    fn validate_maxwell(&self) -> Result<(), ConfigError> {
        // Geometry is required for Maxwell
        if self.geometry.is_none() {
            return Err(ConfigError::InvalidMaxwellConfig(
                "Maxwell solver requires [geometry] section".into(),
            ));
        }

        // Validate ranges
        if let Some(range) = &self.ranges.eps_bg {
            if range.step <= 0.0 {
                return Err(ConfigError::InvalidRange("eps_bg step must be positive".into()));
            }
            if range.min > range.max {
                return Err(ConfigError::InvalidRange("eps_bg min > max".into()));
            }
        }

        if let Some(range) = &self.ranges.resolution {
            if range.step < 1.0 {
                return Err(ConfigError::InvalidRange(
                    "resolution step must be >= 1".into(),
                ));
            }
            if range.min < 4.0 {
                return Err(ConfigError::InvalidRange(
                    "resolution min must be >= 4".into(),
                ));
            }
        }

        // Validate selective output has selections
        if matches!(self.output.mode, OutputMode::Selective) {
            let spec = &self.output.selective;
            if spec.k_indices.is_empty() && spec.k_labels.is_empty() {
                return Err(ConfigError::InvalidOutput(
                    "selective mode requires k_indices or k_labels".into(),
                ));
            }
            if spec.bands.is_empty() {
                return Err(ConfigError::InvalidOutput(
                    "selective mode requires band indices".into(),
                ));
            }
        }

        Ok(())
    }

    /// Validate EA-specific configuration.
    fn validate_ea(&self) -> Result<(), ConfigError> {
        // EA config is required
        self.ea.validate()?;

        // For EA, we don't need geometry or k-path
        // Warn if they're provided (they'll be ignored)
        // (This is just a note - we won't error)

        Ok(())
    }

    /// Get the effective number of threads.
    /// Defaults to physical CPU cores (optimal for CPU-bound workloads).
    pub fn effective_threads(&self) -> usize {
        self.bulk.threads.unwrap_or_else(num_cpus::get_physical)
    }

    /// Get total number of jobs.
    pub fn total_jobs(&self) -> usize {
        self.ranges.total_configurations()
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Configuration parsing errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("TOML file does not contain [bulk] section - not a bulk configuration")]
    NotBulkConfig,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Invalid parameter range: {0}")]
    InvalidRange(String),

    #[error("Invalid output configuration: {0}")]
    InvalidOutput(String),

    #[error("Configuration conflict: {0}")]
    Conflict(String),

    #[error("Invalid Maxwell solver configuration: {0}")]
    InvalidMaxwellConfig(String),

    #[error("Invalid EA solver configuration: {0}")]
    InvalidEAConfig(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_spec_values() {
        let range = RangeSpec {
            min: 0.1,
            max: 0.5,
            step: 0.1,
        };
        let values = range.values();
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.1).abs() < 1e-10);
        assert!((values[4] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn range_spec_count() {
        let range = RangeSpec {
            min: 24.0,
            max: 64.0,
            step: 8.0,
        };
        assert_eq!(range.count(), 6); // 24, 32, 40, 48, 56, 64
    }

    #[test]
    fn not_bulk_config() {
        let content = r#"
polarization = "TM"

[geometry]
eps_bg = 12.0
"#;
        let result = BulkConfig::from_str(content);
        assert!(matches!(result, Err(ConfigError::NotBulkConfig)));
    }

    #[test]
    fn custom_k_path_parsing() {
        // NOTE: k_path must be at top-level in TOML (before any [section] headers)
        // or it will be absorbed into the preceding section!
        let content = r#"
k_path = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]

[bulk]
threads = 1

[geometry]
eps_bg = 1.0

[geometry.lattice]
type = "square"

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.3
eps_inside = 1.0

[grid]
nx = 16
ny = 16

[eigensolver]
n_bands = 2

[ranges]
[[ranges.atoms]]

[output]
mode = "full"
"#;
        let config = BulkConfig::from_str(content).expect("should parse");
        assert_eq!(config.k_path.len(), 3, "k_path should have 3 points");
        assert_eq!(config.k_path[0], [0.0, 0.0]);
        assert_eq!(config.k_path[1], [0.1, 0.0]);
        assert_eq!(config.k_path[2], [0.2, 0.0]);
        assert!(config.is_maxwell(), "default solver should be Maxwell");
    }

    #[test]
    fn ea_solver_config_parsing() {
        let content = r#"
[bulk]
threads = 2

[solver]
type = "ea"

[ea]
potential = "test_data/V.bin"
mass_inv = "test_data/M_inv.bin"
eta = 0.5
domain_size = [10.0, 10.0]

[grid]
nx = 64
ny = 64

[eigensolver]
n_bands = 8
"#;
        let config = BulkConfig::from_str(content).expect("should parse");
        assert!(config.is_ea(), "should be EA solver");
        assert_eq!(config.solver_type(), SolverType::EA);
        assert_eq!(config.ea.eta, 0.5);
        assert_eq!(config.ea.domain_size, [10.0, 10.0]);
        assert_eq!(
            config.ea.potential.as_ref().map(|p| p.to_str().unwrap()),
            Some("test_data/V.bin")
        );
    }

    #[test]
    fn maxwell_requires_geometry() {
        let content = r#"
[bulk]
threads = 1

[solver]
type = "maxwell"

[grid]
nx = 16
ny = 16

[eigensolver]
n_bands = 2
"#;
        let result = BulkConfig::from_str(content);
        assert!(matches!(result, Err(ConfigError::InvalidMaxwellConfig(_))));
    }

    #[test]
    fn ea_requires_input_files() {
        let content = r#"
[bulk]
threads = 1

[solver]
type = "ea"

[ea]
# Missing potential and mass_inv

[grid]
nx = 16
ny = 16

[eigensolver]
n_bands = 2
"#;
        let result = BulkConfig::from_str(content);
        assert!(matches!(result, Err(ConfigError::InvalidEAConfig(_))));
    }

    #[test]
    fn solver_type_display() {
        assert_eq!(format!("{}", SolverType::Maxwell), "Maxwell");
        assert_eq!(format!("{}", SolverType::EA), "EA (Envelope Approximation)");
    }
}
