//! Configuration types for bulk parameter sweeps.
//!
//! This module defines the TOML structure for bulk jobs, including parameter ranges
//! and output configuration. These types are platform-agnostic and shared between
//! native and WASM drivers.
//!
//! # Configuration Format
//!
//! The bulk driver supports two configuration formats:
//!
//! ## New Format (Recommended): Ordered Sweeps with `[[sweeps]]`
//!
//! Sweeps are processed as **nested loops in the order listed**. The first sweep
//! is the outermost loop, the last is the innermost.
//!
//! ```toml
//! [bulk]
//! verbose = true
//!
//! # Sweeps are processed in order: radius is outer loop, eps_bg is inner
//! [[sweeps]]
//! parameter = "atom0.radius"
//! min = 0.2
//! max = 0.4
//! step = 0.1
//!
//! [[sweeps]]
//! parameter = "eps_bg"
//! min = 10.0
//! max = 12.0
//! step = 1.0
//!
//! [defaults.geometry]
//! eps_bg = 12.0
//! # ... rest of config
//! ```
//!
//! ## Legacy Format: `[ranges]` Section
//!
//! The old format with `[ranges]` section is still supported but deprecated.
//! Sweep order is not guaranteed with the legacy format.
//!
//! # Solver Types
//!
//! - **Maxwell** (default): Photonic crystal band structure calculations
//! - **EA (Envelope Approximation)**: Moiré lattice eigenproblems

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EAConfig {
    /// Path to potential data file V(R).
    #[serde(default)]
    pub potential: Option<PathBuf>,

    /// Path to inverse mass tensor data file M⁻¹(R).
    #[serde(default)]
    pub mass_inv: Option<PathBuf>,

    /// Path to group velocity data file v_g(R) for drift term.
    #[serde(default)]
    pub vg: Option<PathBuf>,

    /// Small parameter η in the envelope equation.
    #[serde(default = "default_eta")]
    pub eta: f64,

    /// Physical dimensions of the simulation domain [Lx, Ly].
    #[serde(default = "default_domain_size")]
    pub domain_size: [f64; 2],

    /// Whether to use periodic boundary conditions.
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
        if self.potential.is_none() {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver requires 'potential' file path".into(),
            ));
        }

        if self.mass_inv.is_none() {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver requires 'mass_inv' file path".into(),
            ));
        }

        if self.eta <= 0.0 {
            return Err(ConfigError::InvalidEAConfig(
                "EA solver 'eta' must be positive".into(),
            ));
        }

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
// Ordered Sweep Specification (New Format)
// ============================================================================

/// A single parameter sweep specification.
///
/// Sweeps are processed in the order they appear in the TOML file, forming
/// nested loops. The first sweep is the outermost loop.
///
/// # Parameter Paths
///
/// Parameters are specified using dot notation:
/// - `eps_bg` - Background epsilon
/// - `resolution` - Grid resolution
/// - `polarization` - Polarization (TM/TE)
/// - `lattice_type` - Lattice type
/// - `atom0.radius` - First atom's radius
/// - `atom0.pos_x` - First atom's x position
/// - `atom0.pos_y` - First atom's y position  
/// - `atom0.eps_inside` - First atom's internal epsilon
/// - `atom1.radius` - Second atom's radius (etc.)
///
/// # Example
///
/// ```toml
/// [[sweeps]]
/// parameter = "atom0.radius"
/// min = 0.2
/// max = 0.4
/// step = 0.1
///
/// [[sweeps]]
/// parameter = "polarization"
/// values = ["TM", "TE"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepSpec {
    /// Parameter path using dot notation (e.g., "eps_bg", "atom0.radius")
    pub parameter: String,

    /// Minimum value for range-based sweep
    #[serde(default)]
    pub min: Option<f64>,

    /// Maximum value for range-based sweep
    #[serde(default)]
    pub max: Option<f64>,

    /// Step size for range-based sweep
    #[serde(default)]
    pub step: Option<f64>,

    /// Discrete values for non-numeric or specific value sweeps.
    /// Use for polarization, lattice_type, or when you want specific values.
    #[serde(default)]
    pub values: Option<Vec<toml::Value>>,
}

impl SweepSpec {
    /// Check if this is a range-based sweep (min/max/step).
    pub fn is_range(&self) -> bool {
        self.min.is_some() && self.max.is_some() && self.step.is_some()
    }

    /// Check if this is a discrete values sweep.
    pub fn is_discrete(&self) -> bool {
        self.values.is_some()
    }

    /// Get the number of values in this sweep.
    pub fn count(&self) -> usize {
        if let Some(ref values) = self.values {
            values.len()
        } else if let (Some(min), Some(max), Some(step)) = (self.min, self.max, self.step) {
            if step <= 0.0 || max < min {
                0
            } else {
                ((max - min) / step).floor() as usize + 1
            }
        } else {
            0
        }
    }

    /// Convert to a RangeSpec if this is a range-based sweep.
    pub fn to_range_spec(&self) -> Option<RangeSpec> {
        match (self.min, self.max, self.step) {
            (Some(min), Some(max), Some(step)) => Some(RangeSpec { min, max, step }),
            _ => None,
        }
    }

    /// Validate the sweep specification.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Must have either range or values
        if !self.is_range() && !self.is_discrete() {
            return Err(ConfigError::InvalidSweep(format!(
                "sweep '{}' must specify either (min, max, step) or values",
                self.parameter
            )));
        }

        // Cannot have both
        if self.is_range() && self.is_discrete() {
            return Err(ConfigError::InvalidSweep(format!(
                "sweep '{}' cannot have both range and values",
                self.parameter
            )));
        }

        // Validate range
        if self.is_range() {
            let min = self.min.unwrap();
            let max = self.max.unwrap();
            let step = self.step.unwrap();

            if step <= 0.0 {
                return Err(ConfigError::InvalidSweep(format!(
                    "sweep '{}' step must be positive",
                    self.parameter
                )));
            }
            if min > max {
                return Err(ConfigError::InvalidSweep(format!(
                    "sweep '{}' min ({}) > max ({})",
                    self.parameter, min, max
                )));
            }
        }

        // Validate parameter path
        validate_parameter_path(&self.parameter)?;

        Ok(())
    }
}

/// A concrete value in a sweep dimension.
///
/// This preserves type information for proper output formatting.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepValue {
    /// Floating-point value (eps_bg, radius, pos_x, etc.)
    Float(f64),
    /// Integer value (resolution)
    Int(i64),
    /// String value (polarization, lattice_type)
    String(String),
}

impl SweepValue {
    /// Format as a string for CSV output.
    pub fn to_csv_string(&self) -> String {
        match self {
            SweepValue::Float(v) => format!("{:.6}", v),
            SweepValue::Int(v) => v.to_string(),
            SweepValue::String(s) => s.clone(),
        }
    }

    /// Try to get as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SweepValue::Float(v) => Some(*v),
            SweepValue::Int(v) => Some(*v as f64),
            SweepValue::String(_) => None,
        }
    }

    /// Try to get as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SweepValue::Float(v) => Some(*v as i64),
            SweepValue::Int(v) => Some(*v),
            SweepValue::String(_) => None,
        }
    }

    /// Try to get as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            SweepValue::String(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for SweepValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepValue::Float(v) => write!(f, "{}", v),
            SweepValue::Int(v) => write!(f, "{}", v),
            SweepValue::String(s) => write!(f, "{}", s),
        }
    }
}

/// A sweep dimension with all its values computed.
///
/// This is used during job expansion to iterate over sweep values
/// in the correct order.
#[derive(Debug, Clone)]
pub struct SweepDimension {
    /// Parameter name (e.g., "eps_bg", "atom0.radius")
    pub name: String,
    
    /// Order index (0 = outermost loop, higher = more inner)
    pub order: usize,
    
    /// All values for this dimension
    pub values: Vec<SweepValue>,
}

impl SweepDimension {
    /// Create from a SweepSpec.
    pub fn from_spec(spec: &SweepSpec, order: usize) -> Result<Self, ConfigError> {
        let values = if let Some(ref discrete_values) = spec.values {
            // Convert TOML values to SweepValue
            discrete_values
                .iter()
                .map(|v| toml_to_sweep_value(v, &spec.parameter))
                .collect::<Result<Vec<_>, _>>()?
        } else if let Some(range) = spec.to_range_spec() {
            // Generate range values
            let is_resolution = spec.parameter == "resolution";
            range
                .values()
                .into_iter()
                .map(|v| {
                    if is_resolution {
                        SweepValue::Int(v.round() as i64)
                    } else {
                        SweepValue::Float(v)
                    }
                })
                .collect()
        } else {
            return Err(ConfigError::InvalidSweep(format!(
                "sweep '{}' has no valid range or values",
                spec.parameter
            )));
        };

        Ok(Self {
            name: spec.parameter.clone(),
            order,
            values,
        })
    }

    /// Number of values in this dimension.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if this dimension is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Convert a TOML value to a SweepValue.
fn toml_to_sweep_value(value: &toml::Value, param: &str) -> Result<SweepValue, ConfigError> {
    match value {
        toml::Value::Float(f) => Ok(SweepValue::Float(*f)),
        toml::Value::Integer(i) => {
            // For resolution, keep as int; for others, convert to float
            if param == "resolution" {
                Ok(SweepValue::Int(*i))
            } else {
                Ok(SweepValue::Float(*i as f64))
            }
        }
        toml::Value::String(s) => Ok(SweepValue::String(s.clone())),
        _ => Err(ConfigError::InvalidSweep(format!(
            "unsupported value type for parameter '{}': {:?}",
            param, value
        ))),
    }
}

/// Valid parameter paths for sweeps.
const VALID_GLOBAL_PARAMS: &[&str] = &[
    "eps_bg",
    "resolution",
    "polarization",
    "lattice_type",
];

/// Validate a parameter path.
///
/// Valid paths:
/// - Global: "eps_bg", "resolution", "polarization", "lattice_type"
/// - Atom: "atom0.radius", "atom1.pos_x", etc.
pub fn validate_parameter_path(path: &str) -> Result<(), ConfigError> {
    // Check global parameters
    if VALID_GLOBAL_PARAMS.contains(&path) {
        return Ok(());
    }

    // Check atom parameters (atomN.property)
    if path.starts_with("atom") {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        if parts.len() != 2 {
            return Err(ConfigError::InvalidSweep(format!(
                "invalid atom parameter path '{}': expected 'atomN.property'",
                path
            )));
        }

        let atom_part = parts[0];
        let property = parts[1];

        // Validate atom index
        let index_str = atom_part.strip_prefix("atom").unwrap();
        if index_str.parse::<usize>().is_err() {
            return Err(ConfigError::InvalidSweep(format!(
                "invalid atom index in '{}': expected 'atomN' where N is a number",
                path
            )));
        }

        // Validate property
        const VALID_ATOM_PROPS: &[&str] = &["radius", "pos_x", "pos_y", "eps_inside"];
        if !VALID_ATOM_PROPS.contains(&property) {
            return Err(ConfigError::InvalidSweep(format!(
                "invalid atom property '{}' in '{}': expected one of {:?}",
                property, path, VALID_ATOM_PROPS
            )));
        }

        return Ok(());
    }

    Err(ConfigError::InvalidSweep(format!(
        "unknown parameter '{}': expected one of {:?} or atom path like 'atom0.radius'",
        path, VALID_GLOBAL_PARAMS
    )))
}

/// Parse an atom parameter path into (atom_index, property).
pub fn parse_atom_path(path: &str) -> Option<(usize, &str)> {
    if !path.starts_with("atom") {
        return None;
    }

    let parts: Vec<&str> = path.splitn(2, '.').collect();
    if parts.len() != 2 {
        return None;
    }

    let index_str = parts[0].strip_prefix("atom")?;
    let index = index_str.parse().ok()?;
    Some((index, parts[1]))
}

// ============================================================================
// Defaults Configuration (New Format)
// ============================================================================

/// Default values for parameters when not being swept.
///
/// This replaces the scattered base values in the old format with a
/// centralized `[defaults]` section.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DefaultsConfig {
    /// Default polarization
    #[serde(default)]
    pub polarization: Option<Polarization>,

    /// Default resolution (overrides grid.nx/ny for sweeps)
    #[serde(default)]
    pub resolution: Option<usize>,

    /// Default geometry settings
    #[serde(default)]
    pub geometry: Option<BaseGeometry>,
}

// ============================================================================
// Parameter Ranges
// ============================================================================

/// Range specification for atom parameters.
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParameterRange {
    /// Range for background epsilon
    #[serde(default)]
    pub eps_bg: Option<RangeSpec>,

    /// Range for resolution (integer values)
    #[serde(default)]
    pub resolution: Option<RangeSpec>,

    /// List of polarizations to compute
    #[serde(default)]
    pub polarization: Option<Vec<Polarization>>,

    /// List of lattice types to compute
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IoMode {
    /// Traditional synchronous I/O (write each result immediately)
    #[default]
    Sync,
    /// Batched I/O with background writer
    Batch,
    /// Streaming mode for real-time consumers (Python, WASM)
    Stream,
}

/// Settings for batch mode I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSettings {
    /// Buffer size in bytes before triggering a flush (default: 10 MB)
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,

    /// Maximum time between flushes in seconds (optional)
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

    /// Output directory for full mode
    #[serde(default = "default_output_dir")]
    pub directory: PathBuf,

    /// Output filename for selective mode
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
    /// Note: This is ignored in WASM (always single-threaded)
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
/// # Configuration Formats
///
/// ## New Format (Recommended): Ordered `[[sweeps]]`
///
/// ```toml
/// [bulk]
/// verbose = true
///
/// [[sweeps]]
/// parameter = "atom0.radius"
/// min = 0.2
/// max = 0.4
/// step = 0.1
///
/// [[sweeps]]
/// parameter = "eps_bg"
/// min = 10.0
/// max = 12.0  
/// step = 1.0
///
/// [defaults.geometry]
/// eps_bg = 12.0
/// ```
///
/// Sweeps are processed as nested loops **in the order listed**.
/// First sweep = outermost loop, last sweep = innermost loop.
///
/// ## Legacy Format: `[ranges]` Section
///
/// ```toml
/// [ranges]
/// eps_bg = { min = 10.0, max = 12.0, step = 1.0 }
/// ```
///
/// Still supported but deprecated. Sweep order is not guaranteed.
///
/// # Solver Types
///
/// - **Maxwell** (default): Requires `geometry`, `k_path`/`path`, and `polarization`.
/// - **EA**: Requires `ea` section with input file paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkConfig {
    /// Bulk execution settings (presence marks this as a bulk request)
    pub bulk: BulkSection,

    /// Solver type selection (maxwell or ea)
    #[serde(default)]
    pub solver: SolverSection,

    /// EA (Envelope Approximation) solver configuration.
    #[serde(default)]
    pub ea: EAConfig,

    // ========================================================================
    // NEW FORMAT: Ordered sweeps with [[sweeps]] array
    // ========================================================================

    /// Ordered parameter sweeps (new format).
    ///
    /// Sweeps are processed as nested loops in array order:
    /// - First entry = outermost loop
    /// - Last entry = innermost loop
    ///
    /// Takes precedence over `ranges` if both are present.
    #[serde(default)]
    pub sweeps: Vec<SweepSpec>,

    /// Default values for non-swept parameters (new format).
    #[serde(default)]
    pub defaults: DefaultsConfig,

    // ========================================================================
    // LEGACY FORMAT: [ranges] and [geometry] sections
    // ========================================================================

    /// Base geometry (non-swept parameters).
    /// Required for Maxwell solver, ignored for EA solver.
    /// DEPRECATED: Use `defaults.geometry` instead.
    #[serde(default)]
    pub geometry: Option<BaseGeometry>,

    /// Parameter ranges to sweep.
    /// DEPRECATED: Use `[[sweeps]]` array instead.
    #[serde(default)]
    pub ranges: ParameterRange,

    /// Base polarization (used if not in ranges).
    /// DEPRECATED: Use `defaults.polarization` instead.
    #[serde(default = "default_polarization")]
    pub polarization: Polarization,

    // ========================================================================
    // Common configuration (used by both formats)
    // ========================================================================

    /// Computational grid
    pub grid: Grid2D,

    /// K-path specification.
    #[serde(default)]
    pub path: Option<PathSpec>,

    /// Explicit k-path (overrides path preset).
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,

    /// Eigensolver configuration
    #[serde(default)]
    pub eigensolver: EigensolverConfig,

    /// Dielectric options.
    #[serde(default)]
    pub dielectric: DielectricOptions,

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
        if !s.contains("[bulk]") {
            return Err(ConfigError::NotBulkConfig);
        }

        let config: BulkConfig = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    /// Check if this config uses the new ordered sweeps format.
    pub fn uses_ordered_sweeps(&self) -> bool {
        !self.sweeps.is_empty()
    }

    /// Check if this config uses the legacy ranges format.
    pub fn uses_legacy_ranges(&self) -> bool {
        self.sweeps.is_empty() && self.has_any_legacy_ranges()
    }

    /// Check if any legacy ranges are defined.
    fn has_any_legacy_ranges(&self) -> bool {
        self.ranges.eps_bg.is_some()
            || self.ranges.resolution.is_some()
            || self.ranges.polarization.is_some()
            || self.ranges.lattice_type.is_some()
            || self.ranges.atoms.iter().any(|a| {
                a.radius.is_some() || a.pos_x.is_some() || a.pos_y.is_some() || a.eps_inside.is_some()
            })
    }

    /// Get the effective geometry configuration.
    ///
    /// Prefers `defaults.geometry` over legacy `geometry`.
    pub fn effective_geometry(&self) -> Option<&BaseGeometry> {
        self.defaults.geometry.as_ref().or(self.geometry.as_ref())
    }

    /// Get the effective base polarization.
    ///
    /// Prefers `defaults.polarization` over legacy `polarization`.
    pub fn effective_polarization(&self) -> Polarization {
        self.defaults.polarization.unwrap_or(self.polarization)
    }

    /// Get the effective base resolution.
    ///
    /// Prefers `defaults.resolution` over `grid.nx`.
    pub fn effective_resolution(&self) -> usize {
        self.defaults.resolution.unwrap_or(self.grid.nx)
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
    pub fn geometry(&self) -> &BaseGeometry {
        self.effective_geometry()
            .expect("geometry required for Maxwell solver")
    }

    /// Convert ordered sweeps to SweepDimensions for expansion.
    pub fn build_sweep_dimensions(&self) -> Result<Vec<SweepDimension>, ConfigError> {
        self.sweeps
            .iter()
            .enumerate()
            .map(|(order, spec)| SweepDimension::from_spec(spec, order))
            .collect()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate sweep specifications if using new format
        if self.uses_ordered_sweeps() {
            self.validate_sweeps()?;
        }

        match self.solver.solver_type {
            SolverType::Maxwell => self.validate_maxwell(),
            SolverType::EA => self.validate_ea(),
        }
    }

    /// Validate ordered sweeps.
    fn validate_sweeps(&self) -> Result<(), ConfigError> {
        // Check for empty sweeps
        if self.sweeps.is_empty() {
            // This is fine - means no sweeps, single job
            return Ok(());
        }

        // Check for duplicate parameters
        let mut seen = HashSet::new();
        for sweep in &self.sweeps {
            if !seen.insert(&sweep.parameter) {
                return Err(ConfigError::InvalidSweep(format!(
                    "duplicate sweep parameter '{}'",
                    sweep.parameter
                )));
            }
            sweep.validate()?;
        }

        // Validate atom indices are within bounds
        let max_atom_index = self.sweeps
            .iter()
            .filter_map(|s| parse_atom_path(&s.parameter))
            .map(|(idx, _)| idx)
            .max();

        if let Some(max_idx) = max_atom_index {
            let geom = self.effective_geometry();
            let atom_count = geom.map(|g| g.atoms.len()).unwrap_or(0);
            if atom_count == 0 && max_idx > 0 {
                return Err(ConfigError::InvalidSweep(format!(
                    "sweep references atom{} but no atoms defined in geometry",
                    max_idx
                )));
            }
            // Allow referencing atoms up to max_idx even if not all defined
            // (they'll use defaults)
        }

        Ok(())
    }

    fn validate_maxwell(&self) -> Result<(), ConfigError> {
        if self.effective_geometry().is_none() {
            return Err(ConfigError::InvalidMaxwellConfig(
                "Maxwell solver requires geometry (use [geometry] or [defaults.geometry])".into(),
            ));
        }

        // Validate legacy ranges if using them
        if self.uses_legacy_ranges() {
            if let Some(range) = &self.ranges.eps_bg {
                if range.step <= 0.0 {
                    return Err(ConfigError::InvalidRange(
                        "eps_bg step must be positive".into(),
                    ));
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
        }

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

    fn validate_ea(&self) -> Result<(), ConfigError> {
        self.ea.validate()
    }

    /// Get total number of jobs.
    pub fn total_jobs(&self) -> usize {
        if self.uses_ordered_sweeps() {
            self.sweeps.iter().map(|s| s.count().max(1)).product()
        } else {
            self.ranges.total_configurations()
        }
    }

    /// Get a description of the sweep order for display.
    pub fn sweep_order_description(&self) -> Vec<String> {
        if self.uses_ordered_sweeps() {
            self.sweeps
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let count = s.count();
                    let loop_type = if i == 0 { "outer" } else if i == self.sweeps.len() - 1 { "inner" } else { "middle" };
                    format!("{}: {} ({} values, {} loop)", i + 1, s.parameter, count, loop_type)
                })
                .collect()
        } else {
            // Legacy format - describe in hardcoded order
            let mut desc = Vec::new();
            if self.ranges.eps_bg.is_some() {
                desc.push("eps_bg".to_string());
            }
            if self.ranges.resolution.is_some() {
                desc.push("resolution".to_string());
            }
            if self.ranges.polarization.is_some() {
                desc.push("polarization".to_string());
            }
            if self.ranges.lattice_type.is_some() {
                desc.push("lattice_type".to_string());
            }
            for (i, atom) in self.ranges.atoms.iter().enumerate() {
                if atom.radius.is_some() {
                    desc.push(format!("atom{}.radius", i));
                }
                if atom.pos_x.is_some() {
                    desc.push(format!("atom{}.pos_x", i));
                }
                if atom.pos_y.is_some() {
                    desc.push(format!("atom{}.pos_y", i));
                }
                if atom.eps_inside.is_some() {
                    desc.push(format!("atom{}.eps_inside", i));
                }
            }
            desc
        }
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

    #[error("Invalid sweep specification: {0}")]
    InvalidSweep(String),

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
    fn solver_type_display() {
        assert_eq!(format!("{}", SolverType::Maxwell), "Maxwell");
        assert_eq!(format!("{}", SolverType::EA), "EA (Envelope Approximation)");
    }
}
