//! Configuration types for bulk parameter sweeps.
//!
//! This module defines the TOML structure for bulk jobs, including parameter ranges
//! and output configuration.

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

/// Output configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output mode: "full" or "selective"
    #[serde(default)]
    pub mode: OutputMode,

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

    /// Write output in batches (number of jobs before flushing)
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
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
            directory: default_output_dir(),
            filename: default_output_file(),
            prefix: default_prefix(),
            selective: SelectiveSpec::default(),
            batch_size: default_batch_size(),
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkConfig {
    /// Bulk execution settings (presence marks this as a bulk request)
    pub bulk: BulkSection,

    /// Base geometry (non-swept parameters)
    pub geometry: BaseGeometry,

    /// Computational grid
    pub grid: Grid2D,

    /// Base polarization (used if not in ranges)
    #[serde(default = "default_polarization")]
    pub polarization: Polarization,

    /// K-path specification
    #[serde(default)]
    pub path: Option<PathSpec>,

    /// Explicit k-path (overrides path preset)
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,

    /// Eigensolver configuration
    #[serde(default)]
    pub eigensolver: EigensolverConfig,

    /// Dielectric options
    #[serde(default)]
    pub dielectric: DielectricOptions,

    /// Parameter ranges to sweep
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

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check that ranged parameters are not also in base config conflicts
        // (The actual validation is more nuanced - we allow base values as defaults)

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

    /// Get the effective number of threads.
    pub fn effective_threads(&self) -> usize {
        self.bulk.threads.unwrap_or_else(num_cpus::get)
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
}
