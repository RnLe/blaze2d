//! Configuration file parsing and serialization.
//!
//! This module provides the types needed to load band-structure job
//! configurations from TOML files. The main type is `JobConfig` which
//! can be parsed from a TOML file and converted to a `BandStructureJob`.
//!
//! # File Format
//!
//! ## Legacy format (still supported):
//!
//! ```toml
//! [geometry.lattice]
//! a1 = [1.0, 0.0]
//! a2 = [0.0, 1.0]
//!
//! [[geometry.atoms]]
//! pos = [0.0, 0.0]
//! radius = 0.3
//! eps_inside = 1.0
//! ```
//!
//! ## New typed lattice format (recommended):
//!
//! ```toml
//! # Square lattice (simplest)
//! [geometry.lattice]
//! type = "square"
//! a = 1.0  # optional, default = 1.0
//!
//! # Rectangular lattice
//! [geometry.lattice]
//! type = "rectangular"
//! b = 1.5  # a = 1.0 is fixed
//!
//! # Triangular/hexagonal lattice
//! [geometry.lattice]
//! type = "triangular"
//! a = 1.0  # optional
//!
//! # Oblique lattice
//! [geometry.lattice]
//! type = "oblique"
//! b = 1.2
//! alpha = 1.0472  # 60° in radians
//!
//! # For any lattice type:
//! [[geometry.atoms]]
//! pos = [0.0, 0.0]
//! radius = 0.3
//! eps_inside = 1.0
//!
//! [grid]
//! nx = 32
//! ny = 32
//! lx = 1.0
//! ly = 1.0
//!
//! polarization = "TM"
//!
//! [path]
//! preset = "square"  # or "rectangular", "triangular", "hexagonal"
//! segments_per_leg = 12
//!
//! [eigensolver]
//! n_bands = 8
//! max_iter = 200
//! tol = 1e-6
//! ```

use serde::{Deserialize, Serialize};

use crate::{
    bandstructure::BandStructureJob,
    bravais::LatticeType,
    brillouin::{BrillouinPath, PathPreset as NewPathPreset, generate_path},
    dielectric::DielectricOptions,
    eigensolver::EigensolverConfig,
    geometry::Geometry2D,
    grid::Grid2D,
    polarization::Polarization,
    symmetry::{self, PathType},
};

// ============================================================================
// K-Path Presets
// ============================================================================

/// Preset k-path through the Brillouin zone.
///
/// Now supports rectangular lattice paths in addition to square and hexagonal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathPreset {
    /// Square lattice path: Γ → X → M → Γ
    Square,
    /// Rectangular lattice path: Γ → X → S → Y → Γ
    Rectangular,
    /// Triangular lattice path: Γ → M → K → Γ
    Triangular,
    /// Hexagonal lattice path (alias for triangular): Γ → M → K → Γ
    Hexagonal,
}

impl PathPreset {
    /// Get the Brillouin path type for this preset.
    pub fn to_brillouin_path(&self) -> BrillouinPath {
        match self {
            PathPreset::Square => BrillouinPath::Square,
            PathPreset::Rectangular => BrillouinPath::Rectangular,
            PathPreset::Triangular => BrillouinPath::Triangular,
            PathPreset::Hexagonal => BrillouinPath::Hexagonal,
        }
    }

    /// Get the recommended preset for a lattice type.
    pub fn for_lattice_type(lattice_type: LatticeType) -> Option<Self> {
        match lattice_type {
            LatticeType::Square => Some(PathPreset::Square),
            LatticeType::Rectangular => Some(PathPreset::Rectangular),
            LatticeType::Triangular => Some(PathPreset::Triangular),
            LatticeType::Oblique => None,
        }
    }
}

impl From<PathPreset> for PathType {
    fn from(value: PathPreset) -> Self {
        match value {
            PathPreset::Square => PathType::Square,
            PathPreset::Rectangular => {
                // For backward compatibility with symmetry module
                // We generate the rectangular path directly
                PathType::Custom(BrillouinPath::Rectangular.raw_k_points())
            }
            PathPreset::Triangular | PathPreset::Hexagonal => PathType::Hexagonal,
        }
    }
}

impl From<PathPreset> for NewPathPreset {
    fn from(value: PathPreset) -> Self {
        match value {
            PathPreset::Square => NewPathPreset::Square,
            PathPreset::Rectangular => NewPathPreset::Rectangular,
            PathPreset::Triangular => NewPathPreset::Triangular,
            PathPreset::Hexagonal => NewPathPreset::Hexagonal,
        }
    }
}

/// Specification for a k-path.
///
/// Supports two modes:
/// 1. **Preset path**: Use `preset` to select a standard high-symmetry path
/// 2. **Explicit path**: Use `k_path` to specify arbitrary k-points
///
/// When both are provided, `k_path` takes precedence.
///
/// # Examples
///
/// ```toml
/// # Using a preset
/// [path]
/// preset = "square"
/// segments_per_leg = 12
///
/// # Using an explicit path
/// [path]
/// k_path = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PathSpec {
    /// Which preset path to use (optional if k_path is provided).
    #[serde(default)]
    pub preset: Option<PathPreset>,
    /// Number of k-points per segment between high-symmetry points.
    #[serde(default = "default_segments_per_leg")]
    pub segments_per_leg: usize,
    /// Explicit k-path (overrides preset if provided).
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,
}

fn default_segments_per_leg() -> usize {
    8
}

impl PathSpec {
    /// Generate the k-path from this specification.
    ///
    /// If `k_path` is non-empty, returns it directly.
    /// Otherwise, generates a path from the preset with densification.
    pub fn generate(&self) -> Vec<[f64; 2]> {
        if !self.k_path.is_empty() {
            // Explicit k-path provided - use as-is
            self.k_path.clone()
        } else if let Some(preset) = &self.preset {
            // Generate from preset with densification
            generate_path(&preset.to_brillouin_path(), self.segments_per_leg)
        } else {
            // No path specified - return empty (caller should handle this)
            Vec::new()
        }
    }

    /// Check if this path spec is valid (has either k_path or preset).
    pub fn is_valid(&self) -> bool {
        !self.k_path.is_empty() || self.preset.is_some()
    }
}

// ============================================================================
// Job Configuration
// ============================================================================

/// Configuration for a band-structure job (loadable from TOML).
///
/// This struct is designed for parsing from TOML configuration files.
/// Use the `From<JobConfig>` implementation to convert to a `BandStructureJob`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    /// The photonic crystal geometry.
    pub geometry: Geometry2D,
    /// Computational grid.
    pub grid: Grid2D,
    /// Polarization mode.
    pub polarization: Polarization,
    /// Explicit k-path (overrides `path` if non-empty).
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,
    /// K-path specification using a preset.
    #[serde(default)]
    pub path: Option<PathSpec>,
    /// Eigensolver configuration.
    #[serde(default)]
    pub eigensolver: EigensolverConfig,
    /// Dielectric function options.
    #[serde(default)]
    pub dielectric: DielectricOptions,
}

impl From<JobConfig> for BandStructureJob {
    fn from(value: JobConfig) -> Self {
        // Build k-path from explicit list or preset
        let mut k_path = value.k_path;
        if k_path.is_empty() {
            if let Some(spec) = &value.path {
                // First check if spec has explicit k_path
                if !spec.k_path.is_empty() {
                    k_path = spec.k_path.clone();
                } else if let Some(preset) = &spec.preset {
                    // Use the new path generation for rectangular,
                    // fall back to symmetry module for others
                    match preset {
                        PathPreset::Rectangular => {
                            k_path = spec.generate();
                        }
                        _ => {
                            k_path = symmetry::standard_path(
                                &value.geometry.lattice,
                                preset.clone().into(),
                                spec.segments_per_leg,
                            );
                        }
                    }
                }
            }
        }
        assert!(
            !k_path.is_empty(),
            "JobConfig requires either an explicit k_path or a path preset"
        );

        BandStructureJob {
            geom: value.geometry,
            grid: value.grid,
            pol: value.polarization,
            k_path,
            eigensolver: value.eigensolver,
            dielectric: value.dielectric,
        }
    }
}
