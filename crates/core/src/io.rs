//! Configuration file parsing and serialization.
//!
//! This module provides the types needed to load band-structure job
//! configurations from TOML files. The main type is `JobConfig` which
//! can be parsed from a TOML file and converted to a `BandStructureJob`.
//!
//! # File Format
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
//! preset = "square"
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathPreset {
    /// Square lattice path: Γ → X → M → Γ
    Square,
    /// Hexagonal lattice path: Γ → M → K → Γ
    Hexagonal,
}

impl From<PathPreset> for PathType {
    fn from(value: PathPreset) -> Self {
        match value {
            PathPreset::Square => PathType::Square,
            PathPreset::Hexagonal => PathType::Hexagonal,
        }
    }
}

/// Specification for a k-path using a preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSpec {
    /// Which preset path to use.
    pub preset: PathPreset,
    /// Number of k-points per segment between high-symmetry points.
    #[serde(default = "default_segments_per_leg")]
    pub segments_per_leg: usize,
}

fn default_segments_per_leg() -> usize {
    8
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
                k_path = symmetry::standard_path(
                    &value.geometry.lattice,
                    spec.preset.clone().into(),
                    spec.segments_per_leg,
                );
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
