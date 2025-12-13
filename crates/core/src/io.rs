//! Config serialization helpers.

use serde::{Deserialize, Serialize};

use crate::{
    bandstructure::BandStructureJob,
    eigensolver::EigenOptions,
    geometry::Geometry2D,
    grid::Grid2D,
    metrics::MetricsConfig,
    polarization::Polarization,
    symmetry::{self, PathType},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathPreset {
    Square,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSpec {
    pub preset: PathPreset,
    #[serde(default = "default_segments_per_leg")]
    pub segments_per_leg: usize,
}

fn default_segments_per_leg() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub geometry: Geometry2D,
    pub grid: Grid2D,
    pub polarization: Polarization,
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,
    #[serde(default)]
    pub path: Option<PathSpec>,
    #[serde(default)]
    pub eigensolver: EigenOptions,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

impl From<JobConfig> for BandStructureJob {
    fn from(value: JobConfig) -> Self {
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
        }
    }
}
