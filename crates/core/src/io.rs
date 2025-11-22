//! Config serialization helpers.

use serde::{Deserialize, Serialize};

use crate::{
    bandstructure::BandStructureJob,
    geometry::Geometry2D,
    grid::Grid2D,
    polarization::Polarization,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub geometry: Geometry2D,
    pub grid: Grid2D,
    pub polarization: Polarization,
    pub k_path: Vec<[f64; 2]>,
}

impl From<JobConfig> for BandStructureJob {
    fn from(value: JobConfig) -> Self {
        BandStructureJob {
            geom: value.geometry,
            grid: value.grid,
            pol: value.polarization,
            k_path: value.k_path,
        }
    }
}
