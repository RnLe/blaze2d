//! High-level band-structure job orchestration.

use crate::{
    dielectric::Dielectric2D,
    geometry::Geometry2D,
    grid::Grid2D,
    polarization::Polarization,
};

pub struct BandStructureJob {
    pub geom: Geometry2D,
    pub grid: Grid2D,
    pub pol: Polarization,
    pub k_path: Vec<[f64; 2]>,
}

pub struct BandStructureResult {
    pub k_path: Vec<[f64; 2]>,
    pub bands: Vec<Vec<f64>>,
}

pub fn run(job: &BandStructureJob) -> BandStructureResult {
    let _eps = Dielectric2D::from_geometry(&job.geom, job.grid);
    BandStructureResult {
        k_path: job.k_path.clone(),
        bands: Vec::new(),
    }
}
