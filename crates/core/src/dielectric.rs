//! Real-space dielectric sampling utilities.

use crate::{geometry::Geometry2D, grid::Grid2D};

#[derive(Debug, Clone)]
pub struct Dielectric2D {
    pub eps_r: Vec<f64>,
    pub grid: Grid2D,
}

impl Dielectric2D {
    pub fn from_geometry(geom: &Geometry2D, grid: Grid2D) -> Self {
        // Placeholder implementation; actual sampling logic will arrive in Phase 2.
        let eps_r = vec![geom.eps_bg; grid.len()];
        Self { eps_r, grid }
    }
}
