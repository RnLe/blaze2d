//! Real-space dielectric sampling utilities.

use crate::{geometry::Geometry2D, grid::Grid2D};

#[derive(Debug, Clone)]
pub struct Dielectric2D {
    pub eps_r: Vec<f64>,
    pub inv_eps_r: Vec<f64>,
    pub grid: Grid2D,
}

impl Dielectric2D {
    pub fn from_geometry(geom: &Geometry2D, grid: Grid2D) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid dimensions must be non-zero"
        );
        let mut eps_r = vec![geom.eps_bg; grid.len()];
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let frac = [ix as f64 / grid.nx as f64, iy as f64 / grid.ny as f64];
                let idx = grid.idx(ix, iy);
                eps_r[idx] = geom.relative_permittivity_at_fractional(frac);
            }
        }
        let inv_eps_r = eps_r
            .iter()
            .map(|&val| {
                assert!(val > 0.0, "permittivity must be positive");
                1.0 / val
            })
            .collect();
        Self {
            eps_r,
            inv_eps_r,
            grid,
        }
    }

    pub fn eps(&self) -> &[f64] {
        &self.eps_r
    }

    pub fn inv_eps(&self) -> &[f64] {
        &self.inv_eps_r
    }
}
