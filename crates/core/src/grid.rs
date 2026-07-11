//! Uniform grid helpers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    #[serde(default = "default_length")]
    pub lx: f64,
    #[serde(default = "default_length")]
    pub ly: f64,
    /// If true, coordinates are centered (from -lx/2 to lx/2)
    #[serde(default)]
    pub centered: bool,
}

impl Grid2D {
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        Self { nx, ny, lx, ly, centered: false }
    }

    /// Create a centered grid (coordinates from -lx/2 to lx/2).
    pub fn new_centered(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        Self { nx, ny, lx, ly, centered: true }
    }

    #[inline]
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    #[inline]
    pub fn cartesian_coords(&self, ix: usize, iy: usize) -> [f64; 2] {
        let dx = self.lx / self.nx as f64;
        let dy = self.ly / self.ny as f64;

        if self.centered {
            // Centered: coordinates from -lx/2 to lx/2
            // Cell center at (ix + 0.5) * dx - lx/2
            let x = (ix as f64 + 0.5) * dx - self.lx / 2.0;
            let y = (iy as f64 + 0.5) * dy - self.ly / 2.0;
            [x, y]
        } else {
            // Original: coordinates from 0 to lx (node-based at ix/nx)
            let x = if self.nx > 0 {
                (ix as f64 / self.nx as f64) * self.lx
            } else {
                0.0
            };
            let y = if self.ny > 0 {
                (iy as f64 / self.ny as f64) * self.ly
            } else {
                0.0
            };
            [x, y]
        }
    }
}

fn default_length() -> f64 {
    1.0
}
