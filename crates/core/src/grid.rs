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
}

impl Grid2D {
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        Self { nx, ny, lx, ly }
    }

    #[inline]
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    pub fn len(&self) -> usize {
        self.nx * self.ny
    }
}

fn default_length() -> f64 {
    1.0
}
