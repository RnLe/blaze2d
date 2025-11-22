//! Geometry descriptions (basis atoms, inclusions, etc.).

use serde::{Deserialize, Serialize};

use crate::lattice::Lattice2D;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisAtom {
    pub pos: [f64; 2],
    pub radius: f64,
    #[serde(default = "default_eps_inside")]
    pub eps_inside: f64,
}

fn default_eps_inside() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry2D {
    pub lattice: Lattice2D,
    #[serde(default = "default_eps_bg")]
    pub eps_bg: f64,
    #[serde(default)]
    pub atoms: Vec<BasisAtom>,
}

fn default_eps_bg() -> f64 {
    12.0
}

impl Geometry2D {
    pub fn air_holes_in_dielectric(lattice: Lattice2D, atoms: Vec<BasisAtom>, eps_bg: f64) -> Self {
        Self { lattice, eps_bg, atoms }
    }
}
