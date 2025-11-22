//! Lattice primitives for 2D photonic crystals.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Lattice2D {
    pub a1: [f64; 2],
    pub a2: [f64; 2],
}

impl Lattice2D {
    pub fn square(a: f64) -> Self {
        Self {
            a1: [a, 0.0],
            a2: [0.0, a],
        }
    }

    pub fn rectangular(a: f64, b: f64) -> Self {
        Self {
            a1: [a, 0.0],
            a2: [0.0, b],
        }
    }

    pub fn hexagonal(a: f64) -> Self {
        let half = 0.5 * a;
        let h = (3.0f64).sqrt() * 0.5 * a;
        Self {
            a1: [a, 0.0],
            a2: [half, h],
        }
    }

    pub fn oblique(a1: [f64; 2], a2: [f64; 2]) -> Self {
        Self { a1, a2 }
    }

    pub fn reciprocal(&self) -> ReciprocalLattice2D {
        let det = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0];
        assert!(det.abs() > f64::EPSILON, "primitive vectors are linearly dependent");
        let inv = 2.0 * std::f64::consts::PI / det;
        let b1 = [ self.a2[1] * inv, -self.a2[0] * inv ];
        let b2 = [ -self.a1[1] * inv, self.a1[0] * inv ];
        ReciprocalLattice2D { b1, b2 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReciprocalLattice2D {
    pub b1: [f64; 2],
    pub b2: [f64; 2],
}
