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
        let det = self.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "primitive vectors are linearly dependent"
        );
        let inv = 2.0 * std::f64::consts::PI / det;
        let b1 = [self.a2[1] * inv, -self.a2[0] * inv];
        let b2 = [-self.a1[1] * inv, self.a1[0] * inv];
        ReciprocalLattice2D { b1, b2 }
    }

    pub fn fractional_to_cartesian(&self, frac: [f64; 2]) -> [f64; 2] {
        [
            self.a1[0] * frac[0] + self.a2[0] * frac[1],
            self.a1[1] * frac[0] + self.a2[1] * frac[1],
        ]
    }

    pub fn cartesian_to_fractional(&self, cart: [f64; 2]) -> [f64; 2] {
        let det = self.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "primitive vectors are linearly dependent"
        );
        let inv_det = 1.0 / det;
        let inv = [
            [self.a2[1] * inv_det, -self.a2[0] * inv_det],
            [-self.a1[1] * inv_det, self.a1[0] * inv_det],
        ];
        [
            inv[0][0] * cart[0] + inv[0][1] * cart[1],
            inv[1][0] * cart[0] + inv[1][1] * cart[1],
        ]
    }

    pub fn characteristic_length(&self) -> f64 {
        (self.a1[0] * self.a1[0] + self.a1[1] * self.a1[1]).sqrt()
    }

    fn determinant(&self) -> f64 {
        self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReciprocalLattice2D {
    pub b1: [f64; 2],
    pub b2: [f64; 2],
}
