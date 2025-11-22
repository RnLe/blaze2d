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

impl BasisAtom {
    pub fn radius_cartesian(&self, lattice: &Lattice2D) -> f64 {
        self.radius * lattice.characteristic_length()
    }
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
        Self {
            lattice,
            eps_bg,
            atoms,
        }
    }

    pub fn single_air_hole(lattice: Lattice2D, radius: f64, eps_bg: f64) -> Self {
        Self {
            lattice,
            eps_bg,
            atoms: vec![BasisAtom {
                pos: [0.0, 0.0],
                radius,
                eps_inside: 1.0,
            }],
        }
    }

    pub fn relative_permittivity_at_fractional(&self, frac: [f64; 2]) -> f64 {
        let wrapped = wrap_unit_vec(frac);
        for atom in &self.atoms {
            if self.point_inside_atom(wrapped, atom) {
                return atom.eps_inside;
            }
        }
        self.eps_bg
    }

    pub fn relative_permittivity_at_cartesian(&self, cart: [f64; 2]) -> f64 {
        let frac = self.lattice.cartesian_to_fractional(cart);
        self.relative_permittivity_at_fractional(frac)
    }

    fn point_inside_atom(&self, frac_point: [f64; 2], atom: &BasisAtom) -> bool {
        let delta_frac = [
            wrap_signed(frac_point[0] - atom.pos[0]),
            wrap_signed(frac_point[1] - atom.pos[1]),
        ];
        let delta_cart = self.lattice.fractional_to_cartesian(delta_frac);
        let dist = (delta_cart[0] * delta_cart[0] + delta_cart[1] * delta_cart[1]).sqrt();
        dist <= atom.radius_cartesian(&self.lattice)
    }
}

fn wrap_unit(value: f64) -> f64 {
    value - value.floor()
}

fn wrap_unit_vec(mut frac: [f64; 2]) -> [f64; 2] {
    frac[0] = wrap_unit(frac[0]);
    frac[1] = wrap_unit(frac[1]);
    frac
}

fn wrap_signed(mut delta: f64) -> f64 {
    while delta >= 0.5 {
        delta -= 1.0;
    }
    while delta < -0.5 {
        delta += 1.0;
    }
    delta
}
