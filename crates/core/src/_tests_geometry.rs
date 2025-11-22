#![cfg(test)]

use super::geometry::{BasisAtom, Geometry2D};
use super::lattice::Lattice2D;

fn sample_geometry() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 12.0,
        atoms: vec![BasisAtom {
            pos: [0.25, 0.25],
            radius: 0.2,
            eps_inside: 1.0,
        }],
    }
}

#[test]
fn basis_radius_cartesian_scales_with_lattice_norm() {
    let lattice = Lattice2D::oblique([3.0, 4.0], [0.0, 2.0]);
    let atom = BasisAtom {
        pos: [0.0, 0.0],
        radius: 0.3,
        eps_inside: 2.0,
    };
    // |a1| = 5 so physical radius should be 0.3 * 5.
    assert!((atom.radius_cartesian(&lattice) - 1.5).abs() < 1e-12);
}

#[test]
fn relative_permittivity_wraps_fractional_coordinates_into_unit_cell() {
    let geom = sample_geometry();
    // 1.25 mod 1 -> 0.25, -0.75 mod 1 -> 0.25, placing the point exactly on the atom.
    let eps = geom.relative_permittivity_at_fractional([1.25, -0.75]);
    assert!((eps - 1.0).abs() < 1e-12);
}

#[test]
fn relative_permittivity_uses_shortest_periodic_image() {
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 10.0,
        atoms: vec![BasisAtom {
            pos: [0.02, 0.02],
            radius: 0.05,
            eps_inside: 2.0,
        }],
    };
    // Although 0.98 is far in the direct difference, signed wrap makes the delta -0.04 (inside the hole).
    let eps = geom.relative_permittivity_at_fractional([0.98, 0.02]);
    assert!((eps - 2.0).abs() < 1e-12);
}

#[test]
fn relative_permittivity_falls_back_to_background_outside_atoms() {
    let geom = sample_geometry();
    let eps = geom.relative_permittivity_at_fractional([0.8, 0.8]);
    assert!((eps - geom.eps_bg).abs() < 1e-12);
}

#[test]
fn cartesian_sampling_matches_fractional_sampling() {
    let lattice = Lattice2D::hexagonal(1.0);
    let frac = [0.2, 0.7];
    let geom = Geometry2D {
        lattice,
        eps_bg: 8.0,
        atoms: vec![BasisAtom {
            pos: frac,
            radius: 0.05,
            eps_inside: 1.5,
        }],
    };
    let cart = lattice.fractional_to_cartesian(frac);
    assert!((geom.relative_permittivity_at_cartesian(cart) - 1.5).abs() < 1e-12);
}

#[test]
fn air_holes_builder_preserves_inputs() {
    let lattice = Lattice2D::rectangular(2.0, 3.0);
    let atoms = vec![BasisAtom {
        pos: [0.0, 0.0],
        radius: 0.1,
        eps_inside: 2.0,
    }];
    let geom = Geometry2D::air_holes_in_dielectric(lattice, atoms.clone(), 14.0);
    assert_eq!(geom.lattice.a1, lattice.a1);
    assert_eq!(geom.eps_bg, 14.0);
    assert_eq!(geom.atoms.len(), atoms.len());
    assert_eq!(geom.atoms[0].eps_inside, 2.0);
}

#[test]
fn single_air_hole_builder_defaults_to_air_inside() {
    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D::single_air_hole(lattice, 0.3, 9.0);
    assert_eq!(geom.atoms.len(), 1);
    assert_eq!(geom.atoms[0].pos, [0.0, 0.0]);
    assert_eq!(geom.atoms[0].eps_inside, 1.0);
    assert_eq!(geom.eps_bg, 9.0);
}
