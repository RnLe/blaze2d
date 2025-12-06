#![cfg(test)]

use super::dielectric::Dielectric2D;
use super::geometry::{BasisAtom, Geometry2D};
use super::grid::Grid2D;
use super::lattice::Lattice2D;

fn sample_geom() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 12.0,
        atoms: vec![
            BasisAtom {
                pos: [0.0, 0.0],
                radius: 0.25,
                eps_inside: 1.0,
            },
            BasisAtom {
                pos: [0.5, 0.5],
                radius: 0.25,
                eps_inside: 2.0,
            },
        ],
    }
}

#[test]
fn from_geometry_populates_eps_and_inv_eps_in_row_major_order() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&sample_geom(), grid);
    // Sampling points: (0,0) -> atom 1, (0.5,0) -> background, (0,0.5) -> background, (0.5,0.5) -> atom 2
    assert_eq!(dielectric.eps(), &[1.0, 12.0, 12.0, 2.0]);
    let inv_expected: Vec<f64> = dielectric.eps().iter().map(|v| 1.0 / v).collect();
    assert_eq!(dielectric.inv_eps(), inv_expected.as_slice());
}

#[test]
fn dielectric_retains_grid_metadata() {
    let grid = Grid2D::new(3, 4, 2.0, 1.5);
    let dielectric = Dielectric2D::from_geometry(&sample_geom(), grid);
    assert_eq!(dielectric.grid.nx, 3);
    assert_eq!(dielectric.grid.ny, 4);
    assert!((dielectric.grid.lx - 2.0).abs() < f64::EPSILON);
    assert!((dielectric.grid.ly - 1.5).abs() < f64::EPSILON);
}

#[test]
fn eps_and_inv_eps_slices_expose_internal_storage() {
    let grid = Grid2D::new(1, 2, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&sample_geom(), grid);
    assert_eq!(dielectric.eps().len(), grid.len());
    assert_eq!(dielectric.inv_eps().len(), grid.len());
}

#[test]
#[should_panic(expected = "grid dimensions must be non-zero")]
fn from_geometry_panics_with_zero_sized_grid() {
    let grid = Grid2D::new(0, 2, 1.0, 1.0);
    let _ = Dielectric2D::from_geometry(&sample_geom(), grid);
}

#[test]
#[should_panic(expected = "permittivity must be positive")]
fn from_geometry_panics_on_non_positive_permittivity() {
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 10.0,
        atoms: vec![BasisAtom {
            pos: [0.0, 0.0],
            radius: 0.5,
            eps_inside: 0.0,
        }],
    };
    let grid = Grid2D::new(1, 1, 1.0, 1.0);
    let _ = Dielectric2D::from_geometry(&geom, grid);
}
