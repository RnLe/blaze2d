#![cfg(test)]

use super::dielectric::{Dielectric2D, DielectricOptions};
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

fn unsmoothed_options() -> DielectricOptions {
    let mut opts = DielectricOptions::default();
    opts.smoothing.mesh_size = 1;
    opts
}

#[test]
fn from_geometry_populates_eps_and_inv_eps_in_row_major_order() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let geom = sample_geom();
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &unsmoothed_options());
    let mut expected = Vec::with_capacity(grid.len());
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let frac = [cell_center(ix, grid.nx), cell_center(iy, grid.ny)];
            expected.push(geom.relative_permittivity_at_fractional(frac));
        }
    }
    assert_eq!(dielectric.eps(), expected.as_slice());
    let inv_expected: Vec<f64> = dielectric.eps().iter().map(|v| 1.0 / v).collect();
    assert_eq!(dielectric.inv_eps(), inv_expected.as_slice());
}

#[test]
fn dielectric_retains_grid_metadata() {
    let grid = Grid2D::new(3, 4, 2.0, 1.5);
    let dielectric = Dielectric2D::from_geometry(&sample_geom(), grid, &unsmoothed_options());
    assert_eq!(dielectric.grid.nx, 3);
    assert_eq!(dielectric.grid.ny, 4);
    assert!((dielectric.grid.lx - 2.0).abs() < f64::EPSILON);
    assert!((dielectric.grid.ly - 1.5).abs() < f64::EPSILON);
}

#[test]
fn eps_and_inv_eps_slices_expose_internal_storage() {
    let grid = Grid2D::new(1, 2, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&sample_geom(), grid, &unsmoothed_options());
    assert_eq!(dielectric.eps().len(), grid.len());
    assert_eq!(dielectric.inv_eps().len(), grid.len());
}

#[test]
fn hexagonal_sampling_matches_cartesian_projection() {
    let lattice = Lattice2D::hexagonal(1.0);
    let geom = Geometry2D::air_holes_in_dielectric(
        lattice,
        vec![BasisAtom {
            pos: [0.0, 0.0],
            radius: 0.3,
            eps_inside: 1.0,
        }],
        13.0,
    );
    let grid = Grid2D::new(24, 24, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &unsmoothed_options());
    let (ix, iy) = (0, 0);
    let idx = grid.idx(ix, iy);
    let frac = [cell_center(ix, grid.nx), cell_center(iy, grid.ny)];
    let expected = geom.relative_permittivity_at_fractional(frac);
    assert!((dielectric.eps()[idx] - expected).abs() < 1e-12);
    assert!((expected - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "grid dimensions must be non-zero")]
fn from_geometry_panics_with_zero_sized_grid() {
    let grid = Grid2D::new(0, 2, 1.0, 1.0);
    let _ = Dielectric2D::from_geometry(&sample_geom(), grid, &unsmoothed_options());
}

#[test]
#[should_panic(expected = "permittivity must be positive")]
fn from_geometry_panics_on_non_positive_permittivity() {
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 10.0,
        atoms: vec![BasisAtom {
            pos: [0.5, 0.5],
            radius: 0.5,
            eps_inside: 0.0,
        }],
    };
    let grid = Grid2D::new(1, 1, 1.0, 1.0);
    let _ = Dielectric2D::from_geometry(&geom, grid, &unsmoothed_options());
}

#[test]
fn smoothing_mesh_size_builds_anisotropic_tensor_and_raw_dump() {
    let geom = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 12.0,
        atoms: vec![BasisAtom {
            pos: [0.2, 0.5],
            radius: 0.25,
            eps_inside: 1.0,
        }],
    };
    let grid = Grid2D::new(1, 1, 1.0, 1.0);
    let mut opts = DielectricOptions::default();
    opts.smoothing.mesh_size = 8;
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &opts);
    let tensors = dielectric
        .inv_eps_tensors()
        .expect("smoothing should allocate tensors");
    assert!(dielectric.unsmoothed_eps().is_some());
    let tensor = tensors[0];
    assert!(
        (tensor[0] - tensor[3]).abs() > 1e-3,
        "anisotropic smoothing should yield distinct normal/tangential components, tensor={tensor:?}"
    );
}

fn cell_center(index: usize, count: usize) -> f64 {
    (index as f64 + 0.5) / count as f64
}
