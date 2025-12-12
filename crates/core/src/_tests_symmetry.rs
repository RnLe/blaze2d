#![cfg(test)]

use crate::{
    lattice::Lattice2D,
    symmetry::{PathType, standard_path},
};

fn assert_point_close(actual: [f64; 2], expected: [f64; 2]) {
    let tol = 1e-9;
    assert!(
        (actual[0] - expected[0]).abs() < tol && (actual[1] - expected[1]).abs() < tol,
        "expected {:?} to be close to {:?}",
        actual,
        expected
    );
}

#[test]
fn square_path_respects_segments_per_leg() {
    let lattice = Lattice2D::square(1.0);
    let path = standard_path(&lattice, PathType::Square, 3);

    assert_eq!(path.len(), 10, "3 segments per leg should expand the path");
    assert_point_close(path[0], [0.0, 0.0]);
    assert_point_close(path[3], [0.5, 0.0]);
    assert_point_close(*path.last().unwrap(), [0.0, 0.0]);
}

#[test]
fn hexagonal_path_clamps_to_minimum_segments() {
    let lattice = Lattice2D::hexagonal(1.0);
    let path = standard_path(&lattice, PathType::Hexagonal, 0);

    assert_eq!(
        path.len(),
        4,
        "segments should clamp to at least one per leg"
    );
    assert_point_close(path[1], [0.5, 0.0]);
    assert_point_close(path[2], [1.0 / 3.0, 1.0 / 3.0]);
}

#[test]
fn custom_path_passthrough_skips_densification() {
    let lattice = Lattice2D::square(2.0);
    let custom = vec![[0.1, 0.2], [0.3, 0.4], [0.9, 0.1]];
    let result = standard_path(&lattice, PathType::Custom(custom.clone()), 5);

    assert_eq!(result, custom, "custom points should be returned unchanged");
}
