//! Tests for k-path generation utilities in `symmetry`.
//!
//! The symmetry projector tests that used to live here were retired alongside
//! the projector code itself (see `archive/symmetry_projectors/`).

use crate::{
    lattice::Lattice2D,
    symmetry::{PathType, standard_path},
};

fn assert_point_close(actual: [f64; 2], expected: [f64; 2]) {
    let tol = 1e-9;
    assert!(
        (actual[0] - expected[0]).abs() < tol && (actual[1] - expected[1]).abs() < tol,
        "expected {:?}, got {:?}",
        expected,
        actual
    );
}

#[test]
fn square_path_endpoints() {
    let lattice = Lattice2D::square(1.0);
    let path = standard_path(&lattice, PathType::Square, 3);

    assert_point_close(path[0], [0.0, 0.0]); // Γ
    assert_point_close(path[3], [0.5, 0.0]); // X
    assert_point_close(path[6], [0.5, 0.5]); // M
    assert_point_close(*path.last().unwrap(), [0.0, 0.0]); // Γ
}

#[test]
fn square_path_segment_count() {
    let lattice = Lattice2D::square(1.0);
    let path = standard_path(&lattice, PathType::Square, 5);
    assert_eq!(path.len(), 16, "5 segments per leg should give 16 points");
}

#[test]
fn hexagonal_path_endpoints() {
    let lattice = Lattice2D::hexagonal(1.0);
    let path = standard_path(&lattice, PathType::Hexagonal, 3);

    assert_point_close(path[0], [0.0, 0.0]); // Γ
    assert_point_close(path[3], [0.5, 0.0]); // M
    assert_point_close(path[6], [1.0 / 3.0, 1.0 / 3.0]); // K
    assert_point_close(*path.last().unwrap(), [0.0, 0.0]); // Γ
}

#[test]
fn custom_path_densified() {
    let lattice = Lattice2D::square(1.0);
    let custom = vec![[0.1, 0.2], [0.3, 0.4], [0.9, 0.1]];
    let path = standard_path(&lattice, PathType::Custom(custom.clone()), 10);

    assert_eq!(path.len(), 21, "custom path should be densified");
    assert_point_close(path[0], [0.1, 0.2]);
    assert_point_close(path[10], [0.3, 0.4]);
    assert_point_close(path[20], [0.9, 0.1]);
}

#[test]
fn minimum_segments_clamped() {
    let lattice = Lattice2D::square(1.0);
    let path = standard_path(&lattice, PathType::Square, 0);
    assert!(path.len() >= 4, "should have at least the corner points");
}

#[test]
fn single_segment_path() {
    let lattice = Lattice2D::square(1.0);
    let path = standard_path(&lattice, PathType::Square, 1);

    assert_eq!(path.len(), 4);
    assert_point_close(path[0], [0.0, 0.0]);
    assert_point_close(path[1], [0.5, 0.0]);
    assert_point_close(path[2], [0.5, 0.5]);
    assert_point_close(path[3], [0.0, 0.0]);
}
