#![cfg(test)]

use num_complex::Complex64;

use crate::{
    field::Field2D,
    grid::Grid2D,
    lattice::Lattice2D,
    symmetry::{
        AutoSymmetry, Parity, PathType, ReflectionAxis, ReflectionConstraint, SymmetryOptions,
        SymmetryProjector, standard_path,
    },
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

fn projector_from_reflection(axis: ReflectionAxis, parity: Parity) -> SymmetryProjector {
    let opts = SymmetryOptions {
        reflections: vec![ReflectionConstraint { axis, parity }],
        auto: None,
        ..SymmetryOptions::default()
    };
    SymmetryProjector::from_options(&opts).expect("projector should exist")
}

fn filled_field(grid: Grid2D) -> Field2D {
    let mut field = Field2D::zeros(grid);
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            field.as_mut_slice()[idx] = Complex64::new((ix + iy * grid.nx) as f64, 0.0);
        }
    }
    field
}

#[test]
fn even_reflection_matches_mirror_values() {
    let grid = Grid2D::new(4, 2, 1.0, 1.0);
    let mut field = filled_field(grid);
    let projector = projector_from_reflection(ReflectionAxis::X, Parity::Even);
    projector.apply(&mut field);

    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let mirror_ix = (grid.nx - ix) % grid.nx;
            let mirror_idx = grid.idx(mirror_ix, iy);
            assert!(
                (field.as_slice()[idx] - field.as_slice()[mirror_idx]).norm() < 1e-12,
                "even projector should enforce symmetry at ({ix},{iy})"
            );
        }
    }
}

#[test]
fn odd_reflection_zeroes_mirror_plane() {
    let grid = Grid2D::new(5, 1, 1.0, 1.0);
    let mut field = filled_field(grid);
    let projector = projector_from_reflection(ReflectionAxis::X, Parity::Odd);
    projector.apply(&mut field);

    for ix in 0..grid.nx {
        let idx = grid.idx(ix, 0);
        let mirror_idx = grid.idx((grid.nx - ix) % grid.nx, 0);
        let sum = field.as_slice()[idx] + field.as_slice()[mirror_idx];
        assert!(
            sum.norm() < 1e-12,
            "odd projector should produce antisymmetric pair at column {ix}"
        );
    }
}

#[test]
fn auto_symmetry_populates_reflections_for_rectangular() {
    let lattice = Lattice2D::rectangular(1.0, 2.0);
    let mut opts = SymmetryOptions {
        reflections: Vec::new(),
        auto: Some(AutoSymmetry {
            parity: Parity::Even,
            ..Default::default()
        }),
        ..SymmetryOptions::default()
    };
    opts.resolve_with_lattice(&lattice);
    let selection = opts.selection_for_bloch([0.0, 0.0]);
    assert_eq!(selection.applied_count(), 2);
}

#[test]
fn auto_symmetry_skips_oblique() {
    let lattice = Lattice2D::oblique([1.0, 0.2], [0.7, 0.5]);
    let mut opts = SymmetryOptions {
        reflections: Vec::new(),
        auto: Some(AutoSymmetry {
            parity: Parity::Odd,
            ..Default::default()
        }),
        ..SymmetryOptions::default()
    };
    opts.resolve_with_lattice(&lattice);
    let selection = opts.selection_for_bloch([0.1, 0.2]);
    assert_eq!(selection.applied_count(), 0);
}

#[test]
fn symmetry_defaults_disable_auto_inference() {
    let opts = SymmetryOptions::default();
    assert!(
        opts.auto.is_none(),
        "auto inference should be opt-in by default"
    );
    assert!(
        opts.reflections.is_empty(),
        "defaults should not pre-populate reflections before resolution"
    );
}

#[test]
fn default_auto_populates_supported_lattice_reflections() {
    let lattice = Lattice2D::square(1.0);
    let mut opts = SymmetryOptions {
        auto: Some(AutoSymmetry::default()),
        ..SymmetryOptions::default()
    };
    opts.resolve_with_lattice(&lattice);
    let selection = opts.selection_for_bloch([0.0, 0.0]);
    assert_eq!(
        selection.applied_count(),
        2,
        "square lattice should add two reflection axes by default"
    );
}

#[test]
fn auto_symmetry_skips_off_axis_points() {
    let lattice = Lattice2D::square(1.0);
    let mut opts = SymmetryOptions {
        auto: Some(AutoSymmetry::default()),
        ..SymmetryOptions::default()
    };
    opts.resolve_with_lattice(&lattice);
    let selection = opts.selection_for_bloch([0.25, 0.3]);
    assert_eq!(
        selection.applied_count(),
        0,
        "off-axis k-points should not enforce auto reflections"
    );
    assert_eq!(selection.skipped_count(), 2);
}

#[test]
fn manual_reflections_apply_even_off_axis() {
    let lattice = Lattice2D::square(1.0);
    let mut opts = SymmetryOptions::default();
    opts.reflections = vec![ReflectionConstraint {
        axis: ReflectionAxis::X,
        parity: Parity::Even,
    }];
    opts.resolve_with_lattice(&lattice);
    let selection = opts.selection_for_bloch([0.3, 0.2]);
    assert_eq!(selection.applied_count(), 1);
    assert_eq!(selection.skipped_count(), 0);
}
