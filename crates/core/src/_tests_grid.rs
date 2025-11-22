#![cfg(test)]

use super::grid::Grid2D;

#[test]
fn grid_len_matches_cartesian_product() {
    let grid = Grid2D::new(5, 7, 2.0, 3.5);
    assert_eq!(grid.len(), 35);
    assert_eq!(grid.idx(grid.nx - 1, grid.ny - 1), grid.len() - 1);
}

#[test]
fn grid_indexing_matches_row_major_layout() {
    let grid = Grid2D::new(4, 3, 1.0, 1.0);
    assert_eq!(grid.idx(0, 0), 0);
    assert_eq!(grid.idx(1, 0), 1);
    assert_eq!(grid.idx(0, 1), 4);
    assert_eq!(grid.idx(3, 2), 11);
}

#[test]
fn grid_new_preserves_physical_lengths() {
    let grid = Grid2D::new(2, 2, 0.75, 1.25);
    assert!((grid.lx - 0.75).abs() < f64::EPSILON);
    assert!((grid.ly - 1.25).abs() < f64::EPSILON);
}

#[test]
fn serde_defaults_lengths_to_one() {
    let json = r#"{"nx": 2, "ny": 3}"#;
    let grid: Grid2D = serde_json::from_str(json).expect("grid should deserialize");
    assert_eq!(grid.nx, 2);
    assert_eq!(grid.ny, 3);
    assert_eq!(grid.lx, 1.0);
    assert_eq!(grid.ly, 1.0);
}

#[test]
fn serde_roundtrip_preserves_lengths() {
    let original = Grid2D::new(3, 4, 0.8, 2.4);
    let serialized = serde_json::to_string(&original).expect("grid serializes");
    let roundtrip: Grid2D = serde_json::from_str(&serialized).expect("grid deserializes");
    assert_eq!(roundtrip.nx, 3);
    assert_eq!(roundtrip.ny, 4);
    assert!((roundtrip.lx - 0.8).abs() < f64::EPSILON);
    assert!((roundtrip.ly - 2.4).abs() < f64::EPSILON);
}
