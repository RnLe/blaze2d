#![cfg(test)]

use super::field::{Field2D, FieldReal, FieldScalar};
use super::grid::Grid2D;

#[test]
fn zeros_initializes_all_entries_to_zero() {
    let grid = Grid2D::new(2, 3, 1.0, 1.0);
    let field = Field2D::zeros(grid);
    assert_eq!(field.len(), grid.len());
    assert!(
        field
            .as_slice()
            .iter()
            .all(|value| *value == FieldScalar::new(0.0, 0.0))
    );
}

#[test]
#[should_panic(expected = "data length must match grid size")]
fn from_vec_rejects_mismatched_lengths() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let data = vec![FieldScalar::default(); grid.len() - 1];
    let _ = Field2D::from_vec(grid, data);
}

#[test]
fn field_from_vec_preserves_values() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let data = vec![FieldScalar::new(1.0, -1.0); grid.len()];
    let field = Field2D::from_vec(grid, data.clone());
    assert_eq!(field.len(), data.len());
    assert_eq!(field.as_slice(), data.as_slice());
}

#[test]
fn idx_matches_underlying_grid_row_major_convention() {
    let grid = Grid2D::new(3, 2, 1.0, 1.0);
    let field = Field2D::zeros(grid);
    let grid = field.grid();
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            assert_eq!(field.idx(ix, iy), grid.idx(ix, iy));
        }
    }
}

#[test]
fn get_and_get_mut_operate_on_correct_cell() {
    let grid = Grid2D::new(3, 2, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);
    let grid = field.grid();
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            *field.get_mut(ix, iy) = FieldScalar::new(ix as FieldReal, iy as FieldReal);
        }
    }

    assert_eq!(*field.get(0, 0), FieldScalar::new(0.0, 0.0));
    assert_eq!(*field.get(2, 1), FieldScalar::new(2.0, 1.0));
}

#[test]
fn field_fill_updates_all_entries() {
    let grid = Grid2D::new(3, 1, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);
    field.fill(FieldScalar::new(0.0, 2.0));
    assert!(
        field
            .as_slice()
            .iter()
            .all(|value| *value == FieldScalar::new(0.0, 2.0))
    );
}

#[test]
fn field_into_vec_returns_original_storage() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let data: Vec<_> = (0..grid.len())
        .map(|idx| FieldScalar::new(idx as FieldReal, -(idx as FieldReal)))
        .collect();
    let field = Field2D::from_vec(grid, data.clone());
    let recovered: Vec<FieldScalar> = field.into();
    assert_eq!(recovered, data);
}
