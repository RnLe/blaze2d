#![cfg(test)]

use super::field::Field2D;
use super::grid::Grid2D;
use num_complex::Complex64;

#[test]
fn zeros_initializes_all_entries_to_zero() {
    let grid = Grid2D::new(2, 3, 1.0, 1.0);
    let field = Field2D::zeros(grid);
    assert_eq!(field.len(), grid.len());
    assert!(
        field
            .as_slice()
            .iter()
            .all(|value| *value == Complex64::new(0.0, 0.0))
    );
}

#[test]
#[should_panic(expected = "data length must match grid size")]
fn from_vec_rejects_mismatched_lengths() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let data = vec![Complex64::default(); grid.len() - 1];
    let _ = Field2D::from_vec(grid, data);
}

#[test]
fn field_from_vec_preserves_values() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let data = vec![Complex64::new(1.0, -1.0); grid.len()];
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
            *field.get_mut(ix, iy) = Complex64::new(ix as f64, iy as f64);
        }
    }

    assert_eq!(*field.get(0, 0), Complex64::new(0.0, 0.0));
    assert_eq!(*field.get(2, 1), Complex64::new(2.0, 1.0));
}

#[test]
fn field_fill_updates_all_entries() {
    let grid = Grid2D::new(3, 1, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);
    field.fill(Complex64::new(0.0, 2.0));
    assert!(
        field
            .as_slice()
            .iter()
            .all(|value| *value == Complex64::new(0.0, 2.0))
    );
}

#[test]
fn field_into_vec_returns_original_storage() {
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let data: Vec<_> = (0..grid.len())
        .map(|idx| Complex64::new(idx as f64, -(idx as f64)))
        .collect();
    let field = Field2D::from_vec(grid, data.clone());
    let recovered: Vec<Complex64> = field.into();
    assert_eq!(recovered, data);
}
