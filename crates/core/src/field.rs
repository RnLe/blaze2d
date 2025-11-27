//! Contiguous complex-valued field storage on a uniform 2D grid.

use num_complex::Complex64;

use crate::grid::Grid2D;

#[derive(Debug, Clone)]
pub struct Field2D {
    grid: Grid2D,
    data: Vec<Complex64>,
}

impl Field2D {
    pub fn zeros(grid: Grid2D) -> Self {
        Self {
            data: vec![Complex64::default(); grid.len()],
            grid,
        }
    }

    pub fn from_vec(grid: Grid2D, data: Vec<Complex64>) -> Self {
        assert_eq!(data.len(), grid.len(), "data length must match grid size");
        Self { grid, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        self.grid.idx(ix, iy)
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [Complex64] {
        &mut self.data
    }

    pub fn get(&self, ix: usize, iy: usize) -> &Complex64 {
        let idx = self.idx(ix, iy);
        &self.data[idx]
    }

    pub fn get_mut(&mut self, ix: usize, iy: usize) -> &mut Complex64 {
        let idx = self.idx(ix, iy);
        &mut self.data[idx]
    }

    pub fn fill(&mut self, value: Complex64) {
        self.data.fill(value);
    }
}

impl From<Field2D> for Vec<Complex64> {
    fn from(field: Field2D) -> Self {
        field.data
    }
}
