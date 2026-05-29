//! Contiguous complex-valued field storage on a uniform 2D grid.
//!
//! # Mixed Precision Mode
//!
//! When the `mixed-precision` feature is enabled, fields are stored using
//! `Complex<f32>` instead of `Complex<f64>`. This halves memory bandwidth
//! requirements, which is the primary bottleneck for FFT-heavy eigensolvers.
//!
//! **Important**: Even in mixed precision mode, all accumulation operations
//! (dot products, Gram matrices) must be performed in f64 to maintain
//! numerical stability. See the `SpectralBackend` trait for details.

use num_complex::Complex64;

#[cfg(not(feature = "mixed-precision"))]
use num_complex::Complex64 as FieldScalarAlias;

#[cfg(feature = "mixed-precision")]
use num_complex::Complex32 as FieldScalarAlias;

use crate::grid::Grid2D;

// ============================================================================
// Type aliases for precision-dependent types
// ============================================================================

/// The complex scalar type used for field storage.
/// - f64 (Complex64) in standard mode for maximum precision
/// - f32 (Complex32) in mixed-precision mode for bandwidth efficiency
#[cfg(not(feature = "mixed-precision"))]
pub type FieldScalar = FieldScalarAlias;

#[cfg(feature = "mixed-precision")]
pub type FieldScalar = FieldScalarAlias;

/// The real component type of FieldScalar.
#[cfg(not(feature = "mixed-precision"))]
pub type FieldReal = f64;

#[cfg(feature = "mixed-precision")]
pub type FieldReal = f32;

// Always use f64 for accumulation (dot products, eigenvalues, etc.)
pub use num_complex::Complex64 as AccumScalar;

#[derive(Debug, Clone)]
pub struct Field2D {
    grid: Grid2D,
    data: Vec<FieldScalar>,
}

impl Field2D {
    pub fn zeros(grid: Grid2D) -> Self {
        Self {
            data: vec![FieldScalar::default(); grid.len()],
            grid,
        }
    }

    pub fn from_vec(grid: Grid2D, data: Vec<FieldScalar>) -> Self {
        assert_eq!(data.len(), grid.len(), "data length must match grid size");
        Self { grid, data }
    }

    /// Create a Field2D from f64 complex values, converting to storage precision.
    /// This is useful for initialization from high-precision sources.
    #[cfg(feature = "mixed-precision")]
    pub fn from_f64_vec(grid: Grid2D, data: Vec<Complex64>) -> Self {
        assert_eq!(data.len(), grid.len(), "data length must match grid size");
        let converted: Vec<FieldScalar> = data
            .into_iter()
            .map(|c| FieldScalar::new(c.re as f32, c.im as f32))
            .collect();
        Self {
            grid,
            data: converted,
        }
    }

    #[cfg(not(feature = "mixed-precision"))]
    pub fn from_f64_vec(grid: Grid2D, data: Vec<Complex64>) -> Self {
        Self::from_vec(grid, data)
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

    pub fn as_slice(&self) -> &[FieldScalar] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [FieldScalar] {
        &mut self.data
    }

    pub fn get(&self, ix: usize, iy: usize) -> &FieldScalar {
        let idx = self.idx(ix, iy);
        &self.data[idx]
    }

    pub fn get_mut(&mut self, ix: usize, iy: usize) -> &mut FieldScalar {
        let idx = self.idx(ix, iy);
        &mut self.data[idx]
    }

    pub fn fill(&mut self, value: FieldScalar) {
        self.data.fill(value);
    }

    /// Convert field data to f64 complex for high-precision operations.
    /// In standard mode this is a no-op copy; in mixed-precision mode it upcasts.
    #[cfg(feature = "mixed-precision")]
    pub fn to_f64_vec(&self) -> Vec<Complex64> {
        self.data
            .iter()
            .map(|c| Complex64::new(c.re as f64, c.im as f64))
            .collect()
    }

    #[cfg(not(feature = "mixed-precision"))]
    pub fn to_f64_vec(&self) -> Vec<Complex64> {
        self.data.clone()
    }
}

impl From<Field2D> for Vec<FieldScalar> {
    fn from(field: Field2D) -> Self {
        field.data
    }
}
