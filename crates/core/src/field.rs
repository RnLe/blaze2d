//! Contiguous complex-valued field storage on a uniform 2D grid.
//!
//! Storage precision is selectable at runtime via the [`Real`] trait. The
//! [`Field2D`] container is generic over `R: Real ∈ {f32, f64}` so a single
//! binary can ship both precisions; the eigensolver picks the appropriate
//! monomorphisation at construction time. Dot-product accumulation, eigenvalue
//! storage, and dense Rayleigh–Ritz always run in [`AccumScalar`] (f64) to
//! preserve numerical stability — see [`crate::backend::SpectralBackend::dot`]
//! for the f32-storage / f64-accumulation invariant.
//!
//! `R` defaults to `f64` so existing call sites that wrote `Field2D` keep
//! their meaning unchanged; new generic code should use `Field2D<R>` or the
//! backend's `Buffer` associated type.
//!
//! See [docs/state_report.md](../../docs/state_report.md) §1.3 for context.

use num_complex::{Complex, Complex64};

use crate::grid::Grid2D;

// ============================================================================
// Real scalar trait
// ============================================================================

/// Marker trait for scalar floating-point types that can back a field's
/// storage. Implemented for `f32` and `f64`.
///
/// The trait bundles all the bounds the eigensolver, operators, and
/// preconditioners need from a real scalar: arithmetic, conversion to/from
/// f64 (for the accumulation boundary), and the standard `Float` methods.
pub trait Real:
    Copy
    + Send
    + Sync
    + 'static
    + std::fmt::Debug
    + num_traits::Float
    + num_traits::FloatConst
    + num_traits::FromPrimitive
    + num_traits::Zero
    + num_traits::One
    + num_traits::NumAssign
{
    /// Promote this real to `f64` for accumulation / dense linear algebra.
    fn to_accum(self) -> f64;

    /// Demote an `f64` accumulator back to storage precision.
    fn from_accum(value: f64) -> Self;
}

impl Real for f32 {
    #[inline]
    fn to_accum(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_accum(value: f64) -> Self {
        value as f32
    }
}

impl Real for f64 {
    #[inline]
    fn to_accum(self) -> f64 {
        self
    }
    #[inline]
    fn from_accum(value: f64) -> Self {
        value
    }
}

// ============================================================================
// Accumulation scalar (always f64 — preserved across the f32/f64 split)
// ============================================================================

/// Accumulation scalar used by dot products, Gram matrices, eigenvalue
/// computations, and the dense Rayleigh–Ritz step. Always `Complex<f64>`.
pub use num_complex::Complex64 as AccumScalar;

// ============================================================================
// Legacy aliases (f64-only). New code should use `Complex<R>` / `R` directly.
// ============================================================================

/// Legacy f64 storage scalar. Kept so that callers which have not yet been
/// migrated to `Complex<R>` continue to compile against the `f64`
/// monomorphisation. Equivalent to `Complex<f64>`.
pub type FieldScalar = Complex64;

/// Legacy f64 real component type. Equivalent to `f64`.
pub type FieldReal = f64;

// ============================================================================
// Field2D — generic storage container
// ============================================================================

/// A complex-valued field sampled on a uniform 2D grid. Generic over the
/// storage precision `R`; defaults to `f64` so existing call sites continue
/// to mean the same thing.
#[derive(Debug, Clone)]
pub struct Field2D<R: Real = f64> {
    grid: Grid2D,
    data: Vec<Complex<R>>,
}

impl<R: Real> Field2D<R> {
    pub fn zeros(grid: Grid2D) -> Self {
        let zero = Complex::<R>::new(R::zero(), R::zero());
        Self {
            data: vec![zero; grid.len()],
            grid,
        }
    }

    pub fn from_vec(grid: Grid2D, data: Vec<Complex<R>>) -> Self {
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

    pub fn as_slice(&self) -> &[Complex<R>] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [Complex<R>] {
        &mut self.data
    }

    pub fn get(&self, ix: usize, iy: usize) -> &Complex<R> {
        let idx = self.idx(ix, iy);
        &self.data[idx]
    }

    pub fn get_mut(&mut self, ix: usize, iy: usize) -> &mut Complex<R> {
        let idx = self.idx(ix, iy);
        &mut self.data[idx]
    }

    pub fn fill(&mut self, value: Complex<R>) {
        self.data.fill(value);
    }

    /// Promote field data to `Vec<Complex<f64>>` for high-precision
    /// postprocessing (Python boundary, dense linear algebra, etc.).
    pub fn to_f64_vec(&self) -> Vec<Complex64> {
        self.data
            .iter()
            .map(|c| Complex64::new(c.re.to_accum(), c.im.to_accum()))
            .collect()
    }
}

impl Field2D<f64> {
    /// Build a `Field2D<f64>` from a `Vec<Complex<f64>>`. Convenience for
    /// callers that explicitly want the f64 monomorphisation; identical to
    /// [`Field2D::from_vec`] specialised to `R = f64`.
    pub fn from_f64_vec(grid: Grid2D, data: Vec<Complex64>) -> Self {
        Self::from_vec(grid, data)
    }
}

impl<R: Real> From<Field2D<R>> for Vec<Complex<R>> {
    fn from(field: Field2D<R>) -> Self {
        field.data
    }
}

// ============================================================================
// Boundary helpers — construct storage-precision scalars from f64 components
// ============================================================================

/// Construct a `Complex<R>` at storage precision from f64 components.
/// The canonical way to write a complex literal in generic code.
#[inline]
pub fn cscalar<R: Real>(re: f64, im: f64) -> Complex<R> {
    Complex::new(R::from_accum(re), R::from_accum(im))
}

/// The zero complex scalar at storage precision `R`.
#[inline]
pub fn czero<R: Real>() -> Complex<R> {
    Complex::new(R::zero(), R::zero())
}

/// The unit complex scalar (1 + 0i) at storage precision `R`.
#[inline]
pub fn cone<R: Real>() -> Complex<R> {
    Complex::new(R::one(), R::zero())
}

/// Demote a `Complex<R>` to `Complex<f64>` (accumulation precision).
#[inline]
pub fn to_accum<R: Real>(value: Complex<R>) -> Complex64 {
    Complex64::new(value.re.to_accum(), value.im.to_accum())
}

/// Promote a `Complex<f64>` accumulator back to storage precision `R`.
#[inline]
pub fn from_accum<R: Real>(value: Complex64) -> Complex<R> {
    Complex::new(R::from_accum(value.re), R::from_accum(value.im))
}
