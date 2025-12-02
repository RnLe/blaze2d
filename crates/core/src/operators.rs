//! Operator implementations for various eigenvalue problems.
//!
//! This module provides linear operators for use with the LOBPCG eigensolver.
//! Each operator implements the [`LinearOperator`] trait, which defines the
//! minimal interface needed for iterative eigensolvers.
//!
//! # Available Operators
//!
//! ## Maxwell Operators (Photonic Crystals)
//!
//! The [`maxwell`] submodule provides operators for 2D photonic crystal band
//! structure calculations:
//!
//! - [`ThetaOperator`]: The full Maxwell curl-curl operator for both TM and TE polarizations
//!
//! ## Test Operators
//!
//! The [`test_operators`] submodule provides simple operators for testing:
//!
//! - [`ToyLaplacian`]: Simple periodic Laplacian
//! - [`ToyDiagonalSPD`]: Diagonal SPD operator with known spectrum
//!
//! ## Envelope Approximation Operators (Moiré Lattices)
//!
//! The [`envelope_approximation`] submodule provides operators for moiré lattice
//! band structure calculations using the envelope approximation.
//!
//! # Operator Trait
//!
//! All operators implement [`LinearOperator<B>`], which requires:
//!
//! - `apply(&mut self, input, output)`: Apply the operator A·x → y
//! - `apply_mass(&mut self, input, output)`: Apply the mass matrix B·x → y
//! - `alloc_field(&self)`: Allocate a compatible buffer
//! - `backend(&self)`: Access the spectral backend
//! - `grid(&self)`: Get the computational grid
//! - `bloch(&self)`: Get the Bloch wavevector (for k-space operators)
//!
//! # Example
//!
//! ```ignore
//! use mpb2d_core::operators::{LinearOperator, ThetaOperator};
//!
//! // Create operator
//! let mut theta = ThetaOperator::new(backend, dielectric, pol, bloch);
//!
//! // Apply operator
//! let mut output = theta.alloc_field();
//! theta.apply(&input, &mut output);
//! ```

use crate::backend::SpectralBackend;
use crate::grid::Grid2D;

pub mod envelope_approximation;
pub mod maxwell;
pub mod test_operators;

// Re-export commonly used types
pub use envelope_approximation::{EAOperator, EAOperatorBuilder};
pub use maxwell::ThetaOperator;
pub use test_operators::{ToyDiagonalSPD, ToyLaplacian};

// ============================================================================
// Core Operator Trait
// ============================================================================

/// Floor value for |k+G|² to prevent division by zero at Γ-point.
pub const K_PLUS_G_NEAR_ZERO_FLOOR: f64 = 1e-9;

/// Fraction of effective epsilon to use as mass floor in TM preconditioner.
pub const TM_PRECONDITIONER_MASS_FRACTION: f64 = 1e-2;

/// A linear operator for generalized eigenvalue problems.
///
/// This trait defines the interface for operators A and B in the generalized
/// eigenproblem A·x = λ·B·x. The eigensolver uses this trait to apply both
/// operators without needing to know their internal structure.
///
/// # Type Parameters
///
/// - `B`: The spectral backend type (CPU, CUDA, etc.)
///
/// # Required Methods
///
/// - [`apply`](Self::apply): Apply the main operator A
/// - [`apply_mass`](Self::apply_mass): Apply the mass operator B
/// - [`alloc_field`](Self::alloc_field): Allocate a compatible buffer
/// - [`backend`](Self::backend): Get a reference to the backend
/// - [`backend_mut`](Self::backend_mut): Get a mutable reference to the backend
/// - [`grid`](Self::grid): Get the computational grid
/// - [`bloch`](Self::bloch): Get the Bloch wavevector
///
/// # Optional Methods
///
/// - [`gamma_kernel_transform`](Self::gamma_kernel_transform): Transform factor for
///   Γ-point kernel basis (default: None)
pub trait LinearOperator<B: SpectralBackend> {
    /// Apply the main operator: output = A · input
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer);

    /// Apply the mass operator: output = B · input
    ///
    /// For standard eigenproblems (B = I), this is just a copy.
    /// For generalized eigenproblems, this applies the mass matrix.
    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer);

    /// Allocate a new field buffer compatible with this operator.
    fn alloc_field(&self) -> B::Buffer;

    /// Get a reference to the spectral backend.
    fn backend(&self) -> &B;

    /// Get a mutable reference to the spectral backend.
    fn backend_mut(&mut self) -> &mut B;

    /// Get the computational grid dimensions.
    fn grid(&self) -> Grid2D;

    /// Get the Bloch wavevector (k-point) for this operator.
    fn bloch(&self) -> [f64; 2];

    /// Get the transformation factor for Γ-point kernel basis.
    ///
    /// For operators using a similarity transform (like transformed TE where
    /// A' = ε^{-1/2} A ε^{-1/2}), the kernel basis must also be transformed:
    /// v₀ = ε^{1/2} u₀ instead of the naive constant mode u₀.
    ///
    /// Returns `Some(&[f64])` with the pointwise transformation factor (ε^{1/2}),
    /// or `None` if no transformation is needed (standard formulation).
    fn gamma_kernel_transform(&self) -> Option<&[f64]> {
        None // Default: no transformation needed
    }
}
