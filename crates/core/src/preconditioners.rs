//! Preconditioners for iterative eigensolvers.
//!
//! This module provides preconditioners that accelerate convergence of the
//! LOBPCG eigensolver by approximating the inverse of the operator.
//!
//! # Available Preconditioners
//!
//! ## Fourier-Space Preconditioners
//!
//! - [`FourierDiagonalPreconditioner`]: Simple Fourier-space scaling by `1/|k+G|²`.
//!   O(N log N) cost per application (2 FFTs). Default for TM mode.
//!
//! - [`TransverseProjectionPreconditioner`]: MPB-style physics-informed preconditioner.
//!   O(N log N) cost per application (6 FFTs). Default for TE mode.
//!   Based on Johnson & Joannopoulos, Optics Express 8, 173 (2001).
//!
//! ## Future Preconditioners
//!
//! - `FFTPreconditioner`: For envelope approximation operator (EA)
//!
//! # Preconditioner Trait
//!
//! All preconditioners implement [`OperatorPreconditioner<B>`], which requires:
//!
//! - `apply(&mut self, backend, buffer)`: Apply M^{-1} to the buffer in-place
//!
//! # Example
//!
//! ```ignore
//! use mpb2d_core::preconditioners::{OperatorPreconditioner, FourierDiagonalPreconditioner};
//!
//! // Build preconditioner
//! let mut precond = operator.build_homogeneous_preconditioner_adaptive();
//!
//! // Apply to residual
//! precond.apply(&backend, &mut residual);
//! ```

use crate::backend::SpectralBackend;

pub mod fourier_diagonal;
pub mod transverse_projection;
pub mod fft_preconditioner;

// Re-export commonly used types
pub use fourier_diagonal::{FourierDiagonalPreconditioner, SpectralStats, SHIFT_SMIN_FRACTION};
pub use transverse_projection::TransverseProjectionPreconditioner;

// ============================================================================
// Core Preconditioner Trait
// ============================================================================

/// A preconditioner for iterative eigensolvers.
///
/// Preconditioners approximate the inverse of the operator to accelerate
/// convergence. They transform the residual r → M^{-1} r where M ≈ A.
///
/// # Requirements
///
/// For optimal LOBPCG convergence, preconditioners should:
/// - Be symmetric positive definite (SPD)
/// - Approximate the inverse of the operator
/// - Be cheap to apply (ideally O(N log N) or O(N))
///
/// # Type Parameters
///
/// - `B`: The spectral backend type
pub trait OperatorPreconditioner<B: SpectralBackend> {
    /// Apply the preconditioner to a buffer in-place: buffer ← M^{-1} buffer
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer);
}
