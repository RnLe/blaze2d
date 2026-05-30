//! High-level drivers for eigenvalue problems.
//!
//! This module provides orchestration code that sets up operators, preconditioners,
//! and eigensolvers to solve complete physical problems.
//!
//! # Available Drivers
//!
//! ## Band Structure Driver
//!
//! The [`bandstructure`] submodule provides the Maxwell band structure driver:
//!
//! - [`run_with_options`](bandstructure::run_with_options): Compute photonic band structure along a k-path
//! - [`run_with_k_streaming`](bandstructure::run_with_k_streaming): Same, with a callback after each k-point
//! - [`run_with_diagnostics_and_options`](bandstructure::run_with_diagnostics_and_options): Same with convergence recording
//!
//! ## Single-Solve Driver
//!
//! The [`single_solve`] submodule provides a generic single-shot eigensolver driver
//! used both for one-off Maxwell solves and by the operator-data extraction driver:
//!
//! - [`solve`](single_solve::solve): Solve a single eigenvalue problem
//! - [`solve_with_diagnostics`](single_solve::solve_with_diagnostics): With convergence recording
//! - [`solve_with_warmstart`](single_solve::solve_with_warmstart): Using previous eigenvectors
//! - [`solve_batch`](single_solve::solve_batch): Solve multiple problems in sequence
//!
//! # Example
//!
//! ## Band Structure
//!
//! ```ignore
//! use blaze2d_core::drivers::bandstructure::{run_with_options, BandStructureJob, RunOptions};
//!
//! let result = run_with_options(backend, &job, RunOptions::default());
//! // result.bands[k_index][band_index] gives ω for each (k, band) pair
//! ```

pub mod bandstructure;
pub mod operator_data;
pub mod single_solve;

// Re-export commonly used types from bandstructure
pub use bandstructure::{
    BandStructureJob, BandStructureResult, BandStructureResultWithDiagnostics, KPointResult,
    RunOptions, run_with_diagnostics_and_options, run_with_k_streaming, run_with_options,
};

// Re-export from single_solve
pub use single_solve::{
    BatchSolveResult, OperatorDiagnostics, SingleSolveJob, SingleSolveResult,
    SingleSolveResultWithDiagnostics, create_convergence_study, solve, solve_with_diagnostics,
    solve_with_progress, solve_with_warmstart, solve_with_warmstart_and_diagnostics,
};

// Re-export ProgressInfo from eigensolver for use with solve_with_progress
pub use crate::eigensolver::ProgressInfo;
