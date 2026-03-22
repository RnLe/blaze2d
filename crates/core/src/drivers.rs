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
//! - [`run`](bandstructure::run): Compute photonic band structure along a k-path
//! - [`run_with_options`](bandstructure::run_with_options): With custom preconditioner options
//! - [`run_with_diagnostics`](bandstructure::run_with_diagnostics): With convergence recording
//!
//! ## Single-Solve Driver
//!
//! The [`single_solve`] submodule provides a generic single-shot eigensolver driver:
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
//! use mpb2d_core::drivers::bandstructure::{run, BandStructureJob, Verbosity};
//!
//! let result = run(backend, &job, Verbosity::Verbose);
//! // result.bands[k_index][band_index] gives ω for each (k, band) pair
//! ```
//!
//! ## Single Solve (Envelope Approximation)
//!
//! ```ignore
//! use mpb2d_core::drivers::single_solve::{solve, SingleSolveJob};
//! use mpb2d_core::operators::EAOperator;
//!
//! let job = SingleSolveJob::new(10).with_tolerance(1e-10);
//! let result = solve(&mut ea_operator, Some(&mut preconditioner), &job);
//!
//! for (i, &ev) in result.eigenvalues.iter().enumerate() {
//!     println!("Band {}: E = {:.6}", i, ev);
//! }
//! ```

pub mod bandstructure;
pub mod single_solve;

// Re-export commonly used types from bandstructure
pub use bandstructure::{
    BandStructureJob, BandStructureResult, BandStructureResultWithDiagnostics, RunOptions,
    Verbosity, run, run_with_diagnostics, run_with_options,
};

// Re-export from single_solve
pub use single_solve::{
    BatchSolveResult, OperatorDiagnostics, SingleSolveJob, SingleSolveResult,
    SingleSolveResultWithDiagnostics, create_convergence_study, solve, solve_with_diagnostics,
    solve_with_progress, solve_with_warmstart, solve_with_warmstart_and_diagnostics,
};

// Re-export ProgressInfo from eigensolver for use with solve_with_progress
pub use crate::eigensolver::ProgressInfo;
