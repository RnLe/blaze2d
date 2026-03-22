//! Generic single-shot eigensolver driver.
//!
//! This module provides a flexible driver for solving a single eigenvalue problem
//! without the complexity of k-path iteration. It's designed for use cases like:
//!
//! - Envelope approximation eigenproblems (moiré lattices)
//! - Testing and benchmarking eigensolvers
//! - Any single-shot Hermitian eigenproblem
//! - Parameter sweeps where each point is independent
//!
//! # Design Philosophy
//!
//! This driver is intentionally more generic than the `bandstructure` driver:
//!
//! - Works with any operator implementing [`LinearOperator`]
//! - No assumptions about the physical meaning of eigenvalues
//! - Optional eigenvalue transformation (e.g., λ → √λ for Maxwell)
//! - Full diagnostics support for convergence analysis
//!
//! # Usage
//!
//! ## Basic Usage
//!
//! ```ignore
//! use mpb2d_core::drivers::single_solve::{solve, SingleSolveJob};
//! use mpb2d_core::operators::EAOperator;
//!
//! let job = SingleSolveJob::new(8).with_tolerance(1e-8);
//! let result = solve(&mut operator, None, &job);
//! println!("Eigenvalues: {:?}", result.eigenvalues);
//! ```
//!
//! ## With Diagnostics
//!
//! ```ignore
//! use mpb2d_core::drivers::single_solve::{solve_with_diagnostics, SingleSolveJob};
//!
//! let job = SingleSolveJob::new(8).with_diagnostics();
//! let result = solve_with_diagnostics(&mut operator, None, &job, "my_solve");
//!
//! // Export convergence data for analysis
//! result.diagnostics.save_to_file("convergence.json")?;
//! ```
//!
//! ## Parameter Sweep with Warm-Start
//!
//! ```ignore
//! use mpb2d_core::drivers::single_solve::{solve, solve_with_warmstart, SingleSolveJob};
//!
//! let job = SingleSolveJob::new(8);
//!
//! // First solve (no warm-start)
//! let result1 = solve(&mut operator1, None, &job);
//!
//! // Subsequent solves with warm-start from previous
//! let result2 = solve_with_warmstart(&mut operator2, None, &job, &result1.eigenvectors);
//! ```

use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::backend::SpectralBackend;
use crate::diagnostics::{ConvergenceRun, ConvergenceStudy};
use crate::eigensolver::{
    DiagnosticResult, Eigensolver, EigensolverConfig, EigensolverResult, ProgressInfo,
};
use crate::field::Field2D;
use crate::operators::LinearOperator;
use crate::preconditioners::OperatorPreconditioner;

// ============================================================================
// Job Configuration
// ============================================================================

/// Configuration for a single eigenvalue solve.
///
/// This struct bundles all parameters needed to solve an eigenvalue problem.
/// Use the builder methods for convenient configuration.
///
/// # Example
///
/// ```ignore
/// let job = SingleSolveJob::new(10)
///     .with_tolerance(1e-10)
///     .with_max_iterations(300)
///     .with_diagnostics();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleSolveJob {
    /// Number of eigenvalues/eigenvectors to compute.
    pub n_bands: usize,
    /// Convergence tolerance for relative eigenvalue change.
    pub tolerance: f64,
    /// Maximum number of LOBPCG iterations.
    pub max_iterations: usize,
    /// Block size for LOBPCG (0 = automatic).
    ///
    /// Automatic sizing adds a small slack to n_bands for better convergence.
    /// Set explicitly for fine control over memory usage vs. convergence.
    pub block_size: usize,
    /// Whether to record convergence diagnostics.
    ///
    /// When enabled, per-iteration data (residuals, eigenvalues, etc.) is
    /// recorded for later analysis.
    pub record_diagnostics: bool,
    /// Whether to estimate operator condition number before solving.
    ///
    /// This adds overhead but provides useful diagnostics.
    #[serde(default)]
    pub estimate_condition_number: bool,
    /// Number of power iterations for condition number estimation.
    #[serde(default = "default_power_iterations")]
    pub power_iterations: usize,
    /// Optional label for this solve (used in diagnostics).
    #[serde(skip)]
    pub label: Option<String>,
}

fn default_power_iterations() -> usize {
    15
}

impl Default for SingleSolveJob {
    fn default() -> Self {
        Self {
            n_bands: 8,
            tolerance: 1e-8,
            max_iterations: 200,
            block_size: 0,
            record_diagnostics: false,
            estimate_condition_number: false,
            power_iterations: 15,
            label: None,
        }
    }
}

impl SingleSolveJob {
    /// Create a new job with the given number of bands.
    pub fn new(n_bands: usize) -> Self {
        Self {
            n_bands,
            ..Default::default()
        }
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the maximum iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the block size (0 = automatic).
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Enable convergence diagnostics recording.
    pub fn with_diagnostics(mut self) -> Self {
        self.record_diagnostics = true;
        self
    }

    /// Enable condition number estimation.
    pub fn with_condition_estimation(mut self) -> Self {
        self.estimate_condition_number = true;
        self
    }

    /// Set the number of power iterations for condition estimation.
    pub fn with_power_iterations(mut self, n: usize) -> Self {
        self.power_iterations = n;
        self
    }

    /// Set a label for this solve (used in diagnostics output).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Convert to EigensolverConfig.
    fn to_eigensolver_config(&self) -> EigensolverConfig {
        EigensolverConfig {
            n_bands: self.n_bands,
            max_iter: self.max_iterations,
            tol: self.tolerance,
            block_size: self.block_size,
            record_diagnostics: self.record_diagnostics,
            k_index: None,
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of a single eigenvalue solve.
///
/// Contains the computed eigenvalues, eigenvectors, and solve statistics.
/// The eigenvalues are returned as computed by the operator (no transformation).
#[derive(Debug, Clone)]
pub struct SingleSolveResult {
    /// Computed eigenvalues (sorted ascending).
    ///
    /// These are the raw eigenvalues from the operator. For Maxwell operators,
    /// these are ω². For envelope approximation, these are the band energies.
    pub eigenvalues: Vec<f64>,
    /// Computed eigenvectors as Field2D.
    pub eigenvectors: Vec<Field2D>,
    /// Number of LOBPCG iterations performed.
    pub iterations: usize,
    /// Whether the solver converged within max_iterations.
    pub converged: bool,
    /// Total solve time in seconds (excluding setup).
    pub elapsed_seconds: f64,
    /// Final relative residual for each band (if available).
    pub final_residuals: Vec<f64>,
}

/// Extended result including convergence diagnostics for analysis.
///
/// Returned by [`solve_with_diagnostics`].
#[derive(Debug, Clone)]
pub struct SingleSolveResultWithDiagnostics {
    /// Standard solve result.
    pub result: SingleSolveResult,
    /// Full convergence run data for analysis.
    pub diagnostics: ConvergenceRun,
}

// ============================================================================
// Operator Diagnostics
// ============================================================================

/// Diagnostics about the operator (condition number, etc.).
#[derive(Debug, Clone, Default)]
pub struct OperatorDiagnostics {
    /// Estimated maximum eigenvalue of A.
    pub lambda_max: Option<f64>,
    /// Estimated minimum eigenvalue of A.
    pub lambda_min: Option<f64>,
    /// Estimated condition number κ(A) = λ_max / λ_min.
    pub condition_number: Option<f64>,
    /// Self-adjointness error: |<x, Ay> - <Ax, y>| / (|<x,Ay>| + |<Ax,y>|).
    pub self_adjointness_error: Option<f64>,
    /// Preconditioned condition number κ(M⁻¹A) if preconditioner is used.
    pub preconditioned_condition_number: Option<f64>,
    /// Condition number reduction factor: κ(A) / κ(M⁻¹A).
    pub condition_reduction: Option<f64>,
}

// ============================================================================
// Main Entry Points
// ============================================================================

/// Solve a single eigenvalue problem.
///
/// This is the primary entry point for single-shot eigenvalue problems.
/// It wraps the LOBPCG eigensolver and provides a clean interface for
/// solving `A x = λ B x` where A and B are defined by the operator.
///
/// # Arguments
///
/// - `operator`: The linear operator (must implement [`LinearOperator`])
/// - `preconditioner`: Optional preconditioner for faster convergence
/// - `job`: Configuration for the solve
///
/// # Returns
///
/// A [`SingleSolveResult`] containing eigenvalues, eigenvectors, and solve statistics.
///
/// # Example
///
/// ```ignore
/// use mpb2d_core::drivers::single_solve::{solve, SingleSolveJob};
///
/// let job = SingleSolveJob::new(8).with_tolerance(1e-10);
/// let result = solve(&mut operator, Some(&mut preconditioner), &job);
///
/// if result.converged {
///     println!("Found {} eigenvalues in {} iterations",
///         result.eigenvalues.len(), result.iterations);
/// }
/// ```
pub fn solve<'a, O, B>(
    operator: &'a mut O,
    preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
    job: &SingleSolveJob,
) -> SingleSolveResult
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let label = job.label.as_deref().unwrap_or("single_solve");

    // Optional condition number estimation
    if job.estimate_condition_number {
        estimate_and_log_diagnostics(
            operator,
            preconditioner.is_some(),
            job.power_iterations,
            label,
        );
    }

    let start_time = Instant::now();
    let config = job.to_eigensolver_config();

    // Create and run the eigensolver
    let mut solver = Eigensolver::new(operator, config, preconditioner, None);
    let result: EigensolverResult = solver.solve();
    let eigenvectors = solver.all_eigenvectors();

    let elapsed = start_time.elapsed().as_secs_f64();

    info!(
        "[{}] n_bands={} converged={} iterations={} elapsed={:.3}s",
        label, job.n_bands, result.converged, result.iterations, elapsed
    );

    SingleSolveResult {
        eigenvalues: result.eigenvalues,
        eigenvectors,
        iterations: result.iterations,
        converged: result.converged,
        elapsed_seconds: elapsed,
        final_residuals: result.convergence.relative_residuals,
    }
}

/// Solve a single eigenvalue problem with progress callbacks.
///
/// This variant calls `on_progress` after each LOBPCG iteration, enabling
/// real-time progress displays (e.g., progress bars showing iteration count
/// and trace convergence).
///
/// # Arguments
///
/// - `operator`: The linear operator (must implement [`LinearOperator`])
/// - `preconditioner`: Optional preconditioner for faster convergence
/// - `job`: Configuration for the solve
/// - `on_progress`: Callback invoked after each iteration with [`ProgressInfo`]
///
/// # Returns
///
/// A [`SingleSolveResult`] containing eigenvalues, eigenvectors, and solve statistics.
///
/// # Example
///
/// ```ignore
/// use mpb2d_core::drivers::single_solve::{solve_with_progress, SingleSolveJob};
///
/// let job = SingleSolveJob::new(8).with_tolerance(1e-10);
/// let result = solve_with_progress(&mut operator, None, &job, |progress| {
///     println!("{}", progress.format_compact());
/// });
/// ```
pub fn solve_with_progress<'a, O, B, F>(
    operator: &'a mut O,
    preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
    job: &SingleSolveJob,
    on_progress: F,
) -> SingleSolveResult
where
    O: LinearOperator<B>,
    B: SpectralBackend,
    F: FnMut(&ProgressInfo),
{
    let label = job.label.as_deref().unwrap_or("single_solve");

    // Optional condition number estimation
    if job.estimate_condition_number {
        estimate_and_log_diagnostics(
            operator,
            preconditioner.is_some(),
            job.power_iterations,
            label,
        );
    }

    let start_time = Instant::now();
    let config = job.to_eigensolver_config();

    // Create and run the eigensolver with progress callback
    let mut solver = Eigensolver::new(operator, config, preconditioner, None);
    let result: EigensolverResult = solver.solve_with_progress(on_progress);
    let eigenvectors = solver.all_eigenvectors();

    let elapsed = start_time.elapsed().as_secs_f64();

    info!(
        "[{}] n_bands={} converged={} iterations={} elapsed={:.3}s",
        label, job.n_bands, result.converged, result.iterations, elapsed
    );

    SingleSolveResult {
        eigenvalues: result.eigenvalues,
        eigenvectors,
        iterations: result.iterations,
        converged: result.converged,
        elapsed_seconds: elapsed,
        final_residuals: result.convergence.relative_residuals,
    }
}

/// Solve a single eigenvalue problem with full convergence diagnostics.
///
/// Like [`solve`] but records per-iteration data for convergence analysis.
/// The diagnostics can be exported to JSON for plotting.
///
/// # Arguments
///
/// - `operator`: The linear operator
/// - `preconditioner`: Optional preconditioner
/// - `job`: Configuration (should have `record_diagnostics = true`)
/// - `run_label`: Label for this run in the diagnostics
///
/// # Returns
///
/// A [`SingleSolveResultWithDiagnostics`] containing both the solve result
/// and the full convergence history.
pub fn solve_with_diagnostics<'a, O, B>(
    operator: &'a mut O,
    preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
    job: &SingleSolveJob,
    run_label: impl Into<String>,
) -> SingleSolveResultWithDiagnostics
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let run_label = run_label.into();
    let label = job.label.as_deref().unwrap_or(&run_label);

    // Optional condition number estimation
    if job.estimate_condition_number {
        estimate_and_log_diagnostics(
            operator,
            preconditioner.is_some(),
            job.power_iterations,
            label,
        );
    }

    let start_time = Instant::now();

    // Force diagnostics recording
    let mut config = job.to_eigensolver_config();
    config.record_diagnostics = true;

    // Create and run the eigensolver with diagnostics
    let mut solver = Eigensolver::new(operator, config, preconditioner, None);
    let diag_result: DiagnosticResult = solver.solve_with_diagnostics(&run_label);
    let eigenvectors = solver.all_eigenvectors();

    let elapsed = start_time.elapsed().as_secs_f64();

    info!(
        "[{}] n_bands={} converged={} iterations={} elapsed={:.3}s (diagnostics recorded)",
        label, job.n_bands, diag_result.result.converged, diag_result.result.iterations, elapsed
    );

    let result = SingleSolveResult {
        eigenvalues: diag_result.result.eigenvalues,
        eigenvectors,
        iterations: diag_result.result.iterations,
        converged: diag_result.result.converged,
        elapsed_seconds: elapsed,
        final_residuals: diag_result.result.convergence.relative_residuals,
    };

    SingleSolveResultWithDiagnostics {
        result,
        diagnostics: diag_result.diagnostics,
    }
}

/// Solve with warm-start vectors from a previous solve.
///
/// Warm-starting can significantly accelerate convergence when the operator
/// has changed slightly (e.g., during a parameter sweep). The provided
/// eigenvectors are used as the initial guess for the LOBPCG iteration.
///
/// # Arguments
///
/// - `operator`: The linear operator
/// - `preconditioner`: Optional preconditioner
/// - `job`: Configuration for the solve
/// - `warm_start`: Eigenvectors from a previous solve
///
/// # Returns
///
/// A [`SingleSolveResult`] containing the solve results.
pub fn solve_with_warmstart<'a, O, B>(
    operator: &'a mut O,
    preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
    job: &SingleSolveJob,
    warm_start: &'a [Field2D],
) -> SingleSolveResult
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let label = job.label.as_deref().unwrap_or("single_solve");
    let start_time = Instant::now();

    let config = job.to_eigensolver_config();

    let mut solver = Eigensolver::new(operator, config, preconditioner, Some(warm_start));

    let result = solver.solve();
    let eigenvectors = solver.all_eigenvectors();

    let elapsed = start_time.elapsed().as_secs_f64();

    info!(
        "[{}] n_bands={} converged={} iterations={} elapsed={:.3}s (warm-start: {} vectors)",
        label,
        job.n_bands,
        result.converged,
        result.iterations,
        elapsed,
        warm_start.len()
    );

    SingleSolveResult {
        eigenvalues: result.eigenvalues,
        eigenvectors,
        iterations: result.iterations,
        converged: result.converged,
        elapsed_seconds: elapsed,
        final_residuals: result.convergence.relative_residuals,
    }
}

/// Solve with both warm-start and diagnostics.
pub fn solve_with_warmstart_and_diagnostics<'a, O, B>(
    operator: &'a mut O,
    preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
    job: &SingleSolveJob,
    warm_start: &'a [Field2D],
    run_label: impl Into<String>,
) -> SingleSolveResultWithDiagnostics
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let run_label = run_label.into();
    let label = job.label.as_deref().unwrap_or(&run_label);
    let start_time = Instant::now();

    let mut config = job.to_eigensolver_config();
    config.record_diagnostics = true;

    let mut solver = Eigensolver::new(operator, config, preconditioner, Some(warm_start));

    let diag_result: DiagnosticResult = solver.solve_with_diagnostics(&run_label);
    let eigenvectors = solver.all_eigenvectors();

    let elapsed = start_time.elapsed().as_secs_f64();

    info!(
        "[{}] n_bands={} converged={} iterations={} elapsed={:.3}s (warm-start, diagnostics)",
        label, job.n_bands, diag_result.result.converged, diag_result.result.iterations, elapsed
    );

    let result = SingleSolveResult {
        eigenvalues: diag_result.result.eigenvalues,
        eigenvectors,
        iterations: diag_result.result.iterations,
        converged: diag_result.result.converged,
        elapsed_seconds: elapsed,
        final_residuals: diag_result.result.convergence.relative_residuals,
    };

    SingleSolveResultWithDiagnostics {
        result,
        diagnostics: diag_result.diagnostics,
    }
}

// ============================================================================
// Batch Solving (Parameter Sweeps)
// ============================================================================

/// Result from a batch of eigenvalue solves.
#[derive(Debug, Clone)]
pub struct BatchSolveResult {
    /// Results for each solve in the batch.
    pub results: Vec<SingleSolveResult>,
    /// Total elapsed time for all solves.
    pub total_elapsed_seconds: f64,
    /// Average iterations per solve.
    pub avg_iterations: f64,
    /// Number of converged solves.
    pub n_converged: usize,
}

impl BatchSolveResult {
    /// Create a new BatchSolveResult from a vector of individual results.
    pub fn from_results(results: Vec<SingleSolveResult>, total_elapsed: f64) -> Self {
        let n_solves = results.len();
        let total_iterations: usize = results.iter().map(|r| r.iterations).sum();
        let n_converged = results.iter().filter(|r| r.converged).count();

        Self {
            results,
            total_elapsed_seconds: total_elapsed,
            avg_iterations: if n_solves > 0 {
                total_iterations as f64 / n_solves as f64
            } else {
                0.0
            },
            n_converged,
        }
    }
}

// Note: For batch solving with parameter sweeps, users should iterate manually:
//
// ```ignore
// let mut results = Vec::new();
// let mut prev_eigenvectors: Option<Vec<Field2D>> = None;
//
// for mut op in operators {
//     let result = if let Some(ref warm) = prev_eigenvectors {
//         solve_with_warmstart(&mut op, None, &job, warm)
//     } else {
//         solve(&mut op, None, &job)
//     };
//     prev_eigenvectors = Some(result.eigenvectors.clone());
//     results.push(result);
// }
// ```
//
// This gives full control over operator creation, preconditioner reuse, etc.

// ============================================================================
// Convergence Study Helpers
// ============================================================================

/// Create a convergence study from multiple diagnostic solves.
///
/// This is useful for comparing convergence behavior across different
/// configurations or operators.
pub fn create_convergence_study(
    results: Vec<SingleSolveResultWithDiagnostics>,
    study_name: impl Into<String>,
) -> ConvergenceStudy {
    let mut study = ConvergenceStudy::new(study_name);
    for r in results {
        study.add_run(r.diagnostics);
    }
    study
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Estimate and log operator diagnostics (condition number, self-adjointness).
fn estimate_and_log_diagnostics<O, B>(
    _operator: &mut O,
    has_preconditioner: bool,
    power_iterations: usize,
    label: &str,
) where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    // For now, just log that we would estimate condition number
    // Full implementation would call operator methods similar to bandstructure.rs
    debug!(
        "[{}] Condition number estimation requested ({} power iterations, precond={})",
        label, power_iterations, has_preconditioner
    );

    // TODO: Implement generic condition number estimation
    // This requires the operator to implement methods like:
    // - check_self_adjointness()
    // - estimate_condition_number()
    // - estimate_preconditioned_condition_number()
    //
    // These are currently only implemented on ThetaOperator.
    // For a fully generic driver, we would need to either:
    // 1. Add these methods to the LinearOperator trait (with defaults)
    // 2. Use a separate trait for diagnosable operators
    // 3. Accept a closure for custom diagnostics
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_builder() {
        let job = SingleSolveJob::new(10)
            .with_tolerance(1e-10)
            .with_max_iterations(500)
            .with_block_size(15)
            .with_diagnostics()
            .with_label("test_solve");

        assert_eq!(job.n_bands, 10);
        assert_eq!(job.tolerance, 1e-10);
        assert_eq!(job.max_iterations, 500);
        assert_eq!(job.block_size, 15);
        assert!(job.record_diagnostics);
        assert_eq!(job.label, Some("test_solve".to_string()));
    }

    #[test]
    fn test_job_default() {
        let job = SingleSolveJob::default();

        assert_eq!(job.n_bands, 8);
        assert_eq!(job.tolerance, 1e-8);
        assert_eq!(job.max_iterations, 200);
        assert_eq!(job.block_size, 0);
        assert!(!job.record_diagnostics);
        assert!(!job.estimate_condition_number);
    }

    #[test]
    fn test_to_eigensolver_config() {
        let job = SingleSolveJob::new(12)
            .with_tolerance(1e-6)
            .with_max_iterations(100)
            .with_block_size(20);

        let config = job.to_eigensolver_config();

        assert_eq!(config.n_bands, 12);
        assert_eq!(config.tol, 1e-6);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.block_size, 20);
    }
}
