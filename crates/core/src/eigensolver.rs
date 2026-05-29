//! Eigensolver module for finding the lowest eigenpairs of the Maxwell operator Θ.
//!
//! This module implements a block LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
//! algorithm for solving the generalized eigenvalue problem:
//!
//! ```text
//! A x = λ B x
//! ```
//!
//! where:
//! - `A` is the Maxwell curl-curl operator (Θ operator)
//! - `B` is the mass operator (identity for TM, ε-weighted for TE)
//! - `λ` are the eigenvalues (ω² in physical units)
//! - `x` are the eigenvectors (field modes)
//!
//! # Module Structure
//!
//! The eigensolver is split into semantic submodules:
//! - [`initialization`]: Block initialization (X_0 creation)
//! - [`normalization`]: B-orthonormalization routines (including SVQB)
//! - [`deflation`]: Locking of converged eigenvectors
//! - [`dense`]: Dense Hermitian eigensolver for Rayleigh-Ritz projection

// Submodules
pub mod deflation;
pub mod dense;
pub mod initialization;
pub mod normalization;
pub mod subspace_prediction;

#[cfg(test)]
mod _tests_deflation;
#[cfg(test)]
mod _tests_initialization;
#[cfg(test)]
mod _tests_normalization;

// Re-exports from submodules
pub use deflation::{DeflationSubspace, LockingResult, check_for_locking};
pub use dense::{DenseEigenResult, solve_hermitian_eigen};
pub use initialization::{
    BlockEntry, GAMMA_TOLERANCE, InitializationConfig, InitializationResult, create_gamma_mode,
    is_gamma_point,
};
pub use normalization::{
    SvqbConfig, SvqbResult, b_inner_product, b_norm, normalize_to_unit_b_norm,
    orthogonalize_against_basis, orthonormalize_against_basis, project_out, svqb_orthonormalize,
};
pub use subspace_prediction::{
    PredictionMethod, PredictionResult, SubspaceHistory, compute_complex_overlap_matrix,
    polar_decomposition, polar_decomposition_with_singular_values,
};

// Re-export diagnostics types for convenience
pub use crate::diagnostics::{
    ConvergenceRecorder, ConvergenceRun, ConvergenceStudy, IterationSnapshot, PreconditionerType,
    RunConfig,
};

use crate::timing::Timer;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[cfg(feature = "native-linalg")]
use faer::Mat;

#[cfg(feature = "wasm-linalg")]
use nalgebra::{Complex as NaComplex, DMatrix};

use num_complex::Complex64;

#[cfg(feature = "mixed-precision")]
use num_complex::Complex32;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::Field2D;
use crate::grid::Grid2D;
use crate::operators::LinearOperator;
use crate::preconditioners::OperatorPreconditioner;
use crate::symmetry::SymmetryProjector;

// ============================================================================
// Progress Reporting
// ============================================================================

/// Progress information emitted during LOBPCG iterations.
///
/// This struct captures the current state of the eigensolver for
/// progress reporting to external consumers (e.g., progress bars).
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current iteration number (0-indexed).
    pub iteration: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Number of eigenvalues requested.
    pub n_bands: usize,
    /// Number of bands that have converged (locked).
    pub n_converged: usize,
    /// Current trace (sum of eigenvalues).
    pub trace: f64,
    /// Previous trace (for computing relative change).
    pub prev_trace: Option<f64>,
    /// Relative change in trace: |trace - prev_trace| / |prev_trace|.
    pub trace_rel_change: Option<f64>,
    /// Maximum relative residual across all bands.
    pub max_residual: f64,
    /// Maximum relative eigenvalue change across all bands.
    pub max_eigenvalue_change: f64,
}

impl ProgressInfo {
    /// Format a compact progress string suitable for display.
    pub fn format_compact(&self) -> String {
        let trace_change_str = match self.trace_rel_change {
            Some(change) => format!("Δ={:.2e}", change),
            None => "Δ=--".to_string(),
        };
        format!(
            "iter {:>3}/{} | trace={:.6} ({}) | conv={}/{}",
            self.iteration + 1,
            self.max_iterations,
            self.trace,
            trace_change_str,
            self.n_converged,
            self.n_bands,
        )
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Main configuration for the eigensolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EigensolverConfig {
    /// Number of eigenvalues/eigenvectors to compute.
    pub n_bands: usize,
    /// Maximum number of LOBPCG iterations.
    pub max_iter: usize,
    /// Convergence tolerance for relative residuals.
    pub tol: f64,
    /// Block size (0 = automatic based on n_bands).
    pub block_size: usize,

    // === Feature Toggles (for diagnostics/ablation studies) ===
    /// Enable/disable convergence diagnostics recording.
    /// When true, per-iteration data is collected for analysis.
    #[serde(default)]
    pub record_diagnostics: bool,
    /// Optional k-point index for logging (set by bandstructure code).
    #[serde(skip)]
    pub k_index: Option<usize>,
}

/// Slack to add to block size beyond n_bands for better convergence.
const BLOCK_SIZE_SLACK: usize = 2;

/// Toggle for Γ-point constant-mode deflation.
///
/// At the Γ point (k=0), the constant field is a spurious eigenvector with λ=0.
/// We deflate it out to prevent convergence issues and free up a slot for a
/// physical mode. This is safe when Γ is the **first** k-point in the path
/// (no warm-start to poison), and the converged Γ eigenvectors then serve as
/// valid warm-starts for subsequent k-points.
const ENABLE_GAMMA_DEFLATION: bool = true;

impl Default for EigensolverConfig {
    fn default() -> Self {
        // When using mixed-precision (f32 storage), the noise floor is higher due to
        // |G|^2 amplification of quantization noise. We relax the default tolerance
        // to 1e-4 to ensure robust convergence. The eigenvalue accuracy (bands)
        // remains high (approx 1e-8) due to the variational principle (error is quadratic).
        #[cfg(feature = "mixed-precision")]
        let default_tol = 1e-4;

        #[cfg(not(feature = "mixed-precision"))]
        let default_tol = 1e-6;

        Self {
            n_bands: 8,
            max_iter: 200,
            tol: default_tol,
            block_size: 0,
            record_diagnostics: false,
            k_index: None,
        }
    }
}

impl EigensolverConfig {
    /// Compute the effective block size (auto-sizing if block_size == 0).
    pub fn effective_block_size(&self) -> usize {
        let required = self.n_bands.max(1);
        let target = if self.block_size == 0 {
            required.saturating_add(BLOCK_SIZE_SLACK)
        } else {
            self.block_size
        };
        target.max(required)
    }

    /// Builder method: enable diagnostics recording.
    pub fn with_diagnostics(mut self) -> Self {
        self.record_diagnostics = true;
        self
    }

    /// Builder method: set tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Builder method: set max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder method: set number of bands.
    pub fn with_n_bands(mut self, n_bands: usize) -> Self {
        self.n_bands = n_bands;
        self
    }
}

// ============================================================================
// Debug Logging Helpers
// ============================================================================

/// Determine if we should emit debug logs at this iteration.
///
/// Logging schedule: iterations 1, 2, 3, 4, 5, 10, 20, 50, 100, then every 50.
fn should_log_iteration(iter: usize) -> bool {
    let iter_1based = iter + 1; // Convert 0-based to 1-based
    match iter_1based {
        1..=5 => true,
        10 | 20 | 50 | 100 => true,
        n if n > 100 && n % 50 == 0 => true,
        _ => false,
    }
}

/// Format a slice of f64 values for debug output (shows first few and last).
fn format_values(values: &[f64], max_show: usize) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    if values.len() <= max_show {
        let formatted: Vec<String> = values.iter().map(|v| format!("{:.4e}", v)).collect();
        return format!("[{}]", formatted.join(", "));
    }
    let half = max_show / 2;
    let front: Vec<String> = values[..half]
        .iter()
        .map(|v| format!("{:.4e}", v))
        .collect();
    let back: Vec<String> = values[values.len() - half..]
        .iter()
        .map(|v| format!("{:.4e}", v))
        .collect();
    format!("[{}, ..., {}]", front.join(", "), back.join(", "))
}

/// Calculate frequency range from eigenvalues slice.
/// Frequencies are ω = √λ (eigenvalues are ω²).
fn frequency_range_from_slice(eigenvalues: &[f64]) -> (f64, f64) {
    if eigenvalues.is_empty() {
        return (0.0, 0.0);
    }
    let freqs: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
        .collect();
    let min = freqs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = freqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

// ============================================================================
// Convergence Tracking
// ============================================================================

/// Per-band convergence state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandState {
    /// Band is still converging.
    Active,
    /// Band has converged (relative residual below tolerance).
    Converged,
    /// Band was locked (removed from active iteration).
    Locked,
}

/// Information about the convergence state of the solver.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Per-band convergence state.
    pub band_states: Vec<BandState>,
    /// Per-band relative residual norms: ||r_i||_B / |λ_i|.
    pub relative_residuals: Vec<f64>,
    /// Per-band relative eigenvalue changes: |λ_new - λ_old| / |λ_old|.
    pub relative_eigenvalue_changes: Vec<f64>,
    /// Number of converged bands.
    pub n_converged: usize,
    /// Maximum relative residual among active bands.
    pub max_residual: f64,
    /// Maximum relative eigenvalue change among active bands.
    pub max_eigenvalue_change: f64,
    /// Whether all requested bands have converged.
    pub all_converged: bool,
}

impl ConvergenceInfo {
    /// Create a new convergence info with all bands active.
    pub fn new(n_bands: usize) -> Self {
        Self {
            band_states: vec![BandState::Active; n_bands],
            relative_residuals: vec![f64::INFINITY; n_bands],
            relative_eigenvalue_changes: vec![f64::INFINITY; n_bands],
            n_converged: 0,
            max_residual: f64::INFINITY,
            max_eigenvalue_change: f64::INFINITY,
            all_converged: false,
        }
    }

    /// Update convergence info based on relative eigenvalue changes.
    ///
    /// Convergence is determined by relative eigenvalue change: |λ_new - λ_old| / |λ_old| < tol.
    /// This is more reliable than residual-based convergence for photonic band structure
    /// calculations where eigenvalues stabilize faster than residuals.
    ///
    /// # Arguments
    /// - `relative_residuals`: Per-band relative residual norms (for diagnostics)
    /// - `relative_eigenvalue_changes`: Per-band relative eigenvalue changes
    /// - `tol`: Convergence tolerance for relative eigenvalue change
    pub fn update_with_eigenvalue_changes(
        &mut self,
        relative_residuals: &[f64],
        relative_eigenvalue_changes: &[f64],
        tol: f64,
    ) {
        self.n_converged = 0;
        self.max_residual = 0.0;
        self.max_eigenvalue_change = 0.0;

        for (i, (&res, &ev_change)) in relative_residuals
            .iter()
            .zip(relative_eigenvalue_changes.iter())
            .enumerate()
        {
            if i >= self.band_states.len() {
                break;
            }
            self.relative_residuals[i] = res;
            self.relative_eigenvalue_changes[i] = ev_change;

            // Skip locked bands
            if self.band_states[i] == BandState::Locked {
                continue;
            }

            // Convergence based on eigenvalue change (not residual)
            if ev_change < tol {
                self.band_states[i] = BandState::Converged;
                self.n_converged += 1;
            } else {
                self.band_states[i] = BandState::Active;
                self.max_residual = self.max_residual.max(res);
                self.max_eigenvalue_change = self.max_eigenvalue_change.max(ev_change);
            }
        }

        // Count previously converged bands
        self.n_converged = self
            .band_states
            .iter()
            .filter(|&&s| s == BandState::Converged || s == BandState::Locked)
            .count();

        self.all_converged = self.n_converged >= self.band_states.len();
    }

    /// Update convergence info based on new residual norms (legacy method).
    #[allow(dead_code)]
    pub fn update(&mut self, relative_residuals: &[f64], tol: f64) {
        self.n_converged = 0;
        self.max_residual = 0.0;

        for (i, &res) in relative_residuals.iter().enumerate() {
            if i >= self.band_states.len() {
                break;
            }
            self.relative_residuals[i] = res;

            // Skip locked bands
            if self.band_states[i] == BandState::Locked {
                continue;
            }

            if res < tol {
                self.band_states[i] = BandState::Converged;
                self.n_converged += 1;
            } else {
                self.band_states[i] = BandState::Active;
                self.max_residual = self.max_residual.max(res);
            }
        }

        // Count previously converged bands
        self.n_converged = self
            .band_states
            .iter()
            .filter(|&&s| s == BandState::Converged || s == BandState::Locked)
            .count();

        self.all_converged = self.n_converged >= self.band_states.len();
    }
}

/// Result of the eigensolver.
#[derive(Debug, Clone)]
pub struct EigensolverResult {
    /// Computed eigenvalues (ω²).
    pub eigenvalues: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final convergence info.
    pub convergence: ConvergenceInfo,
    /// Whether the solver converged within max_iter.
    pub converged: bool,
}

/// Extended result including convergence diagnostics for analysis.
///
/// This is returned by [`Eigensolver::solve_with_diagnostics`] and includes
/// per-iteration data for plotting convergence curves.
#[derive(Debug, Clone)]
pub struct DiagnosticResult {
    /// Standard eigensolver result.
    pub result: EigensolverResult,
    /// Full convergence run data for analysis.
    pub diagnostics: ConvergenceRun,
}

// ============================================================================
// The Eigensolver
// ============================================================================

/// The LOBPCG eigensolver.
///
/// This struct holds all the state needed for the iterative solve:
/// - The operator providing A (curl-curl) and B (mass) operations
/// - The preconditioner M^{-1} (optional but recommended)
/// - The current block of eigenvector approximations X
/// - The deflation subspace Y for locked eigenvectors
/// - Configuration parameters
///
/// # Type Parameters
/// - `O`: The linear operator type (must implement `LinearOperator<B>`)
/// - `B`: The spectral backend type
pub struct Eigensolver<'a, O, B>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    /// The operator (provides A and B).
    operator: &'a mut O,
    /// Solver configuration.
    config: EigensolverConfig,
    /// Optional preconditioner M^{-1}.
    preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
    /// Optional warm-start vectors from previous k-point.
    warm_start: Option<&'a [Field2D]>,
    /// Current block of eigenvector approximations (active bands).
    x_block: Vec<BlockEntry<B>>,
    /// Previous search directions (W_k from iteration k-1).
    /// Empty on first iteration.
    w_block: Vec<B::Buffer>,
    /// Deflation subspace Y (locked eigenvectors).
    deflation: DeflationSubspace<B>,
    /// Current eigenvalue estimates for active bands (Rayleigh quotients).
    eigenvalues: Vec<f64>,
    /// Previous iteration's eigenvalues (for eigenvalue-based convergence check).
    previous_eigenvalues: Vec<f64>,
    /// Soft-locked bands: these bands have converged but remain in X block.
    /// For soft-locked bands, we skip residual/preconditioning and set P=W=0.
    /// This is numerically more stable than hard locking (deflation).
    soft_locked: Vec<bool>,
    /// Current iteration count.
    iteration: usize,
    /// Whether the solver has been initialized.
    initialized: bool,
    /// Optional symmetry projector to enforce parity constraints.
    symmetry_projector: Option<SymmetryProjector>,
}

impl<'a, O, B> Eigensolver<'a, O, B>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    /// Create a new eigensolver instance.
    ///
    /// # Arguments
    /// - `operator`: The linear operator providing A and B
    /// - `config`: Solver configuration
    /// - `preconditioner`: Optional preconditioner M^{-1}
    /// - `warm_start`: Optional warm-start vectors from previous k-point
    pub fn new(
        operator: &'a mut O,
        config: EigensolverConfig,
        preconditioner: Option<&'a mut dyn OperatorPreconditioner<B>>,
        warm_start: Option<&'a [Field2D]>,
    ) -> Self {
        Self {
            operator,
            config,
            preconditioner,
            warm_start,
            x_block: Vec::new(),
            w_block: Vec::new(),
            deflation: DeflationSubspace::new(),
            eigenvalues: Vec::new(),
            previous_eigenvalues: Vec::new(),
            soft_locked: Vec::new(),
            iteration: 0,
            initialized: false,
            symmetry_projector: None,
        }
    }

    /// Set the symmetry projector for this solver.
    pub fn set_symmetry_projector(&mut self, projector: SymmetryProjector) {
        self.symmetry_projector = Some(projector);
    }

    /// Apply symmetry projection to a set of vectors.
    fn apply_symmetry(&self, vectors: &mut [B::Buffer]) {
        if let Some(proj) = &self.symmetry_projector {
            for vector in vectors.iter_mut() {
                proj.apply(vector);
            }
        }
    }

    /// Initialize the solver (create initial block X_0).
    ///
    /// This must be called before any iteration steps.
    ///
    /// # Γ-Point Deflation
    ///
    /// At k=0 (the Γ point), the constant field is always an eigenvector with
    /// eigenvalue λ=0. This spurious mode is automatically added to the deflation
    /// subspace to prevent convergence issues. This is NOT optional because:
    ///
    /// 1. The constant mode has exactly λ=0, which can cause division issues
    /// 2. It's always degenerate with other modes in the null space
    /// 3. Including it in the active block would waste computational effort
    pub fn initialize(&mut self) {
        if self.initialized {
            return;
        }

        // Step 1: Check if we're at the Γ-point and add constant mode to deflation
        let bloch = self.operator.bloch();
        if initialization::is_gamma_point(bloch, initialization::GAMMA_TOLERANCE) {
            if ENABLE_GAMMA_DEFLATION {
                // Create B-normalized constant mode y₀
                let (y0, by0, _norm) = initialization::create_gamma_mode(self.operator);

                // Add to deflation if the mode is valid
                let norm = {
                    let backend = self.operator.backend();
                    normalization::b_norm(backend, &y0, &by0)
                };

                if norm > 1e-12 {
                    // Add to deflation subspace with eigenvalue 0 and band index 0
                    // Band index 0 is used as a special marker for the Γ constant mode
                    let backend = self.operator.backend();
                    let added = self.deflation.add_vector(
                        backend,
                        &y0,
                        &by0,
                        0.0,        // eigenvalue λ = 0
                        usize::MAX, // special band index for Γ mode (not a real band)
                    );

                    if added {
                        debug!(
                            "[eigensolver] Γ-point detected (k = [{:.6e}, {:.6e}]): added constant mode to deflation",
                            bloch[0], bloch[1]
                        );
                    } else {
                        warn!(
                            "[eigensolver] Γ-point: failed to add constant mode to deflation (linear dependence?)"
                        );
                    }
                }
            } else {
                // This branch only triggers if ENABLE_GAMMA_DEFLATION is manually set to false.
                // In that case, warn the user that Γ handling is disabled.
                info!("[eigensolver] Γ-point detected but constant-mode deflation is DISABLED.");
                info!(
                    "[eigensolver] Re-enable ENABLE_GAMMA_DEFLATION for correct Γ-point handling."
                );
            }
        }

        // Step 2: Create initial block X_0
        let block_size = self.config.effective_block_size();
        let init_config = InitializationConfig {
            block_size,
            max_random_attempts: block_size * 8,
            zero_tolerance: 1e-12,
        };

        let (mut x_block, _init_result) =
            initialization::initialize_block(self.operator, &init_config, self.warm_start);

        // Step 2b: Apply symmetry projection to initial block
        // This ensures the starting subspace respects the desired parity
        if let Some(proj) = &self.symmetry_projector {
            let backend = self.operator.backend(); // Create backend reference for B-norm (not used here but keeps scope consistent)
            let _ = backend; // Suppress unused warning
            for entry in &mut x_block {
                // Apply P to x
                proj.apply(&mut entry.vector);

                // Recompute B*x and A*x since x changed
                // (Note: initialize_block computed them, but projection invalidates them)
                self.operator.apply_mass(&entry.vector, &mut entry.mass);
                self.operator.apply(&entry.vector, &mut entry.applied);
            }
        }

        // Step 3: Project initial block against Γ mode (if present)
        // We must handle the projection and re-application carefully to avoid borrow conflicts
        if !self.deflation.is_empty() {
            // First pass: project and normalize using immutable backend
            {
                let backend = self.operator.backend();
                for entry in &mut x_block {
                    self.deflation
                        .project_single(backend, &mut entry.vector, &mut entry.mass);

                    // Re-normalize after projection
                    let norm = normalization::b_norm(backend, &entry.vector, &entry.mass);
                    if norm > 1e-12 {
                        let scale = num_complex::Complex64::new(1.0 / norm, 0.0);
                        backend.scale(scale, &mut entry.vector);
                        backend.scale(scale, &mut entry.mass);
                    }
                }
            }
            // Second pass: re-apply operator (requires mutable borrow)
            for entry in &mut x_block {
                self.operator.apply(&entry.vector, &mut entry.applied);
            }
        }

        self.x_block = x_block;
        self.initialized = true;

        // Initialize soft_locked to all false (no bands converged yet)
        self.soft_locked = vec![false; self.x_block.len()];

        // Compute initial Rayleigh quotients as eigenvalue estimates
        self.eigenvalues = self
            .x_block
            .iter()
            .map(|entry| entry.rayleigh_quotient(self.operator.backend()))
            .collect();
    }

    /// Check if the solver has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the current eigenvalue estimates.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Get the grid from the operator.
    pub fn grid(&self) -> Grid2D {
        self.operator.grid()
    }

    /// Get the current block size.
    pub fn block_size(&self) -> usize {
        self.x_block.len()
    }

    /// Get the number of locked (deflated) bands.
    pub fn locked_count(&self) -> usize {
        self.deflation.len()
    }

    /// Get the number of soft-locked bands (converged but still in X block).
    pub fn soft_locked_count(&self) -> usize {
        self.soft_locked.iter().filter(|&&b| b).count()
    }

    /// Get the number of active (non-soft-locked) bands in X block.
    pub fn active_band_count(&self) -> usize {
        self.soft_locked.iter().filter(|&&b| !b).count()
    }

    /// Get the total number of bands (locked + active).
    pub fn total_bands(&self) -> usize {
        self.deflation.len() + self.x_block.len()
    }

    /// Get a reference to the deflation subspace.
    pub fn deflation(&self) -> &DeflationSubspace<B> {
        &self.deflation
    }

    /// Get a reference to the operator.
    pub fn operator(&self) -> &O {
        self.operator
    }

    /// Get a mutable reference to the operator.
    pub fn operator_mut(&mut self) -> &mut O {
        self.operator
    }

    /// Check if a preconditioner is available.
    pub fn has_preconditioner(&self) -> bool {
        self.preconditioner.is_some()
    }

    // ========================================================================
    // Residual Computation
    // ========================================================================

    /// Compute the residual block: R_k = A*X_k - B*X_k * Λ_k
    ///
    /// For each band i:
    ///   r_i = A*x_i - λ_i * B*x_i
    ///
    /// where λ_i is the current Rayleigh quotient estimate.
    ///
    /// **Soft Locking Optimization**: Skips residual computation for soft-locked bands,
    /// returning a zero vector as a placeholder. These zeros won't be used since
    /// P[j] will also be skipped for soft-locked bands.
    ///
    /// Returns only the residual vectors (norms computed separately after deflation).
    fn compute_residuals(&self) -> Vec<B::Buffer> {
        let backend = self.operator.backend();
        let n = self.x_block.len();

        let mut residuals: Vec<B::Buffer> = Vec::with_capacity(n);

        for (i, entry) in self.x_block.iter().enumerate() {
            // Skip expensive computation for soft-locked bands
            if i < self.soft_locked.len() && self.soft_locked[i] {
                // Return zero residual for soft-locked bands (won't be used)
                let mut zero = entry.vector.clone();
                backend.scale(Complex64::ZERO, &mut zero);
                residuals.push(zero);
                continue;
            }

            // r_i = A*x_i (start with applied)
            let mut r = entry.applied.clone();

            // r_i = A*x_i - λ_i * B*x_i
            let lambda = self.eigenvalues[i];
            backend.axpy(
                num_complex::Complex64::new(-lambda, 0.0),
                &entry.mass,
                &mut r,
            );

            residuals.push(r);
        }

        residuals
    }

    /// Compute B-norms of residual vectors: ||r_i||_B = sqrt(r_i^* B r_i)
    ///
    /// This should be called AFTER deflation to get the correct norms
    /// (deflated components are removed, so we measure what actually matters).
    fn compute_residual_b_norms(&mut self, residuals: &[B::Buffer]) -> Vec<f64> {
        let mut norms: Vec<f64> = Vec::with_capacity(residuals.len());

        for r in residuals {
            // Compute B*r
            let mut br = self.operator.alloc_field();
            self.operator.apply_mass(r, &mut br);

            // ||r||_B = sqrt(r^* B r)
            let norm_sq = self.operator.backend().dot(r, &br).re;
            norms.push(norm_sq.max(0.0).sqrt());
        }

        norms
    }

    /// Compute relative residual norms: ||r_i||_B / (|λ_i| * ||x_i||_B)
    ///
    /// This is the standard relative residual for generalized eigenproblems.
    /// Since x_i is B-normalized (||x_i||_B = 1), this simplifies to:
    ///   relative_resid = ||r_i||_B / |λ_i|
    ///
    /// Uses a floor on λ to avoid division by near-zero eigenvalues.
    fn compute_relative_residuals(&self, residual_b_norms: &[f64]) -> Vec<f64> {
        const LAMBDA_FLOOR: f64 = 1e-10;

        residual_b_norms
            .iter()
            .zip(self.eigenvalues.iter())
            .map(|(&r_norm_b, &lambda)| {
                // ||r||_B / |λ| (since ||x||_B = 1 by construction)
                let denom = lambda.abs().max(LAMBDA_FLOOR);
                r_norm_b / denom
            })
            .collect()
    }

    /// Compute relative eigenvalue changes: |λ_new - λ_old| / |λ_old|
    ///
    /// This measures how much the eigenvalue estimates have changed from the
    /// previous iteration. Eigenvalue-based convergence is often faster than
    /// residual-based convergence because eigenvalues stabilize before residuals.
    ///
    /// Returns INFINITY for the first iteration (no previous values) or if
    /// the previous eigenvalue is near zero.
    fn compute_relative_eigenvalue_changes(&self) -> Vec<f64> {
        const LAMBDA_FLOOR: f64 = 1e-10;

        // If no previous eigenvalues, return INFINITY for all bands
        if self.previous_eigenvalues.is_empty() {
            return vec![f64::INFINITY; self.eigenvalues.len()];
        }

        self.eigenvalues
            .iter()
            .zip(self.previous_eigenvalues.iter())
            .map(|(&lambda_new, &lambda_old)| {
                // |λ_new - λ_old| / |λ_old|
                let denom = lambda_old.abs().max(LAMBDA_FLOOR);
                (lambda_new - lambda_old).abs() / denom
            })
            .collect()
    }

    /// Store current eigenvalues for the next iteration's comparison.
    fn store_previous_eigenvalues(&mut self) {
        self.previous_eigenvalues = self.eigenvalues.clone();
    }

    // ========================================================================
    // Preconditioning
    // ========================================================================

    /// Apply preconditioner to residuals: P_k = M^{-1} R_k
    ///
    /// If no preconditioner is available, returns a clone of the residuals
    /// (equivalent to M = I).
    ///
    /// **Soft Locking Optimization**: Only processes residuals for active (non-soft-locked)
    /// bands. Returns a smaller P block containing only active band directions.
    fn precondition_residuals(&mut self, residuals: &[B::Buffer]) -> Vec<B::Buffer> {
        let backend = self.operator.backend();

        // 1. Collect active residuals (cloned)
        let mut p_block: Vec<B::Buffer> = residuals
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                // Skip soft-locked bands entirely (their residuals are zeros anyway)
                if i < self.soft_locked.len() && self.soft_locked[i] {
                    None
                } else {
                    Some(r.clone())
                }
            })
            .collect();

        // 2. Apply preconditioner in batch to leverage optimized backends
        if !p_block.is_empty() {
            if let Some(ref mut precond) = self.preconditioner {
                precond.batch_apply(backend, &mut p_block);
            }
        }

        p_block
    }

    // ========================================================================
    // Deflation Projection
    // ========================================================================

    /// Apply deflation projection to a block of vectors: V ← P_Y V = V - Y(Y^* B V)
    ///
    /// This ensures the vectors are B-orthogonal to all locked eigenvectors in Y.
    /// If the deflation subspace is empty, this is a no-op.
    fn apply_deflation(&self, vectors: &mut [B::Buffer]) {
        if self.deflation.is_empty() {
            return;
        }
        self.deflation
            .project_block_no_mass(self.operator.backend(), vectors);
    }

    /// Soft-lock converged bands (Knyazev's recommended approach).
    ///
    /// Unlike hard locking (deflation), soft locking keeps converged vectors in X
    /// but skips expensive computations for them:
    /// - No residual computation
    /// - No preconditioning
    /// - No P or W directions
    ///
    /// The converged vectors remain in the Rayleigh-Ritz projection, allowing
    /// them to self-adjust slightly for numerical stability.
    ///
    /// # Degeneracy Handling
    /// For bands with near-degenerate eigenvalues (within a relative threshold),
    /// we only soft-lock the entire degenerate cluster at once. This prevents
    /// numerical issues where one band from a degenerate pair locks while the
    /// other can't properly orthogonalize without the P/W directions.
    ///
    /// # Arguments
    /// * `bands_to_lock` - Indices of bands to soft-lock (in the current active block)
    ///
    /// # Returns
    /// The number of bands newly soft-locked (not counting already soft-locked).
    fn soft_lock_bands(&mut self, bands_to_lock: &[usize]) -> usize {
        if bands_to_lock.is_empty() {
            return 0;
        }

        // Relative threshold for detecting degenerate eigenvalues
        // Two eigenvalues λ1, λ2 are considered degenerate if |λ1 - λ2| / max(|λ1|, |λ2|) < threshold
        const DEGENERACY_THRESHOLD: f64 = 1e-4;

        // Build the lock set (bands that want to lock)
        let lock_set: std::collections::HashSet<usize> = bands_to_lock.iter().copied().collect();

        // For each band that wants to lock, check if all near-degenerate partners also want to lock
        let mut safe_to_lock: Vec<usize> = Vec::new();

        for &band_idx in bands_to_lock {
            if band_idx >= self.eigenvalues.len() || band_idx >= self.soft_locked.len() {
                continue;
            }
            if self.soft_locked[band_idx] {
                // Already soft-locked
                continue;
            }

            let lambda_i = self.eigenvalues[band_idx];

            // Check all other bands for degeneracy
            let mut all_degenerate_partners_want_lock = true;

            for (j, &lambda_j) in self.eigenvalues.iter().enumerate() {
                if j == band_idx || j >= self.soft_locked.len() {
                    continue;
                }
                if self.soft_locked[j] {
                    // Already locked, not a concern
                    continue;
                }

                // Check if eigenvalues are near-degenerate
                let max_lambda = lambda_i.abs().max(lambda_j.abs()).max(1e-10);
                let relative_diff = (lambda_i - lambda_j).abs() / max_lambda;

                if relative_diff < DEGENERACY_THRESHOLD {
                    // Band j is near-degenerate with band_idx
                    // Check if j also wants to lock
                    if !lock_set.contains(&j) {
                        // Found a degenerate partner that doesn't want to lock
                        all_degenerate_partners_want_lock = false;
                        break;
                    }
                }
            }

            if all_degenerate_partners_want_lock {
                safe_to_lock.push(band_idx);
            }
        }

        // Now lock only the safe bands
        let mut newly_locked = 0;
        for band_idx in safe_to_lock {
            if !self.soft_locked[band_idx] {
                self.soft_locked[band_idx] = true;
                newly_locked += 1;

                let eigenvalue = self.eigenvalues.get(band_idx).copied().unwrap_or(0.0);
                debug!(
                    "[soft-lock] Band {} soft-locked (λ={:.6e}, ω={:.6e})",
                    band_idx + 1,
                    eigenvalue,
                    if eigenvalue > 0.0 {
                        eigenvalue.sqrt()
                    } else {
                        0.0
                    }
                );
            }
        }

        newly_locked
    }

    /// Explicitly B-orthogonalize vectors against soft-locked eigenvectors.
    ///
    /// This acts like deflation: R <- R - X_soft * (X_soft^T B R)
    ///
    /// For stability in high-condition-number or degenerate problems, active residuals
    /// must be orthogonal to the already-converged soft-locked vectors. Without this,
    /// numerical noise can cause the solver to "rediscover" the converged modes.
    fn apply_soft_deflation(&self, residuals: &mut [B::Buffer]) {
        if self.soft_locked_count() == 0 {
            return;
        }

        let backend = self.operator.backend();

        // For each residual (active band)
        for (i, r) in residuals.iter_mut().enumerate() {
            // Skip if this band itself is soft-locked (r should be 0 anyway)
            if i < self.soft_locked.len() && self.soft_locked[i] {
                continue;
            }

            // Project against every soft-locked vector
            for (j, is_locked) in self.soft_locked.iter().enumerate() {
                if *is_locked && j < self.x_block.len() {
                    let locked_entry = &self.x_block[j];
                    let locked_vec = &locked_entry.vector;
                    let locked_bx = &locked_entry.mass; // B * x_soft

                    // Calculate overlap: <x_soft, r>_B = x_soft^T B r = (B x_soft)^T r
                    // dot(a, b) computes a^H b. So we use dot(locked_bx, r).
                    let overlap = backend.dot(locked_bx, r);

                    // r <- r - overlap * x_soft
                    backend.axpy(-overlap, locked_vec, r);
                }
            }
        }
    }

    /// Lock converged bands and move them to the deflation subspace (HARD locking).
    ///
    /// NOTE: This is the old hard locking approach. Prefer `soft_lock_bands` for
    /// better numerical stability.
    ///
    /// # Arguments
    /// * `bands_to_lock` - Indices of bands to lock (in the current active block)
    ///
    /// # Returns
    /// The number of bands successfully locked.
    #[allow(dead_code)]
    fn lock_converged_bands(&mut self, bands_to_lock: &[usize]) -> usize {
        if bands_to_lock.is_empty() {
            return 0;
        }

        let mut locked_count = 0;

        // Sort in reverse order so we can remove from x_block without index shifting issues
        let mut sorted_indices: Vec<usize> = bands_to_lock.to_vec();
        sorted_indices.sort_by(|a, b| b.cmp(a));

        for &band_idx in &sorted_indices {
            if band_idx >= self.x_block.len() {
                continue;
            }

            // Get the band's data
            let entry = &self.x_block[band_idx];
            let vector = entry.vector.clone();
            let mass = entry.mass.clone();
            let eigenvalue = self.eigenvalues[band_idx];

            // The "global" band index includes previously locked bands
            let global_band_idx = self.deflation.len() + band_idx;

            // Try to add to deflation subspace
            let backend = self.operator.backend();
            let added =
                self.deflation
                    .add_vector(backend, &vector, &mass, eigenvalue, global_band_idx);

            if added {
                locked_count += 1;
                debug!(
                    "[deflation] Locked band {} (λ={:.6e}, ω={:.6e})",
                    global_band_idx + 1,
                    eigenvalue,
                    if eigenvalue > 0.0 {
                        eigenvalue.sqrt()
                    } else {
                        0.0
                    }
                );
            }
        }

        // Now remove the locked bands from x_block and eigenvalues
        // (sorted_indices is in reverse order, so removal is safe)
        for &band_idx in &sorted_indices {
            if band_idx < self.x_block.len() {
                self.x_block.remove(band_idx);
                self.eigenvalues.remove(band_idx);
                // Also remove from soft_locked
                if band_idx < self.soft_locked.len() {
                    self.soft_locked.remove(band_idx);
                }
            }
        }

        // Also need to trim W block if it's larger than new X block
        let new_block_size = self.x_block.len();
        if self.w_block.len() > new_block_size {
            self.w_block.truncate(new_block_size);
        }

        locked_count
    }

    // ========================================================================
    // Search Subspace Construction
    // ========================================================================

    /// Collect the search subspace Z_k = [X_k, P_k, W_k] as owned vectors.
    ///
    /// This clones the vectors to avoid borrow conflicts with subsequent
    /// mutable operations (like orthonormalization).
    ///
    /// # Soft Locking Optimization
    /// For soft-locked bands, we include X[j] but skip P[j] and W[j].
    /// This reduces the subspace dimension while keeping converged vectors
    /// in the Rayleigh-Ritz projection for numerical stability.
    ///
    /// # Returns
    /// - `subspace`: Vectors [X, P_active, W_active]
    /// - `bx_precomputed`: Precomputed B*X vectors (we already have these in entry.mass)
    /// - `block_sizes`: (x_size, p_size, w_size) for tracking per-block statistics
    ///
    /// # Optimization
    /// By returning the precomputed B*X vectors, we avoid recomputing them in SVQB.
    /// This saves m mass applications per iteration.
    fn collect_subspace_with_mass(
        &self,
        p_block: &[B::Buffer],
    ) -> (Vec<B::Buffer>, Vec<B::Buffer>, (usize, usize, usize)) {
        let m = self.x_block.len();

        // Count active (non-soft-locked) bands
        let n_active = self.soft_locked.iter().take(m).filter(|&&b| !b).count();

        // W directions are used for active bands only
        let has_w = !self.w_block.is_empty();

        // Calculate actual P and W sizes (only for non-soft-locked bands)
        // p_block should have the same size as the number of non-soft-locked bands
        let p_size = p_block.len().min(n_active);
        let w_size = if has_w {
            self.w_block.len().min(n_active)
        } else {
            0
        };

        let subspace_dim = m + p_size + w_size;
        let mut subspace: Vec<B::Buffer> = Vec::with_capacity(subspace_dim);
        let mut bx_precomputed: Vec<B::Buffer> = Vec::with_capacity(m);

        // Add ALL X_k vectors (cloned) and their precomputed B*X
        // (including soft-locked bands - they stay in X for Rayleigh-Ritz)
        for entry in &self.x_block {
            subspace.push(entry.vector.clone());
            bx_precomputed.push(entry.mass.clone());
        }

        // Add P_k vectors ONLY for active (non-soft-locked) bands
        // p_block is indexed by the active band indices
        let mut p_idx = 0;
        for (i, &is_locked) in self.soft_locked.iter().take(m).enumerate() {
            if !is_locked && p_idx < p_block.len() {
                subspace.push(p_block[p_idx].clone());
                p_idx += 1;
            }
            // Skip P for soft-locked bands (no residual was computed for them)
            let _ = i; // silence unused variable warning
        }

        // Add W_k vectors ONLY for active (non-soft-locked) bands
        if has_w {
            let mut w_idx = 0;
            for &is_locked in self.soft_locked.iter().take(m) {
                if !is_locked && w_idx < self.w_block.len() {
                    subspace.push(self.w_block[w_idx].clone());
                    w_idx += 1;
                }
                // Skip W for soft-locked bands
            }
        }

        let block_sizes = (m, p_size, w_size);
        (subspace, bx_precomputed, block_sizes)
    }

    /// Get the current subspace dimension.
    ///
    /// Returns 2m on first iteration (no W), 3m otherwise.
    /// With soft locking, returns m + 2*n_active where n_active = m - n_soft_locked.
    pub fn subspace_dimension(&self) -> usize {
        let m = self.x_block.len();
        let n_active = self.soft_locked.iter().take(m).filter(|&&b| !b).count();
        let has_w = !self.w_block.is_empty();
        if has_w {
            m + 2 * n_active
        } else {
            m + n_active
        }
    }

    // ========================================================================
    // Subspace Orthonormalization
    // ========================================================================

    /// B-orthonormalize the search subspace to get Q_k.
    ///
    /// Given Z_k = [X_k, P_k, W_k], compute Q_k such that:
    /// - Q_k^* B Q_k = I  (B-orthonormal)
    /// - range(Q_k) = range(Z_k)  (same subspace)
    ///
    /// This uses SVQB to handle:
    /// - Nearly linearly dependent columns
    /// - Rank deficiency detection and vector dropping
    /// - Numerical stability for ill-conditioned Gram matrices
    ///
    /// # Arguments
    /// * `q_block` - The subspace vectors [X, P, W]
    /// * `bx_precomputed` - Precomputed B*X vectors (saves m mass applications)
    /// * `block_sizes` - (x_size, p_size, w_size)
    ///
    /// # Optimization
    /// The first `x_size` vectors already have their B*v precomputed (from entry.mass).
    /// We only compute B*v for the P and W vectors, saving m mass applications per iteration.
    ///
    /// Returns (Q_k, BQ_k) where BQ_k[i] = B * Q_k[i], and the SVQB result.
    fn orthonormalize_subspace_with_precomputed_mass(
        &mut self,
        mut q_block: Vec<B::Buffer>,
        bx_precomputed: Vec<B::Buffer>,
        (x_size, p_size, w_size): (usize, usize, usize),
    ) -> (Vec<B::Buffer>, Vec<B::Buffer>, SvqbResult) {
        crate::profiler::start_timer("orthonormalize_subspace");

        // Build B*q block, reusing precomputed B*X and computing B*P and B*W fresh
        let mut bq_block: Vec<B::Buffer> = Vec::with_capacity(q_block.len());

        // First x_size vectors: use precomputed B*X (no mass applications needed!)
        for bx in bx_precomputed {
            bq_block.push(bx);
        }

        // Remaining vectors (P and W): compute B*v fresh using batched apply_mass
        let p_w_start = x_size;
        let p_w_count = q_block.len() - x_size;

        if p_w_count > 0 {
            // Allocate outputs for P and W mass application
            let mut b_pw_block = Vec::with_capacity(p_w_count);
            for _ in 0..p_w_count {
                b_pw_block.push(self.operator.alloc_field());
            }

            // Apply mass operator in batch
            self.operator
                .batch_apply_mass(&q_block[p_w_start..], &mut b_pw_block);

            // Append to bq_block
            bq_block.append(&mut b_pw_block);
        }

        // Apply SVQB to B-orthonormalize
        crate::profiler::start_timer("svqb");
        let config = SvqbConfig::default();
        let mut result = svqb_orthonormalize(
            self.operator.backend(),
            &mut q_block,
            &mut bq_block,
            &config,
        );
        crate::profiler::stop_timer("svqb");

        // Compute per-block drop counts from kept_indices
        let block_drops = result.compute_block_drops(x_size, p_size, w_size);
        result.block_drops = Some(block_drops);

        // Truncate to the output rank (SVQB may have dropped vectors)
        q_block.truncate(result.output_rank);
        bq_block.truncate(result.output_rank);

        crate::profiler::stop_timer("orthonormalize_subspace");
        (q_block, bq_block, result)
    }

    // ========================================================================
    // Projected Operator
    // ========================================================================

    /// Compute A*Q for all vectors in the orthonormalized subspace.
    ///
    /// This is factored out so that:
    /// 1. `project_operator` can use it to form A_s = Q^H * AQ
    /// 2. `update_ritz_vectors` can reuse it to compute AX = AQ * Y via GEMM
    ///    instead of recomputing A*X from scratch (saves m operator applications per iteration)
    ///
    /// # Returns
    /// A vector of buffers containing A*q_j for each q_j in q_block.
    fn compute_aq_block(&mut self, q_block: &[B::Buffer]) -> Vec<B::Buffer> {
        crate::profiler::start_timer("compute_aq_block");

        let r = q_block.len();
        let mut aq_block: Vec<B::Buffer> = Vec::with_capacity(r);

        // Pre-allocate output buffers
        for _ in 0..r {
            aq_block.push(self.operator.alloc_field());
        }

        // Apply operator in batch to leverage optimized backends (e.g. batched FFTs)
        self.operator.batch_apply(q_block, &mut aq_block);

        crate::profiler::stop_timer("compute_aq_block");
        aq_block
    }

    /// Form the projected operator A_s = Q_k^* A Q_k from precomputed A*Q.
    ///
    /// This is a small dense Hermitian matrix of size r×r where r = subspace_rank.
    /// The matrix is stored in column-major order as a flat Vec<Complex64>.
    ///
    /// Note: B_s = Q_k^* B Q_k = I by construction (SVQB ensures B-orthonormality),
    /// so we don't need to compute it explicitly.
    ///
    /// # Arguments
    /// * `q_block` - The B-orthonormalized subspace basis Q
    /// * `aq_block` - Precomputed A*Q (from `compute_aq_block`)
    ///
    /// # GPU Optimization
    /// When the `cuda` feature is enabled, the inner product computation
    /// A_s[i,j] = ⟨q_i, A*q_j⟩ is batched using `gram_matrix()`, which maps
    /// to a single cuBLAS ZGEMM call on GPU: A_s = Q^H × (A*Q).
    fn project_operator_with_aq(
        &self,
        q_block: &[B::Buffer],
        aq_block: &[B::Buffer],
    ) -> Vec<num_complex::Complex64> {
        crate::profiler::start_timer("project_operator");

        let r = q_block.len();
        if r == 0 {
            crate::profiler::stop_timer("project_operator");
            return vec![];
        }

        // A_s[i,j] = <q_i, A q_j> = q_i^* (A q_j)
        // The matrix is Hermitian, so A_s[j,i] = conj(A_s[i,j])

        #[cfg(feature = "cuda")]
        let result = {
            // GPU path: use batched gram_matrix (single ZGEMM call)
            let backend = self.operator.backend();
            let a_projected_row_major = backend.gram_matrix(q_block, aq_block);

            // Convert from row-major (gram_matrix output) to column-major (expected by dense solver)
            let mut a_projected = vec![num_complex::Complex64::ZERO; r * r];
            for i in 0..r {
                for j in 0..r {
                    a_projected[i + j * r] = a_projected_row_major[i * r + j];
                }
            }
            a_projected
        };

        #[cfg(all(not(feature = "cuda"), feature = "native-linalg"))]
        let result = {
            // CPU path with faer: use faer GEMM for matrix multiplication
            // A_s = Q^H * AQ where Q is n×r and AQ is n×r
            // Result is r×r in column-major order
            let n = q_block[0].as_slice().len();

            // Build Q matrix (n × r) from q_block - upcast to f64
            let q_mat = Mat::<faer::c64>::from_fn(n, r, |row, col| {
                let c = q_block[col].as_slice()[row];
                faer::c64::new(c.re as f64, c.im as f64)
            });

            // Build AQ matrix (n × r) from aq_block - upcast to f64
            let aq_mat = Mat::<faer::c64>::from_fn(n, r, |row, col| {
                let c = aq_block[col].as_slice()[row];
                faer::c64::new(c.re as f64, c.im as f64)
            });

            // Compute A_s = Q^H * AQ using GEMM
            let a_s = q_mat.adjoint() * &aq_mat; // r × r

            // Convert to column-major Vec<Complex64>
            let mut a_projected = vec![num_complex::Complex64::ZERO; r * r];
            for j in 0..r {
                for i in 0..r {
                    let c = a_s.get(i, j);
                    a_projected[i + j * r] = num_complex::Complex64::new(c.re, c.im);
                }
            }
            a_projected
        };

        #[cfg(all(not(feature = "cuda"), feature = "wasm-linalg"))]
        let result = {
            // CPU path with nalgebra: use nalgebra GEMM for matrix multiplication
            // A_s = Q^H * AQ where Q is n×r and AQ is n×r
            // Result is r×r in column-major order
            let n = q_block[0].as_slice().len();

            // Build Q matrix (n × r) from q_block
            let q_mat = DMatrix::<NaComplex<f64>>::from_fn(n, r, |row, col| {
                let c = q_block[col].as_slice()[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Build AQ matrix (n × r) from aq_block
            let aq_mat = DMatrix::<NaComplex<f64>>::from_fn(n, r, |row, col| {
                let c = aq_block[col].as_slice()[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Compute A_s = Q^H * AQ using GEMM
            let a_s = q_mat.adjoint() * &aq_mat; // r × r

            // Convert to column-major Vec<Complex64>
            let mut a_projected = vec![num_complex::Complex64::ZERO; r * r];
            for j in 0..r {
                for i in 0..r {
                    let c = a_s[(i, j)];
                    a_projected[i + j * r] = num_complex::Complex64::new(c.re, c.im);
                }
            }
            a_projected
        };

        crate::profiler::stop_timer("project_operator");
        result
    }

    // ========================================================================
    // Ritz Vector Update (Step 9)
    // ========================================================================

    /// Update the eigenvector approximations using the Rayleigh-Ritz results.
    ///
    /// Given the B-orthonormal subspace basis Q_k, precomputed A*Q, and the dense eigenpairs (Y, Θ),
    /// compute new Ritz vectors:
    ///
    /// ```text
    /// X_{k+1} = Q_k * Y_1
    /// BX_{k+1} = BQ_k * Y_1
    /// AX_{k+1} = AQ_k * Y_1   (reuses precomputed A*Q, avoiding m operator applications!)
    /// Λ_{k+1} = Θ_1
    /// ```
    ///
    /// where Y_1 is the (r × m) block of eigenvectors corresponding to the m
    /// smallest eigenvalues, and Θ_1 contains those eigenvalues.
    ///
    /// # Key Optimization
    /// Since A*(Q*Y) = (A*Q)*Y by linearity, we can compute A*X via GEMM using
    /// the precomputed A*Q block from `project_operator`, saving m expensive
    /// operator applications per iteration.
    fn update_ritz_vectors(
        &mut self,
        q_block: &[B::Buffer],
        bq_block: &[B::Buffer],
        aq_block: &[B::Buffer],
        dense_result: &DenseEigenResult,
    ) {
        #[cfg(feature = "profiling")]
        crate::profiler::start_timer("update_ritz_vectors");

        let n_bands = self.config.n_bands;
        let r = dense_result.dim; // Subspace dimension
        let m = n_bands.min(r); // Number of Ritz pairs to extract

        if m == 0 || r == 0 {
            #[cfg(feature = "profiling")]
            crate::profiler::stop_timer("update_ritz_vectors");
            return;
        }

        // Update eigenvalues: take the m smallest (already sorted)
        self.eigenvalues.clear();
        for j in 0..m {
            self.eigenvalues.push(dense_result.eigenvalue(j));
        }

        #[cfg(feature = "cuda")]
        let new_x_block = {
            // GPU path: use batched linear_combinations (single ZGEMM call per block)
            let backend = self.operator.backend();
            // Build coefficient matrix in column-major order: coeffs[i + j*r] = Y[i,j]
            let coeffs_x: Vec<num_complex::Complex64> = (0..m)
                .flat_map(|j| (0..r).map(move |i| dense_result.eigenvector(j)[i]))
                .collect();

            // Compute X_new = Q × C, BX_new = BQ × C, and AX_new = AQ × C using batched GEMM
            // Key optimization: AX_new = AQ × C reuses precomputed A*Q, avoiding m operator applications!
            let mut new_x_vectors = backend.linear_combinations(q_block, &coeffs_x, m);
            let mut new_bx_vectors = backend.linear_combinations(bq_block, &coeffs_x, m);
            let mut new_ax_vectors = backend.linear_combinations(aq_block, &coeffs_x, m);

            // Build new block entries (no fresh operator applications needed!)
            let mut new_x_block: Vec<BlockEntry<B>> = Vec::with_capacity(m);
            for _j in (0..m).rev() {
                new_x_block.push(BlockEntry {
                    vector: new_x_vectors.pop().unwrap(),
                    mass: new_bx_vectors.pop().unwrap(),
                    applied: new_ax_vectors.pop().unwrap(),
                });
            }
            new_x_block.reverse();
            new_x_block
        };

        #[cfg(all(not(feature = "cuda"), feature = "native-linalg"))]
        let new_x_block = {
            // CPU path with faer: use faer GEMM for matrix multiplication
            // X_new = Q * Y, BX_new = BQ * Y, AX_new = AQ * Y where matrices are n×r and Y is r×m
            let n = q_block[0].as_slice().len();

            // Build Q matrix (n × r) from q_block - upcast to f64
            let q_mat = Mat::<faer::c64>::from_fn(n, r, |row, col| {
                let c = q_block[col].as_slice()[row];
                faer::c64::new(c.re as f64, c.im as f64)
            });

            // Build BQ matrix (n × r) from bq_block - upcast to f64
            let bq_mat = Mat::<faer::c64>::from_fn(n, r, |row, col| {
                let c = bq_block[col].as_slice()[row];
                faer::c64::new(c.re as f64, c.im as f64)
            });

            // Build AQ matrix (n × r) from aq_block (precomputed - this is the optimization!) - upcast to f64
            let aq_mat = Mat::<faer::c64>::from_fn(n, r, |row, col| {
                let c = aq_block[col].as_slice()[row];
                faer::c64::new(c.re as f64, c.im as f64)
            });

            // Build Y matrix (r × m) from dense_result eigenvectors
            let y_mat = Mat::<faer::c64>::from_fn(r, m, |row, col| {
                let c = dense_result.eigenvector(col)[row];
                faer::c64::new(c.re, c.im)
            });

            // Compute X_new = Q * Y, BX_new = BQ * Y, and AX_new = AQ * Y using GEMM
            // Key optimization: AX_new = AQ * Y reuses precomputed A*Q, avoiding m operator applications!
            let x_new = &q_mat * &y_mat; // n × m
            let bx_new = &bq_mat * &y_mat; // n × m
            let ax_new = &aq_mat * &y_mat; // n × m (no fresh operator applications!)

            // Build new block entries
            let mut new_x_block: Vec<BlockEntry<B>> = Vec::with_capacity(m);

            for j in 0..m {
                // Extract x_j from GEMM result
                let mut x_j = self.operator.alloc_field();
                {
                    let dst = x_j.as_mut_slice();
                    for row in 0..n {
                        let c = x_new.get(row, j);
                        #[cfg(not(feature = "mixed-precision"))]
                        {
                            dst[row] = Complex64::new(c.re, c.im);
                        }
                        #[cfg(feature = "mixed-precision")]
                        {
                            dst[row] = Complex32::new(c.re as f32, c.im as f32);
                        }
                    }
                }

                // Extract bx_j from GEMM result
                let mut bx_j = self.operator.alloc_field();
                {
                    let dst = bx_j.as_mut_slice();
                    for row in 0..n {
                        let c = bx_new.get(row, j);
                        #[cfg(not(feature = "mixed-precision"))]
                        {
                            dst[row] = Complex64::new(c.re, c.im);
                        }
                        #[cfg(feature = "mixed-precision")]
                        {
                            dst[row] = Complex32::new(c.re as f32, c.im as f32);
                        }
                    }
                }

                // Extract ax_j from GEMM result (no fresh operator application needed!)
                let mut ax_j = self.operator.alloc_field();
                {
                    let dst = ax_j.as_mut_slice();
                    for row in 0..n {
                        let c = ax_new.get(row, j);
                        #[cfg(not(feature = "mixed-precision"))]
                        {
                            dst[row] = Complex64::new(c.re, c.im);
                        }
                        #[cfg(feature = "mixed-precision")]
                        {
                            dst[row] = Complex32::new(c.re as f32, c.im as f32);
                        }
                    }
                }

                new_x_block.push(BlockEntry {
                    vector: x_j,
                    mass: bx_j,
                    applied: ax_j,
                });
            }
            new_x_block
        };

        #[cfg(all(not(feature = "cuda"), feature = "wasm-linalg"))]
        let new_x_block = {
            // CPU path with nalgebra: use nalgebra GEMM for matrix multiplication
            // X_new = Q * Y, BX_new = BQ * Y, AX_new = AQ * Y where matrices are n×r and Y is r×m
            let n = q_block[0].as_slice().len();

            // Build Q matrix (n × r) from q_block
            let q_mat = DMatrix::<NaComplex<f64>>::from_fn(n, r, |row, col| {
                let c = q_block[col].as_slice()[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Build BQ matrix (n × r) from bq_block
            let bq_mat = DMatrix::<NaComplex<f64>>::from_fn(n, r, |row, col| {
                let c = bq_block[col].as_slice()[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Build AQ matrix (n × r) from aq_block (precomputed - this is the optimization!)
            let aq_mat = DMatrix::<NaComplex<f64>>::from_fn(n, r, |row, col| {
                let c = aq_block[col].as_slice()[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Build Y matrix (r × m) from dense_result eigenvectors
            let y_mat = DMatrix::<NaComplex<f64>>::from_fn(r, m, |row, col| {
                let c = dense_result.eigenvector(col)[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Compute X_new = Q * Y, BX_new = BQ * Y, and AX_new = AQ * Y using GEMM
            // Key optimization: AX_new = AQ * Y reuses precomputed A*Q, avoiding m operator applications!
            let x_new = &q_mat * &y_mat; // n × m
            let bx_new = &bq_mat * &y_mat; // n × m
            let ax_new = &aq_mat * &y_mat; // n × m (no fresh operator applications!)

            // Build new block entries
            let mut new_x_block: Vec<BlockEntry<B>> = Vec::with_capacity(m);

            for j in 0..m {
                // Extract x_j from GEMM result
                let mut x_j = self.operator.alloc_field();
                {
                    let dst = x_j.as_mut_slice();
                    for row in 0..n {
                        let c = x_new[(row, j)];
                        #[cfg(not(feature = "mixed-precision"))]
                        {
                            dst[row] = Complex64::new(c.re, c.im);
                        }
                        #[cfg(feature = "mixed-precision")]
                        {
                            dst[row] = Complex32::new(c.re as f32, c.im as f32);
                        }
                    }
                }

                // Extract bx_j from GEMM result
                let mut bx_j = self.operator.alloc_field();
                {
                    let dst = bx_j.as_mut_slice();
                    for row in 0..n {
                        let c = bx_new[(row, j)];
                        #[cfg(not(feature = "mixed-precision"))]
                        {
                            dst[row] = Complex64::new(c.re, c.im);
                        }
                        #[cfg(feature = "mixed-precision")]
                        {
                            dst[row] = Complex32::new(c.re as f32, c.im as f32);
                        }
                    }
                }

                // Extract ax_j from GEMM result (no fresh operator application needed!)
                let mut ax_j = self.operator.alloc_field();
                {
                    let dst = ax_j.as_mut_slice();
                    for row in 0..n {
                        let c = ax_new[(row, j)];
                        #[cfg(not(feature = "mixed-precision"))]
                        {
                            dst[row] = Complex64::new(c.re, c.im);
                        }
                        #[cfg(feature = "mixed-precision")]
                        {
                            dst[row] = Complex32::new(c.re as f32, c.im as f32);
                        }
                    }
                }

                new_x_block.push(BlockEntry {
                    vector: x_j,
                    mass: bx_j,
                    applied: ax_j,
                });
            }
            new_x_block
        };

        // Replace old X block with new Ritz vectors
        self.x_block = new_x_block;

        #[cfg(feature = "profiling")]
        crate::profiler::stop_timer("update_ritz_vectors");
    }

    /// Compute new history directions W_{k+1} = Q_k * Y_2.
    ///
    /// The history directions capture the "complementary" search directions
    /// from the Rayleigh-Ritz projection. Different LOBPCG variants use
    /// different strategies:
    ///
    /// 1. **Standard**: W_{k+1} = columns m..2m of Q_k * Y (next m eigenvectors)
    /// 2. **Simplified**: W_{k+1} = P_k (just reuse preconditioned residuals)
    /// 3. **Full**: W_{k+1} = all non-X directions in the projected subspace
    ///
    /// We use the standard approach: take columns m..2m of the eigenvector
    /// matrix Y, which correspond to the "next best" directions after X.
    /// These directions are B-orthogonal to X by construction.
    ///
    /// If the subspace is too small (r < 2m), we take whatever is available
    /// beyond the X directions (columns m..r).
    fn update_history_directions(
        &mut self,
        q_block: &[B::Buffer],
        dense_result: &DenseEigenResult,
    ) {
        let n_bands = self.config.n_bands;
        let r = dense_result.dim; // Subspace dimension
        let m = n_bands.min(r); // Number of X vectors

        // W directions start at column m and go up to min(2m, r)
        let w_start = m;
        let w_end = (2 * m).min(r);
        let n_w = w_end.saturating_sub(w_start);

        if n_w == 0 || r == 0 {
            // No room for W directions (subspace too small)
            self.w_block.clear();
            return;
        }

        #[cfg(feature = "cuda")]
        {
            // GPU path: use batched linear_combinations (single ZGEMM call)
            let backend = self.operator.backend();

            // Build coefficient matrix in column-major order: coeffs[i + j*r] = Y[i, m+j]
            let coeffs_w: Vec<num_complex::Complex64> = (w_start..w_end)
                .flat_map(|j| (0..r).map(move |i| dense_result.eigenvector(j)[i]))
                .collect();

            // Compute W_new = Q × C using batched GEMM
            self.w_block = backend.linear_combinations(q_block, &coeffs_w, n_w);
        }

        #[cfg(all(not(feature = "cuda"), feature = "native-linalg"))]
        {
            // CPU path with faer: use faer GEMM for matrix multiplication
            // W_new = Q * Y_w where Q is n×r and Y_w is r×n_w
            let n = q_block[0].as_slice().len();

            // Build Q matrix (n × r) from q_block - upcast to f64
            let q_mat = Mat::<faer::c64>::from_fn(n, r, |row, col| {
                let c = q_block[col].as_slice()[row];
                faer::c64::new(c.re as f64, c.im as f64)
            });

            // Build Y_w matrix (r × n_w) from dense_result eigenvectors (columns w_start..w_end)
            let y_w_mat = Mat::<faer::c64>::from_fn(r, n_w, |row, col| {
                let c = dense_result.eigenvector(w_start + col)[row];
                faer::c64::new(c.re, c.im)
            });

            // Compute W_new = Q * Y_w using GEMM
            let w_new = &q_mat * &y_w_mat; // n × n_w

            // Extract results into w_block
            let mut new_w_block: Vec<B::Buffer> = Vec::with_capacity(n_w);
            for j in 0..n_w {
                let mut w = self.operator.alloc_field();
                let dst = w.as_mut_slice();
                for row in 0..n {
                    let c = w_new.get(row, j);
                    #[cfg(not(feature = "mixed-precision"))]
                    {
                        dst[row] = Complex64::new(c.re, c.im);
                    }
                    #[cfg(feature = "mixed-precision")]
                    {
                        dst[row] = Complex32::new(c.re as f32, c.im as f32);
                    }
                }
                new_w_block.push(w);
            }
            self.w_block = new_w_block;
        }

        #[cfg(all(not(feature = "cuda"), feature = "wasm-linalg"))]
        {
            // CPU path with nalgebra: use nalgebra GEMM for matrix multiplication
            // W_new = Q * Y_w where Q is n×r and Y_w is r×n_w
            let n = q_block[0].as_slice().len();

            // Build Q matrix (n × r) from q_block
            let q_mat = DMatrix::<NaComplex<f64>>::from_fn(n, r, |row, col| {
                let c = q_block[col].as_slice()[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Build Y_w matrix (r × n_w) from dense_result eigenvectors (columns w_start..w_end)
            let y_w_mat = DMatrix::<NaComplex<f64>>::from_fn(r, n_w, |row, col| {
                let c = dense_result.eigenvector(w_start + col)[row];
                NaComplex::new(c.re as f64, c.im as f64)
            });

            // Compute W_new = Q * Y_w using GEMM
            let w_new = &q_mat * &y_w_mat; // n × n_w

            // Extract results into w_block
            let mut new_w_block: Vec<B::Buffer> = Vec::with_capacity(n_w);
            for j in 0..n_w {
                let mut w = self.operator.alloc_field();
                let dst = w.as_mut_slice();
                for row in 0..n {
                    let c = w_new[(row, j)];
                    #[cfg(not(feature = "mixed-precision"))]
                    {
                        dst[row] = Complex64::new(c.re, c.im);
                    }
                    #[cfg(feature = "mixed-precision")]
                    {
                        dst[row] = Complex32::new(c.re as f32, c.im as f32);
                    }
                }
                new_w_block.push(w);
            }
            self.w_block = new_w_block;
        }
    }

    // ========================================================================
    // Main Solve Loop
    // ========================================================================

    /// Run the LOBPCG iteration to convergence.
    ///
    /// This is the main entry point for solving the eigenvalue problem.
    /// It iterates until either:
    /// - All requested bands have converged (relative residual < tol)
    /// - Maximum iterations reached
    ///
    /// # Deflation Strategy
    ///
    /// The algorithm uses deflation to remove converged components:
    ///
    /// **Residuals (R):**
    /// - Apply deflation once: R ← P_Y R
    ///
    /// **Preconditioned residuals (P):**
    /// - Apply preconditioner first: P = M^{-1} R
    /// - Apply deflation once: P ← P_Y P
    ///
    /// **Search subspace (Z = [X, P, W]):**
    /// - No re-projection needed: X, P, W are already in the deflated
    ///   subspace from previous iterations
    /// - Just orthonormalize via SVQB
    ///
    /// # Algorithm Steps
    ///
    /// 1. **Compute residuals**: R_k = A*X_k - B*X_k * Λ_k
    /// 2. **Apply deflation to R**: R_k ← P_Y R_k
    /// 3. **Compute B-norms**: ||R_k||_B for convergence check
    /// 4. **Check convergence**: relative eigenvalue change < tol?
    /// 5. **Lock converged bands** (optional): move to deflation subspace
    /// 6. **Precondition residuals**: P_k = M^{-1} R_k
    /// 7. **Apply deflation to P**: P_k ← P_Y P_k
    /// 8. **Build search subspace**: Z_k = [X_k, P_k, W_k]
    /// 9. **B-orthonormalize**: Q_k = SVQB(Z_k)
    /// 10. **Project operator**: A_s = Q_k^* A Q_k
    /// 11. **Dense eigenproblem**: A_s Y = Y Θ
    /// 12. **Update Ritz vectors**: X_{k+1} = Q_k * Y_1, Λ_{k+1} = Θ_1
    /// 13. **Update history directions**: W_{k+1} = Q_k * Y_2
    ///
    /// # Returns
    /// An `EigensolverResult` containing the computed eigenvalues and convergence info.
    pub fn solve(&mut self) -> EigensolverResult {
        // Initialize faer with sequential execution (see lib.rs for rationale)
        crate::init_faer_sequential();

        // Ensure we're initialized
        if !self.initialized {
            self.initialize();
        }

        let n_bands_requested = self.config.n_bands;
        let n_bands = n_bands_requested.min(self.x_block.len());
        let mut convergence = ConvergenceInfo::new(n_bands);
        let start_time = Timer::start();
        let bloch = self.operator.bloch();

        // Main LOBPCG iteration loop
        for iter in 0..self.config.max_iter {
            self.iteration = iter;

            // Track the number of active bands (may shrink due to locking)
            let n_active = self.x_block.len();
            if n_active == 0 {
                // All bands have been locked
                break;
            }

            // ================================================================
            // Step 1: Compute residuals R_k = A*X_k - B*X_k * Λ_k
            // ================================================================
            let mut residuals = self.compute_residuals();

            // ================================================================
            // Step 2: Apply deflation to residuals R_k ← P_Y R_k
            // This removes components along locked eigenvectors.
            // ================================================================
            self.apply_deflation(&mut residuals);
            self.apply_soft_deflation(&mut residuals);
            self.apply_symmetry(&mut residuals);

            // ================================================================
            // Step 3: Compute B-norms of deflated residuals (LAZY)
            // OPTIMIZATION: Residual B-norms require mass applications (expensive).
            // Since convergence is based on eigenvalue changes (not residuals),
            // we only compute them when needed for logging/diagnostics.
            // This saves n_active mass applications on ~90% of iterations.
            // ================================================================
            let needs_residual_norms = should_log_iteration(iter);
            let (residual_b_norms, relative_residuals) = if needs_residual_norms {
                let norms = self.compute_residual_b_norms(&residuals);
                let rel = self.compute_relative_residuals(&norms);
                (norms, rel)
            } else {
                // Placeholder zeros - not used for convergence decisions
                (vec![0.0; n_active], vec![0.0; n_active])
            };

            // ================================================================
            // Step 4: Check convergence based on eigenvalue changes
            // Skip iter 0 and 1: we need at least 2 Ritz updates to have
            // meaningful eigenvalue changes (iter 0 initializes, iter 1 first real update)
            // ================================================================
            let n_check = n_active.min(relative_residuals.len());

            // Only check convergence starting from iteration 2
            // iter 0: initial eigenvalues from Rayleigh quotients
            // iter 1: first Ritz update - store these as baseline
            // iter 2+: compare to previous iteration
            if iter >= 2 {
                let relative_eigenvalue_changes = self.compute_relative_eigenvalue_changes();
                convergence.update_with_eigenvalue_changes(
                    &relative_residuals[..n_check],
                    &relative_eigenvalue_changes[..n_check.min(relative_eigenvalue_changes.len())],
                    self.config.tol,
                );
            }

            // Check for overall convergence (all requested bands)
            // Count both hard-locked (deflation) and soft-locked bands as converged
            // Also count bands that converge based on eigenvalue change criterion
            let n_hard_locked = self.deflation.len();
            let n_soft_locked = self.soft_locked_count();
            let total_converged = n_hard_locked + n_soft_locked + convergence.n_converged;
            if total_converged >= n_bands_requested {
                // All requested bands have converged
                let elapsed = start_time.elapsed_secs();
                let max_ev_change = convergence.max_eigenvalue_change;

                // Combine locked and active eigenvalues
                let all_eigenvalues = self.collect_all_eigenvalues();
                let (freq_min, freq_max) = frequency_range_from_slice(&all_eigenvalues);

                // Convert Bloch wavevector to fractional k-point for logging
                let k_frac = [bloch[0] / (2.0 * PI), bloch[1] / (2.0 * PI)];
                let k_idx = self.config.k_index.unwrap_or(0);

                let iters = iter + 1;
                let time_per_iter = elapsed / iters as f64;
                info!(
                    "[eigensolver] k#{:03} ({:+.4},{:+.4}) iters={:>3} Δλ={:+.2e} ω=[{:.4}..{:.4}] elapsed={:.2}s ({:.1}ms/iter, soft_locked={})",
                    k_idx,
                    k_frac[0],
                    k_frac[1],
                    iters,
                    max_ev_change,
                    freq_min,
                    freq_max,
                    elapsed,
                    time_per_iter * 1000.0,
                    n_soft_locked
                );

                return EigensolverResult {
                    eigenvalues: all_eigenvalues[..n_bands_requested.min(all_eigenvalues.len())]
                        .to_vec(),
                    iterations: iter + 1,
                    convergence,
                    converged: true,
                };
            }

            // ================================================================
            // Step 5: Lock converged bands
            // Lock bands that have converged to the deflation subspace
            // Uses eigenvalue-based convergence criterion
            // Only try locking from iteration 2 onward (need eigenvalue history)
            // ================================================================
            if iter >= 2 {
                // Recompute eigenvalue changes for soft locking decision
                let relative_eigenvalue_changes = self.compute_relative_eigenvalue_changes();
                let locking_result = check_for_locking(
                    &relative_eigenvalue_changes[..n_check.min(relative_eigenvalue_changes.len())],
                    self.config.tol,
                );

                if locking_result.has_locks() {
                    // Apply SOFT locking: mark bands as converged but keep them in X
                    // This is numerically more stable than hard locking (deflation)
                    let n_newly_locked = self.soft_lock_bands(&locking_result.bands_to_lock);
                    if n_newly_locked > 0 {
                        debug!(
                            "[iter {:>4}] Soft-locked {} bands (by Δλ), {} active remaining",
                            iter + 1,
                            n_newly_locked,
                            self.active_band_count()
                        );

                        // Check if we just reached full convergence
                        let n_soft_locked_now = self.soft_locked_count();
                        if n_soft_locked_now >= n_bands_requested {
                            let elapsed = start_time.elapsed_secs();
                            let max_ev_change = convergence.max_eigenvalue_change;
                            let all_eigenvalues = self.collect_all_eigenvalues();
                            let (freq_min, freq_max) = frequency_range_from_slice(&all_eigenvalues);
                            let k_frac = [bloch[0] / (2.0 * PI), bloch[1] / (2.0 * PI)];
                            let k_idx = self.config.k_index.unwrap_or(0);
                            let iters = iter + 1;
                            let time_per_iter = elapsed / iters as f64;
                            info!(
                                "[eigensolver] k#{:03} ({:+.4},{:+.4}) iters={:>3} Δλ={:+.2e} ω=[{:.4}..{:.4}] elapsed={:.2}s ({:.1}ms/iter, soft_locked={})",
                                k_idx,
                                k_frac[0],
                                k_frac[1],
                                iters,
                                max_ev_change,
                                freq_min,
                                freq_max,
                                elapsed,
                                time_per_iter * 1000.0,
                                n_soft_locked_now
                            );
                            return EigensolverResult {
                                eigenvalues: all_eigenvalues
                                    [..n_bands_requested.min(all_eigenvalues.len())]
                                    .to_vec(),
                                iterations: iter + 1,
                                convergence,
                                converged: true,
                            };
                        }
                    }
                }
            }

            // ================================================================
            // Step 6: Precondition residuals P_k = M^{-1} R_k
            // Soft-locked bands are skipped (returns smaller P block)
            // ================================================================
            let mut p_block = self.precondition_residuals(&residuals);

            // ================================================================
            // Step 7: Apply deflation to P
            // P_k ← P_Y P_k (preconditioner may have reintroduced components along Y)
            // ================================================================
            self.apply_deflation(&mut p_block);
            self.apply_soft_deflation(&mut p_block);
            self.apply_symmetry(&mut p_block);

            // ================================================================
            // Step 8: Build search subspace Z_k = [X_k, P_k, W_k]
            // Also collects precomputed B*X to avoid recomputing in SVQB
            // ================================================================
            let (subspace, bx_precomputed, block_sizes) = self.collect_subspace_with_mass(&p_block);

            // ================================================================
            // Step 9: B-orthonormalize to get Q_k with Q_k^* B Q_k = I
            // Uses SVQB to handle near-linear-dependence and rank deficiency
            // OPTIMIZATION: Reuses precomputed B*X, saving m mass applications
            // ================================================================
            let (q_block, bq_block, svqb_result) = self
                .orthonormalize_subspace_with_precomputed_mass(
                    subspace,
                    bx_precomputed,
                    block_sizes,
                );
            let subspace_rank = svqb_result.output_rank;

            // ================================================================
            // DIAGNOSTIC: Subspace condition number warning (every iteration)
            // A high condition number indicates near-linear-dependence which
            // can cause eigenvalue jumps and convergence instability.
            // ================================================================
            let condition_number = svqb_result.condition_number();

            // Warn if condition number is dangerously high (near-singular subspace)
            const CONDITION_WARN_THRESHOLD: f64 = 1e10;
            const CONDITION_CRITICAL_THRESHOLD: f64 = 1e14;
            if condition_number > CONDITION_CRITICAL_THRESHOLD {
                warn!(
                    "[iter {:>4}] CRITICAL: Subspace near-singular! κ={:.2e} (dropped {} vectors). \
                    Consider soft restart.",
                    iter + 1,
                    condition_number,
                    svqb_result.dropped_count
                );
            } else if condition_number > CONDITION_WARN_THRESHOLD {
                warn!(
                    "[iter {:>4}] Subspace poorly conditioned: κ={:.2e} (rank={}/{}, dropped={})",
                    iter + 1,
                    condition_number,
                    svqb_result.output_rank,
                    svqb_result.input_count,
                    svqb_result.dropped_count
                );
            }

            // ================================================================
            // Step 10: Compute A*Q and form projected operator A_s = Q_k^* A Q_k
            // Note: B_s = Q_k^* B Q_k = I by construction (SVQB ensures this)
            // Key optimization: A*Q is computed once and reused in step 12 to
            // compute A*X via GEMM, avoiding m redundant operator applications.
            // ================================================================
            let aq_block = self.compute_aq_block(&q_block);
            let a_projected = self.project_operator_with_aq(&q_block, &aq_block);

            // ================================================================
            // Step 11: Solve dense eigenproblem A_s Y = Y Θ
            // ================================================================
            let dense_result = dense::solve_hermitian_eigen(&a_projected, subspace_rank);

            // Store previous frequencies for comparison (for warning about increases)
            let prev_frequencies: Vec<f64> = self
                .eigenvalues
                .iter()
                .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                .collect();

            // ================================================================
            // Step 12: Update Ritz vectors X_{k+1} = Q_k * Y_1, Λ_{k+1} = Θ_1
            // Store previous eigenvalues first for convergence tracking
            // A*X is computed via GEMM using precomputed A*Q (no fresh applies!)
            // ================================================================
            self.store_previous_eigenvalues();
            self.update_ritz_vectors(&q_block, &bq_block, &aq_block, &dense_result);

            // Warn if any frequencies increased (potential variational principle violation)
            // Only warn if the increase is significant compared to the convergence tolerance
            // A band at convergence might fluctuate at ~tol level, so we use tol as threshold
            if iter > 0 && !prev_frequencies.is_empty() {
                let curr_frequencies: Vec<f64> = self
                    .eigenvalues
                    .iter()
                    .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                    .collect();

                // Use convergence tolerance as the threshold for "significant" increase
                // ω = √λ, so relative change in ω ≈ 0.5 * relative change in λ
                // We compare absolute changes scaled by typical frequency magnitude
                let freq_tol = self.config.tol;

                let mut increases: Vec<String> = Vec::new();
                let n_compare = prev_frequencies.len().min(curr_frequencies.len());
                for (band, (&prev, &curr)) in prev_frequencies[..n_compare]
                    .iter()
                    .zip(curr_frequencies[..n_compare].iter())
                    .enumerate()
                {
                    // Only warn if increase is larger than convergence tolerance (relative)
                    // For small frequencies, use absolute tolerance as floor
                    let threshold = (prev * freq_tol).max(1e-12);
                    if curr > prev + threshold {
                        // Band index should account for locked bands
                        let global_band = self.deflation.len() + band;
                        increases.push(format!("b{}:{:.4e}→{:.4e}", global_band + 1, prev, curr));
                    }
                }

                if !increases.is_empty() {
                    warn!(
                        "[iter {:>4}] ω increased: {}",
                        iter + 1,
                        increases.join(", ")
                    );
                }
            }

            // ================================================================
            // Step 13: Compute new history directions W_{k+1} = Q_k * Y_2
            // ================================================================
            self.update_history_directions(&q_block, &dense_result);
            // Apply symmetry to W block (manually to avoid borrow checker issues with self)
            if let Some(proj) = &self.symmetry_projector {
                for vector in self.w_block.iter_mut() {
                    proj.apply(vector);
                }
            }

            // ================================================================
            // Debug logging at selected iterations
            // ================================================================
            if should_log_iteration(iter) {
                let iter_elapsed = start_time.elapsed_secs();
                let n_converged = convergence.n_converged;
                let max_res = convergence.max_residual;
                let max_ev_change = convergence.max_eigenvalue_change;
                let n_soft_locked = self.soft_locked_count();

                // Collect ALL frequencies (locked + active) for complete picture
                let all_eigenvalues = self.collect_all_eigenvalues();
                let all_frequencies: Vec<f64> = all_eigenvalues
                    .iter()
                    .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                    .collect();

                debug!(
                    "[iter {:>4}] elapsed={:.3}s converged={}/{} soft_locked={} max_Δλ={:.2e} max_res={:.2e} subspace_rank={} w_size={}",
                    iter + 1,
                    iter_elapsed,
                    n_converged,
                    n_active,
                    n_soft_locked,
                    max_ev_change,
                    max_res,
                    subspace_rank,
                    self.w_block.len()
                );
                // Log condition number with per-block drop info
                if let Some((x_drop, p_drop, w_drop)) = svqb_result.block_drops {
                    debug!(
                        "[iter {:>4}] subspace κ = {:.2e} (dropped {}: X={}, P={}, W={})",
                        iter + 1,
                        condition_number,
                        svqb_result.dropped_count,
                        x_drop,
                        p_drop,
                        w_drop
                    );
                } else {
                    debug!(
                        "[iter {:>4}] subspace κ = {:.2e} (dropped {} of {} directions)",
                        iter + 1,
                        condition_number,
                        svqb_result.dropped_count,
                        svqb_result.input_count
                    );
                }
                debug!(
                    "[iter {:>4}] frequencies (ω):  {} (first {} soft_locked)",
                    iter + 1,
                    format_values(&all_frequencies, 6),
                    n_soft_locked
                );
                debug!(
                    "[iter {:>4}] residual_B_norms: {}",
                    iter + 1,
                    format_values(&residual_b_norms[..n_active.min(residual_b_norms.len())], 6)
                );
                debug!(
                    "[iter {:>4}] relative_resids:  {}",
                    iter + 1,
                    format_values(
                        &relative_residuals[..n_active.min(relative_residuals.len())],
                        6
                    )
                );

                // ============================================================
                // DIAGNOSTIC: Rayleigh quotient vs Ritz eigenvalue check
                // This is the key invariant: λ_RQ = <x,Ax>_B/<x,Bx>_B should
                // match λ_RR from the dense eigenproblem. If they diverge,
                // the Rayleigh-Ritz projection is not variationally consistent.
                // ============================================================
                let backend = self.operator.backend();
                let mut rq_discrepancies: Vec<String> = Vec::new();
                for (j, entry) in self.x_block.iter().enumerate() {
                    // λ_RQ = (x^* A x) / (x^* B x)
                    // We have entry.applied = A*x and entry.mass = B*x
                    let x_ax = backend.dot(&entry.vector, &entry.applied).re;
                    let x_bx = backend.dot(&entry.vector, &entry.mass).re;
                    let lambda_rq = if x_bx.abs() > 1e-15 { x_ax / x_bx } else { 0.0 };

                    // λ_RR from stored eigenvalue (from Ritz projection)
                    let lambda_rr = self.eigenvalues[j];

                    // Compute relative discrepancy
                    let denom = lambda_rr.abs().max(1e-10);
                    let rel_diff = (lambda_rq - lambda_rr).abs() / denom;

                    // Flag if discrepancy is significant (> 1e-6 relative)
                    if rel_diff > 1e-6 {
                        // Band index (soft-locked bands are still in x_block)
                        let global_band = j;
                        rq_discrepancies.push(format!(
                            "b{}:RQ={:.6e},RR={:.6e},Δ={:.2e}",
                            global_band + 1,
                            lambda_rq,
                            lambda_rr,
                            rel_diff
                        ));
                    }
                }
                if !rq_discrepancies.is_empty() {
                    warn!(
                        "[iter {:>4}] Rayleigh quotient ≠ Ritz eigenvalue: {}",
                        iter + 1,
                        rq_discrepancies.join("; ")
                    );
                }

                // SVQB diagnostics (only log if rank deficiency detected)
                let min_sv = svqb_result.singular_values.last().copied().unwrap_or(0.0);
                let max_sv = svqb_result.singular_values.first().copied().unwrap_or(0.0);
                if svqb_result.dropped_count > 0 || min_sv < 1e-6 {
                    debug!(
                        "[iter {:>4}] SVQB: input={} output={} dropped={} σ=[{:.2e}..{:.2e}]",
                        iter + 1,
                        svqb_result.input_count,
                        svqb_result.output_rank,
                        svqb_result.dropped_count,
                        min_sv,
                        max_sv
                    );
                }
            }
        }

        // End of loop: either max iterations reached or all bands locked
        let elapsed = start_time.elapsed_secs();
        let max_ev_change = convergence.max_eigenvalue_change;
        let n_hard_locked = self.deflation.len();
        let n_soft_locked = self.soft_locked_count();

        // Combine locked and active eigenvalues
        let all_eigenvalues = self.collect_all_eigenvalues();
        let (freq_min, freq_max) = frequency_range_from_slice(&all_eigenvalues);

        // Determine if we actually converged (count both hard and soft locked)
        let total_converged = n_hard_locked + n_soft_locked;
        let converged = total_converged >= n_bands_requested;

        // Convert Bloch wavevector to fractional k-point for logging
        let k_frac = [bloch[0] / (2.0 * PI), bloch[1] / (2.0 * PI)];
        let k_idx = self.config.k_index.unwrap_or(0);

        let iters = if converged {
            self.iteration + 1
        } else {
            self.config.max_iter
        };
        let time_per_iter = elapsed / iters as f64;

        if converged {
            info!(
                "[eigensolver] k#{:03} ({:+.4},{:+.4}) iters={:>3} Δλ={:+.2e} ω=[{:.4}..{:.4}] elapsed={:.2}s ({:.1}ms/iter, soft_locked={})",
                k_idx,
                k_frac[0],
                k_frac[1],
                iters,
                max_ev_change,
                freq_min,
                freq_max,
                elapsed,
                time_per_iter * 1000.0,
                n_soft_locked
            );
        } else {
            info!(
                "[eigensolver] k#{:03} ({:+.4},{:+.4}) iters={:>3} Δλ={:+.2e} ω=[{:.4}..{:.4}] elapsed={:.2}s ({:.1}ms/iter, NOT CONVERGED, soft_locked={})",
                k_idx,
                k_frac[0],
                k_frac[1],
                iters,
                max_ev_change,
                freq_min,
                freq_max,
                elapsed,
                time_per_iter * 1000.0,
                n_soft_locked
            );
        }

        EigensolverResult {
            eigenvalues: all_eigenvalues[..n_bands_requested.min(all_eigenvalues.len())].to_vec(),
            iterations: self.iteration + 1,
            convergence,
            converged,
        }
    }

    /// Solve the eigenvalue problem with progress callbacks.
    ///
    /// This is identical to [`solve`] but calls `on_progress` after each iteration.
    /// The callback receives a [`ProgressInfo`] struct with the current state.
    ///
    /// # Arguments
    /// * `on_progress` - Callback invoked after each iteration
    ///
    /// # Example
    ///
    /// ```ignore
    /// solver.solve_with_progress(|progress| {
    ///     println!("{}", progress.format_compact());
    /// });
    /// ```
    pub fn solve_with_progress<F>(&mut self, mut on_progress: F) -> EigensolverResult
    where
        F: FnMut(&ProgressInfo),
    {
        // Initialize faer with sequential execution (see lib.rs for rationale)
        crate::init_faer_sequential();

        // Ensure we're initialized
        if !self.initialized {
            self.initialize();
        }

        let n_bands_requested = self.config.n_bands;
        let n_bands = n_bands_requested.min(self.x_block.len());
        let mut convergence = ConvergenceInfo::new(n_bands);
        let _start_time = Timer::start();
        let _bloch = self.operator.bloch();

        let mut prev_trace: Option<f64> = None;

        // Main LOBPCG iteration loop
        for iter in 0..self.config.max_iter {
            self.iteration = iter;

            let n_active = self.x_block.len();
            if n_active == 0 {
                break;
            }

            // Compute residuals R_k = A*X_k - B*X_k * Λ_k
            let mut residuals = self.compute_residuals();
            self.apply_deflation(&mut residuals);
            self.apply_soft_deflation(&mut residuals);

            // Compute B-norms of deflated residuals
            let residual_b_norms = self.compute_residual_b_norms(&residuals);
            let relative_residuals = self.compute_relative_residuals(&residual_b_norms);

            let n_check = n_active.min(relative_residuals.len());

            // Check convergence based on eigenvalue changes (starting from iter 2)
            if iter >= 2 {
                let relative_eigenvalue_changes = self.compute_relative_eigenvalue_changes();
                convergence.update_with_eigenvalue_changes(
                    &relative_residuals[..n_check],
                    &relative_eigenvalue_changes[..n_check.min(relative_eigenvalue_changes.len())],
                    self.config.tol,
                );
            }

            // Compute trace for progress reporting
            let all_eigenvalues = self.collect_all_eigenvalues();
            let current_trace: f64 = all_eigenvalues.iter().sum();
            let trace_rel_change = prev_trace.map(|pt| {
                if pt.abs() > 1e-15 {
                    (current_trace - pt).abs() / pt.abs()
                } else {
                    f64::INFINITY
                }
            });

            // Emit progress callback
            let n_soft_locked = self.soft_locked_count();
            let progress = ProgressInfo {
                iteration: iter,
                max_iterations: self.config.max_iter,
                n_bands: n_bands_requested,
                n_converged: n_soft_locked,
                trace: current_trace,
                prev_trace,
                trace_rel_change,
                max_residual: convergence.max_residual,
                max_eigenvalue_change: convergence.max_eigenvalue_change,
            };
            on_progress(&progress);

            prev_trace = Some(current_trace);

            // Check for overall convergence (all requested bands)
            let total_converged = n_soft_locked;
            if total_converged >= n_bands_requested {
                return EigensolverResult {
                    eigenvalues: all_eigenvalues[..n_bands_requested.min(all_eigenvalues.len())]
                        .to_vec(),
                    iterations: iter + 1,
                    convergence,
                    converged: true,
                };
            }

            // Soft-lock converged bands (from iter 2 onward)
            if iter >= 2 {
                let relative_eigenvalue_changes = self.compute_relative_eigenvalue_changes();
                let locking_result = check_for_locking(
                    &relative_eigenvalue_changes[..n_check.min(relative_eigenvalue_changes.len())],
                    self.config.tol,
                );

                if locking_result.has_locks() {
                    self.soft_lock_bands(&locking_result.bands_to_lock);
                }
            }

            // Precondition residuals (skips soft-locked bands)
            let mut p_block = self.precondition_residuals(&residuals);
            self.apply_deflation(&mut p_block);
            self.apply_soft_deflation(&mut p_block);

            // Build search subspace Z_k = [X_k, P_k, W_k] with precomputed B*X
            let (subspace, bx_precomputed, block_sizes) = self.collect_subspace_with_mass(&p_block);

            // B-orthonormalize using SVQB (reuses precomputed B*X)
            let (q_block, bq_block, svqb_result) = self
                .orthonormalize_subspace_with_precomputed_mass(
                    subspace,
                    bx_precomputed,
                    block_sizes,
                );
            let subspace_rank = svqb_result.output_rank;

            if subspace_rank == 0 {
                break;
            }

            // Form projected operator A_s = Q_k^* A Q_k
            // Compute A*Q once and reuse for both projection and Ritz vector update
            let aq_block = self.compute_aq_block(&q_block);
            let a_projected = self.project_operator_with_aq(&q_block, &aq_block);

            // Solve dense eigenproblem A_s Y = Y Θ
            let dense_result = dense::solve_hermitian_eigen(&a_projected, subspace_rank);

            // Store previous eigenvalues and update Ritz vectors
            // A*X computed via GEMM using precomputed A*Q (no fresh applies!)
            self.store_previous_eigenvalues();
            self.update_ritz_vectors(&q_block, &bq_block, &aq_block, &dense_result);
            self.update_history_directions(&q_block, &dense_result);
        }

        // End of loop: either max iterations reached or all bands soft-locked
        let all_eigenvalues = self.collect_all_eigenvalues();
        let n_soft_locked = self.soft_locked_count();
        let converged = n_soft_locked >= n_bands_requested;

        EigensolverResult {
            eigenvalues: all_eigenvalues[..n_bands_requested.min(all_eigenvalues.len())].to_vec(),
            iterations: self.iteration + 1,
            convergence,
            converged,
        }
    }

    /// Collect all eigenvalues (locked + active) in sorted order.
    fn collect_all_eigenvalues(&self) -> Vec<f64> {
        let mut all: Vec<f64> = Vec::with_capacity(self.deflation.len() + self.eigenvalues.len());

        // Add locked eigenvalues
        all.extend_from_slice(self.deflation.eigenvalues());

        // Add active eigenvalues
        all.extend_from_slice(&self.eigenvalues);

        // Sort (locked should already be sorted, but ensure overall ordering)
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        all
    }

    /// Get the current eigenvector approximations as Field2D.
    ///
    /// This extracts the vector component from each BlockEntry.
    /// **Note**: This does NOT include locked (deflated) eigenvectors.
    /// Use [`all_eigenvectors`] to get both locked and active vectors.
    pub fn eigenvectors(&self) -> Vec<Field2D> {
        self.x_block
            .iter()
            .take(self.config.n_bands)
            .map(|entry| {
                let grid = entry.vector.grid();
                Field2D::from_vec(grid, entry.vector.as_slice().to_vec())
            })
            .collect()
    }

    /// Get all eigenvectors (locked + active) as Field2D, sorted by eigenvalue.
    ///
    /// This returns the complete set of eigenvectors including:
    /// - Locked (deflated) vectors (e.g., Γ constant mode with ω=0)
    /// - Active vectors from the current iteration
    ///
    /// The vectors are returned in the same order as [`collect_all_eigenvalues`],
    /// sorted by eigenvalue from smallest to largest.
    pub fn all_eigenvectors(&self) -> Vec<Field2D> {
        let grid = self.operator.grid();

        // Collect (eigenvalue, eigenvector) pairs
        let mut all_pairs: Vec<(f64, Field2D)> =
            Vec::with_capacity(self.deflation.len() + self.x_block.len());

        // Add locked (deflated) vectors
        for (eigenvalue, vector) in self
            .deflation
            .eigenvalues()
            .iter()
            .zip(self.deflation.vectors().iter())
        {
            let field = Field2D::from_vec(grid, vector.as_slice().to_vec());
            all_pairs.push((*eigenvalue, field));
        }

        // Add active vectors
        for (eigenvalue, entry) in self.eigenvalues.iter().zip(self.x_block.iter()) {
            let field = Field2D::from_vec(grid, entry.vector.as_slice().to_vec());
            all_pairs.push((*eigenvalue, field));
        }

        // Sort by eigenvalue
        all_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take n_bands and extract eigenvectors
        all_pairs
            .into_iter()
            .take(self.config.n_bands)
            .map(|(_, vec)| vec)
            .collect()
    }

    /// Create a RunConfig from the current solver state.
    ///
    /// This captures all configuration parameters for diagnostic recording.
    /// Note: The preconditioner type defaults to FourierDiagonalKernelCompensated if a preconditioner
    /// is present. For more control, use `create_run_config_with_precond_type`.
    pub fn create_run_config(&self, label: impl Into<String>) -> RunConfig {
        use crate::diagnostics::PreconditionerType;

        let grid = self.operator.grid();
        let bloch = self.operator.bloch();

        // Default preconditioner type based on presence
        let precond_type = if self.preconditioner.is_some() {
            PreconditionerType::FourierDiagonalKernelCompensated
        } else {
            PreconditionerType::None
        };

        RunConfig::new(label)
            .with_resolution(grid.nx, grid.ny)
            .with_dimensions(grid.lx, grid.ly)
            .with_eigensolver_params(
                self.config.n_bands,
                self.config.max_iter,
                self.config.tol,
                self.config.effective_block_size(),
            )
            .with_toggles(precond_type, self.warm_start.is_some())
            .with_k_point(
                0, // Will be overwritten by caller if part of k-path
                [
                    bloch[0] / (2.0 * std::f64::consts::PI),
                    bloch[1] / (2.0 * std::f64::consts::PI),
                ],
                bloch,
            )
    }

    /// Create a RunConfig with explicit preconditioner type.
    pub fn create_run_config_with_precond_type(
        &self,
        label: impl Into<String>,
        precond_type: crate::diagnostics::PreconditionerType,
    ) -> RunConfig {
        let grid = self.operator.grid();
        let bloch = self.operator.bloch();

        RunConfig::new(label)
            .with_resolution(grid.nx, grid.ny)
            .with_dimensions(grid.lx, grid.ly)
            .with_eigensolver_params(
                self.config.n_bands,
                self.config.max_iter,
                self.config.tol,
                self.config.effective_block_size(),
            )
            .with_toggles(precond_type, self.warm_start.is_some())
            .with_k_point(
                0,
                [
                    bloch[0] / (2.0 * std::f64::consts::PI),
                    bloch[1] / (2.0 * std::f64::consts::PI),
                ],
                bloch,
            )
    }

    /// Run the LOBPCG iteration with full diagnostics recording.
    ///
    /// This is similar to [`solve`], but records per-iteration snapshots
    /// for later analysis and plotting. Use this when you want to study
    /// convergence behavior in detail.
    ///
    /// # Deflation Strategy
    ///
    /// The algorithm uses deflation to remove converged components:
    ///
    /// **Residuals (R):**
    /// - Apply deflation once: R ← P_Y R
    ///
    /// **Preconditioned residuals (P):**
    /// - Apply preconditioner first: P = M^{-1} R
    /// - Apply deflation once: P ← P_Y P
    ///
    /// **Search subspace (Z = [X, P, W]):**
    /// - No re-projection needed: X, P, W are already in the deflated
    ///   subspace from previous iterations
    /// - Just orthonormalize via SVQB
    ///
    /// # Arguments
    /// - `label`: A human-readable label for this run (e.g., "baseline", "no_precond")
    ///
    /// # Returns
    /// A [`DiagnosticResult`] containing both the standard result and
    /// the full convergence history.
    ///
    /// # Example
    /// ```ignore
    /// let diag_result = solver.solve_with_diagnostics("baseline_run");
    ///
    /// // Standard result
    /// println!("Converged: {}", diag_result.result.converged);
    ///
    /// // Export diagnostics to JSON
    /// let json = serde_json::to_string_pretty(&diag_result.diagnostics)?;
    /// std::fs::write("convergence.json", json)?;
    /// ```
    pub fn solve_with_diagnostics(&mut self, label: impl Into<String>) -> DiagnosticResult {
        // Initialize faer with sequential execution (see lib.rs for rationale)
        crate::init_faer_sequential();

        // Ensure we're initialized
        if !self.initialized {
            self.initialize();
        }

        let n_bands_requested = self.config.n_bands;
        let n_bands = n_bands_requested.min(self.x_block.len());
        let mut convergence = ConvergenceInfo::new(n_bands);
        let start_time = Timer::start();
        let bloch = self.operator.bloch();

        // Set up diagnostics recorder
        let run_config = self.create_run_config(label);
        let mut recorder = ConvergenceRecorder::new(run_config);
        recorder.start();

        // Threshold for warning about poor subspace conditioning
        const CONDITION_WARN_THRESHOLD: f64 = 1e10;
        const CONDITION_CRITICAL_THRESHOLD: f64 = 1e14;

        // Main LOBPCG iteration loop
        for iter in 0..self.config.max_iter {
            self.iteration = iter;

            // Track the number of active bands (may shrink due to locking)
            let n_active = self.x_block.len();
            if n_active == 0 {
                // All bands have been locked
                break;
            }

            // ================================================================
            // Step 1: Compute residuals R_k = A*X_k - B*X_k * Λ_k
            // ================================================================
            let mut residuals = self.compute_residuals();

            // ================================================================
            // Step 2: Apply deflation to residuals R_k ← P_Y R_k
            // This removes components along locked eigenvectors.
            // ================================================================
            self.apply_deflation(&mut residuals);
            self.apply_soft_deflation(&mut residuals);

            // ================================================================
            // Step 3: Compute B-norms of deflated residuals
            // This is the correct metric: we measure what's left after
            // projecting out the locked subspace
            // ================================================================
            let residual_b_norms = self.compute_residual_b_norms(&residuals);
            let relative_residuals = self.compute_relative_residuals(&residual_b_norms);

            // ================================================================
            // Step 4: Check convergence based on eigenvalue changes
            // Skip iter 0 and 1: we need at least 2 Ritz updates to have
            // meaningful eigenvalue changes (iter 0 initializes, iter 1 first real update)
            // ================================================================
            let n_check = n_active.min(relative_residuals.len());

            // Only check convergence starting from iteration 2
            // iter 0: initial eigenvalues from Rayleigh quotients
            // iter 1: first Ritz update - store these as baseline
            // iter 2+: compare to previous iteration
            if iter >= 2 {
                let relative_eigenvalue_changes = self.compute_relative_eigenvalue_changes();
                convergence.update_with_eigenvalue_changes(
                    &relative_residuals[..n_check],
                    &relative_eigenvalue_changes[..n_check.min(relative_eigenvalue_changes.len())],
                    self.config.tol,
                );
            }

            // Check for overall convergence
            let n_soft_locked = self.soft_locked_count();
            let total_converged = n_soft_locked;

            // Prepare subspace info for recording (will be filled in after SVQB)
            let subspace_dim_input = self.subspace_dimension();
            let w_size = self.w_block.len();

            // Run the rest of the iteration to get subspace info
            if total_converged >= n_bands_requested {
                // Record final snapshot before returning
                let snapshot = IterationSnapshot::new(iter)
                    .with_eigenvalues(self.eigenvalues.clone())
                    .with_residuals(residual_b_norms.clone(), relative_residuals.clone())
                    .with_convergence_counts(convergence.n_converged, n_soft_locked, n_active)
                    .with_subspace_info(subspace_dim_input, subspace_dim_input, 0, w_size);
                recorder.record_iteration(snapshot);

                // All requested bands have converged
                let elapsed = start_time.elapsed_secs();
                let max_rel = convergence.max_residual;
                let all_eigenvalues = self.collect_all_eigenvalues();
                let (freq_min, freq_max) = frequency_range_from_slice(&all_eigenvalues);

                info!(
                    "[solve_diag] k#{:03} k=({:+.3},{:+.3}) iters={:>3} rel={:+10.3e} frequencies=[{:>8.3}..{:>8.3}] elapsed={:.2}s (soft_locked={})",
                    iter + 1,
                    bloch[0],
                    bloch[1],
                    iter + 1,
                    max_rel,
                    freq_min,
                    freq_max,
                    elapsed,
                    n_soft_locked
                );

                let result = EigensolverResult {
                    eigenvalues: all_eigenvalues[..n_bands_requested.min(all_eigenvalues.len())]
                        .to_vec(),
                    iterations: iter + 1,
                    convergence,
                    converged: true,
                };

                let diagnostics = recorder.finalize_with_result(iter + 1, true);
                return DiagnosticResult {
                    result,
                    diagnostics,
                };
            }

            // ================================================================
            // Step 5: Soft-lock converged bands
            // Uses eigenvalue-based convergence criterion
            // Only try locking from iteration 2 onward (need eigenvalue history)
            // ================================================================
            if iter >= 2 {
                // Recompute eigenvalue changes for soft-locking decision
                let relative_eigenvalue_changes = self.compute_relative_eigenvalue_changes();
                let locking_result = check_for_locking(
                    &relative_eigenvalue_changes[..n_check.min(relative_eigenvalue_changes.len())],
                    self.config.tol,
                );

                if locking_result.has_locks() {
                    let n_newly_locked = self.soft_lock_bands(&locking_result.bands_to_lock);
                    if n_newly_locked > 0 {
                        debug!(
                            "[iter {:>4}] Soft-locked {} bands (by Δλ), {} active remaining",
                            iter + 1,
                            n_newly_locked,
                            self.active_band_count()
                        );
                    }
                }
            }

            // ================================================================
            // Step 6: Precondition residuals P_k = M^{-1} R_k
            // Soft-locked bands are skipped
            // ================================================================
            let mut p_block = self.precondition_residuals(&residuals);

            // ================================================================
            // Step 7: Apply deflation to P
            // P_k ← P_Y P_k (preconditioner may have reintroduced components along Y)
            // ================================================================
            self.apply_deflation(&mut p_block);
            self.apply_soft_deflation(&mut p_block);

            // ================================================================
            // Step 8: Build search subspace Z_k = [X_k, P_k, W_k]
            // Also collects precomputed B*X to avoid recomputing in SVQB
            // ================================================================
            let (subspace, bx_precomputed, block_sizes) = self.collect_subspace_with_mass(&p_block);

            // ================================================================
            // Step 9: B-orthonormalize to get Q_k
            // OPTIMIZATION: Reuses precomputed B*X, saving m mass applications
            // ================================================================
            let (q_block, bq_block, svqb_result) = self
                .orthonormalize_subspace_with_precomputed_mass(
                    subspace,
                    bx_precomputed,
                    block_sizes,
                );
            let subspace_rank = svqb_result.output_rank;

            // Check subspace conditioning (detect potential near-linear dependence)
            let condition_number = svqb_result.condition_number();
            if condition_number > CONDITION_CRITICAL_THRESHOLD {
                warn!(
                    "[iter {:>4}] CRITICAL: Subspace near-singular! κ={:.2e} (dropped {} vectors). \
                    Consider soft restart.",
                    iter + 1,
                    condition_number,
                    svqb_result.dropped_count
                );
            } else if condition_number > CONDITION_WARN_THRESHOLD {
                warn!(
                    "[iter {:>4}] Subspace poorly conditioned: κ={:.2e} (rank={}/{}, dropped={})",
                    iter + 1,
                    condition_number,
                    svqb_result.output_rank,
                    svqb_result.input_count,
                    svqb_result.dropped_count
                );
            }

            // ================================================================
            // Record iteration snapshot
            // ================================================================
            let snapshot = IterationSnapshot::new(iter)
                .with_eigenvalues(self.eigenvalues.clone())
                .with_residuals(residual_b_norms.clone(), relative_residuals.clone())
                .with_convergence_counts(
                    convergence.n_converged,
                    self.soft_locked_count(),
                    n_active,
                )
                .with_subspace_info(
                    svqb_result.input_count,
                    svqb_result.output_rank,
                    svqb_result.dropped_count,
                    w_size,
                );
            recorder.record_iteration(snapshot);

            // ================================================================
            // Step 10: Form projected operator A_s = Q_k^* A Q_k
            // Compute A*Q once and reuse for both projection and Ritz vector update
            // ================================================================
            let aq_block = self.compute_aq_block(&q_block);
            let a_projected = self.project_operator_with_aq(&q_block, &aq_block);

            // ================================================================
            // Step 11: Solve dense eigenproblem A_s Y = Y Θ
            // ================================================================
            let dense_result = dense::solve_hermitian_eigen(&a_projected, subspace_rank);

            // Store previous frequencies for comparison (for warning about increases)
            let prev_frequencies: Vec<f64> = self
                .eigenvalues
                .iter()
                .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                .collect();

            // ================================================================
            // Step 12: Update Ritz vectors X_{k+1} = Q_k * Y_1, Λ_{k+1} = Θ_1
            // Store previous eigenvalues first for convergence tracking
            // A*X computed via GEMM using precomputed A*Q (no fresh applies!)
            // ================================================================
            self.store_previous_eigenvalues();
            self.update_ritz_vectors(&q_block, &bq_block, &aq_block, &dense_result);

            // Warn if any frequencies increased (potential variational principle violation)
            // Only warn if the increase is significant compared to the convergence tolerance
            // A band at convergence might fluctuate at ~tol level, so we use tol as threshold
            if iter > 0 && !prev_frequencies.is_empty() {
                let curr_frequencies: Vec<f64> = self
                    .eigenvalues
                    .iter()
                    .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                    .collect();

                // Use convergence tolerance as the threshold for "significant" increase
                // ω = √λ, so relative change in ω ≈ 0.5 * relative change in λ
                // We compare absolute changes scaled by typical frequency magnitude
                let freq_tol = self.config.tol;

                let mut increases: Vec<String> = Vec::new();
                let n_compare = prev_frequencies.len().min(curr_frequencies.len());
                for (band, (&prev, &curr)) in prev_frequencies[..n_compare]
                    .iter()
                    .zip(curr_frequencies[..n_compare].iter())
                    .enumerate()
                {
                    // Only warn if increase is larger than convergence tolerance (relative)
                    // For small frequencies, use absolute tolerance as floor
                    let threshold = (prev * freq_tol).max(1e-12);
                    if curr > prev + threshold {
                        // Band index should account for locked bands
                        let global_band = self.deflation.len() + band;
                        increases.push(format!("b{}:{:.4e}→{:.4e}", global_band + 1, prev, curr));
                    }
                }

                if !increases.is_empty() {
                    warn!(
                        "[iter {:>4}] ω increased: {}",
                        iter + 1,
                        increases.join(", ")
                    );
                }
            }

            // ================================================================
            // Step 13: Update history directions W_{k+1}
            // ================================================================
            self.update_history_directions(&q_block, &dense_result);

            // Debug logging at selected iterations
            if should_log_iteration(iter) {
                let iter_elapsed = start_time.elapsed_secs();
                let n_converged = convergence.n_converged;
                let max_res = convergence.max_residual;
                let max_ev_change = convergence.max_eigenvalue_change;
                let n_soft_locked = self.soft_locked_count();

                // Collect ALL frequencies (locked + active) for complete picture
                let all_eigenvalues = self.collect_all_eigenvalues();
                let all_frequencies: Vec<f64> = all_eigenvalues
                    .iter()
                    .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                    .collect();

                debug!(
                    "[iter {:>4}] elapsed={:.3}s converged={}/{} soft_locked={} max_Δλ={:.2e} max_res={:.2e} subspace_rank={} w_size={}",
                    iter + 1,
                    iter_elapsed,
                    n_converged,
                    n_active,
                    n_soft_locked,
                    max_ev_change,
                    max_res,
                    subspace_rank,
                    self.w_block.len()
                );
                debug!(
                    "[iter {:>4}] frequencies (ω):  {} (first {} soft_locked)",
                    iter + 1,
                    format_values(&all_frequencies, 6),
                    n_soft_locked
                );
                // Log condition number with per-block drop info
                if let Some((x_drop, p_drop, w_drop)) = svqb_result.block_drops {
                    debug!(
                        "[iter {:>4}] subspace κ = {:.2e} (dropped {}: X={}, P={}, W={})",
                        iter + 1,
                        condition_number,
                        svqb_result.dropped_count,
                        x_drop,
                        p_drop,
                        w_drop
                    );
                } else {
                    debug!(
                        "[iter {:>4}] subspace κ = {:.2e} (dropped {} of {} directions)",
                        iter + 1,
                        condition_number,
                        svqb_result.dropped_count,
                        svqb_result.input_count
                    );
                }
            }
        }

        // End of loop: either max iterations reached or all bands locked
        let elapsed = start_time.elapsed_secs();
        let max_ev_change = convergence.max_eigenvalue_change;
        let n_locked = self.deflation.len();

        let all_eigenvalues = self.collect_all_eigenvalues();
        let (freq_min, freq_max) = frequency_range_from_slice(&all_eigenvalues);

        let total_converged = n_locked + convergence.n_converged;
        let converged = total_converged >= n_bands_requested;

        if converged {
            info!(
                "[solve_diag] k#{:03} k=({:+.3},{:+.3}) iters={:>3} Δλ={:+10.3e} frequencies=[{:>8.3}..{:>8.3}] elapsed={:.2}s (locked={})",
                self.iteration + 1,
                bloch[0],
                bloch[1],
                self.iteration + 1,
                max_ev_change,
                freq_min,
                freq_max,
                elapsed,
                n_locked
            );
        } else {
            info!(
                "[solve_diag] k#{:03} k=({:+.3},{:+.3}) iters={:>3} Δλ={:+10.3e} frequencies=[{:>8.3}..{:>8.3}] elapsed={:.2}s (NOT CONVERGED, locked={})",
                self.config.max_iter,
                bloch[0],
                bloch[1],
                self.config.max_iter,
                max_ev_change,
                freq_min,
                freq_max,
                elapsed,
                n_locked
            );
        }

        let result = EigensolverResult {
            eigenvalues: all_eigenvalues[..n_bands_requested.min(all_eigenvalues.len())].to_vec(),
            iterations: self.iteration + 1,
            convergence,
            converged,
        };

        let diagnostics = recorder.finalize_with_result(self.iteration + 1, converged);
        DiagnosticResult {
            result,
            diagnostics,
        }
    }
}
