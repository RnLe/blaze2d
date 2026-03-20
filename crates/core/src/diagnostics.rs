//! Diagnostics and convergence tracking for the LOBPCG eigensolver.
//!
//! This module provides infrastructure for recording iteration-by-iteration
//! data during eigensolver runs, enabling detailed analysis of convergence
//! behavior under different configurations.
//!
//! # Overview
//!
//! The main components are:
//!
//! - [`ConvergenceRecorder`]: Accumulates per-iteration snapshots during a solve
//! - [`IterationSnapshot`]: Single iteration's eigenvalues, residuals, timing
//! - [`RunConfig`]: Configuration snapshot (resolution, tolerances, toggles)
//! - [`ConvergenceRun`]: Complete run data ready for serialization/plotting
//!
//! # Usage
//!
//! ```ignore
//! use mpb2d_core::diagnostics::{ConvergenceRecorder, RunConfig};
//! use mpb2d_core::eigensolver::EigensolverConfig;
//!
//! // Create a recorder with configuration metadata
//! let config = RunConfig::new("experiment_1")
//!     .with_resolution(24, 24)
//!     .with_eigensolver_config(&eigensolver_config);
//!
//! let mut recorder = ConvergenceRecorder::new(config);
//!
//! // The eigensolver will call recorder.record_iteration(...) each iteration
//! // After solve completes:
//! let run_data = recorder.finalize();
//!
//! // Export to JSON for Python analysis
//! let json = serde_json::to_string_pretty(&run_data)?;
//! ```
//!
//! # Python Integration
//!
//! The output JSON format is designed for easy loading in Python:
//!
//! ```python
//! import json
//! import matplotlib.pyplot as plt
//! import numpy as np
//!
//! with open('convergence_run.json') as f:
//!     data = json.load(f)
//!
//! # Plot residuals vs iterations for each band
//! for band_idx in range(data['config']['n_bands']):
//!     residuals = [snap['relative_residuals'][band_idx]
//!                  for snap in data['snapshots']]
//!     plt.semilogy(residuals, label=f'Band {band_idx}')
//! plt.xlabel('Iteration')
//! plt.ylabel('Relative Residual')
//! plt.legend()
//! ```

use std::time::Instant;

use serde::{Deserialize, Serialize};

// ============================================================================
// Preconditioner Type
// ============================================================================

/// Type of preconditioner used in the eigensolver.
///
/// # Background
///
/// For plane-wave photonic crystal solvers, the preconditioner approximates
/// the inverse of the Maxwell operator in Fourier space. The "transverse-projection"
/// preconditioner from Johnson's thesis projects onto divergence-free fields and
/// inverts the Laplacian symbol |k+G|².
///
/// For 2D scalar TE/TM problems, this simplifies to a scalar Laplacian inverse:
/// - **TM**: M⁻¹(q) = 1 / (|q|² + σ²)  (A = -Δ, no ε in operator)
/// - **TE**: M⁻¹(q) = ε_eff / (|q|² + σ²)  (A = -∇·(ε⁻¹∇))
///
/// The Fourier-diagonal kernel-compensated variant explicitly zeros the Γ-mode (q=0)
/// in Fourier space at the Γ-point only, relying on deflation to handle the null space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerType {
    /// Automatic selection based on polarization (default)
    ///
    /// - **TM**: Uses FourierDiagonalKernelCompensated (symmetry-compatible)
    /// - **TE**: Uses TransverseProjection (best condition reduction)
    Auto,
    /// No preconditioner (identity)
    None,
    /// Fourier-diagonal kernel-compensated preconditioner
    ///
    /// Uses M⁻¹(q) = ε_eff / (|q|² + σ²) with adaptive k-dependent shift σ².
    /// At the Γ-point, explicitly zeros the DC component (q=0) in Fourier space,
    /// relying on deflation to handle the null space. Away from Γ, uses only
    /// the regularization shift since |k|² > 0 provides natural regularization.
    ///
    /// - **TM**: M⁻¹(q) = 1 / (|q|² + σ²) for q ≠ 0, zero at q = 0 (Γ only)
    /// - **TE**: M⁻¹(q) = ε_eff / (|q|² + σ²) for q ≠ 0, zero at q = 0 (Γ only)
    ///
    /// Combines well with explicit Γ-mode deflation for robust convergence.
    /// Cost: 2 FFTs per application.
    FourierDiagonalKernelCompensated,
    /// MPB-style transverse-projection preconditioner
    ///
    /// The most effective preconditioner for TE mode with high dielectric contrast.
    /// Accounts for the spatial variation of ε(r) by computing:
    /// 1. Invert gradient in Fourier space: G = -i(k+G)/|k+G|² · r̂
    /// 2. Multiply by ε(r) in real space
    /// 3. Invert divergence in Fourier space
    ///
    /// For TM mode, falls back to the kernel-compensated Fourier-diagonal.
    ///
    /// Cost: 6 FFTs for TE (vs 2 for diagonal), but typically 5-10× fewer iterations.
    ///
    /// Based on Johnson & Joannopoulos, Optics Express 8, 173 (2001).
    TransverseProjection,
}

impl Default for PreconditionerType {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for PreconditionerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::None => write!(f, "none"),
            Self::FourierDiagonalKernelCompensated => {
                write!(f, "fourier_diagonal_kernel_compensated")
            }
            Self::TransverseProjection => write!(f, "transverse_projection"),
        }
    }
}

impl PreconditionerType {
    /// Resolve `Auto` to the appropriate preconditioner for the given polarization.
    ///
    /// - **TM**: Returns `FourierDiagonalKernelCompensated` (symmetry-compatible, works with transformed operator)
    /// - **TE**: Returns `TransverseProjection` (best condition reduction for ε-dependent operator)
    ///
    /// For non-Auto types, returns self unchanged.
    pub fn resolve_for_polarization(self, pol: crate::polarization::Polarization) -> Self {
        match self {
            Self::Auto => match pol {
                crate::polarization::Polarization::TM => Self::FourierDiagonalKernelCompensated,
                crate::polarization::Polarization::TE => Self::TransverseProjection,
            },
            other => other,
        }
    }
}

// ============================================================================
// Run Configuration (metadata for the run)
// ============================================================================

/// Configuration and metadata for a convergence run.
///
/// This captures all the settings that might affect convergence behavior,
/// allowing comparison between different configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Human-readable label for this run (e.g., "baseline", "no_preconditioner")
    pub label: String,

    // === Grid/Resolution ===
    /// Grid resolution in x direction
    pub nx: usize,
    /// Grid resolution in y direction  
    pub ny: usize,
    /// Physical size in x (lattice units)
    pub lx: f64,
    /// Physical size in y (lattice units)
    pub ly: f64,
    /// Mesh size (dx = lx/nx, dy = ly/ny) - computed for convenience
    pub mesh_size: [f64; 2],

    // === Eigensolver parameters ===
    /// Number of bands requested
    pub n_bands: usize,
    /// Maximum iterations allowed
    pub max_iter: usize,
    /// Convergence tolerance for relative residuals
    pub convergence_tol: f64,
    /// Locking tolerance (may differ from convergence tolerance)
    pub locking_tol: f64,
    /// Block size used (after auto-sizing)
    pub block_size: usize,

    // === Feature toggles ===
    /// Type of preconditioner used
    pub preconditioner_type: PreconditionerType,
    /// Whether W (history) directions are enabled
    pub w_history_enabled: bool,
    /// Whether warm-start from previous k-point is enabled
    pub warm_start_enabled: bool,
    /// Whether locking (deflation of converged bands) is enabled
    pub locking_enabled: bool,
    /// Whether Γ-point constant mode deflation is enabled (always recommended)
    pub gamma_deflation_enabled: bool,

    // === K-point info ===
    /// K-point index in the path (0-based)
    pub k_index: Option<usize>,
    /// K-point in fractional coordinates [kx, ky]
    pub k_point: Option<[f64; 2]>,
    /// Bloch wavevector [bx, by] (in 2π/a units)
    pub bloch: Option<[f64; 2]>,

    // === Polarization ===
    /// Polarization mode ("TM" or "TE")
    pub polarization: Option<String>,

    // === Additional metadata ===
    /// Any extra notes or description
    pub notes: Option<String>,
}

impl RunConfig {
    /// Create a new run configuration with a label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            nx: 0,
            ny: 0,
            lx: 1.0,
            ly: 1.0,
            mesh_size: [0.0, 0.0],
            n_bands: 0,
            max_iter: 0,
            convergence_tol: 0.0,
            locking_tol: 0.0,
            block_size: 0,
            preconditioner_type: PreconditionerType::Auto,
            w_history_enabled: true,
            warm_start_enabled: true,
            locking_enabled: true,
            gamma_deflation_enabled: true,
            k_index: None,
            k_point: None,
            bloch: None,
            polarization: None,
            notes: None,
        }
    }

    /// Set grid resolution.
    pub fn with_resolution(mut self, nx: usize, ny: usize) -> Self {
        self.nx = nx;
        self.ny = ny;
        // Update mesh size if dimensions are set
        if self.lx > 0.0 && nx > 0 {
            self.mesh_size[0] = self.lx / nx as f64;
        }
        if self.ly > 0.0 && ny > 0 {
            self.mesh_size[1] = self.ly / ny as f64;
        }
        self
    }

    /// Set physical dimensions.
    pub fn with_dimensions(mut self, lx: f64, ly: f64) -> Self {
        self.lx = lx;
        self.ly = ly;
        // Update mesh size if resolution is set
        if self.nx > 0 {
            self.mesh_size[0] = lx / self.nx as f64;
        }
        if self.ny > 0 {
            self.mesh_size[1] = ly / self.ny as f64;
        }
        self
    }

    /// Set eigensolver parameters from config.
    pub fn with_eigensolver_params(
        mut self,
        n_bands: usize,
        max_iter: usize,
        tol: f64,
        block_size: usize,
    ) -> Self {
        self.n_bands = n_bands;
        self.max_iter = max_iter;
        self.convergence_tol = tol;
        self.locking_tol = tol; // Default: same as convergence
        self.block_size = block_size;
        self
    }

    /// Set tolerances separately.
    pub fn with_tolerances(mut self, convergence_tol: f64, locking_tol: f64) -> Self {
        self.convergence_tol = convergence_tol;
        self.locking_tol = locking_tol;
        self
    }

    /// Set preconditioner type.
    pub fn with_preconditioner(mut self, precond_type: PreconditionerType) -> Self {
        self.preconditioner_type = precond_type;
        self
    }

    /// Set feature toggles (locking is always enabled).
    pub fn with_toggles(
        mut self,
        preconditioner_type: PreconditionerType,
        warm_start: bool,
    ) -> Self {
        self.preconditioner_type = preconditioner_type;
        self.w_history_enabled = true; // W history is always enabled
        self.warm_start_enabled = warm_start;
        self.locking_enabled = true; // Locking is always enabled
        self
    }

    /// Set k-point information.
    pub fn with_k_point(mut self, k_index: usize, k_point: [f64; 2], bloch: [f64; 2]) -> Self {
        self.k_index = Some(k_index);
        self.k_point = Some(k_point);
        self.bloch = Some(bloch);
        self
    }

    /// Set polarization.
    pub fn with_polarization(mut self, pol: &str) -> Self {
        self.polarization = Some(pol.to_string());
        self
    }

    /// Add notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Set gamma deflation toggle.
    pub fn with_gamma_deflation(mut self, enabled: bool) -> Self {
        self.gamma_deflation_enabled = enabled;
        self
    }
}

// ============================================================================
// Iteration Snapshot
// ============================================================================

/// Snapshot of eigensolver state at a single iteration.
///
/// This captures all the per-iteration data needed to analyze convergence:
/// - Current eigenvalue estimates (λ = ω²)
/// - Corresponding frequencies (ω = √λ)
/// - Residual norms (both absolute and relative)
/// - Timing information
/// - Subspace information (rank, locked count)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationSnapshot {
    /// Iteration number (0-based)
    pub iteration: usize,

    // === Eigenvalues ===
    /// Current eigenvalue estimates (λ = ω²) for active bands
    pub eigenvalues: Vec<f64>,
    /// Current frequency estimates (ω = √λ) for active bands
    pub frequencies: Vec<f64>,

    // === Residuals ===
    /// B-norm of residuals ||r_i||_B for each active band
    pub residual_b_norms: Vec<f64>,
    /// Relative residuals ||r_i||_B / |λ_i| for each active band
    pub relative_residuals: Vec<f64>,
    /// Maximum relative residual (worst-converged band)
    pub max_relative_residual: f64,
    /// Mean relative residual across active bands
    pub mean_relative_residual: f64,

    // === Convergence state ===
    /// Number of bands that have converged (active, not locked)
    pub n_converged: usize,
    /// Number of bands currently locked (deflated)
    pub n_locked: usize,
    /// Number of active bands still iterating
    pub n_active: usize,

    // === Subspace info ===
    /// Subspace dimension before SVQB (should be 2m or 3m)
    pub subspace_dim_input: usize,
    /// Subspace rank after SVQB (may be reduced due to linear dependence)
    pub subspace_rank_output: usize,
    /// Number of vectors dropped by SVQB
    pub svqb_dropped: usize,
    /// W block size (0 on first iteration)
    pub w_size: usize,

    // === Timing ===
    /// Cumulative wall-clock time from start of solve (seconds)
    pub elapsed_secs: f64,
}

impl IterationSnapshot {
    /// Create a new snapshot with minimal data.
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            eigenvalues: Vec::new(),
            frequencies: Vec::new(),
            residual_b_norms: Vec::new(),
            relative_residuals: Vec::new(),
            max_relative_residual: f64::INFINITY,
            mean_relative_residual: f64::INFINITY,
            n_converged: 0,
            n_locked: 0,
            n_active: 0,
            subspace_dim_input: 0,
            subspace_rank_output: 0,
            svqb_dropped: 0,
            w_size: 0,
            elapsed_secs: 0.0,
        }
    }

    /// Set eigenvalue data.
    pub fn with_eigenvalues(mut self, eigenvalues: Vec<f64>) -> Self {
        self.frequencies = eigenvalues
            .iter()
            .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
            .collect();
        self.eigenvalues = eigenvalues;
        self
    }

    /// Set residual data.
    pub fn with_residuals(mut self, b_norms: Vec<f64>, relative: Vec<f64>) -> Self {
        self.max_relative_residual = relative.iter().cloned().fold(0.0_f64, f64::max);
        self.mean_relative_residual = if relative.is_empty() {
            0.0
        } else {
            relative.iter().sum::<f64>() / relative.len() as f64
        };
        self.residual_b_norms = b_norms;
        self.relative_residuals = relative;
        self
    }

    /// Set convergence counts.
    pub fn with_convergence_counts(
        mut self,
        n_converged: usize,
        n_locked: usize,
        n_active: usize,
    ) -> Self {
        self.n_converged = n_converged;
        self.n_locked = n_locked;
        self.n_active = n_active;
        self
    }

    /// Set subspace information.
    pub fn with_subspace_info(
        mut self,
        dim_input: usize,
        rank_output: usize,
        dropped: usize,
        w_size: usize,
    ) -> Self {
        self.subspace_dim_input = dim_input;
        self.subspace_rank_output = rank_output;
        self.svqb_dropped = dropped;
        self.w_size = w_size;
        self
    }

    /// Set timing.
    pub fn with_elapsed(mut self, elapsed_secs: f64) -> Self {
        self.elapsed_secs = elapsed_secs;
        self
    }
}

// ============================================================================
// Convergence Recorder
// ============================================================================

/// Records per-iteration data during an eigensolver run.
///
/// Create one of these before starting a solve, then call `record_iteration`
/// at each step of the LOBPCG loop. After the solve completes, call
/// `finalize()` to get the complete run data.
#[derive(Debug, Clone)]
pub struct ConvergenceRecorder {
    /// Configuration metadata for this run
    config: RunConfig,
    /// Accumulated iteration snapshots
    snapshots: Vec<IterationSnapshot>,
    /// Start time for elapsed time tracking
    start_time: Option<Instant>,
    /// Whether recording is enabled
    enabled: bool,
}

impl ConvergenceRecorder {
    /// Create a new recorder with the given configuration.
    pub fn new(config: RunConfig) -> Self {
        Self {
            config,
            snapshots: Vec::new(),
            start_time: None,
            enabled: true,
        }
    }

    /// Create a disabled recorder (no-op for all operations).
    pub fn disabled() -> Self {
        Self {
            config: RunConfig::new("disabled"),
            snapshots: Vec::new(),
            start_time: None,
            enabled: false,
        }
    }

    /// Check if recording is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start the timer (call at beginning of solve).
    pub fn start(&mut self) {
        if self.enabled {
            self.start_time = Some(Instant::now());
        }
    }

    /// Get elapsed time since start (or 0 if not started).
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Record an iteration snapshot.
    pub fn record_iteration(&mut self, mut snapshot: IterationSnapshot) {
        if !self.enabled {
            return;
        }
        // Fill in elapsed time
        snapshot.elapsed_secs = self.elapsed_secs();
        self.snapshots.push(snapshot);
    }

    /// Get the number of recorded snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &RunConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut RunConfig {
        &mut self.config
    }

    /// Finalize the recording and return the complete run data.
    pub fn finalize(self) -> ConvergenceRun {
        let total_elapsed = self.elapsed_secs();
        ConvergenceRun {
            config: self.config,
            snapshots: self.snapshots,
            total_elapsed_secs: total_elapsed,
            final_iteration: 0, // Will be set by caller
            converged: false,   // Will be set by caller
        }
    }

    /// Finalize with final result information.
    pub fn finalize_with_result(self, final_iteration: usize, converged: bool) -> ConvergenceRun {
        let total_elapsed = self.elapsed_secs();
        ConvergenceRun {
            config: self.config,
            snapshots: self.snapshots,
            total_elapsed_secs: total_elapsed,
            final_iteration,
            converged,
        }
    }
}

// ============================================================================
// Complete Run Data
// ============================================================================

/// Complete data for a single eigensolver run.
///
/// This is the final output of the [`ConvergenceRecorder`], ready for
/// serialization to JSON and analysis in Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceRun {
    /// Configuration and metadata
    pub config: RunConfig,
    /// Per-iteration snapshots
    pub snapshots: Vec<IterationSnapshot>,
    /// Total wall-clock time for the solve
    pub total_elapsed_secs: f64,
    /// Final iteration count (1-based)
    pub final_iteration: usize,
    /// Whether the solve converged
    pub converged: bool,
}

impl ConvergenceRun {
    /// Get eigenvalue trajectory for a specific band.
    ///
    /// Returns (iterations, eigenvalues) where iterations[i] corresponds to
    /// eigenvalues[i].
    pub fn eigenvalue_trajectory(&self, band: usize) -> (Vec<usize>, Vec<f64>) {
        let mut iters = Vec::new();
        let mut values = Vec::new();
        for snap in &self.snapshots {
            if band < snap.eigenvalues.len() {
                iters.push(snap.iteration);
                values.push(snap.eigenvalues[band]);
            }
        }
        (iters, values)
    }

    /// Get frequency trajectory for a specific band.
    pub fn frequency_trajectory(&self, band: usize) -> (Vec<usize>, Vec<f64>) {
        let mut iters = Vec::new();
        let mut values = Vec::new();
        for snap in &self.snapshots {
            if band < snap.frequencies.len() {
                iters.push(snap.iteration);
                values.push(snap.frequencies[band]);
            }
        }
        (iters, values)
    }

    /// Get relative residual trajectory for a specific band.
    pub fn residual_trajectory(&self, band: usize) -> (Vec<usize>, Vec<f64>) {
        let mut iters = Vec::new();
        let mut values = Vec::new();
        for snap in &self.snapshots {
            if band < snap.relative_residuals.len() {
                iters.push(snap.iteration);
                values.push(snap.relative_residuals[band]);
            }
        }
        (iters, values)
    }

    /// Get max residual trajectory (worst band at each iteration).
    pub fn max_residual_trajectory(&self) -> (Vec<usize>, Vec<f64>) {
        self.snapshots
            .iter()
            .map(|s| (s.iteration, s.max_relative_residual))
            .unzip()
    }

    /// Get mean residual trajectory.
    pub fn mean_residual_trajectory(&self) -> (Vec<usize>, Vec<f64>) {
        self.snapshots
            .iter()
            .map(|s| (s.iteration, s.mean_relative_residual))
            .unzip()
    }
}

// ============================================================================
// Multi-Run Collection
// ============================================================================

/// Collection of multiple convergence runs for comparison.
///
/// Useful for comparing different configurations (e.g., with/without
/// preconditioner) or different k-points in a band structure calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStudy {
    /// Study name/description
    pub name: String,
    /// Individual runs in this study
    pub runs: Vec<ConvergenceRun>,
}

impl ConvergenceStudy {
    /// Create a new study with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            runs: Vec::new(),
        }
    }

    /// Add a run to the study.
    pub fn add_run(&mut self, run: ConvergenceRun) {
        self.runs.push(run);
    }

    /// Get runs matching a label pattern.
    pub fn runs_by_label(&self, pattern: &str) -> Vec<&ConvergenceRun> {
        self.runs
            .iter()
            .filter(|r| r.config.label.contains(pattern))
            .collect()
    }

    /// Get all unique labels.
    pub fn labels(&self) -> Vec<&str> {
        let mut labels: Vec<&str> = self.runs.iter().map(|r| r.config.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        labels
    }

    /// Save to JSON file.
    pub fn save_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Load from JSON file.
    pub fn load_json(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_json::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save iteration data to CSV file for easy plotting.
    ///
    /// Format: One file per k-point, columns are:
    /// `iteration,elapsed_secs,band1_eigenvalue,band1_frequency,band1_residual,band2_...`
    ///
    /// If `base_path` is `/path/to/study`, outputs will be:
    /// - `/path/to/study_k000.csv`
    /// - `/path/to/study_k001.csv`
    /// - etc.
    pub fn save_iteration_csv(&self, base_path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;

        for run in &self.runs {
            // Determine k-point index from label (format: "study_k###")
            let k_suffix = run
                .config
                .k_index
                .map(|idx| format!("_k{:03}", idx))
                .unwrap_or_default();

            // Build output path
            let mut path = base_path.to_path_buf();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("study");
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("csv");
            path.set_file_name(format!("{}{}.{}", stem, k_suffix, ext));

            let file = std::fs::File::create(&path)?;
            let mut writer = std::io::BufWriter::new(file);

            // Determine max bands from snapshots
            let max_bands = run
                .snapshots
                .iter()
                .map(|s| s.eigenvalues.len())
                .max()
                .unwrap_or(0);

            // Write header
            write!(writer, "iteration,elapsed_secs")?;
            for b in 0..max_bands {
                write!(
                    writer,
                    ",band{}_eigenvalue,band{}_frequency,band{}_residual",
                    b + 1,
                    b + 1,
                    b + 1
                )?;
            }
            writeln!(writer)?;

            // Write data rows
            for snap in &run.snapshots {
                write!(writer, "{},{:.6}", snap.iteration, snap.elapsed_secs)?;
                for b in 0..max_bands {
                    let ev = snap.eigenvalues.get(b).copied().unwrap_or(f64::NAN);
                    let freq = snap.frequencies.get(b).copied().unwrap_or(f64::NAN);
                    let res = snap.relative_residuals.get(b).copied().unwrap_or(f64::NAN);
                    write!(writer, ",{:.12e},{:.12e},{:.12e}", ev, freq, res)?;
                }
                writeln!(writer)?;
            }

            writer.flush()?;
        }

        Ok(())
    }

    /// Save aggregated iteration data to a single CSV file.
    ///
    /// Format: All k-points in one file, columns are:
    /// `k_index,iteration,elapsed_secs,band1_eigenvalue,band1_frequency,band1_residual,...`
    pub fn save_iteration_csv_combined(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;

        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        // Determine max bands across all runs
        let max_bands = self
            .runs
            .iter()
            .flat_map(|r| r.snapshots.iter())
            .map(|s| s.eigenvalues.len())
            .max()
            .unwrap_or(0);

        // Write header
        write!(writer, "k_index,iteration,elapsed_secs")?;
        for b in 0..max_bands {
            write!(
                writer,
                ",band{}_eigenvalue,band{}_frequency,band{}_residual",
                b + 1,
                b + 1,
                b + 1
            )?;
        }
        writeln!(writer)?;

        // Write data rows
        for run in &self.runs {
            let k_idx = run.config.k_index.unwrap_or(0);
            for snap in &run.snapshots {
                write!(
                    writer,
                    "{},{},{:.6}",
                    k_idx, snap.iteration, snap.elapsed_secs
                )?;
                for b in 0..max_bands {
                    let ev = snap.eigenvalues.get(b).copied().unwrap_or(f64::NAN);
                    let freq = snap.frequencies.get(b).copied().unwrap_or(f64::NAN);
                    let res = snap.relative_residuals.get(b).copied().unwrap_or(f64::NAN);
                    write!(writer, ",{:.12e},{:.12e},{:.12e}", ev, freq, res)?;
                }
                writeln!(writer)?;
            }
        }

        writer.flush()
    }
}

// ============================================================================
// Builder for diagnostic config from EigensolverConfig
// ============================================================================

/// Extension trait for building RunConfig from eigensolver types.
///
/// This allows easy creation of RunConfig from existing configuration objects.
pub trait IntoRunConfig {
    /// Convert to a RunConfig with the given label.
    fn into_run_config(&self, label: impl Into<String>) -> RunConfig;
}
