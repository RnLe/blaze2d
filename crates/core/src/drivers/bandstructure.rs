//! Band-structure computation for 2D photonic crystals.
//!
//! This module provides the high-level orchestration for computing photonic band
//! structures. A band structure is the dispersion relation ω(k) that describes
//! how the frequency of electromagnetic modes varies with the Bloch wavevector k.
//!
//! # Overview
//!
//! The band-structure calculation proceeds as follows:
//!
//! 1. **Dielectric preparation**: Sample the geometry onto the computational grid
//!    to create the dielectric function ε(r).
//!
//! 2. **K-point loop**: For each k-point along the high-symmetry path:
//!    - Construct the Maxwell operator Θ with Bloch boundary conditions
//!    - Build a preconditioner for faster convergence
//!    - Run the eigensolver to find the lowest eigenvalues (ω²) and modes
//!    - Optionally use warm-start from the previous k-point
//!
//! 3. **Result collection**: Collect all eigenfrequencies into a band structure.
//!
//! # Physical Background
//!
//! In a periodic dielectric structure, the electromagnetic modes satisfy:
//!
//! ```text
//! ∇ × (ε⁻¹ ∇ × H) = (ω/c)² H     (master equation for H-field)
//! ```
//!
//! With Bloch boundary conditions, H(r + R) = e^{ik·R} H(r), where R is any
//! lattice vector. The eigenvalue problem becomes:
//!
//! ```text
//! Θ_k H_n,k = (ω_n,k / c)² H_n,k
//! ```
//!
//! where n is the band index and k is the Bloch wavevector.
//!
//! # Usage
//!
//! ```ignore
//! use mpb2d_core::bandstructure::{run, BandStructureJob, Verbosity};
//!
//! let result = run(backend, &job, Verbosity::Verbose);
//! // result.bands[k_index][band_index] gives ω for each (k, band) pair
//! ```

use log::{debug, info, warn};
use std::time::Instant;

use crate::{
    backend::SpectralBackend,
    band_tracking::{apply_permutation, track_bands_with_frequencies},
    diagnostics::{ConvergenceStudy, PreconditionerType},
    dielectric::{Dielectric2D, DielectricOptions},
    eigensolver::{Eigensolver, EigensolverConfig, SubspaceHistory},
    field::Field2D,
    geometry::Geometry2D,
    grid::Grid2D,
    operators::ThetaOperator,
    polarization::Polarization,
};

// ============================================================================
// Γ-Point Detection
// ============================================================================

/// Threshold for detecting Γ-point (k ≈ 0).
///
/// At the Γ point, the Maxwell operator becomes singular with a null space
/// (the DC/constant mode with ω = 0). This causes numerical issues and
/// produces unreliable eigenvectors for band tracking.
const GAMMA_THRESHOLD: f64 = 1e-8;

/// Check if a k-point is at or very near the Γ point (k = 0).
///
/// Returns true if |k|² < GAMMA_THRESHOLD.
#[inline]
fn is_gamma_point(k_frac: [f64; 2]) -> bool {
    let k_norm_sq = k_frac[0] * k_frac[0] + k_frac[1] * k_frac[1];
    k_norm_sq < GAMMA_THRESHOLD
}

// ============================================================================
// Job Configuration
// ============================================================================

/// Configuration for a band-structure calculation.
///
/// This struct bundles all the parameters needed to compute a photonic band
/// structure, including geometry, grid resolution, polarization, k-path,
/// and eigensolver settings.
///
/// # Fields
///
/// - `geom`: The 2D geometry (lattice + atoms)
/// - `grid`: The computational grid resolution
/// - `pol`: Polarization mode (TM or TE)
/// - `k_path`: List of k-points in fractional coordinates
/// - `eigensolver`: Configuration for the LOBPCG eigensolver
/// - `dielectric`: Dielectric smoothing options
#[derive(Debug, Clone)]
pub struct BandStructureJob {
    /// The 2D photonic crystal geometry.
    pub geom: Geometry2D,
    /// Computational grid (Nx × Ny points, Lx × Ly physical size).
    pub grid: Grid2D,
    /// Polarization: TM (E-field out of plane) or TE (H-field out of plane).
    pub pol: Polarization,
    /// Path through the Brillouin zone in fractional coordinates.
    /// Each entry is [kx, ky] where kx, ky ∈ [0, 1).
    pub k_path: Vec<[f64; 2]>,
    /// Configuration for the eigensolver.
    pub eigensolver: EigensolverConfig,
    /// Dielectric function options (smoothing, etc.).
    pub dielectric: DielectricOptions,
}

// ============================================================================
// Result
// ============================================================================

/// Result of a band-structure calculation.
///
/// Contains the k-path, accumulated distances along the path (for plotting),
/// and the computed eigenfrequencies organized by k-point.
#[derive(Debug, Clone)]
pub struct BandStructureResult {
    /// The k-path used for the calculation (fractional coordinates).
    pub k_path: Vec<[f64; 2]>,
    /// Cumulative distance along the k-path (for plotting).
    /// `distances[i]` is the distance from k_path[0] to k_path[i].
    pub distances: Vec<f64>,
    /// Computed eigenfrequencies ω (not ω²) organized as bands[k_index][band].
    pub bands: Vec<Vec<f64>>,
}

// ============================================================================
// Verbosity Control
// ============================================================================

/// Controls the verbosity of progress output during band-structure computation.
///
/// **Note:** This is now deprecated. Use the `log` crate with appropriate log levels instead.
/// The bandstructure module now uses `log::info!`, `log::debug!`, and `log::warn!`
/// for all output. Configure your log filter to control verbosity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// No progress output (deprecated: use log filter instead).
    Quiet,
    /// Print progress information (deprecated: use log filter instead).
    Verbose,
}

impl Verbosity {
    /// Returns true if verbose output is enabled.
    #[allow(dead_code)]
    fn enabled(self) -> bool {
        matches!(self, Verbosity::Verbose)
    }
}

/// Options for band structure computation.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Preconditioner type (None, FourierDiagonalKernelCompensated, TransverseProjection).
    pub precond_type: PreconditionerType,
    /// Enable subspace prediction for accelerated warm-start.
    ///
    /// When enabled, uses complex eigenvector overlaps and polar decomposition to
    /// compute a rotation matrix that aligns the previous subspace to the current one.
    /// This provides a better warm-start than simple copying, especially when bands
    /// reorder between k-points.
    pub use_subspace_prediction: bool,
    /// Enable linear extrapolation in subspace prediction (Stage 2).
    ///
    /// When enabled (and `use_subspace_prediction` is also enabled), predicts the
    /// next k-point's eigenvectors by extrapolating from the aligned subspace:
    /// X_pred = (1+α)X̃_n - α X_{n-1}
    ///
    /// This is automatically disabled at k-path corners and near degeneracies.
    pub use_extrapolation: bool,
    /// Enable band-window-based preconditioner shift.
    ///
    /// When enabled, uses eigenvalues from the previous k-point to compute a
    /// band-window-informed shift σ² for the preconditioner. This can improve
    /// convergence by tuning the preconditioner to the actual spectral range
    /// of the bands being computed.
    ///
    /// The shift is computed as:
    ///   σ² = β · σ²_adaptive + (1-β) · c · λ_median
    /// where λ_median is the median eigenvalue from the previous k-point,
    /// c is a band-scale factor (default 0.5), and β is the blend factor (default 0.5).
    ///
    /// For the first k-point (no previous eigenvalues), falls back to adaptive shift.
    pub use_band_window_shift: bool,
    /// Blend factor β for band-window shift (0.0 = pure band-window, 1.0 = pure adaptive).
    ///
    /// Only used when `use_band_window_shift` is enabled.
    pub band_window_blend: f64,
    /// Scale factor c for band-window eigenvalue contribution.
    ///
    /// The band-window shift component is c × λ_median. Default is 0.5.
    /// Only used when `use_band_window_shift` is enabled.
    pub band_window_scale: f64,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            precond_type: PreconditionerType::default(),
            use_subspace_prediction: true, // Stage 1: rotation-based warm-start
            use_extrapolation: true,       // Stage 2: linear extrapolation
            use_band_window_shift: false,  // Disabled by default (experimental)
            band_window_blend: 0.5,        // Equal mix of adaptive and band-window
            band_window_scale: 0.5,        // Conservative scaling of median eigenvalue
        }
    }
}

impl RunOptions {
    /// Create default run options (fourier-diagonal preconditioner with adaptive shift).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the preconditioner type.
    pub fn with_preconditioner(mut self, precond_type: PreconditionerType) -> Self {
        self.precond_type = precond_type;
        self
    }

    /// Enable subspace prediction for accelerated warm-start.
    ///
    /// This uses rotation-based subspace tracking to provide better initial
    /// guesses for the eigensolver at each k-point.
    pub fn with_subspace_prediction(mut self, enabled: bool) -> Self {
        self.use_subspace_prediction = enabled;
        self
    }

    /// Enable linear extrapolation in subspace prediction (Stage 2).
    ///
    /// When enabled, predicts eigenvectors by extrapolating along the k-path.
    /// Automatically disabled at corners and near degeneracies.
    pub fn with_extrapolation(mut self, enabled: bool) -> Self {
        self.use_extrapolation = enabled;
        self
    }

    /// Enable band-window-based preconditioner shift (experimental).
    ///
    /// When enabled, uses eigenvalues from the previous k-point to tune the
    /// preconditioner shift to the spectral range of the bands being computed.
    pub fn with_band_window_shift(mut self, enabled: bool) -> Self {
        self.use_band_window_shift = enabled;
        self
    }

    /// Set the blend factor for band-window shift.
    ///
    /// The blend factor β controls how much weight is given to the adaptive shift
    /// versus the band-window shift: σ² = β·σ²_adaptive + (1-β)·σ²_band_window.
    ///
    /// - β = 0.0: Pure band-window shift (uses eigenvalue median)
    /// - β = 1.0: Pure adaptive shift (uses smin²)
    /// - β = 0.5 (default): Equal blend
    pub fn with_band_window_blend(mut self, blend: f64) -> Self {
        self.band_window_blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Set the scale factor for band-window eigenvalue contribution.
    ///
    /// The band-window shift component is c × λ_median, where c is this scale factor.
    /// Lower values (e.g., 0.25) are more conservative, higher values (e.g., 1.0)
    /// give more weight to the eigenvalue spectrum.
    pub fn with_band_window_scale(mut self, scale: f64) -> Self {
        self.band_window_scale = scale.max(0.0);
        self
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

// ============================================================================
// Main Entry Point
// ============================================================================

/// Compute the photonic band structure.
///
/// This is the main entry point for band-structure calculations. It iterates
/// over all k-points in the job's k_path, solving the eigenvalue problem at
/// each point to obtain the photonic band frequencies.
///
/// Uses default options: fourier-diagonal preconditioner with adaptive k-dependent shift.
///
/// # Arguments
///
/// - `backend`: The spectral backend (CPU, CUDA, etc.)
/// - `job`: The band-structure job configuration
/// - `verbosity`: Controls progress output
///
/// # Returns
///
/// A `BandStructureResult` containing the k-path, path distances, and
/// computed eigenfrequencies for each k-point and band.
///
/// # Algorithm
///
/// For each k-point:
/// 1. Construct the Θ operator with Bloch wavevector k
/// 2. Build the preconditioner M^{-1}
/// 3. Create and run the eigensolver
/// 4. Extract eigenfrequencies ω = √λ from eigenvalues λ = ω²
/// 5. Store eigenvectors for warm-starting the next k-point
pub fn run<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
) -> BandStructureResult {
    run_with_options(backend, job, verbosity, RunOptions::default())
}

/// Compute the photonic band structure with custom options.
///
/// Like [`run`] but allows customization of preconditioner type and shift mode.
pub fn run_with_options<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    _verbosity: Verbosity,
    options: RunOptions,
) -> BandStructureResult {
    // ========================================================================
    // Setup Phase
    // ========================================================================

    // Resolve Auto preconditioner type based on polarization
    let precond_type = options.precond_type.resolve_for_polarization(job.pol);

    // Compute reciprocal lattice for proper k-space coordinate conversion.
    // This is essential for non-orthogonal lattices (hexagonal, oblique) where
    // the fractional k-coordinates must be transformed using the reciprocal
    // lattice vectors, not just multiplied by 2π.
    let reciprocal = job.geom.lattice.reciprocal();

    info!(
        "[bandstructure] grid={}x{} pol={:?} bands={} k_points={} lattice={:?}",
        job.grid.nx,
        job.grid.ny,
        job.pol,
        job.eigensolver.n_bands,
        job.k_path.len(),
        job.geom.lattice.classify(),
    );

    // Sample the dielectric function from geometry
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    // Compute dielectric contrast for diagnostics
    let eps = dielectric.eps();
    let eps_min = eps.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_contrast = if eps_min > 1e-15 {
        eps_max / eps_min
    } else {
        f64::INFINITY
    };

    info!(
        "[bandstructure] dielectric: ε=[{:.3}, {:.3}] contrast={:.1}x precond={:?}",
        eps_min, eps_max, eps_contrast, precond_type,
    );

    // ========================================================================
    // Operator Diagnostics (one-time, before k-point loop)
    // ========================================================================
    // Use a non-Γ k-point for condition number estimates (Γ has κ → ∞ due to DC mode)
    {
        // Find first non-Γ k-point for diagnostics
        let diag_k_idx = job.k_path.iter().position(|&k| !is_gamma_point(k));
        if let Some(idx) = diag_k_idx {
            let k_frac = job.k_path[idx];
            // Convert fractional k-coordinates to Cartesian Bloch wavevector
            // using the reciprocal lattice (essential for non-orthogonal lattices)
            let bloch = reciprocal.fractional_to_cartesian(k_frac);

            // Create temporary operator for diagnostics
            let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch);

            // Build preconditioner for diagnostics (always using adaptive shift)
            let mut preconditioner_opt: Option<
                Box<dyn crate::preconditioners::OperatorPreconditioner<B>>,
            > = match precond_type {
                PreconditionerType::Auto => {
                    unreachable!("Auto should be resolved before this point")
                }
                PreconditionerType::None => None,
                PreconditionerType::FourierDiagonalKernelCompensated => {
                    Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                }
                PreconditionerType::TransverseProjection => Some(Box::new(
                    theta.build_transverse_projection_preconditioner_adaptive(),
                )),
            };

            const POWER_ITERATIONS: usize = 15;

            // Self-adjointness check
            let self_adj_err = theta.check_self_adjointness();
            let self_adj_status = if self_adj_err < 1e-12 {
                "exact"
            } else if self_adj_err < 1e-6 {
                "good"
            } else {
                "WARN"
            };

            // Condition number estimates
            let (lambda_max, lambda_min, kappa) = theta.estimate_condition_number(POWER_ITERATIONS);

            if let Some(ref mut precond) = preconditioner_opt {
                let (_pm_max, _pm_min, pm_kappa) = theta
                    .estimate_preconditioned_condition_number(&mut **precond, POWER_ITERATIONS);
                let reduction = kappa / pm_kappa;
                info!(
                    "[bandstructure] κ(A)≈{:.1} κ(M⁻¹A)≈{:.1} ({:.1}x reduction) self-adjoint={} ({:.1e})",
                    kappa, pm_kappa, reduction, self_adj_status, self_adj_err
                );
            } else {
                info!(
                    "[bandstructure] κ(A)≈{:.1} (λ_max≈{:.2e}, λ_min≈{:.2e}) self-adjoint={} ({:.1e})",
                    kappa, lambda_max, lambda_min, self_adj_status, self_adj_err
                );
            }
        }
    }

    // ========================================================================
    // K-Point Loop
    // ========================================================================

    // Storage for warm-start vectors from previous k-point (simple mode)
    let warm_start_limit = job.eigensolver.n_bands;
    let mut warm_start_store: Vec<Field2D> = Vec::new();

    // Subspace prediction history (rotation-based warm-start)
    let mut subspace_history: Option<SubspaceHistory> = if options.use_subspace_prediction {
        let mut history = SubspaceHistory::new(warm_start_limit);
        if options.use_extrapolation {
            history.enable_extrapolation();
            info!("[bandstructure] Subspace prediction enabled (rotation + extrapolation)");
        } else {
            info!("[bandstructure] Subspace prediction enabled (rotation-only)");
        }
        Some(history)
    } else {
        None
    };

    // Storage for band tracking: eigenvectors from previous k-point
    // Note: When starting at Γ with proper deflation, Γ eigenvectors are reliable
    // and can be used for warm-starting subsequent k-points.
    let mut prev_eigenvectors: Option<Vec<Field2D>> = None;

    // Storage for previous frequencies (ω) for degenerate block disambiguation
    // These are used by track_bands_with_frequencies to break ties when
    // singular values indicate near-degenerate bands.
    let mut prev_omegas: Option<Vec<f64>> = None;

    // Storage for previous eigenvalues (ω² = λ) for band-window preconditioner shift
    // These are used to inform the preconditioner shift at the next k-point.
    let mut prev_eigenvalues: Option<Vec<f64>> = None;

    // Log if band-window shift is enabled
    if options.use_band_window_shift {
        info!(
            "[bandstructure] Band-window shift enabled (blend={:.2}, scale={:.2})",
            options.band_window_blend, options.band_window_scale
        );
    }

    // Get dielectric epsilon for B-weighted overlaps
    // - TE mode: B = I, use standard inner product (None)
    // - TM mode (generalized): B = ε, use ε-weighted inner product
    let eps_for_tracking: Option<Vec<f64>> = if job.pol == Polarization::TM {
        Some(dielectric.eps().to_vec())
    } else {
        None
    };

    // Detect if we can reuse the first Γ-point result for the last k-point
    // This avoids redundant computation when the path is e.g., Γ→X→M→Γ
    let first_is_gamma = job.k_path.first().map_or(false, |&k| is_gamma_point(k));
    let last_is_gamma = job.k_path.last().map_or(false, |&k| is_gamma_point(k));
    let reuse_gamma = first_is_gamma && last_is_gamma && job.k_path.len() > 1;
    let last_k_idx = job.k_path.len().saturating_sub(1);

    // Storage for first Γ-point frequencies (to reuse for last k-point if applicable)
    let mut first_gamma_omegas: Option<Vec<f64>> = None;

    // Accumulate results
    let mut bands: Vec<Vec<f64>> = Vec::with_capacity(job.k_path.len());
    let mut total_iterations = 0usize;

    // Start timing the solve phase (excludes setup/diagnostics)
    let solve_start = Instant::now();

    for (k_idx, &k_frac) in job.k_path.iter().enumerate() {
        // Check if this is a Γ-point (k ≈ 0)
        // At Γ, the constant mode (DC) is deflated by the eigensolver.
        let is_gamma = is_gamma_point(k_frac);

        // Reuse first Γ-point result for the last k-point (avoid duplicate solve)
        // This is valid when the path loops back to Γ (e.g., Γ→X→M→Γ)
        if reuse_gamma && k_idx == last_k_idx {
            if let Some(ref gamma_omegas) = first_gamma_omegas {
                debug!(
                    "[bandstructure] k#{:03} is duplicate Γ-point: reusing result from k#000",
                    k_idx
                );
                bands.push(gamma_omegas.clone());
                // Don't update warm-start or tracking state - not needed for last point
                continue;
            }
        }

        if is_gamma {
            debug!(
                "[bandstructure] k#{:03} is Γ-point: constant mode will be deflated",
                k_idx
            );
        }

        // Convert fractional k-point to Cartesian Bloch wavevector using the
        // reciprocal lattice. This is essential for non-orthogonal lattices
        // (hexagonal, oblique) where M=(0.5,0) in fractional coordinates does
        // NOT map to [π, 0] in Cartesian k-space.
        let bloch = reciprocal.fractional_to_cartesian(k_frac);

        // Prepare warm-start slice (if available from previous k-point)
        // Use subspace prediction if enabled, otherwise fall back to simple copy
        let predicted_vectors: Option<Vec<Field2D>>;
        let warm_slice: Option<&[Field2D]> = if let Some(ref mut history) = subspace_history {
            // Use subspace prediction for warm-start
            let prediction = history.predict(k_frac, eps_for_tracking.as_deref());
            if prediction.has_prediction() {
                debug!(
                    "[bandstructure] k#{:03} using {:?} prediction (σ_min={:.4})",
                    k_idx, prediction.method_used, prediction.singular_value_min
                );
                predicted_vectors = Some(prediction.predicted_vectors);
                predicted_vectors.as_deref()
            } else {
                predicted_vectors = None;
                None
            }
        } else if !warm_start_store.is_empty() {
            // Simple copy warm-start (original behavior)
            predicted_vectors = None;
            Some(warm_start_store.as_slice())
        } else {
            predicted_vectors = None;
            None
        };

        // Configure eigensolver
        let eigensolver_config = {
            let mut cfg = job.eigensolver.clone();
            cfg.k_index = Some(k_idx);
            cfg
        };

        // Construct the Maxwell operator Θ for this k-point
        let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch);

        // Build preconditioner for this operator
        // Use band-window shift if enabled AND we have previous eigenvalues,
        // otherwise fall back to adaptive shift.
        let use_band_window = options.use_band_window_shift && prev_eigenvalues.is_some();
        let mut preconditioner_opt: Option<
            Box<dyn crate::preconditioners::OperatorPreconditioner<B>>,
        > = match precond_type {
            PreconditionerType::Auto => unreachable!("Auto should be resolved before this point"),
            PreconditionerType::None => None,
            PreconditionerType::FourierDiagonalKernelCompensated => {
                if use_band_window {
                    // Use eigenvalues from previous k-point to tune the preconditioner
                    let eigenvalues = prev_eigenvalues.as_ref().unwrap();
                    Some(Box::new(
                        theta.build_homogeneous_preconditioner_band_window(
                            eigenvalues,
                            Some(options.band_window_blend),
                            Some(options.band_window_scale),
                        ),
                    ))
                } else {
                    Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                }
            }
            PreconditionerType::TransverseProjection => {
                if use_band_window {
                    let eigenvalues = prev_eigenvalues.as_ref().unwrap();
                    Some(Box::new(
                        theta.build_transverse_projection_preconditioner_band_window(
                            eigenvalues,
                            Some(options.band_window_blend),
                            Some(options.band_window_scale),
                        ),
                    ))
                } else {
                    Some(Box::new(
                        theta.build_transverse_projection_preconditioner_adaptive(),
                    ))
                }
            }
        };

        // Create and run the eigensolver
        let mut solver = Eigensolver::new(
            &mut theta,
            eigensolver_config,
            preconditioner_opt
                .as_mut()
                .map(|p| &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>),
            warm_slice,
        );

        let result = solver.solve();
        let k_iterations = result.iterations;

        // Convert eigenvalues (ω²) to frequencies (ω)
        let mut omegas: Vec<f64> = result
            .eigenvalues
            .iter()
            .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
            .collect();

        // Use all_eigenvectors() to include deflated vectors (e.g., Γ constant mode)
        let mut eigenvectors = solver.all_eigenvectors();

        total_iterations += k_iterations;

        // ====================================================================
        // Band Tracking: reorder by overlap with previous k-point
        // Uses polar decomposition + frequency continuity for degenerate blocks.
        // ====================================================================
        if let (Some(prev_vecs), Some(prev_freqs)) = (&prev_eigenvectors, &prev_omegas) {
            let tracking_result = track_bands_with_frequencies(
                prev_vecs,
                &eigenvectors,
                prev_freqs,
                &omegas,
                eps_for_tracking.as_deref(),
            );

            // Always log σ_min for diagnostic purposes
            let perm_str = if tracking_result.had_swaps {
                format!("{:?}", tracking_result.permutation)
            } else {
                "identity".to_string()
            };
            let blocks_str = if tracking_result.degenerate_blocks.is_empty() {
                String::new()
            } else {
                format!(" blocks={:?}", tracking_result.degenerate_blocks)
            };
            info!(
                "[band_tracking] k#{:03} σ_min={:.6} perm={}{}",
                k_idx, tracking_result.sigma_min, perm_str, blocks_str
            );

            if tracking_result.had_swaps {
                apply_permutation(&tracking_result.permutation, &mut omegas, &mut eigenvectors);

                // Log warnings for near-degenerate regions (low sigma_min)
                if tracking_result.sigma_min < 0.1 {
                    warn!(
                        "[bandstructure] k#{:03} band tracking: near-degeneracy (σ_min={:.4}), may be unreliable",
                        k_idx, tracking_result.sigma_min
                    );
                }
            }
        }

        // Store eigenvectors for band tracking at next k-point
        prev_eigenvectors = Some(eigenvectors.clone());

        // Store frequencies (ω) for degenerate block disambiguation at next k-point
        prev_omegas = Some(omegas.clone());

        // Store eigenvalues (ω²) for band-window preconditioner shift at next k-point
        // We store the raw eigenvalues (λ = ω²), not frequencies.
        prev_eigenvalues = Some(result.eigenvalues.clone());

        // Update subspace history or warm-start store for next k-point
        if let Some(ref mut history) = subspace_history {
            // Update subspace history with converged eigenvectors
            history.update(k_frac, &eigenvectors);
        } else {
            // Simple copy for traditional warm-start
            warm_start_store.clear();
            for vec in eigenvectors.iter().take(warm_start_limit) {
                warm_start_store.push(vec.clone());
            }
        }

        // Store first Γ-point result for potential reuse at end of path
        if reuse_gamma && k_idx == 0 && is_gamma {
            first_gamma_omegas = Some(omegas.clone());
        }

        bands.push(omegas);
    } // end k-point loop

    // ========================================================================
    // Finalize
    // ========================================================================

    let solve_elapsed = solve_start.elapsed().as_secs_f64();
    let time_per_iter_ms = if total_iterations > 0 {
        solve_elapsed * 1000.0 / total_iterations as f64
    } else {
        0.0
    };

    info!(
        "[bandstructure] complete: {} k-points, {} total iterations, {:.2}s ({:.1}ms/iter)",
        job.k_path.len(),
        total_iterations,
        solve_elapsed,
        time_per_iter_ms
    );

    // Build initial result
    let result = BandStructureResult {
        k_path: job.k_path.clone(),
        distances: compute_k_path_distances(&job.k_path),
        bands,
    };

    // Rotate output to start from Γ-point (if path was internally rotated)
    rotate_result_to_gamma(result)
}

// ============================================================================
// Band Structure with Diagnostics
// ============================================================================

/// Result of a band-structure calculation with convergence diagnostics.
#[derive(Debug, Clone)]
pub struct BandStructureResultWithDiagnostics {
    /// The standard band structure result (k-path, distances, bands).
    pub result: BandStructureResult,
    /// Convergence study containing per-k-point diagnostic data.
    pub study: ConvergenceStudy,
}

/// Compute the photonic band structure with full convergence diagnostics.
///
/// This is similar to [`run`] but additionally records per-iteration data
/// for each k-point solve, producing a [`ConvergenceStudy`] that can be
/// serialized to JSON for analysis.
///
/// # Arguments
///
/// - `backend`: The spectral backend (CPU, CUDA, etc.)
/// - `job`: The band-structure job configuration
/// - `verbosity`: Controls progress output
/// - `study_name`: Name for the convergence study
/// - `precond_type`: Type of preconditioner to use (None, FourierDiagonalKernelCompensated, TransverseProjection)
///
/// # Returns
///
/// A [`BandStructureResultWithDiagnostics`] containing both the band structure
/// and the full convergence study data.
pub fn run_with_diagnostics<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    study_name: impl Into<String>,
    precond_type: PreconditionerType,
) -> BandStructureResultWithDiagnostics {
    run_with_diagnostics_and_options(
        backend,
        job,
        verbosity,
        study_name,
        RunOptions::new().with_preconditioner(precond_type),
    )
}

/// Compute the photonic band structure with full convergence diagnostics and custom options.
///
/// Like [`run_with_diagnostics`] but uses RunOptions for configuration.
pub fn run_with_diagnostics_and_options<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    _verbosity: Verbosity,
    study_name: impl Into<String>,
    options: RunOptions,
) -> BandStructureResultWithDiagnostics {
    // ========================================================================
    // Setup Phase
    // ========================================================================

    let study_name = study_name.into();

    // Resolve Auto preconditioner type based on polarization
    let precond_type = options.precond_type.resolve_for_polarization(job.pol);

    // Compute reciprocal lattice for proper k-space coordinate conversion.
    // This is essential for non-orthogonal lattices (hexagonal, oblique) where
    // the fractional k-coordinates must be transformed using the reciprocal
    // lattice vectors, not just multiplied by 2π.
    let reciprocal = job.geom.lattice.reciprocal();

    info!(
        "[bandstructure] grid={}x{} pol={:?} bands={} k_points={} lattice={:?} (diagnostics={})",
        job.grid.nx,
        job.grid.ny,
        job.pol,
        job.eigensolver.n_bands,
        job.k_path.len(),
        job.geom.lattice.classify(),
        study_name
    );

    debug!("[bandstructure] precond={:?}", precond_type);

    // Sample the dielectric function from geometry
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    // Compute dielectric contrast for diagnostics
    let eps = dielectric.eps();
    let eps_min = eps.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_contrast = if eps_min > 1e-15 {
        eps_max / eps_min
    } else {
        f64::INFINITY
    };

    info!(
        "[bandstructure] dielectric: ε=[{:.3}, {:.3}] contrast={:.1}x precond={:?}",
        eps_min, eps_max, eps_contrast, precond_type,
    );

    // ========================================================================
    // Operator Diagnostics (one-time, before k-point loop)
    // ========================================================================
    // Use a non-Γ k-point for condition number estimates (Γ has κ → ∞ due to DC mode)
    {
        // Find first non-Γ k-point for diagnostics
        let diag_k_idx = job.k_path.iter().position(|&k| !is_gamma_point(k));
        if let Some(idx) = diag_k_idx {
            let k_frac = job.k_path[idx];
            // Convert fractional k-coordinates to Cartesian Bloch wavevector
            // using the reciprocal lattice (essential for non-orthogonal lattices)
            let bloch = reciprocal.fractional_to_cartesian(k_frac);

            // Create temporary operator for diagnostics
            let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch);

            // Build preconditioner for diagnostics (always using adaptive shift)
            let mut preconditioner_opt: Option<
                Box<dyn crate::preconditioners::OperatorPreconditioner<B>>,
            > = match precond_type {
                PreconditionerType::Auto => {
                    unreachable!("Auto should be resolved before this point")
                }
                PreconditionerType::None => None,
                PreconditionerType::FourierDiagonalKernelCompensated => {
                    Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                }
                PreconditionerType::TransverseProjection => Some(Box::new(
                    theta.build_transverse_projection_preconditioner_adaptive(),
                )),
            };

            const POWER_ITERATIONS: usize = 15;

            // Self-adjointness check
            let self_adj_err = theta.check_self_adjointness();
            let self_adj_status = if self_adj_err < 1e-12 {
                "exact"
            } else if self_adj_err < 1e-6 {
                "good"
            } else {
                "WARN"
            };

            // Condition number estimates
            let (lambda_max, lambda_min, kappa) = theta.estimate_condition_number(POWER_ITERATIONS);

            if let Some(ref mut precond) = preconditioner_opt {
                let (_pm_max, _pm_min, pm_kappa) = theta
                    .estimate_preconditioned_condition_number(&mut **precond, POWER_ITERATIONS);
                let reduction = kappa / pm_kappa;
                info!(
                    "[bandstructure] κ(A)≈{:.1} κ(M⁻¹A)≈{:.1} ({:.1}x reduction) self-adjoint={} ({:.1e})",
                    kappa, pm_kappa, reduction, self_adj_status, self_adj_err
                );
            } else {
                info!(
                    "[bandstructure] κ(A)≈{:.1} (λ_max≈{:.2e}, λ_min≈{:.2e}) self-adjoint={} ({:.1e})",
                    kappa, lambda_max, lambda_min, self_adj_status, self_adj_err
                );
            }
        }
    }

    // Initialize convergence study
    let mut study = ConvergenceStudy::new(&study_name);

    // ========================================================================
    // K-Point Loop
    // ========================================================================

    // Storage for warm-start vectors from previous k-point (simple mode)
    let warm_start_limit = job.eigensolver.n_bands;
    let mut warm_start_store: Vec<Field2D> = Vec::new();

    // Subspace prediction history (rotation-based warm-start)
    let mut subspace_history: Option<SubspaceHistory> = if options.use_subspace_prediction {
        let mut history = SubspaceHistory::new(warm_start_limit);
        if options.use_extrapolation {
            history.enable_extrapolation();
            info!("[bandstructure] Subspace prediction enabled (rotation + extrapolation)");
        } else {
            info!("[bandstructure] Subspace prediction enabled (rotation-only)");
        }
        Some(history)
    } else {
        None
    };

    // Track if previous k-point was Γ (to skip warm-start after Γ)
    // Γ eigenvectors span a different subspace that doesn't overlap well with k≠0
    let mut prev_was_gamma = false;

    // Storage for band tracking: eigenvectors from previous k-point
    // Note: When starting at Γ with proper deflation, Γ eigenvectors are reliable
    // and can be used for warm-starting subsequent k-points.
    let mut prev_eigenvectors: Option<Vec<Field2D>> = None;

    // Storage for previous frequencies (ω) for degenerate block disambiguation
    // These are used by track_bands_with_frequencies to break ties when
    // singular values indicate near-degenerate bands.
    let mut prev_omegas: Option<Vec<f64>> = None;

    // Storage for previous eigenvalues (ω² = λ) for band-window preconditioner shift
    let mut prev_eigenvalues: Option<Vec<f64>> = None;

    // Log if band-window shift is enabled
    if options.use_band_window_shift {
        info!(
            "[bandstructure] Band-window shift enabled (blend={:.2}, scale={:.2})",
            options.band_window_blend, options.band_window_scale
        );
    }

    // Get dielectric epsilon for B-weighted overlaps
    // - TE mode: B = I, use standard inner product (None)
    // - TM mode (generalized): B = ε, use ε-weighted inner product
    let eps_for_tracking: Option<Vec<f64>> = if job.pol == Polarization::TM {
        Some(dielectric.eps().to_vec())
    } else {
        None
    };

    // Detect if we can reuse the first Γ-point result for the last k-point
    // This avoids redundant computation when the path is e.g., Γ→X→M→Γ
    let first_is_gamma = job.k_path.first().map_or(false, |&k| is_gamma_point(k));
    let last_is_gamma = job.k_path.last().map_or(false, |&k| is_gamma_point(k));
    let reuse_gamma = first_is_gamma && last_is_gamma && job.k_path.len() > 1;
    let last_k_idx = job.k_path.len().saturating_sub(1);

    // Storage for first Γ-point frequencies (to reuse for last k-point if applicable)
    let mut first_gamma_omegas: Option<Vec<f64>> = None;

    // Accumulate results
    let mut bands: Vec<Vec<f64>> = Vec::with_capacity(job.k_path.len());
    let mut total_iterations = 0usize;

    // Start timing the solve phase (excludes setup/diagnostics)
    let solve_start = Instant::now();

    for (k_idx, &k_frac) in job.k_path.iter().enumerate() {
        // Check if this is a Γ-point (k ≈ 0)
        // At Γ, the constant mode (DC) is deflated by the eigensolver.
        let is_gamma = is_gamma_point(k_frac);

        // Reuse first Γ-point result for the last k-point (avoid duplicate solve)
        // This is valid when the path loops back to Γ (e.g., Γ→X→M→Γ)
        if reuse_gamma && k_idx == last_k_idx {
            if let Some(ref gamma_omegas) = first_gamma_omegas {
                debug!(
                    "[bandstructure] k#{:03} is duplicate Γ-point: reusing result from k#000",
                    k_idx
                );
                bands.push(gamma_omegas.clone());
                // Don't update warm-start or tracking state - not needed for last point
                continue;
            }
        }

        if is_gamma {
            debug!(
                "[bandstructure] k#{:03} is Γ-point: constant mode will be deflated",
                k_idx
            );
        }

        // Convert fractional k-point to Cartesian Bloch wavevector using the
        // reciprocal lattice. This is essential for non-orthogonal lattices
        // (hexagonal, oblique) where M=(0.5,0) in fractional coordinates does
        // NOT map to [π, 0] in Cartesian k-space.
        let bloch = reciprocal.fractional_to_cartesian(k_frac);

        // Construct the Maxwell operator Θ for this k-point
        let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch);

        // Prepare warm-start using subspace prediction or simple copy
        let predicted_vectors: Option<Vec<Field2D>>;
        let warm_start_used: bool;
        let warm_slice: Option<&[Field2D]> = if prev_was_gamma {
            // Skip warm-start when coming from Γ: those eigenvectors don't overlap well with k≠0
            debug!(
                "[bandstructure] k#{:03}: skipping warm-start (previous was Γ-point)",
                k_idx
            );
            predicted_vectors = None;
            warm_start_used = false;
            None
        } else if let Some(ref mut history) = subspace_history {
            // Use subspace prediction for warm-start
            let prediction = history.predict(k_frac, eps_for_tracking.as_deref());
            if prediction.has_prediction() {
                debug!(
                    "[bandstructure] k#{:03} using {:?} prediction (σ_min={:.4})",
                    k_idx, prediction.method_used, prediction.singular_value_min
                );
                predicted_vectors = Some(prediction.predicted_vectors);
                warm_start_used = true;
                predicted_vectors.as_deref()
            } else {
                predicted_vectors = None;
                warm_start_used = false;
                None
            }
        } else if !warm_start_store.is_empty() {
            // Simple copy warm-start (original behavior)
            predicted_vectors = None;
            warm_start_used = true;
            Some(warm_start_store.as_slice())
        } else {
            predicted_vectors = None;
            warm_start_used = false;
            None
        };
        // Silence unused variable warning
        let _ = warm_start_used;

        // Create label for this k-point run
        let run_label = format!("{}_k{:03}", study_name, k_idx);

        // Create eigensolver config with diagnostics enabled
        let mut eigensolver_config = job.eigensolver.clone();
        eigensolver_config.record_diagnostics = true;

        // Build preconditioner
        // Use band-window shift if enabled AND we have previous eigenvalues,
        // otherwise fall back to adaptive shift.
        let use_band_window = options.use_band_window_shift && prev_eigenvalues.is_some();
        let mut preconditioner_opt: Option<
            Box<dyn crate::preconditioners::OperatorPreconditioner<B>>,
        > = match precond_type {
            PreconditionerType::Auto => unreachable!("Auto should be resolved before this point"),
            PreconditionerType::None => None,
            PreconditionerType::FourierDiagonalKernelCompensated => {
                if use_band_window {
                    let eigenvalues = prev_eigenvalues.as_ref().unwrap();
                    Some(Box::new(
                        theta.build_homogeneous_preconditioner_band_window(
                            eigenvalues,
                            Some(options.band_window_blend),
                            Some(options.band_window_scale),
                        ),
                    ))
                } else {
                    Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                }
            }
            PreconditionerType::TransverseProjection => {
                if use_band_window {
                    let eigenvalues = prev_eigenvalues.as_ref().unwrap();
                    Some(Box::new(
                        theta.build_transverse_projection_preconditioner_band_window(
                            eigenvalues,
                            Some(options.band_window_blend),
                            Some(options.band_window_scale),
                        ),
                    ))
                } else {
                    Some(Box::new(
                        theta.build_transverse_projection_preconditioner_adaptive(),
                    ))
                }
            }
        };

        // Create and run the eigensolver
        let mut solver = Eigensolver::new(
            &mut theta,
            eigensolver_config,
            preconditioner_opt
                .as_mut()
                .map(|p| &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>),
            warm_slice,
        );

        let diag_result = solver.solve_with_diagnostics(&run_label);
        let mut omegas: Vec<f64> = diag_result
            .result
            .eigenvalues
            .iter()
            .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
            .collect();
        // Use all_eigenvectors() to include deflated vectors (e.g., Γ constant mode)
        let mut eigenvectors = solver.all_eigenvectors();

        // ====================================================================
        // Band Tracking: reorder by overlap with previous k-point
        // Uses polar decomposition + frequency continuity for degenerate blocks.
        // ====================================================================
        if let (Some(prev_vecs), Some(prev_freqs)) = (&prev_eigenvectors, &prev_omegas) {
            let tracking_result = track_bands_with_frequencies(
                prev_vecs,
                &eigenvectors,
                prev_freqs,
                &omegas,
                eps_for_tracking.as_deref(),
            );

            // Always log σ_min for diagnostic purposes
            let perm_str = if tracking_result.had_swaps {
                format!("{:?}", tracking_result.permutation)
            } else {
                "identity".to_string()
            };
            let blocks_str = if tracking_result.degenerate_blocks.is_empty() {
                String::new()
            } else {
                format!(" blocks={:?}", tracking_result.degenerate_blocks)
            };
            info!(
                "[band_tracking] k#{:03} σ_min={:.6} perm={}{}",
                k_idx, tracking_result.sigma_min, perm_str, blocks_str
            );

            if tracking_result.had_swaps {
                apply_permutation(&tracking_result.permutation, &mut omegas, &mut eigenvectors);

                // Log warnings for near-degenerate regions (low sigma_min)
                if tracking_result.sigma_min < 0.1 {
                    warn!(
                        "[bandstructure] k#{:03} band tracking: near-degeneracy (σ_min={:.4}), may be unreliable",
                        k_idx, tracking_result.sigma_min
                    );
                }
            }
        }

        // Store eigenvectors for band tracking at next k-point
        prev_eigenvectors = Some(eigenvectors.clone());

        // Store frequencies (ω) for degenerate block disambiguation at next k-point
        prev_omegas = Some(omegas.clone());

        // Store eigenvalues (ω²) for band-window preconditioner shift at next k-point
        prev_eigenvalues = Some(diag_result.result.eigenvalues.clone());

        // Update subspace history or warm-start store for next k-point
        if let Some(ref mut history) = subspace_history {
            // Update subspace history with converged eigenvectors
            history.update(k_frac, &eigenvectors);
        } else {
            // Simple copy for traditional warm-start
            warm_start_store.clear();
            for vec in eigenvectors.iter().take(warm_start_limit) {
                warm_start_store.push(vec.clone());
            }
        }

        // Track if this k-point was Γ for next iteration
        prev_was_gamma = is_gamma;

        total_iterations += diag_result.result.iterations;

        // Update the diagnostics with k-point info and add to study
        let mut run_data = diag_result.diagnostics;
        run_data.config.k_index = Some(k_idx);
        run_data.config.k_point = Some(k_frac);
        run_data.config.bloch = Some(bloch);
        run_data.config.polarization = Some(format!("{:?}", job.pol));
        run_data.config.preconditioner_type = options.precond_type;
        run_data.config.warm_start_enabled = warm_start_used;
        study.add_run(run_data);

        // Store first Γ-point result for potential reuse at end of path
        if reuse_gamma && k_idx == 0 && is_gamma {
            first_gamma_omegas = Some(omegas.clone());
        }

        bands.push(omegas);
    } // end k-point loop

    // ========================================================================
    // Finalize
    // ========================================================================

    let solve_elapsed = solve_start.elapsed().as_secs_f64();
    let time_per_iter_ms = if total_iterations > 0 {
        solve_elapsed * 1000.0 / total_iterations as f64
    } else {
        0.0
    };

    info!(
        "[bandstructure] complete: {} k-points, {} total iterations, {:.2}s ({:.1}ms/iter) (diagnostics recorded)",
        job.k_path.len(),
        total_iterations,
        solve_elapsed,
        time_per_iter_ms
    );

    // Build initial result
    let result = BandStructureResult {
        k_path: job.k_path.clone(),
        distances: compute_k_path_distances(&job.k_path),
        bands,
    };

    // Rotate output to start from Γ-point (if path was internally rotated)
    let result = rotate_result_to_gamma(result);

    BandStructureResultWithDiagnostics { result, study }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute cumulative distances along the k-path.
///
/// Returns a vector where `distances[i]` is the Euclidean distance from
/// `k_path[0]` to `k_path[i]`, summed along the path segments.
///
/// # Arguments
///
/// - `k_path`: The k-path in fractional coordinates
///
/// # Returns
///
/// Vector of cumulative distances, with `distances[0] = 0.0`.
pub(crate) fn compute_k_path_distances(k_path: &[[f64; 2]]) -> Vec<f64> {
    if k_path.is_empty() {
        return Vec::new();
    }

    let mut distances = Vec::with_capacity(k_path.len());
    let mut cumulative = 0.0;

    distances.push(0.0);

    for window in k_path.windows(2) {
        let dx = window[1][0] - window[0][0];
        let dy = window[1][1] - window[0][1];
        cumulative += (dx * dx + dy * dy).sqrt();
        distances.push(cumulative);
    }

    distances
}

/// Find the index of the first Γ-point (k ≈ 0) in the path.
/// Returns None if no Γ-point is found.
fn find_gamma_index(k_path: &[[f64; 2]]) -> Option<usize> {
    const GAMMA_TOL: f64 = 1e-9;
    k_path
        .iter()
        .position(|k| k[0].abs() < GAMMA_TOL && k[1].abs() < GAMMA_TOL)
}

/// Rotate the band structure result so it starts from the Γ-point.
///
/// Rotate band structure result so that it starts from the Γ-point.
///
/// This function is a legacy helper for when k-paths were internally rotated
/// to start from a non-Γ point. With the current approach of starting at Γ,
/// this function typically returns early (γ_idx == 0).
///
/// For closed paths (where first ≈ last), the duplicate endpoint is handled
/// so the output has the same length as the input.
fn rotate_result_to_gamma(result: BandStructureResult) -> BandStructureResult {
    let gamma_idx = match find_gamma_index(&result.k_path) {
        Some(idx) => idx,
        None => return result, // No Γ-point found, return as-is
    };

    if gamma_idx == 0 {
        return result; // Already starts at Γ
    }

    let n = result.k_path.len();
    if n < 2 {
        return result;
    }

    // Check if path is closed (first ≈ last)
    let is_closed = {
        let first = result.k_path.first().unwrap();
        let last = result.k_path.last().unwrap();
        (first[0] - last[0]).abs() < 1e-9 && (first[1] - last[1]).abs() < 1e-9
    };

    // Rotate k_path and bands
    let mut new_k_path = Vec::with_capacity(n);
    let mut new_bands = Vec::with_capacity(n);

    if is_closed {
        // For closed paths: take [gamma_idx..n-1] ++ [0..gamma_idx] ++ [gamma_idx]
        // This gives us a path that starts and ends at Γ
        for i in gamma_idx..(n - 1) {
            new_k_path.push(result.k_path[i]);
            new_bands.push(result.bands[i].clone());
        }
        for i in 0..=gamma_idx {
            new_k_path.push(result.k_path[i]);
            new_bands.push(result.bands[i].clone());
        }
    } else {
        // For open paths: simple rotation
        for i in gamma_idx..n {
            new_k_path.push(result.k_path[i]);
            new_bands.push(result.bands[i].clone());
        }
        for i in 0..gamma_idx {
            new_k_path.push(result.k_path[i]);
            new_bands.push(result.bands[i].clone());
        }
    }

    // Recompute distances for the new ordering
    let new_distances = compute_k_path_distances(&new_k_path);

    BandStructureResult {
        k_path: new_k_path,
        distances: new_distances,
        bands: new_bands,
    }
}
