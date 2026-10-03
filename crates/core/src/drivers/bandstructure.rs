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
//! use blaze2d_core::bandstructure::{run_with_options, BandStructureJob, RunOptions};
//!
//! let result = run_with_options(backend, &job, RunOptions::default());
//! // result.bands[k_index][band_index] gives ω for each (k, band) pair
//! ```

use log::{debug, info, warn};

use crate::{
    backend::SpectralBackend,
    band_tracking::{apply_permutation, track_bands_with_frequencies},
    diagnostics::ConvergenceStudy,
    dielectric::{Dielectric2D, DielectricOptions},
    eigensolver::{Eigensolver, EigensolverConfig, SubspaceHistory},
    field::Field2D,
    geometry::Geometry2D,
    grid::Grid2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::reuse::{CachedSpectrum, KReuse, ReuseReport, SpectralCache, SymmetryOptions,candidate_operations},
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

/// Incremental result emitted after a single k-point solve.
#[derive(Debug, Clone)]
pub struct KPointResult {
    /// Present only for an explicitly selected metric Fourier support. Its
    /// projected certificate and full-rectangle diagnostic have different roles.
    pub discretization: Option<super::support_bands::SupportPointReport>,
    /// Actual reuse and qualification decisions; reused points take zero iterations.
    pub symmetry: ReuseReport,
    /// Zero-based index of the completed k-point in the path.
    pub k_index: usize,
    /// Total number of k-points in the path.
    pub total_k_points: usize,
    /// The solved k-point in fractional coordinates.
    pub k_point: [f64; 2],
    /// Cumulative distance along the k-path for this point.
    pub distance: f64,
    /// Computed eigenfrequencies ω (not ω²) for this k-point.
    pub omegas: Vec<f64>,
    /// All eigensolver iterations at this point, including rejected attempts.
    pub iterations: usize,
    /// Solver convergence state and relative residuals in returned band order.
    pub converged: bool,
    pub residuals: Vec<f64>,
    pub absolute_residuals: Vec<f64>,
    pub b_orthogonality_defect: f64,
    /// Retained only when requested through the run options.
    pub eigenvectors: Option<Vec<Field2D>>,
    /// Whether this point is at Γ and uses the DC-mode deflation path.
    pub is_gamma: bool,
    /// Wall time spent on this point, including rejected attempts and certification.
    pub elapsed_seconds: f64,
    /// Numerical work spent on this point.
    pub work: crate::work::Work,
    /// The preconditioner used at this point, absent for reused points.
    pub preconditioner: Option<crate::preconditioners::PreconditionerInfo>,
}

// ============================================================================
// Verbosity Control
// ============================================================================

// (Previously a `Verbosity` enum lived here.) All progress reporting is now
// done through the `log` crate; configure RUST_LOG/log filters to control output.

/// Options for band structure computation.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Independently requested, qualified symmetry policies. Defaults to off.
    pub symmetry: SymmetryOptions,
    /// Preconditioner kind and shift rule, built per k-point.
    pub preconditioner: crate::preconditioners::PreconditionerSpec,
    /// Run the one-off operator and preconditioner power-iteration diagnostics.
    pub operator_diagnostics: bool,
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
    /// Disable band tracking after eigensolver convergence.
    ///
    /// When enabled, skips the polar decomposition + Hungarian algorithm band
    /// tracking step that reorders bands based on eigenvector overlap. This is
    /// useful for irregular k-paths (e.g., 5×5 grids around Γ) where there's no
    /// natural "previous" k-point and band tracking can cause spurious band hopping.
    ///
    /// Default: false (band tracking is enabled)
    pub disable_band_tracking: bool,
    /// Reuse the first Γ-point result for the last k-point when the path is closed (Γ→...→Γ).
    ///
    /// When enabled, skips the (often redundant) solve at the closing Γ and copies the
    /// first Γ frequencies. This is faster but breaks line continuity if bands cross
    /// along the path, since indices at the start and end Γ may not correspond to the
    /// same physical bands after tracking.
    ///
    /// Default: true.
    pub reuse_gamma: bool,
    /// Include fields in incremental results. Disabled to avoid extra copies.
    pub retain_eigenvectors: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            symmetry: SymmetryOptions::default(),
            preconditioner: crate::preconditioners::PreconditionerSpec::default(),
            operator_diagnostics: false,
            use_subspace_prediction: true, // Stage 1: rotation-based warm-start
            use_extrapolation: true,       // Stage 2: linear extrapolation
            disable_band_tracking: false,  // Band tracking enabled by default
            reuse_gamma: true,             // Reuse first Γ for closing Γ (fast path)
            retain_eigenvectors: false,
        }
    }
}

impl RunOptions {
    /// Create default run options (the default `PreconditionerSpec`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the preconditioner specification.
    pub fn with_preconditioner(mut self, spec: crate::preconditioners::PreconditionerSpec) -> Self {
        self.preconditioner = spec;
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

    /// Disable band tracking after eigensolver convergence.
    ///
    /// When enabled, skips the polar decomposition + Hungarian algorithm that
    /// reorders bands based on eigenvector overlap. Useful for irregular k-paths
    /// (e.g., 5×5 grids around Γ) where band tracking can cause spurious hopping.
    pub fn with_disable_band_tracking(mut self, disabled: bool) -> Self {
        self.disable_band_tracking = disabled;
        self
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Compute the photonic band structure.
///
/// This is the main entry point for band-structure calculations. It iterates
/// over all k-points in the job's k_path, solving the eigenvalue problem at
/// each point to obtain the photonic band frequencies.
///
/// Uses default options, including the default `PreconditionerSpec`.
///
/// # Arguments
///
/// - `backend`: The spectral backend (CPU, CUDA, etc.)
/// - `job`: The band-structure job configuration
/// - `options`: Preconditioner, warm-start, and band-tracking options
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
pub fn run_with_options<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    options: RunOptions,
) -> BandStructureResult {
    run_core(backend, job, options, None, None, None)
}

/// Compute the photonic band structure while emitting each completed k-point.
pub fn run_with_k_streaming<B, F>(
    backend: B,
    job: &BandStructureJob,
    options: RunOptions,
    mut on_k_point: F,
) -> BandStructureResult
where
    B: SpectralBackend + Clone,
    F: FnMut(KPointResult),
{
    run_core(
        backend,
        job,
        options,
        None,
        Some(&mut on_k_point as &mut dyn FnMut(KPointResult)),
        None,
    )
}

/// Execute a full path with a worker-owned history of certified geometries.
pub fn run_with_continuation<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    options: RunOptions,
    continuation: Option<&mut crate::symmetry::continuation::BandContinuation>,
    mut on_k_point: impl FnMut(KPointResult),
) -> BandStructureResult {
    run_core(backend, job, options, None, Some(&mut on_k_point), continuation)
}

/// Shared implementation behind [`run_with_options`] and
/// [`run_with_diagnostics_and_options`].
///
/// When `study` is `Some`, per-iteration convergence diagnostics are recorded
/// into it via the eigensolver's `solve_with_diagnostics` path; when `None`,
/// the faster plain `solve` path is used. The two solver paths differ in their
/// internal convergence accounting, so this function branches on `study`
/// rather than always recording.
fn run_core<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    mut options: RunOptions,
    mut study: Option<&mut ConvergenceStudy>,
    mut on_k_point: Option<&mut dyn FnMut(KPointResult)>,
    mut continuation: Option<&mut crate::symmetry::continuation::BandContinuation>,
) -> BandStructureResult {
    if study.is_some() || job.eigensolver.record_diagnostics || std::mem::size_of::<B::Real>() != 8 {
        continuation = None;
    }
    let requested_reuse=options.symmetry.k_reuse;
    let diagnostic_fallback=(study.is_some() || job.eigensolver.record_diagnostics) &&
        (options.symmetry.k_reuse!=KReuse::Off || options.symmetry.sectors!=crate::symmetry::sectors::SectorPolicy::Off ||
            options.symmetry.arithmetic!=crate::symmetry::fourier::Arithmetic::SpatialComplex ||
            options.symmetry.k_traversal!=crate::symmetry::traversal::KTraversal::Given || options.reuse_gamma);
    let diagnostic_note="per-iteration diagnostics use unrestricted spatial solves in the supplied order; spectrum reuse, sectors, Fourier arithmetic and legacy Gamma reuse are disabled for this study";
    if diagnostic_fallback {
        options.symmetry.k_reuse=KReuse::Off;
        options.symmetry.sectors=crate::symmetry::sectors::SectorPolicy::Off;
        options.symmetry.arithmetic=crate::symmetry::fourier::Arithmetic::SpatialComplex;
        options.symmetry.k_traversal=crate::symmetry::traversal::KTraversal::Given;
        options.reuse_gamma=false;
        warn!("[bandstructure] {diagnostic_note}");
        if let Some(study)=study.as_deref_mut() {study.notes.push(diagnostic_note.into());}
    }
    let mut traversal_rejection=None;
    if options.symmetry.k_traversal==crate::symmetry::traversal::KTraversal::Nearest {
        if study.is_some() || job.eigensolver.record_diagnostics {
            traversal_rejection=Some("convergence diagnostics keep the supplied k-point order".to_string());
        } else {
            match crate::symmetry::traversal::run_reordered(backend.clone(),job,options.clone(),&mut on_k_point,continuation.as_deref_mut()) {
                Ok(result)=>return result,
                Err(reason)=>traversal_rejection=Some(reason),
            }
        }
    }
    crate::profile_scope!("run.wall");
    crate::profiler::start_timer("run.setup");
    // ========================================================================
    // Setup Phase
    // ========================================================================

    // Resolve Auto preconditioner type based on polarization
    let precond_kind = options.preconditioner.resolve_kind(job.pol);

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
        eps_min, eps_max, eps_contrast, precond_kind,
    );

    if options.disable_band_tracking {
        warn!("[bandstructure] Band tracking DISABLED - bands will be sorted by eigenvalue only");
    }

    // ========================================================================
    // Operator Diagnostics (one-time, before k-point loop)
    // ========================================================================
    // Use a non-Γ k-point for condition number estimates (Γ has κ → ∞ due to DC mode)
    if options.operator_diagnostics {
        // Find first non-Γ k-point for diagnostics
        let diag_k_idx = job.k_path.iter().position(|&k| !is_gamma_point(k));
        if let Some(idx) = diag_k_idx {
            let k_frac = job.k_path[idx];
            // Convert fractional k-coordinates to Cartesian Bloch wavevector
            // using the reciprocal lattice (essential for non-orthogonal lattices)
            let bloch = reciprocal.fractional_to_cartesian(k_frac);

            // Create temporary operator for diagnostics
            crate::profile_scope!("run.operator_diagnostics");
            let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch);

            // Build the job's preconditioner for diagnostics (no band-window history here)
            let (mut preconditioner_opt, _) = theta.build_preconditioner(&options.preconditioner, None);

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
    if options.preconditioner.shift == crate::preconditioners::ShiftRule::BandWindow {
        info!(
            "[bandstructure] Band-window shift enabled (blend={:.2}, scale={:.2})",
            options.preconditioner.band_window_blend, options.preconditioner.band_window_scale
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
    let reuse_gamma =
        options.reuse_gamma && options.symmetry.k_reuse == KReuse::Off && first_is_gamma && last_is_gamma && job.k_path.len() > 1;
    let last_k_idx = job.k_path.len().saturating_sub(1);
    let distances = compute_k_path_distances(&job.k_path);
    let total_k_points = job.k_path.len();

    // Storage for first Γ-point frequencies (to reuse for last k-point if applicable)
    let mut first_gamma_omegas: Option<Vec<f64>> = None;
    let mut first_gamma_quality = (false, Vec::new(), 0.0, Vec::new());
    let mut first_gamma_fields = None;

    // Accumulate results
    let mut bands: Vec<Vec<f64>> = Vec::with_capacity(job.k_path.len());
    let mut total_iterations = 0usize;
    let mut symmetry_cache = SpectralCache::new(options.symmetry.max_cache_bytes);
    let mut arithmetic_rejection = None;
    let key = |k: [f64; 2]| k.map(|x| if x == 0.0 { 0 } else { x.to_bits() });
    let mut exact_uses = std::collections::HashMap::<_, usize>::new();
    if options.symmetry.k_reuse == KReuse::Identical {
        for &k in &job.k_path { *exact_uses.entry(key(k)).or_default() += 1; }
    }
    let geometry_parameters: Vec<_> = std::iter::once(job.geom.eps_bg)
        .chain(job.geom.atoms.iter().flat_map(|atom| [atom.pos[0], atom.pos[1], atom.radius, atom.eps_inside]))
        .collect();
    let symmetry_operations = if matches!(options.symmetry.k_reuse,KReuse::Unitary|KReuse::TimeReversal) {
        let mut operations=candidate_operations(&job.geom.lattice,options.symmetry.plane_group);
        if options.symmetry.k_reuse==KReuse::TimeReversal {
            let antiunitary:Vec<_>=operations.iter().map(|op|op.with_conjugation(true)).collect();
            operations.extend(antiunitary);
        }
        operations
    } else {Vec::new()};

    // Start timing the solve phase (excludes setup/diagnostics)
    let solve_start = crate::timing::Timer::start();

    crate::profiler::stop_timer("run.setup");
    for (k_idx, &k_frac) in job.k_path.iter().enumerate() {
        let point_start = crate::timing::Timer::start();
        let work_start = crate::work::snapshot();
        let mut point_continuation = continuation.as_deref_mut()
            .filter(|state| !state.first_point_only || k_idx == 0);
        crate::profile_scope!("run.k_point");
        let mut reuse_report = ReuseReport { requested: requested_reuse, ..Default::default() };
        if point_continuation.is_some() { reuse_report.continuation = Some(Default::default()); }
        if let Some(reason) = &arithmetic_rejection {
            reuse_report.rejected.push(format!("automatic arithmetic fallback for this geometry: {reason}"));
        }
        if let Some(uses) = exact_uses.get_mut(&key(k_frac)) { *uses -= 1; }
        if diagnostic_fallback {reuse_report.rejected.push(diagnostic_note.into());}
        if let Some(reason)=&traversal_rejection {reuse_report.rejected.push(reason.clone());}
        let mut reused = if options.symmetry.k_reuse != KReuse::Off {
            symmetry_cache.exact(k_frac)
        } else { None };
        if let Some(ref entry) = reused { reuse_report.source_index = Some(entry.source_index); }
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
                if let Some(ref mut on_k_point) = on_k_point {
                    (**on_k_point)(KPointResult {
                        discretization: None,
                        symmetry: ReuseReport::default(),
                        k_index: k_idx,
                        total_k_points,
                        k_point: k_frac,
                        distance: distances[k_idx],
                        omegas: gamma_omegas.clone(),
                        iterations: 0,
                        converged: first_gamma_quality.0,
                        residuals: first_gamma_quality.1.clone(),
                        absolute_residuals: first_gamma_quality.3.clone(),
                        b_orthogonality_defect: first_gamma_quality.2,
                        eigenvectors: first_gamma_fields.clone(),
                        is_gamma: true,
                        elapsed_seconds: point_start.elapsed_secs(),
                        work: crate::work::snapshot().since(work_start),
                        preconditioner: None,
                    });
                }
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
        // Note: _predicted_vectors owns the data that warm_slice borrows from
        let _predicted_vectors: Option<Vec<Field2D>>;
        let warm_start_used: bool;
        let warm_slice: Option<&[Field2D]> = if let Some(ref mut history) = subspace_history {
            // Use subspace prediction for warm-start
            let prediction = history.predict(k_frac, eps_for_tracking.as_deref());
            if prediction.has_prediction() {
                debug!(
                    "[bandstructure] k#{:03} using {:?} prediction (σ_min={:.4})",
                    k_idx, prediction.method_used, prediction.singular_value_min
                );
                _predicted_vectors = Some(prediction.predicted_vectors);
                warm_start_used = true;
                _predicted_vectors.as_deref()
            } else {
                _predicted_vectors = None;
                warm_start_used = false;
                None
            }
        } else if !warm_start_store.is_empty() {
            // Simple copy warm-start (original behavior)
            _predicted_vectors = None;
            warm_start_used = true;
            Some(warm_start_store.as_slice())
        } else {
            _predicted_vectors = None;
            warm_start_used = false;
            None
        };

        // Construct the Maxwell operator Θ for this k-point.
        let mut theta = {
            crate::profile_scope!("run.operator_setup");
            ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch)
        };
        let path_warm_slice = warm_slice;
        if reused.is_none() {
            if let Some(state) = point_continuation.as_deref_mut() {
                match state.cache.reuse_exact(&mut theta, k_frac, job.eigensolver.n_bands, &options.symmetry) {
                    Ok(Some(hit)) => {
                        reuse_report.continuation.as_mut().unwrap().exact_source = Some(hit.source_id);
                        reused = Some(hit.spectrum);
                    }
                    Ok(None) => {}
                    Err(reason) => reuse_report.rejected.push(reason),
                }
            }
        }
        if reused.is_none() && matches!(options.symmetry.k_reuse,KReuse::Unitary|KReuse::TimeReversal) {
            reused = symmetry_cache.related(&mut theta,&job.geom.lattice,&dielectric,k_frac,&symmetry_operations,&options.symmetry,&mut reuse_report);
        }
        let geometry_seed = if reused.is_none() {
            point_continuation.as_deref_mut().and_then(|state| {
                match state.cache.prepare(&mut theta, crate::symmetry::continuation::SeedHistory::Latest,
                    job.eigensolver.n_bands, options.symmetry.relative_rank_tolerance) {
                    Ok(Some(seed)) => {
                        reuse_report.continuation.as_mut().unwrap().seed = Some(seed.report.clone());
                        Some(seed)
                    }
                    Ok(None) => None,
                    Err(reason) => { reuse_report.rejected.push(reason); None }
                }
            })
        } else { None };
        let warm_slice = geometry_seed.as_ref().map(|s| s.fields.as_slice()).or(warm_slice);
        let rank_seed = if reused.is_none() && geometry_seed.is_none() && options.symmetry.seed_transfer==crate::symmetry::seed::SeedTransfer::RankAware {
            warm_slice.and_then(|input|match crate::symmetry::seed::rank_aware_seed(&mut theta,input,options.symmetry.relative_rank_tolerance) {
                Ok(seed)=>{reuse_report.seed=Some(seed.report.clone());Some(seed)},
                Err(error)=>{reuse_report.rejected.push(error.to_string());None},
            })
        } else {None};
        let iteration_start=crate::eigensolver::iteration_total();
        let coarse_seed=if k_idx==0 && reused.is_none() && warm_slice.is_none() {
            options.symmetry.coarse_start.and_then(|coarse| {
                let (seed, report)=crate::symmetry::coarse::prepare_initializer(
                    &mut theta,&job.geom,&job.dielectric,k_frac,&job.eigensolver,coarse,
                    options.symmetry.relative_rank_tolerance);
                reuse_report.solver_attempts.push(crate::symmetry::reuse::SolveAttempt {
                    method:"coarse_initializer",iterations:report.iterations,retained:false,
                    rejection:report.rejection.clone()});
                if let Some(reason)=&report.rejection {reuse_report.rejected.push(reason.clone());}
                reuse_report.coarse_start=Some(report);
                seed
            })
        } else {None};
        let warm_slice=coarse_seed.as_ref().or(rank_seed.as_ref()).map(|s|s.fields.as_slice()).or(warm_slice);
        let warm_start_used=(warm_start_used || coarse_seed.is_some() || geometry_seed.is_some()) && warm_slice.is_some_and(|s|!s.is_empty());

        // Build preconditioner for this operator. The band-window shift uses the
        // previous k-point's eigenvalues and falls back to the legacy shift.
        crate::profiler::start_timer("run.preconditioner_setup");
        let (mut preconditioner_opt, precond_info) =
            theta.build_preconditioner(&options.preconditioner, prev_eigenvalues.as_deref());

        crate::profiler::stop_timer("run.preconditioner_setup");

        let mut sector_result=None;
        if reused.is_none() && options.symmetry.sectors!=crate::symmetry::sectors::SectorPolicy::Off {
            use crate::symmetry::{sectors::SectorPolicy,affine::{AffineGroup,Rational}};
            let mut groups=Vec::new();
            if matches!(options.symmetry.sectors,SectorPolicy::ConservativeGroup|SectorPolicy::AdaptiveGroup) {
                if let Some(group)=options.symmetry.plane_group {
                    if let Ok(little)=group.group().little_group(k_frac,1e-12) {if little.order()>1 {groups.push(little);}}
                } else {reuse_report.rejected.push("group sectors require an explicit plane_group setting".into());}
            } else {
                for op in crate::symmetry::reuse::candidate_operations(&job.geom.lattice,options.symmetry.plane_group).into_iter().filter(|op|op.determinant()==-1 && op.translation()==[Rational::ZERO;2]) {
                    let group=AffineGroup::from_generators(&[op],2).expect("order-two mirror");
                    if group.little_group(k_frac,1e-12).is_ok_and(|g|g.order()==2) {groups.push(group);}
                }
            }
            for group in groups {
                let attempt_start=crate::eigensolver::iteration_total();
                let attempt=crate::symmetry::sectors::solve_conservative(&mut theta,&job.geom.lattice,k_frac,group,&job.eigensolver,warm_slice,&options.symmetry,preconditioner_opt.as_mut().map(|p|&mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>));
                reuse_report.solver_attempts.push(crate::symmetry::reuse::SolveAttempt {method:"sectors",iterations:crate::eigensolver::iteration_total().wrapping_sub(attempt_start),retained:attempt.is_ok(),rejection:attempt.as_ref().err().cloned()});
                match attempt {
                    Ok(result)=>{reuse_report.sectors=Some(result.report.clone());sector_result=Some(result);break;},
                    Err(reason)=>reuse_report.rejected.push(reason),
                }
            }
            if sector_result.is_none() {reuse_report.rejected.push("no qualified sector solve; using unrestricted solver".into());}
        }

        if sector_result.as_ref().is_some_and(|s|s.report.sectors.iter().any(|s|s.coordinates.is_none())) && options.symmetry.arithmetic!=crate::symmetry::fourier::Arithmetic::SpatialComplex {
            reuse_report.rejected.push("requested Fourier representation is not combined with the retained sector solve".into());
        }
        let coordinate_result=if reused.is_none() && sector_result.is_none() && options.symmetry.arithmetic!=crate::symmetry::fourier::Arithmetic::SpatialComplex {
            let attempt_start=crate::eigensolver::iteration_total();
            let attempt=crate::symmetry::fourier::solve_fourier(&mut theta,&job.geom.lattice,k_frac,&job.eigensolver,warm_slice,&options.symmetry,preconditioner_opt.as_mut().map(|p|&mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>));
            reuse_report.solver_attempts.push(crate::symmetry::reuse::SolveAttempt {method:"fourier",iterations:crate::eigensolver::iteration_total().wrapping_sub(attempt_start),retained:attempt.is_ok(),rejection:attempt.as_ref().err().cloned()});
            match attempt {
                Ok(result)=>{reuse_report.coordinates=Some(result.report.clone());Some(result)},
                Err(reason)=>{
                    if options.symmetry.automatic_policy {
                        arithmetic_rejection = Some(reason.clone());
                        options.symmetry.arithmetic = crate::symmetry::fourier::Arithmetic::SpatialComplex;
                    }
                    reuse_report.rejected.push(reason);None
                },
            }
        } else {None};

        // Solve the eigenproblem at this k-point.
        //
        // Branch on diagnostics: the recording path uses
        // `solve_with_diagnostics` (eager residual norms + per-iteration
        // snapshots), the plain path uses `solve`. Both use the same stopping
        // rule, while the plain path avoids recording iteration snapshots.
        let unrestricted=reused.is_none() && sector_result.is_none() && coordinate_result.is_none();
        let unrestricted_start=crate::eigensolver::iteration_total();
        let k_iterations;
        let mut k_converged;
        let mut k_residuals;
        let mut k_absolute_residuals = Vec::new();
        let mut final_evals: Vec<f64>;
        let mut eigenvectors: Vec<Field2D>;
        // A coordinate solve certifies its returned pairs with fresh applications of `theta`.
        let mut coordinate_defect = None;
        if let Some(ref entry) = reused {
            k_iterations = 0;
            k_converged = true;
            k_residuals = entry.residuals.clone();
            k_absolute_residuals = entry.absolute_residuals.clone();
            final_evals = entry.eigenvalues.clone();
            eigenvectors = entry.fields.clone();
            crate::profiler::count("symmetry.reused_solves", 1);
        } else if let Some(result)=sector_result {
            k_iterations=result.iterations;k_converged=true;
            k_residuals=result.spectrum.residuals;k_absolute_residuals=result.spectrum.absolute_residuals;
            final_evals=result.spectrum.eigenvalues;eigenvectors=result.spectrum.fields;
        } else if let Some(result)=coordinate_result {
            k_iterations=result.iterations;k_converged=true;k_residuals=result.spectrum.residuals;k_absolute_residuals=result.spectrum.absolute_residuals;final_evals=result.spectrum.eigenvalues;eigenvectors=result.spectrum.fields;
            coordinate_defect=Some(result.spectrum.orthogonality_defect);
        } else if let Some(study) = study.as_deref_mut() {
            let mut cfg = job.eigensolver.clone();
            cfg.k_index = Some(k_idx);
            cfg.record_diagnostics = true;
            let run_label = format!("{}_k{:03}", study.name, k_idx);

            let mut solver = Eigensolver::new(
                &mut theta,
                cfg,
                preconditioner_opt.as_mut().map(|p| {
                    &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>
                }),
                warm_slice,
            );

            let diag_result = solver.solve_with_diagnostics(&run_label);
            k_iterations = diag_result.result.iterations;
            k_converged = diag_result.result.converged;
            k_residuals = diag_result.result.convergence.relative_residuals.clone();
            final_evals = diag_result.result.eigenvalues.clone();
            eigenvectors = solver.all_eigenvectors();

            // Record per-k-point diagnostics into the study.
            let mut run_data = diag_result.diagnostics;
            run_data.config.k_index = Some(k_idx);
            run_data.config.k_point = Some(k_frac);
            run_data.config.bloch = Some(bloch);
            run_data.config.polarization = Some(format!("{:?}", job.pol));
            run_data.config.preconditioner_type = options.preconditioner.diagnostics_type(job.pol);
            run_data.config.warm_start_enabled = warm_start_used;
            study.add_run(run_data);
        } else {
            let mut cfg = job.eigensolver.clone();
            cfg.k_index = Some(k_idx);

            let mut solver = Eigensolver::new(
                &mut theta,
                cfg,
                preconditioner_opt.as_mut().map(|p| {
                    &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>
                }),
                warm_slice,
            );

            let result = solver.solve();
            k_iterations = result.iterations;
            k_converged = result.converged;
            k_residuals = result.convergence.relative_residuals;
            final_evals = result.eigenvalues;
            eigenvectors = solver.all_eigenvectors();
        }

        if unrestricted {
            reuse_report.solver_attempts.push(crate::symmetry::reuse::SolveAttempt {method:"unrestricted",iterations:crate::eigensolver::iteration_total().wrapping_sub(unrestricted_start),retained:true,rejection:None});
        }
        reuse_report.retained_iterations=k_iterations;
        reuse_report.attempted_iterations=crate::eigensolver::iteration_total().wrapping_sub(iteration_start);

        // The iteration cache contains soft-lock placeholders. Certify the
        // returned fields without altering eigenvalues, gauges, or warm starts.
        let mut k_b_orthogonality_defect = if let Some(ref entry) = reused {
            entry.orthogonality_defect
        } else if let Some(defect) = coordinate_defect {
            defect
        } else if on_k_point.is_some() || options.symmetry.k_reuse != KReuse::Off || point_continuation.is_some() {
            let certificate = crate::eigensolver::refine::certify_without_refinement(
                &mut theta,
                &final_evals,
                &eigenvectors,
            );
            k_absolute_residuals = certificate.absolute_residuals;
            k_residuals = certificate.residuals;
            certificate.b_orthogonality_defect
        } else {
            0.0
        };

        // A geometry prediction is optional initialization. Retry the ordinary
        // path if its solve or fresh certificate fails, retaining all work.
        let acceptable = || {
            k_converged && final_evals.len() == job.eigensolver.n_bands
                && eigenvectors.len() == job.eigensolver.n_bands && k_b_orthogonality_defect.is_finite()
                && k_b_orthogonality_defect <= options.symmetry.max_orthogonality_defect
                && final_evals.iter().enumerate().all(|(i, &value)| {
                    let (residual, limit) = if value == 0.0 && is_gamma {
                        (k_absolute_residuals.get(i), options.symmetry.max_zero_mode_absolute_residual)
                    } else { (k_residuals.get(i), options.symmetry.max_relative_residual) };
                    residual.is_some_and(|r| r.is_finite() && *r <= limit)
                })
        };
        if geometry_seed.is_some() && !acceptable() {
            let reason = "geometry-seeded solve failed convergence or fresh certification";
            reuse_report.continuation.as_mut().unwrap().fallback = Some(reason.into());
            for attempt in &mut reuse_report.solver_attempts { attempt.retained = false; }
            reuse_report.coordinates = None;
            reuse_report.sectors = None;
            let retry_start = crate::eigensolver::iteration_total();
            let mut cfg = job.eigensolver.clone();
            cfg.k_index = Some(k_idx);
            let mut solver = Eigensolver::new(&mut theta, cfg,
                preconditioner_opt.as_mut().map(|p| &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>),
                path_warm_slice);
            let retry = solver.solve();
            eigenvectors = solver.all_eigenvectors();
            drop(solver);
            final_evals = retry.eigenvalues;
            k_converged = retry.converged;
            let certificate = crate::eigensolver::refine::certify_without_refinement(&mut theta, &final_evals, &eigenvectors);
            k_residuals = certificate.residuals;
            k_absolute_residuals = certificate.absolute_residuals;
            k_b_orthogonality_defect = certificate.b_orthogonality_defect;
            let iterations = crate::eigensolver::iteration_total().wrapping_sub(retry_start);
            reuse_report.retained_iterations = iterations;
            reuse_report.solver_attempts.push(crate::symmetry::reuse::SolveAttempt {
                method: "continuation_fallback", iterations, retained: true, rejection: None,
            });
        }
        reuse_report.attempted_iterations = crate::eigensolver::iteration_total().wrapping_sub(iteration_start);
        let k_iterations = reuse_report.attempted_iterations;

        if let Some(state) = point_continuation.as_deref_mut() {
            if k_converged {
                let entry = CachedSpectrum { source_index: k_idx, k: k_frac,
                    eigenvalues: final_evals.clone(), fields: eigenvectors.clone(),
                    residuals: k_residuals.clone(), absolute_residuals: k_absolute_residuals.clone(),
                    orthogonality_defect: k_b_orthogonality_defect };
                match state.cache.insert_latest_certified(&mut theta, format!("{}:{k_idx}", state.job_index),
                    geometry_parameters.clone(), entry, &options.symmetry) {
                    Ok(report) => reuse_report.continuation.as_mut().unwrap().insertion = Some(report),
                    Err(reason) => reuse_report.rejected.push(reason),
                }
            }
            reuse_report.continuation.as_mut().unwrap().cache_payload_bytes = state.cache.payload_bytes();
        }

        // Cache sorted physical pairs before path tracking permutes the output.
        let needed_again = options.symmetry.k_reuse != KReuse::Identical
            || exact_uses.get(&key(k_frac)).is_some_and(|&uses| uses > 0);
        if options.symmetry.k_reuse != KReuse::Off && needed_again && reused.is_none() && k_converged {
            let entry = CachedSpectrum { source_index: k_idx, k: k_frac,
                eigenvalues: final_evals.clone(), fields: eigenvectors.clone(),
                residuals: k_residuals.clone(), absolute_residuals: k_absolute_residuals.clone(),
                orthogonality_defect: k_b_orthogonality_defect };
            if entry.passes(&options.symmetry) { symmetry_cache.insert(entry); }
            else { reuse_report.rejected.push("source spectrum failed fresh quality gates".into()); }
        }
        reuse_report.cache_bytes = symmetry_cache.bytes();
        reuse_report.cache_entries = symmetry_cache.len();

        // Convert eigenvalues (ω²) to frequencies (ω)
        let mut omegas: Vec<f64> = final_evals
            .iter()
            .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
            .collect();

        // (eigenvectors are already set)

        total_iterations += k_iterations;

        // ====================================================================
        // Band Tracking: reorder by overlap with previous k-point
        // Uses polar decomposition + frequency continuity for degenerate blocks.
        // ====================================================================
        if !options.disable_band_tracking {
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
                    let original = k_residuals.clone();
                    let absolute_original = k_absolute_residuals.clone();
                    for (dst, &src) in tracking_result.permutation.iter().enumerate() {
                        if dst < k_residuals.len() && src < original.len() {
                            k_residuals[dst] = original[src];
                        }
                        if dst < k_absolute_residuals.len() && src < absolute_original.len() {
                            k_absolute_residuals[dst] = absolute_original[src];
                        }
                    }

                    // Log warnings for near-degenerate regions (low sigma_min)
                    if tracking_result.sigma_min < 0.1 {
                        warn!(
                            "[bandstructure] k#{:03} band tracking: near-degeneracy (σ_min={:.4}), may be unreliable",
                            k_idx, tracking_result.sigma_min
                        );
                    }
                }
            }
        }

        crate::profiler::start_timer("run.cache_update");
        // Store eigenvectors for band tracking at next k-point
        prev_eigenvectors = Some(eigenvectors.clone());

        // Store frequencies (ω) for degenerate block disambiguation at next k-point
        prev_omegas = Some(omegas.clone());

        // Store eigenvalues (ω²) for band-window preconditioner shift at next k-point
        // We store the raw eigenvalues (λ = ω²), not frequencies.
        prev_eigenvalues = Some(final_evals.clone());

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

        crate::profiler::stop_timer("run.cache_update");

        // Store first Γ-point result for potential reuse at end of path
        if reuse_gamma && k_idx == 0 && is_gamma {
            first_gamma_omegas = Some(omegas.clone());
            first_gamma_quality = (k_converged, k_residuals.clone(), k_b_orthogonality_defect, k_absolute_residuals.clone());
            first_gamma_fields = options.retain_eigenvectors.then(|| eigenvectors.clone());
        }

        if let Some(ref mut on_k_point) = on_k_point {
            (**on_k_point)(KPointResult {
                discretization: None,
                symmetry: reuse_report,
                k_index: k_idx,
                total_k_points,
                k_point: k_frac,
                distance: distances[k_idx],
                omegas: omegas.clone(),
                iterations: k_iterations,
                converged: k_converged,
                residuals: k_residuals,
                absolute_residuals: k_absolute_residuals,
                b_orthogonality_defect: k_b_orthogonality_defect,
                eigenvectors: options.retain_eigenvectors.then(|| eigenvectors.clone()),
                is_gamma,
                elapsed_seconds: point_start.elapsed_secs(),
                work: crate::work::snapshot().since(work_start),
                preconditioner: reused.is_none().then_some(precond_info),
            });
        }

        bands.push(omegas);
    } // end k-point loop

    // ========================================================================
    // Finalize
    // ========================================================================

    let solve_elapsed = solve_start.elapsed_secs();
    let time_per_iter_ms = if total_iterations > 0 {
        solve_elapsed * 1000.0 / total_iterations as f64
    } else {
        0.0
    };

    let diag_suffix = if study.is_some() {
        " (diagnostics recorded)"
    } else {
        ""
    };
    info!(
        "[bandstructure] complete: {} k-points, {} total iterations, {:.2}s ({:.1}ms/iter){}",
        job.k_path.len(),
        total_iterations,
        solve_elapsed,
        time_per_iter_ms,
        diag_suffix
    );

    // Build initial result
    let result = BandStructureResult {
        k_path: job.k_path.clone(),
        distances,
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
/// Records per-iteration data for each k-point solve into a
/// [`ConvergenceStudy`] that can be serialized to JSON for analysis. This is a
/// thin wrapper over [`run_core`]; the only difference from [`run_with_options`]
/// is that a study recorder is threaded through the solve.
pub fn run_with_diagnostics_and_options<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    study_name: impl Into<String>,
    options: RunOptions,
) -> BandStructureResultWithDiagnostics {
    let mut study = ConvergenceStudy::new(study_name);
    let result = run_core(backend, job, options, Some(&mut study), None, None);
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
