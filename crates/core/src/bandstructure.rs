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
use std::f64::consts::PI;
use std::time::Instant;

use crate::{
    backend::SpectralBackend,
    band_tracking::{apply_permutation, track_bands},
    diagnostics::{ConvergenceStudy, PreconditionerShiftMode, PreconditionerType},
    dielectric::{Dielectric2D, DielectricOptions},
    eigensolver::{Eigensolver, EigensolverConfig},
    field::Field2D,
    geometry::Geometry2D,
    grid::Grid2D,
    operator::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        Parity, SectorSchedule, SymmetryConfig, SymmetryProjector, enumerate_sectors,
        projector_for_sector,
    },
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
    /// Preconditioner type (None, FourierDiagonal, Structured).
    pub precond_type: PreconditionerType,
    /// Preconditioner shift mode (Adaptive or Legacy).
    pub shift_mode: PreconditionerShiftMode,
    /// Use transformed TE mode (similarity transform to standard eigenproblem).
    /// When true, converts Ax = λBx to A'y = λy by applying B^{-1/2} · A · B^{-1/2}.
    /// This simplifies orthogonalization since B = I.
    pub use_transformed_te: bool,
    /// Disable symmetry projections.
    /// When true (default), no symmetry projections are applied even at high-symmetry k-points.
    /// Set to false to enable symmetry projections with multi-sector handling.
    pub disable_symmetry: bool,
    /// Symmetry configuration (parity, tolerance, etc.).
    /// Note: When multi_sector is true, the parity field is ignored
    /// because all sectors are enumerated automatically.
    pub symmetry_config: SymmetryConfig,
    /// Use multi-sector symmetry handling.
    /// When true, runs LOBPCG once per symmetry sector at each k-point
    /// and merges the results. This gives complete, correct bands.
    /// When false (default), symmetry handling is disabled for simpler operation.
    pub multi_sector: bool,
    /// Round-trip mode: traverse the k-path twice in a loop.
    /// The warm start carries over from the first pass, so the second pass
    /// benefits from well-converged initial vectors. Only the second pass
    /// results are returned.
    pub round_trip: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            precond_type: PreconditionerType::default(),
            shift_mode: PreconditionerShiftMode::default(),
            // Default: use UNTRANSFORMED generalized eigenproblem for TE (A x = λ B x, B = ε).
            // The transformed version (A' = ε^{-1/2} A ε^{-1/2}, B' = I) was causing systematic
            // eigenvalue shifts because pointwise ε^{-1/2} is NOT the true matrix square root
            // of the plane-wave mass matrix. See doc/te_operator_report.md for details.
            use_transformed_te: false,
            disable_symmetry: true, // Default: no symmetry projections (simpler, more predictable)
            symmetry_config: SymmetryConfig::default(),
            multi_sector: false, // Default: no multi-sector (requires explicit opt-in)
            round_trip: false,
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

    /// Set the shift mode.
    pub fn with_shift_mode(mut self, shift_mode: PreconditionerShiftMode) -> Self {
        self.shift_mode = shift_mode;
        self
    }

    /// Use legacy global shift (σ = 1e-3).
    pub fn with_legacy_shift(mut self) -> Self {
        self.shift_mode = PreconditionerShiftMode::Legacy;
        self
    }

    /// Use adaptive k-dependent shift (σ(k) = β * s_median).
    pub fn with_adaptive_shift(mut self) -> Self {
        self.shift_mode = PreconditionerShiftMode::Adaptive;
        self
    }

    /// Use transformed TE mode (similarity transform to standard eigenproblem).
    pub fn with_transformed_te(mut self) -> Self {
        self.use_transformed_te = true;
        self
    }

    /// Use generalized TE mode (original Ax = λBx formulation).
    pub fn without_transformed_te(mut self) -> Self {
        self.use_transformed_te = false;
        self
    }

    /// Disable symmetry projections entirely.
    pub fn without_symmetry(mut self) -> Self {
        self.disable_symmetry = true;
        self
    }

    /// Enable symmetry projections (default).
    pub fn with_symmetry(mut self) -> Self {
        self.disable_symmetry = false;
        self
    }

    /// Set symmetry configuration.
    pub fn with_symmetry_config(mut self, config: SymmetryConfig) -> Self {
        self.symmetry_config = config;
        self.disable_symmetry = false;
        self
    }

    /// Set parity for symmetry projection (even or odd).
    /// Note: This is ignored when multi_sector is true (default).
    pub fn with_parity(mut self, parity: Parity) -> Self {
        self.symmetry_config.parity = parity;
        self
    }

    /// Enable multi-sector symmetry handling (default).
    ///
    /// With multi-sector enabled, the solver runs LOBPCG once per symmetry
    /// sector (irreducible representation) at each k-point and merges all
    /// eigenpairs. This gives complete, correct bands comparable to MPB.
    ///
    /// Sectors are determined by the little group of each k-point:
    /// - At Γ: 4 sectors (2 mirrors × 2 parities each)
    /// - On Γ-X: 2 sectors (1 mirror × 2 parities)
    /// - At generic k: 1 sector (full space)
    pub fn with_multi_sector(mut self) -> Self {
        self.multi_sector = true;
        self.disable_symmetry = false;
        self
    }

    /// Disable multi-sector, use single-parity projection (legacy).
    ///
    /// This uses the old behavior where only one parity is projected.
    /// Results will be incomplete (missing modes from other parities).
    pub fn without_multi_sector(mut self) -> Self {
        self.multi_sector = false;
        self
    }

    /// Enable round-trip mode.
    ///
    /// In round-trip mode, the k-path is traversed twice. The first pass
    /// builds up well-converged warm-start vectors, and the second pass
    /// benefits from these improved initial guesses. Only results from the
    /// second pass are returned.
    pub fn with_round_trip(mut self) -> Self {
        self.round_trip = true;
        self
    }

    /// Disable round-trip mode (default).
    pub fn without_round_trip(mut self) -> Self {
        self.round_trip = false;
        self
    }
}

// ============================================================================
// Multi-Sector K-Point Solver
// ============================================================================

/// Result of solving a single k-point with multi-sector symmetry.
///
/// Contains the merged eigenvalues/eigenvectors from all symmetry sectors,
/// sorted by eigenvalue to give the lowest N bands.
#[derive(Debug, Clone)]
pub struct MultiSectorKPointResult {
    /// The eigenvalues (ω²) sorted from lowest to highest.
    pub eigenvalues: Vec<f64>,
    /// The eigenvectors corresponding to the sorted eigenvalues.
    pub eigenvectors: Vec<Field2D>,
    /// Total iterations across all sectors.
    pub total_iterations: usize,
    /// Number of sectors that were solved.
    pub n_sectors: usize,
    /// Whether all sectors converged.
    pub all_converged: bool,
}

/// Solve a single k-point using multi-sector symmetry.
///
/// This function:
/// 1. Enumerates all symmetry sectors for the k-point's little group
/// 2. Runs LOBPCG once per sector (with appropriate projector)
/// 3. Merges all eigenpairs and sorts by eigenvalue
/// 4. Returns the lowest n_bands eigenpairs
///
/// # PARALLELIZATION OPPORTUNITY
/// The sector solves are independent and could be parallelized using rayon:
/// ```ignore
/// sectors.par_iter().map(|sector| solve_sector(...)).collect()
/// ```
/// This would give approximately linear speedup with the number of sectors.
///
/// # Note on Implementation
/// Each sector requires its own mutable operator and preconditioner instances
/// due to Rust's borrow checker. This is handled by the caller providing
/// a factory function or by rebuilding per-sector. For the initial implementation,
/// we rebuild operators per sector. Future optimization could share read-only
/// parts and only rebuild the mutable state.
fn solve_k_point_multi_sector<B: SpectralBackend + Clone>(
    backend: &B,
    dielectric: &Dielectric2D,
    pol: Polarization,
    bloch: [f64; 2],
    use_transformed_te: bool,
    eigensolver_config: &EigensolverConfig,
    precond_type: PreconditionerType,
    shift_mode: PreconditionerShiftMode,
    warm_slice: Option<&[Field2D]>,
    schedule: &SectorSchedule,
    grid: Grid2D,
    _verbosity: Verbosity,
    k_idx: usize,
) -> MultiSectorKPointResult {
    let n_bands = eigensolver_config.n_bands;
    let n_sectors = schedule.n_sectors();

    // Storage for all eigenpairs from all sectors
    // Each entry: (eigenvalue, eigenvector)
    let mut all_eigenpairs: Vec<(f64, Field2D)> = Vec::with_capacity(n_bands * n_sectors);
    let mut total_iterations = 0usize;
    let mut all_converged = true;

    // ========================================================================
    // PARALLELIZATION POINT: This loop over sectors is embarrassingly parallel.
    // Each sector is independent and could run on a separate thread/core.
    // With rayon:
    //   schedule.sectors.par_iter().enumerate().map(|(sector_idx, sector)| {
    //       // Create operator, preconditioner, solver for this sector
    //       // Return (eigenvalues, eigenvectors, iterations, converged)
    //   }).collect()
    // ========================================================================
    for (sector_idx, sector) in schedule.sectors.iter().enumerate() {
        // Build projector for this sector (None for trivial sector)
        // NOTE: Skip symmetry projection for transformed TE mode - the projection
        // doesn't commute with the ε^{1/2} similarity transform.
        let projector = if use_transformed_te && pol == Polarization::TE {
            None
        } else {
            projector_for_sector(grid, sector)
        };

        if n_sectors > 1 {
            debug!(
                "[bandstructure] k#{:03} sector {}/{}: {}",
                k_idx,
                sector_idx + 1,
                n_sectors,
                sector.label
            );
        }

        // Create a fresh operator for this sector
        // This is necessary because the Eigensolver takes &mut operator
        let mut theta = if use_transformed_te && pol == Polarization::TE {
            ThetaOperator::new_transformed(backend.clone(), dielectric.clone(), pol, bloch)
        } else {
            ThetaOperator::new(backend.clone(), dielectric.clone(), pol, bloch)
        };

        // Build preconditioner for this sector
        let mut preconditioner_opt: Option<Box<dyn crate::preconditioner::OperatorPreconditioner<B>>> =
            match (precond_type, shift_mode) {
                (PreconditionerType::Auto, _) => unreachable!("Auto should be resolved before this point"),
                (PreconditionerType::None, _) => None,
                (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Adaptive)
                | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Adaptive) => {
                    Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                }
                (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Legacy)
                | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Legacy) => {
                    Some(Box::new(theta.build_homogeneous_preconditioner()))
                }
                (PreconditionerType::Structured, PreconditionerShiftMode::Adaptive) => {
                    Some(Box::new(theta.build_structured_preconditioner_adaptive()))
                }
                (PreconditionerType::Structured, PreconditionerShiftMode::Legacy) => {
                    Some(Box::new(theta.build_structured_preconditioner()))
                }
                (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Adaptive) => {
                    Some(Box::new(theta.build_transverse_projection_preconditioner_adaptive()))
                }
                (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Legacy) => {
                    Some(Box::new(theta.build_transverse_projection_preconditioner()))
                }
            };

        // Create and run the eigensolver for this sector
        let sector_config = eigensolver_config.clone();
        let mut solver = Eigensolver::new(
            &mut theta,
            sector_config,
            preconditioner_opt
                .as_mut()
                .map(|p| &mut **p as &mut dyn crate::preconditioner::OperatorPreconditioner<B>),
            warm_slice,
        );

        // Apply sector projector if non-trivial
        if let Some(proj) = projector {
            solver = solver.with_symmetry_projector(proj);
        }

        let result = solver.solve();
        total_iterations += result.iterations;
        all_converged = all_converged && result.converged;

        // Collect eigenpairs from this sector
        // Use all_eigenvectors() to include deflated vectors (e.g., Γ constant mode)
        let eigenvectors = solver.all_eigenvectors();
        for (band_idx, (&eigenvalue, eigenvector)) in result
            .eigenvalues
            .iter()
            .zip(eigenvectors.into_iter())
            .enumerate()
        {
            // Include eigenvalue = 0 (Γ constant mode) - it's physically valid
            if eigenvalue >= 0.0 {
                all_eigenpairs.push((eigenvalue, eigenvector));
            }

            if band_idx >= n_bands {
                break;
            }
        }
    }

    // Sort all eigenpairs by eigenvalue (ascending)
    all_eigenpairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take the lowest n_bands
    all_eigenpairs.truncate(n_bands);

    // Separate into eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors): (Vec<f64>, Vec<Field2D>) = all_eigenpairs.into_iter().unzip();

    if n_sectors > 1 {
        let omega_min = eigenvalues.first().map(|&ev| ev.sqrt()).unwrap_or(0.0);
        let omega_max = eigenvalues.last().map(|&ev| ev.sqrt()).unwrap_or(0.0);
        debug!(
            "[bandstructure] k#{:03} multi-sector merged: {} sectors, {} eigenpairs, ω=[{:.4}..{:.4}]",
            k_idx,
            n_sectors,
            eigenvalues.len(),
            omega_min,
            omega_max
        );
    }

    MultiSectorKPointResult {
        eigenvalues,
        eigenvectors,
        total_iterations,
        n_sectors,
        all_converged,
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
    verbosity: Verbosity,
    options: RunOptions,
) -> BandStructureResult {
    // ========================================================================
    // Setup Phase
    // ========================================================================

    // Resolve Auto preconditioner type based on polarization
    let precond_type = options.precond_type.resolve_for_polarization(job.pol);
    let shift_mode = options.shift_mode;

    // Determine number of passes (1 normally, 2 for round-trip)
    let n_passes = if options.round_trip { 2 } else { 1 };

    let symmetry_mode = if options.disable_symmetry {
        "disabled"
    } else if options.multi_sector {
        "multi-sector"
    } else {
        "single-parity"
    };

    info!(
        "[bandstructure] grid={}x{} pol={:?} bands={} k_points={} symmetry={}{}",
        job.grid.nx,
        job.grid.ny,
        job.pol,
        job.eigensolver.n_bands,
        job.k_path.len(),
        symmetry_mode,
        if options.round_trip { " round-trip" } else { "" }
    );

    // Sample the dielectric function from geometry
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    // Compute dielectric contrast for diagnostics
    let eps = dielectric.eps();
    let eps_min = eps.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_contrast = if eps_min > 1e-15 { eps_max / eps_min } else { f64::INFINITY };

    info!(
        "[bandstructure] dielectric: ε=[{:.3}, {:.3}] contrast={:.1}x precond={:?}",
        eps_min,
        eps_max,
        eps_contrast,
        precond_type,
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
            let bloch = [2.0 * PI * k_frac[0], 2.0 * PI * k_frac[1]];

            // Create temporary operator for diagnostics
            let mut theta = if options.use_transformed_te && job.pol == Polarization::TE {
                ThetaOperator::new_transformed(
                    backend.clone(),
                    dielectric.clone(),
                    job.pol,
                    bloch,
                )
            } else {
                ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch)
            };

            // Build preconditioner for diagnostics
            let mut preconditioner_opt: Option<Box<dyn crate::preconditioner::OperatorPreconditioner<B>>> =
                match (precond_type, shift_mode) {
                    (PreconditionerType::Auto, _) => unreachable!("Auto should be resolved before this point"),
                    (PreconditionerType::None, _) => None,
                    (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Adaptive)
                    | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                    }
                    (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Legacy)
                    | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_homogeneous_preconditioner()))
                    }
                    (PreconditionerType::Structured, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_structured_preconditioner_adaptive()))
                    }
                    (PreconditionerType::Structured, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_structured_preconditioner()))
                    }
                    (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_transverse_projection_preconditioner_adaptive()))
                    }
                    (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_transverse_projection_preconditioner()))
                    }
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
            let (lambda_max, lambda_min, kappa) =
                theta.estimate_condition_number(POWER_ITERATIONS);

            if let Some(ref mut precond) = preconditioner_opt {
                let (_pm_max, _pm_min, pm_kappa) =
                    theta.estimate_preconditioned_condition_number(&mut **precond, POWER_ITERATIONS);
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
    // K-Point Loop (with optional round-trip)
    // ========================================================================

    // Storage for warm-start vectors from previous k-point
    let warm_start_limit = job.eigensolver.n_bands;
    let mut warm_start_store: Vec<Field2D> = Vec::new();

    // Storage for band tracking: eigenvectors from previous k-point
    // Note: When starting at Γ with proper deflation, Γ eigenvectors are reliable
    // and can be used for warm-starting subsequent k-points.
    let mut prev_eigenvectors: Option<Vec<Field2D>> = None;

    // Get dielectric epsilon for B-weighted overlaps
    // - TM mode: B = I, use standard inner product (None)
    // - TE mode (generalized): B = ε, use ε-weighted inner product
    // - TE mode (transformed): B = I, use standard inner product (None)
    let eps_for_tracking: Option<Vec<f64>> =
        if job.pol == Polarization::TE && !options.use_transformed_te {
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

    // Accumulate results (only final pass results are kept)
    let mut bands: Vec<Vec<f64>> = Vec::with_capacity(job.k_path.len());
    let mut total_iterations = 0usize;

    // Start timing the solve phase (excludes setup/diagnostics)
    let solve_start = Instant::now();

    // Outer loop for round-trip: pass 0 = warm-up, pass 1 = final results
    for pass in 0..n_passes {
        // Clear bands at start of each pass (only last pass is kept)
        bands.clear();

        // For warm-up pass, use minimal iterations (just enough to get reasonable warm-start vectors)
        let is_warmup_pass = n_passes > 1 && pass == 0;
        const WARMUP_ITERATIONS: usize = 5;

        if n_passes > 1 {
            let pass_label = if is_warmup_pass { "warm-up" } else { "final" };
            debug!("[bandstructure] starting pass {}/{} ({})", pass + 1, n_passes, pass_label);
        }

        for (k_idx, &k_frac) in job.k_path.iter().enumerate() {
            // Check if this is a Γ-point (k ≈ 0)
            // At Γ, the constant mode (DC) is deflated by the eigensolver.
            let is_gamma = is_gamma_point(k_frac);

            // Reuse first Γ-point result for the last k-point (avoid duplicate solve)
            // This is valid when the path loops back to Γ (e.g., Γ→X→M→Γ)
            if reuse_gamma && k_idx == last_k_idx && !is_warmup_pass {
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
                debug!("[bandstructure] k#{:03} is Γ-point: constant mode will be deflated", k_idx);
            }

            // Convert fractional k-point to Bloch wavevector (in 2π/a units)
            let bloch = [2.0 * PI * k_frac[0], 2.0 * PI * k_frac[1]];

            // Prepare warm-start slice (if available from previous k-point)
            // Note: Warm-start from Γ is now enabled - eigenvectors from Γ (with deflation)
            // provide useful starting guesses even though the subspaces differ at k≠0.
            let warm_slice: Option<&[Field2D]> = if !warm_start_store.is_empty() {
                Some(warm_start_store.as_slice())
            } else {
                None
            };

            // Configure eigensolver - use reduced iterations for warm-up pass
            let mut eigensolver_config = if is_warmup_pass {
                let mut config = job.eigensolver.clone();
                config.max_iter = WARMUP_ITERATIONS;
                config
            } else {
                job.eigensolver.clone()
            };
            // Set k-point index for logging
            eigensolver_config.k_index = Some(k_idx);

            // Determine the lattice class for symmetry detection
            let lattice_class = job.geom.lattice.classify();

            // ====================================================================
            // Dispatch: Multi-Sector vs Single-Sector (Legacy)
            // ====================================================================
            // NOTE: Skip multi-sector for transformed TE mode - the symmetry projection
            // doesn't commute with the ε^{1/2} similarity transform, so we must use
            // the single-sector path without symmetry projection.
            let use_multi_sector = options.multi_sector
                && !options.disable_symmetry
                && !(options.use_transformed_te && job.pol == Polarization::TE);

            let (mut omegas, mut eigenvectors, k_iterations) = if use_multi_sector
            {
                // Multi-sector path: enumerate sectors and solve each
                let schedule = enumerate_sectors(
                    k_frac,
                    lattice_class,
                    options.symmetry_config.bloch_tolerance,
                );

                if schedule.n_sectors() > 1 {
                    debug!(
                        "[bandstructure] k#{:03} ({:+.4},{:+.4}): {} sectors to solve",
                        k_idx,
                        k_frac[0],
                        k_frac[1],
                        schedule.n_sectors()
                    );
                }

                let ms_result = solve_k_point_multi_sector(
                    &backend,
                    &dielectric,
                    job.pol,
                    bloch,
                    options.use_transformed_te,
                    &eigensolver_config,
                    precond_type,
                    shift_mode,
                    warm_slice,
                    &schedule,
                    job.grid,
                    verbosity,
                    k_idx,
                );

                // Convert eigenvalues to frequencies
                let omegas: Vec<f64> = ms_result
                    .eigenvalues
                    .iter()
                    .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                    .collect();

                // Note: Per-k-point logging is handled by the eigensolver via log crate

                (omegas, ms_result.eigenvectors, ms_result.total_iterations)
            } else {
                // Single-sector path (legacy behavior or symmetry disabled)
                // Construct the Maxwell operator Θ for this k-point
                let mut theta = if options.use_transformed_te && job.pol == Polarization::TE {
                    ThetaOperator::new_transformed(
                        backend.clone(),
                        dielectric.clone(),
                        job.pol,
                        bloch,
                    )
                } else {
                    ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch)
                };

                // Build preconditioner for this operator based on options
                let mut preconditioner_opt: Option<Box<dyn crate::preconditioner::OperatorPreconditioner<B>>> =
                    match (precond_type, shift_mode) {
                        (PreconditionerType::Auto, _) => unreachable!("Auto should be resolved before this point"),
                        (PreconditionerType::None, _) => None,
                        (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Adaptive)
                        | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Adaptive) => {
                            Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                        }
                        (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Legacy)
                        | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Legacy) => {
                            Some(Box::new(theta.build_homogeneous_preconditioner()))
                        }
                        (PreconditionerType::Structured, PreconditionerShiftMode::Adaptive) => {
                            Some(Box::new(theta.build_structured_preconditioner_adaptive()))
                        }
                        (PreconditionerType::Structured, PreconditionerShiftMode::Legacy) => {
                            Some(Box::new(theta.build_structured_preconditioner()))
                        }
                        (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Adaptive) => {
                            Some(Box::new(theta.build_transverse_projection_preconditioner_adaptive()))
                        }
                        (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Legacy) => {
                            Some(Box::new(theta.build_transverse_projection_preconditioner()))
                        }
                    };

                // Create symmetry projector for this k-point (legacy single-parity mode)
                // NOTE: Skip symmetry projection for transformed TE mode - the projection
                // doesn't commute with the ε^{1/2} similarity transform.
                let symmetry_projector = if options.disable_symmetry
                    || (options.use_transformed_te && job.pol == Polarization::TE)
                {
                    None
                } else {
                    SymmetryProjector::for_k_point(
                        job.grid,
                        k_frac,
                        &options.symmetry_config,
                        lattice_class,
                    )
                };

                // Create and run the eigensolver
                let mut solver = Eigensolver::new(
                    &mut theta,
                    eigensolver_config,
                    preconditioner_opt
                        .as_mut()
                        .map(|p| &mut **p as &mut dyn crate::preconditioner::OperatorPreconditioner<B>),
                    warm_slice,
                );

                // Apply symmetry projector if one was created for this k-point
                if let Some(projector) = symmetry_projector {
                    solver = solver.with_symmetry_projector(projector);
                }

                let result = solver.solve();
                let k_iterations = result.iterations;

                // Convert eigenvalues (ω²) to frequencies (ω)
                let omegas: Vec<f64> = result
                    .eigenvalues
                    .iter()
                    .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
                    .collect();

                // Note: Per-k-point logging is handled by the eigensolver via log crate

                // Use all_eigenvectors() to include deflated vectors (e.g., Γ constant mode)
                let eigenvectors = solver.all_eigenvectors();

                (omegas, eigenvectors, k_iterations)
            };

            total_iterations += k_iterations;

            // ====================================================================
            // Band Tracking: reorder by overlap with previous k-point
            // ====================================================================
            // Skip tracking if we don't have valid reference eigenvectors.
            // Γ-point eigenvectors are unreliable due to operator singularity.
            //
            // MULTI-SECTOR NOTE: When using multi-sector symmetry, bands are already
            // sorted by eigenvalue across all sectors. Eigenvector-based tracking fails
            // because vectors from different sectors are orthogonal (overlap ≈ 0).
            // We skip the permutation step in multi-sector mode.
            let is_multi_sector_active = options.multi_sector && !options.disable_symmetry;
            if let Some(ref prev_vecs) = prev_eigenvectors {
                let tracking_result = track_bands(
                    &backend,
                    prev_vecs,
                    &eigenvectors,
                    eps_for_tracking.as_deref(),
                );

                // Only apply permutation in single-sector mode
                // In multi-sector mode, eigenvalue ordering is authoritative
                if tracking_result.had_swaps && !is_multi_sector_active {
                    apply_permutation(&tracking_result.permutation, &mut omegas, &mut eigenvectors);

                    // Log low overlap warnings via log crate
                    if tracking_result.min_overlap < 0.1 {
                        warn!(
                            "[bandstructure] k#{:03} band tracking: low overlap ({:.4}), may be unreliable",
                            k_idx, tracking_result.min_overlap
                        );
                    } else {
                        debug!(
                            "[bandstructure] k#{:03} band tracking: swaps applied, min_overlap={:.4}",
                            k_idx, tracking_result.min_overlap
                        );
                    }
                }
            }

            // Store eigenvectors for band tracking at next k-point
            // With proper Γ-point deflation (constant mode removed), the eigenvectors
            // are reliable and can be used for band tracking and warm-starting.
            prev_eigenvectors = Some(eigenvectors.clone());

            // Store eigenvectors for warm-starting the next k-point
            // Use tracked order (already reordered by band identity)
            warm_start_store.clear();
            for vec in eigenvectors.iter().take(warm_start_limit) {
                warm_start_store.push(vec.clone());
            }

            // Store first Γ-point result for potential reuse at end of path
            if reuse_gamma && k_idx == 0 && is_gamma && !is_warmup_pass {
                first_gamma_omegas = Some(omegas.clone());
            }

            bands.push(omegas);
        } // end k-point loop
    } // end pass loop

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
/// Uses legacy shift mode for backward compatibility.
///
/// # Arguments
///
/// - `backend`: The spectral backend (CPU, CUDA, etc.)
/// - `job`: The band-structure job configuration
/// - `verbosity`: Controls progress output
/// - `study_name`: Name for the convergence study
/// - `precond_type`: Type of preconditioner to use (None, FourierDiagonal, Structured)
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
        RunOptions::new()
            .with_preconditioner(precond_type)
            .with_legacy_shift(),
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
    let shift_mode = options.shift_mode;

    // Determine number of passes (1 normally, 2 for round-trip)
    let n_passes = if options.round_trip { 2 } else { 1 };

    info!(
        "[bandstructure] grid={}x{} pol={:?} bands={} k_points={} (diagnostics={}){}",
        job.grid.nx,
        job.grid.ny,
        job.pol,
        job.eigensolver.n_bands,
        job.k_path.len(),
        study_name,
        if options.round_trip { " round-trip" } else { "" }
    );

    debug!(
        "[bandstructure] precond={:?} shift={:?} w_history={} locking={}",
        precond_type,
        shift_mode,
        job.eigensolver.use_w_history,
        job.eigensolver.use_locking
    );

    // Sample the dielectric function from geometry
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    // Compute dielectric contrast for diagnostics
    let eps = dielectric.eps();
    let eps_min = eps.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_contrast = if eps_min > 1e-15 { eps_max / eps_min } else { f64::INFINITY };

    info!(
        "[bandstructure] dielectric: ε=[{:.3}, {:.3}] contrast={:.1}x precond={:?}",
        eps_min,
        eps_max,
        eps_contrast,
        precond_type,
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
            let bloch = [2.0 * PI * k_frac[0], 2.0 * PI * k_frac[1]];

            // Create temporary operator for diagnostics
            let mut theta = if options.use_transformed_te && job.pol == Polarization::TE {
                ThetaOperator::new_transformed(
                    backend.clone(),
                    dielectric.clone(),
                    job.pol,
                    bloch,
                )
            } else {
                ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch)
            };

            // Build preconditioner for diagnostics
            let mut preconditioner_opt: Option<Box<dyn crate::preconditioner::OperatorPreconditioner<B>>> =
                match (precond_type, shift_mode) {
                    (PreconditionerType::Auto, _) => unreachable!("Auto should be resolved before this point"),
                    (PreconditionerType::None, _) => None,
                    (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Adaptive)
                    | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                    }
                    (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Legacy)
                    | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_homogeneous_preconditioner()))
                    }
                    (PreconditionerType::Structured, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_structured_preconditioner_adaptive()))
                    }
                    (PreconditionerType::Structured, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_structured_preconditioner()))
                    }
                    (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_transverse_projection_preconditioner_adaptive()))
                    }
                    (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_transverse_projection_preconditioner()))
                    }
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
            let (lambda_max, lambda_min, kappa) =
                theta.estimate_condition_number(POWER_ITERATIONS);

            if let Some(ref mut precond) = preconditioner_opt {
                let (_pm_max, _pm_min, pm_kappa) =
                    theta.estimate_preconditioned_condition_number(&mut **precond, POWER_ITERATIONS);
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
    // K-Point Loop (with optional round-trip)
    // ========================================================================

    // Storage for warm-start vectors from previous k-point
    let warm_start_limit = job.eigensolver.n_bands;
    let mut warm_start_store: Vec<Field2D> = Vec::new();

    // Track if previous k-point was Γ (to skip warm-start after Γ)
    // Γ eigenvectors span a different subspace that doesn't overlap well with k≠0
    let mut prev_was_gamma = false;

    // Storage for band tracking: eigenvectors from previous k-point
    // Note: When starting at Γ with proper deflation, Γ eigenvectors are reliable
    // and can be used for warm-starting subsequent k-points.
    let mut prev_eigenvectors: Option<Vec<Field2D>> = None;

    // Get dielectric epsilon for B-weighted overlaps
    // - TM mode: B = I, use standard inner product (None)
    // - TE mode (generalized): B = ε, use ε-weighted inner product
    // - TE mode (transformed): B = I, use standard inner product (None)
    let eps_for_tracking: Option<Vec<f64>> =
        if job.pol == Polarization::TE && !options.use_transformed_te {
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

    // Accumulate results (only final pass results are kept)
    let mut bands: Vec<Vec<f64>> = Vec::with_capacity(job.k_path.len());
    let mut total_iterations = 0usize;

    // Start timing the solve phase (excludes setup/diagnostics)
    let solve_start = Instant::now();

    // Outer loop for round-trip: pass 0 = warm-up, pass 1 = final results
    for pass in 0..n_passes {
        // Clear bands at start of each pass (only last pass is kept)
        bands.clear();

        // For diagnostics, only record the final pass
        let is_final_pass = pass == n_passes - 1;

        // For warm-up pass, use minimal iterations (just enough to get reasonable warm-start vectors)
        let is_warmup_pass = n_passes > 1 && pass == 0;
        const WARMUP_ITERATIONS: usize = 5;

        if n_passes > 1 {
            let pass_label = if is_warmup_pass { "warm-up" } else { "final" };
            debug!("[bandstructure] starting pass {}/{} ({})", pass + 1, n_passes, pass_label);
        }

        for (k_idx, &k_frac) in job.k_path.iter().enumerate() {
            // Check if this is a Γ-point (k ≈ 0)
            // At Γ, the constant mode (DC) is deflated by the eigensolver.
            let is_gamma = is_gamma_point(k_frac);

            // Reuse first Γ-point result for the last k-point (avoid duplicate solve)
            // This is valid when the path loops back to Γ (e.g., Γ→X→M→Γ)
            if reuse_gamma && k_idx == last_k_idx && !is_warmup_pass {
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
                debug!("[bandstructure] k#{:03} is Γ-point: constant mode will be deflated", k_idx);
            }

            // Convert fractional k-point to Bloch wavevector (in 2π/a units)
            let bloch = [2.0 * PI * k_frac[0], 2.0 * PI * k_frac[1]];

            // Construct the Maxwell operator Θ for this k-point
            let mut theta = if options.use_transformed_te && job.pol == Polarization::TE {
                ThetaOperator::new_transformed(backend.clone(), dielectric.clone(), job.pol, bloch)
            } else {
                ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch)
            };

            // Track if warm start was used (before we borrow)
            // Skip warm-start when coming from Γ: those eigenvectors don't overlap well with k≠0
            let warm_start_used = if prev_was_gamma {
                debug!("[bandstructure] k#{:03}: skipping warm-start (previous was Γ-point)", k_idx);
                false
            } else {
                !warm_start_store.is_empty()
            };

            // Create label for this k-point run
            let run_label = format!("{}_k{:03}", study_name, k_idx);

            // Create eigensolver config with diagnostics enabled
            // Use reduced iterations for warm-up pass
            let mut eigensolver_config = job.eigensolver.clone();
            eigensolver_config.record_diagnostics = true;
            if is_warmup_pass {
                eigensolver_config.max_iter = WARMUP_ITERATIONS;
            }

            // Build preconditioner based on options
            let mut preconditioner_opt: Option<Box<dyn crate::preconditioner::OperatorPreconditioner<B>>> =
                match (precond_type, shift_mode) {
                    (PreconditionerType::Auto, _) => unreachable!("Auto should be resolved before this point"),
                    (PreconditionerType::None, _) => None,
                    (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Adaptive)
                    | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_homogeneous_preconditioner_adaptive()))
                    }
                    (PreconditionerType::FourierDiagonal, PreconditionerShiftMode::Legacy)
                    | (PreconditionerType::KernelCompensated, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_homogeneous_preconditioner()))
                    }
                    (PreconditionerType::Structured, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_structured_preconditioner_adaptive()))
                    }
                    (PreconditionerType::Structured, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_structured_preconditioner()))
                    }
                    (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Adaptive) => {
                        Some(Box::new(theta.build_transverse_projection_preconditioner_adaptive()))
                    }
                    (PreconditionerType::TransverseProjection, PreconditionerShiftMode::Legacy) => {
                        Some(Box::new(theta.build_transverse_projection_preconditioner()))
                    }
                };

            // Create and run the eigensolver
            let warm_slice: Option<&[Field2D]> = if warm_start_used {
                Some(warm_start_store.as_slice())
            } else {
                None
            };

            // Create symmetry projector for this k-point (if symmetry is enabled)
            // NOTE: Skip symmetry projection for transformed TE mode - the projection
            // doesn't commute with the ε^{1/2} similarity transform.
            let symmetry_projector = if options.disable_symmetry
                || (options.use_transformed_te && job.pol == Polarization::TE)
            {
                None
            } else {
                let lattice_class = job.geom.lattice.classify();
                SymmetryProjector::for_k_point(
                    job.grid,
                    k_frac,
                    &options.symmetry_config,
                    lattice_class,
                )
            };

            let mut solver = Eigensolver::new(
                &mut theta,
                eigensolver_config,
                preconditioner_opt
                    .as_mut()
                    .map(|p| &mut **p as &mut dyn crate::preconditioner::OperatorPreconditioner<B>),
                warm_slice,
            );

            // Apply symmetry projector if one was created for this k-point
            if let Some(projector) = symmetry_projector {
                solver = solver.with_symmetry_projector(projector);
            }

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
            // ====================================================================
            // Skip tracking if we don't have valid reference eigenvectors.
            // Γ-point eigenvectors are unreliable due to operator singularity.
            if let Some(ref prev_vecs) = prev_eigenvectors {
                let tracking_result = track_bands(
                    &backend,
                    prev_vecs,
                    &eigenvectors,
                    eps_for_tracking.as_deref(),
                );

                if tracking_result.had_swaps {
                    apply_permutation(&tracking_result.permutation, &mut omegas, &mut eigenvectors);

                    // Log low overlap warnings unconditionally via log crate
                    if tracking_result.min_overlap < 0.1 {
                        warn!(
                            "k#{:03} band tracking: low overlap ({:.4}), tracking may be unreliable",
                            k_idx, tracking_result.min_overlap
                        );
                    }

                    debug!(
                        "[bandstructure] k#{:03} band tracking: swaps detected, min_overlap={:.4}",
                        k_idx, tracking_result.min_overlap
                    );
                }
            }

            // Store eigenvectors for band tracking at next k-point
            // With proper Γ-point deflation (constant mode removed), the eigenvectors
            // are reliable and can be used for band tracking and warm-starting.
            prev_eigenvectors = Some(eigenvectors.clone());

            // Store eigenvectors for warm-starting the next k-point
            // Use tracked order (already reordered by band identity)
            warm_start_store.clear();
            for vec in eigenvectors.iter().take(warm_start_limit) {
                warm_start_store.push(vec.clone());
            }

            // Track if this k-point was Γ for next iteration
            prev_was_gamma = is_gamma;

            total_iterations += diag_result.result.iterations;

            // Update the diagnostics with k-point info and add to study
            // Only record diagnostics on the final pass
            if is_final_pass {
                let mut run_data = diag_result.diagnostics;
                run_data.config.k_index = Some(k_idx);
                run_data.config.k_point = Some(k_frac);
                run_data.config.bloch = Some(bloch);
                run_data.config.polarization = Some(format!("{:?}", job.pol));
                run_data.config.preconditioner_type = options.precond_type;
                run_data.config.warm_start_enabled = warm_start_used;
                study.add_run(run_data);
            }

            // Store first Γ-point result for potential reuse at end of path
            if reuse_gamma && k_idx == 0 && is_gamma && !is_warmup_pass {
                first_gamma_omegas = Some(omegas.clone());
            }

            bands.push(omegas);
        } // end k-point loop
    } // end pass loop

    // ========================================================================
    // Finalize
    // ========================================================================

    let solve_elapsed = solve_start.elapsed().as_secs_f64();
    let time_per_iter_ms = if total_iterations > 0 {
        solve_elapsed * 1000.0 / total_iterations as f64
    } else {
        0.0
    };

    {
        let pass_info = if n_passes > 1 {
            format!(" ({} passes)", n_passes)
        } else {
            String::new()
        };
        info!(
            "[bandstructure] complete: {} k-points, {} total iterations, {:.2}s ({:.1}ms/iter) (diagnostics recorded){}",
            job.k_path.len(),
            total_iterations,
            solve_elapsed,
            time_per_iter_ms,
            pass_info
        );
    }

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
