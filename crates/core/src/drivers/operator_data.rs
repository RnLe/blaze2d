//! Envelope-approximation Hamiltonian extraction driver.
//!
//! This driver orchestrates the complete pipeline for extracting EA Hamiltonian
//! ingredients at a given (R, k₀) point:
//!
//! 1. Build dielectric and (optionally) dielectric derivatives
//! 2. Construct ThetaOperator at k₀
//! 3. Solve the eigenproblem (LOBPCG)
//! 4. Extract EA ingredients (velocity, mass tensor, Born–Huang, etc.)
//!
//! # Example
//!
//! ```ignore
//! use blaze2d_core::drivers::operator_data::{run, OperatorDataJob};
//!
//! let job = OperatorDataJob {
//!     geom: geometry,
//!     grid: Grid2D::new(32, 32, 1.0, 1.0),
//!     pol: Polarization::TE,
//!     k0: [0.0, 0.0],
//!     registry: [0.0, 0.0],
//!     operator_data_config: OperatorDataConfig::default(),
//!     eigensolver: EigensolverConfig::default(),
//!     dielectric: DielectricOptions::default(),
//!     fd_step: 0.001,
//!     atom_index: 0,
//!     compute_dielectric_derivatives: true,
//! };
//! let result = run(backend, &job);
//! ```

use crate::backend::SpectralBackend;
use crate::band_tracking::{apply_permutation, track_bands_with_frequencies};
use crate::dielectric::{Dielectric2D, DielectricDerivative, DielectricOptions};
use crate::operator_data::{OperatorDataConfig, OperatorDataExtractor, OperatorData};
use crate::eigensolver::{Eigensolver, EigensolverConfig};
use crate::field::Field2D;
use crate::geometry::Geometry2D;
use crate::grid::Grid2D;
use crate::operators::maxwell::ThetaOperator;
use crate::polarization::Polarization;
use crate::preconditioners::OperatorPreconditioner;

// ============================================================================
// Job Configuration
// ============================================================================

/// Full job specification for EA Hamiltonian extraction at one (R, k₀) point.
#[derive(Debug, Clone)]
pub struct OperatorDataJob {
    /// The 2D photonic crystal geometry (with atoms at the registry position).
    pub geom: Geometry2D,
    /// Computational grid.
    pub grid: Grid2D,
    /// Polarization mode.
    pub pol: Polarization,
    /// Carrier momentum k₀ in Cartesian reciprocal-space units (2π/a).
    pub k0: [f64; 2],
    /// Registry point (fractional atom shift coordinates) — metadata only.
    /// The geometry should already have atoms at the correct positions.
    pub registry: [f64; 2],
    /// EA extraction configuration (n_retained, n_remote, what to compute).
    pub operator_data_config: OperatorDataConfig,
    /// Eigensolver configuration.
    pub eigensolver: EigensolverConfig,
    /// Dielectric construction options.
    pub dielectric: DielectricOptions,
    /// Finite-difference step for dielectric derivatives (fractional coords).
    pub fd_step: f64,
    /// Which atom to differentiate w.r.t. for R-derivatives.
    pub atom_index: usize,
    /// Whether to compute dielectric derivatives (needed for R-derivatives and Born–Huang).
    pub compute_dielectric_derivatives: bool,
}

impl OperatorDataJob {
    /// Total number of bands to solve for.
    pub fn n_total_bands(&self) -> usize {
        self.operator_data_config.n_retained + self.operator_data_config.n_remote
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of an EA extraction run.
#[derive(Debug, Clone)]
pub struct OperatorDataDriverResult {
    /// The extracted EA Hamiltonian ingredients.
    pub ingredients: OperatorData,
    /// Total wall-clock time for the eigensolver (seconds).
    pub solve_time_seconds: f64,
    /// Total wall-clock time for the extraction (seconds).
    pub extract_time_seconds: f64,
}

// ============================================================================
// Driver Functions
// ============================================================================

/// Run the full EA extraction pipeline at a single (R, k₀) point.
///
/// This is the main entry point. It:
/// 1. Builds the dielectric from geometry
/// 2. Optionally computes dielectric derivatives via FD
/// 3. Constructs ThetaOperator at k₀
/// 4. Solves for eigenvalues/eigenvectors
/// 5. Extracts all requested EA ingredients
pub fn run<B: SpectralBackend>(backend: B, job: &OperatorDataJob) -> OperatorDataDriverResult {
    run_with_reference(backend, job, None)
}

/// Run the EA extraction with optional reference eigenvectors for overlap matrix.
///
/// The reference eigenvectors are from a neighboring R-point, used to build
/// the gauge-transport overlap matrix S_{mn} = ⟨uₘ(R)|uₙ(R')⟩.
pub fn run_with_reference<B: SpectralBackend>(
    backend: B,
    job: &OperatorDataJob,
    reference_eigvecs: Option<&[Field2D]>,
) -> OperatorDataDriverResult {
    let n_total = job.n_total_bands();

    // 1. Build dielectric
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    // 2. Compute dielectric derivatives (if requested)
    let diel_derivs = if job.compute_dielectric_derivatives {
        Some([
            DielectricDerivative::from_finite_difference(
                &job.geom,
                job.grid,
                &job.dielectric,
                job.atom_index,
                0, // x-direction
                job.fd_step,
            ),
            DielectricDerivative::from_finite_difference(
                &job.geom,
                job.grid,
                &job.dielectric,
                job.atom_index,
                1, // y-direction
                job.fd_step,
            ),
        ])
    } else {
        None
    };

    // 3. Construct operator at k₀
    let mut theta = ThetaOperator::new(backend, dielectric, job.pol, job.k0);

    // 4. Build preconditioner and solve
    let mut preconditioner: Box<dyn OperatorPreconditioner<B>> =
        Box::new(theta.build_homogeneous_preconditioner_adaptive());

    let mut eigensolver_config = job.eigensolver.clone();
    // Ensure we solve for enough bands
    if eigensolver_config.n_bands < n_total {
        eigensolver_config.n_bands = n_total;
    }

    let solve_start = std::time::Instant::now();
    let mut solver = Eigensolver::new(
        &mut theta,
        eigensolver_config,
        Some(&mut *preconditioner as &mut dyn OperatorPreconditioner<B>),
        None,
    );
    let result = solver.solve();
    let eigenvectors = solver.all_eigenvectors();
    let solve_time = solve_start.elapsed().as_secs_f64();

    let n_iterations = result.iterations;
    let converged = result.converged;
    let eigenvalues = result.eigenvalues;

    // 5. Extract EA ingredients
    let extract_start = std::time::Instant::now();
    let mut extractor = OperatorDataExtractor::new(
        &mut theta,
        &eigenvectors,
        &eigenvalues,
        job.operator_data_config.clone(),
    );

    let ingredients = extractor.extract(
        job.k0,
        job.registry,
        diel_derivs.as_ref(),
        reference_eigvecs,
        n_iterations,
        converged,
    );
    let extract_time = extract_start.elapsed().as_secs_f64();

    OperatorDataDriverResult {
        ingredients,
        solve_time_seconds: solve_time,
        extract_time_seconds: extract_time,
    }
}

/// Run EA extraction with warm-start from previous eigenvectors.
///
/// This is useful for R-point sweeps where adjacent R-points have similar
/// eigenpairs, enabling faster convergence.
pub fn run_with_warmstart<B: SpectralBackend>(
    backend: B,
    job: &OperatorDataJob,
    warmstart: &[Field2D],
    reference_eigvecs: Option<&[Field2D]>,
) -> OperatorDataDriverResult {
    let n_total = job.n_total_bands();

    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    let diel_derivs = if job.compute_dielectric_derivatives {
        Some([
            DielectricDerivative::from_finite_difference(
                &job.geom,
                job.grid,
                &job.dielectric,
                job.atom_index,
                0,
                job.fd_step,
            ),
            DielectricDerivative::from_finite_difference(
                &job.geom,
                job.grid,
                &job.dielectric,
                job.atom_index,
                1,
                job.fd_step,
            ),
        ])
    } else {
        None
    };

    let mut theta = ThetaOperator::new(backend, dielectric, job.pol, job.k0);

    let mut preconditioner: Box<dyn OperatorPreconditioner<B>> =
        Box::new(theta.build_homogeneous_preconditioner_adaptive());

    let mut eigensolver_config = job.eigensolver.clone();
    if eigensolver_config.n_bands < n_total {
        eigensolver_config.n_bands = n_total;
    }

    let solve_start = std::time::Instant::now();
    let mut solver = Eigensolver::new(
        &mut theta,
        eigensolver_config,
        Some(&mut *preconditioner as &mut dyn OperatorPreconditioner<B>),
        Some(warmstart),
    );
    let result = solver.solve();
    let eigenvectors = solver.all_eigenvectors();
    let solve_time = solve_start.elapsed().as_secs_f64();

    let n_iterations = result.iterations;
    let converged = result.converged;
    let eigenvalues = result.eigenvalues;

    let extract_start = std::time::Instant::now();
    let mut extractor = OperatorDataExtractor::new(
        &mut theta,
        &eigenvectors,
        &eigenvalues,
        job.operator_data_config.clone(),
    );

    let ingredients = extractor.extract(
        job.k0,
        job.registry,
        diel_derivs.as_ref(),
        reference_eigvecs,
        n_iterations,
        converged,
    );
    let extract_time = extract_start.elapsed().as_secs_f64();

    OperatorDataDriverResult {
        ingredients,
        solve_time_seconds: solve_time,
        extract_time_seconds: extract_time,
    }
}

// ============================================================================
// k-Stencil Sweep
// ============================================================================

/// Result of a k-stencil sweep.
#[derive(Debug, Clone)]
pub struct KStencilResult {
    /// EA result at the center k₀.
    pub center: OperatorDataDriverResult,
    /// Center eigenvectors (for downstream warm-starting, e.g., at neighboring R-points).
    pub center_eigenvectors: Vec<Field2D>,
    /// EA results at each stencil neighbor point.
    pub neighbors: Vec<OperatorDataDriverResult>,
    /// Stencil k-points (absolute, not relative). Center is NOT included.
    pub neighbor_k_points: Vec<[f64; 2]>,
}

/// Run EA extraction on a k-stencil centered at k₀.
///
/// The stencil consists of `n_points` per axis (must be odd, ≥ 1), spaced evenly
/// within `[-delta_k, +delta_k]` along each axis. `n_points=1` means center only.
/// `n_points=3` → center + 8 neighbors (3×3 grid minus center), etc.
///
/// **Optimizations:**
/// - The dielectric (and its derivatives) are built once and reused.
/// - The center k₀ is solved first (cold start), then all stencil neighbors
///   are warm-started from the center eigenvectors.
///
/// # Arguments
///
/// * `backend` - Spectral backend (will be cloned for each k-point)
/// * `job` - Base OperatorDataJob. The `k0` field is used as the stencil center.
/// * `n_points` - Number of stencil points per axis (odd, ≥ 1). Total stencil
///   neighbors = n_points² − 1.
/// * `delta_k` - Maximum stencil displacement from center in each direction.
///   Units: same as k₀ (Cartesian reciprocal-space, 2π/a).
pub fn run_k_stencil<B: SpectralBackend + Clone>(
    backend: B,
    job: &OperatorDataJob,
    n_points: usize,
    delta_k: f64,
) -> KStencilResult {
    assert!(n_points >= 1 && n_points % 2 == 1, "n_points must be odd and >= 1");

    #[derive(Clone)]
    struct StencilPointResult {
        ea: OperatorDataDriverResult,
        eigenvectors: Vec<Field2D>,
        omegas: Vec<f64>,
    }

    fn build_adjacent_stencil_walk(half: isize) -> Vec<(isize, isize)> {
        if half <= 0 {
            return Vec::new();
        }

        let mut walk = Vec::with_capacity(((2 * half + 1) * (2 * half + 1) - 1) as usize);
        let mut x = 0isize;
        let mut y = 0isize;
        let mut step_len = 1isize;
        let directions = [(-1isize, 0isize), (0, 1), (1, 0), (0, -1)];

        while walk.len() < ((2 * half + 1) * (2 * half + 1) - 1) as usize {
            for (dir_idx, (dx, dy)) in directions.iter().enumerate() {
                for _ in 0..step_len {
                    x += dx;
                    y += dy;
                    if x.abs() <= half && y.abs() <= half {
                        walk.push((x, y));
                        if walk.len() == ((2 * half + 1) * (2 * half + 1) - 1) as usize {
                            return walk;
                        }
                    }
                }
                if dir_idx % 2 == 1 {
                    step_len += 1;
                }
            }
        }

        walk
    }

    let n_total = job.n_total_bands();

    // -- 1. Build dielectric and derivatives ONCE --
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

    let diel_derivs = if job.compute_dielectric_derivatives {
        Some([
            DielectricDerivative::from_finite_difference(
                &job.geom,
                job.grid,
                &job.dielectric,
                job.atom_index,
                0,
                job.fd_step,
            ),
            DielectricDerivative::from_finite_difference(
                &job.geom,
                job.grid,
                &job.dielectric,
                job.atom_index,
                1,
                job.fd_step,
            ),
        ])
    } else {
        None
    };

    // -- 2. Solve at center k₀ (cold start) --
    let (center_ea, center_eigvecs) = {
        let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, job.k0);
        let mut preconditioner: Box<dyn OperatorPreconditioner<B>> =
            Box::new(theta.build_homogeneous_preconditioner_adaptive());

        let mut eigensolver_config = job.eigensolver.clone();
        if eigensolver_config.n_bands < n_total {
            eigensolver_config.n_bands = n_total;
        }

        let solve_start = std::time::Instant::now();
        let mut solver = Eigensolver::new(
            &mut theta,
            eigensolver_config,
            Some(&mut *preconditioner as &mut dyn OperatorPreconditioner<B>),
            None,
        );
        let result = solver.solve();
        let eigenvectors = solver.all_eigenvectors();
        let solve_time = solve_start.elapsed().as_secs_f64();

        let extract_start = std::time::Instant::now();
        let mut extractor = OperatorDataExtractor::new(
            &mut theta,
            &eigenvectors,
            &result.eigenvalues,
            job.operator_data_config.clone(),
        );
        let ingredients = extractor.extract(
            job.k0,
            job.registry,
            diel_derivs.as_ref(),
            None,
            result.iterations,
            result.converged,
        );
        let extract_time = extract_start.elapsed().as_secs_f64();

        (OperatorDataDriverResult {
            ingredients,
            solve_time_seconds: solve_time,
            extract_time_seconds: extract_time,
        }, eigenvectors)
    };

    // n_points=1 → no neighbors
    if n_points == 1 {
        return KStencilResult {
            center: center_ea,
            center_eigenvectors: center_eigvecs,
            neighbors: Vec::new(),
            neighbor_k_points: Vec::new(),
        };
    }

    // -- 3. Generate stencil k-points --
    let half = (n_points / 2) as isize;
    let step = delta_k / half as f64;

    let mut neighbor_k_points = Vec::new();
    let mut neighbor_offsets = Vec::new();
    for ix in -half..=half {
        for iy in -half..=half {
            if ix == 0 && iy == 0 {
                continue; // skip center
            }
            neighbor_offsets.push((ix, iy));
            neighbor_k_points.push([
                job.k0[0] + ix as f64 * step,
                job.k0[1] + iy as f64 * step,
            ]);
        }
    }

    // -- 4. Solve neighbors by local transport over the stencil graph --
    // Each point is warm-started and tracked against an adjacent parent point,
    // not always against the center. This preserves label continuity much more
    // reliably for near-degenerate manifolds.
    let center_omegas: Vec<f64> = center_ea
        .ingredients
        .eigenvalues
        .iter()
        .map(|&lambda| lambda.max(0.0).sqrt())
        .collect();
    let eps_for_tracking: Option<Vec<f64>> = if job.pol == Polarization::TM {
        Some(dielectric.eps().to_vec())
    } else {
        None
    };

    let adjacent_walk = build_adjacent_stencil_walk(half);
    let solve_order: Vec<usize> = adjacent_walk
        .iter()
        .map(|offset| {
            neighbor_offsets
                .iter()
                .position(|ofs| ofs == offset)
                .expect("adjacent stencil walk offset must exist")
        })
        .collect();

    let mut solved_points: Vec<Option<StencilPointResult>> = vec![None; neighbor_k_points.len()];

    for (order_pos, idx) in solve_order.iter().copied().enumerate() {
        let k_point = neighbor_k_points[idx];
        let parent_state = if order_pos == 0 {
            None::<&StencilPointResult>
        } else {
            let parent_idx = solve_order[order_pos - 1];
            Some(
                solved_points[parent_idx]
                    .as_ref()
                    .expect("previous stencil point should be solved before child"),
            )
        };

        let warmstart = parent_state
            .map(|state| state.eigenvectors.as_slice())
            .unwrap_or(center_eigvecs.as_slice());
        let ref_vecs = parent_state
            .map(|state| state.eigenvectors.as_slice())
            .unwrap_or(center_eigvecs.as_slice());
        let ref_omegas = parent_state
            .map(|state| state.omegas.as_slice())
            .unwrap_or(center_omegas.as_slice());

        let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, k_point);
        let mut preconditioner: Box<dyn OperatorPreconditioner<B>> =
            Box::new(theta.build_homogeneous_preconditioner_adaptive());

        let mut eigensolver_config = job.eigensolver.clone();
        if eigensolver_config.n_bands < n_total {
            eigensolver_config.n_bands = n_total;
        }

        let solve_start = std::time::Instant::now();
        let mut solver = Eigensolver::new(
            &mut theta,
            eigensolver_config,
            Some(&mut *preconditioner as &mut dyn OperatorPreconditioner<B>),
            Some(warmstart),
        );
        let result = solver.solve();
        let mut eigenvalues = result.eigenvalues;
        let mut eigenvectors = solver.all_eigenvectors();
        let mut omegas: Vec<f64> = eigenvalues
            .iter()
            .map(|&lambda| lambda.max(0.0).sqrt())
            .collect();

        let tracking_result = track_bands_with_frequencies(
            ref_vecs,
            &eigenvectors,
            ref_omegas,
            &omegas,
            eps_for_tracking.as_deref(),
        );

        if tracking_result.had_swaps {
            apply_permutation(&tracking_result.permutation, &mut omegas, &mut eigenvectors);

            let eigenvalues_orig = eigenvalues.clone();
            for (idx, &src) in tracking_result.permutation.iter().enumerate() {
                if idx < eigenvalues.len() && src < eigenvalues_orig.len() {
                    eigenvalues[idx] = eigenvalues_orig[src];
                }
            }
        }

        let solve_time = solve_start.elapsed().as_secs_f64();

        let extract_start = std::time::Instant::now();
        let mut extractor = OperatorDataExtractor::new(
            &mut theta,
            &eigenvectors,
            &eigenvalues,
            job.operator_data_config.clone(),
        );
        let ingredients = extractor.extract(
            k_point,
            job.registry,
            diel_derivs.as_ref(),
            Some(ref_vecs),
            result.iterations,
            result.converged,
        );
        let extract_time = extract_start.elapsed().as_secs_f64();

        solved_points[idx] = Some(StencilPointResult {
            ea: OperatorDataDriverResult {
                ingredients,
                solve_time_seconds: solve_time,
                extract_time_seconds: extract_time,
            },
            eigenvectors,
            omegas,
        });
    }

    let neighbors = solved_points
        .into_iter()
        .map(|entry| entry.expect("all stencil points should be solved").ea)
        .collect();

    KStencilResult {
        center: center_ea,
        center_eigenvectors: center_eigvecs,
        neighbors,
        neighbor_k_points,
    }
}

// ============================================================================
// k-Path Band Solver (lean eigenvalue-only sweep)
// ============================================================================

/// Result of a k-path eigenvalue sweep.
#[derive(Debug, Clone)]
pub struct KPathResult {
    /// Eigenvalues at each k-point. Outer index: k-point, inner: band.
    pub eigenvalues: Vec<Vec<f64>>,
    /// The k-points that were solved (echoed back for convenience).
    pub k_points: Vec<[f64; 2]>,
    /// Wall-clock solve time per k-point (seconds).
    pub solve_times: Vec<f64>,
    /// Number of eigensolver iterations per k-point.
    pub iterations: Vec<usize>,
    /// Convergence status per k-point.
    pub converged: Vec<bool>,
}

/// Solve eigenvalues along a k-path efficiently.
///
/// This function is optimised for band-diagram computation:
/// - The dielectric is built **once** and reused for every k-point.
/// - Consecutive k-points are **warm-started** from the previous solution.
/// - **No EA extraction** is performed (no velocity/mass-tensor/Born–Huang).
///
/// # Arguments
///
/// * `backend` – Spectral backend (cloned per k-point).
/// * `geom` – 2-D photonic crystal geometry.
/// * `grid` – Computational grid.
/// * `pol` – Polarization mode.
/// * `k_points` – Ordered list of Bloch wave-vectors (Cartesian, 2π/a).
/// * `n_bands` – Number of eigenvalues to compute at each k-point.
/// * `eigensolver_config` – Convergence / iteration parameters.
/// * `dielectric_opts` – Smoothing options for the dielectric construction.
pub fn run_k_path<B: SpectralBackend + Clone>(
    backend: B,
    geom: &Geometry2D,
    grid: Grid2D,
    pol: Polarization,
    k_points: &[[f64; 2]],
    n_bands: usize,
    eigensolver_config: &EigensolverConfig,
    dielectric_opts: &DielectricOptions,
) -> KPathResult {
    // 1. Build dielectric ONCE
    let dielectric = Dielectric2D::from_geometry(geom, grid, dielectric_opts);

    let n_k = k_points.len();
    let mut all_eigenvalues = Vec::with_capacity(n_k);
    let mut solve_times = Vec::with_capacity(n_k);
    let mut iterations = Vec::with_capacity(n_k);
    let mut converged = Vec::with_capacity(n_k);
    let mut prev_eigenvectors: Option<Vec<Field2D>> = None;

    for &k in k_points {
        let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), pol, k);
        let mut preconditioner: Box<dyn OperatorPreconditioner<B>> =
            Box::new(theta.build_homogeneous_preconditioner_adaptive());

        let mut cfg = eigensolver_config.clone();
        if cfg.n_bands < n_bands {
            cfg.n_bands = n_bands;
        }

        let solve_start = std::time::Instant::now();
        let mut solver = Eigensolver::new(
            &mut theta,
            cfg,
            Some(&mut *preconditioner as &mut dyn OperatorPreconditioner<B>),
            prev_eigenvectors
                .as_deref()
                .map(|v| v as &[Field2D]),
        );
        let result = solver.solve();
        let eigenvectors = solver.all_eigenvectors();
        let elapsed = solve_start.elapsed().as_secs_f64();

        all_eigenvalues.push(result.eigenvalues);
        solve_times.push(elapsed);
        iterations.push(result.iterations);
        converged.push(result.converged);

        // Warm-start next k-point from this one's eigenvectors
        prev_eigenvectors = Some(eigenvectors);
    }

    KPathResult {
        eigenvalues: all_eigenvalues,
        k_points: k_points.to_vec(),
        solve_times,
        iterations,
        converged,
    }
}
