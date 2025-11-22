//! High-level band-structure job orchestration.

use std::{any::type_name, f64::consts::PI, time::Instant};

use crate::{
    backend::SpectralBackend,
    dielectric::Dielectric2D,
    eigensolver::{
        DeflationWorkspace, EigenOptions, GammaContext, PreconditionerKind,
        build_deflation_workspace, solve_lowest_eigenpairs,
    },
    field::Field2D,
    geometry::Geometry2D,
    grid::Grid2D,
    metrics::{MetricsEvent, MetricsRecorder},
    operator::ThetaOperator,
    polarization::Polarization,
};

#[derive(Debug, Clone)]
pub struct BandStructureJob {
    pub geom: Geometry2D,
    pub grid: Grid2D,
    pub pol: Polarization,
    pub k_path: Vec<[f64; 2]>,
    pub eigensolver: EigenOptions,
}

#[derive(Debug, Clone)]
pub struct BandStructureResult {
    pub k_path: Vec<[f64; 2]>,
    pub distances: Vec<f64>,
    pub bands: Vec<Vec<f64>>, // raw ω values
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Quiet,
    Verbose,
}

impl Verbosity {
    fn enabled(self) -> bool {
        matches!(self, Verbosity::Verbose)
    }
}

#[cfg(not(test))]
pub fn run<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
) -> BandStructureResult {
    run_impl(backend, job, verbosity, None)
}

#[cfg(test)]
pub fn run<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
) -> BandStructureResult {
    run_impl(backend, job, verbosity, None, None)
}

#[cfg(not(test))]
pub fn run_with_metrics<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    metrics: Option<&MetricsRecorder>,
) -> BandStructureResult {
    run_impl(backend, job, verbosity, metrics)
}

#[cfg(test)]
pub fn run_with_metrics<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    metrics: Option<&MetricsRecorder>,
) -> BandStructureResult {
    run_impl(backend, job, verbosity, metrics, None)
}

#[cfg(test)]
#[derive(Default, Debug)]
pub(crate) struct RunDebugProbe {
    pub warm_start_hits: usize,
    pub theta_instances: usize,
}

#[cfg(test)]
pub(crate) fn run_with_debug<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    probe: &mut RunDebugProbe,
) -> BandStructureResult {
    run_impl(backend, job, verbosity, None, Some(probe))
}

#[cfg(not(test))]
fn run_impl<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    metrics: Option<&MetricsRecorder>,
) -> BandStructureResult {
    run_impl_inner(backend, job, verbosity, metrics)
}

#[cfg(test)]
fn run_impl<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    metrics: Option<&MetricsRecorder>,
    debug: Option<&mut RunDebugProbe>,
) -> BandStructureResult {
    run_impl_inner(backend, job, verbosity, metrics, debug)
}

fn run_impl_inner<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    verbosity: Verbosity,
    metrics: Option<&MetricsRecorder>,
    #[cfg(test)] mut debug: Option<&mut RunDebugProbe>,
) -> BandStructureResult {
    let pipeline_start = Instant::now();
    if verbosity.enabled() {
        eprintln!(
            "[setup] backend={} grid={}x{} pol={:?} bands={} max_iter={} tol={} k_points={}",
            type_name::<B>(),
            job.grid.nx,
            job.grid.ny,
            job.pol,
            job.eigensolver.n_bands,
            job.eigensolver.max_iter,
            job.eigensolver.tol,
            job.k_path.len()
        );
        eprintln!(
            "[setup] lattice a1={:?} a2={:?} atoms={}",
            job.geom.lattice.a1,
            job.geom.lattice.a2,
            job.geom.atoms.len()
        );
    }
    if let Some(recorder) = metrics {
        recorder.emit(MetricsEvent::PipelineStart {
            backend: type_name::<B>(),
            grid_nx: job.grid.nx,
            grid_ny: job.grid.ny,
            polarization: job.pol,
            n_bands: job.eigensolver.n_bands,
            max_iter: job.eigensolver.max_iter,
            tol: job.eigensolver.tol,
            k_points: job.k_path.len(),
            atoms: job.geom.atoms.len(),
        });
    }

    let dielectric_start = Instant::now();
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid);
    let dielectric_elapsed = dielectric_start.elapsed();
    if verbosity.enabled() {
        eprintln!(
            "[setup] dielectric sampled in {:.2?} (grid_len={})",
            dielectric_elapsed,
            job.grid.len()
        );
    }
    if let Some(recorder) = metrics {
        recorder.emit(MetricsEvent::DielectricSample {
            duration_ms: dielectric_elapsed.as_secs_f64() * 1000.0,
            grid_points: job.grid.len(),
        });
    }

    let deflation_limit = job
        .eigensolver
        .deflation
        .effective_limit(job.eigensolver.n_bands);
    let warm_start_limit = job
        .eigensolver
        .warm_start
        .effective_limit(job.eigensolver.n_bands);
    let mut deflation_store: Vec<Field2D> = Vec::new();
    let mut warm_start_store: Vec<Field2D> = Vec::new();
    let mut bands = Vec::with_capacity(job.k_path.len());
    let mut total_iters = 0usize;
    let mut cumulative_distance = 0.0;
    let mut prev_k: Option<[f64; 2]> = None;
    for (idx, kp) in job.k_path.iter().enumerate() {
        #[cfg(test)]
        if let Some(probe) = debug.as_mut() {
            probe.theta_instances += 1;
        }
        if let Some(prev) = prev_k {
            let dx = kp[0] - prev[0];
            let dy = kp[1] - prev[1];
            cumulative_distance += (dx * dx + dy * dy).sqrt();
        }
        prev_k = Some(*kp);
        let bloch = [2.0 * PI * kp[0], 2.0 * PI * kp[1]];
        let k_timer = Instant::now();
        let workspace_timer = metrics.map(|_| Instant::now());
        let mut theta = ThetaOperator::new(backend.clone(), dielectric.clone(), job.pol, bloch);
        let workspace_elapsed = workspace_timer.map(|timer| timer.elapsed());
        if let (Some(recorder), Some(duration)) = (metrics, workspace_elapsed) {
            recorder.emit(MetricsEvent::FftWorkspace {
                duration_ms: duration.as_secs_f64() * 1000.0,
                grid_nx: job.grid.nx,
                grid_ny: job.grid.ny,
            });
        }
        if idx == 0 && verbosity.enabled() {
            eprintln!(
                "[fft] spectral workspace prepared (grid={}x{}, points={})",
                job.grid.nx,
                job.grid.ny,
                job.grid.len()
            );
        }
        let bloch_norm = (bloch[0] * bloch[0] + bloch[1] * bloch[1]).sqrt();
        let gamma_context = GammaContext::new(job.eigensolver.gamma.should_deflate(bloch_norm));
        let deflation_workspace: Option<DeflationWorkspace<B>> =
            if deflation_limit > 0 && !deflation_store.is_empty() {
                let mut recent: Vec<&Field2D> =
                    deflation_store.iter().rev().take(deflation_limit).collect();
                recent.reverse();
                let workspace = build_deflation_workspace(&mut theta, recent);
                if workspace.is_empty() {
                    None
                } else {
                    Some(workspace)
                }
            } else {
                None
            };
        let workspace_ref = deflation_workspace.as_ref();
        let warm_slice = if warm_start_limit > 0 && !warm_start_store.is_empty() {
            #[cfg(test)]
            if let Some(probe) = debug.as_mut() {
                probe.warm_start_hits += 1;
            }
            Some(warm_start_store.as_slice())
        } else {
            None
        };
        let eig = match job.eigensolver.preconditioner {
            PreconditionerKind::None => solve_lowest_eigenpairs(
                &mut theta,
                &job.eigensolver,
                None,
                gamma_context,
                warm_slice,
                workspace_ref,
            ),
            PreconditionerKind::FourierDiagonal => {
                let mut preconditioner = theta.build_fourier_diagonal_preconditioner();
                solve_lowest_eigenpairs(
                    &mut theta,
                    &job.eigensolver,
                    Some(&mut preconditioner),
                    gamma_context,
                    warm_slice,
                    workspace_ref,
                )
            }
        };
        total_iters += eig.iterations;
        let k_elapsed = k_timer.elapsed();
        if verbosity.enabled() {
            let freq_summary = format_frequency_summary(&eig.omegas);
            eprintln!(
                "[solve] k#{:03} k=({:+.3},{:+.3}) pol={:?} iters={} bands={} {} elapsed={:.2?}",
                idx,
                kp[0],
                kp[1],
                job.pol,
                eig.iterations,
                eig.omegas.len(),
                freq_summary,
                k_elapsed
            );
        }
        if let Some(recorder) = metrics {
            recorder.emit(MetricsEvent::KPointSolve {
                k_index: idx,
                kx: kp[0],
                ky: kp[1],
                distance: cumulative_distance,
                polarization: job.pol,
                iterations: eig.iterations,
                bands: eig.omegas.len(),
                duration_ms: k_elapsed.as_secs_f64() * 1000.0,
                max_residual: eig.diagnostics.max_residual,
                avg_residual: eig.diagnostics.avg_residual(),
                max_mass_error: eig.diagnostics.max_mass_error(),
                duplicate_modes_skipped: eig.diagnostics.duplicate_modes_skipped,
                negative_modes_skipped: eig.diagnostics.negative_modes_skipped,
                freq_tolerance: eig.diagnostics.freq_tolerance,
            });
            for iter_info in &eig.diagnostics.iterations {
                recorder.emit(MetricsEvent::EigenIteration {
                    k_index: idx,
                    iteration: iter_info.iteration,
                    max_residual: iter_info.max_residual,
                    avg_residual: iter_info.avg_residual,
                    block_size: iter_info.block_size,
                    new_directions: iter_info.new_directions,
                });
            }
        }
        if deflation_limit > 0 {
            for mode in &eig.modes {
                deflation_store.push(mode.clone());
            }
            if deflation_store.len() > deflation_limit {
                let excess = deflation_store.len() - deflation_limit;
                deflation_store.drain(0..excess);
            }
        }
        if warm_start_limit > 0 {
            warm_start_store.clear();
            warm_start_store.extend(eig.modes.iter().take(warm_start_limit).cloned());
        }
        bands.push(eig.omegas);
    }
    if verbosity.enabled() {
        eprintln!(
            "[done] solved {} k-points in {:.2?} (total_iters={})",
            job.k_path.len(),
            pipeline_start.elapsed(),
            total_iters
        );
    }
    if let Some(recorder) = metrics {
        recorder.emit(MetricsEvent::PipelineDone {
            total_k: job.k_path.len(),
            total_iterations: total_iters,
            duration_ms: pipeline_start.elapsed().as_secs_f64() * 1000.0,
        });
    }
    BandStructureResult {
        k_path: job.k_path.clone(),
        distances: accumulate_distances(&job.k_path),
        bands,
    }
}

const FREQUENCY_PLACEHOLDER: &str = "   --.--";

fn format_frequency_summary(values: &[f64]) -> String {
    let mut summary = String::from("frequencies=[");
    match values.len() {
        0 => {
            summary.push_str(FREQUENCY_PLACEHOLDER);
            summary.push_str("..");
            summary.push_str(FREQUENCY_PLACEHOLDER);
        }
        _ => {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &value in values {
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }
            let low = format_frequency_value(min_val);
            let high = format_frequency_value(max_val);
            summary.push_str(&low);
            summary.push_str("..");
            summary.push_str(&high);
        }
    }
    summary.push(']');
    summary
}

fn format_frequency_value(value: f64) -> String {
    let abs = value.abs();
    if abs >= 1e-3 && abs <= 1e2 {
        format!("{value:8.3}")
    } else {
        format!("{value:8.2e}")
    }
}

fn accumulate_distances(k_path: &[[f64; 2]]) -> Vec<f64> {
    if k_path.is_empty() {
        return Vec::new();
    }
    let mut distances = Vec::with_capacity(k_path.len());
    let mut total = 0.0;
    distances.push(0.0);
    for pair in k_path.windows(2) {
        let dx = pair[1][0] - pair[0][0];
        let dy = pair[1][1] - pair[0][1];
        total += (dx * dx + dy * dy).sqrt();
        distances.push(total);
    }
    distances
}
