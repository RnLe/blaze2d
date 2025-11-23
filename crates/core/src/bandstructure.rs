//! High-level band-structure job orchestration.

use std::{
    any::type_name,
    cmp::Ordering,
    f64::consts::PI,
    fs::{self, File},
    io::{self, BufWriter, Write},
    mem,
    path::{Path, PathBuf},
    time::Instant,
};

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::{Dielectric2D, DielectricOptions},
    eigensolver::{
        DeflationWorkspace, EigenOptions, GammaContext, PreconditionerKind, solve_lowest_eigenpairs,
    },
    field::Field2D,
    geometry::Geometry2D,
    grid::Grid2D,
    metrics::{MetricsEvent, MetricsRecorder},
    operator::{K_PLUS_G_NEAR_ZERO_FLOOR, ThetaOperator},
    operator_inspection::{dump_iteration_trace, dump_operator_snapshots},
    polarization::Polarization,
    preconditioner::FourierDiagonalPreconditioner,
    symmetry::SymmetryProjector,
};

use num_complex::Complex64;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct BandStructureJob {
    pub geom: Geometry2D,
    pub grid: Grid2D,
    pub pol: Polarization,
    pub k_path: Vec<[f64; 2]>,
    pub eigensolver: EigenOptions,
    pub dielectric: DielectricOptions,
    pub inspection: InspectionOptions,
}

#[derive(Debug, Clone, Default)]
pub struct InspectionOptions {
    pub output_dir: Option<PathBuf>,
    pub dump_eps_real: bool,
    pub dump_eps_fourier: bool,
    pub dump_fft_workspace_raw: bool,
    pub dump_fft_workspace_report: bool,
    pub operator: OperatorInspectionOptions,
}

impl InspectionOptions {
    pub fn is_enabled(&self) -> bool {
        self.output_dir
            .as_ref()
            .filter(|_| {
                self.dump_eps_real
                    || self.dump_eps_fourier
                    || self.dump_fft_workspace_raw
                    || self.dump_fft_workspace_report
                    || self.operator.is_enabled()
            })
            .is_some()
    }
}

#[derive(Debug, Clone)]
pub struct OperatorInspectionOptions {
    pub dump_snapshots: bool,
    pub dump_iteration_traces: bool,
    pub snapshot_k_limit: usize,
    pub snapshot_mode_limit: usize,
}

impl Default for OperatorInspectionOptions {
    fn default() -> Self {
        Self {
            dump_snapshots: false,
            dump_iteration_traces: false,
            snapshot_k_limit: 1,
            snapshot_mode_limit: 2,
        }
    }
}

impl OperatorInspectionOptions {
    pub fn is_enabled(&self) -> bool {
        (self.dump_snapshots && self.snapshot_k_limit > 0 && self.snapshot_mode_limit > 0)
            || self.dump_iteration_traces
    }

    pub fn should_dump_snapshots(&self, k_index: usize) -> bool {
        self.dump_snapshots
            && self.snapshot_k_limit > 0
            && self.snapshot_mode_limit > 0
            && k_index < self.snapshot_k_limit
    }
}

#[derive(Debug, Clone)]
pub struct BandStructureResult {
    pub k_path: Vec<[f64; 2]>,
    pub distances: Vec<f64>,
    pub bands: Vec<Vec<f64>>, // raw ω values
}

fn dump_dielectric_artifacts<B: SpectralBackend>(
    backend: &B,
    dielectric: &Dielectric2D,
    opts: &InspectionOptions,
) -> io::Result<()> {
    if !opts.is_enabled() {
        return Ok(());
    }
    let Some(dir) = opts.output_dir.as_ref() else {
        return Ok(());
    };
    fs::create_dir_all(dir)?;
    if opts.dump_eps_real {
        write_eps_real_csv(
            &dir.join("epsilon_real.csv"),
            dielectric.eps(),
            dielectric.grid,
        )?;
        if let Some(raw) = dielectric.unsmoothed_eps() {
            write_eps_real_csv(&dir.join("epsilon_real_raw.csv"), raw, dielectric.grid)?;
        }
    }
    if opts.dump_eps_fourier {
        write_eps_fourier_csv(backend, &dir.join("epsilon_fourier.csv"), dielectric)?;
    }
    Ok(())
}

fn dump_fft_workspace_raw(
    grid: Grid2D,
    kx_shifted: &[f64],
    ky_shifted: &[f64],
    k_plus_g_sq: &[f64],
    clamp_floor: f64,
    opts: &InspectionOptions,
    k_index: usize,
) -> io::Result<Option<PathBuf>> {
    let Some(dir) = opts.output_dir.as_ref() else {
        return Ok(None);
    };
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("fft_workspace_raw_k{k_index:03}.csv"));
    write_fft_workspace_raw(
        &path,
        grid,
        kx_shifted,
        ky_shifted,
        k_plus_g_sq,
        clamp_floor,
    )?;
    Ok(Some(path))
}

fn dump_fft_workspace_report(
    grid: Grid2D,
    pol: Polarization,
    k_fractional: [f64; 2],
    bloch: [f64; 2],
    kx_shifted: &[f64],
    ky_shifted: &[f64],
    k_plus_g_sq: &[f64],
    mesh_size: usize,
    opts: &InspectionOptions,
    k_index: usize,
) -> io::Result<Option<PathBuf>> {
    let Some(dir) = opts.output_dir.as_ref() else {
        return Ok(None);
    };
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("fft_workspace_report_k{k_index:03}.json"));
    write_fft_workspace_report(
        &path,
        grid,
        pol,
        k_fractional,
        bloch,
        kx_shifted,
        ky_shifted,
        k_plus_g_sq,
        mesh_size,
        k_index,
    )?;
    Ok(Some(path))
}

fn write_eps_real_csv(path: &Path, values: &[f64], grid: Grid2D) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "ix,iy,x,y,epsilon")?;
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let [x, y] = grid.cartesian_coords(ix, iy);
            let eps = values[idx];
            writeln!(writer, "{ix},{iy},{x},{y},{eps}")?;
        }
    }
    writer.flush()
}

fn write_eps_fourier_csv<B: SpectralBackend>(
    backend: &B,
    path: &Path,
    dielectric: &Dielectric2D,
) -> io::Result<()> {
    let grid = dielectric.grid;
    let mut buffer = backend.alloc_field(grid);
    for (value, &eps) in buffer.as_mut_slice().iter_mut().zip(dielectric.eps()) {
        *value = Complex64::new(eps, 0.0);
    }
    backend.forward_fft_2d(&mut buffer);
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "ix,iy,gx,gy,real,imag")?;
    for iy in 0..grid.ny {
        let gy = reciprocal_component(iy, grid.ny, grid.ly);
        for ix in 0..grid.nx {
            let gx = reciprocal_component(ix, grid.nx, grid.lx);
            let idx = grid.idx(ix, iy);
            let value = buffer.as_slice()[idx];
            writeln!(writer, "{ix},{iy},{gx},{gy},{},{})", value.re, value.im)?;
        }
    }
    writer.flush()
}

fn write_fft_workspace_raw(
    path: &Path,
    grid: Grid2D,
    kx_shifted: &[f64],
    ky_shifted: &[f64],
    k_plus_g_sq: &[f64],
    clamp_floor: f64,
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "ix,iy,kx_plus_g,ky_plus_g,k_plus_g_sq_raw,k_plus_g_sq,clamped"
    )?;
    for iy in 0..grid.ny {
        let ky = ky_shifted.get(iy).copied().unwrap_or(0.0);
        for ix in 0..grid.nx {
            let kx = kx_shifted.get(ix).copied().unwrap_or(0.0);
            let idx = grid.idx(ix, iy);
            let clamped = k_plus_g_sq.get(idx).copied().unwrap_or(0.0);
            let raw = kx * kx + ky * ky;
            let clamped_flag = if raw <= clamp_floor { 1 } else { 0 };
            writeln!(writer, "{ix},{iy},{kx},{ky},{raw},{clamped},{clamped_flag}")?;
        }
    }
    writer.flush()
}

fn write_fft_workspace_report(
    path: &Path,
    grid: Grid2D,
    pol: Polarization,
    k_fractional: [f64; 2],
    bloch: [f64; 2],
    kx_shifted: &[f64],
    ky_shifted: &[f64],
    k_plus_g_sq: &[f64],
    mesh_size: usize,
    k_index: usize,
) -> io::Result<()> {
    let (k_sq_min, k_sq_max, k_sq_mean) = summary_stats(k_plus_g_sq).unwrap_or((0.0, 0.0, 0.0));
    let (kx_min, kx_max) = min_max(kx_shifted).unwrap_or((0.0, 0.0));
    let (ky_min, ky_max) = min_max(ky_shifted).unwrap_or((0.0, 0.0));
    let bloch_norm = (bloch[0] * bloch[0] + bloch[1] * bloch[1]).sqrt();
    let buffer_count = 3usize;
    let elements_per_buffer = grid.len();
    let bytes_per_complex = mem::size_of::<Complex64>();
    let approx_bytes = buffer_count * elements_per_buffer * bytes_per_complex;

    let report = json!({
        "k_index": k_index,
        "k_fractional": {"kx": k_fractional[0], "ky": k_fractional[1]},
        "bloch_vector": {"kx": bloch[0], "ky": bloch[1], "norm": bloch_norm},
        "grid": {"nx": grid.nx, "ny": grid.ny, "points": elements_per_buffer},
        "polarization": format!("{:?}", pol),
        "dielectric_mesh_size": mesh_size,
        "kx_shifted": {"min": kx_min, "max": kx_max},
        "ky_shifted": {"min": ky_min, "max": ky_max},
        "k_plus_g_sq": {"min": k_sq_min, "max": k_sq_max, "mean": k_sq_mean},
        "workspace_buffers": {
            "count": buffer_count,
            "elements_each": elements_per_buffer,
            "bytes_total": approx_bytes
        }
    });

    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.flush()
}

fn summary_stats(values: &[f64]) -> Option<(f64, f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mut min = values[0];
    let mut max = values[0];
    let mut sum = 0.0;
    for &value in values {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
        sum += value;
    }
    Some((min, max, sum / values.len() as f64))
}

fn min_max(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mut min = values[0];
    let mut max = values[0];
    for &value in values {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    Some((min, max))
}

fn reciprocal_component(index: usize, len: usize, length: f64) -> f64 {
    if len == 0 || length == 0.0 {
        return 0.0;
    }
    let len_i = len as isize;
    let mut k = index as isize;
    if k > len_i / 2 {
        k -= len_i;
    }
    let two_pi = 2.0 * PI;
    two_pi * k as f64 / length
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
        eprintln!(
            "[setup] dielectric smoothing mesh_size={} ({})",
            job.dielectric.smoothing.mesh_size,
            if job.dielectric.smoothing_enabled() {
                "enabled"
            } else {
                "disabled"
            }
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
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);
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

    if let Err(err) = dump_dielectric_artifacts(&backend, &dielectric, &job.inspection) {
        eprintln!("[inspect] failed to dump dielectric data: {err}");
    } else if job.inspection.is_enabled() && verbosity.enabled() {
        if let Some(path) = &job.inspection.output_dir {
            eprintln!(
                "[inspect] dielectric snapshots written to {}",
                path.display()
            );
        }
    }

    let warm_start_limit = job
        .eigensolver
        .warm_start
        .effective_limit(job.eigensolver.n_bands);
    let mut warm_start_store: Vec<Field2D> = Vec::new();
    let mut bands = Vec::with_capacity(job.k_path.len());
    let mut total_iters = 0usize;
    let mut cumulative_distance = 0.0;
    let mut prev_k: Option<[f64; 2]> = None;
    let mut fft_workspace_raw_dumped = false;
    let mut fft_workspace_report_dumped = false;
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
        if job.inspection.dump_fft_workspace_raw && !fft_workspace_raw_dumped {
            match dump_fft_workspace_raw(
                job.grid,
                theta.kx_shifted(),
                theta.ky_shifted(),
                theta.k_plus_g_squares(),
                K_PLUS_G_NEAR_ZERO_FLOOR,
                &job.inspection,
                idx,
            ) {
                Ok(Some(path)) => {
                    if verbosity.enabled() {
                        eprintln!(
                            "[inspect] fft workspace raw snapshot k#{idx:03} -> {}",
                            path.display()
                        );
                    }
                    fft_workspace_raw_dumped = true;
                }
                Ok(None) => {}
                Err(err) => eprintln!("[inspect] failed to dump fft workspace raw: {err}"),
            }
        }
        if job.inspection.dump_fft_workspace_report && !fft_workspace_report_dumped {
            match dump_fft_workspace_report(
                job.grid,
                job.pol,
                *kp,
                theta.bloch(),
                theta.kx_shifted(),
                theta.ky_shifted(),
                theta.k_plus_g_squares(),
                job.dielectric.smoothing.mesh_size,
                &job.inspection,
                idx,
            ) {
                Ok(Some(path)) => {
                    if verbosity.enabled() {
                        eprintln!(
                            "[inspect] fft workspace report k#{idx:03} -> {}",
                            path.display()
                        );
                    }
                    fft_workspace_report_dumped = true;
                }
                Ok(None) => {}
                Err(err) => eprintln!("[inspect] failed to dump fft workspace report: {err}"),
            }
        }
        let bloch_norm = (bloch[0] * bloch[0] + bloch[1] * bloch[1]).sqrt();
        let gamma_context = GammaContext::new(job.eigensolver.gamma.should_deflate(bloch_norm));
        let workspace_ref: Option<&DeflationWorkspace<B>> = None;
        let deflation_size = 0;
        let warm_slice = if warm_start_limit > 0 && !warm_start_store.is_empty() {
            #[cfg(test)]
            if let Some(probe) = debug.as_mut() {
                probe.warm_start_hits += 1;
            }
            Some(warm_start_store.as_slice())
        } else {
            None
        };
        let warm_seed_count = warm_slice.map_or(0, |slice| slice.len());
        let symmetry_selection = job.eigensolver.symmetry.selection_for_bloch(bloch);
        let symmetry_projector =
            SymmetryProjector::from_reflections(&symmetry_selection.reflections);
        let symmetry_reflections = symmetry_projector
            .as_ref()
            .map(|proj| proj.len())
            .unwrap_or(0);
        let symmetry_skipped = symmetry_selection.skipped_count();
        let mut preconditioner_storage: FourierDiagonalPreconditioner;
        let eig = match job.eigensolver.preconditioner {
            PreconditionerKind::None => solve_lowest_eigenpairs(
                &mut theta,
                &job.eigensolver,
                None,
                gamma_context,
                warm_slice,
                workspace_ref,
                symmetry_projector.as_ref(),
            ),
            PreconditionerKind::HomogeneousJacobi => {
                preconditioner_storage = theta.build_homogeneous_preconditioner();
                solve_lowest_eigenpairs(
                    &mut theta,
                    &job.eigensolver,
                    Some(&mut preconditioner_storage),
                    gamma_context,
                    warm_slice,
                    workspace_ref,
                    symmetry_projector.as_ref(),
                )
            }
            PreconditionerKind::StructuredDiagonal => {
                preconditioner_storage = theta.build_structured_preconditioner();
                solve_lowest_eigenpairs(
                    &mut theta,
                    &job.eigensolver,
                    Some(&mut preconditioner_storage),
                    gamma_context,
                    warm_slice,
                    workspace_ref,
                    symmetry_projector.as_ref(),
                )
            }
        };
        if job.inspection.operator.should_dump_snapshots(idx) {
            match dump_operator_snapshots(
                &mut theta,
                job.pol,
                idx,
                *kp,
                &eig.modes,
                &job.inspection,
            ) {
                Ok(paths) => {
                    if !paths.is_empty() && verbosity.enabled() {
                        eprintln!(
                            "[inspect] operator snapshots k#{idx:03} -> {} files",
                            paths.len()
                        );
                    }
                }
                Err(err) => eprintln!("[inspect] failed to dump operator snapshots: {err}"),
            }
        }
        if job.inspection.operator.dump_iteration_traces {
            if let Err(err) = dump_iteration_trace(
                &job.inspection,
                job.pol,
                idx,
                *kp,
                &eig.diagnostics.iterations,
            ) {
                eprintln!("[inspect] failed to dump iteration trace: {err}");
            }
        }
        total_iters += eig.iterations;
        let k_elapsed = k_timer.elapsed();
        if verbosity.enabled() {
            let freq_summary = format_frequency_summary(&eig.omegas);
            let rel_change = eig
                .diagnostics
                .iterations
                .last()
                .map(|info| info.max_relative_residual)
                .unwrap_or(0.0);
            eprintln!(
                "[solve] k#{:03} k=({:+.3},{:+.3}) iters={} rel={:+10.3e} {} elapsed={:.2?}",
                idx, kp[0], kp[1], eig.iterations, rel_change, freq_summary, k_elapsed
            );
        }
        let mut preconditioner_trials = 0usize;
        let mut preconditioner_new_directions = 0usize;
        let mut preconditioner_before_sum = 0.0;
        let mut preconditioner_after_sum = 0.0;
        for iter_info in &eig.diagnostics.iterations {
            preconditioner_trials += iter_info.preconditioner_trials;
            preconditioner_new_directions += iter_info.new_directions;
            if iter_info.preconditioner_trials > 0 {
                let trials = iter_info.preconditioner_trials as f64;
                preconditioner_before_sum += iter_info.preconditioner_avg_before * trials;
                preconditioner_after_sum += iter_info.preconditioner_avg_after * trials;
            }
        }
        let preconditioner_avg_before = if preconditioner_trials > 0 {
            preconditioner_before_sum / preconditioner_trials as f64
        } else {
            0.0
        };
        let preconditioner_avg_after = if preconditioner_trials > 0 {
            preconditioner_after_sum / preconditioner_trials as f64
        } else {
            0.0
        };

        // Projector and preconditioner diagnostics continue to flow through metrics events.

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
                max_relative_residual: eig.diagnostics.max_relative_residual,
                avg_relative_residual: eig.diagnostics.avg_relative_residual(),
                max_mass_error: eig.diagnostics.max_mass_error(),
                duplicate_modes_skipped: eig.diagnostics.duplicate_modes_skipped,
                negative_modes_skipped: eig.diagnostics.negative_modes_skipped,
                freq_tolerance: eig.diagnostics.freq_tolerance,
                gamma_deflated: eig.gamma_deflated,
                seed_count: warm_seed_count,
                warm_start_hits: eig.warm_start_hits,
                deflation_workspace: deflation_size,
                symmetry_reflections,
                symmetry_reflections_skipped: symmetry_skipped,
                preconditioner_new_directions,
                preconditioner_trials,
                preconditioner_avg_before,
                preconditioner_avg_after,
            });
            for iter_info in &eig.diagnostics.iterations {
                recorder.emit(MetricsEvent::EigenIteration {
                    k_index: idx,
                    iteration: iter_info.iteration,
                    max_residual: iter_info.max_residual,
                    avg_residual: iter_info.avg_residual,
                    max_relative_residual: iter_info.max_relative_residual,
                    avg_relative_residual: iter_info.avg_relative_residual,
                    block_size: iter_info.block_size,
                    new_directions: iter_info.new_directions,
                    preconditioner_trials: iter_info.preconditioner_trials,
                    preconditioner_avg_before: iter_info.preconditioner_avg_before,
                    preconditioner_avg_after: iter_info.preconditioner_avg_after,
                    preconditioner_accepted: iter_info.new_directions,
                });
            }
        }
        if warm_start_limit > 0 {
            let mut order: Vec<usize> = (0..eig.modes.len()).collect();
            order.sort_by(|&a, &b| {
                eig.omegas[a]
                    .partial_cmp(&eig.omegas[b])
                    .unwrap_or(Ordering::Equal)
            });
            warm_start_store.clear();
            for idx in order.into_iter().take(warm_start_limit) {
                warm_start_store.push(eig.modes[idx].clone());
            }
        } else {
            warm_start_store.clear();
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
