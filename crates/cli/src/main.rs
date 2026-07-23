//! MPB-like 2D solver command-line interface.
//!
//! This CLI provides access to the band structure solver from the command line.
//! It reads a TOML configuration file and outputs computed band structures as CSV.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use blaze2d_bulk_driver_core::{
    Config, ExpandedJobType, PathPresetSpec, PathSection, Precision, SmoothingKind, expand_jobs,
};
use blaze2d_core::{
    brillouin::BrillouinPath,
    diagnostics::{ConvergenceStudy, PreconditionerType},
    dielectric::Dielectric2D,
    drivers::bandstructure::{self, BandStructureJob, BandStructureResult, RunOptions},
};
use clap::{Parser, ValueEnum};
use env_logger::Builder;
use log::{error, info, warn};

use blaze2d_backend_cpu::CpuBackend;

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "blaze",
    about = "Blaze 2D photonic crystal band structure solver CLI"
)]
struct Cli {
    /// Path to a TOML configuration file
    #[arg(short, long)]
    config: PathBuf,

    /// Path to CSV output (defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override k-path with a preset (square, rectangular, triangular, or hexagonal)
    #[arg(long, value_enum)]
    path: Option<PathArg>,

    /// Number of interpolation segments per high-symmetry leg (used with --path)
    #[arg(long, default_value_t = 12)]
    segments_per_leg: usize,

    /// Suppress progress logs (stderr)
    #[arg(long)]
    quiet: bool,

    /// Override dielectric smoothing mesh size (1 disables smoothing)
    #[arg(long)]
    mesh_size: Option<usize>,

    // ========================================================================
    // Diagnostics Options
    // ========================================================================
    /// Record per-iteration convergence diagnostics and save to JSON
    #[arg(long)]
    record_diagnostics: bool,

    /// Path to JSON diagnostics output (required if --record-diagnostics is set)
    #[arg(long)]
    diagnostics_output: Option<PathBuf>,

    /// Study name/label for diagnostics (default: derived from config filename)
    #[arg(long)]
    study_name: Option<String>,

    /// Preconditioner type: auto (default), none, fourier-diagonal-kernel-compensated, or transverse-projection
    /// Auto selects fourier-diagonal-kernel-compensated for TE, transverse-projection for TM
    #[arg(long, value_enum, default_value = "auto")]
    preconditioner: PrecondArg,

    /// Storage precision override: f64 or f32.
    ///
    /// `f32` uses single-precision Complex<f32> field storage and FFTs with
    /// f64 accumulation in dot products and Rayleigh–Ritz (the standard HPC
    /// mixed-precision pattern); eigenvalues remain f64 either way. When not
    /// given, the `[solver].precision` TOML key decides.
    #[arg(long, value_enum)]
    precision: Option<PrecisionArg>,

    /// Write logs to a file instead of stderr
    ///
    /// Creates a timestamped log file with all log output. Useful for debugging
    /// convergence issues. The log level is controlled by RUST_LOG environment
    /// variable (e.g., RUST_LOG=debug).
    #[arg(long)]
    log_file: Option<PathBuf>,

    // ========================================================================
    // Export Options
    // ========================================================================
    /// Export per-iteration convergence data to CSV (requires --record-diagnostics)
    ///
    /// Creates CSV files with iteration-by-iteration eigenvalues, frequencies,
    /// and residuals for each k-point. Useful for plotting convergence curves.
    /// Output files: {base}_k000.csv, {base}_k001.csv, etc.
    #[arg(long)]
    iteration_csv: Option<PathBuf>,

    /// Export combined iteration data to a single CSV file (requires --record-diagnostics)
    ///
    /// Like --iteration-csv but all k-points in one file with a k_index column.
    #[arg(long)]
    iteration_csv_combined: Option<PathBuf>,

    /// Export epsilon(r) data to CSV
    ///
    /// Saves the dielectric function ε(r) at each grid point. If smoothing is
    /// enabled, both raw and smoothed values are exported.
    /// Format: ix,iy,frac_x,frac_y,eps_smoothed,inv_eps_smoothed[,eps_raw,inv_eps_raw]
    #[arg(long)]
    export_epsilon: Option<PathBuf>,

    /// Export inverse epsilon tensor data to CSV (only if smoothing is enabled)
    ///
    /// Saves the 2x2 inverse permittivity tensor at each grid point.
    /// Format: ix,iy,frac_x,frac_y,inv_eps_xx,inv_eps_xy,inv_eps_yx,inv_eps_yy
    #[arg(long)]
    export_epsilon_tensor: Option<PathBuf>,

    /// Skip the eigensolver and only export epsilon data
    ///
    /// Use with --export-epsilon and/or --export-epsilon-tensor to export
    /// dielectric data without computing band structures.
    #[arg(long)]
    skip_solve: bool,

    /// Enable subspace prediction for accelerated warm-start
    ///
    /// Uses rotation-based subspace tracking to provide better initial guesses
    /// for the eigensolver at each k-point. This can reduce the number of
    /// iterations needed for convergence, especially when bands reorder.
    /// Enabled by default.
    #[arg(long)]
    subspace_prediction: bool,

    /// Disable subspace prediction (use simple warm-start copy)
    #[arg(long)]
    no_subspace_prediction: bool,

    /// Disable linear extrapolation in subspace prediction
    ///
    /// When set, uses rotation-only prediction (Stage 1) instead of
    /// extrapolation (Stage 2). Useful for debugging or when extrapolation
    /// causes instability.
    #[arg(long)]
    no_extrapolation: bool,

    /// Enable band-window-based preconditioner shift (experimental)
    ///
    /// Uses eigenvalues from the previous k-point to tune the preconditioner
    /// shift to the spectral range of the bands being computed. This can
    /// improve convergence, especially near the Γ-point.
    #[arg(long)]
    band_window_shift: bool,

    /// Blend factor for band-window shift (0.0 = pure band-window, 1.0 = pure adaptive)
    ///
    /// Controls how much weight is given to the adaptive shift vs the
    /// band-window shift. Default is 0.5 (equal blend).
    #[arg(long, default_value = "0.5")]
    band_window_blend: f64,

    /// Scale factor for band-window eigenvalue contribution
    ///
    /// The band-window shift component is c × λ_median, where c is this scale.
    /// Lower values are more conservative. Default is 0.5.
    #[arg(long, default_value = "0.5")]
    band_window_scale: f64,

    /// Skip calculating the final Γ-point (copy from initial Γ instead)
    ///
    /// When the k-path loops back to Γ (e.g., Γ→X→M→Γ), by default the final
    /// Γ-point is fully calculated. Use this flag to copy the initial Γ-point
    /// result instead, which is faster but may miss eigenvector differences.
    #[arg(long)]
    skip_final_gamma: bool,

    /// Disable band tracking between k-points
    ///
    /// Skips the polar decomposition + Hungarian algorithm band tracking step.
    /// Bands will be output in the order the eigensolver produces them (typically
    /// sorted by eigenvalue). Use this for non-sequential k-paths (e.g., 2D grids
    /// around a k-point) where the tracking algorithm may incorrectly swap bands
    /// due to large jumps or direction changes in k-space.
    #[arg(long)]
    no_band_tracking: bool,

    /// Enable profiling and write JSON report to the specified path
    ///
    /// Records timing information for key operations (FFTs, eigensolver,
    /// operator applications) and outputs a JSON report at the end.
    /// Requires the `profiling` feature to be enabled at compile time.
    #[arg(long)]
    profile: Option<PathBuf>,
}

/// CLI path argument supporting all four lattice types.
#[derive(Clone, Debug, ValueEnum)]
enum PathArg {
    /// Square lattice: Γ → X → M → Γ
    Square,
    /// Rectangular lattice: Γ → X → S → Y → Γ
    Rectangular,
    /// Triangular lattice: Γ → M → K → Γ
    Triangular,
    /// Hexagonal lattice (alias for triangular): Γ → M → K → Γ
    Hexagonal,
}

impl From<PathArg> for PathPresetSpec {
    fn from(value: PathArg) -> Self {
        match value {
            PathArg::Square => PathPresetSpec::Square,
            PathArg::Rectangular => PathPresetSpec::Rectangular,
            PathArg::Triangular => PathPresetSpec::Triangular,
            PathArg::Hexagonal => PathPresetSpec::Hexagonal,
        }
    }
}

impl From<PathArg> for BrillouinPath {
    fn from(value: PathArg) -> Self {
        match value {
            PathArg::Square => BrillouinPath::Square,
            PathArg::Rectangular => BrillouinPath::Rectangular,
            PathArg::Triangular => BrillouinPath::Triangular,
            PathArg::Hexagonal => BrillouinPath::Hexagonal,
        }
    }
}

/// CLI-friendly preconditioner type argument
#[derive(Clone, Debug, ValueEnum, Default)]
enum PrecondArg {
    /// Auto-select based on polarization (TM=fourier-diagonal-kernel-compensated, TE=transverse-projection)
    #[default]
    Auto,
    /// No preconditioner
    None,
    /// Fourier-diagonal kernel-compensated (with proper null-space handling at Γ)
    FourierDiagonalKernelCompensated,
    /// Transverse projection (MPB's full 6-FFT algorithm)
    TransverseProjection,
}

impl From<PrecondArg> for PreconditionerType {
    fn from(value: PrecondArg) -> Self {
        match value {
            PrecondArg::Auto => PreconditionerType::Auto,
            PrecondArg::None => PreconditionerType::None,
            PrecondArg::FourierDiagonalKernelCompensated => {
                PreconditionerType::FourierDiagonalKernelCompensated
            }
            PrecondArg::TransverseProjection => PreconditionerType::TransverseProjection,
        }
    }
}

/// CLI-friendly storage-precision argument.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum PrecisionArg {
    /// Single precision: Complex<f32> storage (f64 accumulation).
    F32,
    /// Double precision: Complex<f64> storage (default).
    #[default]
    F64,
}

/// Run the band-structure solver on a backend of the chosen precision,
/// returning the (always-f64) result plus an optional convergence study.
///
/// Generic over the backend so the same code serves `CpuBackend<f32>` and
/// `CpuBackend<f64>`; the precision dispatch happens at the call site.
fn run_solver<B: blaze2d_core::backend::SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    run_options: RunOptions,
    record_diagnostics: bool,
    study_name: &str,
) -> (BandStructureResult, Option<ConvergenceStudy>) {
    if record_diagnostics {
        let diag = bandstructure::run_with_diagnostics_and_options(
            backend,
            job,
            study_name.to_string(),
            run_options,
        );
        (diag.result, Some(diag.study))
    } else {
        (bandstructure::run_with_options(backend, job, run_options), None)
    }
}

// ============================================================================
// Logging Setup
// ============================================================================

/// Initialize logging to stderr or to a file.
///
/// If `log_file` is Some, writes logs to that file with timestamps.
/// Otherwise, uses standard env_logger to stderr.
/// Log level is controlled by RUST_LOG environment variable.
fn initialize_logging(log_file: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    match log_file {
        Some(path) => {
            // Create or truncate the log file
            let file = File::create(path)?;
            let file = std::sync::Mutex::new(file);

            // Build logger that writes to file with timestamps
            // Default to Debug level when logging to file for diagnostic purposes
            Builder::new()
                .format(move |buf, record| {
                    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

                    // Write to the log file
                    {
                        let mut file = file.lock().unwrap();
                        let _ = writeln!(
                            file,
                            "[{} {:5} {}] {}",
                            timestamp,
                            record.level(),
                            record.target(),
                            record.args()
                        );
                    }

                    // Also write to the buffer (for stderr output)
                    writeln!(
                        buf,
                        "[{} {:5} {}] {}",
                        timestamp,
                        record.level(),
                        record.target(),
                        record.args()
                    )
                })
                .filter_level(log::LevelFilter::Debug) // Enable debug level for file logging
                .parse_default_env() // Allow RUST_LOG to override
                .init();

            info!("logging to file: {}", path.display());
        }
        None => {
            // Standard stderr logging with compact format (no timestamp, no module path)
            // with colored log levels using ANSI escape codes
            Builder::from_default_env()
                .format(|buf, record| {
                    use std::io::Write;

                    let level = record.level();

                    // ANSI color codes for log levels
                    let (color_start, color_end) = match level {
                        log::Level::Error => ("\x1b[1;31m", "\x1b[0m"), // Bold red
                        log::Level::Warn => ("\x1b[1;33m", "\x1b[0m"),  // Bold yellow
                        log::Level::Info => ("\x1b[32m", "\x1b[0m"),    // Green
                        log::Level::Debug => ("\x1b[36m", "\x1b[0m"),   // Cyan
                        log::Level::Trace => ("\x1b[35m", "\x1b[0m"),   // Magenta
                    };

                    writeln!(
                        buf,
                        "{}{:5}{} {}",
                        color_start,
                        level,
                        color_end,
                        record.args()
                    )
                })
                .init();
        }
    }

    Ok(())
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize logging - either to file or stderr
    initialize_logging(cli.log_file.as_deref())?;

    // Validate diagnostics options
    if cli.record_diagnostics && cli.diagnostics_output.is_none() {
        error!("--diagnostics-output is required when using --record-diagnostics");
        std::process::exit(1);
    }

    // Validate iteration CSV options (require diagnostics)
    if (cli.iteration_csv.is_some() || cli.iteration_csv_combined.is_some())
        && !cli.record_diagnostics
    {
        error!("--iteration-csv and --iteration-csv-combined require --record-diagnostics");
        std::process::exit(1);
    }

    // Load configuration
    if !cli.quiet {
        info!("loading config {}", cli.config.display());
    }
    let raw = fs::read_to_string(&cli.config)?;
    let mut config = Config::from_str(&raw).map_err(|e| {
        error!("{}", e);
        e
    })?;

    // Apply k-path override if specified
    if let Some(preset) = cli.path.clone() {
        let brillouin_path = BrillouinPath::from(preset.clone());
        config.path = Some(PathSection {
            preset: Some(PathPresetSpec::from(preset.clone())),
            points_per_segment: Some(cli.segments_per_leg),
            points: vec![],
        });
        if !cli.quiet {
            info!(
                "overriding k-path via preset {:?} ({}) with {} points/segment",
                preset,
                brillouin_path.name(),
                cli.segments_per_leg
            );
        }
    }

    // Apply dielectric smoothing overrides
    if let Some(mesh_size) = cli.mesh_size {
        let clamped = mesh_size.max(1);
        if clamped == 1 {
            config.dielectric.smoothing = SmoothingKind::Disabled;
        } else {
            config.dielectric.mesh_size = clamped;
            if config.dielectric.smoothing == SmoothingKind::Disabled {
                config.dielectric.smoothing = SmoothingKind::Analytic;
            }
        }
        if !cli.quiet {
            info!(
                "overriding dielectric mesh_size -> {}{}",
                clamped,
                if clamped == 1 {
                    " (smoothing disabled)"
                } else {
                    ""
                }
            );
        }
    }

    // Expand the (single) job. Sweeps and operator-data extraction run via
    // the bulk driver, not this CLI.
    let jobs = expand_jobs(&config);
    if jobs.len() > 1 {
        error!(
            "this config expands to {} jobs; parameter sweeps run via the blaze2d-bulk driver",
            jobs.len()
        );
        std::process::exit(1);
    }
    let job: BandStructureJob = match jobs.into_iter().next().map(|j| j.job_type) {
        Some(ExpandedJobType::Maxwell(job)) => job,
        _ => {
            error!("solver.type = \"operator_data\" runs via the blaze2d-bulk driver");
            std::process::exit(1);
        }
    };

    // Export epsilon data if requested (before solver runs)
    if cli.export_epsilon.is_some() || cli.export_epsilon_tensor.is_some() {
        // Sample dielectric for export
        let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

        if let Some(ref eps_path) = cli.export_epsilon {
            if !cli.quiet {
                info!("exporting epsilon data to {}", eps_path.display());
            }
            dielectric.save_csv(eps_path)?;
        }

        if let Some(ref tensor_path) = cli.export_epsilon_tensor {
            if !cli.quiet {
                info!("exporting epsilon tensor data to {}", tensor_path.display());
            }
            match dielectric.save_tensor_csv(tensor_path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                    warn!("{}", e);
                    warn!("tensor data requires smoothing to be enabled");
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    // Skip solver if --skip-solve is set
    if cli.skip_solve {
        if cli.export_epsilon.is_none() && cli.export_epsilon_tensor.is_none() {
            warn!("--skip-solve specified but no export options given");
            warn!("use --export-epsilon and/or --export-epsilon-tensor");
        }
        if !cli.quiet {
            info!("skipping eigensolver (--skip-solve)");
        }
        return Ok(());
    }

    if !cli.quiet {
        if let Some(dest) = &cli.output {
            info!("writing CSV to {}", dest.display());
        } else {
            info!("streaming CSV to stdout");
        }
    }

    let verbosity = cli.quiet;
    let _ = verbosity; // verbosity is now controlled via RUST_LOG; kept here only to satisfy the `cli.quiet` reads below.

    // Build run options
    // Subspace prediction is ON by default unless --no-subspace-prediction is set
    // --subspace-prediction flag is kept for explicit enabling (overrides --no-subspace-prediction)
    let use_subspace_pred = if cli.no_subspace_prediction && !cli.subspace_prediction {
        false
    } else {
        true // default on
    };
    let use_extrapolation = !cli.no_extrapolation;

    let precond_type = PreconditionerType::from(cli.preconditioner.clone());
    let run_options = RunOptions::new()
        .with_preconditioner(precond_type)
        .with_subspace_prediction(use_subspace_pred)
        .with_extrapolation(use_extrapolation)
        .with_band_window_shift(cli.band_window_shift)
        .with_band_window_blend(cli.band_window_blend)
        .with_band_window_scale(cli.band_window_scale)
        .with_disable_band_tracking(cli.no_band_tracking || config.run.disable_band_tracking);

    if !cli.quiet && use_subspace_pred {
        if use_extrapolation {
            info!("subspace prediction enabled (rotation + extrapolation)");
        } else {
            info!("subspace prediction enabled (rotation-only)");
        }
    }

    if !cli.quiet && cli.no_band_tracking {
        info!("band tracking disabled (bands output in eigensolver order)");
    }

    // Derive study name (used only when recording diagnostics).
    let study_name = if cli.record_diagnostics {
        cli.study_name.clone().unwrap_or_else(|| {
            cli.config
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "study".to_string())
        })
    } else {
        String::new()
    };

    // CLI --precision overrides [solver].precision from the TOML.
    let precision = match cli.precision {
        Some(PrecisionArg::F32) => Precision::F32,
        Some(PrecisionArg::F64) => Precision::F64,
        None => config.precision(),
    };

    if !cli.quiet {
        info!("using CPU backend (precision={})", precision);
        if cli.record_diagnostics {
            info!(
                "recording diagnostics: study={} preconditioner={:?}",
                study_name, precond_type
            );
        }
    }

    // Run the solver on the precision-correct backend monomorphisation.
    // The result is always f64 regardless of storage precision.
    let (result, study) = match precision {
        Precision::F32 => run_solver(
            CpuBackend::<f32>::new(),
            &job,
            run_options,
            cli.record_diagnostics,
            &study_name,
        ),
        Precision::F64 => run_solver(
            CpuBackend::<f64>::new(),
            &job,
            run_options,
            cli.record_diagnostics,
            &study_name,
        ),
    };

    // Save diagnostics artifacts (JSON + optional iteration CSVs), if recorded.
    if let Some(study) = study {
        let diag_path = cli.diagnostics_output.as_ref().unwrap();
        if !cli.quiet {
            info!("writing diagnostics to {}", diag_path.display());
        }
        study.save_json(diag_path)?;

        if let Some(ref csv_path) = cli.iteration_csv {
            if !cli.quiet {
                info!(
                    "writing per-k-point iteration CSV to {}",
                    csv_path.display()
                );
            }
            study.save_iteration_csv(csv_path)?;
        }

        if let Some(ref csv_path) = cli.iteration_csv_combined {
            if !cli.quiet {
                info!("writing combined iteration CSV to {}", csv_path.display());
            }
            study.save_iteration_csv_combined(csv_path)?;
        }

        if !cli.quiet {
            info!(
                "diagnostics saved: {} runs, {} k-points",
                study.runs.len(),
                result.k_path.len()
            );
        }
    }

    // Write CSV output
    emit_csv(&result, cli.output.as_deref())?;

    if !cli.quiet {
        if let Some(path) = cli.output {
            info!("wrote {} rows to {}", result.k_path.len(), path.display());
        } else {
            info!("wrote {} rows to stdout", result.k_path.len());
        }
    }

    // Write profiling report if requested
    #[cfg(feature = "profiling")]
    if let Some(ref profile_path) = cli.profile {
        use blaze2d_core::profiler::get_profile_json;
        let json = get_profile_json();
        fs::write(profile_path, &json)?;
        if !cli.quiet {
            info!("wrote profiling report to {}", profile_path.display());
        }
    }

    #[cfg(not(feature = "profiling"))]
    if cli.profile.is_some() {
        warn!("--profile flag requires the `profiling` feature to be enabled");
        warn!("recompile with: cargo build --release --features profiling");
    }

    Ok(())
}

// ============================================================================
// Output Functions
// ============================================================================

/// Write the band structure result to CSV format.
///
/// The output format is:
/// - Column 1: k-point index (0-indexed)
/// - Column 2: kx (fractional coordinate)
/// - Column 3: ky (fractional coordinate)
/// - Column 4: k_distance (cumulative path distance)
/// - Remaining columns: band1, band2, ... (normalized frequencies ω/2π)
fn emit_csv(result: &BandStructureResult, dest: Option<&Path>) -> io::Result<()> {
    let mut writer: Box<dyn Write> = match dest {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout())),
    };

    // Write header
    let max_bands = result.bands.iter().map(|row| row.len()).max().unwrap_or(0);
    write!(writer, "k_index,kx,ky,k_distance")?;
    for band_idx in 0..max_bands {
        write!(writer, ",band{}", band_idx + 1)?;
    }
    writeln!(writer)?;

    // Write data rows
    for (idx, ((kx, ky), bands)) in result
        .k_path
        .iter()
        .map(|k| (k[0], k[1]))
        .zip(result.bands.iter())
        .enumerate()
    {
        let distance = result.distances.get(idx).copied().unwrap_or_default();
        write!(writer, "{idx},{kx},{ky},{distance}")?;

        for omega in bands {
            // Normalize: output ω/(2π) so frequencies are in units of c/a
            let normalized = omega / (2.0 * std::f64::consts::PI);
            write!(writer, ",{}", normalized)?;
        }

        // Pad with empty columns if this k-point has fewer bands
        if bands.len() < max_bands {
            for _ in bands.len()..max_bands {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }

    writer.flush()
}
