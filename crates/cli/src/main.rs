//! MPB-like 2D solver command-line interface.
//!
//! This CLI provides access to the band structure solver from the command line.
//! It reads a TOML configuration file and outputs computed band structures as CSV.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use env_logger::Builder;
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{
    bandstructure::{self, BandStructureResult, RunOptions, Verbosity},
    diagnostics::{PreconditionerShiftMode, PreconditionerType},
    dielectric::Dielectric2D,
    io::{JobConfig, PathPreset},
    symmetry,
};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "mpb2d-lite", about = "MPB-like 2D solver CLI")]
struct Cli {
    /// Path to a TOML configuration file
    #[arg(short, long)]
    config: PathBuf,

    /// Path to CSV output (defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override k-path with a preset (square or hexagonal)
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

    /// Disable dielectric smoothing (shorthand for --mesh-size=1)
    #[arg(long)]
    no_smoothing: bool,

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

    /// Preconditioner type: auto (default), none, fourier-diagonal, kernel-compensated, structured, or transverse-projection
    /// Auto selects kernel-compensated for TE, transverse-projection for TM
    #[arg(long, value_enum, default_value = "auto")]
    preconditioner: PrecondArg,

    /// Disable W (history) directions in LOBPCG (use steepest descent)
    #[arg(long)]
    no_w_history: bool,

    /// Disable locking (deflation) of converged bands
    #[arg(long)]
    no_locking: bool,

    /// Write logs to a file instead of stderr
    ///
    /// Creates a timestamped log file with all log output. Useful for debugging
    /// convergence issues. The log level is controlled by RUST_LOG environment
    /// variable (e.g., RUST_LOG=debug).
    #[arg(long)]
    log_file: Option<PathBuf>,

    /// Use legacy global preconditioner shift (σ = 1e-3)
    ///
    /// By default, the preconditioner uses an adaptive k-dependent shift:
    /// σ(k) = β * s_median(k), where s_median is the median of |k+G|².
    /// This flag reverts to the legacy fixed shift for compatibility.
    #[arg(long)]
    legacy_shift: bool,

    /// Enable transformed TE mode (use similarity transform instead of generalized eigenproblem)
    ///
    /// By default, TE mode uses the generalized eigenproblem A x = λ B x where B = ε(r).
    /// This is mathematically correct and matches MPB's eigenvalue spectrum.
    ///
    /// This flag enables the similarity transform A' = ε^{-1/2} A ε^{-1/2} to convert
    /// to a standard eigenproblem A'y = λy. CAUTION: This transform is broken because
    /// pointwise ε^{-1/2} is NOT the true matrix square root of the plane-wave mass
    /// matrix, causing systematic eigenvalue shifts compared to MPB.
    #[arg(long)]
    transformed_te: bool,

    /// [DEPRECATED] Use --transformed-te instead. This flag is now a no-op since
    /// untransformed TE is the default.
    #[arg(long, hide = true)]
    no_transformed_te: bool,

    /// Enable symmetry projections in LOBPCG
    ///
    /// By default, symmetry projections are disabled for simpler, more predictable
    /// behavior. This flag enables symmetry handling at high-symmetry k-points.
    /// When combined with --multi-sector, the solver runs one LOBPCG per symmetry
    /// sector and merges results for complete, correct bands comparable to MPB.
    #[arg(long)]
    symmetry: bool,

    /// Enable multi-sector symmetry handling
    ///
    /// When symmetry is enabled (--symmetry), this flag enables multi-sector mode
    /// which runs one LOBPCG per symmetry sector at each k-point and merges all
    /// eigenpairs. This gives complete bands (both even and odd modes).
    ///
    /// Without this flag, symmetry uses single-parity projection, which only finds
    /// modes matching the configured parity (default: even). Results may be
    /// incomplete, missing modes from other symmetry sectors.
    #[arg(long)]
    multi_sector: bool,

    /// Traverse the k-path twice for improved convergence
    ///
    /// In round-trip mode, the k-path is traversed twice. The first pass builds
    /// up well-converged warm-start vectors at each k-point. The second pass then
    /// benefits from these improved initial guesses, especially at high-symmetry
    /// points like Γ. Only the second pass results are returned.
    ///
    /// This is useful when convergence is poor at the first k-point (e.g., starting
    /// at Γ), as the warm start from the first pass significantly improves the second.
    #[arg(long)]
    round_trip: bool,

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
}

#[derive(Clone, Debug, ValueEnum)]
enum PathArg {
    Square,
    Hexagonal,
}

impl From<PathArg> for PathPreset {
    fn from(value: PathArg) -> Self {
        match value {
            PathArg::Square => PathPreset::Square,
            PathArg::Hexagonal => PathPreset::Hexagonal,
        }
    }
}

/// CLI-friendly preconditioner type argument
#[derive(Clone, Debug, ValueEnum, Default)]
enum PrecondArg {
    /// Auto-select based on polarization (TE=kernel-compensated, TM=transverse-projection)
    #[default]
    Auto,
    /// No preconditioner
    None,
    /// Fourier diagonal (homogeneous approximation)
    FourierDiagonal,
    /// Structured preconditioner with spatial weights
    Structured,
    /// Kernel-compensated (same as fourier-diagonal with proper null-space handling)
    KernelCompensated,
    /// Transverse projection (MPB's full 6-FFT algorithm)
    TransverseProjection,
}

impl From<PrecondArg> for PreconditionerType {
    fn from(value: PrecondArg) -> Self {
        match value {
            PrecondArg::Auto => PreconditionerType::Auto,
            PrecondArg::None => PreconditionerType::None,
            PrecondArg::FourierDiagonal => PreconditionerType::FourierDiagonal,
            PrecondArg::Structured => PreconditionerType::Structured,
            PrecondArg::KernelCompensated => PreconditionerType::KernelCompensated,
            PrecondArg::TransverseProjection => PreconditionerType::TransverseProjection,
        }
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

            eprintln!("[cli] logging to file: {}", path.display());
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
                        log::Level::Warn  => ("\x1b[1;33m", "\x1b[0m"), // Bold yellow
                        log::Level::Info  => ("\x1b[32m", "\x1b[0m"),   // Green
                        log::Level::Debug => ("\x1b[36m", "\x1b[0m"),   // Cyan
                        log::Level::Trace => ("\x1b[35m", "\x1b[0m"),   // Magenta
                    };

                    writeln!(buf, "{}{:5}{} {}", color_start, level, color_end, record.args())
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
        eprintln!("error: --diagnostics-output is required when using --record-diagnostics");
        std::process::exit(1);
    }

    // Validate iteration CSV options (require diagnostics)
    if (cli.iteration_csv.is_some() || cli.iteration_csv_combined.is_some())
        && !cli.record_diagnostics
    {
        eprintln!(
            "error: --iteration-csv and --iteration-csv-combined require --record-diagnostics"
        );
        std::process::exit(1);
    }

    // Load configuration
    if !cli.quiet {
        eprintln!("[cli] loading config {}", cli.config.display());
    }
    let raw = fs::read_to_string(&cli.config)?;
    let mut config: JobConfig = toml::from_str(&raw)?;

    // Apply k-path override if specified
    if let Some(preset) = cli.path.clone() {
        let path_type = PathPreset::from(preset.clone());
        let samples = symmetry::standard_path(
            &config.geometry.lattice,
            path_type.into(),
            cli.segments_per_leg,
        );
        config.k_path = samples;
        config.path = None;
        if !cli.quiet {
            eprintln!(
                "[cli] overriding k-path via preset {:?} (segments_per_leg={})",
                preset, cli.segments_per_leg
            );
        }
    }

    // Apply dielectric smoothing overrides
    if let Some(mesh_size) = cli.mesh_size {
        let clamped = mesh_size.max(1);
        config.dielectric.smoothing.mesh_size = clamped;
        if !cli.quiet {
            eprintln!(
                "[cli] overriding dielectric mesh_size -> {}{}",
                clamped,
                if clamped == 1 {
                    " (smoothing disabled)"
                } else {
                    ""
                }
            );
        }
    }
    if cli.no_smoothing {
        config.dielectric.smoothing.mesh_size = 1;
        if !cli.quiet {
            eprintln!("[cli] disabling dielectric smoothing (mesh_size=1)");
        }
    }

    // Apply eigensolver toggle overrides
    if cli.no_w_history {
        config.eigensolver.use_w_history = false;
        if !cli.quiet {
            eprintln!("[cli] disabling W history (steepest descent mode)");
        }
    }
    if cli.no_locking {
        config.eigensolver.use_locking = false;
        if !cli.quiet {
            eprintln!("[cli] disabling locking (no band deflation)");
        }
    }

    // Determine shift mode
    let shift_mode = if cli.legacy_shift {
        if !cli.quiet {
            eprintln!("[cli] using legacy preconditioner shift (σ = 1e-3)");
        }
        PreconditionerShiftMode::Legacy
    } else {
        PreconditionerShiftMode::Adaptive
    };

    // Create the job
    let job = bandstructure::BandStructureJob::from(config.clone());

    // Export epsilon data if requested (before solver runs)
    if cli.export_epsilon.is_some() || cli.export_epsilon_tensor.is_some() {
        // Sample dielectric for export
        let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);

        if let Some(ref eps_path) = cli.export_epsilon {
            if !cli.quiet {
                eprintln!("[cli] exporting epsilon data to {}", eps_path.display());
            }
            dielectric.save_csv(eps_path)?;
        }

        if let Some(ref tensor_path) = cli.export_epsilon_tensor {
            if !cli.quiet {
                eprintln!(
                    "[cli] exporting epsilon tensor data to {}",
                    tensor_path.display()
                );
            }
            match dielectric.save_tensor_csv(tensor_path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                    eprintln!("warning: {}", e);
                    eprintln!("         (tensor data requires smoothing to be enabled)");
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    // Skip solver if --skip-solve is set
    if cli.skip_solve {
        if cli.export_epsilon.is_none() && cli.export_epsilon_tensor.is_none() {
            eprintln!("warning: --skip-solve specified but no export options given");
            eprintln!("         use --export-epsilon and/or --export-epsilon-tensor");
        }
        if !cli.quiet {
            eprintln!("[cli] skipping eigensolver (--skip-solve)");
        }
        return Ok(());
    }

    if !cli.quiet {
        if let Some(dest) = &cli.output {
            eprintln!("[cli] writing CSV to {}", dest.display());
        } else {
            eprintln!("[cli] streaming CSV to stdout");
        }
    }

    let verbosity = if cli.quiet {
        Verbosity::Quiet
    } else {
        Verbosity::Verbose
    };

    // Build run options
    let precond_type = PreconditionerType::from(cli.preconditioner.clone());
    let mut run_options = RunOptions::new()
        .with_preconditioner(precond_type)
        .with_shift_mode(shift_mode);

    // Default: untransformed TE (generalized eigenproblem A x = λ B x, B = ε)
    // The transformed version causes systematic eigenvalue shifts because pointwise
    // ε^{-1/2} is NOT the true matrix square root of the plane-wave mass matrix.
    if cli.transformed_te {
        run_options = run_options.with_transformed_te();
        if !cli.quiet {
            eprintln!("[cli] WARNING: using transformed TE mode (BROKEN - eigenvalues will be shifted)");
        }
    } else if !cli.quiet {
        eprintln!("[cli] using untransformed TE mode (generalized eigenproblem, default)");
    }

    if cli.symmetry {
        run_options = run_options.with_symmetry();
        // Symmetry enabled - check multi-sector mode
        if cli.multi_sector {
            run_options = run_options.with_multi_sector();
            if !cli.quiet {
                eprintln!("[cli] symmetry enabled, multi-sector mode (complete bands)");
            }
        } else {
            // Single-parity mode (legacy)
            run_options = run_options.without_multi_sector();
            if !cli.quiet {
                eprintln!("[cli] symmetry enabled, single-parity mode");
                eprintln!("[cli]   NOTE: results may be incomplete (only one parity sector)");
                eprintln!("[cli]   Use --multi-sector for complete bands");
            }
        }
    } else {
        // Default: no symmetry projections
        if !cli.quiet {
            eprintln!("[cli] symmetry projections disabled (default)");
        }
    }

    if cli.round_trip {
        run_options = run_options.with_round_trip();
        if !cli.quiet {
            eprintln!("[cli] round-trip mode enabled (k-path traversed twice)");
        }
    }

    // Run the solver (with or without diagnostics)
    let result = if cli.record_diagnostics {
        // Derive study name from config filename if not specified
        let study_name = cli.study_name.unwrap_or_else(|| {
            cli.config
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "study".to_string())
        });

        if !cli.quiet {
            eprintln!(
                "[cli] recording diagnostics: study={} preconditioner={:?} shift={:?}",
                study_name, precond_type, shift_mode
            );
        }

        let diag_result = bandstructure::run_with_diagnostics_and_options(
            CpuBackend::new(),
            &job,
            verbosity,
            &study_name,
            run_options,
        );

        // Save diagnostics to JSON
        let diag_path = cli.diagnostics_output.as_ref().unwrap();
        if !cli.quiet {
            eprintln!("[cli] writing diagnostics to {}", diag_path.display());
        }

        diag_result.study.save_json(diag_path)?;

        // Save iteration CSV if requested
        if let Some(ref csv_path) = cli.iteration_csv {
            if !cli.quiet {
                eprintln!(
                    "[cli] writing per-k-point iteration CSV to {}",
                    csv_path.display()
                );
            }
            diag_result.study.save_iteration_csv(csv_path)?;
        }

        if let Some(ref csv_path) = cli.iteration_csv_combined {
            if !cli.quiet {
                eprintln!(
                    "[cli] writing combined iteration CSV to {}",
                    csv_path.display()
                );
            }
            diag_result.study.save_iteration_csv_combined(csv_path)?;
        }

        if !cli.quiet {
            eprintln!(
                "[cli] diagnostics saved: {} runs, {} k-points",
                diag_result.study.runs.len(),
                diag_result.result.k_path.len()
            );
        }

        diag_result.result
    } else {
        bandstructure::run_with_options(CpuBackend::new(), &job, verbosity, run_options)
    };

    // Write CSV output
    emit_csv(&result, cli.output.as_deref())?;

    if !cli.quiet {
        if let Some(path) = cli.output {
            eprintln!("wrote {} rows to {}", result.k_path.len(), path.display());
        } else {
            eprintln!("wrote {} rows to stdout", result.k_path.len());
        }
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
