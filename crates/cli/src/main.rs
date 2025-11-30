//! MPB-like 2D solver command-line interface.
//!
//! This CLI provides access to the band structure solver from the command line.
//! It reads a TOML configuration file and outputs computed band structures as CSV.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use env_logger::Builder;
use log::{error, info, warn};
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{
    bandstructure::{self, BandStructureResult, RunOptions, Verbosity},
    diagnostics::PreconditionerType,
    dielectric::Dielectric2D,
    io::{JobConfig, PathPreset},
    symmetry::standard_path,
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
            PrecondArg::FourierDiagonalKernelCompensated => PreconditionerType::FourierDiagonalKernelCompensated,
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
    let mut config: JobConfig = toml::from_str(&raw)?;

    // Apply k-path override if specified
    if let Some(preset) = cli.path.clone() {
        let path_type = PathPreset::from(preset.clone());
        let samples = standard_path(
            &config.geometry.lattice,
            path_type.into(),
            cli.segments_per_leg,
        );
        config.k_path = samples;
        config.path = None;
        if !cli.quiet {
            info!(
                "overriding k-path via preset {:?} (segments_per_leg={})",
                preset, cli.segments_per_leg
            );
        }
    }

    // Apply dielectric smoothing overrides
    if let Some(mesh_size) = cli.mesh_size {
        let clamped = mesh_size.max(1);
        config.dielectric.smoothing.mesh_size = clamped;
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

    // Create the job
    let job = bandstructure::BandStructureJob::from(config.clone());

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
                info!(
                    "exporting epsilon tensor data to {}",
                    tensor_path.display()
                );
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

    let verbosity = if cli.quiet {
        Verbosity::Quiet
    } else {
        Verbosity::Verbose
    };

    // Build run options
    let precond_type = PreconditionerType::from(cli.preconditioner.clone());
    let run_options = RunOptions::new()
        .with_preconditioner(precond_type);

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
            info!(
                "recording diagnostics: study={} preconditioner={:?}",
                study_name, precond_type
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
            info!("writing diagnostics to {}", diag_path.display());
        }

        diag_result.study.save_json(diag_path)?;

        // Save iteration CSV if requested
        if let Some(ref csv_path) = cli.iteration_csv {
            if !cli.quiet {
                info!(
                    "writing per-k-point iteration CSV to {}",
                    csv_path.display()
                );
            }
            diag_result.study.save_iteration_csv(csv_path)?;
        }

        if let Some(ref csv_path) = cli.iteration_csv_combined {
            if !cli.quiet {
                info!(
                    "writing combined iteration CSV to {}",
                    csv_path.display()
                );
            }
            diag_result.study.save_iteration_csv_combined(csv_path)?;
        }

        if !cli.quiet {
            info!(
                "diagnostics saved: {} runs, {} k-points",
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
            info!("wrote {} rows to {}", result.k_path.len(), path.display());
        } else {
            info!("wrote {} rows to stdout", result.k_path.len());
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
