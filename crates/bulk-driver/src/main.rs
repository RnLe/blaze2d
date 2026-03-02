//! MPB2D Bulk Driver - Multi-threaded parameter sweep CLI.
//!
//! This tool reads a bulk configuration TOML file and executes many band structure
//! calculations in parallel, ideal for parameter studies.
//!
//! # Usage
//!
//! ```bash
//! mpb2d-bulk --config sweep.toml
//! mpb2d-bulk --config sweep.toml --threads -1  # Adaptive threading
//! ```
//!
//! Use `--dry-run` to see how many jobs would be executed without running them.

use std::path::PathBuf;
use std::process;
use std::time::Duration;

use clap::Parser;
use env_logger::Builder;
use log::{error, warn};

use mpb2d_bulk_driver::{BulkConfig, BulkDriver};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "mpb2d-bulk",
    about = "Multi-threaded parameter sweep driver for MPB2D",
    version
)]
struct Cli {
    /// Path to bulk configuration TOML file
    #[arg(short, long)]
    config: PathBuf,

    /// Dry run: show job count and parameter statistics without executing
    #[arg(long)]
    dry_run: bool,

    /// Number of threads (-1 = adaptive/smart, 0 = all physical cores, N = fixed count)
    #[arg(short = 'j', long, default_value = "0", allow_hyphen_values = true)]
    threads: i32,

    /// Verbose output (show debug logs and thread adjustments)
    #[arg(short, long)]
    verbose: bool,

    /// Quiet mode: suppress all output except errors
    #[arg(short, long)]
    quiet: bool,

    /// Override output directory (for full mode)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override output filename (for selective mode)
    #[arg(long)]
    output_file: Option<PathBuf>,

    /// Show first N expanded job configurations (for debugging)
    #[arg(long)]
    show_jobs: Option<usize>,

    /// Stress test mode: simulate jobs without execution (use with --dry-run)
    #[arg(long)]
    stress_test: bool,

    /// Duration in ms for simulated jobs in stress test mode
    #[arg(long, default_value = "50")]
    stress_duration_ms: u64,

    /// Vary job duration randomly in stress test mode (±50%)
    #[arg(long)]
    stress_vary: bool,

    /// Benchmark mode: run real solves but skip file output
    #[arg(long)]
    benchmark: bool,
}

// ============================================================================
// Logging
// ============================================================================

fn init_logging(verbose: bool, quiet: bool) {
    use log::LevelFilter;

    // We control what we show
    let our_level = if quiet {
        LevelFilter::Error
    } else if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Warn  // Show our own warnings
    };

    // Suppress noisy library logs unless verbose
    let lib_level = if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Error  // Only errors from core library
    };

    Builder::new()
        .filter_module("mpb2d_bulk_driver", our_level)
        .filter_module("mpb2d_core", lib_level)
        .filter_level(lib_level)
        .format(|buf, record| {
            use std::io::Write;
            let level = record.level();
            let (color_start, color_end) = match level {
                log::Level::Error => ("\x1b[1;31m", "\x1b[0m"),
                log::Level::Warn => ("\x1b[1;33m", "\x1b[0m"),
                log::Level::Info => ("\x1b[32m", "\x1b[0m"),
                log::Level::Debug => ("\x1b[36m", "\x1b[0m"),
                log::Level::Trace => ("\x1b[35m", "\x1b[0m"),
            };
            writeln!(buf, "{}{:5}{} {}", color_start, level, color_end, record.args())
        })
        .init();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let cli = Cli::parse();

    init_logging(cli.verbose, cli.quiet);

    // Load configuration
    let mut config = match BulkConfig::from_file(&cli.config) {
        Ok(c) => c,
        Err(e) => {
            error!("failed to load configuration: {}", e);
            process::exit(1);
        }
    };

    // Apply CLI overrides
    if cli.verbose {
        config.bulk.verbose = true;
    }
    if cli.dry_run && !cli.stress_test {
        config.bulk.dry_run = true;
    }
    if let Some(ref output) = cli.output {
        config.output.directory = output.clone();
    }
    if let Some(ref output_file) = cli.output_file {
        config.output.filename = output_file.clone();
    }

    // Determine thread mode from CLI
    let requested_threads = if cli.threads == 0 {
        // 0 means use default from config, or all cores
        config.bulk.threads.map(|t| t as i32)
    } else {
        Some(cli.threads)
    };

    // Create driver with thread mode
    let driver = BulkDriver::new(config, requested_threads);

    // Show job details if requested
    if let Some(n) = cli.show_jobs {
        println!("Showing first {} job configurations:\n", n);
        for job in driver.jobs().iter().take(n) {
            println!("Job {}:", job.index);
            println!("  eps_bg: {:.4}", job.params.eps_bg);
            println!("  resolution: {}", job.params.resolution);
            println!("  polarization: {:?}", job.params.polarization);
            if let Some(ref lt) = job.params.lattice_type {
                println!("  lattice_type: {}", lt);
            }
            for atom in &job.params.atoms {
                println!(
                    "  atom{}: pos=[{:.4}, {:.4}], r={:.4}, eps={:.4}",
                    atom.index, atom.pos[0], atom.pos[1], atom.radius, atom.eps_inside
                );
            }
            println!();
        }
        if driver.job_count() > n {
            println!("... and {} more jobs\n", driver.job_count() - n);
        }
    }

    // Stress test mode
    if cli.stress_test {
        let duration = Duration::from_millis(cli.stress_duration_ms);
        match driver.stress_test(duration, cli.stress_vary) {
            Ok(_stats) => {
                // Stats already printed by stress_test
            }
            Err(e) => {
                error!("stress test error: {}", e);
                process::exit(1);
            }
        }
        return;
    }

    // Benchmark mode (real solves, no file output)
    if cli.benchmark {
        match driver.benchmark() {
            Ok(stats) => {
                if stats.failed > 0 {
                    warn!(
                        "benchmark completed with errors: {}/{} jobs succeeded",
                        stats.completed, stats.total_jobs
                    );
                    process::exit(1);
                }
            }
            Err(e) => {
                error!("benchmark error: {}", e);
                process::exit(1);
            }
        }
        return;
    }

    // Dry run mode
    if cli.dry_run {
        let stats = driver.dry_run();
        stats.print_report();
        return;
    }

    // Execute
    match driver.run() {
        Ok(stats) => {
            if stats.failed > 0 {
                warn!(
                    "completed with errors: {}/{} jobs succeeded",
                    stats.completed, stats.total_jobs
                );
                process::exit(1);
            }
            // Success message already printed by driver.run()
        }
        Err(e) => {
            error!("driver error: {}", e);
            process::exit(1);
        }
    }
}
