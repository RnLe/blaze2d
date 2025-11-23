use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{
    bandstructure::{self, BandStructureResult, Verbosity},
    eigensolver::PreconditionerKind,
    io::{JobConfig, PathPreset},
    symmetry,
};

#[derive(Parser, Debug)]
#[command(name = "mpb2d-lite", about = "MPB-like 2D solver CLI (skeleton)")]
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
    /// Disable automatic symmetry reflections inferred from the lattice
    #[arg(long)]
    no_auto_symmetry: bool,
    /// Write optional pipeline inspection artifacts to the provided directory
    #[arg(long = "dump-pipeline")]
    dump_pipeline: Option<PathBuf>,
    /// Override dielectric smoothing mesh size (1 disables smoothing)
    #[arg(long)]
    mesh_size: Option<usize>,
    /// Disable dielectric smoothing (shorthand for --mesh-size=1)
    #[arg(long)]
    no_smoothing: bool,
    /// Override the eigensolver preconditioner kind
    #[arg(long, value_enum)]
    preconditioner: Option<PreconditionerArg>,
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

#[derive(Clone, Debug, ValueEnum)]
enum PreconditionerArg {
    None,
    #[value(alias = "homogeneous_jacobi")]
    HomogeneousJacobi,
    #[value(alias = "structured_diagonal")]
    StructuredDiagonal,
}

impl From<PreconditionerArg> for PreconditionerKind {
    fn from(value: PreconditionerArg) -> Self {
        match value {
            PreconditionerArg::None => PreconditionerKind::None,
            PreconditionerArg::HomogeneousJacobi => PreconditionerKind::HomogeneousJacobi,
            PreconditionerArg::StructuredDiagonal => PreconditionerKind::StructuredDiagonal,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if !cli.quiet {
        eprintln!("[cli] loading config {}", cli.config.display());
    }
    let raw = fs::read_to_string(&cli.config)?;
    let mut config: JobConfig = toml::from_str(&raw)?;
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
    if let Some(kind) = cli.preconditioner.clone() {
        config.eigensolver.preconditioner = kind.into();
        if !cli.quiet {
            eprintln!(
                "[cli] overriding preconditioner -> {:?}",
                config.eigensolver.preconditioner
            );
        }
    }
    if cli.no_auto_symmetry {
        config.eigensolver.symmetry.disable_auto();
        if !cli.quiet {
            eprintln!("[cli] disabling automatic symmetry reflections");
        }
    }
    if let Some(dir) = cli.dump_pipeline.clone() {
        config.inspection.enable_with_dir(dir.clone());
        if !cli.quiet {
            eprintln!("[cli] pipeline inspection dumps -> {}", dir.display());
        }
    }
    let metrics_cfg = config.metrics.clone();
    let metrics_recorder = metrics_cfg.build_recorder()?;
    let job = bandstructure::BandStructureJob::from(config);
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
    let result = bandstructure::run_with_metrics(
        CpuBackend::new(),
        &job,
        verbosity,
        metrics_recorder.as_ref(),
    );
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

fn emit_csv(result: &BandStructureResult, dest: Option<&Path>) -> io::Result<()> {
    let mut writer: Box<dyn Write> = match dest {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout())),
    };
    let max_bands = result.bands.iter().map(|row| row.len()).max().unwrap_or(0);
    write!(writer, "k_index,kx,ky,k_distance")?;
    for band_idx in 0..max_bands {
        write!(writer, ",band{}", band_idx + 1)?;
    }
    writeln!(writer)?;

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
            let normalized = omega / (2.0 * std::f64::consts::PI);
            write!(writer, ",{}", normalized)?;
        }
        if bands.len() < max_bands {
            for _ in bands.len()..max_bands {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }

    writer.flush()
}
