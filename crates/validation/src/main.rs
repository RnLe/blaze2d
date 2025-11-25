mod geometry_sampling;
mod validation_geometry;
mod validation_kspace;
mod validation_lattice;
mod validation_precompute;
mod validation_reciprocal;
mod validation_smoothing;
mod validation_utils;

use clap::{Parser, Subcommand};
use validation_geometry::GeometryArgs;
use validation_kspace::KSpaceArgs;
use validation_lattice::LatticeArgs;
use validation_precompute::PrecomputeArgs;
use validation_reciprocal::ReciprocalArgs;
use validation_smoothing::SmoothingArgs;

fn main() {
    if let Err(err) = run() {
        eprintln!("validation error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::LatticeData(args) => validation_lattice::run(args),
        Command::GeometryData(args) => validation_geometry::run(args),
        Command::SmoothingData(args) => validation_smoothing::run(args),
        Command::KSpaceData(args) => validation_kspace::run(args),
        Command::PrecomputeData(args) => validation_precompute::run(args),
        Command::ReciprocalData(args) => validation_reciprocal::run(args),
    }
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Module-level validation helpers for the MPB 2D pipeline."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate validation data for real/reciprocal lattice bases.
    LatticeData(LatticeArgs),
    /// Generate sampled dielectric grids for canonical geometries.
    GeometryData(GeometryArgs),
    /// Evaluate dielectric smoothing impact for canonical geometries.
    SmoothingData(SmoothingArgs),
    /// Generate k-space meshes and symmetry paths for validation plots.
    KSpaceData(KSpaceArgs),
    /// Capture FFT/precompute artifacts for module-level validation.
    PrecomputeData(PrecomputeArgs),
    /// Tabulate k+G data across symmetry paths for Module D validation.
    ReciprocalData(ReciprocalArgs),
}
