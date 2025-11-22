use std::fs;
use std::path::PathBuf;

use clap::Parser;
use mpb2d_core::io::JobConfig;

#[derive(Parser, Debug)]
#[command(name = "mpb2d-lite", about = "MPB-like 2D solver CLI (skeleton)")]
struct Cli {
    /// Path to a TOML configuration file
    #[arg(short, long)]
    config: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let raw = fs::read_to_string(&cli.config)?;
    let config: JobConfig = toml::from_str(&raw)?;
    println!("Loaded config for grid {}x{} ({} k-points)", config.grid.nx, config.grid.ny, config.k_path.len());
    Ok(())
}
