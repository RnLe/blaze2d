use crate::geometry_sampling::sample_geometry;
use crate::validation_lattice::LatticeKind;
use crate::validation_utils::{classify_to_string, lattice_cell_area};
use clap::Args;
use mpb2d_core::geometry::{BasisAtom, Geometry2D};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct GeometryArgs {
    /// Lattice preset to validate.
    #[arg(long, value_enum)]
    pub lattice: LatticeKind,
    /// Grid resolution used for sampling (overrides lattice-specific defaults).
    #[arg(long)]
    pub resolution: Option<usize>,
    /// Fractional radius for circular inclusion (default derives from lattice).
    #[arg(long)]
    pub radius: Option<f64>,
    /// Background permittivity ε_bg.
    #[arg(long, value_name = "EPS_BG")]
    pub eps_bg: Option<f64>,
    /// Inclusion permittivity ε_hole.
    #[arg(long, value_name = "EPS_IN")]
    pub eps_inside: Option<f64>,
    /// Optional output path for the generated JSON. Defaults to stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: GeometryArgs) -> Result<(), Box<dyn std::error::Error>> {
    let defaults = GeometryPreset::from_kind(args.lattice);
    let resolution = args.resolution.unwrap_or(defaults.resolution);
    if resolution < 8 {
        return Err("resolution must be at least 8 samples".into());
    }
    let radius = args.radius.unwrap_or(defaults.radius);
    if !(radius.is_finite() && radius > 0.0) {
        return Err("radius must be positive and finite".into());
    }
    let eps_bg = args.eps_bg.unwrap_or(defaults.eps_bg);
    let eps_inside = args.eps_inside.unwrap_or(defaults.eps_inside);
    if !(eps_bg.is_finite() && eps_inside.is_finite() && eps_bg > 0.0 && eps_inside > 0.0) {
        return Err("permittivities must be positive and finite".into());
    }

    let lattice = args.lattice.build(1.0);
    let geometry = Geometry2D::air_holes_in_dielectric(
        lattice,
        vec![BasisAtom {
            pos: [0.0, 0.0],
            radius,
            eps_inside,
        }],
        eps_bg,
    );

    let eps_inclusion = geometry.atoms[0].eps_inside;
    let radius_fractional = geometry.atoms[0].radius;
    let sample = sample_geometry(&geometry, resolution, eps_inclusion);
    let cell_area = lattice_cell_area(&geometry.lattice);
    let radius_cartesian = geometry.atoms[0].radius_cartesian(&geometry.lattice);
    let expected_fill = std::f64::consts::PI * radius_cartesian * radius_cartesian / cell_area;
    let fill_error = sample.fill_fraction - expected_fill;

    let report = GeometryReport {
        lattice: args.lattice.as_str().to_string(),
        classification: classify_to_string(geometry.lattice.classify()),
        resolution,
        eps_bg: geometry.eps_bg,
        eps_inside: eps_inclusion,
        radius_fractional,
        radius_cartesian,
        cell_area,
        expected_fill_fraction: expected_fill,
        measured_fill_fraction: sample.fill_fraction,
        fill_fraction_error: fill_error,
        epsilon_stats: FieldStats {
            min: sample.min,
            max: sample.max,
            mean: sample.mean,
        },
        grid: sample.grid,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json)?;
        eprintln!(
            "Saved geometry validation report to {} (fill fraction Δ = {:+.3e})",
            path.display(),
            fill_error
        );
    } else {
        println!("{}", json);
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct GeometryPreset {
    pub resolution: usize,
    pub radius: f64,
    pub eps_bg: f64,
    pub eps_inside: f64,
}

impl GeometryPreset {
    pub fn from_kind(kind: LatticeKind) -> Self {
        match kind {
            LatticeKind::Square => Self {
                resolution: 24,
                radius: 0.28,
                eps_bg: 11.4,
                eps_inside: 1.0,
            },
            LatticeKind::Triangular => Self {
                resolution: 40,
                radius: 0.22,
                eps_bg: 13.5,
                eps_inside: 2.25,
            },
        }
    }
}

#[derive(Serialize)]
struct GeometryReport {
    lattice: String,
    classification: String,
    resolution: usize,
    eps_bg: f64,
    eps_inside: f64,
    radius_fractional: f64,
    radius_cartesian: f64,
    cell_area: f64,
    expected_fill_fraction: f64,
    measured_fill_fraction: f64,
    fill_fraction_error: f64,
    epsilon_stats: FieldStats,
    grid: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct FieldStats {
    min: f64,
    max: f64,
    mean: f64,
}
