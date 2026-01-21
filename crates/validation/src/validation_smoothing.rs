use crate::geometry_sampling::{SampledField, sample_geometry};
use crate::validation_geometry::GeometryPreset;
use crate::validation_lattice::LatticeKind;
use clap::Args;
use mpb2d_core::dielectric::{Dielectric2D, DielectricOptions, SmoothingOptions};
use mpb2d_core::geometry::{BasisAtom, Geometry2D};
use mpb2d_core::grid::Grid2D;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct SmoothingArgs {
    /// Lattice preset to validate.
    #[arg(long, value_enum)]
    pub lattice: LatticeKind,
    /// Grid resolution (defaults to lattice preset).
    #[arg(long)]
    pub resolution: Option<usize>,
    /// Fractional radius for circular inclusion (defaults to preset).
    #[arg(long)]
    pub radius: Option<f64>,
    /// Background permittivity ε_bg.
    #[arg(long, value_name = "EPS_BG")]
    pub eps_bg: Option<f64>,
    /// Inclusion permittivity ε_hole.
    #[arg(long, value_name = "EPS_IN")]
    pub eps_inside: Option<f64>,
    /// Dielectric smoothing mesh size (1 disables smoothing).
    #[arg(long, default_value_t = 4)]
    pub mesh_size: usize,
    /// Optional override for interface tolerance used by smoothing.
    #[arg(long)]
    pub interface_tolerance: Option<f64>,
    /// Output file for JSON report (stdout if omitted).
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: SmoothingArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.mesh_size == 0 {
        return Err("mesh_size must be >= 1".into());
    }
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

    let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);
    let mut smoothing = SmoothingOptions::default();
    smoothing.mesh_size = args.mesh_size.max(1);
    if let Some(tol) = args.interface_tolerance {
        smoothing.interface_tolerance = tol.max(1e-12);
    }
    let mut dielectric_opts = DielectricOptions::default();
    dielectric_opts.smoothing = smoothing.clone();

    let dielectric = Dielectric2D::from_geometry(&geometry, grid, &dielectric_opts);
    let raw_sample = sample_geometry(&geometry, resolution, eps_inside);
    let smoothed_slice = dielectric.eps();
    let raw_flat: Vec<f64> = raw_sample
        .grid
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    let smoothed_grid = if smoothing.mesh_size > 1 {
        Some(reshape(smoothed_slice, grid.nx, grid.ny))
    } else {
        None
    };

    let delta_stats = if smoothing.mesh_size > 1 {
        Some(compute_delta_stats(smoothed_slice, &raw_flat))
    } else {
        None
    };

    let report = SmoothingReport {
        lattice: args.lattice.as_str().to_string(),
        resolution,
        mesh_size: smoothing.mesh_size,
        interface_tolerance: smoothing.interface_tolerance,
        eps_bg: geometry.eps_bg,
        eps_inside,
        radius_fractional: geometry.atoms[0].radius,
        raw_stats: FieldStats::from_sample(&raw_sample),
        smoothed_stats: smoothed_grid
            .as_ref()
            .map(|_| stats_from_slice(smoothed_slice)),
        delta_stats,
        raw_grid: raw_sample.grid,
        smoothed_grid,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json)?;
        eprintln!(
            "Saved smoothing validation report to {} (mesh_size={})",
            path.display(),
            smoothing.mesh_size
        );
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn reshape(values: &[f64], nx: usize, ny: usize) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(ny);
    for iy in 0..ny {
        let start = iy * nx;
        let end = start + nx;
        rows.push(values[start..end].to_vec());
    }
    rows
}

fn stats_from_slice(values: &[f64]) -> FieldStats {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    FieldStats {
        min,
        max,
        mean: sum / values.len() as f64,
        fill_fraction: None,
    }
}

fn compute_delta_stats(smoothed: &[f64], raw: &[f64]) -> DeltaStats {
    assert_eq!(smoothed.len(), raw.len());
    let mut max_abs: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for (&s, &r) in smoothed.iter().zip(raw.iter()) {
        let delta = s - r;
        let abs = delta.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        sum_sq += delta * delta;
    }
    let len = smoothed.len() as f64;
    DeltaStats {
        l2_norm: (sum_sq / len).sqrt(),
        mean_abs: sum_abs / len,
        max_abs,
    }
}

#[derive(Serialize)]
struct SmoothingReport {
    lattice: String,
    resolution: usize,
    mesh_size: usize,
    interface_tolerance: f64,
    eps_bg: f64,
    eps_inside: f64,
    radius_fractional: f64,
    raw_stats: FieldStats,
    smoothed_stats: Option<FieldStats>,
    delta_stats: Option<DeltaStats>,
    raw_grid: Vec<Vec<f64>>,
    smoothed_grid: Option<Vec<Vec<f64>>>,
}

#[derive(Serialize)]
struct FieldStats {
    min: f64,
    max: f64,
    mean: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    fill_fraction: Option<f64>,
}

impl FieldStats {
    fn from_sample(sample: &SampledField) -> Self {
        Self {
            min: sample.min,
            max: sample.max,
            mean: sample.mean,
            fill_fraction: Some(sample.fill_fraction),
        }
    }
}

#[derive(Serialize)]
struct DeltaStats {
    l2_norm: f64,
    mean_abs: f64,
    max_abs: f64,
}
