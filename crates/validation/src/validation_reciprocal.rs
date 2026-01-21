use crate::validation_lattice::LatticeKind;
use crate::validation_utils::{build_k_plus_g_tables, reciprocal_component};
use clap::Args;
use mpb2d_core::grid::Grid2D;
use mpb2d_core::lattice::{Lattice2D, ReciprocalLattice2D};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct ReciprocalArgs {
    /// Lattice preset whose reciprocal tables should be inspected.
    #[arg(long, value_enum)]
    pub lattice: LatticeKind,
    /// Grid resolution used for FFT indexing (must be >= 8).
    #[arg(long, default_value_t = 32)]
    pub resolution: usize,
    /// Mesh size parameter persisted for traceability with dielectric data.
    #[arg(long, default_value_t = 4)]
    pub mesh_size: usize,
    /// Number of samples per symmetry-leg (>= 2).
    #[arg(long, default_value_t = 3)]
    pub points_per_leg: usize,
    /// Optional JSON output path (stdout when omitted).
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Serialize)]
struct ReciprocalReport {
    lattice: String,
    resolution: usize,
    mesh_size: usize,
    points_per_leg: usize,
    reciprocal_basis: ReciprocalBasis,
    grid: GridMetadata,
    index_map: Vec<IndexEntry>,
    samples: Vec<KPointSample>,
    validation: ReportValidation,
}

#[derive(Serialize)]
struct ReciprocalBasis {
    b1: [f64; 2],
    b2: [f64; 2],
    cell_area: f64,
}

#[derive(Serialize)]
struct GridMetadata {
    shape: [usize; 2],
    extent: [f64; 2],
}

#[derive(Serialize)]
struct IndexEntry {
    ix: usize,
    iy: usize,
    mx: i32,
    my: i32,
    gx: f64,
    gy: f64,
}

#[derive(Serialize)]
struct HistogramBin {
    min: f64,
    max: f64,
    count: usize,
}

#[derive(Serialize)]
struct Histogram {
    bins: Vec<HistogramBin>,
}

#[derive(Serialize, Default, Clone, Copy)]
struct SummaryStats {
    min: f64,
    max: f64,
    mean: f64,
}

#[derive(Serialize)]
struct KPointSample {
    label: String,
    segment: String,
    fractional: [f64; 2],
    cartesian: [f64; 2],
    k_plus_g_x: Vec<f64>,
    k_plus_g_y: Vec<f64>,
    k_plus_g_sq: Vec<f64>,
    clamp_mask: Vec<bool>,
    magnitude_stats: SummaryStats,
    histogram: Histogram,
    aliasing_ok: bool,
    non_negative: bool,
}

#[derive(Serialize)]
struct ReportValidation {
    sample_count: usize,
    aliasing_ok: bool,
    non_negative: bool,
}

#[derive(Clone, Copy)]
struct NamedNode {
    coord: [f64; 2],
    label: &'static str,
}

const SQUARE_PATH: [NamedNode; 4] = [
    NamedNode {
        coord: [0.0, 0.0],
        label: "Γ",
    },
    NamedNode {
        coord: [0.5, 0.0],
        label: "X",
    },
    NamedNode {
        coord: [0.5, 0.5],
        label: "M",
    },
    NamedNode {
        coord: [0.0, 0.0],
        label: "Γ",
    },
];

const TRIANGULAR_PATH: [NamedNode; 4] = [
    NamedNode {
        coord: [0.0, 0.0],
        label: "Γ",
    },
    NamedNode {
        coord: [0.5, 0.0],
        label: "M",
    },
    NamedNode {
        coord: [1.0 / 3.0, 1.0 / 3.0],
        label: "K",
    },
    NamedNode {
        coord: [0.0, 0.0],
        label: "Γ",
    },
];

pub fn run(args: ReciprocalArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.mesh_size == 0 {
        return Err("mesh_size must be >= 1".into());
    }
    if args.resolution < 8 {
        return Err("resolution must be at least 8".into());
    }
    if args.points_per_leg < 2 {
        return Err("points_per_leg must be >= 2".into());
    }

    let lattice = args.lattice.build(1.0);
    let reciprocal = lattice.reciprocal();
    let report = build_report(&args, lattice, reciprocal);
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json)?;
        eprintln!(
            "Saved reciprocal validation data to {} ({} samples)",
            path.display(),
            report.samples.len()
        );
    } else {
        println!("{}", json);
    }
    Ok(())
}

fn build_report(
    args: &ReciprocalArgs,
    lattice: Lattice2D,
    reciprocal: ReciprocalLattice2D,
) -> ReciprocalReport {
    let grid = Grid2D::new(args.resolution, args.resolution, 1.0, 1.0);
    let basis = ReciprocalBasis {
        b1: reciprocal.b1,
        b2: reciprocal.b2,
        cell_area: lattice.cell_area(),
    };
    let grid_meta = GridMetadata {
        shape: [grid.nx, grid.ny],
        extent: [grid.lx, grid.ly],
    };
    let index_map = build_index_map(grid);
    let samples = build_samples(args, &reciprocal, grid);
    let sample_count = samples.len();
    let aliasing_ok = samples.iter().all(|s| s.aliasing_ok);
    let non_negative = samples.iter().all(|s| s.non_negative);

    ReciprocalReport {
        lattice: format!("{:?}", lattice.classify()),
        resolution: args.resolution,
        mesh_size: args.mesh_size,
        points_per_leg: args.points_per_leg,
        reciprocal_basis: basis,
        grid: grid_meta,
        index_map,
        samples,
        validation: ReportValidation {
            sample_count,
            aliasing_ok,
            non_negative,
        },
    }
}

fn build_index_map(grid: Grid2D) -> Vec<IndexEntry> {
    let mut entries = Vec::with_capacity(grid.len());
    for iy in 0..grid.ny {
        let my = centered_index(iy, grid.ny);
        let gy = reciprocal_component(iy, grid.ny, grid.ly);
        for ix in 0..grid.nx {
            let mx = centered_index(ix, grid.nx);
            let gx = reciprocal_component(ix, grid.nx, grid.lx);
            entries.push(IndexEntry {
                ix,
                iy,
                mx,
                my,
                gx,
                gy,
            });
        }
    }
    entries
}

fn build_samples(
    args: &ReciprocalArgs,
    reciprocal: &ReciprocalLattice2D,
    grid: Grid2D,
) -> Vec<KPointSample> {
    let nodes = match args.lattice {
        LatticeKind::Square => &SQUARE_PATH[..],
        LatticeKind::Triangular => &TRIANGULAR_PATH[..],
    };
    let per_leg = args.points_per_leg.max(2);
    let waypoints = sample_path(nodes, per_leg, reciprocal);
    waypoints
        .into_iter()
        .map(|wp| build_sample(wp, grid))
        .collect()
}

struct Waypoint {
    label: String,
    segment: String,
    fractional: [f64; 2],
    cartesian: [f64; 2],
}

fn build_sample(wp: Waypoint, grid: Grid2D) -> KPointSample {
    let (k_plus_g_x, k_plus_g_y, k_plus_g_sq, clamp_mask) =
        build_k_plus_g_tables(grid, wp.cartesian);
    let magnitude_stats = summary_stats(&k_plus_g_sq);
    let histogram = build_histogram(&k_plus_g_sq, 12);
    let aliasing_ok = aliasing_check(grid, wp.cartesian, &k_plus_g_x, &k_plus_g_y, &clamp_mask);
    let non_negative = k_plus_g_sq.iter().all(|value| *value >= -1e-12);

    KPointSample {
        label: wp.label,
        segment: wp.segment,
        fractional: wp.fractional,
        cartesian: wp.cartesian,
        k_plus_g_x,
        k_plus_g_y,
        k_plus_g_sq,
        clamp_mask,
        magnitude_stats,
        histogram,
        aliasing_ok,
        non_negative,
    }
}

fn aliasing_check(
    grid: Grid2D,
    bloch: [f64; 2],
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
    clamp_mask: &[bool],
) -> bool {
    for iy in 0..grid.ny {
        let ky_expected = reciprocal_component(iy, grid.ny, grid.ly) + bloch[1];
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            if clamp_mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let kx_expected = reciprocal_component(ix, grid.nx, grid.lx) + bloch[0];
            let kx = k_plus_g_x[idx];
            let ky = k_plus_g_y[idx];
            if (kx - kx_expected).abs() > 1e-10 || (ky - ky_expected).abs() > 1e-10 {
                return false;
            }
            let expected_sq = kx_expected * kx_expected + ky_expected * ky_expected;
            let actual_sq = k_plus_g_x[idx] * k_plus_g_x[idx] + k_plus_g_y[idx] * k_plus_g_y[idx];
            if (expected_sq - actual_sq).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

fn summary_stats(values: &[f64]) -> SummaryStats {
    if values.is_empty() {
        return SummaryStats::default();
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
    SummaryStats {
        min,
        max,
        mean: sum / values.len() as f64,
    }
}

fn build_histogram(values: &[f64], bins: usize) -> Histogram {
    if values.is_empty() || bins == 0 {
        return Histogram { bins: Vec::new() };
    }
    let stats = summary_stats(values);
    if (stats.max - stats.min).abs() < 1e-15 {
        return Histogram {
            bins: vec![HistogramBin {
                min: stats.min,
                max: stats.max,
                count: values.len(),
            }],
        };
    }
    let width = (stats.max - stats.min) / bins as f64;
    let mut bucket = vec![0usize; bins];
    for &value in values {
        let mut idx = ((value - stats.min) / width).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        bucket[idx] += 1;
    }
    let mut hist_bins = Vec::with_capacity(bins);
    for idx in 0..bins {
        let min = stats.min + idx as f64 * width;
        let max = min + width;
        hist_bins.push(HistogramBin {
            min,
            max,
            count: bucket[idx],
        });
    }
    Histogram { bins: hist_bins }
}

fn sample_path(
    nodes: &[NamedNode],
    per_leg: usize,
    reciprocal: &ReciprocalLattice2D,
) -> Vec<Waypoint> {
    if nodes.len() < 2 {
        return Vec::new();
    }
    let mut waypoints = Vec::new();
    for (leg_idx, pair) in nodes.windows(2).enumerate() {
        let start = pair[0];
        let end = pair[1];
        let segment_label = format!("{}→{}", start.label, end.label);
        for step in 0..per_leg {
            if leg_idx > 0 && step == 0 {
                continue;
            }
            let t = step as f64 / (per_leg - 1) as f64;
            let frac = [
                (1.0 - t) * start.coord[0] + t * end.coord[0],
                (1.0 - t) * start.coord[1] + t * end.coord[1],
            ];
            let cart = fractional_to_cart(frac, reciprocal);
            let label = if step == 0 {
                start.label.to_string()
            } else if step == per_leg - 1 {
                end.label.to_string()
            } else {
                format!("{} midpoint", segment_label)
            };
            waypoints.push(Waypoint {
                label,
                segment: segment_label.clone(),
                fractional: frac,
                cartesian: cart,
            });
        }
    }
    waypoints
}

fn fractional_to_cart(frac: [f64; 2], reciprocal: &ReciprocalLattice2D) -> [f64; 2] {
    [
        reciprocal.b1[0] * frac[0] + reciprocal.b2[0] * frac[1],
        reciprocal.b1[1] * frac[0] + reciprocal.b2[1] * frac[1],
    ]
}

fn centered_index(idx: usize, len: usize) -> i32 {
    let len_i = len as i32;
    let half = len_i / 2;
    let idx_i = idx as i32;
    if idx_i <= half { idx_i } else { idx_i - len_i }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliasing_holds_for_gamma() {
        let args = ReciprocalArgs {
            lattice: LatticeKind::Square,
            resolution: 8,
            mesh_size: 1,
            points_per_leg: 2,
            output: None,
        };
        let grid = Grid2D::new(args.resolution, args.resolution, 1.0, 1.0);
        let wp = Waypoint {
            label: "Γ".to_string(),
            segment: "Γ→X".to_string(),
            fractional: [0.0, 0.0],
            cartesian: [0.0, 0.0],
        };
        let sample = build_sample(wp, grid);
        assert!(sample.aliasing_ok);
        assert!(sample.non_negative);
    }

    #[test]
    fn path_sampling_produces_expected_count() {
        let reciprocal = LatticeKind::Square.build(1.0).reciprocal();
        let samples = sample_path(&SQUARE_PATH, 3, &reciprocal);
        // Γ, midpoint, X, midpoint, M, midpoint, Γ
        assert_eq!(samples.len(), 7);
    }
}
