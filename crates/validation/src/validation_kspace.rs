use crate::validation_lattice::LatticeKind;
use clap::{Args, ValueEnum};
use mpb2d_core::lattice::{Lattice2D, ReciprocalLattice2D};
use serde::Serialize;
use std::{collections::HashSet, fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct KSpaceArgs {
    /// Lattice preset to use when building reciprocal data.
    #[arg(long, value_enum)]
    pub lattice: LatticeKind,
    /// Generate either a uniform k-mesh or a high-symmetry path.
    #[arg(long = "mode", value_enum)]
    pub mode: KSpaceMode,
    /// Optional override for mesh resolution in u (b1) direction.
    #[arg(long)]
    pub mesh_nx: Option<usize>,
    /// Optional override for mesh resolution in v (b2) direction.
    #[arg(long)]
    pub mesh_ny: Option<usize>,
    /// Half-extent (in reduced coordinates) covered by the mesh along each axis.
    #[arg(long)]
    pub mesh_extent: Option<f64>,
    /// Segments per leg when densifying preset paths (defaults to 12).
    #[arg(long)]
    pub segments_per_leg: Option<usize>,
    /// Path preset to use (defaults to lattice-appropriate option when omitted).
    #[arg(long, value_enum)]
    pub path_kind: Option<PathKind>,
    /// Custom path nodes expressed as "u,v;u,v;..." in reduced coordinates.
    #[arg(long = "custom-path")]
    pub custom_path: Option<String>,
    /// Output file for JSON report (stdout if omitted).
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum KSpaceMode {
    Mesh,
    Path,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PathKind {
    Square,
    Triangular,
    Custom,
}

#[derive(Serialize)]
struct KSpaceReport {
    lattice: String,
    mode: KSpaceMode,
    reciprocal_basis: ReciprocalBasis,
    reciprocal_grid: Vec<ReciprocalPoint>,
    mesh: Option<MeshMetadata>,
    path: Option<PathMetadata>,
    points: Vec<PointEntry>,
    validation: ValidationSummary,
}

#[derive(Serialize)]
struct ReciprocalBasis {
    b1: [f64; 2],
    b2: [f64; 2],
    cell_area: f64,
}

#[derive(Serialize)]
struct ReciprocalPoint {
    indices: [i32; 2],
    cartesian: [f64; 2],
}

#[derive(Serialize)]
struct MeshMetadata {
    shape: [usize; 2],
    extent: f64,
}

#[derive(Serialize)]
struct PathMetadata {
    preset: Option<String>,
    segments_per_leg: usize,
    total_length: f64,
    segments: Vec<SegmentSummary>,
}

#[derive(Serialize, Clone)]
struct SegmentSummary {
    name: String,
    num_samples: usize,
    length: f64,
}

#[derive(Serialize, Clone)]
struct PointEntry {
    fractional: [f64; 2],
    cartesian: [f64; 2],
    segment: Option<String>,
    cumulative_length: Option<f64>,
}

#[derive(Serialize)]
struct ValidationSummary {
    unique_points: usize,
    total_points: usize,
    includes_gamma: bool,
    wraps_back_to_gamma: Option<bool>,
    adjacency_ok: Option<bool>,
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

#[derive(Clone, Copy)]
struct NamedNode {
    coord: [f64; 2],
    label: &'static str,
}

pub fn run(args: KSpaceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let lattice = args.lattice.build(1.0);
    let reciprocal = lattice.reciprocal();
    let reciprocal_basis = ReciprocalBasis {
        b1: reciprocal.b1,
        b2: reciprocal.b2,
        cell_area: lattice.cell_area(),
    };
    let reciprocal_grid = generate_reciprocal_grid(&reciprocal);

    let report = match args.mode {
        KSpaceMode::Mesh => build_mesh_report(
            &lattice,
            &reciprocal,
            reciprocal_basis,
            reciprocal_grid,
            &args,
        )?,
        KSpaceMode::Path => build_path_report(
            &lattice,
            &reciprocal,
            reciprocal_basis,
            reciprocal_grid,
            &args,
        )?,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json)?;
        eprintln!("Saved k-space report to {}", path.display());
    } else {
        println!("{}", json);
    }

    Ok(())
}

fn build_mesh_report(
    lattice: &Lattice2D,
    reciprocal: &ReciprocalLattice2D,
    reciprocal_basis: ReciprocalBasis,
    reciprocal_grid: Vec<ReciprocalPoint>,
    args: &KSpaceArgs,
) -> Result<KSpaceReport, Box<dyn std::error::Error>> {
    let nx = args.mesh_nx.unwrap_or(13).max(2);
    let ny = args.mesh_ny.unwrap_or(9).max(2);
    let extent = args.mesh_extent.unwrap_or(0.55).abs().max(0.1);
    let mesh_points = build_mesh_points(nx, ny, extent, reciprocal);
    let unique_points = count_unique(&mesh_points);
    let includes_gamma = includes_gamma(&mesh_points);
    let validation = ValidationSummary {
        unique_points,
        total_points: mesh_points.len(),
        includes_gamma,
        wraps_back_to_gamma: None,
        adjacency_ok: None,
    };

    Ok(KSpaceReport {
        lattice: format!("{:?}", lattice.classify()),
        mode: KSpaceMode::Mesh,
        reciprocal_basis,
        reciprocal_grid,
        mesh: Some(MeshMetadata {
            shape: [nx, ny],
            extent,
        }),
        path: None,
        points: mesh_points,
        validation,
    })
}

fn build_path_report(
    lattice: &Lattice2D,
    reciprocal: &ReciprocalLattice2D,
    reciprocal_basis: ReciprocalBasis,
    reciprocal_grid: Vec<ReciprocalPoint>,
    args: &KSpaceArgs,
) -> Result<KSpaceReport, Box<dyn std::error::Error>> {
    let segments_per_leg = args.segments_per_leg.unwrap_or(12).max(1);
    let resolved_kind = resolve_path_kind(args);
    let (points, segments) = match resolved_kind {
        PathKind::Square => densify_with_labels(&SQUARE_PATH, segments_per_leg, reciprocal),
        PathKind::Triangular => densify_with_labels(&TRIANGULAR_PATH, segments_per_leg, reciprocal),
        PathKind::Custom => build_custom_path(args, reciprocal)?,
    };

    let total_length = points
        .last()
        .and_then(|p| p.cumulative_length)
        .unwrap_or(0.0);
    let includes_gamma = includes_gamma(&points);
    let wraps_back = wraps_back_to_gamma(&points);
    let adjacency_ok = match resolved_kind {
        PathKind::Square => Some(segments_aligned(&points, &SQUARE_PATH)),
        PathKind::Triangular => Some(segments_aligned(&points, &TRIANGULAR_PATH)),
        PathKind::Custom => None,
    };
    let validation = ValidationSummary {
        unique_points: count_unique(&points),
        total_points: points.len(),
        includes_gamma,
        wraps_back_to_gamma: Some(wraps_back),
        adjacency_ok,
    };

    Ok(KSpaceReport {
        lattice: format!("{:?}", lattice.classify()),
        mode: KSpaceMode::Path,
        reciprocal_basis,
        reciprocal_grid,
        mesh: None,
        path: Some(PathMetadata {
            preset: path_kind_label(resolved_kind, args),
            segments_per_leg,
            total_length,
            segments,
        }),
        points,
        validation,
    })
}

fn resolve_path_kind(args: &KSpaceArgs) -> PathKind {
    if let Some(kind) = args.path_kind {
        return kind;
    }
    if args.custom_path.is_some() {
        return PathKind::Custom;
    }
    match args.lattice {
        LatticeKind::Square => PathKind::Square,
        LatticeKind::Triangular => PathKind::Triangular,
    }
}

fn path_kind_label(kind: PathKind, args: &KSpaceArgs) -> Option<String> {
    match kind {
        PathKind::Square => Some("square".to_string()),
        PathKind::Triangular => Some("triangular".to_string()),
        PathKind::Custom => args
            .custom_path
            .as_ref()
            .map(|s| format!("custom ({} nodes)", s.split(';').count())),
    }
}

fn build_mesh_points(
    nx: usize,
    ny: usize,
    extent: f64,
    reciprocal: &ReciprocalLattice2D,
) -> Vec<PointEntry> {
    let mut points = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let v = interpolate(-extent, extent, iy, ny);
        for ix in 0..nx {
            let u = interpolate(-extent, extent, ix, nx);
            let cart = fractional_to_cart([u, v], reciprocal);
            points.push(PointEntry {
                fractional: [u, v],
                cartesian: cart,
                segment: None,
                cumulative_length: None,
            });
        }
    }
    points
}

fn build_custom_path(
    args: &KSpaceArgs,
    reciprocal: &ReciprocalLattice2D,
) -> Result<(Vec<PointEntry>, Vec<SegmentSummary>), Box<dyn std::error::Error>> {
    let raw = args
        .custom_path
        .as_ref()
        .ok_or("custom-path must be provided when path_kind=custom")?;
    let nodes = parse_custom_path(raw)?;
    if nodes.len() < 2 {
        return Err("custom path must contain at least two nodes".into());
    }
    let mut points = Vec::new();
    let mut summaries = Vec::new();
    let mut prev_cart = fractional_to_cart(nodes[0], reciprocal);
    points.push(PointEntry {
        fractional: nodes[0],
        cartesian: prev_cart,
        segment: Some(format!("P0 ({:.2},{:.2})", nodes[0][0], nodes[0][1])),
        cumulative_length: Some(0.0),
    });
    let mut cumulative = 0.0;
    for idx in 0..nodes.len() - 1 {
        let end = nodes[idx + 1];
        let label = format!("P{idx}→P{}", idx + 1);
        let end_cart = fractional_to_cart(end, reciprocal);
        let segment_len = distance(prev_cart, end_cart);
        cumulative += segment_len;
        points.push(PointEntry {
            fractional: end,
            cartesian: end_cart,
            segment: Some(label.clone()),
            cumulative_length: Some(cumulative),
        });
        summaries.push(SegmentSummary {
            name: label,
            num_samples: 2,
            length: segment_len,
        });
        prev_cart = end_cart;
    }
    Ok((points, summaries))
}

fn densify_with_labels(
    nodes: &[NamedNode],
    segments_per_leg: usize,
    reciprocal: &ReciprocalLattice2D,
) -> (Vec<PointEntry>, Vec<SegmentSummary>) {
    let mut points = Vec::new();
    let mut summaries = Vec::new();
    if nodes.is_empty() {
        return (points, summaries);
    }
    let mut cumulative = 0.0;
    let start_cart = fractional_to_cart(nodes[0].coord, reciprocal);
    points.push(PointEntry {
        fractional: nodes[0].coord,
        cartesian: start_cart,
        segment: Some(nodes[0].label.to_string()),
        cumulative_length: Some(0.0),
    });
    let mut prev_cart = start_cart;
    for pair in nodes.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let label = format!("{}→{}", start.label, end.label);
        let mut segment_len = 0.0;
        let mut samples = 0usize;
        for step in 1..=segments_per_leg {
            let t = step as f64 / segments_per_leg as f64;
            let frac = [
                (1.0 - t) * start.coord[0] + t * end.coord[0],
                (1.0 - t) * start.coord[1] + t * end.coord[1],
            ];
            let cart = fractional_to_cart(frac, reciprocal);
            let delta = distance(prev_cart, cart);
            cumulative += delta;
            segment_len += delta;
            samples += 1;
            points.push(PointEntry {
                fractional: frac,
                cartesian: cart,
                segment: Some(label.clone()),
                cumulative_length: Some(cumulative),
            });
            prev_cart = cart;
        }
        summaries.push(SegmentSummary {
            name: label,
            num_samples: samples,
            length: segment_len,
        });
    }
    (points, summaries)
}

fn generate_reciprocal_grid(reciprocal: &ReciprocalLattice2D) -> Vec<ReciprocalPoint> {
    let mut points = Vec::with_capacity(9);
    for j in -1..=1 {
        for i in -1..=1 {
            let cart = fractional_to_cart([i as f64, j as f64], reciprocal);
            points.push(ReciprocalPoint {
                indices: [i, j],
                cartesian: cart,
            });
        }
    }
    points
}

fn fractional_to_cart(frac: [f64; 2], reciprocal: &ReciprocalLattice2D) -> [f64; 2] {
    [
        reciprocal.b1[0] * frac[0] + reciprocal.b2[0] * frac[1],
        reciprocal.b1[1] * frac[0] + reciprocal.b2[1] * frac[1],
    ]
}

fn interpolate(min: f64, max: f64, idx: usize, count: usize) -> f64 {
    if count <= 1 {
        return min;
    }
    let t = idx as f64 / (count - 1) as f64;
    min + t * (max - min)
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn count_unique(points: &[PointEntry]) -> usize {
    let mut uniq = HashSet::new();
    for point in points {
        let key = (
            (point.fractional[0] * 1e6).round() as i64,
            (point.fractional[1] * 1e6).round() as i64,
        );
        uniq.insert(key);
    }
    uniq.len()
}

fn includes_gamma(points: &[PointEntry]) -> bool {
    points
        .iter()
        .any(|p| p.fractional[0].abs() < 1e-9 && p.fractional[1].abs() < 1e-9)
}

fn wraps_back_to_gamma(points: &[PointEntry]) -> bool {
    if points.is_empty() {
        return false;
    }
    let first = &points[0];
    let last = &points[points.len() - 1];
    (first.fractional[0] - last.fractional[0]).abs() < 1e-9
        && (first.fractional[1] - last.fractional[1]).abs() < 1e-9
}

fn segments_aligned(points: &[PointEntry], definition: &[NamedNode]) -> bool {
    if definition.len() < 2 {
        return true;
    }
    for pair in definition.windows(2) {
        let label = format!("{}→{}", pair[0].label, pair[1].label);
        let expected = [
            pair[1].coord[0] - pair[0].coord[0],
            pair[1].coord[1] - pair[0].coord[1],
        ];
        let mut last_point = pair[0].coord;
        for point in points
            .iter()
            .filter(|p| p.segment.as_deref() == Some(&label))
        {
            let delta = [
                point.fractional[0] - last_point[0],
                point.fractional[1] - last_point[1],
            ];
            let cross = expected[0] * delta[1] - expected[1] * delta[0];
            if cross.abs() > 1e-6 {
                return false;
            }
            last_point = point.fractional;
        }
    }
    true
}

fn parse_custom_path(raw: &str) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error>> {
    let mut nodes = Vec::new();
    for entry in raw.split(';') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split(',');
        let u: f64 = parts.next().ok_or("missing u coordinate")?.trim().parse()?;
        let v: f64 = parts.next().ok_or("missing v coordinate")?.trim().parse()?;
        nodes.push([u, v]);
    }
    if nodes.is_empty() {
        return Err("custom path string produced no nodes".into());
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_points_include_gamma() {
        let lattice = Lattice2D::square(1.0);
        let reciprocal = lattice.reciprocal();
        let points = build_mesh_points(5, 5, 0.5, &reciprocal);
        assert!(includes_gamma(&points));
        assert_eq!(count_unique(&points), points.len());
    }

    #[test]
    fn custom_path_parser_roundtrips() {
        let nodes = parse_custom_path("0,0; 0.5,0.0; 0.5,0.5; 0,0").unwrap();
        assert_eq!(nodes.len(), 4);
        assert!((nodes[1][0] - 0.5).abs() < 1e-12);
        assert!((nodes[2][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn square_segments_align() {
        let lattice = Lattice2D::square(1.0);
        let reciprocal = lattice.reciprocal();
        let (points, _) = densify_with_labels(&SQUARE_PATH, 4, &reciprocal);
        assert!(segments_aligned(&points, &SQUARE_PATH));
        assert!(wraps_back_to_gamma(&points));
    }
}
