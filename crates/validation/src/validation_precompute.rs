use crate::validation_geometry::GeometryPreset;
use crate::validation_lattice::LatticeKind;
use crate::validation_utils::{build_k_plus_g_tables, reciprocal_component};
use clap::Args;
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::backend::SpectralBackend;
use mpb2d_core::dielectric::{Dielectric2D, DielectricOptions, SmoothingOptions};
use mpb2d_core::geometry::{BasisAtom, Geometry2D};
use mpb2d_core::grid::Grid2D;
use num_complex::Complex64;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::mem;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct PrecomputeArgs {
    /// Lattice preset to sample.
    #[arg(long, value_enum)]
    pub lattice: LatticeKind,
    /// Grid resolution (defaults to lattice preset).
    #[arg(long)]
    pub resolution: Option<usize>,
    /// Fractional radius for the circular inclusion.
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
    /// Optional tag recorded in the JSON payload.
    #[arg(long)]
    pub tag: Option<String>,
    /// Output file for the JSON report (stdout if omitted).
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: PrecomputeArgs) -> Result<(), Box<dyn std::error::Error>> {
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
    let te_eps_eff = arithmetic_mean(dielectric.eps()).unwrap_or(1.0);
    let tm_eps_eff = harmonic_mean(dielectric.inv_eps()).unwrap_or(1.0);

    let backend = CpuBackend::new();
    let backend_for_eps = backend.clone();
    let bloch = [0.0, 0.0];
    let (k_plus_g_x, k_plus_g_y, k_plus_g_sq, clamp_mask) = build_k_plus_g_tables(grid, bloch);
    let kx_shifted = build_shifted_axis(grid.nx, grid.lx, bloch[0]);
    let ky_shifted = build_shifted_axis(grid.ny, grid.ly, bloch[1]);

    let k_stats = summary_stats(&k_plus_g_sq);
    let kx_range = min_max(&kx_shifted);
    let ky_range = min_max(&ky_shifted);
    let clamp_fraction = clamp_mask.iter().filter(|&&flag| flag).count() as f64 / grid.len() as f64;

    let mut files = ArtifactFiles::default();
    if let Some(output_path) = args.output.clone() {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let stem = output_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("precompute");
        let eps_csv_name = format!("{}_epsilon_fourier.csv", stem);
        let fft_csv_name = format!("{}_fft_workspace_raw.csv", stem);
        let eps_path = sibling(&output_path, &eps_csv_name);
        let fft_path = sibling(&output_path, &fft_csv_name);
        write_eps_fourier_csv(&backend_for_eps, &eps_path, &dielectric)?;
        write_fft_workspace_raw(
            &fft_path,
            grid,
            &k_plus_g_x,
            &k_plus_g_y,
            &k_plus_g_sq,
            &clamp_mask,
        )?;
        files.epsilon_fourier_csv = Some(eps_csv_name);
        files.fft_workspace_csv = Some(fft_csv_name);
    }

    let workspace = WorkspaceSummary {
        buffers: 3,
        elements_each: grid.len(),
        bytes_total: 3 * grid.len() * mem::size_of::<Complex64>(),
    };

    let report = PrecomputeReport {
        lattice: args.lattice.as_str().to_string(),
        resolution,
        mesh_size: smoothing.mesh_size,
        interface_tolerance: smoothing.interface_tolerance,
        eps_bg: geometry.eps_bg,
        eps_inside,
        radius_fractional: geometry.atoms[0].radius,
        tag: args.tag,
        fft_backend: "cpu_rustfft".to_string(),
        bloch_fractional: [0.0, 0.0],
        te_eps_eff,
        tm_eps_eff,
        clamp_fraction,
        k_plus_g_sq: k_stats,
        kx_shifted: kx_range,
        ky_shifted: ky_range,
        workspace,
        files,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        fs::write(&path, json)?;
        eprintln!(
            "Saved precompute validation report to {} (mesh_size={})",
            path.display(),
            smoothing.mesh_size
        );
    } else {
        println!("{}", json);
    }

    Ok(())
}

#[derive(Serialize)]
struct PrecomputeReport {
    lattice: String,
    resolution: usize,
    mesh_size: usize,
    interface_tolerance: f64,
    eps_bg: f64,
    eps_inside: f64,
    radius_fractional: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tag: Option<String>,
    fft_backend: String,
    bloch_fractional: [f64; 2],
    te_eps_eff: f64,
    tm_eps_eff: f64,
    clamp_fraction: f64,
    k_plus_g_sq: SummaryStats,
    kx_shifted: MinMax,
    ky_shifted: MinMax,
    workspace: WorkspaceSummary,
    files: ArtifactFiles,
}

#[derive(Serialize, Default)]
struct ArtifactFiles {
    #[serde(skip_serializing_if = "Option::is_none")]
    epsilon_fourier_csv: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fft_workspace_csv: Option<String>,
}

#[derive(Serialize)]
struct WorkspaceSummary {
    buffers: usize,
    elements_each: usize,
    bytes_total: usize,
}

#[derive(Serialize, Default)]
struct SummaryStats {
    min: f64,
    max: f64,
    mean: f64,
}

#[derive(Serialize, Default)]
struct MinMax {
    min: f64,
    max: f64,
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

fn min_max(values: &[f64]) -> MinMax {
    if values.is_empty() {
        return MinMax::default();
    }
    let mut min = values[0];
    let mut max = values[0];
    for &value in values {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    MinMax { min, max }
}

fn harmonic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for &value in values {
        if value <= 0.0 {
            return None;
        }
        sum += value;
    }
    Some(1.0 / (sum / values.len() as f64))
}

fn arithmetic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for &value in values {
        sum += value;
    }
    Some(sum / values.len() as f64)
}

fn sibling(path: &Path, file_name: &str) -> PathBuf {
    if let Some(parent) = path.parent() {
        parent.join(file_name)
    } else {
        PathBuf::from(file_name)
    }
}

fn write_eps_fourier_csv(
    backend: &CpuBackend,
    path: &Path,
    dielectric: &Dielectric2D,
) -> std::io::Result<()> {
    let grid = dielectric.grid;
    let mut buffer = backend.alloc_field(grid);
    for (value, &eps) in buffer.as_mut_slice().iter_mut().zip(dielectric.eps()) {
        *value = Complex64::new(eps, 0.0);
    }
    backend.forward_fft_2d(&mut buffer);
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "ix,iy,gx,gy,real,imag")?;
    for iy in 0..grid.ny {
        let gy = reciprocal_component(iy, grid.ny, grid.ly);
        for ix in 0..grid.nx {
            let gx = reciprocal_component(ix, grid.nx, grid.lx);
            let idx = grid.idx(ix, iy);
            let value = buffer.as_slice()[idx];
            writeln!(writer, "{ix},{iy},{gx},{gy},{},{})", value.re, value.im)?;
        }
    }
    writer.flush()
}

fn write_fft_workspace_raw(
    path: &Path,
    grid: Grid2D,
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
    k_plus_g_sq: &[f64],
    clamp_mask: &[bool],
) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "ix,iy,kx_plus_g,ky_plus_g,k_plus_g_sq,clamped")?;
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let kx = k_plus_g_x.get(idx).copied().unwrap_or(0.0);
            let ky = k_plus_g_y.get(idx).copied().unwrap_or(0.0);
            let k_sq = k_plus_g_sq.get(idx).copied().unwrap_or(0.0);
            let clamped_flag = if clamp_mask.get(idx).copied().unwrap_or(false) {
                1
            } else {
                0
            };
            writeln!(writer, "{ix},{iy},{kx},{ky},{k_sq},{clamped_flag}")?;
        }
    }
    writer.flush()
}

fn build_shifted_axis(n: usize, length: f64, bloch_shift: f64) -> Vec<f64> {
    (0..n)
        .map(|idx| reciprocal_component(idx, n, length) + bloch_shift)
        .collect()
}
