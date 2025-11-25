//! Real-space dielectric sampling utilities + MPB-style smoothing.

use crate::{geometry::Geometry2D, grid::Grid2D};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DielectricOptions {
    pub smoothing: SmoothingOptions,
}

impl Default for DielectricOptions {
    fn default() -> Self {
        Self {
            smoothing: SmoothingOptions::default(),
        }
    }
}

impl DielectricOptions {
    pub fn smoothing_enabled(&self) -> bool {
        self.smoothing.is_enabled()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SmoothingOptions {
    pub mesh_size: usize,
    pub interface_tolerance: f64,
}

impl Default for SmoothingOptions {
    fn default() -> Self {
        Self {
            mesh_size: 4,
            interface_tolerance: 1e-6,
        }
    }
}

impl SmoothingOptions {
    pub fn is_enabled(&self) -> bool {
        self.mesh_size > 1
    }

    pub fn effective_mesh(&self) -> usize {
        self.mesh_size.max(1)
    }

    pub fn tolerance(&self) -> f64 {
        self.interface_tolerance.max(1e-12)
    }
}

#[derive(Debug, Clone)]
pub struct Dielectric2D {
    eps_r: Vec<f64>,
    inv_eps_r: Vec<f64>,
    inv_eps_tensors: Option<Vec<[f64; 4]>>, // row-major [[xx, xy], [yx, yy]]
    unsmoothed_eps: Option<Vec<f64>>,
    unsmoothed_inv_eps: Option<Vec<f64>>,
    pub grid: Grid2D,
}

impl Dielectric2D {
    pub fn from_geometry(geom: &Geometry2D, grid: Grid2D, opts: &DielectricOptions) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid dimensions must be non-zero"
        );
        let raw_eps = sample_raw_eps(geom, grid);
        if !opts.smoothing_enabled() {
            let inv_eps_r = raw_eps
                .iter()
                .map(|&val| {
                    assert!(val > 0.0, "permittivity must be positive");
                    1.0 / val
                })
                .collect();
            return Self {
                eps_r: raw_eps,
                inv_eps_r,
                inv_eps_tensors: None,
                unsmoothed_eps: None,
                unsmoothed_inv_eps: None,
                grid,
            };
        }

        assert!(
            grid.lx > 0.0 && grid.ly > 0.0,
            "grid lengths must be positive when smoothing is enabled"
        );

        let raw_inv_eps: Vec<f64> = raw_eps
            .iter()
            .map(|&val| {
                assert!(val > 0.0, "permittivity must be positive");
                1.0 / val
            })
            .collect();

        let (smoothed_eps, smoothed_inv, tensors) =
            build_smoothed_dielectric(geom, grid, &opts.smoothing);

        Self {
            eps_r: smoothed_eps,
            inv_eps_r: smoothed_inv,
            inv_eps_tensors: Some(tensors),
            unsmoothed_eps: Some(raw_eps),
            unsmoothed_inv_eps: Some(raw_inv_eps),
            grid,
        }
    }

    pub fn eps(&self) -> &[f64] {
        &self.eps_r
    }

    pub fn inv_eps(&self) -> &[f64] {
        &self.inv_eps_r
    }

    pub fn inv_eps_tensors(&self) -> Option<&[[f64; 4]]> {
        self.inv_eps_tensors.as_deref()
    }

    pub fn unsmoothed_eps(&self) -> Option<&[f64]> {
        self.unsmoothed_eps.as_deref()
    }

    pub fn unsmoothed_inv_eps(&self) -> Option<&[f64]> {
        self.unsmoothed_inv_eps.as_deref()
    }
}

fn sample_raw_eps(geom: &Geometry2D, grid: Grid2D) -> Vec<f64> {
    let mut eps_r = vec![geom.eps_bg; grid.len()];
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let frac = [
                fractional_cell_center(ix, grid.nx),
                fractional_cell_center(iy, grid.ny),
            ];
            let idx = grid.idx(ix, iy);
            eps_r[idx] = geom.relative_permittivity_at_fractional(frac);
        }
    }
    eps_r
}

fn build_smoothed_dielectric(
    geom: &Geometry2D,
    grid: Grid2D,
    smoothing: &SmoothingOptions,
) -> (Vec<f64>, Vec<f64>, Vec<[f64; 4]>) {
    let mesh = smoothing.effective_mesh();
    let len = grid.len();
    let mut eps_avg = vec![0.0; len];
    let mut inv_avg = vec![0.0; len];
    let mut tensors = vec![[0.0; 4]; len];

    let unit_cell_area = geom.lattice.cell_area();
    let cell_area = unit_cell_area / (grid.nx * grid.ny) as f64;
    let sub_area = cell_area / (mesh * mesh) as f64;
    let frac_dx = fractional_cell_size(grid.nx);
    let frac_dy = fractional_cell_size(grid.ny);
    let frac_sub_dx = frac_dx / mesh as f64;
    let frac_sub_dy = frac_dy / mesh as f64;
    let spacing_x = vector_norm(geom.lattice.a1) / grid.nx as f64;
    let spacing_y = vector_norm(geom.lattice.a2) / grid.ny as f64;
    let length_scale = spacing_x.max(spacing_y).max(1e-12);
    let tol = smoothing.tolerance();

    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let frac_origin_x = fractional_cell_origin(ix, grid.nx);
            let frac_origin_y = fractional_cell_origin(iy, grid.ny);
            let center_frac = [frac_origin_x + 0.5 * frac_dx, frac_origin_y + 0.5 * frac_dy];
            let center_cart = geom.lattice.fractional_to_cartesian(center_frac);

            let mut eps_sum = 0.0;
            let mut inv_sum = 0.0;
            let mut dipole = [0.0, 0.0];
            let mut eps_min = f64::MAX;
            let mut eps_max = f64::MIN;

            for sub_y in 0..mesh {
                for sub_x in 0..mesh {
                    let sample_frac = [
                        frac_origin_x + (sub_x as f64 + 0.5) * frac_sub_dx,
                        frac_origin_y + (sub_y as f64 + 0.5) * frac_sub_dy,
                    ];
                    let eps = geom.relative_permittivity_at_fractional(sample_frac);
                    assert!(eps > 0.0, "permittivity must be positive");
                    eps_sum += eps * sub_area;
                    inv_sum += (1.0 / eps) * sub_area;
                    eps_min = eps_min.min(eps);
                    eps_max = eps_max.max(eps);
                    let sample_cart = geom.lattice.fractional_to_cartesian(sample_frac);
                    let rel_x = sample_cart[0] - center_cart[0];
                    let rel_y = sample_cart[1] - center_cart[1];
                    dipole[0] += eps * rel_x * sub_area;
                    dipole[1] += eps * rel_y * sub_area;
                }
            }

            let avg_eps = eps_sum / cell_area;
            let avg_inv = inv_sum / cell_area;
            eps_avg[idx] = avg_eps;
            inv_avg[idx] = avg_inv;

            let contrast = (eps_max - eps_min).abs();
            let dip_norm = (dipole[0] * dipole[0] + dipole[1] * dipole[1]).sqrt();
            let dip_scale = avg_eps.abs() * cell_area * length_scale;
            let normalized_dip = if dip_scale > 0.0 {
                dip_norm / dip_scale
            } else {
                0.0
            };
            let use_tensor = contrast > tol && normalized_dip > tol;

            if use_tensor {
                let nx = dipole[0] / dip_norm;
                let ny = dipole[1] / dip_norm;
                tensors[idx] = build_anisotropic_inv_tensor([nx, ny], avg_eps, avg_inv);
            } else {
                tensors[idx] = isotropic_tensor(avg_inv);
            }
        }
    }

    (eps_avg, inv_avg, tensors)
}

fn build_anisotropic_inv_tensor(normal: [f64; 2], avg_eps: f64, avg_inv: f64) -> [f64; 4] {
    let inv_tangential = if avg_eps > 0.0 {
        1.0 / avg_eps
    } else {
        avg_inv
    };
    let base = inv_tangential;
    let delta = avg_inv - inv_tangential;
    let nx = normal[0];
    let ny = normal[1];
    let p_xx = nx * nx;
    let p_xy = nx * ny;
    let p_yy = ny * ny;
    [
        base + delta * p_xx,
        delta * p_xy,
        delta * p_xy,
        base + delta * p_yy,
    ]
}

fn isotropic_tensor(value: f64) -> [f64; 4] {
    [value, 0.0, 0.0, value]
}

fn fractional_cell_center(index: usize, count: usize) -> f64 {
    (index as f64 + 0.5) / count as f64
}

fn fractional_cell_origin(index: usize, count: usize) -> f64 {
    index as f64 / count as f64
}

fn fractional_cell_size(count: usize) -> f64 {
    1.0 / count as f64
}

fn vector_norm(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}
