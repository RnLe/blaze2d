//! Real-space dielectric sampling utilities + MPB-style smoothing.
//!
//! This module provides two smoothing modes:
//! - **Subgrid sampling**: Uses numerical integration over a sub-grid
//! - **Analytic geometry** (default): Uses exact geometric formulas for known shapes (circles)
//!
//! The analytic mode provides more accurate results at material interfaces by
//! computing exact filling fractions and interface normals, following the MPB/Meep
//! approach described in Farjadpour et al., Optics Letters 31, 2972 (2006).
//!
//! # Grid Sampling Convention
//!
//! We sample epsilon at **grid nodes**: position `i/N` for index `i`.
//! This matches MPB's convention for direct comparison.

use crate::{
    analytic_geometry::{AnalyticShape, Circle, compute_smoothed_dielectric},
    geometry::Geometry2D,
    grid::Grid2D,
};
use serde::{Deserialize, Serialize};

/// Smoothing method for material interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SmoothingMethod {
    /// Numerical subgrid sampling (original method)
    Subgrid,
    /// Analytic geometry-aware smoothing (MPB-style)
    /// Uses exact filling fractions and interface normals for circles.
    #[default]
    Analytic,
}

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
    /// Subgrid mesh size for numerical integration (only used with Subgrid method)
    pub mesh_size: usize,
    /// Tolerance for detecting material interfaces
    pub interface_tolerance: f64,
    /// Smoothing method to use
    pub method: SmoothingMethod,
}

impl Default for SmoothingOptions {
    fn default() -> Self {
        Self {
            mesh_size: 3,
            interface_tolerance: 1e-6,
            method: SmoothingMethod::default(),
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
    /// Reciprocal lattice vectors b1, b2 for computing G-vectors.
    /// G = n1*b1 + n2*b2 for integer indices n1, n2.
    reciprocal_b1: [f64; 2],
    reciprocal_b2: [f64; 2],
}

impl Dielectric2D {
    pub fn from_geometry(geom: &Geometry2D, grid: Grid2D, opts: &DielectricOptions) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid dimensions must be non-zero"
        );

        // Compute reciprocal lattice vectors for G-vector construction
        let reciprocal = geom.lattice.reciprocal();
        let reciprocal_b1 = reciprocal.b1;
        let reciprocal_b2 = reciprocal.b2;

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
                reciprocal_b1,
                reciprocal_b2,
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

        let (smoothed_eps, smoothed_inv, tensors) = match opts.smoothing.method {
            SmoothingMethod::Subgrid => {
                build_smoothed_dielectric_subgrid(geom, grid, &opts.smoothing)
            }
            SmoothingMethod::Analytic => {
                build_smoothed_dielectric_analytic(geom, grid, &opts.smoothing)
            }
        };

        Self {
            eps_r: smoothed_eps,
            inv_eps_r: smoothed_inv,
            inv_eps_tensors: Some(tensors),
            unsmoothed_eps: Some(raw_eps),
            unsmoothed_inv_eps: Some(raw_inv_eps),
            grid,
            reciprocal_b1,
            reciprocal_b2,
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

    /// Get the first reciprocal lattice vector b1.
    /// G-vectors are computed as G = n1*b1 + n2*b2.
    pub fn reciprocal_b1(&self) -> [f64; 2] {
        self.reciprocal_b1
    }

    /// Get the second reciprocal lattice vector b2.
    /// G-vectors are computed as G = n1*b1 + n2*b2.
    pub fn reciprocal_b2(&self) -> [f64; 2] {
        self.reciprocal_b2
    }

    /// Export epsilon data to CSV file.
    ///
    /// Format: `ix,iy,frac_x,frac_y,eps_smoothed,inv_eps_smoothed[,eps_raw,inv_eps_raw]`
    ///
    /// The raw (unsmoothed) columns are only present if smoothing was enabled.
    pub fn save_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;

        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        let has_raw = self.unsmoothed_eps.is_some();

        // Write header
        if has_raw {
            writeln!(
                writer,
                "ix,iy,frac_x,frac_y,eps_smoothed,inv_eps_smoothed,eps_raw,inv_eps_raw"
            )?;
        } else {
            writeln!(writer, "ix,iy,frac_x,frac_y,eps,inv_eps")?;
        }

        // Write data rows
        for iy in 0..self.grid.ny {
            for ix in 0..self.grid.nx {
                let idx = self.grid.idx(ix, iy);
                let frac_x = (ix as f64 + 0.5) / self.grid.nx as f64;
                let frac_y = (iy as f64 + 0.5) / self.grid.ny as f64;

                if has_raw {
                    let eps_raw = self
                        .unsmoothed_eps
                        .as_ref()
                        .map(|v| v[idx])
                        .unwrap_or(f64::NAN);
                    let inv_eps_raw = self
                        .unsmoothed_inv_eps
                        .as_ref()
                        .map(|v| v[idx])
                        .unwrap_or(f64::NAN);
                    writeln!(
                        writer,
                        "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                        ix,
                        iy,
                        frac_x,
                        frac_y,
                        self.eps_r[idx],
                        self.inv_eps_r[idx],
                        eps_raw,
                        inv_eps_raw
                    )?;
                } else {
                    writeln!(
                        writer,
                        "{},{},{:.12e},{:.12e},{:.12e},{:.12e}",
                        ix, iy, frac_x, frac_y, self.eps_r[idx], self.inv_eps_r[idx]
                    )?;
                }
            }
        }

        writer.flush()
    }

    /// Export epsilon tensor data to CSV file.
    ///
    /// Only meaningful when smoothing is enabled (tensors exist).
    /// Format: `ix,iy,frac_x,frac_y,inv_eps_xx,inv_eps_xy,inv_eps_yx,inv_eps_yy`
    pub fn save_tensor_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;

        let tensors = match &self.inv_eps_tensors {
            Some(t) => t,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "No tensor data available (smoothing was disabled)",
                ));
            }
        };

        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        // Write header
        writeln!(
            writer,
            "ix,iy,frac_x,frac_y,inv_eps_xx,inv_eps_xy,inv_eps_yx,inv_eps_yy"
        )?;

        // Write data rows
        for iy in 0..self.grid.ny {
            for ix in 0..self.grid.nx {
                let idx = self.grid.idx(ix, iy);
                let frac_x = (ix as f64 + 0.5) / self.grid.nx as f64;
                let frac_y = (iy as f64 + 0.5) / self.grid.ny as f64;
                let t = &tensors[idx];

                writeln!(
                    writer,
                    "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
                    ix, iy, frac_x, frac_y, t[0], t[1], t[2], t[3]
                )?;
            }
        }

        writer.flush()
    }
}

fn sample_raw_eps(geom: &Geometry2D, grid: Grid2D) -> Vec<f64> {
    let mut eps_r = vec![geom.eps_bg; grid.len()];
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let frac = [
                fractional_position(ix, grid.nx),
                fractional_position(iy, grid.ny),
            ];
            let idx = grid.idx(ix, iy);
            eps_r[idx] = geom.relative_permittivity_at_fractional(frac);
        }
    }
    eps_r
}

/// Build smoothed dielectric using subgrid numerical integration.
///
/// This is the original method that uses a sub-grid mesh to sample the geometry
/// and computes approximate filling fractions and interface normals from the
/// dipole moment of the eps distribution.
fn build_smoothed_dielectric_subgrid(
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
            // The pixel origin is at i/n, and we integrate over the cell
            let frac_origin_x = fractional_cell_origin(ix, grid.nx);
            let frac_origin_y = fractional_cell_origin(iy, grid.ny);
            // Node-based: the pixel "center" for dipole calculation is at the node itself
            let center_frac = [frac_origin_x, frac_origin_y];
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

/// Build smoothed dielectric using analytic geometry calculations.
///
/// This method computes exact filling fractions and interface normals for
/// circles (BasisAtoms), providing more accurate results at material interfaces.
///
/// For each pixel, we:
/// 1. Check if any atom (circle) intersects the pixel
/// 2. Use exact circle-rectangle intersection formulas for filling fraction
/// 3. Compute the exact interface normal from the geometry
/// 4. Apply the MPB smoothing formula: ε̃⁻¹ = P⟨ε⁻¹⟩ + (1-P)⟨ε⟩⁻¹
fn build_smoothed_dielectric_analytic(
    geom: &Geometry2D,
    grid: Grid2D,
    smoothing: &SmoothingOptions,
) -> (Vec<f64>, Vec<f64>, Vec<[f64; 4]>) {
    let len = grid.len();
    let mut eps_avg = vec![0.0; len];
    let mut inv_avg = vec![0.0; len];
    let mut tensors = vec![[0.0; 4]; len];

    // Compute pixel size in Cartesian coordinates
    // For general lattices, we approximate using axis-aligned bounding
    let pixel_size_x = vector_norm(geom.lattice.a1) / grid.nx as f64;
    let pixel_size_y = vector_norm(geom.lattice.a2) / grid.ny as f64;
    let pixel_size = [pixel_size_x, pixel_size_y];

    let tol = smoothing.tolerance();

    // Pre-compute circles for all atoms (in Cartesian coordinates)
    // Handle periodic images by including the 9 nearest copies
    let mut circles: Vec<(Circle, f64)> = Vec::new(); // (circle, eps_inside)
    for atom in &geom.atoms {
        let radius_cart = atom.radius_cartesian(&geom.lattice);
        // Add the atom and its periodic images
        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                let frac_pos = [atom.pos[0] + di as f64, atom.pos[1] + dj as f64];
                let center_cart = geom.lattice.fractional_to_cartesian(frac_pos);
                circles.push((Circle::new(center_cart, radius_cart), atom.eps_inside));
            }
        }
    }

    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);

            // Compute pixel center in Cartesian coordinates (node-based: i/N)
            let frac_center = [
                fractional_position(ix, grid.nx),
                fractional_position(iy, grid.ny),
            ];
            let pixel_center = geom.lattice.fractional_to_cartesian(frac_center);

            // Check all circles for intersection with this pixel
            let mut best_intersection: Option<(f64, [f64; 2], f64)> = None; // (fill_frac, normal, eps_inside)
            let mut total_fill = 0.0;

            for (circle, eps_inside) in &circles {
                let intersection = circle.intersect_pixel(pixel_center, pixel_size);

                if intersection.filling_fraction > 1e-10 {
                    total_fill += intersection.filling_fraction;

                    // Keep track of the dominant intersection (largest filling fraction)
                    let dominated = best_intersection
                        .as_ref()
                        .map(|(f, _, _)| intersection.filling_fraction > *f)
                        .unwrap_or(true);

                    if dominated {
                        let normal = intersection.interface_normal.unwrap_or_else(|| {
                            // For fully-inside pixels, compute normal from center
                            circle.normal_at(pixel_center)
                        });
                        best_intersection =
                            Some((intersection.filling_fraction, normal, *eps_inside));
                    }
                }
            }

            // Clamp total fill to [0, 1] (might exceed 1 due to overlapping circles)
            total_fill = total_fill.clamp(0.0, 1.0);

            if let Some((fill_frac, normal, eps_inside)) = best_intersection {
                // Material interface: use analytic smoothing formula
                let fill_frac = fill_frac.min(total_fill);
                let (avg_eps, avg_inv, tensor) =
                    compute_smoothed_dielectric(fill_frac, eps_inside, geom.eps_bg, normal);

                eps_avg[idx] = avg_eps;
                inv_avg[idx] = avg_inv;

                // Check if this is actually at an interface (partial fill)
                let is_interface = fill_frac > tol && fill_frac < 1.0 - tol;
                if is_interface {
                    tensors[idx] = tensor;
                } else {
                    // Uniform region: use isotropic tensor
                    tensors[idx] = isotropic_tensor(avg_inv);
                }
            } else {
                // No circle intersection: use background permittivity
                eps_avg[idx] = geom.eps_bg;
                inv_avg[idx] = 1.0 / geom.eps_bg;
                tensors[idx] = isotropic_tensor(1.0 / geom.eps_bg);
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

/// Get the fractional position for a grid index.
///
/// Uses node-based sampling: position = index / count (MPB-compatible)
fn fractional_position(index: usize, count: usize) -> f64 {
    index as f64 / count as f64
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
