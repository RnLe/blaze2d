use mpb2d_core::grid::Grid2D;
use mpb2d_core::lattice::{Lattice2D, LatticeClass};
use std::f64::consts::TAU;

pub const K_PLUS_G_NEAR_ZERO_FLOOR: f64 = 1e-9;

pub fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub fn determinant(a1: [f64; 2], a2: [f64; 2]) -> f64 {
    a1[0] * a2[1] - a1[1] * a2[0]
}

pub fn lattice_cell_area(lattice: &Lattice2D) -> f64 {
    determinant(lattice.a1, lattice.a2).abs()
}

pub fn classify_to_string(class: LatticeClass) -> String {
    match class {
        LatticeClass::Square => "Square",
        LatticeClass::Rectangular => "Rectangular",
        LatticeClass::Triangular => "Triangular",
        LatticeClass::Oblique => "Oblique",
    }
    .to_string()
}

pub fn reciprocal_component(index: usize, len: usize, length: f64) -> f64 {
    if len == 0 || length == 0.0 {
        return 0.0;
    }
    let len_i = len as isize;
    let mut k = index as isize;
    if k > len_i / 2 {
        k -= len_i;
    }
    TAU * k as f64 / length
}

pub fn clamp_gradient_components(kx: f64, ky: f64) -> (f64, f64) {
    if !kx.is_finite() || !ky.is_finite() {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        return (magnitude, 0.0);
    }

    let norm_sq = kx * kx + ky * ky;
    if norm_sq >= K_PLUS_G_NEAR_ZERO_FLOOR {
        (kx, ky)
    } else if norm_sq == 0.0 {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        (magnitude, 0.0)
    } else {
        let scale = (K_PLUS_G_NEAR_ZERO_FLOOR / norm_sq).sqrt();
        (kx * scale, ky * scale)
    }
}

pub fn build_k_plus_g_tables(
    grid: Grid2D,
    bloch: [f64; 2],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let len = grid.len();
    let mut k_plus_g_x = vec![0.0; len];
    let mut k_plus_g_y = vec![0.0; len];
    let mut k_plus_g_sq = vec![0.0; len];
    let mut clamp_mask = vec![false; len];
    for iy in 0..grid.ny {
        let ky_shifted = reciprocal_component(iy, grid.ny, grid.ly) + bloch[1];
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let kx_shifted = reciprocal_component(ix, grid.nx, grid.lx) + bloch[0];
            let raw_sq = kx_shifted * kx_shifted + ky_shifted * ky_shifted;
            let (clamped_kx, clamped_ky) = clamp_gradient_components(kx_shifted, ky_shifted);
            k_plus_g_x[idx] = clamped_kx;
            k_plus_g_y[idx] = clamped_ky;
            k_plus_g_sq[idx] = clamped_kx * clamped_kx + clamped_ky * clamped_ky;
            if raw_sq <= K_PLUS_G_NEAR_ZERO_FLOOR {
                clamp_mask[idx] = true;
            }
        }
    }
    (k_plus_g_x, k_plus_g_y, k_plus_g_sq, clamp_mask)
}
