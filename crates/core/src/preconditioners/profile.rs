//! Parametric spectral profile: the learnable part of the preconditioner.
//!
//! T' = W^½ T W^½ with W = F⁻¹ diag(exp τ(s)) F and s = |k+G|² / |b_min|², the
//! squared wavevector in units of the shortest reciprocal lattice vector. τ is
//! piecewise linear in ln s through `PROFILE_KNOTS` knots at s = 4^(j−2) and constant
//! outside them. Every profile keeps T' Hermitian positive semidefinite and commutes
//! with the symmetries of T; τ ≡ 0 reproduces T. Offline fitting lives in
//! `benchmarks/preconditioner/ml`.

pub const PROFILE_KNOTS: usize = 8;

/// Band count `TE_PROFILE` was fitted for. Below it the automatic profile follows the
/// lower band window: s is scaled by `TE_PROFILE_BANDS / bands` (measured on 4 to 40
/// bands; scaling up above it moved the damped low modes into the window and cost work).
pub const TE_PROFILE_BANDS: usize = 12;

/// Built-in profile of the TE transverse projection (continuous shift, tensor weight),
/// fitted to the FFT work of 16² band runs on 16 training geometries. Over the 384 TE
/// diagrams of the study matrix it needed 14% fewer FFTs in total than no profile and
/// more on no diagram.
pub const TE_PROFILE: [f64; PROFILE_KNOTS] = [-0.7859, -1.6483, -1.6596, 0.4915, 0.1951, 0.0596, 0.0, 0.0];

/// Knot position s_j = 4^(j−2).
pub fn profile_knot(j: usize) -> f64 {
    4f64.powi(j as i32 - 2)
}

/// τ(s) for knot values `tau`.
pub fn profile_value(s: f64, tau: &[f64; PROFILE_KNOTS]) -> f64 {
    let x = s.max(f64::MIN_POSITIVE).ln() / 4f64.ln() + 2.0;
    if x <= 0.0 {
        return tau[0];
    }
    if x >= (PROFILE_KNOTS - 1) as f64 {
        return tau[PROFILE_KNOTS - 1];
    }
    let j = x.floor() as usize;
    let t = x - j as f64;
    tau[j] * (1.0 - t) + tau[j + 1] * t
}

/// |b_min|², the squared length of the shortest nonzero vector of the lattice spanned
/// by b1 and b2, by Lagrange–Gauss reduction (so any basis of the lattice gives it).
pub fn shortest_reciprocal_sq(b1: [f64; 2], b2: [f64; 2]) -> f64 {
    let dot = |u: [f64; 2], v: [f64; 2]| u[0] * v[0] + u[1] * v[1];
    let (mut u, mut v) = if dot(b1, b1) <= dot(b2, b2) { (b1, b2) } else { (b2, b1) };
    for _ in 0..64 {
        let m = (dot(u, v) / dot(u, u)).round();
        v = [v[0] - m * u[0], v[1] - m * u[1]];
        if dot(v, v) >= dot(u, u) {
            break;
        }
        std::mem::swap(&mut u, &mut v);
    }
    dot(u, u)
}

/// exp(τ(s)/2) per Fourier mode, the factor W^½. The smoothers fold it into their
/// Fourier symbols: the diagonal gains W, each 1/|k+G|² factor of the transverse
/// projection gains W^½, so a profile adds no work per application.
pub fn profile_root(k_plus_g_sq: &[f64], b_min_sq: f64, tau: &[f64; PROFILE_KNOTS]) -> Vec<f64> {
    k_plus_g_sq.iter().map(|&q2| (0.5 * profile_value(q2 / b_min_sq, tau)).exp()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_interpolates_in_log_s_and_is_constant_outside() {
        let tau = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(profile_value(1e-9, &tau), 1.0);
        assert_eq!(profile_value(1e9, &tau), 8.0);
        assert!((profile_value(1.0, &tau) - 3.0).abs() < 1e-12);
        assert!((profile_value(2.0, &tau) - 3.5).abs() < 1e-12);
        assert!((shortest_reciprocal_sq([1.0, 0.0], [0.5, 3f64.sqrt() / 2.0]) - 1.0).abs() < 1e-12);
        // A non-reduced basis of the square lattice and a strongly oblique one.
        assert!((shortest_reciprocal_sq([2.0, 1.0], [5.0, 3.0]) - 1.0).abs() < 1e-12);
        let angle = 10f64.to_radians();
        let b2 = [3.0 * angle.cos() + 7.0, 3.0 * angle.sin()];
        let direct = shortest_reciprocal_sq([1.0, 0.0], [3.0 * angle.cos(), 3.0 * angle.sin()]);
        assert!((shortest_reciprocal_sq([1.0, 0.0], b2) - direct).abs() < 1e-12);
    }
}
