#![cfg(test)]

use super::lattice::Lattice2D;

const TAU: f64 = std::f64::consts::PI * 2.0;

#[test]
fn reciprocal_of_square_lattice_matches_expected() {
    let lattice = Lattice2D::square(1.0);
    let reciprocal = lattice.reciprocal();
    assert!((reciprocal.b1[0] - TAU).abs() < 1e-12);
    assert!(reciprocal.b1[1].abs() < 1e-12);
    assert!((reciprocal.b2[1] - TAU).abs() < 1e-12);
    assert!(reciprocal.b2[0].abs() < 1e-12);
}

#[test]
fn cartesian_fractional_roundtrip_is_identity() {
    let lattice = Lattice2D::hexagonal(2.0);
    let frac = [0.35, 0.4];
    let cart = lattice.fractional_to_cartesian(frac);
    let recovered = lattice.cartesian_to_fractional(cart);
    assert!((recovered[0] - frac[0]).abs() < 1e-12);
    assert!((recovered[1] - frac[1]).abs() < 1e-12);
}

#[test]
fn reciprocal_vectors_form_dual_basis() {
    let lattice = Lattice2D::oblique([2.0, 0.5], [0.25, 1.5]);
    let reciprocal = lattice.reciprocal();
    let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
    assert!((dot(lattice.a1, reciprocal.b1) - TAU).abs() < 1e-12);
    assert!(dot(lattice.a1, reciprocal.b2).abs() < 1e-12);
    assert!(dot(lattice.a2, reciprocal.b1).abs() < 1e-12);
    assert!((dot(lattice.a2, reciprocal.b2) - TAU).abs() < 1e-12);
}

#[test]
fn characteristic_length_matches_a1_norm() {
    let lattice = Lattice2D::oblique([3.0, 4.0], [1.0, 0.0]);
    assert!((lattice.characteristic_length() - 5.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "primitive vectors are linearly dependent")]
fn reciprocal_panics_for_linearly_dependent_vectors() {
    let lattice = Lattice2D::oblique([1.0, 0.0], [2.0, 0.0]);
    let _ = lattice.reciprocal();
}

#[test]
#[should_panic(expected = "primitive vectors are linearly dependent")]
fn cartesian_to_fractional_panics_for_singular_lattice() {
    let lattice = Lattice2D::oblique([0.0, 0.0], [2.0, 0.0]);
    let _ = lattice.cartesian_to_fractional([0.0, 0.0]);
}
