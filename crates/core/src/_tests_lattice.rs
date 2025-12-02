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
fn classify_distinguishes_basic_lattices() {
    let square = Lattice2D::square(1.0);
    assert_eq!(square.classify(), super::lattice::LatticeClass::Square);
    let rect = Lattice2D::rectangular(1.0, 2.0);
    assert_eq!(rect.classify(), super::lattice::LatticeClass::Rectangular);
    let tri = Lattice2D::hexagonal(1.0);
    assert_eq!(tri.classify(), super::lattice::LatticeClass::Triangular);
    let oblique = Lattice2D::oblique([1.0, 0.2], [0.3, 0.9]);
    assert_eq!(oblique.classify(), super::lattice::LatticeClass::Oblique);
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

#[test]
fn reciprocal_fractional_to_cartesian_square_lattice() {
    // Square lattice: reciprocal is also square
    // b1 = [2π, 0], b2 = [0, 2π]
    let lattice = Lattice2D::square(1.0);
    let recip = lattice.reciprocal();

    // M-point: (0.5, 0) → [π, 0]
    let m_point = recip.fractional_to_cartesian([0.5, 0.0]);
    assert!((m_point[0] - std::f64::consts::PI).abs() < 1e-12);
    assert!(m_point[1].abs() < 1e-12);

    // X-point: (0.5, 0.5) → [π, π]
    let x_point = recip.fractional_to_cartesian([0.5, 0.5]);
    assert!((x_point[0] - std::f64::consts::PI).abs() < 1e-12);
    assert!((x_point[1] - std::f64::consts::PI).abs() < 1e-12);
}

#[test]
fn reciprocal_fractional_to_cartesian_hexagonal_lattice() {
    // Hexagonal lattice (60° convention, as used by MPB):
    // a1 = [1, 0], a2 = [0.5, √3/2]
    // This gives: b1 = [2π, -2π/√3], b2 = [0, 4π/√3]
    let sqrt3 = 3.0_f64.sqrt();
    let lattice = Lattice2D::oblique([1.0, 0.0], [0.5, sqrt3 / 2.0]);
    let recip = lattice.reciprocal();

    // Verify reciprocal vectors
    assert!((recip.b1[0] - TAU).abs() < 1e-10, "b1[0] should be 2π");
    assert!(
        (recip.b1[1] + TAU / sqrt3).abs() < 1e-10,
        "b1[1] should be -2π/√3"
    );
    assert!(recip.b2[0].abs() < 1e-10, "b2[0] should be 0");
    assert!(
        (recip.b2[1] - 2.0 * TAU / sqrt3).abs() < 1e-10,
        "b2[1] should be 4π/√3"
    );

    // M-point: (0.5, 0) → [π, -π/√3] (NOT [π, 0]!)
    let m_point = recip.fractional_to_cartesian([0.5, 0.0]);
    assert!(
        (m_point[0] - std::f64::consts::PI).abs() < 1e-10,
        "M-point x should be π"
    );
    assert!(
        (m_point[1] + std::f64::consts::PI / sqrt3).abs() < 1e-10,
        "M-point y should be -π/√3"
    );

    // K-point: (1/3, 1/3) → [(2/3)π, (2/3)π/√3]
    // k = (1/3)*b1 + (1/3)*b2 = (1/3)*[2π, -2π/√3] + (1/3)*[0, 4π/√3]
    //   = [(2/3)π, -(2/3)π/√3 + (4/3)π/√3] = [(2/3)π, (2/3)π/√3]
    let k_point = recip.fractional_to_cartesian([1.0 / 3.0, 1.0 / 3.0]);
    assert!(
        (k_point[0] - 2.0 * std::f64::consts::PI / 3.0).abs() < 1e-10,
        "K-point x should be 2π/3"
    );
    assert!(
        (k_point[1] - 2.0 * std::f64::consts::PI / (3.0 * sqrt3)).abs() < 1e-10,
        "K-point y should be 2π/(3√3)"
    );
}
