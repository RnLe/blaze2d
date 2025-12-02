//! Comprehensive tests for the symmetry projector module.
//!
//! These tests cover:
//! - Mirror partner table construction and correctness
//! - Even/odd parity projections
//! - Self-partner (on-axis) handling
//! - Multiple reflection constraints
//! - K-point detection and little group logic
//! - Edge cases: 1D grids, odd dimensions, Nyquist frequencies
//! - Path generation utilities

#![cfg(test)]

use num_complex::Complex64;

use crate::{
    field::Field2D,
    grid::Grid2D,
    lattice::{Lattice2D, LatticeClass},
    symmetry::{
        MirrorPartnerTable, Parity, PathType, ReflectionAxis, ReflectionConstraint, SymmetryConfig,
        SymmetryProjector, is_high_symmetry_point, is_on_symmetry_line, standard_path,
    },
};

// ============================================================================
// Helper Functions
// ============================================================================

fn assert_complex_eq(actual: Complex64, expected: Complex64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).norm() < tol,
        "{}: expected {:.6} + {:.6}i, got {:.6} + {:.6}i",
        msg,
        expected.re,
        expected.im,
        actual.re,
        actual.im
    );
}

fn assert_complex_zero(actual: Complex64, tol: f64, msg: &str) {
    assert!(
        actual.norm() < tol,
        "{}: expected ~0, got {:.6} + {:.6}i (norm={:.2e})",
        msg,
        actual.re,
        actual.im,
        actual.norm()
    );
}

fn assert_point_close(actual: [f64; 2], expected: [f64; 2]) {
    let tol = 1e-9;
    assert!(
        (actual[0] - expected[0]).abs() < tol && (actual[1] - expected[1]).abs() < tol,
        "expected {:?} to be close to {:?}",
        actual,
        expected
    );
}

/// Create a simple projector with a single reflection constraint.
fn single_reflection_projector(
    grid: Grid2D,
    axis: ReflectionAxis,
    parity: Parity,
) -> SymmetryProjector {
    SymmetryProjector::new(grid, vec![ReflectionConstraint { axis, parity }])
        .expect("projector should be created")
}

#[allow(dead_code)]
/// Create a field where each coefficient is its linear index (for easy debugging).
fn indexed_field(grid: Grid2D) -> Field2D {
    let mut field = Field2D::zeros(grid);
    for idx in 0..grid.len() {
        field.as_mut_slice()[idx] = Complex64::new(idx as f64, 0.0);
    }
    field
}

/// Create a field with distinct complex values for each index.
fn complex_indexed_field(grid: Grid2D) -> Field2D {
    let mut field = Field2D::zeros(grid);
    for idx in 0..grid.len() {
        // Use different real and imaginary parts for each index
        field.as_mut_slice()[idx] = Complex64::new(idx as f64, (idx as f64) * 0.1 + 1.0);
    }
    field
}

// ============================================================================
// Mirror Partner Table Tests
// ============================================================================

mod mirror_partner_table {
    use super::*;

    #[test]
    fn square_grid_y_mirror_partners() {
        // 4x4 grid: test y → -y mirror (ReflectionAxis::X)
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // iy=0 → iy=0 (self-partner, G_y = 0)
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 0)), grid.idx(0, 0));
        assert_eq!(table.partner_under_y_mirror(grid.idx(1, 0)), grid.idx(1, 0));
        assert_eq!(table.partner_under_y_mirror(grid.idx(2, 0)), grid.idx(2, 0));
        assert_eq!(table.partner_under_y_mirror(grid.idx(3, 0)), grid.idx(3, 0));

        // iy=1 → iy=3 (G_y = 1 → G_y = -1 = 3 mod 4)
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 1)), grid.idx(0, 3));
        assert_eq!(table.partner_under_y_mirror(grid.idx(2, 1)), grid.idx(2, 3));

        // iy=2 → iy=2 (self-partner at Nyquist: G_y = ±N/2)
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 2)), grid.idx(0, 2));
        assert_eq!(table.partner_under_y_mirror(grid.idx(1, 2)), grid.idx(1, 2));

        // iy=3 → iy=1
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 3)), grid.idx(0, 1));
        assert_eq!(table.partner_under_y_mirror(grid.idx(3, 3)), grid.idx(3, 1));
    }

    #[test]
    fn square_grid_x_mirror_partners() {
        // 4x4 grid: test x → -x mirror (ReflectionAxis::Y)
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // ix=0 → ix=0 (self-partner, G_x = 0)
        assert_eq!(table.partner_under_x_mirror(grid.idx(0, 0)), grid.idx(0, 0));
        assert_eq!(table.partner_under_x_mirror(grid.idx(0, 1)), grid.idx(0, 1));

        // ix=1 → ix=3
        assert_eq!(table.partner_under_x_mirror(grid.idx(1, 0)), grid.idx(3, 0));
        assert_eq!(table.partner_under_x_mirror(grid.idx(1, 2)), grid.idx(3, 2));

        // ix=2 → ix=2 (Nyquist)
        assert_eq!(table.partner_under_x_mirror(grid.idx(2, 0)), grid.idx(2, 0));

        // ix=3 → ix=1
        assert_eq!(table.partner_under_x_mirror(grid.idx(3, 0)), grid.idx(1, 0));
    }

    #[test]
    fn rectangular_grid_partners() {
        // Non-square grid: 6x4
        let grid = Grid2D::new(6, 4, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // Y-mirror in 4-sized dimension
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 1)), grid.idx(0, 3));
        assert_eq!(table.partner_under_y_mirror(grid.idx(5, 1)), grid.idx(5, 3));

        // X-mirror in 6-sized dimension
        assert_eq!(table.partner_under_x_mirror(grid.idx(1, 0)), grid.idx(5, 0));
        assert_eq!(table.partner_under_x_mirror(grid.idx(2, 2)), grid.idx(4, 2));
        assert_eq!(table.partner_under_x_mirror(grid.idx(3, 1)), grid.idx(3, 1)); // Nyquist
    }

    #[test]
    fn odd_dimension_grid_partners() {
        // 5x5 grid: odd dimensions have no Nyquist frequency
        let grid = Grid2D::new(5, 5, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // iy=0 is self-partner
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 0)), grid.idx(0, 0));

        // iy=1 → iy=4, iy=2 → iy=3
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 1)), grid.idx(0, 4));
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 2)), grid.idx(0, 3));

        // No Nyquist self-partners for odd N (except 0)
        assert_ne!(table.partner_under_y_mirror(grid.idx(0, 2)), grid.idx(0, 2));
    }

    #[test]
    fn tiny_1x1_grid() {
        // Degenerate 1x1 grid: only one element
        let grid = Grid2D::new(1, 1, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // The single element is its own partner under both mirrors
        assert_eq!(table.partner_under_y_mirror(0), 0);
        assert_eq!(table.partner_under_x_mirror(0), 0);
        assert!(table.is_on_y_axis(0));
        assert!(table.is_on_x_axis(0));
    }

    #[test]
    fn tiny_2x2_grid() {
        // 2x2 grid: all elements are either DC or Nyquist
        let grid = Grid2D::new(2, 2, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // DC corner (0,0)
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 0)), grid.idx(0, 0));
        assert_eq!(table.partner_under_x_mirror(grid.idx(0, 0)), grid.idx(0, 0));

        // Nyquist corners: (1,0), (0,1), (1,1) are all self-partners
        assert_eq!(table.partner_under_x_mirror(grid.idx(1, 0)), grid.idx(1, 0));
        assert_eq!(table.partner_under_y_mirror(grid.idx(0, 1)), grid.idx(0, 1));
        assert_eq!(table.partner_under_x_mirror(grid.idx(1, 1)), grid.idx(1, 1));
        assert_eq!(table.partner_under_y_mirror(grid.idx(1, 1)), grid.idx(1, 1));
    }

    #[test]
    fn axis_detection() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        // On y=0 axis (G_y = 0): iy = 0
        assert!(table.is_on_y_axis(grid.idx(0, 0)));
        assert!(table.is_on_y_axis(grid.idx(3, 0)));
        assert!(!table.is_on_y_axis(grid.idx(0, 1)));

        // Also on y-axis: Nyquist (iy = ny/2)
        assert!(table.is_on_y_axis(grid.idx(0, 2)));

        // On x=0 axis (G_x = 0): ix = 0
        assert!(table.is_on_x_axis(grid.idx(0, 0)));
        assert!(table.is_on_x_axis(grid.idx(0, 3)));
        assert!(!table.is_on_x_axis(grid.idx(1, 0)));

        // Also on x-axis: Nyquist (ix = nx/2)
        assert!(table.is_on_x_axis(grid.idx(2, 0)));
    }

    #[test]
    fn involution_property() {
        // Mirror applied twice should return to original
        let grid = Grid2D::new(8, 6, 1.0, 1.0);
        let table = MirrorPartnerTable::new(grid);

        for idx in 0..grid.len() {
            let partner_y = table.partner_under_y_mirror(idx);
            let double_y = table.partner_under_y_mirror(partner_y);
            assert_eq!(
                double_y, idx,
                "y-mirror should be involution at idx={}",
                idx
            );

            let partner_x = table.partner_under_x_mirror(idx);
            let double_x = table.partner_under_x_mirror(partner_x);
            assert_eq!(
                double_x, idx,
                "x-mirror should be involution at idx={}",
                idx
            );
        }
    }
}

// ============================================================================
// Even Parity Projection Tests
// ============================================================================

mod even_parity {
    use super::*;

    #[test]
    fn basic_even_y_projection() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Set (0,1) = 1+2i, (0,3) = 3+4i (they are partners under y-mirror)
        data[grid.idx(0, 1)] = Complex64::new(1.0, 2.0);
        data[grid.idx(0, 3)] = Complex64::new(3.0, 4.0);

        projector.apply(&mut field);

        // After even projection: both = (1+3)/2 + i(2+4)/2 = 2+3i
        let expected = Complex64::new(2.0, 3.0);
        assert_complex_eq(field.as_slice()[grid.idx(0, 1)], expected, 1e-12, "(0,1)");
        assert_complex_eq(field.as_slice()[grid.idx(0, 3)], expected, 1e-12, "(0,3)");
    }

    #[test]
    fn even_projection_preserves_symmetric_field() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Create an already-symmetric field
        data[grid.idx(1, 1)] = Complex64::new(5.0, 3.0);
        data[grid.idx(1, 3)] = Complex64::new(5.0, 3.0); // Same as partner

        let original = data[grid.idx(1, 1)];
        projector.apply(&mut field);

        // Should be unchanged
        assert_complex_eq(
            field.as_slice()[grid.idx(1, 1)],
            original,
            1e-12,
            "symmetric preserved",
        );
        assert_complex_eq(
            field.as_slice()[grid.idx(1, 3)],
            original,
            1e-12,
            "partner preserved",
        );
    }

    #[test]
    fn even_projection_self_partner_preserved() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Self-partner at iy=0 (G_y = 0 axis)
        data[grid.idx(2, 0)] = Complex64::new(7.0, -2.0);
        let original = data[grid.idx(2, 0)];

        projector.apply(&mut field);

        // Even projection preserves self-partners
        assert_complex_eq(
            field.as_slice()[grid.idx(2, 0)],
            original,
            1e-12,
            "self-partner at iy=0",
        );
    }

    #[test]
    fn even_projection_nyquist_self_partner() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Nyquist frequency iy=2 is also a self-partner
        data[grid.idx(1, 2)] = Complex64::new(3.5, 1.2);
        let original = data[grid.idx(1, 2)];

        projector.apply(&mut field);

        assert_complex_eq(
            field.as_slice()[grid.idx(1, 2)],
            original,
            1e-12,
            "Nyquist self-partner",
        );
    }

    #[test]
    fn even_x_projection() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::Y, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Partners under x-mirror: (1,0) ↔ (3,0)
        data[grid.idx(1, 0)] = Complex64::new(10.0, 0.0);
        data[grid.idx(3, 0)] = Complex64::new(20.0, 0.0);

        projector.apply(&mut field);

        let expected = Complex64::new(15.0, 0.0);
        assert_complex_eq(field.as_slice()[grid.idx(1, 0)], expected, 1e-12, "(1,0)");
        assert_complex_eq(field.as_slice()[grid.idx(3, 0)], expected, 1e-12, "(3,0)");
    }

    #[test]
    fn even_projection_idempotent() {
        let grid = Grid2D::new(6, 6, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = complex_indexed_field(grid);

        projector.apply(&mut field);
        let after_first: Vec<Complex64> = field.as_slice().to_vec();

        projector.apply(&mut field);

        // Second application should not change the field
        for (idx, &val) in field.as_slice().iter().enumerate() {
            assert_complex_eq(
                val,
                after_first[idx],
                1e-12,
                &format!("idempotent at {}", idx),
            );
        }
    }
}

// ============================================================================
// Odd Parity Projection Tests
// ============================================================================

mod odd_parity {
    use super::*;

    #[test]
    fn basic_odd_y_projection() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Set (0,1) = 1+2i, (0,3) = 3+4i
        data[grid.idx(0, 1)] = Complex64::new(1.0, 2.0);
        data[grid.idx(0, 3)] = Complex64::new(3.0, 4.0);

        projector.apply(&mut field);

        // Odd: c_i → (c_i - c_j)/2, c_j → -(c_i - c_j)/2
        // (0,1) → (1-3)/2 + i(2-4)/2 = -1 - i
        // (0,3) → -(-1 - i) = 1 + i
        assert_complex_eq(
            field.as_slice()[grid.idx(0, 1)],
            Complex64::new(-1.0, -1.0),
            1e-12,
            "(0,1)",
        );
        assert_complex_eq(
            field.as_slice()[grid.idx(0, 3)],
            Complex64::new(1.0, 1.0),
            1e-12,
            "(0,3)",
        );
    }

    #[test]
    fn odd_projection_self_partner_zeroed() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Self-partner at iy=0: must be zero for odd parity
        data[grid.idx(1, 0)] = Complex64::new(5.0, 3.0);
        data[grid.idx(2, 0)] = Complex64::new(7.0, -1.0);

        projector.apply(&mut field);

        assert_complex_zero(
            field.as_slice()[grid.idx(1, 0)],
            1e-12,
            "self-partner (1,0)",
        );
        assert_complex_zero(
            field.as_slice()[grid.idx(2, 0)],
            1e-12,
            "self-partner (2,0)",
        );
    }

    #[test]
    fn odd_projection_nyquist_zeroed() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Nyquist (iy=2) is a self-partner, must be zero for odd
        data[grid.idx(0, 2)] = Complex64::new(100.0, 200.0);

        projector.apply(&mut field);

        assert_complex_zero(
            field.as_slice()[grid.idx(0, 2)],
            1e-12,
            "Nyquist self-partner",
        );
    }

    #[test]
    fn odd_projection_antisymmetry() {
        let grid = Grid2D::new(6, 6, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);
        let table = MirrorPartnerTable::new(grid);

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        // Check that c_i = -c_j for all pairs
        for idx in 0..grid.len() {
            let partner = table.partner_under_y_mirror(idx);
            if idx != partner {
                let ci = field.as_slice()[idx];
                let cj = field.as_slice()[partner];
                assert_complex_eq(
                    ci,
                    -cj,
                    1e-12,
                    &format!("antisymmetry at {} ↔ {}", idx, partner),
                );
            }
        }
    }

    #[test]
    fn odd_projection_idempotent() {
        let grid = Grid2D::new(6, 6, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);

        let mut field = complex_indexed_field(grid);

        projector.apply(&mut field);
        let after_first: Vec<Complex64> = field.as_slice().to_vec();

        projector.apply(&mut field);

        for (idx, &val) in field.as_slice().iter().enumerate() {
            assert_complex_eq(
                val,
                after_first[idx],
                1e-12,
                &format!("idempotent at {}", idx),
            );
        }
    }

    #[test]
    fn odd_projection_preserves_antisymmetric_field() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Create already-antisymmetric field
        data[grid.idx(1, 1)] = Complex64::new(5.0, 3.0);
        data[grid.idx(1, 3)] = Complex64::new(-5.0, -3.0); // -partner

        let original = data[grid.idx(1, 1)];
        projector.apply(&mut field);

        assert_complex_eq(
            field.as_slice()[grid.idx(1, 1)],
            original,
            1e-12,
            "antisym preserved",
        );
        assert_complex_eq(
            field.as_slice()[grid.idx(1, 3)],
            -original,
            1e-12,
            "partner preserved",
        );
    }
}

// ============================================================================
// Multiple Reflections Tests
// ============================================================================

mod multiple_reflections {
    use super::*;

    #[test]
    fn both_mirrors_even_parity() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = SymmetryProjector::new(
            grid,
            vec![
                ReflectionConstraint {
                    axis: ReflectionAxis::X,
                    parity: Parity::Even,
                },
                ReflectionConstraint {
                    axis: ReflectionAxis::Y,
                    parity: Parity::Even,
                },
            ],
        )
        .unwrap();

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        // After both even projections, field should be symmetric under both mirrors
        let table = MirrorPartnerTable::new(grid);

        for idx in 0..grid.len() {
            let y_partner = table.partner_under_y_mirror(idx);
            let x_partner = table.partner_under_x_mirror(idx);

            let ci = field.as_slice()[idx];
            let cy = field.as_slice()[y_partner];
            let cx = field.as_slice()[x_partner];

            assert_complex_eq(ci, cy, 1e-12, &format!("y-symmetry at {}", idx));
            assert_complex_eq(ci, cx, 1e-12, &format!("x-symmetry at {}", idx));
        }
    }

    #[test]
    fn both_mirrors_odd_parity() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = SymmetryProjector::new(
            grid,
            vec![
                ReflectionConstraint {
                    axis: ReflectionAxis::X,
                    parity: Parity::Odd,
                },
                ReflectionConstraint {
                    axis: ReflectionAxis::Y,
                    parity: Parity::Odd,
                },
            ],
        )
        .unwrap();

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        // All axis points (G_x=0 or G_y=0) should be zero
        let table = MirrorPartnerTable::new(grid);

        for idx in 0..grid.len() {
            if table.is_on_x_axis(idx) || table.is_on_y_axis(idx) {
                assert_complex_zero(field.as_slice()[idx], 1e-12, &format!("axis point {}", idx));
            }
        }
    }

    #[test]
    fn mixed_parity_x_even_y_odd() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = SymmetryProjector::new(
            grid,
            vec![
                ReflectionConstraint {
                    axis: ReflectionAxis::X,
                    parity: Parity::Odd,
                }, // y → -y: odd
                ReflectionConstraint {
                    axis: ReflectionAxis::Y,
                    parity: Parity::Even,
                }, // x → -x: even
            ],
        )
        .unwrap();

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        let table = MirrorPartnerTable::new(grid);

        for idx in 0..grid.len() {
            let y_partner = table.partner_under_y_mirror(idx);
            let x_partner = table.partner_under_x_mirror(idx);

            let ci = field.as_slice()[idx];

            // X-mirror (y → -y) is odd: c_i = -c_{y_partner}
            if idx != y_partner {
                let cy = field.as_slice()[y_partner];
                assert_complex_eq(ci, -cy, 1e-12, &format!("y-antisym at {}", idx));
            }

            // Y-mirror (x → -x) is even: c_i = c_{x_partner}
            let cx = field.as_slice()[x_partner];
            assert_complex_eq(ci, cx, 1e-12, &format!("x-sym at {}", idx));
        }
    }
}

// ============================================================================
// K-Point Detection Tests
// ============================================================================

mod k_point_detection {
    use super::*;

    #[test]
    fn gamma_point_detection() {
        let tol = 1e-6;

        assert!(is_high_symmetry_point([0.0, 0.0], tol));
        assert!(is_high_symmetry_point([1e-8, -1e-9], tol));
        assert!(!is_high_symmetry_point([0.1, 0.0], tol));
    }

    #[test]
    fn x_point_detection() {
        let tol = 1e-6;

        assert!(is_high_symmetry_point([0.5, 0.0], tol));
        assert!(is_high_symmetry_point([0.5 - 1e-8, 1e-9], tol));
        assert!(!is_high_symmetry_point([0.5, 0.1], tol));
    }

    #[test]
    fn m_point_detection() {
        let tol = 1e-6;

        assert!(is_high_symmetry_point([0.5, 0.5], tol));
        assert!(is_high_symmetry_point([0.5 + 1e-8, 0.5 - 1e-9], tol));
        assert!(!is_high_symmetry_point([0.5, 0.4], tol));
    }

    #[test]
    fn symmetry_line_detection() {
        let tol = 1e-6;

        // Γ-X line (k_y = 0)
        assert!(is_on_symmetry_line([0.3, 0.0], tol));
        assert!(is_on_symmetry_line([0.0, 0.0], tol));

        // Γ-Y line (k_x = 0)
        assert!(is_on_symmetry_line([0.0, 0.3], tol));

        // Γ-M diagonal (k_x = k_y)
        assert!(is_on_symmetry_line([0.3, 0.3], tol));
        assert!(is_on_symmetry_line([0.25, 0.25], tol));

        // Generic point
        assert!(!is_on_symmetry_line([0.3, 0.2], tol));
        assert!(!is_on_symmetry_line([0.1, 0.4], tol));
    }

    #[test]
    fn projector_for_gamma_point() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let config = SymmetryConfig::even();

        let proj = SymmetryProjector::for_k_point(grid, [0.0, 0.0], &config, LatticeClass::Square);

        assert!(proj.is_some(), "Γ-point should have projector");
        let proj = proj.unwrap();
        assert_eq!(proj.len(), 2, "Γ-point should have 2 reflections");
    }

    #[test]
    fn projector_for_x_point() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let config = SymmetryConfig::even();

        let proj = SymmetryProjector::for_k_point(grid, [0.5, 0.0], &config, LatticeClass::Square);

        assert!(proj.is_some(), "X-point should have projector");
        let proj = proj.unwrap();
        assert_eq!(
            proj.len(),
            1,
            "X-point should have 1 reflection (y-mirror only)"
        );
        assert_eq!(proj.reflections()[0].axis, ReflectionAxis::X);
    }

    #[test]
    fn projector_for_m_point_no_mirror() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let config = SymmetryConfig::even();

        let proj = SymmetryProjector::for_k_point(grid, [0.5, 0.5], &config, LatticeClass::Square);

        // M point has neither k_x=0 nor k_y=0, so no simple mirrors apply
        assert!(
            proj.is_none(),
            "M-point should have no simple mirror projector"
        );
    }

    #[test]
    fn projector_for_generic_k_point() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let config = SymmetryConfig::even();

        let proj = SymmetryProjector::for_k_point(grid, [0.3, 0.2], &config, LatticeClass::Square);

        assert!(proj.is_none(), "generic k-point should have no projector");
    }

    #[test]
    fn projector_disabled_config() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let config = SymmetryConfig::disabled();

        let proj = SymmetryProjector::for_k_point(grid, [0.0, 0.0], &config, LatticeClass::Square);

        assert!(proj.is_none(), "disabled config should return no projector");
    }

    #[test]
    fn projector_oblique_lattice() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let config = SymmetryConfig::even();

        let proj = SymmetryProjector::for_k_point(grid, [0.0, 0.0], &config, LatticeClass::Oblique);

        assert!(
            proj.is_none(),
            "oblique lattice should have no mirror symmetry"
        );
    }
}

// ============================================================================
// Path Generation Tests
// ============================================================================

mod path_generation {
    use super::*;

    #[test]
    fn square_path_endpoints() {
        let lattice = Lattice2D::square(1.0);
        let path = standard_path(&lattice, PathType::Square, 3);

        assert_point_close(path[0], [0.0, 0.0]); // Γ
        assert_point_close(path[3], [0.5, 0.0]); // X
        assert_point_close(path[6], [0.5, 0.5]); // M
        assert_point_close(*path.last().unwrap(), [0.0, 0.0]); // Γ
    }

    #[test]
    fn square_path_segment_count() {
        let lattice = Lattice2D::square(1.0);
        let path = standard_path(&lattice, PathType::Square, 5);

        // 3 legs × 5 segments + 1 starting point = 16 points
        // But first point of each leg after first is shared, so: 1 + 5 + 5 + 5 = 16
        // Actually the densify_path adds segments points after each node
        assert_eq!(path.len(), 16, "5 segments per leg should give 16 points");
    }

    #[test]
    fn hexagonal_path_endpoints() {
        let lattice = Lattice2D::hexagonal(1.0);
        let path = standard_path(&lattice, PathType::Hexagonal, 3);

        assert_point_close(path[0], [0.0, 0.0]); // Γ
        assert_point_close(path[3], [0.5, 0.0]); // M
        assert_point_close(path[6], [1.0 / 3.0, 1.0 / 3.0]); // K
        assert_point_close(*path.last().unwrap(), [0.0, 0.0]); // Γ
    }

    #[test]
    fn custom_path_passthrough() {
        let lattice = Lattice2D::square(1.0);
        let custom = vec![[0.1, 0.2], [0.3, 0.4], [0.9, 0.1]];
        let path = standard_path(&lattice, PathType::Custom(custom.clone()), 10);

        assert_eq!(path, custom, "custom path should pass through unchanged");
    }

    #[test]
    fn minimum_segments_clamped() {
        let lattice = Lattice2D::square(1.0);
        let path = standard_path(&lattice, PathType::Square, 0);

        // Should clamp to at least 1 segment per leg
        assert!(path.len() >= 4, "should have at least the corner points");
    }

    #[test]
    fn single_segment_path() {
        let lattice = Lattice2D::square(1.0);
        let path = standard_path(&lattice, PathType::Square, 1);

        // 1 segment per leg = just the corner points
        assert_eq!(path.len(), 4);
        assert_point_close(path[0], [0.0, 0.0]);
        assert_point_close(path[1], [0.5, 0.0]);
        assert_point_close(path[2], [0.5, 0.5]);
        assert_point_close(path[3], [0.0, 0.0]);
    }
}

// ============================================================================
// Edge Cases and Numerical Stability
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn projection_of_zero_field() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        // Field is already zero

        projector.apply(&mut field);

        // Should remain zero
        for &val in field.as_slice() {
            assert_complex_zero(val, 1e-15, "zero field");
        }
    }

    #[test]
    fn projection_of_large_values() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Large values
        data[grid.idx(0, 1)] = Complex64::new(1e15, 1e15);
        data[grid.idx(0, 3)] = Complex64::new(1e15, 1e15);

        projector.apply(&mut field);

        // Should be preserved (already symmetric)
        assert_complex_eq(
            field.as_slice()[grid.idx(0, 1)],
            Complex64::new(1e15, 1e15),
            1e3, // Relative tolerance
            "large values",
        );
    }

    #[test]
    fn projection_of_tiny_values() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Tiny values
        data[grid.idx(0, 1)] = Complex64::new(1e-15, 1e-15);
        data[grid.idx(0, 3)] = Complex64::new(3e-15, 3e-15);

        projector.apply(&mut field);

        let expected = Complex64::new(2e-15, 2e-15);
        assert_complex_eq(
            field.as_slice()[grid.idx(0, 1)],
            expected,
            1e-16,
            "tiny values",
        );
    }

    #[test]
    fn projection_with_pure_imaginary() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Odd);

        let mut field = Field2D::zeros(grid);
        let data = field.as_mut_slice();

        // Pure imaginary values
        data[grid.idx(0, 1)] = Complex64::new(0.0, 5.0);
        data[grid.idx(0, 3)] = Complex64::new(0.0, 3.0);

        projector.apply(&mut field);

        // Odd: (5i - 3i)/2 = i, partner = -i
        assert_complex_eq(
            field.as_slice()[grid.idx(0, 1)],
            Complex64::new(0.0, 1.0),
            1e-12,
            "pure imag",
        );
        assert_complex_eq(
            field.as_slice()[grid.idx(0, 3)],
            Complex64::new(0.0, -1.0),
            1e-12,
            "partner",
        );
    }

    #[test]
    fn empty_projector_returns_none() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);

        let proj = SymmetryProjector::new(grid, vec![]);
        assert!(proj.is_none(), "empty reflections should return None");
    }

    #[test]
    fn very_narrow_grid_1xn() {
        let grid = Grid2D::new(1, 8, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        // Should still work correctly
        let table = MirrorPartnerTable::new(grid);
        for idx in 0..grid.len() {
            let partner = table.partner_under_y_mirror(idx);
            let ci = field.as_slice()[idx];
            let cj = field.as_slice()[partner];
            assert_complex_eq(ci, cj, 1e-12, &format!("1xN symmetry at {}", idx));
        }
    }

    #[test]
    fn very_narrow_grid_nx1() {
        let grid = Grid2D::new(8, 1, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::Y, Parity::Odd);

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        // For x-mirror with odd parity, all points on y=0 line are self-partners
        // and should be zeroed
        let table = MirrorPartnerTable::new(grid);
        for idx in 0..grid.len() {
            if table.is_on_x_axis(idx) {
                assert_complex_zero(
                    field.as_slice()[idx],
                    1e-12,
                    &format!("Nx1 zeroed at {}", idx),
                );
            }
        }
    }

    #[test]
    fn large_grid_performance() {
        // Just verify it doesn't crash or take too long for larger grids
        let grid = Grid2D::new(64, 64, 1.0, 1.0);
        let projector = SymmetryProjector::new(
            grid,
            vec![
                ReflectionConstraint {
                    axis: ReflectionAxis::X,
                    parity: Parity::Even,
                },
                ReflectionConstraint {
                    axis: ReflectionAxis::Y,
                    parity: Parity::Even,
                },
            ],
        )
        .unwrap();

        let mut field = complex_indexed_field(grid);
        projector.apply(&mut field);

        // Just check a few spots
        let table = MirrorPartnerTable::new(grid);
        let test_idx = grid.idx(10, 15);
        let partner = table.partner_under_y_mirror(test_idx);
        assert_complex_eq(
            field.as_slice()[test_idx],
            field.as_slice()[partner],
            1e-12,
            "large grid symmetry",
        );
    }
}

// ============================================================================
// SymmetryConfig Tests
// ============================================================================

mod config {
    use super::*;

    #[test]
    fn default_config() {
        let config = SymmetryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.parity, Parity::Even);
        assert!(config.reflections.is_empty());
    }

    #[test]
    fn disabled_config() {
        let config = SymmetryConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn even_config() {
        let config = SymmetryConfig::even();
        assert!(config.enabled);
        assert_eq!(config.parity, Parity::Even);
    }

    #[test]
    fn odd_config() {
        let config = SymmetryConfig::odd();
        assert!(config.enabled);
        assert_eq!(config.parity, Parity::Odd);
    }

    #[test]
    fn builder_pattern() {
        let config = SymmetryConfig::default()
            .with_parity(Parity::Odd)
            .without_symmetry()
            .with_symmetry();

        assert!(config.enabled);
        assert_eq!(config.parity, Parity::Odd);
    }
}

// ============================================================================
// Block Operations Tests
// ============================================================================

mod block_operations {
    use super::*;

    #[test]
    fn apply_block_to_multiple_fields() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut fields: Vec<Field2D> = (0..3)
            .map(|i| {
                let mut f = Field2D::zeros(grid);
                for idx in 0..grid.len() {
                    f.as_mut_slice()[idx] = Complex64::new((i * grid.len() + idx) as f64, 0.0);
                }
                f
            })
            .collect();

        projector.apply_block(&mut fields);

        // Each field should now be symmetric
        let table = MirrorPartnerTable::new(grid);
        for (fi, field) in fields.iter().enumerate() {
            for idx in 0..grid.len() {
                let partner = table.partner_under_y_mirror(idx);
                assert_complex_eq(
                    field.as_slice()[idx],
                    field.as_slice()[partner],
                    1e-12,
                    &format!("field {} at {}", fi, idx),
                );
            }
        }
    }

    #[test]
    fn apply_block_empty() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let projector = single_reflection_projector(grid, ReflectionAxis::X, Parity::Even);

        let mut fields: Vec<Field2D> = vec![];
        projector.apply_block(&mut fields);
        // Should not crash
    }
}
