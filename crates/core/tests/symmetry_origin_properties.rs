use blaze2d_core::{
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{
        affine::{AffineOp, PlaneGroup, Rational},
        support::{CoefficientAction, FourierSupport, SupportBudget},
    },
};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

fn lattice(name: PlaneGroup) -> Lattice2D {
    match name {
        PlaneGroup::P1 | PlaneGroup::P2 => Lattice2D::oblique([1.0, 0.0], [0.3, 1.1]),
        PlaneGroup::Cm | PlaneGroup::Cmm => Lattice2D::oblique([0.5, 0.7], [-0.5, 0.7]),
        PlaneGroup::P4 | PlaneGroup::P4m | PlaneGroup::P4g => Lattice2D::square(1.0),
        PlaneGroup::P3 | PlaneGroup::P3m1 | PlaneGroup::P31m | PlaneGroup::P6 | PlaneGroup::P6m => {
            Lattice2D::hexagonal(1.0)
        }
        _ => Lattice2D::rectangular(1.0, 1.4),
    }
}

// An integer common-denominator oracle keeps the expected carries independent
// of Rational::checked_add/checked_scale and AffineOp::compose.
fn shifted(op: AffineOp, numerator: [i64; 2], denominator: i64) -> AffineOp {
    let unit = 2 * denominator;
    let w = op.matrix();
    let tau = op.translation();
    let translated = std::array::from_fn(|i| {
        assert_eq!(unit % tau[i].denominator(), 0);
        let value = tau[i].numerator() * (unit / tau[i].denominator())
            + 2 * (numerator[i] - (0..2).map(|j| w[i][j] as i64 * numerator[j]).sum::<i64>());
        Rational::new(value, unit).unwrap()
    });
    AffineOp::new(w, translated).unwrap()
}

fn integer_translation(op: AffineOp, unit: i64) -> [i64; 2] {
    op.translation().map(|t| {
        assert_eq!(unit % t.denominator(), 0);
        t.numerator() * (unit / t.denominator())
    })
}

const ORIGINS: [([i64; 2], i64); 5] = [
    ([0, 0], 1),
    ([1, -1], 2),
    ([-2, 4], 3),
    ([7, -11], 5),
    ([-13, 17], 7),
];

#[test]
fn every_plane_group_keeps_exact_products_and_inverse_carries_after_origin_changes() {
    for name in PlaneGroup::ALL {
        let group = name.group();
        for (origin, denominator) in ORIGINS {
            let unit = 2 * denominator;
            let operations: Vec<_> = group
                .operations
                .iter()
                .map(|&op| shifted(op, origin, denominator))
                .collect();
            for (i, &left) in operations.iter().enumerate() {
                let inverse = (0..group.order())
                    .find(|&j| group.product_index[i][j] == 0 && group.product_index[j][i] == 0)
                    .unwrap();
                assert_eq!(
                    left.compose(operations[inverse]).unwrap().0,
                    AffineOp::IDENTITY
                );
                for (j, &right) in operations.iter().enumerate() {
                    let a = integer_translation(left, unit);
                    let b = integer_translation(right, unit);
                    let w = left.matrix();
                    let unwrapped: [i64; 2] = std::array::from_fn(|axis| {
                        a[axis] + (0..2).map(|d| w[axis][d] as i64 * b[d]).sum::<i64>()
                    });
                    let (actual, carry) = left.compose(right).unwrap();
                    assert_eq!(
                        actual, operations[group.product_index[i][j]],
                        "{name:?} {origin:?}"
                    );
                    assert_eq!(carry, unwrapped.map(|x| x.div_euclid(unit)));
                    assert_eq!(
                        integer_translation(actual, unit),
                        unwrapped.map(|x| x.rem_euclid(unit))
                    );
                }
            }
        }
    }
}

#[test]
fn origin_shifted_unitary_and_antiunitary_inverse_actions_restore_full_coefficients() {
    let grid = Grid2D::new(17, 17, 1.0, 1.0);
    let mut nontrivial_carries = 0;
    for name in PlaneGroup::ALL {
        let group = name.group();
        for k in [
            [0.0, 0.0],
            [0.13, -0.21],
            [0.5, 0.0],
            [2.0 / 3.0, 1.0 / 3.0],
        ] {
            let source = FourierSupport::metric_ball(
                grid,
                lattice(name),
                k,
                TAU * 2.35,
                SupportBudget::default(),
            )
            .unwrap();
            let input: Vec<_> = (0..source.len())
                .map(|i| C::new((1.37 * i as f64).sin(), (0.73 * (i + 1) as f64).cos()))
                .collect();
            let norm = input.iter().map(C::norm_sqr).sum::<f64>();
            for (origin, denominator) in ORIGINS {
                let operations: Vec<_> = group
                    .operations
                    .iter()
                    .map(|&op| shifted(op, origin, denominator))
                    .collect();
                for (i, &op) in operations.iter().enumerate() {
                    let inverse = operations[(0..group.order())
                        .find(|&j| group.product_index[j][i] == 0)
                        .unwrap()];
                    for antiunitary in [false, true] {
                        let forward_op = op.with_conjugation(antiunitary);
                        let backward_op = inverse.with_conjugation(antiunitary);
                        let (identity, carry) = backward_op.compose(forward_op).unwrap();
                        assert_eq!(identity, AffineOp::IDENTITY);
                        let destination = FourierSupport::metric_ball(
                            grid,
                            lattice(name),
                            forward_op.map_wavevector(k),
                            TAU * 2.35,
                            SupportBudget::default(),
                        )
                        .unwrap();
                        let phase = C::from_polar(
                            1.0,
                            -TAU * (k[0] * carry[0] as f64 + k[1] * carry[1] as f64),
                        );
                        for pol in [Polarization::TM, Polarization::TE] {
                            let forward = CoefficientAction::new(
                                forward_op,
                                &source,
                                &destination,
                                pol,
                                1e-12,
                            )
                            .unwrap();
                            let backward = CoefficientAction::new(
                                backward_op,
                                &destination,
                                &source,
                                pol,
                                1e-12,
                            )
                            .unwrap();
                            let mut transformed = vec![C::ZERO; destination.len()];
                            let mut restored = vec![C::ZERO; source.len()];
                            forward.apply(&input, &mut transformed);
                            backward.apply(&transformed, &mut restored);
                            assert!(
                                (transformed.iter().map(C::norm_sqr).sum::<f64>() - norm).abs()
                                    < 3e-13 * norm
                            );
                            let defect = restored
                                .iter()
                                .zip(&input)
                                .map(|(a, b)| (*a - phase * b).norm_sqr())
                                .sum::<f64>()
                                .sqrt()
                                / norm.sqrt();
                            assert!(
                                defect < 2e-12,
                                "{name:?} {origin:?}/{denominator} {k:?} {pol:?} {antiunitary}: {defect}"
                            );
                            // Omitting the carry must actually change these fixtures.
                            if (phase - C::ONE).norm() > 0.1 {
                                let uncorrected = restored
                                    .iter()
                                    .zip(&input)
                                    .map(|(a, b)| (*a - b).norm_sqr())
                                    .sum::<f64>()
                                    .sqrt()
                                    / norm.sqrt();
                                assert!(uncorrected > 0.09);
                                nontrivial_carries += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(nontrivial_carries > 100);
}
