use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend,
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{
        action::{BlochActionPlan, fft_indices},
        affine::{AffineOp, PlaneGroup},
        support::{CoefficientAction, FourierSupport, SupportBudget},
    },
};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

fn support(grid: Grid2D, lattice: Lattice2D, k: [f64; 2], radius: f64) -> FourierSupport {
    FourierSupport::metric_ball(grid, lattice, k, TAU * radius, SupportBudget::default()).unwrap()
}

fn coefficients(n: usize) -> Vec<C> {
    (0..n)
        .map(|i| C::new(((i + 1) as f64 * 1.37).sin(), ((i + 3) as f64 * 0.73).cos()))
        .collect()
}

#[test]
fn metric_ball_is_nested_and_never_aliases_missing_nyquist_or_clips_edges() {
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let lattice = Lattice2D::hexagonal(1.0);
    let a = support(grid, lattice, [0.0, 0.0], 1.5);
    let b = support(grid, lattice, [0.0, 0.0], 3.0);
    assert!(a.len() < b.len());
    assert!(a.modes().iter().all(|g| b.modes().contains(g)));
    for (g, &index) in b.modes().iter().zip(b.embedding_indices()) {
        assert_eq!(
            *g,
            [
                fft_indices(grid.nx)[index % grid.nx],
                fft_indices(grid.ny)[index / grid.nx]
            ]
        );
    }
    assert!(
        FourierSupport::metric_ball(grid, lattice, [0.0; 2], TAU * 10.0, Default::default())
            .is_err()
    );
    let even = Grid2D::new(8, 8, 1.0, 1.0);
    let square = Lattice2D::square(1.0);
    assert!(
        FourierSupport::metric_ball(even, square, [0.0; 2], TAU * 4.0, Default::default()).is_err()
    );
    assert!(
        FourierSupport::metric_ball(even, square, [0.5, 0.0], TAU * 3.6, Default::default())
            .is_err()
    );
    let boundary = support(even, square, [-0.5, 0.0], 3.6);
    for operation in PlaneGroup::Pmm
        .group()
        .little_group([-0.5, 0.0], 1e-12)
        .unwrap()
        .operations
    {
        CoefficientAction::new(operation, &boundary, &boundary, Polarization::TM, 1e-12).unwrap();
    }
    let a = support(grid, square, [0.0; 2], 1.01);
    let b = support(grid, square, [0.0; 2], 1.05);
    assert_eq!(a.identity(), b.identity());
    assert_ne!(a.angular_cutoff(), b.angular_cutoff());
}

#[test]
fn retained_hexagonal_and_glide_actions_obey_the_actual_multiplier() {
    for (name, lattice, k, n) in [
        (PlaneGroup::P6m, Lattice2D::hexagonal(1.0), [0.0, 0.0], 15),
        (
            PlaneGroup::P6m,
            Lattice2D::hexagonal(1.0),
            [2.0 / 3.0, 1.0 / 3.0],
            15,
        ),
        (PlaneGroup::Pg, Lattice2D::square(1.0), [0.5, 0.0], 12),
    ] {
        let grid = Grid2D::new(n, n, 1.0, 1.0);
        let space = support(grid, lattice, k, 2.6);
        let group = name.group().little_group(k, 1e-12).unwrap();
        if name == PlaneGroup::P6m {
            assert!(
                group.operations.iter().any(|&g| BlochActionPlan::new(
                    g,
                    k,
                    k,
                    grid,
                    Polarization::TM,
                    1e-12
                )
                .is_err())
            );
        }
        for pol in [Polarization::TM, Polarization::TE] {
            let actions: Vec<_> = group
                .operations
                .iter()
                .map(|&g| CoefficientAction::new(g, &space, &space, pol, 1e-12).unwrap())
                .collect();
            let input = coefficients(space.len());
            let mut first = vec![C::ZERO; space.len()];
            let mut composed = first.clone();
            let mut product = first.clone();
            for g in 0..group.order() {
                for h in 0..group.order() {
                    actions[h].apply(&input, &mut first);
                    actions[g].apply(&first, &mut composed);
                    actions[group.product_index[g][h]].apply(&input, &mut product);
                    let carry = group.translation_carry[g][h];
                    let phase = C::from_polar(
                        1.0,
                        -TAU * (k[0] * carry[0] as f64 + k[1] * carry[1] as f64),
                    );
                    let error = composed
                        .iter()
                        .zip(&product)
                        .map(|(a, b)| (*a - phase * b).norm())
                        .fold(0.0, f64::max);
                    assert!(error < 2e-12, "{name:?} {k:?} {pol:?} {g} {h}: {error}");
                }
            }
        }
    }
}

#[test]
fn plane_wave_embedding_is_isometric_and_restriction_is_its_adjoint() {
    let backend = CpuBackend::<f64>::new();
    for n in [12, 15] {
        let grid = Grid2D::new(n, n, 1.0, 1.0);
        let space = support(grid, Lattice2D::hexagonal(1.0), [2.0 / 3.0, 1.0 / 3.0], 2.6);
        let v = coefficients(space.len());
        let mut full = backend.alloc_field(grid);
        let mut scratch = backend.alloc_field(grid);
        space.expand(&backend, &v, &mut full);
        let expected = v.iter().map(C::norm_sqr).sum::<f64>();
        let actual = full.as_slice().iter().map(C::norm_sqr).sum::<f64>();
        assert!((actual - expected).abs() < 2e-13 * expected);
        let mut returned = vec![C::ZERO; space.len()];
        space.restrict(&backend, &full, &mut scratch, &mut returned);
        assert!(
            returned
                .iter()
                .zip(&v)
                .all(|(a, b)| (*a - *b).norm() < 2e-13)
        );
        let y = coefficients(grid.len());
        let mut physical_y = backend.alloc_field(grid);
        physical_y.as_mut_slice().copy_from_slice(&y);
        space.restrict(&backend, &physical_y, &mut scratch, &mut returned);
        let lhs: C = full
            .as_slice()
            .iter()
            .zip(&y)
            .map(|(a, b)| a.conj() * b)
            .sum();
        let rhs: C = v.iter().zip(&returned).map(|(a, b)| a.conj() * b).sum();
        assert!((lhs - rhs).norm() < 2e-12);
    }
}

#[test]
fn antiunitary_maps_and_resource_rejections_do_not_grant_operator_qualification() {
    let grid = Grid2D::new(15, 15, 1.0, 1.0);
    let lattice = Lattice2D::square(1.0);
    let k = [0.23, 0.11];
    let a = support(grid, lattice, k, 2.6);
    let b = support(grid, lattice, k.map(|x| -x), 2.6);
    let tr = AffineOp::IDENTITY.with_conjugation(true);
    for pol in [Polarization::TM, Polarization::TE] {
        let forward = CoefficientAction::new(tr, &a, &b, pol, 1e-12).unwrap();
        let backward = CoefficientAction::new(tr, &b, &a, pol, 1e-12).unwrap();
        assert!(forward.trace().is_err());
        let v = coefficients(a.len());
        let mut transformed = vec![C::ZERO; a.len()];
        let mut restored = transformed.clone();
        forward.apply(&v, &mut transformed);
        backward.apply(&transformed, &mut restored);
        assert!(v.iter().zip(restored).all(|(x, y)| (*x - y).norm() < 1e-13));
    }
    assert!(CoefficientAction::new(tr, &a, &a, Polarization::TM, 1e-12).is_err());
    for budget in [
        SupportBudget {
            max_coefficients: 1,
            ..Default::default()
        },
        SupportBudget {
            max_embedding_points: 1,
            ..Default::default()
        },
        SupportBudget {
            max_enumerated_indices: 1,
            ..Default::default()
        },
    ] {
        assert!(FourierSupport::metric_ball(grid, lattice, k, TAU * 2.6, budget).is_err());
    }
    assert!(
        FourierSupport::metric_ball(grid, lattice, [f64::NAN, 0.0], TAU, Default::default())
            .is_err()
    );
    assert!(
        FourierSupport::metric_ball(
            Grid2D::new_centered(15, 15, 1.0, 1.0),
            lattice,
            k,
            TAU,
            Default::default()
        )
        .is_err()
    );
    assert!(FourierSupport::metric_ball(grid, lattice, k, 0.01, Default::default()).is_err());
}
