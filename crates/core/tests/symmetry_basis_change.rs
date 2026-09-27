//! G09: a passive primitive-basis change must preserve the same finite pencil.
//! L' = L M, x' = M^-1 x, k' = M^T k, W' = M^-1 W M, tau' = M^-1 tau.
//! Metric supports are transported exactly; an FFT rectangle is not invariant
//! under a shear and is therefore not the numerical pencil compared here.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::EigensolverConfig,
    field::Field2D,
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::{AffineGroup, AffineOp, PlaneGroup},
        galerkin::{GalerkinApplication, GalerkinOperator, dense_pencil, solve_dense},
        support::{CoefficientAction, FourierSupport},
        support_qualification::QualifiedSupportGroup,
        support_sectors::{SupportSectorOptions, SupportSectorPlan, solve_with_plan},
    },
};
use num_complex::Complex64 as C;
use std::{collections::HashMap, f64::consts::TAU};

type Matrix = [[i32; 2]; 2];
const ID: Matrix = [[1, 0], [0, 1]];

fn multiply(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..2).map(|k| a[i][k] * b[k][j]).sum()))
}
fn inverse(m: Matrix) -> Matrix {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    assert_eq!(d.abs(), 1);
    [[d * m[1][1], -d * m[0][1]], [-d * m[1][0], d * m[0][0]]]
}
fn dual(m: Matrix, k: [f64; 2]) -> [f64; 2] {
    std::array::from_fn(|i| (0..2).map(|j| m[j][i] as f64 * k[j]).sum())
}
fn lattice(base: Lattice2D, m: Matrix) -> Lattice2D {
    Lattice2D::oblique(
        std::array::from_fn(|i| base.a1[i] * m[0][0] as f64 + base.a2[i] * m[1][0] as f64),
        std::array::from_fn(|i| base.a1[i] * m[0][1] as f64 + base.a2[i] * m[1][1] as f64),
    )
}
fn operation(op: AffineOp, m: Matrix) -> (AffineOp, [i64; 2]) {
    let inv = inverse(m);
    let t = op.translation();
    let raw = std::array::from_fn(|i| {
        t[0].checked_scale(inv[i][0])
            .unwrap()
            .checked_add(t[1].checked_scale(inv[i][1]).unwrap())
            .unwrap()
    });
    let changed = AffineOp::new(multiply(multiply(inv, op.matrix()), m), raw)
        .unwrap()
        .with_conjugation(op.conjugates());
    let carry =
        std::array::from_fn(|i| (raw[i].value() - changed.translation()[i].value()).round() as i64);
    (changed, carry)
}
fn sample_permutation(n: usize, m: Matrix) -> Vec<usize> {
    (0..n * n)
        .map(|j| {
            let x = (j % n) as i32;
            let y = (j / n) as i32;
            let xx = (m[0][0] * x + m[0][1] * y).rem_euclid(n as i32) as usize;
            let yy = (m[1][0] * x + m[1][1] * y).rem_euclid(n as i32) as usize;
            yy * n + xx
        })
        .collect()
}
fn coefficient_permutation(a: &FourierSupport, b: &FourierSupport, m: Matrix) -> Vec<usize> {
    let lookup: HashMap<_, _> = b
        .modes()
        .iter()
        .copied()
        .enumerate()
        .map(|(i, g)| (g, i))
        .collect();
    let perm: Vec<_> = a
        .modes()
        .iter()
        .map(|g| {
            let mapped: [i64; 2] =
                std::array::from_fn(|i| (0..2).map(|j| m[j][i] as i64 * g[j]).sum());
            *lookup
                .get(&mapped)
                .expect("complete metric support must map without wrapping or clipping")
        })
        .collect();
    assert_eq!(perm.len(), b.len());
    let mut unique = perm.clone();
    unique.sort_unstable();
    assert_eq!(unique, (0..b.len()).collect::<Vec<_>>());
    perm
}
fn transport(x: &[C], permutation: &[usize]) -> Vec<C> {
    let mut y = vec![C::ZERO; x.len()];
    for (i, &j) in permutation.iter().enumerate() {
        y[j] = x[i];
    }
    y
}
fn material(grid: Grid2D, repeated: bool, angle: f64) -> (Vec<f64>, Vec<[f64; 4]>) {
    let mut eps = Vec::new();
    let mut tensor = Vec::new();
    let q = [[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let f = if repeated { 2.0 } else { 1.0 };
            let x = TAU * f * ix as f64 / grid.nx as f64;
            let y = TAU * f * iy as f64 / grid.ny as f64;
            eps.push(3.0 + 0.2 * (x.cos() + y.cos()) + 0.07 * x.cos() * y.cos());
            let xy = 0.007 * x.sin() * y.sin();
            let t = [
                [0.35 + 0.025 * x.cos() + 0.01 * y.cos(), xy],
                [xy, 0.35 + 0.01 * x.cos() + 0.025 * y.cos()],
            ];
            let cart: [[f64; 2]; 2] = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    (0..2)
                        .flat_map(|a| (0..2).map(move |b| q[i][a] * t[a][b] * q[j][b]))
                        .sum()
                })
            });
            tensor.push([cart[0][0], cart[0][1], cart[1][0], cart[1][1]]);
        }
    }
    (eps, tensor)
}
fn pencil(
    grid: Grid2D,
    l: Lattice2D,
    k: [f64; 2],
    pol: Polarization,
    eps: Vec<f64>,
    tensor: Vec<[f64; 4]>,
) -> ThetaOperator<CpuBackend<f64>> {
    let b = l.reciprocal();
    ThetaOperator::new(
        CpuBackend::new(),
        Dielectric2D::from_sampled_epsilon(
            grid,
            b.b1,
            b.b2,
            eps,
            (pol == Polarization::TE).then_some(tensor),
        ),
        pol,
        b.fractional_to_cartesian(k),
    )
}
fn matrices(op: &mut ThetaOperator<CpuBackend<f64>>, s: &FourierSupport) -> (Vec<C>, Vec<C>) {
    let mut p = GalerkinOperator::new(op, s, GalerkinApplication::DirectFourier, 1 << 28).unwrap();
    dense_pencil(&mut p, 1 << 28).unwrap()
}
fn matrix_error(a: &[C], b: &[C], perm: &[usize]) -> f64 {
    let m = perm.len();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for j in 0..m {
        for i in 0..m {
            numerator += (a[i + m * j] - b[perm[i] + m * perm[j]]).norm_sqr();
            denominator += a[i + m * j].norm_sqr();
        }
    }
    numerator.sqrt() / denominator.sqrt().max(1.0)
}

#[test]
fn primitive_basis_changes_preserve_scalar_tensor_pencils_spectra_and_bloch_actions() {
    let angle: f64 = 0.37;
    let scale = 1.3;
    let base = Lattice2D::oblique(
        [scale * angle.cos(), scale * angle.sin()],
        [-scale * angle.sin(), scale * angle.cos()],
    );
    let grid = Grid2D::new(24, 24, 1.0, 1.0);
    // Two shears exercise both translations; the third basis reverses orientation.
    let changes = [[[1, 1], [0, 1]], [[1, 0], [1, 1]], [[0, 1], [1, 0]]];
    let mut nonzero_carries = 0;
    for group in [PlaneGroup::P4m, PlaneGroup::Pgg] {
        let (eps, tensor) = material(grid, group == PlaneGroup::Pgg, angle);
        for k in [[0.0, 0.0], [0.5, 0.0], [0.17, 0.09]] {
            let s =
                FourierSupport::metric_ball(grid, base, k, TAU * 2.25 / scale, Default::default())
                    .unwrap();
            let old_group = group.group().little_group(k, 1e-12).unwrap();
            for pol in [Polarization::TM, Polarization::TE] {
                let mut op = pencil(grid, base, k, pol, eps.clone(), tensor.clone());
                let original = matrices(&mut op, &s);
                let qualified = QualifiedSupportGroup::qualify(
                    &mut op,
                    &s,
                    old_group.clone(),
                    Default::default(),
                )
                .unwrap();
                assert!(qualified.report().authorized);
                let config = EigensolverConfig {
                    n_bands: s.len(),
                    tol: 1e-14,
                    max_iter: 500,
                    ..Default::default()
                };
                let a = solve_dense(
                    &mut op,
                    &s,
                    &config,
                    GalerkinApplication::DirectFourier,
                    Default::default(),
                )
                .unwrap();
                assert!(a.report.accepted);
                for m in changes {
                    assert_eq!(multiply(m, inverse(m)), ID);
                    let changed_lattice = lattice(base, m);
                    let changed_k = dual(m, k);
                    let b0 = base.reciprocal().fractional_to_cartesian(k);
                    let b1 = changed_lattice
                        .reciprocal()
                        .fractional_to_cartesian(changed_k);
                    assert!((b0[0] - b1[0]).hypot(b0[1] - b1[1]) < 2e-14);
                    let t = FourierSupport::metric_ball(
                        grid,
                        changed_lattice,
                        changed_k,
                        TAU * 2.25 / scale,
                        Default::default(),
                    )
                    .unwrap();
                    let permutation = coefficient_permutation(&s, &t, m);
                    let nodes = sample_permutation(grid.nx, m);
                    let eps2: Vec<_> = nodes.iter().map(|&i| eps[i]).collect();
                    let tensor2: Vec<_> = nodes.iter().map(|&i| tensor[i]).collect();
                    let mut changed =
                        pencil(grid, changed_lattice, changed_k, pol, eps2.clone(), tensor2);
                    let transformed = matrices(&mut changed, &t);
                    assert!(
                        matrix_error(&original.0, &transformed.0, &permutation) < 3e-12,
                        "{group:?} {pol:?} {k:?} {m:?} stiffness"
                    );
                    assert!(
                        matrix_error(&original.1, &transformed.1, &permutation) < 3e-12,
                        "{group:?} {pol:?} {k:?} {m:?} mass"
                    );
                    let generators: Vec<_> = old_group
                        .operations
                        .iter()
                        .map(|&op| operation(op, m).0)
                        .collect();
                    let changed_group =
                        AffineGroup::from_generators(&generators, old_group.order()).unwrap();
                    let qualified = QualifiedSupportGroup::qualify(
                        &mut changed,
                        &t,
                        changed_group,
                        Default::default(),
                    )
                    .unwrap();
                    assert!(qualified.report().authorized);
                    let probe: Vec<_> = (0..s.len())
                        .map(|i| C::new((i as f64 + 0.3).sin(), (i as f64 + 0.7).cos()))
                        .collect();
                    let transported = transport(&probe, &permutation);
                    let backend = CpuBackend::<f64>::new();
                    let mut field = Field2D::zeros(grid);
                    let mut field2 = Field2D::zeros(grid);
                    s.expand(&backend, &probe, &mut field);
                    t.expand(&backend, &transported, &mut field2);
                    for (j, &i) in nodes.iter().enumerate() {
                        assert!((field2.as_slice()[j] - field.as_slice()[i]).norm() < 3e-13);
                    }
                    for &g in &old_group.operations {
                        let (h, carry) = operation(g, m);
                        nonzero_carries += usize::from(carry != [0, 0]);
                        let old_action = CoefficientAction::new(g, &s, &s, pol, 1e-12).unwrap();
                        let new_action = CoefficientAction::new(h, &t, &t, pol, 1e-12).unwrap();
                        let mut left = vec![C::ZERO; s.len()];
                        let mut right = left.clone();
                        old_action.apply(&probe, &mut left);
                        new_action.apply(&transported, &mut right);
                        let left = transport(&left, &permutation);
                        let carry_phase = C::from_polar(
                            1.0,
                            -TAU * (changed_k[0] * carry[0] as f64
                                + changed_k[1] * carry[1] as f64),
                        );
                        for (x, y) in left.iter().zip(right) {
                            assert!(
                                (*x - carry_phase * y).norm() < 3e-12,
                                "{group:?} {pol:?} {k:?} carry {carry:?}"
                            );
                        }
                    }
                    let b = solve_dense(
                        &mut changed,
                        &t,
                        &config,
                        GalerkinApplication::DirectFourier,
                        Default::default(),
                    )
                    .unwrap();
                    assert!(b.report.accepted);
                    for (&x, &y) in a.eigenvalues.iter().zip(&b.eigenvalues) {
                        assert!((x - y).abs() / x.abs().max(y.abs()).max(1.0) < 3e-11);
                    }
                    let plan =
                        SupportSectorPlan::new(qualified, &changed, &t, true, 1 << 28).unwrap();
                    let reduced = solve_with_plan(
                        &mut changed,
                        &t,
                        &plan,
                        &config,
                        None,
                        None,
                        SupportSectorOptions {
                            complete_dense: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert!(reduced.report.accepted && reduced.spectrum.report.accepted);
                    assert_eq!(reduced.report.sum_sector_dimensions, s.len());
                    assert_eq!(reduced.spectrum.eigenvalues.len(), s.len());
                    for (&x, &y) in a.eigenvalues.iter().zip(&reduced.spectrum.eigenvalues) {
                        assert!((x - y).abs() / x.abs().max(y.abs()).max(1.0) < 3e-11);
                    }
                    // Compare complete eigenclusters in the common physical B metric.
                    // Individual vectors/phases inside a degenerate cluster are arbitrary.
                    let mut first = 0;
                    while first < s.len() {
                        let mut end = first + 1;
                        while end < s.len()
                            && [&a.eigenvalues, &b.eigenvalues].iter().any(|values| {
                                (values[end] - values[end - 1]).abs()
                                    <= 1e-7 * values[end].abs().max(1.0)
                            })
                        {
                            end += 1;
                        }
                        let mut norm = 0.0;
                        for i in first..end {
                            for j in first..end {
                                let overlap: C = nodes
                                    .iter()
                                    .enumerate()
                                    .map(|(new, &old)| {
                                        a.fields[i].as_slice()[old].conj()
                                            * b.fields[j].as_slice()[new]
                                            * if pol == Polarization::TM {
                                                eps2[new]
                                            } else {
                                                1.0
                                            }
                                    })
                                    .sum();
                                norm += overlap.norm_sqr();
                            }
                        }
                        assert!(
                            (norm - (end - first) as f64).abs() < 1e-8,
                            "{group:?} {pol:?} {k:?} physical cluster {first}..{end}"
                        );
                        first = end;
                    }
                }
            }
        }
    }
    assert!(
        nonzero_carries > 0,
        "fractional-translation basis changes must exercise canonical carries"
    );
}

#[test]
fn failing_to_transform_operations_or_material_is_detected() {
    let grid = Grid2D::new(24, 24, 1.0, 1.0);
    let base = Lattice2D::square(1.0);
    let m = [[1, 1], [0, 1]];
    let changed = lattice(base, m);
    let k = [0.0, 0.0];
    let s = FourierSupport::metric_ball(grid, base, k, TAU * 2.25, Default::default()).unwrap();
    let t = FourierSupport::metric_ball(grid, changed, k, TAU * 2.25, Default::default()).unwrap();
    let (eps, tensor) = material(grid, false, 0.0);
    let nodes = sample_permutation(grid.nx, m);
    let right_eps: Vec<_> = nodes.iter().map(|&i| eps[i]).collect();
    let right_tensor: Vec<_> = nodes.iter().map(|&i| tensor[i]).collect();
    let mut right = pencil(grid, changed, k, Polarization::TM, right_eps, right_tensor);
    assert!(
        QualifiedSupportGroup::qualify(&mut right, &t, PlaneGroup::P4m.group(), Default::default())
            .is_err()
    );
    let mut old = pencil(grid, base, k, Polarization::TM, eps.clone(), tensor.clone());
    let mut wrong = pencil(grid, changed, k, Polarization::TM, eps, tensor);
    let p = coefficient_permutation(&s, &t, m);
    assert!(matrix_error(&matrices(&mut old, &s).1, &matrices(&mut wrong, &t).1, &p) > 1e-3);
    let generators: Vec<_> = PlaneGroup::P4m
        .group()
        .operations
        .iter()
        .map(|&op| operation(op, m).0)
        .collect();
    let transformed = AffineGroup::from_generators(&generators, 8).unwrap();
    assert!(
        QualifiedSupportGroup::qualify(&mut wrong, &t, transformed, Default::default()).is_err()
    );
}
