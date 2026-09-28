use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend,
    dielectric::{Dielectric2D, DielectricOptions, SmoothingMethod},
    eigensolver::dense::solve_hermitian_eigen,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    geometry_warm_start::{
        predict::rayleigh_ritz_trial,
        response::predict,
        snapshot::{BasisGeneration, NativeSnapshot, Quality},
    },
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::response::ResponseConfig,
};
use num_complex::Complex64 as C;
type Op = ThetaOperator<CpuBackend<f64>>;
fn operator(n: usize, pol: Polarization, radius: f64) -> Op {
    operator_at(n, pol, radius, Lattice2D::square(1.), [0., 0.], [0., 0.])
}
fn operator_at(
    n: usize,
    pol: Polarization,
    radius: f64,
    lattice: Lattice2D,
    k: [f64; 2],
    position: [f64; 2],
) -> Op {
    let mut options = DielectricOptions::default();
    options.smoothing.method = SmoothingMethod::Voronoi;
    let d = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.,
            atoms: vec![BasisAtom {
                pos: position,
                radius,
                eps_inside: 8.9,
            }],
        },
        Grid2D::new(n, n, 1., 1.),
        &options,
    );
    ThetaOperator::new(
        CpuBackend::new(),
        d,
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}
fn dense(op: &mut Op, bands: usize) -> Vec<(f64, Field2D)> {
    let n = op.grid().len();
    let mut a = vec![C::ZERO; n * n];
    let mut x = op.alloc_field();
    let mut y = op.alloc_field();
    let inverse: Vec<_> = (0..n)
        .map(|i| {
            if op.polarization() == Polarization::TM {
                op.dielectric().eps()[i].sqrt().recip()
            } else {
                1.
            }
        })
        .collect();
    for j in 0..n {
        x.as_mut_slice().fill(C::ZERO);
        x.as_mut_slice()[j] = C::new(inverse[j], 0.);
        op.apply(&x, &mut y);
        for i in 0..n {
            a[i + j * n] = y.as_slice()[i] * inverse[i];
        }
    }
    for j in 0..n {
        for i in 0..=j {
            let v = 0.5 * (a[i + j * n] + a[j + i * n].conj());
            a[i + j * n] = v;
            a[j + i * n] = v.conj();
        }
    }
    let eig = solve_hermitian_eigen(&a, n);
    (0..bands)
        .map(|j| {
            let gamma = j == 0 && op.analytic_kernel_available();
            let values = if gamma {
                let norm = if op.polarization() == Polarization::TM {
                    op.dielectric().eps().iter().sum::<f64>().sqrt()
                } else {
                    (n as f64).sqrt()
                };
                vec![C::new(norm.recip(), 0.); n]
            } else {
                (0..n)
                    .map(|i| eig.eigenvectors[i + j * n] * inverse[i])
                    .collect()
            };
            (
                if gamma { 0. } else { eig.eigenvalues[j] },
                Field2D::from_vec(op.grid(), values),
            )
        })
        .collect()
}
fn span_error(op: &mut Op, actual: &[Field2D], reference: &[(f64, Field2D)]) -> f64 {
    let mut total = 0.;
    let mut mass = op.alloc_field();
    for field in actual {
        let mut residual = field.clone();
        for _ in 0..2 {
            op.apply_mass(&residual, &mut mass);
            for (_, q) in reference {
                let c = op.backend().dot(q, &mass);
                op.backend().axpy(-c, q, &mut residual);
            }
        }
        op.apply_mass(&residual, &mut mass);
        total += op.backend().dot(&residual, &mass).re;
    }
    total.max(0.).sqrt()
}
#[test]
fn native_gamma_response_has_second_order_low_span_and_valid_products() {
    for n in [7, 9] {
        for pol in [Polarization::TM, Polarization::TE] {
            let mut source = operator(n, pol, 0.203);
            let pairs = dense(&mut source, 7);
            // The requested window must contain the whole first doublet. A cut
            // inside it compares arbitrary bases, not a separated invariant span.
            assert!(pairs[4].0 - pairs[3].0 > 0.1);
            eprintln!(
                "spectrum {n} {pol:?}: {:?}",
                pairs.iter().map(|p| p.0).collect::<Vec<_>>()
            );
            let snapshot = NativeSnapshot::capture(
                &mut source,
                pairs,
                4,
                BasisGeneration {
                    job: 0,
                    k_index: 0,
                    accepted_generation: 1,
                },
                vec![],
                vec![0.203],
                true,
                true,
            )
            .unwrap();
            assert!(
                snapshot.admitted(Quality::default()),
                "{:?}",
                snapshot.certificate()
            );
            let mut copy_errors = Vec::new();
            let mut response_errors = Vec::new();
            for h in [0.001, 0.0005, 0.00025, 0.000125] {
                let mut target = operator(n, pol, 0.203 + h);
                let reference = dense(&mut target, 4);
                let response = predict(
                    &mut target,
                    &snapshot,
                    ResponseConfig {
                        max_inner_iterations: 256,
                        inner_relative_tolerance: 1e-11,
                    },
                    Quality::default(),
                    true,
                    1 << 24,
                )
                .unwrap();
                assert!(response.report.includes_gamma_kernel);
                assert_eq!(response.report.excluded_columns, 7);
                assert_eq!(response.report.response_columns, 3);
                assert_eq!(response.report.copied_guards, 3);
                assert!(
                    response
                        .report
                        .inner_columns
                        .iter()
                        .all(|c| c.inner_converged && c.excluded_space_defect < 1e-10)
                );
                if let Some(actions) = &response.actions {
                    let mut fresh = target.alloc_field();
                    for (x, ax) in response.vectors.iter().zip(actions) {
                        target.apply(x, &mut fresh);
                        target.backend().axpy(-C::ONE, ax, &mut fresh);
                        assert!(target.backend().dot(&fresh, &fresh).re.sqrt() < 1e-9);
                    }
                }
                let copy = rayleigh_ritz_trial(&mut target, snapshot.fields().to_vec(), 6, 1 << 24)
                    .unwrap();
                let response =
                    rayleigh_ritz_trial(&mut target, response.vectors, 6, 1 << 24).unwrap();
                copy_errors.push(span_error(&mut target, &copy.vectors[..3], &reference[1..]));
                response_errors.push(span_error(
                    &mut target,
                    &response.vectors[..3],
                    &reference[1..],
                ));
            }
            let copy_order = (copy_errors[1] / copy_errors[3]).log2() / 2.;
            let response_order = (response_errors[1] / response_errors[3]).log2() / 2.;
            eprintln!(
                "{n} {pol:?} copy={copy_errors:?} response={response_errors:?} orders=({copy_order},{response_order})"
            );
            assert!((0.9..1.1).contains(&copy_order));
            assert!((1.8..2.2).contains(&response_order));
            assert!(response_errors[3] < 0.05 * copy_errors[3]);
            let mut target = operator(n, pol, 0.204);
            for budget in [1, 2, 4] {
                let r = predict(
                    &mut target,
                    &snapshot,
                    ResponseConfig {
                        max_inner_iterations: budget,
                        inner_relative_tolerance: 1e-11,
                    },
                    Quality::default(),
                    false,
                    1 << 24,
                )
                .unwrap();
                assert!(r.actions.is_none());
                assert!(
                    r.report
                        .inner_columns
                        .iter()
                        .all(|c| c.iterations <= budget && c.excluded_space_defect < 1e-10)
                );
                assert!(r.report.inner_columns.iter().any(|c| !c.inner_converged));
            }
        }
    }
}

#[test]
fn nonzero_response_uses_its_own_pencil_through_radius_and_displacement_changes() {
    for (name, lattice) in [
        ("square", Lattice2D::square(1.)),
        ("triangular", Lattice2D::hexagonal(1.)),
    ] {
        for pol in [Polarization::TM, Polarization::TE] {
            for (kind, k) in [("radius", [0.5, 0.]), ("displacement", [0.17, 0.13])] {
                let mut source = operator_at(7, pol, 0.203, lattice, k, [0., 0.]);
                let pairs = dense(&mut source, 8);
                let requested = (2..=4)
                    .rev()
                    .find(|&p| pairs[p].0 - pairs[p - 1].0 > 0.05)
                    .unwrap();
                let snapshot = NativeSnapshot::capture(
                    &mut source,
                    pairs,
                    requested,
                    BasisGeneration {
                        job: 0,
                        k_index: 1,
                        accepted_generation: 1,
                    },
                    vec![],
                    vec![0.203],
                    true,
                    true,
                )
                .unwrap();
                assert!(snapshot.admitted(Quality::default()));
                assert!(snapshot.kernel().is_none());
                let mut errors = Vec::new();
                for h in [0.0004, 0.0002, 0.0001] {
                    let (radius, position) = if kind == "radius" {
                        (0.203 + h, [0., 0.])
                    } else {
                        (0.203, [h, 0.])
                    };
                    let mut target = operator_at(7, pol, radius, lattice, k, position);
                    let reference = dense(&mut target, requested);
                    let response = predict(
                        &mut target,
                        &snapshot,
                        ResponseConfig {
                            max_inner_iterations: 256,
                            inner_relative_tolerance: 1e-11,
                        },
                        Quality::default(),
                        true,
                        1 << 24,
                    )
                    .unwrap();
                    assert!(!response.report.includes_gamma_kernel);
                    assert_eq!(response.report.response_columns, requested);
                    assert!(
                        response
                            .report
                            .inner_columns
                            .iter()
                            .all(|c| c.inner_converged && c.excluded_space_defect < 1e-10)
                    );
                    let trial =
                        rayleigh_ritz_trial(&mut target, response.vectors, requested, 1 << 24)
                            .unwrap();
                    errors.push(span_error(&mut target, &trial.vectors, &reference));
                }
                let order = (errors[0] / errors[2]).log2() / 2.;
                eprintln!(
                    "{name} {pol:?} {kind} k={k:?} requested={requested} errors={errors:?} response_order={order}"
                );
                assert!((1.8..2.2).contains(&order));
            }
        }
    }
}
