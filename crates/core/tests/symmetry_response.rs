use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::EigensolverConfig,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        response::{ResponseConfig, predict_response, predict_response_symmetric},
        reuse::CachedSpectrum,
        sectors::solve_conservative,
        seed::rank_aware_seed,
    },
};
use num_complex::Complex64 as C;

fn operator(eps: f64, pol: Polarization, k: [f64; 2]) -> ThetaOperator<CpuBackend<f64>> {
    operator_on(5, eps, pol, k)
}

fn operator_on(
    n: usize,
    eps: f64,
    pol: Polarization,
    k: [f64; 2],
) -> ThetaOperator<CpuBackend<f64>> {
    let lattice = Lattice2D::square(1.0);
    ThetaOperator::new(
        CpuBackend::<f64>::new(),
        Dielectric2D::from_geometry(
            &Geometry2D {
                lattice,
                eps_bg: 1.0,
                atoms: vec![BasisAtom {
                    pos: [0.0; 2],
                    radius: 0.24,
                    eps_inside: eps,
                }],
            },
            Grid2D::new(n, n, 1.0, 1.0),
            &Default::default(),
        ),
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}

fn dense(op: &mut ThetaOperator<CpuBackend<f64>>, k: [f64; 2]) -> CachedSpectrum {
    solve_conservative(
        op,
        &Lattice2D::square(1.0),
        k,
        PlaneGroup::P1.group(),
        &EigensolverConfig {
            n_bands: op.grid().len(),
            tol: 1e-14,
            ..Default::default()
        },
        None,
        &Default::default(),
        None,
    )
    .unwrap()
    .spectrum
}

fn prefix(mut full: CachedSpectrum, count: usize) -> CachedSpectrum {
    full.eigenvalues.truncate(count);
    full.fields.truncate(count);
    full.residuals.truncate(count);
    full.absolute_residuals.truncate(count);
    full
}

// Frobenius norm of the component outside the exact target eigenspace, in
// the target mass metric. Form the residual vectors explicitly: subtracting
// squared overlaps from one would lose the second-order signal to roundoff.
fn span_error(
    op: &mut ThetaOperator<CpuBackend<f64>>,
    approximate: &[Field2D],
    exact: &[Field2D],
) -> f64 {
    let mass: Vec<_> = exact
        .iter()
        .map(|x| {
            let mut bx = op.alloc_field();
            op.apply_mass(x, &mut bx);
            bx
        })
        .collect();
    let mut error = 0.0;
    for field in approximate {
        let mut residual = field.clone();
        for (x, bx) in exact.iter().zip(&mass) {
            let overlap: C = bx
                .as_slice()
                .iter()
                .zip(field.as_slice())
                .map(|(b, a)| b.conj() * a)
                .sum();
            for (r, v) in residual.as_mut_slice().iter_mut().zip(x.as_slice()) {
                *r -= overlap * v;
            }
        }
        let mut br = op.alloc_field();
        op.apply_mass(&residual, &mut br);
        error += residual
            .as_slice()
            .iter()
            .zip(br.as_slice())
            .map(|(r, b)| (r.conj() * b).re)
            .sum::<f64>();
    }
    error.max(0.0).sqrt()
}

#[test]
fn response_has_second_order_span_error_for_a_separated_lowest_cluster() {
    let k = [0.13, 0.17];
    for n in [5, 7, 9] {
        for pol in [Polarization::TM, Polarization::TE] {
            let mut source = operator_on(n, 6.0, pol, k);
            let full = dense(&mut source, k);
            assert!(full.eigenvalues[4] - full.eigenvalues[3] > 0.05);
            let old = prefix(full, 4);
            let mut errors = Vec::new();
            for delta in [0.04, 0.02, 0.01] {
                let mut target = operator_on(n, 6.0 + delta, pol, k);
                let exact = prefix(dense(&mut target, k), 4);
                let copied = rank_aware_seed(&mut target, &old.fields, 1e-9).unwrap();
                let response = predict_response(
                    &mut source,
                    &mut target,
                    &old,
                    ResponseConfig {
                        max_inner_iterations: 400,
                        inner_relative_tolerance: 1e-11,
                    },
                    &Default::default(),
                )
                .unwrap();
                assert!(
                    response
                        .report
                        .columns
                        .iter()
                        .all(|r| r.inner_relative_residual < 2e-11
                            && r.excluded_space_defect < 1e-12)
                );
                assert_eq!(
                    response.report.work.inner_a_columns,
                    response.report.inner_iterations + 4
                );
                let copy_error = span_error(&mut target, &copied.fields, &exact.fields);
                let response_error = span_error(&mut target, &response.fields, &exact.fields);
                eprintln!(
                    "grid={n} {pol:?} delta={delta}: copy={copy_error:e}, response={response_error:e}"
                );
                assert!(response_error < copy_error * 0.02);
                errors.push((copy_error, response_error));
            }
            for pair in errors.windows(2) {
                assert!((1.9..2.1).contains(&(pair[0].0 / pair[1].0)));
                assert!((3.7..4.3).contains(&(pair[0].1 / pair[1].1)));
            }
        }
    }
}

#[test]
fn internal_degeneracy_does_not_make_the_excluded_space_equation_singular() {
    let k = [0.0; 2];
    for pol in [Polarization::TM, Polarization::TE] {
        let mut source = operator(1.0, pol, k);
        let full = dense(&mut source, k);
        assert!((full.eigenvalues[4] - full.eigenvalues[1]).abs() < 1e-10);
        assert!(full.eigenvalues[5] - full.eigenvalues[4] > 1.0);
        let old = prefix(full, 5);
        let mut errors = Vec::new();
        for delta in [0.02, 0.01] {
            let mut target = operator(1.0 + delta, pol, k);
            let exact = prefix(dense(&mut target, k), 5);
            let response = predict_response(
                &mut source,
                &mut target,
                &old,
                ResponseConfig {
                    max_inner_iterations: 100,
                    inner_relative_tolerance: 1e-11,
                },
                &Default::default(),
            )
            .unwrap();
            assert!(response.report.columns.iter().all(|r| r.inner_converged));
            assert!(response.report.target_seed.target_absolute_residuals[0] < 1e-9);
            errors.push(span_error(&mut target, &response.fields, &exact.fields));
        }
        assert!(
            (3.7..4.3).contains(&(errors[0] / errors[1])),
            "{pol:?}: {errors:?}"
        );
    }
}

#[test]
fn a_small_budget_is_reported_as_approximate_and_failures_keep_attempted_work() {
    let k = [0.13, 0.17];
    let mut source = operator(6.0, Polarization::TM, k);
    let full = dense(&mut source, k);
    let old = prefix(full.clone(), 4);
    let mut target = operator(6.2, Polarization::TM, k);
    let response = predict_response(
        &mut source,
        &mut target,
        &old,
        ResponseConfig {
            max_inner_iterations: 1,
            inner_relative_tolerance: 1e-12,
        },
        &Default::default(),
    )
    .unwrap();
    assert_eq!(response.report.inner_iterations, 4);
    assert_eq!(response.report.work.inner_a_columns, 8);
    assert!(response.report.columns.iter().all(|r| !r.inner_converged));

    let mut highest = full;
    let last = highest.fields.len() - 1;
    highest.fields = vec![highest.fields[last].clone()];
    highest.eigenvalues = vec![highest.eigenvalues[last]];
    let error = predict_response(
        &mut source,
        &mut target,
        &highest,
        Default::default(),
        &Default::default(),
    )
    .err()
    .unwrap();
    assert!(error.reason.contains("positive curvature"), "{error}");
    assert_eq!(error.work.source_certificate_columns, 1);
    assert_eq!(error.work.difference_a_columns, 2);
    assert_eq!(error.work.inner_a_columns, 1);

    let mut other_k = operator(6.2, Polarization::TM, [0.21, 0.17]);
    assert!(
        predict_response(
            &mut source,
            &mut other_k,
            &old,
            Default::default(),
            &Default::default()
        )
        .is_err()
    );
    let mut broken = old;
    broken.fields[0].as_mut_slice()[0] = C::new(f64::NAN, 0.0);
    assert!(
        predict_response(
            &mut source,
            &mut target,
            &broken,
            Default::default(),
            &Default::default()
        )
        .is_err()
    );
}

#[test]
fn selection_rules_preserve_full_response_accuracy_and_reduce_inner_coordinates() {
    let lattice = Lattice2D::square(1.0);
    let k = [0.17, 0.0];
    for pol in [Polarization::TM, Polarization::TE] {
        let mut source = operator_on(9, 6.0, pol, k);
        let old = prefix(
            solve_conservative(
                &mut source,
                &lattice,
                k,
                PlaneGroup::Pm.group(),
                &EigensolverConfig {
                    n_bands: 81,
                    tol: 1e-14,
                    ..Default::default()
                },
                None,
                &Default::default(),
                None,
            )
            .unwrap()
            .spectrum,
            8,
        );
        let mut target = operator_on(9, 6.2, pol, k);
        for budget in [4, 200] {
            let config = ResponseConfig {
                max_inner_iterations: budget,
                inner_relative_tolerance: 1e-10,
            };
            let full =
                predict_response(&mut source, &mut target, &old, config, &Default::default())
                    .unwrap();
            let reduced = predict_response_symmetric(
                &mut source,
                &mut target,
                &old,
                config,
                &Default::default(),
                &lattice,
                PlaneGroup::Pm.group(),
            )
            .unwrap();
            let error = span_error(&mut target, &reduced.fields, &full.fields);
            assert!(error < 2e-9, "{pol:?} budget={budget}: {error:e}");
            let selection = reduced.report.selection.unwrap();
            assert!(selection.removed_deflation_pairs > 0);
            assert!(selection.maximum_source_sector_defect < 1e-12);
            assert!(selection.maximum_rhs_sector_defect < 1e-11);
            assert_eq!(reduced.report.work.qualification_a_columns, 16);
            assert_eq!(reduced.report.work.qualification_b_columns, 16);
            for (a, b) in full.report.columns.iter().zip(&reduced.report.columns) {
                assert!(b.coordinate_rows < a.coordinate_rows);
                assert!(b.deflation_columns < a.deflation_columns);
                assert!(b.cg_workspace_payload_bytes < a.cg_workspace_payload_bytes);
            }
        }
    }
}

#[test]
fn selection_preserves_a_frozen_production_size_response_span() {
    let lattice = Lattice2D::square(1.0);
    let k = [0.13, 0.0];
    for (n, bands) in [(31, 8), (65, 16)] {
        for pol in [Polarization::TM, Polarization::TE] {
            let mut source = operator_on(n, 6.0, pol, k);
            let mut preconditioner = source.build_homogeneous_preconditioner_adaptive();
            let old = solve_conservative(
                &mut source,
                &lattice,
                k,
                PlaneGroup::Pm.group(),
                &EigensolverConfig {
                    n_bands: bands,
                    tol: 1e-14,
                    max_iter: 500,
                    ..Default::default()
                },
                None,
                &Default::default(),
                Some(&mut preconditioner),
            )
            .unwrap()
            .spectrum;
            let mut target = operator_on(n, 6.2, pol, k);
            let config = ResponseConfig {
                max_inner_iterations: 16,
                inner_relative_tolerance: 1e-6,
            };
            let full =
                predict_response(&mut source, &mut target, &old, config, &Default::default())
                    .unwrap();
            let selected = predict_response_symmetric(
                &mut source,
                &mut target,
                &old,
                config,
                &Default::default(),
                &lattice,
                PlaneGroup::Pm.group(),
            )
            .unwrap();
            let error = span_error(&mut target, &selected.fields, &full.fields);
            eprintln!("frozen response {n}²/{bands} {pol:?}: span difference {error:e}");
            assert!(error < 1e-8, "{n}²/{bands} {pol:?}: {error:e}");
            assert_eq!(
                full.report.inner_iterations,
                selected.report.inner_iterations
            );
        }
    }
}

#[test]
fn selection_rules_reject_changed_symmetry_and_mixed_degenerate_modes() {
    let lattice = Lattice2D::square(1.0);
    let k = [0.0; 2];
    let mut source = operator(1.0, Polarization::TM, k);
    let mut old = prefix(
        solve_conservative(
            &mut source,
            &lattice,
            k,
            PlaneGroup::Pm.group(),
            &EigensolverConfig {
                n_bands: 25,
                tol: 1e-14,
                ..Default::default()
            },
            None,
            &Default::default(),
            None,
        )
        .unwrap()
        .spectrum,
        5,
    );
    let mut target = operator(1.02, Polarization::TM, k);
    predict_response_symmetric(
        &mut source,
        &mut target,
        &old,
        Default::default(),
        &Default::default(),
        &lattice,
        PlaneGroup::Pm.group(),
    )
    .unwrap();
    let parity = |f: &Field2D| -> f64 {
        let g = f.grid();
        f.as_slice()
            .iter()
            .enumerate()
            .map(|(i, z)| (z.conj() * f.as_slice()[g.idx(i % g.nx, (g.ny - i / g.nx) % g.ny)]).re)
            .sum()
    };
    let even = (1..5).find(|&i| parity(&old.fields[i]) > 0.5).unwrap();
    let odd = (1..5).find(|&i| parity(&old.fields[i]) < -0.5).unwrap();
    assert!((old.eigenvalues[even] - old.eigenvalues[odd]).abs() < 1e-10);
    let a = old.fields[even].clone();
    let b = old.fields[odd].clone();
    for i in 0..25 {
        old.fields[even].as_mut_slice()[i] = (a.as_slice()[i] + b.as_slice()[i]) / 2.0_f64.sqrt();
        old.fields[odd].as_mut_slice()[i] = (a.as_slice()[i] - b.as_slice()[i]) / 2.0_f64.sqrt();
    }
    let error = predict_response_symmetric(
        &mut source,
        &mut target,
        &old,
        Default::default(),
        &Default::default(),
        &lattice,
        PlaneGroup::Pm.group(),
    )
    .err()
    .unwrap();
    assert!(error.reason.contains("mix inequivalent irreps"), "{error}");

    let mut source = operator_on(9, 6.0, Polarization::TM, [0.17, 0.0]);
    let old = prefix(dense(&mut source, [0.17, 0.0]), 4);
    let mut broken = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        Dielectric2D::from_geometry(
            &Geometry2D {
                lattice,
                eps_bg: 1.0,
                atoms: vec![BasisAtom {
                    pos: [0.0, 0.13],
                    radius: 0.24,
                    eps_inside: 6.2,
                }],
            },
            Grid2D::new(9, 9, 1.0, 1.0),
            &Default::default(),
        ),
        Polarization::TM,
        lattice.reciprocal().fractional_to_cartesian([0.17, 0.0]),
    );
    let error = predict_response_symmetric(
        &mut source,
        &mut broken,
        &old,
        Default::default(),
        &Default::default(),
        &lattice,
        PlaneGroup::Pm.group(),
    )
    .err()
    .unwrap();
    assert!(error.reason.contains("MaterialMismatch"), "{error}");
    assert!(error.work.qualification_a_columns > 0);
}

#[test]
fn complex_glide_and_projective_response_use_the_same_physical_pencil() {
    let lattice = Lattice2D::square(1.0);
    for group in [PlaneGroup::Pg, PlaneGroup::Pgg] {
        let (grid, k) = if group == PlaneGroup::Pg {
            (Grid2D::new(6, 5, 1.0, 1.0), [0.5, 0.0])
        } else {
            (Grid2D::new(6, 6, 1.0, 1.0), [-0.5, -0.5])
        };
        for pol in [Polarization::TM, Polarization::TE] {
            let make = |eps| {
                let atoms = group
                    .group()
                    .operations
                    .iter()
                    .map(|g| {
                        let w = g.matrix();
                        let t = g.translation();
                        BasisAtom {
                            pos: std::array::from_fn(|i| {
                                w[i][0] as f64 * 0.13 + w[i][1] as f64 * 0.17 + t[i].value()
                            }),
                            radius: 0.16,
                            eps_inside: eps,
                        }
                    })
                    .collect();
                ThetaOperator::new(
                    CpuBackend::<f64>::new(),
                    Dielectric2D::from_geometry(
                        &Geometry2D {
                            lattice,
                            eps_bg: 1.0,
                            atoms,
                        },
                        grid,
                        &Default::default(),
                    ),
                    pol,
                    lattice.reciprocal().fractional_to_cartesian(k),
                )
            };
            let mut source = make(6.0);
            let mut target = make(6.02);
            let full = solve_conservative(
                &mut source,
                &lattice,
                k,
                group.group(),
                &EigensolverConfig {
                    n_bands: grid.len(),
                    tol: 1e-14,
                    ..Default::default()
                },
                None,
                &Default::default(),
                None,
            )
            .unwrap()
            .spectrum;
            let mut count = 4;
            while count < grid.len() && full.eigenvalues[count] - full.eigenvalues[count - 1] < 1e-7
            {
                count += 1;
            }
            assert!(count < grid.len());
            let old = prefix(full, count);
            let config = ResponseConfig {
                max_inner_iterations: 200,
                inner_relative_tolerance: 1e-9,
            };
            let reference =
                predict_response(&mut source, &mut target, &old, config, &Default::default())
                    .unwrap();
            let selected = predict_response_symmetric(
                &mut source,
                &mut target,
                &old,
                config,
                &Default::default(),
                &lattice,
                group.group(),
            )
            .unwrap();
            let error = span_error(&mut target, &selected.fields, &reference.fields);
            assert!(error < 1e-8, "{group:?} {pol:?}: {error:e}");
            assert!(
                selected
                    .report
                    .columns
                    .iter()
                    .all(|r| r.inner_relative_residual < 2e-9)
            );
        }
    }
}
