use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        action::{ActionError, BlochActionPlan},
        affine::{AffineGroup, AffineOp, PlaneGroup, Rational},
        support::FourierSupport,
        support_qualification::{QualifiedSupportGroup, SupportQualificationOptions},
    },
};
use std::f64::consts::TAU;

fn rod(lattice: Lattice2D, grid: Grid2D, raw: bool) -> Dielectric2D {
    let mut options = DielectricOptions::default();
    if raw {
        options.smoothing.mesh_size = 1;
    }
    Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid,
        &options,
    )
}
fn operator(
    d: Dielectric2D,
    lattice: Lattice2D,
    k: [f64; 2],
    pol: Polarization,
) -> ThetaOperator<CpuBackend<f64>> {
    ThetaOperator::new(
        CpuBackend::new(),
        d,
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}
fn support(lattice: Lattice2D, grid: Grid2D, k: [f64; 2], radius: f64) -> FourierSupport {
    FourierSupport::metric_ball(grid, lattice, k, TAU * radius, Default::default()).unwrap()
}

#[test]
fn hexagonal_raw_pencils_qualify_without_authorizing_the_full_rectangle() {
    let lattice = Lattice2D::hexagonal(1.0);
    for n in [12, 13] {
        let grid = Grid2D::new(n, n, 1.0, 1.0);
        for k in [[0.0; 2], [2.0 / 3.0, 1.0 / 3.0], [1.0, 0.0]] {
            let space = support(lattice, grid, k, 2.6);
            for pol in [Polarization::TM, Polarization::TE] {
                let group = PlaneGroup::P6m.group().little_group(k, 1e-12).unwrap();
                let mut full = operator(rod(lattice, grid, true), lattice, k, pol);
                let qualified =
                    QualifiedSupportGroup::qualify(&mut full, &space, group, Default::default())
                        .unwrap();
                qualified.validate_for(&full, &space).unwrap();
                let report = qualified.report();
                assert!(report.authorized && report.complete_column_probes);
                assert!(report.maximum_multiplier_defect < 1e-12);
                assert!(report.actions.iter().all(|a| a.sample_permutation_exact
                    && a.support_closed
                    && a.probe_count == space.len()
                    && a.operator_defect < 1e-12
                    && a.mass_defect < 1e-12));
            }
        }
        let rotation = PlaneGroup::P6.group().operations[1];
        assert!(matches!(
            BlochActionPlan::new(rotation, [0.0; 2], [0.0; 2], grid, Polarization::TM, 1e-12),
            Err(ActionError::FourierSupportNotClosed { .. })
        ));
    }
}

#[test]
fn analytic_hex_material_remains_an_explicit_negative_control() {
    let lattice = Lattice2D::hexagonal(1.0);
    let grid = Grid2D::new(17, 17, 1.0, 1.0);
    for pol in [Polarization::TM, Polarization::TE] {
        for k in [[0.0; 2], [2.0 / 3.0, 1.0 / 3.0]] {
            let space = support(lattice, grid, k, 3.5);
            let mut full = operator(rod(lattice, grid, false), lattice, k, pol);
            let group = PlaneGroup::P6m.group().little_group(k, 1e-12).unwrap();
            let report =
                QualifiedSupportGroup::qualify(&mut full, &space, group, Default::default())
                    .unwrap_err();
            assert!(!report.authorized);
            assert!(
                report
                    .rejection
                    .unwrap()
                    .contains("sampled scalar material")
            );
            assert!(report.actions.iter().any(|a| a.material_defect > 1e-5));
        }
    }
}

#[test]
fn square_chiral_material_keeps_complex_cyclic_labels_and_rejects_mirrors() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(17, 17, 1.0, 1.0);
    let space = support(lattice, grid, [0.0; 2], 3.1);
    let mut pos = [0.23, 0.11];
    let mut atoms = Vec::new();
    for _ in 0..4 {
        atoms.push(BasisAtom {
            pos,
            radius: 0.08,
            eps_inside: 8.9,
        });
        pos = [-pos[1], pos[0]];
    }
    let d = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms,
        },
        grid,
        &Default::default(),
    );
    for pol in [Polarization::TM, Polarization::TE] {
        let mut full = operator(d.clone(), lattice, [0.0; 2], pol);
        let p = QualifiedSupportGroup::qualify(
            &mut full,
            &space,
            PlaneGroup::P4.group(),
            Default::default(),
        )
        .unwrap();
        assert!(
            p.representation().irreps.iter().any(|d| d
                .matrices
                .iter()
                .flatten()
                .any(|z| z.im.abs() > 0.5))
        );
        let failure = QualifiedSupportGroup::qualify(
            &mut full,
            &space,
            PlaneGroup::P4m.group(),
            Default::default(),
        )
        .unwrap_err();
        assert!(failure.rejection.unwrap().contains("material"));
    }
}

#[test]
fn boundary_glide_qualifies_its_multiplier_only_on_a_compatible_sample_grid() {
    let lattice = Lattice2D::square(1.0);
    let k = [0.5, 0.0];
    for n in [15, 16] {
        let grid = Grid2D::new(n, n, 1.0, 1.0);
        let reciprocal = lattice.reciprocal();
        let eps = (0..grid.len())
            .map(|i| {
                let x = TAU * (i % n) as f64 / n as f64;
                let y = TAU * (i / n) as f64 / n as f64;
                3.0 + 0.2 * x.sin() * y.sin() + 0.1 * (2.0 * x).cos() * y.cos()
            })
            .collect();
        let d = Dielectric2D::from_sampled_epsilon(grid, reciprocal.b1, reciprocal.b2, eps, None);
        let space = support(lattice, grid, k, 2.6);
        for pol in [Polarization::TM, Polarization::TE] {
            let mut full = operator(d.clone(), lattice, k, pol);
            let result = QualifiedSupportGroup::qualify(
                &mut full,
                &space,
                PlaneGroup::Pg.group(),
                Default::default(),
            );
            if n % 2 == 1 {
                assert!(
                    result
                        .unwrap_err()
                        .rejection
                        .unwrap()
                        .contains("sample nodes")
                );
            } else {
                let p = result.unwrap();
                assert!(
                    p.representation()
                        .multiplier
                        .iter()
                        .flatten()
                        .any(|z| z.re < -0.99)
                );
                assert!(
                    p.representation().irreps.iter().any(|d| d
                        .matrices
                        .iter()
                        .flatten()
                        .any(|z| z.im.abs() > 0.5))
                );
            }
        }
    }
}

#[test]
fn qualification_rejects_changed_identity_metric_tensor_and_nonfinite_operator() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let space = support(lattice, grid, [0.0; 2], 2.6);
    let d = rod(lattice, grid, false);
    let mut full = operator(d.clone(), lattice, [0.0; 2], Polarization::TE);
    let p = QualifiedSupportGroup::qualify(
        &mut full,
        &space,
        PlaneGroup::P4m.group(),
        Default::default(),
    )
    .unwrap();
    p.validate_for(&full, &space).unwrap();
    let larger = support(lattice, grid, [0.0; 2], 3.5);
    assert!(p.validate_for(&full, &larger).is_err());
    let reciprocal = lattice.reciprocal();
    let changed = Dielectric2D::from_sampled_epsilon(
        grid,
        reciprocal.b1,
        reciprocal.b2,
        vec![2.0; grid.len()],
        None,
    );
    assert!(
        p.validate_for(
            &operator(changed, lattice, [0.0; 2], Polarization::TE),
            &space
        )
        .is_err()
    );
    assert!(
        p.validate_for(
            &operator(d.clone(), lattice, [0.0; 2], Polarization::TM),
            &space
        )
        .is_err()
    );
    for tensor in [
        [0.5, 0.0, 0.0, 1.0],
        [1.0, 3.0, 3.0, 1.0],
        [1e308, 0.0, 0.0, 1e308],
    ] {
        let d = Dielectric2D::from_sampled_epsilon(
            grid,
            reciprocal.b1,
            reciprocal.b2,
            vec![2.0; grid.len()],
            Some(vec![tensor; grid.len()]),
        );
        let mut full = operator(d, lattice, [0.0; 2], Polarization::TE);
        assert!(
            QualifiedSupportGroup::qualify(
                &mut full,
                &space,
                PlaneGroup::P4.group(),
                Default::default()
            )
            .is_err()
        );
    }
    let lattice = Lattice2D::hexagonal(1.0);
    let space = support(lattice, grid, [0.0; 2], 2.6);
    let mut full = operator(
        rod(lattice, grid, true),
        lattice,
        [0.0; 2],
        Polarization::TM,
    );
    let wrong_reflection = AffineOp::new([[-1, 0], [0, 1]], [Rational::ZERO; 2]).unwrap();
    let group = AffineGroup::from_generators(&[wrong_reflection], 2).unwrap();
    let failure =
        QualifiedSupportGroup::qualify(&mut full, &space, group, Default::default()).unwrap_err();
    assert!(failure.rejection.unwrap().contains("Cartesian metric"));
}

#[test]
fn malformed_groups_and_insufficient_budgets_reject_before_probe_work() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let space = support(lattice, grid, [0.0; 2], 2.6);
    let mut full = operator(
        rod(lattice, grid, false),
        lattice,
        [0.0; 2],
        Polarization::TM,
    );
    let mut malformed = PlaneGroup::P4.group();
    malformed.product_index[1][1] = 100;
    let antiunitary =
        AffineGroup::from_generators(&[AffineOp::IDENTITY.with_conjugation(true)], 2).unwrap();
    for group in [malformed, antiunitary] {
        let report = QualifiedSupportGroup::qualify(&mut full, &space, group, Default::default())
            .unwrap_err();
        assert!(report.actions.is_empty());
    }
    for options in [
        SupportQualificationOptions {
            max_estimated_payload_bytes: 1,
            ..Default::default()
        },
        SupportQualificationOptions {
            tolerance: f64::NAN,
            ..Default::default()
        },
        SupportQualificationOptions {
            max_group_order: 2,
            ..Default::default()
        },
    ] {
        let report =
            QualifiedSupportGroup::qualify(&mut full, &space, PlaneGroup::P4.group(), options)
                .unwrap_err();
        assert!(report.actions.is_empty());
    }
    let options = SupportQualificationOptions {
        complete_probe_dimension: 0,
        deterministic_probes: 3,
        ..Default::default()
    };
    let p =
        QualifiedSupportGroup::qualify(&mut full, &space, PlaneGroup::P4.group(), options).unwrap();
    assert!(!p.report().complete_column_probes);
    assert!(p.report().actions.iter().all(|a| a.probe_count == 3));
}
