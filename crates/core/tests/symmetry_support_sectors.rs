use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup, support::FourierSupport, support_qualification::QualifiedSupportGroup,
        support_sectors::SupportSectorPlan,
    },
};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

fn setup(
    name: PlaneGroup,
    k: [f64; 2],
    pol: Polarization,
    radius: f64,
) -> (
    ThetaOperator<CpuBackend<f64>>,
    FourierSupport,
    QualifiedSupportGroup,
) {
    let lattice = if matches!(
        name,
        PlaneGroup::P3 | PlaneGroup::P3m1 | PlaneGroup::P31m | PlaneGroup::P6 | PlaneGroup::P6m
    ) {
        Lattice2D::hexagonal(1.0)
    } else {
        Lattice2D::square(1.0)
    };
    let grid = Grid2D::new(12, 12, 1.0, 1.0);
    let reciprocal = lattice.reciprocal();
    let d = Dielectric2D::from_sampled_epsilon(
        grid,
        reciprocal.b1,
        reciprocal.b2,
        vec![3.7; grid.len()],
        None,
    );
    let mut full = ThetaOperator::new(
        CpuBackend::new(),
        d,
        pol,
        reciprocal.fractional_to_cartesian(k),
    );
    let support =
        FourierSupport::metric_ball(grid, lattice, k, TAU * radius, Default::default()).unwrap();
    let group = name.group().little_group(k, 1e-12).unwrap();
    let qualified =
        QualifiedSupportGroup::qualify(&mut full, &support, group, Default::default()).unwrap();
    (full, support, qualified)
}

fn verify_complete_basis(plan: &SupportSectorPlan, m: usize) {
    let mut complete = Vec::new();
    let probe: Vec<_> = (0..m)
        .map(|i| C::new((i as f64 + 0.5).sin(), (i as f64 + 0.75).cos()))
        .collect();
    for (s, descriptor) in plan.descriptors().enumerate() {
        assert_eq!(descriptor.arithmetic, "complex_f64");
        let basis = plan.basis(s);
        let mut small = vec![C::ZERO; basis.rank()];
        let mut full = vec![C::ZERO; m];
        let mut projected = full.clone();
        let mut scratch = full.clone();
        basis.restrict(&probe, &mut small);
        basis.expand(&small, &mut full);
        plan.project(s, &probe, &mut projected, &mut scratch);
        for (a, b) in full.iter().zip(&projected) {
            assert!((*a - *b).norm() < 3e-12);
        }
        for j in 0..basis.rank() {
            small.fill(C::ZERO);
            small[j] = C::ONE;
            basis.expand(&small, &mut full);
            basis.restrict(&full, &mut small);
            for (i, &z) in small.iter().enumerate() {
                assert!((z - if i == j { C::ONE } else { C::ZERO }).norm() < 3e-12);
            }
            for row in 0..descriptor.partner_rows {
                let mut output = vec![C::ZERO; m];
                plan.reconstruct_partner(s, row, &full, &mut output, &mut scratch)
                    .unwrap();
                complete.push(output);
            }
        }
    }
    assert_eq!(complete.len(), m);
    for i in 0..m {
        for j in 0..m {
            let dot: C = complete[i]
                .iter()
                .zip(&complete[j])
                .map(|(x, y)| x.conj() * y)
                .sum();
            assert!(
                (dot - if i == j { C::ONE } else { C::ZERO }).norm() < 3e-12,
                "{i} {j}: {dot}"
            );
        }
    }
}

#[test]
fn complete_orbit_bases_cover_all_plane_groups_with_both_partner_policies() {
    for name in PlaneGroup::ALL {
        for pol in [Polarization::TM, Polarization::TE] {
            let (full, space, qualified) = setup(name, [0.0; 2], pol, 1.6);
            for partners in [false, true] {
                let plan =
                    SupportSectorPlan::new(qualified.clone(), &full, &space, partners, 1 << 28)
                        .unwrap();
                verify_complete_basis(&plan, space.len());
                assert_eq!(
                    plan.descriptors()
                        .filter(|s| s.owns_analytic_kernel)
                        .count(),
                    1
                );
                let owner = plan.descriptors().find(|s| s.owns_analytic_kernel).unwrap();
                let irrep = &plan.group().representation().irreps[owner.irrep];
                assert_eq!(irrep.dimension, 1);
                for (g, d) in plan.group().actions().iter().zip(&irrep.matrices) {
                    let expected = if pol == Polarization::TM {
                        1.0
                    } else {
                        g.operation().determinant() as f64
                    };
                    assert!((d[0] - C::new(expected, 0.0)).norm() < 3e-12);
                }
            }
        }
    }
}

#[test]
fn folded_boundary_and_projective_partner_bases_retain_full_multiplicity() {
    for (name, k) in [
        (PlaneGroup::P6m, [2.0 / 3.0, 1.0 / 3.0]),
        (PlaneGroup::P6m, [1.0, 0.0]),
        (PlaneGroup::P4m, [0.5, 0.5]),
        (PlaneGroup::Pg, [0.5, 0.0]),
        (PlaneGroup::Pgg, [0.5, 0.0]),
    ] {
        for pol in [Polarization::TM, Polarization::TE] {
            let (full, space, qualified) = setup(name, k, pol, 2.1);
            for partners in [false, true] {
                let plan =
                    SupportSectorPlan::new(qualified.clone(), &full, &space, partners, 1 << 28)
                        .unwrap();
                verify_complete_basis(&plan, space.len());
                assert_eq!(
                    plan.descriptors()
                        .filter(|s| s.owns_analytic_kernel)
                        .count(),
                    usize::from(k == [1.0, 0.0])
                );
                if name == PlaneGroup::Pgg && partners {
                    assert!(
                        plan.descriptors()
                            .any(|s| s.partner_rows == 2 && s.rank > 0)
                    );
                }
            }
        }
    }
}

#[test]
fn nonreal_cyclic_and_glide_rows_stay_complex_and_plan_admission_is_bounded() {
    for (name, k) in [
        (PlaneGroup::P3, [0.0; 2]),
        (PlaneGroup::P4, [0.0; 2]),
        (PlaneGroup::P6, [0.0; 2]),
        (PlaneGroup::Pg, [0.5, 0.0]),
    ] {
        let (full, space, qualified) = setup(name, k, Polarization::TM, 2.1);
        assert!(SupportSectorPlan::new(qualified.clone(), &full, &space, true, 1).is_err());
        let plan = SupportSectorPlan::new(qualified, &full, &space, true, 1 << 28).unwrap();
        assert!(
            plan.descriptors()
                .any(|s| s.representation_has_nonreal_entries && s.rank > 0)
        );
        assert!(plan.descriptors().all(|s| s.arithmetic == "complex_f64"));
        plan.validate_for(&full, &space).unwrap();
        let changed = FourierSupport::metric_ball(
            space.grid(),
            space.lattice(),
            k,
            TAU * 2.8,
            Default::default(),
        )
        .unwrap();
        assert!(plan.validate_for(&full, &changed).is_err());
    }
}
