use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::EigensolverConfig,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        action::BlochActionPlan,
        affine::{AffineGroup, AffineOp, PlaneGroup, Rational},
        fourier::Arithmetic,
        reuse::SymmetryOptions,
        sectors::{SectorPolicy, SectorResult, solve_conservative},
    },
};

fn operator(
    grid: Grid2D,
    pol: Polarization,
    k: [f64; 2],
    center: [f64; 2],
) -> ThetaOperator<CpuBackend<f64>> {
    let lattice = Lattice2D::square(1.0);
    let material = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: center,
                radius: 0.23,
                eps_inside: 8.9,
            }],
        },
        grid,
        &Default::default(),
    );
    ThetaOperator::new(
        CpuBackend::new(),
        material,
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}
fn config(bands: usize) -> EigensolverConfig {
    EigensolverConfig {
        n_bands: bands,
        tol: 1e-14,
        max_iter: 500,
        ..Default::default()
    }
}
fn agree(a: &SectorResult, b: &SectorResult) {
    assert_eq!(a.spectrum.fields.len(), b.spectrum.fields.len());
    assert_eq!(
        a.report.sum_sector_dimensions,
        b.report.sum_sector_dimensions
    );
    assert_eq!(a.report.kernel_owners, b.report.kernel_owners);
    for (&x, &y) in a.spectrum.eigenvalues.iter().zip(&b.spectrum.eigenvalues) {
        assert!((x - y).abs() / x.abs().max(1.0) < 1e-7, "{x} versus {y}");
    }
}
fn pt_invariant(result: &SectorResult, pt: AffineOp, k: [f64; 2], pol: Polarization) {
    let grid = result.spectrum.fields[0].grid();
    let action = BlochActionPlan::new(pt, k, k, grid, pol, 1e-12).unwrap();
    for v in &result.spectrum.fields {
        let mut out = Field2D::zeros(grid);
        action.apply_spatial(v, &mut out);
        let error = v
            .as_slice()
            .iter()
            .zip(out.as_slice())
            .map(|(x, y)| (*x - *y).norm_sqr())
            .sum::<f64>();
        let scale = v.as_slice().iter().map(|x| x.norm_sqr()).sum::<f64>();
        assert!(
            (error / scale).sqrt() < 1e-8,
            "exported field left PT gauge"
        );
    }
}

#[test]
fn complete_fourier_and_pt_sector_spectra_preserve_complex_characters_and_partners() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(5, 5, 1.0, 1.0);
    let k = [0.0; 2];
    for pol in [Polarization::TM, Polarization::TE] {
        let mut op = operator(grid, pol, k, [0.0; 2]);
        let reference = solve_conservative(
            &mut op,
            &lattice,
            k,
            PlaneGroup::P1.group(),
            &config(grid.len()),
            None,
            &Default::default(),
            None,
        )
        .unwrap();
        for group in [
            PlaneGroup::Pm,
            PlaneGroup::P2,
            PlaneGroup::P4,
            PlaneGroup::P4m,
        ] {
            for partners in [false, true] {
                for arithmetic in [Arithmetic::FourierComplex, Arithmetic::PtRealTe] {
                    let options = SymmetryOptions {
                        arithmetic,
                        reconstruct_partners: partners,
                        ..Default::default()
                    };
                    let result = solve_conservative(
                        &mut op,
                        &lattice,
                        k,
                        group.group(),
                        &config(grid.len()),
                        None,
                        &options,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("{pol:?} {group:?} {arithmetic:?} {partners}: {e}"));
                    agree(&reference, &result);
                    assert!(result.spectrum.passes(&options));
                    assert!(
                        result
                            .report
                            .sectors
                            .iter()
                            .all(|s| s.dense_complete && s.coordinates.is_some())
                    );
                    if arithmetic == Arithmetic::PtRealTe {
                        assert!(
                            result.report.sectors.iter().any(|s| s
                                .coordinates
                                .as_ref()
                                .unwrap()
                                .representation
                                == Arithmetic::PtRealTe)
                        );
                        for s in &result.report.sectors {
                            let c = s.coordinates.as_ref().unwrap();
                            if c.representation == Arithmetic::PtRealTe {
                                assert_eq!(s.vector_column_bytes, grid.len().div_ceil(2) * 16);
                                assert!(s.pt_projector_defect.unwrap() < 1e-10);
                            } else {
                                assert!(
                                    s.coordinate_rejection
                                        .as_ref()
                                        .unwrap()
                                        .contains("not invariant")
                                );
                                assert!(s.pt_projector_defect.unwrap() > 1e-3);
                            }
                        }
                        if group == PlaneGroup::P4 {
                            assert_eq!(
                                result
                                    .report
                                    .sectors
                                    .iter()
                                    .filter(|s| s.coordinate_rejection.is_some())
                                    .count(),
                                2
                            );
                        }
                        if !partners && matches!(group, PlaneGroup::Pm | PlaneGroup::P2) {
                            pt_invariant(
                                &result,
                                PlaneGroup::P2.group().operations[1].with_conjugation(true),
                                k,
                                pol,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn iterative_pt_sectors_keep_warm_rank_guards_and_one_gamma_kernel() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    for pol in [Polarization::TM, Polarization::TE] {
        let mut warm = None;
        for k in [[0.0; 2], [0.17, 0.0], [0.0; 2]] {
            let mut op = operator(grid, pol, k, [0.0; 2]);
            let mut p = op.build_homogeneous_preconditioner_adaptive();
            let mut options = SymmetryOptions {
                sectors: SectorPolicy::AdaptiveGroup,
                sector_guard_budget: Some(2),
                ..Default::default()
            };
            let reference = solve_conservative(
                &mut op,
                &lattice,
                k,
                PlaneGroup::Pm.group(),
                &config(8),
                None,
                &options,
                Some(&mut p),
            )
            .unwrap();
            options.arithmetic = Arithmetic::PtRealTe;
            let result = solve_conservative(
                &mut op,
                &lattice,
                k,
                PlaneGroup::Pm.group(),
                &config(8),
                warm.as_deref(),
                &options,
                Some(&mut p),
            )
            .unwrap_or_else(|e| panic!("{pol:?} {k:?}: {e}"));
            agree(&reference, &result);
            assert!(result.spectrum.passes(&options));
            assert!(result.report.sectors.iter().all(|s| !s.dense_complete
                && s.coordinates.as_ref().unwrap().representation == Arithmetic::PtRealTe));
            assert_eq!(result.report.kernel_owners, usize::from(k == [0.0; 2]));
            pt_invariant(
                &result,
                PlaneGroup::P2.group().operations[1].with_conjugation(true),
                k,
                pol,
            );
            // A dependent block must not become duplicate physical modes when
            // its real/imaginary parts are offered to packed initialization.
            let fields = result.spectrum.fields;
            warm = Some(fields.iter().chain(&fields).cloned().collect::<Vec<_>>());
        }
    }
}

#[test]
fn shifted_pt_gauge_and_noninversion_material_keep_valid_mirror_sectors() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(12, 13, 1.0, 1.0);
    let k = [0.17, 0.0];
    let pt = AffineOp::new([[-1, 0], [0, -1]], [Rational::HALF, Rational::ZERO])
        .unwrap()
        .with_conjugation(true);
    for pol in [Polarization::TM, Polarization::TE] {
        let mut op = operator(grid, pol, k, [0.25, 0.0]);
        let mut p = op.build_homogeneous_preconditioner_adaptive();
        let reference = solve_conservative(
            &mut op,
            &lattice,
            k,
            PlaneGroup::Pm.group(),
            &config(5),
            None,
            &Default::default(),
            Some(&mut p),
        )
        .unwrap();
        let mut options = SymmetryOptions {
            arithmetic: Arithmetic::PtRealTe,
            ..Default::default()
        };
        let fallback = solve_conservative(
            &mut op,
            &lattice,
            k,
            PlaneGroup::Pm.group(),
            &config(5),
            None,
            &options,
            Some(&mut p),
        )
        .unwrap();
        agree(&reference, &fallback);
        assert!(fallback.report.sectors.iter().all(|s| {
            s.coordinate_rejection
                .as_ref()
                .is_some_and(|r| r.contains("PT qualification failed"))
                && s.coordinates.as_ref().unwrap().representation == Arithmetic::FourierComplex
        }));
        options.pt_operation = Some(pt);
        let packed = solve_conservative(
            &mut op,
            &lattice,
            k,
            PlaneGroup::Pm.group(),
            &config(5),
            None,
            &options,
            Some(&mut p),
        )
        .unwrap();
        agree(&reference, &packed);
        assert!(
            packed
                .report
                .sectors
                .iter()
                .all(|s| s.coordinate_rejection.is_none())
        );
        pt_invariant(&packed, pt, k, pol);
    }
}

#[test]
fn complex_glide_labels_can_have_pt_real_projectors_at_the_zone_boundary() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(6, 5, 1.0, 1.0);
    let k = [0.5, 0.0];
    // Homogeneous material has PT and the glide. Its boundary glide characters
    // are nonreal, but the Bloch phase cancels that phase in each projector.
    // Reality must be tested on the gauged projector, not on its label alone.
    let representation = blaze2d_core::symmetry::representation::RepresentationPlan::projective(
        PlaneGroup::Pg.group(),
        k,
    )
    .unwrap();
    assert!(
        representation
            .irreps
            .iter()
            .all(|irrep| (0..representation.group.order())
                .any(|g| irrep.character(g).im.abs() > 0.9))
    );
    for pol in [Polarization::TM, Polarization::TE] {
        let material = Dielectric2D::from_geometry(
            &Geometry2D {
                lattice,
                eps_bg: 2.0,
                atoms: vec![],
            },
            grid,
            &Default::default(),
        );
        let mut op = ThetaOperator::new(
            CpuBackend::<f64>::new(),
            material,
            pol,
            lattice.reciprocal().fractional_to_cartesian(k),
        );
        let reference = solve_conservative(
            &mut op,
            &lattice,
            k,
            AffineGroup::from_generators(&[], 1).unwrap(),
            &config(grid.len()),
            None,
            &Default::default(),
            None,
        )
        .unwrap();
        let options = SymmetryOptions {
            arithmetic: Arithmetic::PtRealTe,
            ..Default::default()
        };
        let result = solve_conservative(
            &mut op,
            &lattice,
            k,
            PlaneGroup::Pg.group(),
            &config(grid.len()),
            None,
            &options,
            None,
        )
        .unwrap();
        agree(&reference, &result);
        assert_eq!(result.report.kernel_owners, 0);
        assert!(
            result
                .report
                .sectors
                .iter()
                .all(|s| s.coordinate_rejection.is_none()
                    && s.coordinates.as_ref().unwrap().representation == Arithmetic::PtRealTe)
        );
        pt_invariant(
            &result,
            PlaneGroup::P2.group().operations[1].with_conjugation(true),
            k,
            pol,
        );
    }
}
