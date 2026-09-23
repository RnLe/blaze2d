use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::{EigensolverConfig, iteration_total},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        reuse::SymmetryOptions,
        sectors::solve_conservative,
        shared::{SharedSolverConfig, solve_shared},
        trial::{GramMode, RitzMode},
    },
};

#[test]
fn shared_iterations_preserve_lowest_modes_and_sector_labels() {
    let lattice = Lattice2D::square(1.0);
    for (family, grid, k, atoms) in [
        (
            PlaneGroup::Pm,
            Grid2D::new(13, 13, 1.0, 1.0),
            [0.17, 0.0],
            vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.24,
                eps_inside: 8.9,
            }],
        ),
        (
            PlaneGroup::P4m,
            Grid2D::new(9, 9, 1.0, 1.0),
            [0.0; 2],
            vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.24,
                eps_inside: 8.9,
            }],
        ),
        (
            PlaneGroup::Pg,
            Grid2D::new(6, 5, 1.0, 1.0),
            [0.5, 0.0],
            vec![
                BasisAtom {
                    pos: [0.0, 0.2],
                    radius: 0.16,
                    eps_inside: 5.0,
                },
                BasisAtom {
                    pos: [0.5, -0.2],
                    radius: 0.16,
                    eps_inside: 5.0,
                },
            ],
        ),
        (
            PlaneGroup::Pgg,
            Grid2D::new(6, 6, 1.0, 1.0),
            [-0.5, -0.5],
            vec![],
        ),
    ] {
        for polarization in [Polarization::TM, Polarization::TE] {
            let material = Dielectric2D::from_geometry(
                &Geometry2D {
                    lattice,
                    eps_bg: 1.0,
                    atoms: atoms.clone(),
                },
                grid,
                &Default::default(),
            );
            let mut op = ThetaOperator::new(
                CpuBackend::<f64>::new(),
                material,
                polarization,
                lattice.reciprocal().fractional_to_cartesian(k),
            );
            let mut preconditioner = op.build_homogeneous_preconditioner_adaptive();
            let options = SymmetryOptions::default();
            let reference = solve_conservative(
                &mut op,
                &lattice,
                k,
                family.group(),
                &EigensolverConfig {
                    n_bands: 5,
                    tol: 1e-14,
                    max_iter: 500,
                    ..Default::default()
                },
                None,
                &options,
                Some(&mut preconditioner),
            )
            .unwrap();
            for (gram, ritz) in [
                (GramMode::Full, RitzMode::Full),
                (GramMode::Blocks, RitzMode::Full),
                (GramMode::Blocks, RitzMode::Blocks),
            ] {
                let config = SharedSolverConfig {
                    n_bands: 5,
                    gram,
                    ritz,
                    ..Default::default()
                };
                let before = iteration_total();
                let result = solve_shared(
                    &mut op,
                    &lattice,
                    k,
                    family.group(),
                    &config,
                    &options,
                    None,
                    Some(&mut preconditioner),
                )
                .unwrap_or_else(|e| panic!("{family:?} {polarization:?} {gram:?} {ritz:?}: {e}"));
                assert_eq!(iteration_total().wrapping_sub(before), result.report.rounds);
                assert_eq!(result.report.sector_ranks.iter().sum::<usize>(), grid.len());
                assert!(result.report.maximum_mass_defect < 1e-8);
                assert!(result.report.maximum_leakage < 1e-8);
                assert!(result.spectrum.passes(&options));
                if family == PlaneGroup::Pm {
                    assert!(result.report.rounds > 1);
                    assert!(result.report.maximum_within_block_offdiagonal_ratio > 0.01);
                    assert!(
                        result.report.maximum_trial_columns
                            > result.report.working_columns.iter().sum::<usize>()
                    );
                }
                for (a, b) in reference
                    .spectrum
                    .eigenvalues
                    .iter()
                    .zip(&result.spectrum.eigenvalues)
                {
                    assert!(
                        (a - b).abs() / a.abs().max(1.0) < 1e-7,
                        "{family:?} {polarization:?} {gram:?} {ritz:?}: {a} vs {b}"
                    );
                }
                assert_eq!(
                    result
                        .spectrum
                        .eigenvalues
                        .iter()
                        .filter(|&&v| v == 0.0)
                        .count(),
                    usize::from(k == [0.0; 2])
                );
            }
        }
    }
}

#[test]
fn shared_solver_rejects_excess_workspace_and_unsupported_policies() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(9, 9, 1.0, 1.0);
    let material = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![],
        },
        grid,
        &Default::default(),
    );
    let mut op = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        material,
        Polarization::TE,
        [0.0; 2],
    );
    let capped = SharedSolverConfig {
        max_workspace_bytes: 1,
        ..Default::default()
    };
    let rejected = solve_shared(
        &mut op,
        &lattice,
        [0.0; 2],
        PlaneGroup::P4m.group(),
        &capped,
        &Default::default(),
        None,
        None,
    );
    assert!(rejected.err().unwrap().contains("exceeds cap"));
    let partners = SymmetryOptions {
        reconstruct_partners: true,
        ..Default::default()
    };
    let rejected = solve_shared(
        &mut op,
        &lattice,
        [0.0; 2],
        PlaneGroup::P4m.group(),
        &Default::default(),
        &partners,
        None,
        None,
    );
    assert!(rejected.err().unwrap().contains("unsupported"));
}
