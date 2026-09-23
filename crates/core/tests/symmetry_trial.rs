use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        trial::{GramMode, RitzMode, SharedSectorTrial},
    },
};
use num_complex::Complex64 as C;

fn candidates(grid: Grid2D, count: usize) -> Vec<Field2D> {
    (0..count)
        .map(|j| {
            Field2D::from_vec(
                grid,
                (0..grid.len())
                    .map(|i| {
                        let a = (i + 1) as f64;
                        let b = (j + 1) as f64;
                        C::new((a * b * 0.723).sin(), (a * (b + 0.317) * 1.31).cos())
                    })
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn shared_gram_and_ritz_match_on_ordinary_and_projective_trials() {
    for (group, grid, k, atoms) in [
        (
            PlaneGroup::Pm,
            Grid2D::new(9, 9, 1.0, 1.0),
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
        for pol in [Polarization::TM, Polarization::TE] {
            let lattice = Lattice2D::square(1.0);
            let backend = CpuBackend::<f64>::new();
            let d = Dielectric2D::from_geometry(
                &Geometry2D {
                    lattice,
                    eps_bg: 1.0,
                    atoms: atoms.clone(),
                },
                grid,
                &Default::default(),
            );
            let mut op = ThetaOperator::new(
                backend.clone(),
                d,
                pol,
                lattice.reciprocal().fractional_to_cartesian(k),
            );
            let trial = SharedSectorTrial::prepare(
                &mut op,
                &lattice,
                k,
                group.group(),
                &candidates(grid, 8),
                8,
                &Default::default(),
            )
            .unwrap();
            assert!(trial.report.projection_leakage < 1e-9);
            assert!(trial.report.mass_orthogonality_defect < 1e-8);
            for mass in [false, true] {
                let full = trial.project(&backend, GramMode::Full, mass);
                let block = trial.project(&backend, GramMode::Blocks, mass);
                let scale = full.matrix().iter().map(C::norm_sqr).sum::<f64>().sqrt();
                let error = full
                    .matrix()
                    .iter()
                    .zip(block.matrix())
                    .map(|(a, b)| (a - b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(error / scale < 1e-11, "{group:?} {pol:?} mass={mass}");
                assert!(full.cross_block_defect() < 1e-11);
            }
            let projected = trial.project(&backend, GramMode::Full, false);
            let full = projected.solve(RitzMode::Full).unwrap();
            let block = projected.solve(RitzMode::Blocks).unwrap();
            for (x, y) in full.eigenvalues.iter().zip(&block.eigenvalues) {
                assert!(
                    (x - y).abs() / x.abs().max(1.0) < 1e-10,
                    "{group:?} {pol:?}"
                );
            }
            // Compare invariant spans within each eigenvalue cluster, allowing
            // arbitrary rotations of a degenerate Ritz multiplet.
            let n = full.dim;
            for j in 0..n {
                let overlap: f64 = (0..n)
                    .filter(|&i| {
                        (full.eigenvalues[j] - block.eigenvalues[i]).abs()
                            < 1e-8 * full.eigenvalues[j].abs().max(1.0)
                    })
                    .map(|i| {
                        (0..n)
                            .map(|r| {
                                full.eigenvectors[r + n * j].conj() * block.eigenvectors[r + n * i]
                            })
                            .sum::<C>()
                            .norm_sqr()
                    })
                    .sum();
                assert!(
                    overlap > 1.0 - 1e-8,
                    "{group:?} {pol:?} mode {j}: {overlap}"
                );
            }
            assert_eq!(trial.reconstruct(&backend, &block).len(), n);
        }
    }
}

#[test]
fn shared_trial_rejects_unqualified_material_and_unsupported_precision() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(9, 9, 1.0, 1.0);
    let d = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.13, 0.07],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid,
        &Default::default(),
    );
    let mut op = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        d.clone(),
        Polarization::TE,
        [0.0; 2],
    );
    assert!(
        SharedSectorTrial::prepare(
            &mut op,
            &lattice,
            [0.0; 2],
            PlaneGroup::Pm.group(),
            &candidates(grid, 4),
            4,
            &Default::default()
        )
        .is_err()
    );
    let mut op32 = ThetaOperator::new(CpuBackend::<f32>::new(), d, Polarization::TE, [0.0; 2]);
    let error = SharedSectorTrial::prepare(
        &mut op32,
        &lattice,
        [0.0; 2],
        PlaneGroup::P1.group(),
        &candidates(grid, 4),
        4,
        &Default::default(),
    )
    .err()
    .unwrap();
    assert!(error.contains("f64"));
}
