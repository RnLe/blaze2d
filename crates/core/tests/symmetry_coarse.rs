use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{coarse::prolongate_fourier, seed::rank_aware_seed},
};
use num_complex::Complex64 as C;

fn wave(grid: Grid2D, mode: [i32; 2]) -> Field2D {
    Field2D::from_vec(
        grid,
        (0..grid.len())
            .map(|j| {
                C::from_polar(
                    1.0,
                    std::f64::consts::TAU
                        * (mode[0] as f64 * (j % grid.nx) as f64 / grid.nx as f64
                            + mode[1] as f64 * (j / grid.nx) as f64 / grid.ny as f64),
                )
            })
            .collect(),
    )
}

#[test]
fn zero_padding_preserves_odd_and_even_signed_modes_and_amplitudes() {
    let backend = CpuBackend::<f64>::new();
    for (coarse, fine, modes) in [
        (
            Grid2D::new(3, 5, 1.0, 1.0),
            Grid2D::new(7, 11, 1.0, 1.0),
            [[0, 0], [1, -2], [-1, 2]],
        ),
        (
            Grid2D::new(4, 6, 1.0, 1.0),
            Grid2D::new(9, 13, 1.0, 1.0),
            [[0, 0], [2, 3], [-1, -2]],
        ),
    ] {
        for mode in modes {
            let source = wave(coarse, mode);
            let expected = wave(fine, mode);
            let interpolated = prolongate_fourier(&backend, &[source], fine).unwrap();
            let max_error = interpolated[0]
                .as_slice()
                .iter()
                .zip(expected.as_slice())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            assert!(max_error < 2e-14, "{mode:?}: {max_error:e}");
        }
    }
    let field = wave(Grid2D::new(4, 4, 1.0, 1.0), [1, 1]);
    assert!(prolongate_fourier(&backend, &[field.clone()], Grid2D::new(3, 3, 1.0, 1.0)).is_err());
    assert!(
        prolongate_fourier(
            &backend,
            &[field.clone()],
            Grid2D::new_centered(9, 9, 1.0, 1.0)
        )
        .is_err()
    );
    assert!(prolongate_fourier(&backend, &[field], Grid2D::new(9, 9, 2.0, 1.0)).is_err());
}

#[test]
fn coarse_seed_keeps_the_fine_operator_spectrum_and_gamma_kernel() {
    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: vec![BasisAtom {
            pos: [0.0; 2],
            radius: 0.2,
            eps_inside: 8.9,
        }],
    };
    let coarse = Grid2D::new(7, 7, 1.0, 1.0);
    let fine = Grid2D::new(13, 13, 1.0, 1.0);
    let config = SingleSolveJob::new(6)
        .with_tolerance(1e-14)
        .with_max_iterations(500);
    for pol in [Polarization::TM, Polarization::TE] {
        let backend = CpuBackend::<f64>::new();
        let make = |grid| {
            ThetaOperator::new(
                backend.clone(),
                Dielectric2D::from_geometry(&geom, grid, &Default::default()),
                pol,
                [0.0; 2],
            )
        };
        let mut coarse_op = make(coarse);
        let mut p = coarse_op.build_homogeneous_preconditioner_adaptive();
        let low = solve(&mut coarse_op, Some(&mut p), &config);
        assert!(low.converged);
        let mut fine_op = make(fine);
        let mut p = fine_op.build_homogeneous_preconditioner_adaptive();
        let reference = solve(&mut fine_op, Some(&mut p), &config);
        let lifted = prolongate_fourier(&backend, &low.eigenvectors, fine).unwrap();
        let seed = rank_aware_seed(&mut fine_op, &lifted, 1e-9).unwrap();
        let corrected = solve_with_warmstart(&mut fine_op, Some(&mut p), &config, &seed.fields);
        assert!(corrected.converged);
        assert!(corrected.final_residuals.iter().skip(1).all(|r| *r < 1e-5));
        assert!(corrected.b_orthogonality_defect < 1e-8);
        assert!(corrected.eigenvalues[0].abs() < 1e-8);
        let cert = blaze2d_core::eigensolver::refine::certify_without_refinement(
            &mut fine_op,
            &corrected.eigenvalues,
            &corrected.eigenvectors,
        );
        assert!(cert.absolute_residuals[0] < 1e-7);
        for (a, b) in reference.eigenvalues.iter().zip(corrected.eigenvalues) {
            assert!((a - b).abs() / a.abs().max(1.0) < 1e-7);
        }
    }
}
