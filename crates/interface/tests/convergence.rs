use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::eigensolver::*;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
};

fn solve_mode(mode: usize, pol: Polarization, k: [f64; 2], max_iter: usize) -> EigensolverResult {
    let geometry = Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 1.0,
        atoms: vec![BasisAtom {
            pos: [0.0, 0.0],
            radius: 0.16,
            eps_inside: 8.9,
        }],
    };
    let grid = Grid2D::new(16, 16, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&geometry, grid, &DielectricOptions::default());
    let mut operator = ThetaOperator::new(CpuBackend::<f64>::new(), dielectric, pol, k);
    let mut preconditioner = operator.build_homogeneous_preconditioner_adaptive();
    let config = EigensolverConfig {
        n_bands: 8,
        tol: 1e-6,
        max_iter,
        ..Default::default()
    };
    let mut solver = Eigensolver::new(&mut operator, config, Some(&mut preconditioner), None);
    let result = match mode {
        0 => solver.solve(),
        1 => solver.solve_with_progress(|progress| assert!(progress.n_converged <= 8)),
        _ => solver.solve_with_diagnostics("convergence").result,
    };
    if result.converged {
        let hard = usize::from(k == [0.0, 0.0]);
        for i in 0..8 - hard {
            assert!(
                result.convergence.band_states[i] == BandState::Converged,
                "Band {i} was still active when the calculation reported convergence"
            );
        }
    }
    result
}

#[test]
fn ordinary_progress_and_diagnostic_solves_have_the_same_stopping_rule() {
    for pol in [Polarization::TM, Polarization::TE] {
        for k in [[0.0, 0.0], [2.1, 1.7]] {
            let ordinary = solve_mode(0, pol, k, 200);
            assert!(ordinary.converged);
            for mode in [1, 2] {
                let observed = solve_mode(mode, pol, k, 200);
                assert_eq!(ordinary.iterations, observed.iterations);
                assert_eq!(ordinary.converged, observed.converged);
                assert_eq!(ordinary.eigenvalues, observed.eigenvalues);
            }
        }
    }
}

#[test]
fn iteration_limit_does_not_certify_active_bands() {
    for mode in 0..3 {
        let result = solve_mode(mode, Polarization::TM, [0.0, 0.0], 1);
        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
    }
}
