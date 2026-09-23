use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    drivers::bandstructure::{BandStructureJob, RunOptions, run_with_k_streaming},
    eigensolver::{EigensolverConfig, refine::certify_without_refinement},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        reuse::{KReuse, SymmetryOptions},
        traversal::KTraversal,
    },
};

fn job(pol: Polarization) -> BandStructureJob {
    BandStructureJob {
        geom: Geometry2D {
            lattice: Lattice2D::square(1.0),
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid: Grid2D::new(9, 9, 1.0, 1.0),
        pol,
        k_path: vec![
            [0.11, 0.17],
            [0.39, 0.27],
            [-0.17, 0.11],
            [0.12, 0.17],
            [-0.39, -0.27],
            [0.11, 0.17],
        ],
        eigensolver: EigensolverConfig {
            n_bands: 4,
            max_iter: 500,
            tol: 1e-14,
            ..Default::default()
        },
        dielectric: DielectricOptions::default(),
    }
}

#[test]
fn traversal_restores_stream_order_fields_certificates_and_cache_indices() {
    for pol in [Polarization::TM, Polarization::TE] {
        let job = job(pol);
        for tracking in [false, true] {
            let mut options = RunOptions {
                reuse_gamma: false,
                disable_band_tracking: !tracking,
                retain_eigenvectors: true,
                symmetry: SymmetryOptions {
                    k_reuse: KReuse::TimeReversal,
                    ..Default::default()
                },
                ..Default::default()
            };
            let reference =
                run_with_k_streaming(CpuBackend::<f64>::new(), &job, options.clone(), |_| {});
            options.symmetry.k_traversal = KTraversal::Nearest;
            let mut points = Vec::new();
            let result =
                run_with_k_streaming(CpuBackend::<f64>::new(), &job, options, |p| points.push(p));
            assert_eq!(result.k_path, job.k_path);
            assert_eq!(result.distances, reference.distances);
            assert_eq!(
                points.iter().map(|p| p.k_index).collect::<Vec<_>>(),
                (0..job.k_path.len()).collect::<Vec<_>>()
            );
            assert!(
                points
                    .iter()
                    .any(|p| p.symmetry.traversal.as_ref().unwrap().solve_position != p.k_index)
            );
            for (i, p) in points.iter().enumerate() {
                assert_eq!(p.k_point, job.k_path[i]);
                assert_eq!(p.distance, result.distances[i]);
                assert_eq!(p.omegas, result.bands[i]);
                assert!(p.converged);
                let mut a = p.omegas.clone();
                let mut b = reference.bands[i].clone();
                a.sort_by(f64::total_cmp);
                b.sort_by(f64::total_cmp);
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() / x.abs().max(1.0) < 1e-6);
                }
                let d = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);
                let mut op = ThetaOperator::new(
                    CpuBackend::<f64>::new(),
                    d,
                    pol,
                    job.geom
                        .lattice
                        .reciprocal()
                        .fractional_to_cartesian(p.k_point),
                );
                let values: Vec<_> = p.omegas.iter().map(|w| w * w).collect();
                let cert =
                    certify_without_refinement(&mut op, &values, p.eigenvectors.as_ref().unwrap());
                assert!(cert.residuals.iter().all(|r| *r < 1e-5));
                assert!(cert.b_orthogonality_defect < 1e-8);
                for (actual, reported) in cert.residuals.iter().zip(&p.residuals) {
                    assert!((actual - reported).abs() < 1e-10);
                }
                if let Some(source) = p.symmetry.source_index {
                    let source_position = points[source]
                        .symmetry
                        .traversal
                        .as_ref()
                        .unwrap()
                        .solve_position;
                    assert!(
                        source_position < p.symmetry.traversal.as_ref().unwrap().solve_position
                    );
                    let mapped = p.symmetry.operation.map_or(points[source].k_point, |g| {
                        g.map_wavevector(points[source].k_point)
                    });
                    for axis in 0..2 {
                        let d = mapped[axis] - p.k_point[axis];
                        assert!((d - d.round()).abs() < 1e-12);
                    }
                }
            }
        }
    }
}

#[test]
fn traversal_budget_rejects_before_solving_and_tracking_can_hide_fields() {
    let job = job(Polarization::TM);
    let mut options = RunOptions {
        reuse_gamma: false,
        retain_eigenvectors: true,
        symmetry: SymmetryOptions {
            k_traversal: KTraversal::Nearest,
            max_traversal_field_bytes: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut points = Vec::new();
    run_with_k_streaming(CpuBackend::<f64>::new(), &job, options.clone(), |p| {
        points.push(p)
    });
    assert!(points.iter().all(|p| {
        p.symmetry.traversal.is_none()
            && p.symmetry
                .rejected
                .iter()
                .any(|r| r.contains("exceeds budget"))
    }));
    assert!(points.iter().all(|p| p.symmetry.solver_attempts.len() == 1));
    options.symmetry.max_traversal_field_bytes = 64 * 1024 * 1024;
    options.retain_eigenvectors = false;
    run_with_k_streaming(CpuBackend::<f64>::new(), &job, options, |p| {
        assert!(p.symmetry.traversal.is_some());
        assert!(p.eigenvectors.is_none());
    });
}
