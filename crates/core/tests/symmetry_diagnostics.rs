use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    drivers::bandstructure::{
        BandStructureJob, RunOptions, run_with_diagnostics_and_options, run_with_k_streaming,
    },
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{
        fourier::Arithmetic,
        reuse::{KReuse, SymmetryOptions},
        sectors::SectorPolicy,
        traversal::KTraversal,
    },
};

#[test]
fn requested_diagnostics_survive_symmetry_options_and_repeated_points() {
    let mut job = BandStructureJob {
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
        pol: Polarization::TM,
        k_path: vec![[0.0; 2], [0.17, 0.0], [0.0; 2]],
        eigensolver: EigensolverConfig {
            n_bands: 4,
            max_iter: 500,
            tol: 1e-12,
            ..Default::default()
        },
        dielectric: Default::default(),
    };
    let options = RunOptions {
        symmetry: SymmetryOptions {
            k_reuse: KReuse::TimeReversal,
            k_traversal: KTraversal::Nearest,
            sectors: SectorPolicy::ConservativeMirror,
            arithmetic: Arithmetic::PtRealTe,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = run_with_diagnostics_and_options(
        CpuBackend::<f64>::new(),
        &job,
        "symmetry_diagnostics",
        options.clone(),
    );
    assert_eq!(result.study.runs.len(), job.k_path.len());
    assert_eq!(result.study.notes.len(), 1);
    assert!(result.study.notes[0].contains("unrestricted spatial"));
    for (i, run) in result.study.runs.iter().enumerate() {
        assert!(!run.snapshots.is_empty());
        assert!(run.converged);
        assert_eq!(run.config.k_index, Some(i));
        assert_eq!(run.config.k_point, Some(job.k_path[i]));
    }
    let encoded = serde_json::to_string(&result.study).unwrap();
    let restored: blaze2d_core::diagnostics::ConvergenceStudy =
        serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored.notes, result.study.notes);
    job.eigensolver.record_diagnostics = true;
    run_with_k_streaming(CpuBackend::<f64>::new(), &job, options, |point| {
        assert!(point.iterations > 0 && point.symmetry.source_index.is_none());
        assert!(point.symmetry.sectors.is_none() && point.symmetry.coordinates.is_none());
        assert!(
            point
                .symmetry
                .rejected
                .iter()
                .any(|reason| reason.contains("per-iteration diagnostics"))
        );
    });
}
