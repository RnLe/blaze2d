#![cfg(feature = "profiling")]

use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    drivers::bandstructure::{BandStructureJob, RunOptions, run_with_k_streaming},
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    profiler,
};

#[test]
fn complete_band_runs_close_every_profile_scope_and_can_reset() {
    let job = BandStructureJob {
        geom: Geometry2D {
            lattice: Lattice2D::square(1.0), eps_bg: 1.0,
            atoms: vec![BasisAtom { pos: [0.0; 2], radius: 0.2, eps_inside: 8.9 }],
        },
        grid: Grid2D::new(7, 7, 1.0, 1.0),
        pol: Polarization::TM,
        k_path: vec![[0.0; 2], [0.13, 0.21]],
        eigensolver: EigensolverConfig { n_bands: 3, max_iter: 200, ..Default::default() },
        dielectric: Default::default(),
    };
    for _ in 0..2 {
        profiler::reset();
        run_with_k_streaming(CpuBackend::<f64>::new(), &job, RunOptions::default(), |_| {});
        assert_eq!(profiler::get_counters().get("profiler.unbalanced_timers"), None);
        let profile: serde_json::Value = serde_json::from_str(&profiler::get_profile_json()).unwrap();
        assert_eq!(profile["run.wall"]["calls"], 1);
        assert_eq!(profile["run.setup"]["calls"], 1);
        assert_eq!(profile["run.operator_diagnostics"]["calls"], 1);
        assert_eq!(profile["run.operator_setup"]["calls"], 2);
        assert_eq!(profile["run.preconditioner_setup"]["calls"], 2);
        profiler::reset();
    }
}
