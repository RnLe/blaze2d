//! Complete first-point cost, including a separate coarse solve when requested.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    eigensolver::refine::certify_without_refinement,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{coarse::prolongate_fourier, seed::rank_aware_seed},
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("gamma");
    let k = match case {
        "gamma" => [0.0; 2],
        "generic" => [0.13, 0.17],
        _ => panic!("expected gamma or generic"),
    };
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(31);
    let bands: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    let pol = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("expected TM or TE"),
    };
    let mode = args.get(5).map(String::as_str).unwrap_or("random");
    assert!(matches!(mode, "random" | "coarse"));
    let tolerance: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    assert!(tolerance.is_finite() && tolerance > 0.0);
    assert!(n >= 7 && bands > 0);
    let mut coarse_n = n / 2;
    if coarse_n % 2 == 0 {
        coarse_n -= 1;
    }
    assert!(coarse_n >= 3 && bands + 2 < coarse_n * coarse_n);
    let backend = CpuBackend::<f64>::new();
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
    let fine = Grid2D::new(n, n, 1.0, 1.0);
    let coarse = Grid2D::new(coarse_n, coarse_n, 1.0, 1.0);
    let config = SingleSolveJob::new(bands)
        .with_tolerance(tolerance)
        .with_max_iterations(500);
    let make = |grid| {
        ThetaOperator::new(
            backend.clone(),
            Dielectric2D::from_geometry(&geom, grid, &Default::default()),
            pol,
            lattice.reciprocal().fractional_to_cartesian(k),
        )
    };
    let start = std::time::Instant::now();
    let mut op = make(fine);
    let mut p = op.build_homogeneous_preconditioner_adaptive();
    let mut coarse_iterations = 0;
    let mut coarse_report = None;
    let seed = if mode == "coarse" {
        let coarse_start = std::time::Instant::now();
        let mut low_op = make(coarse);
        let mut low_p = low_op.build_homogeneous_preconditioner_adaptive();
        let low = solve(&mut low_op, Some(&mut low_p), &config);
        coarse_iterations = low.iterations;
        let interpolation = prolongate_fourier(&backend, &low.eigenvectors, fine).unwrap();
        let prepared = rank_aware_seed(&mut op, &interpolation, 1e-9).unwrap();
        coarse_report = Some(
            json!({"grid":[coarse_n,coarse_n],"iterations":low.iterations,"converged":low.converged,
            "residuals":low.final_residuals,"orthogonality":low.b_orthogonality_defect,
            "setup_solve_interpolation_and_preparation_seconds":coarse_start.elapsed().as_secs_f64()}),
        );
        Some(prepared)
    } else {
        None
    };
    let result = if let Some(seed) = &seed {
        solve_with_warmstart(&mut op, Some(&mut p), &config, &seed.fields)
    } else {
        solve(&mut op, Some(&mut p), &config)
    };
    let cert = certify_without_refinement(&mut op, &result.eigenvalues, &result.eigenvectors);
    let spectrum = vec![
        result
            .eigenvalues
            .iter()
            .map(|v| v.max(0.0).sqrt())
            .collect::<Vec<_>>(),
    ];
    let seconds = start.elapsed().as_secs_f64();
    let rss_kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        });
    println!(
        "{}",
        json!({"case":case,"mode":mode,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),"precision":"f64",
        "tolerance":tolerance,"seconds":seconds,"peak_rss_kib":rss_kib,"coarse":coarse_report,"predictor":seed.map(|s|s.report),
        "spectrum":spectrum,"points":[{"k":k,"iterations":coarse_iterations+result.iterations,"fine_iterations":result.iterations,
            "converged":result.converged,"residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,
            "b_orthogonality_defect":cert.b_orthogonality_defect,"field_count":result.eigenvectors.len()}],
        "counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),
        "scope":"fine first-point solve including all coarse setup, solve, prolongation, target preparation and final certification; no serialized field export"})
    );
}
