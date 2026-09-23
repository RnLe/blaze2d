//! Precision policy changes on an unchanged physical and finite-grid target.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    eigensolver::{iteration_total, refine::certify_without_refinement},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::precision::{F32SeedConfig, prepare_f32_seed},
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("generic");
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
    let mode = args.get(5).map(String::as_str).unwrap_or("f64");
    assert!(matches!(mode, "f64" | "f32" | "mixed16" | "mixed64"));
    let tolerance: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    assert!(tolerance.is_finite() && tolerance > 0.0);
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let lattice = Lattice2D::square(1.0);
    let geometry = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: vec![BasisAtom {
            pos: [0.0; 2],
            radius: 0.2,
            eps_inside: 8.9,
        }],
    };
    let job = SingleSolveJob::new(bands)
        .with_tolerance(tolerance)
        .with_max_iterations(500);
    let start = std::time::Instant::now();
    let iteration_start = iteration_total();
    let dielectric = Dielectric2D::from_geometry(&geometry, grid, &Default::default());
    let mut target = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        dielectric.clone(),
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    );
    let mut seed_report = None;
    let mut rejection = None;
    let mut preparation_seconds = 0.0;
    let result = if mode == "f32" {
        let mut low = ThetaOperator::new(CpuBackend::<f32>::new(), dielectric, pol, target.bloch());
        let mut preconditioner = low.build_homogeneous_preconditioner_adaptive();
        solve(&mut low, Some(&mut preconditioner), &job)
    } else {
        let seed = if mode.starts_with("mixed") {
            let preparation_start = std::time::Instant::now();
            let attempted = prepare_f32_seed(
                &mut target,
                CpuBackend::<f32>::new(),
                &job,
                F32SeedConfig {
                    max_iterations: mode.strip_prefix("mixed").unwrap().parse().unwrap(),
                    ..Default::default()
                },
            );
            preparation_seconds = preparation_start.elapsed().as_secs_f64();
            match attempted {
                Ok(seed) => {
                    seed_report = Some(seed.report);
                    Some(seed.fields)
                }
                Err(reason) => {
                    rejection = Some(reason);
                    None
                }
            }
        } else {
            None
        };
        let mut preconditioner = target.build_homogeneous_preconditioner_adaptive();
        if let Some(fields) = seed {
            solve_with_warmstart(&mut target, Some(&mut preconditioner), &job, &fields)
        } else {
            solve(&mut target, Some(&mut preconditioner), &job)
        }
    };
    // Native f32 outputs are promoted without repairing their values. All
    // modes face the same original f64-pencil quality gate and output metric.
    let cert = certify_without_refinement(&mut target, &result.eigenvalues, &result.eigenvectors);
    let total_iterations = iteration_total().wrapping_sub(iteration_start);
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
        json!({"case":case,"mode":mode,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),
        "precision":mode,"tolerance":tolerance,"seconds":seconds,"peak_rss_kib":rss_kib,"expected_points":1,
        "requested_path_success":rejection.is_none(),"preparation_seconds":preparation_seconds,"seed":seed_report,"rejection":rejection,
        "native_final_residuals":result.final_residuals,"native_final_mass_orthogonality_defect":result.b_orthogonality_defect,
        "spectrum":[result.eigenvalues.iter().map(|v|v.max(0.0).sqrt()).collect::<Vec<_>>()],
        "points":[{"k":k,"iterations":total_iterations,"final_precision_iterations":result.iterations,"converged":result.converged,
            "residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,"b_orthogonality_defect":cert.b_orthogonality_defect,"field_count":result.eigenvectors.len()}],
        "counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),
        "scope":"complete first-point solve including low-precision work, promotion, target preparation and f64 original-pencil certification; no serialized field export"})
    );
}
