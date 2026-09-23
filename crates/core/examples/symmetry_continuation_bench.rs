//! Complete geometry sweeps: source solves and predictor work are all timed.
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
    symmetry::{
        continuation::{GeometryContinuationCache, SeedHistory},
        reuse::CachedSpectrum,
    },
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("contrast-sweep");
    assert!(matches!(case, "contrast-sweep" | "radius-sweep"));
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(31);
    let bands: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    let pol = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("expected TM or TE"),
    };
    let mode = args.get(5).map(String::as_str).unwrap_or("latest");
    let history = match mode {
        "latest" => SeedHistory::Latest,
        "enriched" => SeedHistory::Recent { snapshots: 2 },
        _ => panic!("expected latest or enriched"),
    };
    let tolerance: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    assert!(tolerance.is_finite() && tolerance > 0.0);
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let lattice = Lattice2D::square(1.0);
    let k = [0.13, 0.17];
    let parameters: Vec<_> = (0..7)
        .map(|i| {
            if case == "contrast-sweep" {
                6.0 + 0.2 * i as f64
            } else {
                0.18 + 0.005 * i as f64
            }
        })
        .collect();
    // Equal resident-cache budget in both modes, sufficient for two complete
    // snapshots including sampled scalar/tensor identity and source metadata.
    let cache_limit = 2 * (grid.len() * (bands * 16 + 48) + bands * 24 + 256);
    let mut cache = GeometryContinuationCache::new(cache_limit);
    let config = SingleSolveJob::new(bands)
        .with_tolerance(tolerance)
        .with_max_iterations(500);
    let mut points = Vec::new();
    let mut spectrum = Vec::new();
    let start = std::time::Instant::now();
    for (index, &parameter) in parameters.iter().enumerate() {
        let (radius, epsilon) = if case == "contrast-sweep" {
            (0.2, parameter)
        } else {
            (parameter, 8.9)
        };
        let dielectric = Dielectric2D::from_geometry(
            &Geometry2D {
                lattice,
                eps_bg: 1.0,
                atoms: vec![BasisAtom {
                    pos: [0.0; 2],
                    radius,
                    eps_inside: epsilon,
                }],
            },
            grid,
            &Default::default(),
        );
        let mut op = ThetaOperator::new(
            CpuBackend::<f64>::new(),
            dielectric,
            pol,
            lattice.reciprocal().fractional_to_cartesian(k),
        );
        let mut preconditioner = op.build_homogeneous_preconditioner_adaptive();
        let predictor_start = std::time::Instant::now();
        let seed = cache.prepare(&mut op, history, bands, 1e-9).unwrap();
        let predictor_seconds = predictor_start.elapsed().as_secs_f64();
        let result = if let Some(seed) = &seed {
            solve_with_warmstart(&mut op, Some(&mut preconditioner), &config, &seed.fields)
        } else {
            solve(&mut op, Some(&mut preconditioner), &config)
        };
        let cert = certify_without_refinement(&mut op, &result.eigenvalues, &result.eigenvectors);
        let entry = CachedSpectrum {
            source_index: index,
            k,
            eigenvalues: result.eigenvalues.clone(),
            fields: result.eigenvectors,
            residuals: cert.residuals.clone(),
            absolute_residuals: cert.absolute_residuals.clone(),
            orthogonality_defect: cert.b_orthogonality_defect,
        };
        let field_count = entry.fields.len();
        let insertion = cache.insert(
            &mut op,
            format!("geometry{index}"),
            vec![parameter],
            entry,
            &Default::default(),
        );
        let failed = insertion.is_err();
        let insertion = match insertion {
            Ok(report) => serde_json::to_value(report).unwrap(),
            Err(reason) => json!({"stored":false,"rejection":reason}),
        };
        spectrum.push(
            result
                .eigenvalues
                .iter()
                .map(|v| v.max(0.0).sqrt())
                .collect::<Vec<_>>(),
        );
        points.push(json!({"k":k,"parameter":parameter,"iterations":result.iterations,"converged":result.converged,
            "residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,"b_orthogonality_defect":cert.b_orthogonality_defect,
            "field_count":field_count,"predictor_seconds":predictor_seconds,"predictor":seed.map(|s|s.report),"insertion":insertion}));
        // Preserve the failed point and preceding work without contaminating
        // later geometry history with an uncertified source.
        if failed {
            break;
        }
    }
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
        "tolerance":tolerance,"seconds":seconds,"peak_rss_kib":rss_kib,"cache_limit":cache_limit,"parameters":parameters,
        "spectrum":spectrum,"points":points,"counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),
        "scope":"complete geometry sweep including source solves, cache insertion, target preparation and final certification; no serialized field export"})
    );
}
