//! Complete sweeps with the same one-snapshot parent for copy and response.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    eigensolver::{EigensolverConfig, refine::certify_without_refinement},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        response::{ResponseConfig, predict_response, predict_response_symmetric},
        reuse::CachedSpectrum,
        sectors::solve_conservative,
        seed::rank_aware_seed,
    },
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("contrast-sweep");
    assert!(matches!(
        case,
        "contrast-sweep" | "radius-sweep" | "gamma-sweep" | "mirror-sweep" | "square-sweep"
    ));
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(31);
    let bands: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    let pol = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("expected TM or TE"),
    };
    let mode = args.get(5).map(String::as_str).unwrap_or("copy");
    let selected = mode.starts_with("selected");
    let response_config = if mode == "copy" {
        None
    } else {
        Some(ResponseConfig {
            max_inner_iterations: mode
                .strip_prefix("response")
                .or_else(|| mode.strip_prefix("selected"))
                .expect("expected copy, response<budget> or selected<budget>")
                .parse()
                .unwrap(),
            inner_relative_tolerance: 1e-6,
        })
    };
    let tolerance: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    assert!(tolerance.is_finite() && tolerance > 0.0);
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let lattice = Lattice2D::square(1.0);
    let (k, group) = match case {
        "mirror-sweep" => ([0.13, 0.0], Some(PlaneGroup::Pm)),
        "square-sweep" => ([0.0, 0.0], Some(PlaneGroup::P4m)),
        "gamma-sweep" => ([0.0, 0.0], None),
        _ => ([0.13, 0.17], None),
    };
    assert!(
        !selected || group.is_some(),
        "selection rules need a sector-source sweep"
    );
    let parameters: Vec<_> = (0..7)
        .map(|i| {
            if case == "radius-sweep" {
                0.18 + 0.005 * i as f64
            } else {
                6.0 + 0.2 * i as f64
            }
        })
        .collect();
    let config = SingleSolveJob::new(bands)
        .with_tolerance(tolerance)
        .with_max_iterations(500);
    let quality = Default::default();
    let mut previous: Option<(ThetaOperator<CpuBackend<f64>>, CachedSpectrum)> = None;
    let mut points = Vec::new();
    let mut spectrum = Vec::new();
    let mut requested_path_success = true;
    let mut snapshot_peak_payload = 0;
    let start = std::time::Instant::now();
    for (index, &parameter) in parameters.iter().enumerate() {
        let (radius, epsilon) = if case == "radius-sweep" {
            (parameter, 8.9)
        } else {
            (0.2, parameter)
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
        let mut predictor = serde_json::Value::Null;
        let mut rejection = serde_json::Value::Null;
        let seed = if let Some((old_op, old)) = previous.as_mut() {
            let response = response_config.map(|cfg| {
                if selected {
                    predict_response_symmetric(
                        old_op,
                        &mut op,
                        old,
                        cfg,
                        &quality,
                        &lattice,
                        group.unwrap().group(),
                    )
                } else {
                    predict_response(old_op, &mut op, old, cfg, &quality)
                }
            });
            match response {
                Some(Ok(seed)) => {
                    predictor = serde_json::to_value(seed.report).unwrap();
                    Some(seed.fields)
                }
                attempt => {
                    if let Some(Err(error)) = attempt {
                        requested_path_success = false;
                        rejection = serde_json::to_value(error).unwrap();
                    }
                    let copied = rank_aware_seed(&mut op, &old.fields, 1e-9).unwrap();
                    predictor = serde_json::to_value(copied.report).unwrap();
                    Some(copied.fields)
                }
            }
        } else {
            None
        };
        let predictor_seconds = predictor_start.elapsed().as_secs_f64();
        let iteration_start = blaze2d_core::eigensolver::iteration_total();
        let mut sector_rejection = None;
        let sectors = group.and_then(|group| {
            match solve_conservative(
                &mut op,
                &lattice,
                k,
                group.group(),
                &EigensolverConfig {
                    n_bands: bands,
                    tol: tolerance,
                    max_iter: 500,
                    ..Default::default()
                },
                seed.as_deref(),
                &quality,
                Some(&mut preconditioner),
            ) {
                Ok(result) => Some(result),
                Err(reason) => {
                    requested_path_success = false;
                    sector_rejection = Some(reason);
                    None
                }
            }
        });
        let (eigenvalues, fields, converged) = if let Some(result) = sectors {
            (result.spectrum.eigenvalues, result.spectrum.fields, true)
        } else {
            let result = if let Some(seed) = seed {
                solve_with_warmstart(&mut op, Some(&mut preconditioner), &config, &seed)
            } else {
                solve(&mut op, Some(&mut preconditioner), &config)
            };
            (result.eigenvalues, result.eigenvectors, result.converged)
        };
        let iterations = blaze2d_core::eigensolver::iteration_total().wrapping_sub(iteration_start);
        let cert = certify_without_refinement(&mut op, &eigenvalues, &fields);
        let entry = CachedSpectrum {
            source_index: index,
            k,
            eigenvalues: eigenvalues.clone(),
            fields,
            residuals: cert.residuals.clone(),
            absolute_residuals: cert.absolute_residuals.clone(),
            orthogonality_defect: cert.b_orthogonality_defect,
        };
        let valid = converged && entry.passes(&quality);
        snapshot_peak_payload = snapshot_peak_payload.max(entry.bytes());
        spectrum.push(
            eigenvalues
                .iter()
                .map(|v| v.max(0.0).sqrt())
                .collect::<Vec<_>>(),
        );
        points.push(json!({"k":k,"parameter":parameter,"iterations":iterations,"converged":converged,"sector_rejection":sector_rejection,
            "residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,"b_orthogonality_defect":cert.b_orthogonality_defect,
            "field_count":entry.fields.len(),"predictor_seconds":predictor_seconds,"predictor":predictor,"rejection":rejection}));
        // Both modes retain one complete operator and one certified snapshot.
        // No next geometry may consume a failed source.
        if !valid {
            break;
        }
        previous = Some((op, entry));
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
        "tolerance":tolerance,"seconds":seconds,"peak_rss_kib":rss_kib,"parameters":parameters,"expected_points":parameters.len(),
        "requested_path_success":requested_path_success,"snapshot_peak_payload_bytes":snapshot_peak_payload,"retained_snapshots":1,"source_sector_group":group.map(|g|format!("{g:?}")),
        "spectrum":spectrum,"points":points,"counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),
        "scope":"complete geometry sweep including source solves, response work, target preparation and final certification; one retained source operator in both controls; no serialized field export"})
    );
}
