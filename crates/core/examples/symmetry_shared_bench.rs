//! End-to-end block-Gram/Ritz experiments in one shared iteration engine.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve},
    eigensolver::{EigensolverConfig, iteration_total, refine::certify_without_refinement},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        reuse::SymmetryOptions,
        sectors::solve_conservative,
        shared::{SharedSolverConfig, solve_shared},
        trial::{GramMode, RitzMode},
    },
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("mirror");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(31);
    let bands: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    let polarization = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("expected TM or TE"),
    };
    let mode = args.get(5).map(String::as_str).unwrap_or("full");
    assert!(matches!(mode, "full" | "gram" | "blocks" | "independent"));
    let tolerance: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-8);
    let lattice = Lattice2D::square(1.0);
    let (family, k, grid, atoms) = match case {
        "mirror" => (
            PlaneGroup::Pm,
            [0.17, 0.0],
            Grid2D::new(n, n, 1.0, 1.0),
            vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.24,
                eps_inside: 8.9,
            }],
        ),
        "square" => (
            PlaneGroup::P4m,
            [0.0; 2],
            Grid2D::new(n, n, 1.0, 1.0),
            vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.24,
                eps_inside: 8.9,
            }],
        ),
        "glide" => (
            PlaneGroup::Pg,
            [0.5, 0.0],
            Grid2D::new(n, n - 1, 1.0, 1.0),
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
        "projective" => (
            PlaneGroup::Pgg,
            [-0.5, -0.5],
            Grid2D::new(n, n, 1.0, 1.0),
            vec![],
        ),
        _ => panic!("expected mirror, square, glide or projective"),
    };
    let start = std::time::Instant::now();
    let iteration_start = iteration_total();
    let material = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms,
        },
        grid,
        &Default::default(),
    );
    let mut op = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        material,
        polarization,
        lattice.reciprocal().fractional_to_cartesian(k),
    );
    let mut preconditioner = op.build_homogeneous_preconditioner_adaptive();
    let options = SymmetryOptions::default();
    let mut report = None;
    let mut rejection = None;
    let attempted = if mode == "independent" {
        solve_conservative(
            &mut op,
            &lattice,
            k,
            family.group(),
            &EigensolverConfig {
                n_bands: bands,
                tol: 1e-14,
                max_iter: 500,
                ..Default::default()
            },
            None,
            &options,
            Some(&mut preconditioner),
        )
        .map(|r| {
            report = Some(json!(r.report));
            r.spectrum
        })
    } else {
        solve_shared(
            &mut op,
            &lattice,
            k,
            family.group(),
            &SharedSolverConfig {
                n_bands: bands,
                residual_tolerance: tolerance,
                gram: if mode == "full" {
                    GramMode::Full
                } else {
                    GramMode::Blocks
                },
                ritz: if mode == "blocks" {
                    RitzMode::Blocks
                } else {
                    RitzMode::Full
                },
                ..Default::default()
            },
            &options,
            None,
            Some(&mut preconditioner),
        )
        .map(|r| {
            report = Some(json!(r.report));
            r.spectrum
        })
    };
    let (values, fields, converged) = match attempted {
        Ok(r) => (r.eigenvalues, r.fields, true),
        Err(e) => {
            rejection = Some(e);
            let fallback = solve(
                &mut op,
                Some(&mut preconditioner),
                &SingleSolveJob::new(bands)
                    .with_tolerance(1e-14)
                    .with_max_iterations(500),
            );
            (
                fallback.eigenvalues,
                fallback.eigenvectors,
                fallback.converged,
            )
        }
    };
    let cert = certify_without_refinement(&mut op, &values, &fields);
    let iterations = iteration_total().wrapping_sub(iteration_start);
    let seconds = start.elapsed().as_secs_f64();
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u64>().ok())
        });
    println!(
        "{}",
        json!({"case":case,"grid":[grid.nx,grid.ny],"bands":bands,
        "polarization":format!("{polarization:?}"),"mode":mode,"precision":"f64",
        "tolerance":tolerance,"independent_eigenvalue_tolerance":1e-14,
        "seconds":seconds,"peak_rss_kib":rss,"expected_points":1,
        "requested_path_success":rejection.is_none(),"rejection":rejection,"report":report,
        "spectrum":[values.iter().map(|v|v.max(0.0).sqrt()).collect::<Vec<_>>()],
        "points":[{"k":k,"iterations":iterations,"converged":converged,
            "residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,
            "b_orthogonality_defect":cert.b_orthogonality_defect,"field_count":fields.len()}],
        "counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),
        "scope":"complete solve including qualification, actual iterative search matrices, reconstruction, failures and fresh certification; no serialized field export; independent reference changes the iteration engine"})
    );
}
