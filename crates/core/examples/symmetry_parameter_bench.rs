//! Complete fixed-job scheduling/cache experiments, including physical output.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    drivers::single_solve::SingleSolveJob,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{
        continuation::{GeometryEviction, SeedHistory},
        parameter::{
            ParameterMetric, ParameterOrder,
            sweep::{
                ParameterJob, SweepPolicy,
                concurrency::{Sharding, SweepExecutionBudget, run as run_parallel},
                run,
            },
        },
    },
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("revisits");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(31);
    let bands: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    let pol = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("expected TM or TE"),
    };
    let mode = args.get(5).map(String::as_str).unwrap_or("given");
    let tolerance: f64 = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    let metric = ParameterMetric::new(vec![1.0, 0.1]).unwrap();
    let (order, eviction, history) = match mode {
        "given" | "chain1" | "chain2" | "chain4" | "chain2-contiguous" => (
            ParameterOrder::Given,
            GeometryEviction::Fifo,
            SeedHistory::Latest,
        ),
        "route" | "route-chain2" => (
            ParameterOrder::Nearest,
            GeometryEviction::Fifo,
            SeedHistory::Latest,
        ),
        "lru" => (
            ParameterOrder::Given,
            GeometryEviction::LeastRecentlyUsed,
            SeedHistory::Latest,
        ),
        "near-evict" => (
            ParameterOrder::Given,
            GeometryEviction::FarthestFromIncoming {
                metric: metric.clone(),
            },
            SeedHistory::Latest,
        ),
        "near-seed" => (
            ParameterOrder::Given,
            GeometryEviction::Fifo,
            SeedHistory::Nearest { snapshots: 1 },
        ),
        _ => panic!("expected given, route, lru, near-evict or near-seed"),
    };
    let sequence: Vec<usize> = match case {
        "ordered" => (0..9).collect(),
        "shuffled" => vec![0, 6, 2, 7, 1, 5, 3, 8, 4],
        "revisits" => vec![0, 6, 0, 2, 6, 0, 4, 2, 5, 4, 0, 6],
        "grid" => vec![0, 8, 2, 10, 1, 7, 3, 11, 4, 9, 5, 6],
        _ => panic!("expected ordered, shuffled, revisits or grid"),
    };
    let jobs: Vec<_> = sequence
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let epsilon = 6.0
                + 0.2
                    * if case == "grid" {
                        (x / 3) as f64
                    } else {
                        x as f64
                    };
            let radius = if case == "grid" {
                0.19 + 0.01 * (x % 3) as f64
            } else {
                0.2
            };
            ParameterJob {
                id: format!("job{i:04}"),
                parameters: vec![epsilon, radius],
                geometry: Geometry2D {
                    lattice: Lattice2D::square(1.0),
                    eps_bg: 1.0,
                    atoms: vec![BasisAtom {
                        pos: [0.0; 2],
                        radius,
                        eps_inside: epsilon,
                    }],
                },
            }
        })
        .collect();
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    // Equal resident numerical/string cache allowance, holding two snapshots.
    let cache_limit = 2 * (grid.len() * (bands * 16 + 48) + bands * 24 + 256);
    let config = SingleSolveJob::new(bands)
        .with_tolerance(tolerance)
        .with_max_iterations(500);
    let k = [0.13, 0.17];
    let policy = SweepPolicy {
        order,
        metric,
        eviction,
        history,
        exact_reuse: true,
        max_cache_payload_bytes: cache_limit,
        max_output_field_bytes: 64 * 1024 * 1024,
    };
    let complete_start = std::time::Instant::now();
    let mut execution = None;
    let outcome = if mode.contains("chain") {
        let workers = if mode == "chain1" {
            1
        } else if mode == "chain4" {
            4
        } else {
            2
        };
        let result = run_parallel(
            CpuBackend::<f64>::new,
            &jobs,
            grid,
            k,
            pol,
            &Default::default(),
            &config,
            &policy,
            SweepExecutionBudget {
                workers,
                logical_cpu_budget: 4,
                max_total_payload_bytes: 512 * 1024 * 1024,
            },
            if mode == "chain2-contiguous" {
                Sharding::Contiguous
            } else {
                Sharding::RoundRobin
            },
        )
        .unwrap();
        execution = Some(result.execution);
        result.result
    } else {
        run(
            CpuBackend::<f64>::new,
            &jobs,
            grid,
            k,
            pol,
            &Default::default(),
            &config,
            &policy,
        )
        .unwrap()
    };
    let complete_seconds = complete_start.elapsed().as_secs_f64();
    let spectrum: Vec<_> = outcome
        .outputs
        .iter()
        .map(|o| {
            o.spectrum
                .eigenvalues
                .iter()
                .map(|v| v.max(0.0).sqrt())
                .collect::<Vec<_>>()
        })
        .collect();
    let points: Vec<_> = outcome.outputs.iter().map(|o| json!({"k":k,"iterations":o.report.iterations,
        "converged":o.report.converged,"residuals":o.spectrum.residuals,"absolute_residuals":o.spectrum.absolute_residuals,
        "b_orthogonality_defect":o.spectrum.orthogonality_defect,"field_count":o.spectrum.fields.len(),"job":o.report})).collect();
    std::hint::black_box(&outcome.outputs);
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        });
    println!(
        "{}",
        json!({"case":case,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),"mode":mode,
            "precision":"f64","tolerance":tolerance,"seconds":complete_seconds,"peak_rss_kib":rss,"expected_points":jobs.len(),
        "requested_path_success":outcome.report.rejection.is_none() && outcome.report.completed_jobs==jobs.len(),
            "spectrum":spectrum,"points":points,"report":outcome.report,"execution":execution,
            "counters":blaze2d_core::profiler::get_all_counters(),"stage_counters":blaze2d_core::profiler::get_all_stage_counters(),
            "scope":"complete parameter jobs, fixed aggregate cache/field-output caps, cold sources and predictor/corrector/certification costs; all fields restored to original order; optional private parallel chains with partitioned cache; no serialized field export"})
    );
}
