use blaze2d_interface::*;
use blaze2d_runner::{start, Options, ErrorPolicy};
use serde_json::json;

fn study() -> Config {
    let mut c = Config::default();
    c.grid.resolution = Resolution::Uniform(12);
    c.bands.as_mut().unwrap().path = KPath { points:Some(vec![vec![0.19,0.13]]), intervals_per_segment:None, ..Default::default() };
    c.sweeps = vec![Sweep { name:"radius".into(), target:"geometry.objects.rod.radius".into(),
        values:Some(vec![json!(0.18),json!(0.2),json!(0.22)]), linspace:None }];
    c
}

#[test]
fn bounded_parallel_delivery_preserves_all_jobs_and_single_equivalence() {
    let c = study();
    let plan = Plan::new(c.clone(), Platform::Native).unwrap();
    let stream = start(plan.clone(), Options {threads:2,queue_capacity:1,..Default::default()}).unwrap();
    let mut results = Vec::new();
    while let Some(event) = stream.next_event() {
        match event {
            Event::Result {result} => results.push(result),
            Event::JobFailure {error} => panic!("{}", error.diagnostic),
            Event::Terminal {status,completed,failed} => {
                assert_eq!(status, RunStatus::Completed);
                assert_eq!((completed,failed),(3,0));
            }, _=>{}
        }
    }
    results.sort_by_key(|r|r.job_index);
    assert_eq!(results.len(),3);
    let single = execute(blaze2d_backend_cpu::CpuBackend::<f64>::new(), "cpu", &plan.job(1).unwrap(), |_|{}).unwrap();
    assert_eq!(results[1].arrays["frequencies"],single.arrays["frequencies"]);
    assert_eq!(results[1].metadata["runner"]["threads"],2);
}

#[test]
fn runtime_failure_has_distinct_terminal_status() {
    let mut v = serde_json::to_value(study()).unwrap();
    v["task"] = json!("operators");
    v.as_object_mut().unwrap().remove("bands");
    v["operators"] = json!({"k_point":{"value":[0.19,0.13],"basis":"reciprocal_fractional"},"retained_bands":2,"remote_bands":2,
        "quantities":["velocity"],"fail_on_residual":1e-30});
    let config = Config::from_value(v).unwrap();
    for policy in [ErrorPolicy::Stop,ErrorPolicy::Continue] {
        let stream = start(Plan::new(config.clone(),Platform::Native).unwrap(), Options {threads:1,error_policy:policy,..Default::default()}).unwrap();
        let mut failed = 0;
        while let Some(e) = stream.next_event() {
            match e {
                Event::JobFailure {..} => failed+=1,
                Event::Terminal {status,completed,..} => {
                    assert_eq!(completed,0);
                    assert_eq!(status,if policy==ErrorPolicy::Stop {RunStatus::Failed} else {RunStatus::CompletedWithErrors});
                }, _=>{}
            }
        }
        assert_eq!(failed,if policy==ErrorPolicy::Stop {1} else {3});
    }
}

#[test]
fn cancel_does_not_report_completion() {
    let stream = start(Plan::new(study(),Platform::Native).unwrap(), Options {threads:1,queue_capacity:1,..Default::default()}).unwrap();
    stream.cancel();
    let mut terminal = None;
    while let Some(e) = stream.next_event() { if let Event::Terminal {status,..} = e { terminal = Some(status); } }
    assert_eq!(terminal,Some(RunStatus::Cancelled));
}

#[test]
fn geometry_chains_carry_full_paths_with_bounded_fields_and_original_job_ids() {
    let mut c = study();
    c.grid.resolution = Resolution::Uniform(9);
    c.eigensolver.tolerance = Some(1e-13);
    c.eigensolver.max_iterations = Some(400);
    let b = c.bands.as_mut().unwrap();
    b.count = 4;
    b.tracking = false;
    b.path.points = Some(vec![vec![0.0, 0.0], vec![0.19, 0.13], vec![0.31, 0.07]]);
    c.results.eigenvectors = true;
    c.sweeps.push(Sweep { name:"polarization".into(), target:"polarization".into(),
        values:Some(vec![json!("TM"), json!("TE")]), linspace:None });
    let reference = blaze2d_runner::collect(Plan::new(c.clone(), Platform::Native).unwrap(),
        Options {threads:1,..Default::default()}).unwrap();
    assert!(reference.errors.is_empty());
    c.bands.as_mut().unwrap().symmetry.continuation = ContinuationPolicy::PreviousGeometry;
    let budget = 256 * 1024;
    c.bands.as_mut().unwrap().symmetry.max_continuation_bytes = budget;
    for threads in [1, 2] {
        let plan = Plan::new(c.clone(), Platform::Native).unwrap();
        assert_eq!((0..6).map(|i|plan.continuation_job_index(i)).collect::<Vec<_>>(), vec![0,2,4,1,3,5]);
        let candidate = blaze2d_runner::collect(plan, Options {threads,..Default::default()}).unwrap();
        assert!(candidate.errors.is_empty(), "{:?}", candidate.errors);
        let mut seeded = 0;
        for (expected, actual) in reference.results.iter().zip(&candidate.results) {
            assert_eq!(actual.job_index, expected.job_index);
            assert_eq!(actual.arrays["eigenvectors"].shape, vec![3,4,9,9]);
            assert_eq!(actual.metadata["runner"]["parallelism"], "geometry_chains");
            for (&a, &b) in actual.arrays["frequencies"].data.iter().zip(&expected.arrays["frequencies"].data) {
                assert!((a-b).abs() < 2e-7, "frequency {a} != {b}");
            }
            assert_eq!(actual.metadata["converged"], true);
            assert!(actual.metadata["certification"]["b_orthogonality_defect"].as_f64().unwrap() < 1e-8);
            for point in actual.metadata["symmetry"]["points"].as_array().unwrap() {
                let continuation = &point["continuation"];
                assert!(continuation["cache_payload_bytes"].as_u64().unwrap() <= (budget / threads) as u64);
                if !continuation["seed"].is_null() { seeded += 1; }
            }
        }
        assert_eq!(seeded, 12, "two later geometries per polarization, each with three k points");
    }
}
