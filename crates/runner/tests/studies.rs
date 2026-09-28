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
    for metric in [false, true] {
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
    if metric {
        let mut value = serde_json::to_value(&c).unwrap();
        value["geometry"]["lattice"] = json!({"type":"triangular","a":1.0});
        value["dielectric"]["smoothing"] = json!("voronoi");
        value["bands"]["discretization"] = json!({"kind":"metric_ball","angular_cutoff":std::f64::consts::TAU * 2.8});
        c = Config::from_value(value).unwrap();
    }
    let reference = blaze2d_runner::collect(Plan::new(c.clone(), Platform::Native).unwrap(),
        Options {threads:1,..Default::default()}).unwrap();
    assert!(reference.errors.is_empty());
    c.bands.as_mut().unwrap().symmetry.continuation = ContinuationPolicy::PreviousGeometry;
    if metric {
        c.bands.as_mut().unwrap().symmetry.arithmetic = SymmetryArithmetic::PtReal;
        c.bands.as_mut().unwrap().symmetry.coarse_start = Some(CoarseStart { resolution: [5,5], max_seed_field_bytes: 1<<20 });
    }
    let budget = 256 * 1024;
    c.bands.as_mut().unwrap().symmetry.max_continuation_bytes = budget;
    for (threads, policy, expected_seeds) in [(1, ContinuationPolicy::PreviousGeometry, 12),
        (2, ContinuationPolicy::PreviousGeometry, 12), (1, ContinuationPolicy::FirstPoint, 4)] {
        c.bands.as_mut().unwrap().symmetry.continuation = policy;
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
                assert!(continuation["cache_payload_bytes"].as_u64().unwrap_or(0) <= (budget / threads) as u64);
                if !continuation["seed"].is_null() { seeded += 1; }
            }
        }
        assert_eq!(seeded, expected_seeds, "seed scope must match the selected continuation policy");
    }    }

}

#[test]
fn automatic_policy_uses_qualified_pt_and_stops_retrying_broken_material() {
    for metric in [false,true] {
    let mut config = Config::default();
    config.grid.resolution = Resolution::Uniform(31);
    config.eigensolver.tolerance = Some(1e-13);
    let bands = config.bands.as_mut().unwrap();
    bands.path = KPath { points:Some(vec![vec![0.19,0.13], vec![0.27,0.13], vec![0.19,0.13]]),
        intervals_per_segment:None, ..Default::default() };
    bands.symmetry.policy = SymmetryPolicy::Auto;
    if metric {
        let mut value = serde_json::to_value(&config).unwrap();
        value["geometry"]["lattice"] = json!({"type":"triangular","a":1.0});
        value["dielectric"]["smoothing"] = json!("voronoi");
        value["bands"]["discretization"] = json!({"kind":"metric_ball","angular_cutoff":std::f64::consts::TAU * 14.0});
        config = Config::from_value(value).unwrap();
    }
    let method = if metric { "metric_support_pt_real" } else { "fourier" };
    let solve = |c| {
        let job = Plan::new(c, Platform::Native).unwrap().job(0).unwrap();
        execute(blaze2d_backend_cpu::CpuBackend::<f64>::new(), "cpu", &job, |_|{}).unwrap()
    };
    let mut loose = config.clone();
    loose.eigensolver.tolerance = Some(1e-6);
    let loose_job = Plan::new(loose, Platform::Native).unwrap().job(0).unwrap();
    let (selected, decision) = blaze2d_interface::policy::resolve(&loose_job, "cpu");
    assert_eq!(selected.arithmetic, SymmetryArithmetic::SpatialComplex);
    assert!(decision.unwrap().reasons.iter().any(|r|r.contains("stopping tolerances")));
    let qualified = solve(config.clone());
    assert_eq!(qualified.metadata["symmetry"]["automatic"]["arithmetic"], "pt_real");
    assert!(!qualified.metadata["symmetry"]["points"][0]["coordinates"].is_null());
    assert_eq!(qualified.metadata["iterations"][2], 0);
    let mut manual = config.clone();
    manual.bands.as_mut().unwrap().symmetry.policy = SymmetryPolicy::Manual;
    manual.bands.as_mut().unwrap().symmetry.arithmetic = SymmetryArithmetic::PtReal;
    manual.bands.as_mut().unwrap().symmetry.k_reuse = KReuse::Identical;
    assert_eq!(qualified.arrays, solve(manual).arrays);

    config.geometry.objects[0].center = vec![0.13,0.07];
    let fallback = solve(config.clone());
    let points = fallback.metadata["symmetry"]["points"].as_array().unwrap();
    assert!(points[0]["solver_attempts"].as_array().unwrap().iter().any(|a| a["method"] == method && a["retained"] == false));
    assert!(points[1]["solver_attempts"].as_array().unwrap().iter().all(|a| a["method"] != method));
    assert!(points[1]["rejected"].as_array().unwrap().iter().any(|v| v.as_str().unwrap().contains("automatic arithmetic fallback")));
    config.bands.as_mut().unwrap().symmetry.policy = SymmetryPolicy::Manual;
    config.bands.as_mut().unwrap().symmetry.k_reuse = KReuse::Identical;
    if metric { config.bands.as_mut().unwrap().symmetry.arithmetic = SymmetryArithmetic::FourierComplex; }
    assert_eq!(fallback.arrays, solve(config).arrays);
    }
}

#[test]
fn native_geometry_history_preserves_public_paths_guards_and_resume_roots() {
    let mut c=study();
    c.grid.resolution=Resolution::Uniform(9);
    c.eigensolver.tolerance=Some(1e-14);
    c.eigensolver.max_iterations=Some(500);
    c.results.eigenvectors=true;
    let b=c.bands.as_mut().unwrap();
    b.count=4; b.tracking=false; b.accuracy=Some(BandAccuracy::default());
    b.path.points=Some(vec![vec![0.,0.],vec![0.19,0.13],vec![0.,0.]]);
    let reference=blaze2d_runner::collect(Plan::new(c.clone(),Platform::Native).unwrap(),
        Options {threads:1,..Default::default()}).unwrap();
    assert!(reference.errors.is_empty(),"{:?}",reference.errors);
    c.bands.as_mut().unwrap().warm_start=Some(GeometryWarmStart {scope:GeometryScope::Path,
        guards:2,max_cache_bytes:256*1024,..Default::default()});
    for (method,threads,innovations) in [(GeometryMethod::Copy,1,0),(GeometryMethod::Copy,2,0),(GeometryMethod::Metric,1,0),(GeometryMethod::Secant,1,0),(GeometryMethod::History,1,0),(GeometryMethod::Residual,1,0),(GeometryMethod::Response,1,0),(GeometryMethod::Response,2,0),(GeometryMethod::History,1,4)] {
        let warm=c.bands.as_mut().unwrap().warm_start.as_mut().unwrap();
        warm.method=method;warm.retain_tm_actions=threads==2;
        warm.innovation_columns=innovations;
        let candidate=blaze2d_runner::collect(Plan::new(c.clone(),Platform::Native).unwrap(),
            Options {threads,..Default::default()}).unwrap();
        assert!(candidate.errors.is_empty(),"{:?}",candidate.errors);
        let mut seeds=0;
        let mut advanced=0;
        let mut compressed_seeds=0;
        for (a,b) in reference.results.iter().zip(&candidate.results) {
            assert_eq!(a.job_index,b.job_index);
            assert_eq!(b.arrays["eigenvectors"].shape,vec![3,4,9,9]);
            for (&x,&y) in a.arrays["frequencies"].data.iter().zip(&b.arrays["frequencies"].data) {
                assert!((x-y).abs()<2e-7,"{x} != {y}");
            }
            for point in b.metadata["geometry_warm_start"]["points"].as_array().unwrap() {
                seeds+=usize::from(!point["source"].is_null());
                compressed_seeds+=point["prediction"]["innovation_columns"].as_u64().unwrap_or(0);
                if (point["prediction"].is_object() && point["prediction"]["fallback"].is_null()) || point["response"].is_object() {advanced+=1;}
                if threads==2 && !point["source"].is_null() {
                    assert!(point[if method==GeometryMethod::Response {"preparation_actions_reused"} else {"tm_actions_reused"}].as_u64().unwrap()>0);
                }
                assert_eq!(point["certified_guards"],2,"{point}");
                assert!(point["insertion"]["payload_bytes"].as_u64().unwrap()<=256*1024/threads as u64);
            }
        }
        assert!(seeds>0);
        if innovations>0 {assert!(compressed_seeds>0,"compressed directions were not consumed");}
        if method!=GeometryMethod::Copy {assert!(advanced>0,"{method:?} never admitted a prediction");}
    }
    let resumed=blaze2d_runner::collect(Plan::new(c,Platform::Native).unwrap(),
        Options {threads:1,completed_jobs:vec![0,1],..Default::default()}).unwrap();
    assert!(resumed.errors.is_empty());
    assert_eq!(resumed.results.len(),1);
    assert_eq!(resumed.results[0].job_index,2);
    assert_eq!(resumed.results[0].metadata["geometry_warm_start"]["points"][0]["cold_root"],true);
}

#[test]
fn geometry_anchors_choose_donors_and_expand_a_split_boundary_with_bounded_work() {
    let mut c=study();c.grid.resolution=Resolution::Uniform(9);
    c.eigensolver.tolerance=Some(1e-14);c.eigensolver.max_iterations=Some(500);
    let b=c.bands.as_mut().unwrap();b.count=3;b.tracking=false;b.accuracy=Some(BandAccuracy::default());
    b.path.points=Some(vec![vec![0.,0.],vec![0.19,0.13],vec![0.2,0.14],vec![0.,0.]]);
    let run=|c:Config|blaze2d_runner::collect(Plan::new(c,Platform::Native).unwrap(),Options{threads:1,..Default::default()}).unwrap();
    let reference=run(c.clone());assert!(reference.errors.is_empty());
    for (scope,selection,adaptive) in [(GeometryScope::Anchors,GeometryDonorSelection::SameK,true),
        (GeometryScope::Path,GeometryDonorSelection::PreviousK,false),
        (GeometryScope::Path,GeometryDonorSelection::Screened,false)] {
        c.bands.as_mut().unwrap().warm_start=Some(GeometryWarmStart {scope,anchors:vec![[0.19,0.13]],donor_selection:selection,
            guards:if adaptive {0}else{2},max_guards:adaptive.then_some(4),..Default::default()});
        let candidate=run(c.clone());assert!(candidate.errors.is_empty(),"{:?}",candidate.errors);
        let mut expansions=0;let mut selected_path=0;let mut selected_geometry=0;
        for (a,b) in reference.results.iter().zip(&candidate.results) {
            for (&x,&y) in a.arrays["frequencies"].data.iter().zip(&b.arrays["frequencies"].data) {assert!((x-y).abs()<2e-7);}
            for (i,p) in b.metadata["geometry_warm_start"]["points"].as_array().unwrap().iter().enumerate() {
                if scope==GeometryScope::Anchors && i==2 {
                    assert_eq!(p["geometry_anchor"],false); assert_eq!(p["solved_guards"],0);
                    assert!(p["path"].is_object()); continue;
                }
                assert!(p.is_object());
                selected_path+=usize::from(p["selection"]["selected"]=="previous_k");
                selected_geometry+=usize::from(p["selection"]["selected"]=="same_k_geometry");
                let attempts=p["guard_attempts"].as_array().unwrap();expansions+=attempts.len()-1;
                assert!(attempts.len()<=3);assert!(p["solved_guards"].as_u64().unwrap()<=4);
                let rounds:u64=attempts.iter().map(|a|a["iterations"].as_u64().unwrap()).sum();
                assert_eq!(p["iterations"],rounds,"{p}");
                if attempts.len()>1 {assert!(p["discarded_iterations"].as_u64().unwrap()>0);}
            }
        }
        assert!(selected_geometry>0);
        if adaptive {assert!(expansions>0,"split doublet was not detected");}
        if selection!=GeometryDonorSelection::SameK {assert!(selected_path>0);}
    }
}
