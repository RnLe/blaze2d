use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;

fn bands(lattice: &str, polarization: &str, n: usize, extra: &str) -> Config {
    Config::from_toml(&source(lattice, polarization, n, extra)).unwrap()
}

fn source(lattice: &str, polarization: &str, n: usize, extra: &str) -> String {
    format!(r#"
schema = "blaze2d/1"
task = "bands"
polarization = "{polarization}"
[geometry]
background_epsilon = 13.0
lattice = {{ type = "{lattice}", a = 1.0 }}
[[geometry.objects]]
name = "rod"
kind = "circle"
center = [0.0, 0.0]
radius = 0.3
epsilon = 1.0
[grid]
resolution = {n}
[bands]
count = 8
path = {{ preset = "{lattice}", intervals_per_segment = 2 }}
{extra}
"#)
}

#[test]
fn default_band_jobs_use_residual_stopping_and_certified_pt_arithmetic() {
    for lattice in ["square", "triangular"] {
        for polarization in ["TE", "TM"] {
            for n in [16, 17] {
                let plan = Plan::new(bands(lattice, polarization, n, ""), Platform::Native).unwrap();
                let job = plan.job(0).unwrap();
                assert_eq!(job.resolved.config.eigensolver.stopping, Stopping::Residual);
                let result = execute(CpuBackend::<f64>::new(), "cpu", &job, |_| {}).unwrap();
                let m = &result.metadata;
                let case = format!("{lattice} {polarization} {n}");
                assert_eq!(m["converged"], true, "{case}");
                assert_eq!(m["symmetry"]["resolved"]["arithmetic"], "pt_real", "{case}");
                let frequencies = &result.arrays["frequencies"].data;
                for (i, (&f, &r)) in frequencies.iter().zip(&result.arrays["residuals"].data).enumerate() {
                    if f != 0.0 { assert!(r <= 1.5e-7, "{case} entry {i}: residual {r:e}"); }
                }
                for (&f, &r) in frequencies.iter().zip(&result.arrays["absolute_residuals"].data) {
                    if f == 0.0 { assert!(r <= 1e-7, "{case}: zero-mode absolute residual {r:e}"); }
                }
                for (i, point) in m["symmetry"]["points"].as_array().unwrap().iter().enumerate() {
                    let attempts = point["solver_attempts"].as_array().unwrap();
                    assert!(attempts.iter().all(|a| a["retained"] == true), "{case} point {i} rejected an attempt");
                    if point["source_index"].is_null() {
                        assert!(attempts.iter().any(|a| a["method"] == "fourier"), "{case} point {i}");
                        let leakage = point["coordinates"]["maximum_real_subspace_leakage"].as_f64().unwrap();
                        assert!(leakage <= 1e-10, "{case} point {i}: leakage {leakage:e}");
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_choices_keep_their_previous_meaning() {
    let explicit = Plan::new(bands("square", "TM", 16, "[eigensolver]\ntolerance = 1e-8\n"), Platform::Native).unwrap();
    assert_eq!(explicit.job(0).unwrap().resolved.config.eigensolver.stopping, Stopping::EigenvalueChange);
    let browser = Plan::new(bands("square", "TM", 16, ""), Platform::Browser).unwrap();
    assert_eq!(browser.job(0).unwrap().resolved.config.eigensolver.stopping, Stopping::EigenvalueChange);
    let manual = Plan::new(bands("square", "TM", 16, "[bands.symmetry]\nk_reuse = \"identical\"\n"), Platform::Native).unwrap();
    assert_eq!(manual.job(0).unwrap().resolved.config.bands.unwrap().symmetry.policy, Some(SymmetryPolicy::Manual));
    let automatic = Plan::new(bands("square", "TM", 16, ""), Platform::Native).unwrap();
    assert_eq!(automatic.job(0).unwrap().resolved.config.bands.unwrap().symmetry.policy, Some(SymmetryPolicy::Auto));
    assert!(Plan::new(bands("square", "TM", 16, "[bands.symmetry]\npolicy = \"auto\"\nk_reuse = \"identical\"\n"), Platform::Native).is_err());
}

fn resolved_symmetry(config: Config) -> Symmetry {
    Plan::new(config, Platform::Native).unwrap().job(0).unwrap().resolved.config.bands.unwrap().symmetry
}

#[test]
fn written_mechanism_keys_select_manual_even_at_their_defaults() {
    let toml = bands("square", "TM", 16, "[bands.symmetry]\narithmetic = \"spatial_complex\"\nk_reuse = \"off\"\n");
    assert_eq!(resolved_symmetry(toml.clone()).policy, Some(SymmetryPolicy::Manual));
    let job = Plan::new(toml, Platform::Native).unwrap().job(0).unwrap();
    let result = execute(CpuBackend::<f64>::new(), "cpu", &job, |_| {}).unwrap();
    assert!(result.metadata["symmetry"]["automatic"].is_null());
    assert_eq!(result.metadata["symmetry"]["requested"]["arithmetic"], "spatial_complex");
    let mut value = serde_json::to_value(bands("square", "TM", 16, "")).unwrap();
    value["bands"]["symmetry"] = serde_json::json!({"k_reuse": "off"});
    assert_eq!(resolved_symmetry(Config::from_value(value.clone()).unwrap()).policy, Some(SymmetryPolicy::Manual));
    value["bands"]["symmetry"] = serde_json::json!({"max_cache_bytes": 1024});
    assert_eq!(resolved_symmetry(Config::from_value(value).unwrap()).policy, Some(SymmetryPolicy::Auto));
}

#[test]
fn sweeps_keep_the_stopping_rule_they_control() {
    let sweep = "[[sweeps]]\nname = \"tol\"\ntarget = \"eigensolver.tolerance\"\nvalues = [1e-4, 1e-10]\n";
    let plan = Plan::new(bands("square", "TM", 16, sweep), Platform::Native).unwrap();
    for i in 0..2 {
        assert_eq!(plan.job(i).unwrap().resolved.config.eigensolver.stopping, Stopping::EigenvalueChange);
    }
    assert_eq!(plan.job(1).unwrap().resolved.config.eigensolver.tolerance, Some(1e-10));
    let rules = "[[sweeps]]\nname = \"rule\"\ntarget = \"eigensolver.stopping\"\nvalues = [\"eigenvalue_change\", \"auto\"]\n";
    let plan = Plan::new(bands("square", "TM", 16, rules), Platform::Native).unwrap();
    assert_eq!(plan.job(0).unwrap().resolved.config.eigensolver.stopping, Stopping::EigenvalueChange);
    assert_eq!(plan.job(1).unwrap().resolved.config.eigensolver.stopping, Stopping::Residual);
}

#[test]
fn exported_configurations_keep_automatic_choices_editable() {
    let plan = Plan::new(bands("square", "TM", 16, ""), Platform::Native).unwrap();
    assert_eq!(plan.config.eigensolver.stopping, Stopping::Auto);
    assert_eq!(plan.config.eigensolver.tolerance, None);
    assert_eq!(plan.config.bands.as_ref().unwrap().symmetry.policy, None);
    let exported = plan.config.to_toml().unwrap();
    let mut edited = Config::from_toml(&exported).unwrap();
    assert_eq!(edited, plan.config);
    edited.eigensolver.tolerance = Some(1e-9);
    edited.bands.as_mut().unwrap().symmetry.k_reuse = KReuse::Off;
    edited.eigensolver.precision = Precision::F32;
    let job = Plan::new(edited, Platform::Native).unwrap().job(0).unwrap();
    assert_eq!(job.resolved.config.eigensolver.stopping, Stopping::EigenvalueChange);
    assert_eq!(job.resolved.config.eigensolver.tolerance, Some(1e-9));
    let value = serde_json::to_value(&plan.config).unwrap();
    let job = Plan::new(Config::from_value(value).unwrap(), Platform::Native).unwrap().job(0).unwrap();
    assert_eq!(job.resolved.config.eigensolver.stopping, Stopping::Residual);
    assert_eq!(job.resolved.config.bands.unwrap().symmetry.policy, Some(SymmetryPolicy::Auto));
}

#[test]
fn automatic_pt_stays_inside_its_measured_envelope() {
    let decide = |config: Config| {
        let job = Plan::new(config, Platform::Native).unwrap().job(0).unwrap();
        let (selected, decision) = blaze2d_interface::policy::resolve(&job, "cpu");
        (selected.arithmetic, decision.unwrap().reasons.join("; "))
    };
    let tight = decide(bands("square", "TM", 16, "[eigensolver]\nresidual_tolerance = 1e-11\n"));
    assert_eq!(tight.0, SymmetryArithmetic::SpatialComplex);
    assert!(tight.1.contains("stopping tolerances"), "{}", tight.1);
    let shifted = source("square", "TM", 16, "").replace("center = [0.0, 0.0]", "center = [0.2, 0.1]");
    let off_centre = decide(Config::from_toml(&shifted).unwrap());
    assert_eq!(off_centre.0, SymmetryArithmetic::SpatialComplex);
    assert!(off_centre.1.contains("inversion"), "{}", off_centre.1);
    let job = Plan::new(Config::from_toml(&shifted).unwrap(), Platform::Native).unwrap().job(0).unwrap();
    let result = execute(CpuBackend::<f64>::new(), "cpu", &job, |_| {}).unwrap();
    for point in result.metadata["symmetry"]["points"].as_array().unwrap() {
        assert!(point["solver_attempts"].as_array().unwrap().iter().all(|a| a["retained"] == true));
    }
}

#[test]
fn automatic_policy_reserves_the_reuse_cache_only_for_repeated_points() {
    let peak = |extra: &str, repeated: bool| {
        let path = if repeated { "[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]" } else { "[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]" };
        let text = source("square", "TM", 16, extra).replace("path = { preset = \"square\", intervals_per_segment = 2 }",
            &format!("path = {{ vertices = [{path}], intervals_per_segment = 2 }}"));
        Plan::new(Config::from_toml(&text).unwrap(), Platform::Native).unwrap().summary.estimated_peak_bytes
    };
    // Auto selects PT here, so it reserves what manual PT reserves, plus the cache
    // only when a point repeats.
    let manual = "[bands.symmetry]\npolicy = \"manual\"\narithmetic = \"pt_real\"\n";
    let reuse = "[bands.symmetry]\npolicy = \"manual\"\narithmetic = \"pt_real\"\nk_reuse = \"identical\"\n";
    assert_eq!(peak("", false), peak(manual, false));
    assert_eq!(peak("", true), peak(reuse, true));
    assert!(peak("", true) > peak(manual, true));
}

#[test]
fn automatic_reuse_respects_the_metric_support_payload_allowance() {
    let decide = |budget: usize| {
        let mut value = serde_json::to_value(bands("triangular", "TM", 31, "[eigensolver]\ntolerance = 1e-13\n")).unwrap();
        value["dielectric"]["smoothing"] = serde_json::json!("voronoi");
        value["bands"]["discretization"] = serde_json::json!({"kind": "metric_ball",
            "angular_cutoff": std::f64::consts::TAU * 13.5, "max_estimated_payload_bytes": budget});
        let job = Plan::new(Config::from_value(value).unwrap(), Platform::Native).unwrap().job(0).unwrap();
        let (selected, decision) = blaze2d_interface::policy::resolve(&job, "cpu");
        (selected.k_reuse, decision.unwrap().reasons.join("; "))
    };
    assert_eq!(decide(1 << 30).0, KReuse::Identical);
    let (reuse, reasons) = decide(48 << 20);
    assert_eq!(reuse, KReuse::Off);
    assert!(reasons.contains("payload allowance"), "{reasons}");
}
