use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;

fn bands(extra: &str) -> Config {
    Config::from_toml(&format!(r#"
schema = "blaze2d/1"
task = "bands"
polarization = "TE"
[geometry]
background_epsilon = 13.0
lattice = {{ type = "triangular", a = 1.0 }}
[[geometry.objects]]
name = "rod"
kind = "circle"
center = [0.0, 0.0]
radius = 0.3
epsilon = 1.0
[grid]
resolution = 16
[bands]
count = 6
path = {{ preset = "triangular", intervals_per_segment = 2 }}
{extra}
"#)).unwrap()
}

fn run(c: Config) -> ResultRecord {
    let plan = Plan::new(c, Platform::Native).unwrap();
    execute(CpuBackend::<f64>::new(), "cpu", &plan.job(0).unwrap(), |_| {}).unwrap()
}

#[test]
fn defaults_reproduce_the_built_in_preconditioner_and_stopping_rule() {
    let plan = Plan::new(bands(""), Platform::Native).unwrap();
    let job = plan.job(0).unwrap();
    assert_eq!(job.preconditioner(), blaze2d_core::preconditioners::PreconditionerSpec::default());
    let solver = job.eigensolver();
    assert_eq!(solver.stopping, blaze2d_core::eigensolver::StoppingRule::Residual);
    assert_eq!(solver.history, blaze2d_core::eigensolver::HistoryRule::Ritz);
    let result = run(bands(""));
    assert_eq!(result.metadata["stopping_criterion"], "relative_residual");
    let points = result.metadata["point_seconds"].as_array().unwrap();
    assert_eq!(points.len(), result.metadata["iterations"].as_array().unwrap().len());
    let work = &result.metadata["work"];
    let iterations: u64 = result.metadata["iterations"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(work["iterations"].as_u64().unwrap(), iterations);
    assert!(work["a_columns"].as_u64().unwrap() > 0 && work["m_columns"].as_u64().unwrap() > 0);
    assert_eq!(result.metadata["preconditioner"]["points"][1]["kind"], "transverse");
}

#[test]
fn residual_stopping_meets_its_target_in_fresh_certificates() {
    let extra = "[eigensolver]\nstopping = \"residual\"\nresidual_tolerance = 1e-8\nmax_iterations = 400\nhistory = \"conjugate\"\n[eigensolver.preconditioner]\nkind = \"diagonal\"\nshift = \"fixed\"\nshift_value = 0.0\n";
    let result = run(bands(extra));
    assert_eq!(result.metadata["stopping_criterion"], "relative_residual");
    assert_eq!(result.metadata["converged"], true);
    let max = result.metadata["certification"]["max_residual"].as_f64().unwrap();
    assert!(max <= 1.5e-8, "max fresh residual {max:e}");
    assert_eq!(result.metadata["preconditioner"]["points"][1]["kind"], "diagonal");
    assert_eq!(result.metadata["preconditioner"]["points"][1]["shift"], 0.0);
}

#[test]
fn invalid_settings_are_rejected() {
    for extra in [
        "[eigensolver.preconditioner]\nband_window_blend = 1.5\n",
        "[eigensolver]\nresidual_tolerance = -1.0\n",
        "[eigensolver]\nprecision = \"f32\"\nstopping = \"residual\"\nresidual_tolerance = 1e-7\n",
    ] {
        assert!(Plan::new(bands(extra), Platform::Native).is_err(), "{extra}");
    }
    let tm = bands("[eigensolver.preconditioner]\nkind = \"transverse\"\n").to_toml().unwrap().replace("polarization = \"TE\"", "polarization = \"TM\"");
    assert!(Plan::new(Config::from_toml(&tm).unwrap(), Platform::Native).is_err());
}

#[test]
fn preconditioner_settings_are_sweepable() {
    let extra = "[[sweeps]]\nname = \"alpha\"\ntarget = \"eigensolver.preconditioner.shift_fraction\"\nvalues = [0.1, 0.5]\n";
    let plan = Plan::new(bands(extra), Platform::Native).unwrap();
    assert_eq!(plan.job(0).unwrap().preconditioner().shift_fraction, 0.1);
    assert_eq!(plan.job(1).unwrap().preconditioner().shift_fraction, 0.5);
}
