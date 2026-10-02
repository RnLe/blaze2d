use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;
use serde_json::json;

fn configuration(pol: Polarization) -> Config {
    let mut c = Config::default();
    c.polarization = pol;
    c.grid.resolution = Resolution::Uniform(13);
    c.eigensolver.tolerance = Some(1e-14);
    c.eigensolver.max_iterations = Some(500);
    let bands = c.bands.as_mut().unwrap();
    bands.count = 5;
    bands.tracking = false;
    bands.path = KPath {
        points: Some(vec![vec![0.0, 0.0], vec![0.17, 0.09], vec![0.0, 0.0]]),
        intervals_per_segment: None,
        ..Default::default()
    };
    c.results.eigenvectors = true;
    c
}

fn run(c: Config) -> Result<ResultRecord, JobFailure> {
    let job = Plan::new(c, Platform::Native).unwrap().job(0).unwrap();
    execute(CpuBackend::<f64>::new(), "cpu", &job, |_| {})
}

#[test]
fn acceptance_preserves_fields_and_covers_final_fallbacks() {
    for pol in [Polarization::TM, Polarization::TE] {
        for fallback in [false, true] {
            let mut c = configuration(pol);
            if fallback {
                c.geometry.objects[0].center = vec![0.13, 0.07];
                c.bands.as_mut().unwrap().symmetry.arithmetic = SymmetryArithmetic::PtReal;
            } else {
                c.bands.as_mut().unwrap().symmetry.k_reuse = KReuse::Identical;
            }
            let control = run(c.clone()).unwrap();
            assert_eq!(control.metadata["accuracy"]["status"], "not_requested");
            c.bands.as_mut().unwrap().accuracy = Some(BandAccuracy::default());
            let accepted = run(c.clone()).unwrap();
            assert_eq!(control.arrays, accepted.arrays);
            assert_eq!(accepted.metadata["accuracy"]["accepted"], true);
            assert_eq!(accepted.metadata["execution"]["accuracy"], "accepted");
            if fallback {
                assert_eq!(accepted.metadata["execution"]["coordinate_blocks"], json!({}));
                assert!(accepted.metadata["execution"]["rejected_solver_attempts"].as_u64().unwrap()>0);
                assert!(
                    accepted.metadata["symmetry"]["points"][0]["solver_attempts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|a| a["retained"] == false)
                );
            } else {
                assert_eq!(accepted.metadata["iterations"][2], 0);
                assert_eq!(accepted.metadata["execution"]["reused_points"], 1);
            }
            let policy = c.bands.as_mut().unwrap().accuracy.as_mut().unwrap();
            policy.max_relative_residual = 1e-30;
            policy.max_absolute_residual = 1e-30;
            let rejected = run(c).unwrap_err();
            assert_eq!(rejected.diagnostic.code, "accuracy_gate");
            let partial = rejected.partial_result.unwrap();
            assert_eq!(partial.arrays, control.arrays);
            assert_eq!(partial.metadata["accuracy"]["accepted"], false);
            assert_eq!(partial.metadata["execution"]["accuracy"], "rejected");
            assert!(
                partial.metadata["accuracy"]["failed_residuals"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        }
    }
}

#[test]
fn projected_and_external_pencils_enforce_the_same_policy() {
    for pol in [Polarization::TM, Polarization::TE] {
        let mut c = configuration(pol);
        c.bands.as_mut().unwrap().accuracy = Some(BandAccuracy::default());
        c.bands.as_mut().unwrap().path.points = Some(vec![vec![0.17, 0.09]]);
        let mut value = serde_json::to_value(&c).unwrap();
        value["geometry"]["lattice"] = json!({"type":"triangular", "a":1.0});
        value["dielectric"]["smoothing"] = json!("voronoi");
        value["bands"]["discretization"] =
            json!({"kind":"metric_ball", "angular_cutoff":std::f64::consts::TAU*3.5});
        let metric = run(Config::from_value(value).unwrap()).unwrap();
        assert_eq!(metric.metadata["accuracy"]["accepted"], true);
        assert_eq!(
            metric.metadata["accuracy"]["scope"],
            "selected_metric_fourier_support"
        );

        c.dielectric.source = DielectricSource::External;
        c.geometry.objects.clear();
        let mut unsupported = c.clone();
        unsupported.bands.as_mut().unwrap().symmetry.policy = Some(SymmetryPolicy::Auto);
        assert!(Plan::new(unsupported, Platform::Native).is_err());
        let job = Plan::new(c.clone(), Platform::Native)
            .unwrap()
            .job(0)
            .unwrap();
        let accepted = external::execute_sampled(
            CpuBackend::<f64>::new(),
            "cpu",
            &job,
            vec![2.0; 13 * 13],
            None,
        )
        .unwrap();
        assert_eq!(accepted.metadata["accuracy"]["accepted"], true);
        assert_eq!(accepted.arrays["absolute_residuals"].shape, [1, 5]);
        c.bands
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap()
            .max_orthogonality_defect = 1e-30;
        let job = Plan::new(c, Platform::Native).unwrap().job(0).unwrap();
        let rejected = external::execute_sampled(
            CpuBackend::<f64>::new(),
            "cpu",
            &job,
            vec![2.0; 13 * 13],
            None,
        )
        .unwrap_err();
        assert_eq!(rejected.code, "accuracy_gate");
    }
}

#[test]
fn gamma_absolute_branch_and_missing_certificates_fail_closed() {
    let mut c = configuration(Polarization::TM);
    c.bands.as_mut().unwrap().path.points = Some(vec![vec![0.0, 0.0]]);
    let mut result = run(c).unwrap();
    let policy = BandAccuracy::default();
    // A normalized residual of one is possible at the exact zero eigenvalue.
    result.arrays.get_mut("residuals").unwrap().data[0] = 1.0;
    result.arrays.get_mut("absolute_residuals").unwrap().data[0] = 1e-11;
    assert!(accuracy::assess(&mut result, Some(&policy)).is_none());
    assert!(
        result.metadata["accuracy"]["absolute_only_acceptances"]
            .as_u64()
            .unwrap()
            > 0
    );
    result.arrays.get_mut("absolute_residuals").unwrap().data[0] = 1e-2;
    assert!(accuracy::assess(&mut result, Some(&policy)).is_some());
    result.arrays.get_mut("absolute_residuals").unwrap().data[0] = 1e-11;
    result.metadata["certification"]["b_orthogonality_per_point"] = json!([null]);
    assert!(accuracy::assess(&mut result, Some(&policy)).is_some());
    result.metadata["certification"]["b_orthogonality_per_point"] = json!([0.0]);
    result.arrays.get_mut("residuals").unwrap().data[0] = f64::NAN;
    assert!(accuracy::assess(&mut result, Some(&policy)).is_some());
}

#[test]
fn acceptance_requires_stopping_convergence_and_valid_limits() {
    let mut c = configuration(Polarization::TE);
    c.eigensolver.max_iterations = Some(1);
    assert_eq!(run(c.clone()).unwrap().metadata["converged"], false);
    c.bands.as_mut().unwrap().accuracy = Some(BandAccuracy {
        max_relative_residual: 10.0,
        max_absolute_residual: 1e6,
        max_orthogonality_defect: 10.0,
    });
    let failure = run(c.clone()).unwrap_err();
    assert_eq!(failure.diagnostic.code, "accuracy_gate");
    assert_eq!(
        failure.partial_result.unwrap().metadata["accuracy"]["first_violation"]["reason"],
        "eigenvalue-change stopping did not converge at every requested point"
    );
    for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        c.bands
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap()
            .max_relative_residual = limit;
        assert!(c.resolve().is_err());
    }
    let mut value = serde_json::to_value(configuration(Polarization::TE)).unwrap();
    value["bands"]["accuracy"] = json!({"max_relative_residual":1e-8});
    let config = Config::from_value(value).unwrap();
    assert_eq!(
        Config::from_toml(&config.to_toml().unwrap()).unwrap(),
        config
    );
    assert_eq!(
        config
            .bands
            .unwrap()
            .accuracy
            .unwrap()
            .max_absolute_residual,
        1e-7
    );
}
