use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;
use serde_json::json;

fn config() -> Config {
    let mut c = Config::from_toml(include_str!(
        "../../../examples/calculations/metric-support.toml"
    ))
    .unwrap();
    c.grid.resolution = Resolution::Uniform(17);
    let b = c.bands.as_mut().unwrap();
    b.count = 5;
    b.discretization=serde_json::from_value(json!({"kind":"metric_ball","angular_cutoff":std::f64::consts::TAU*3.1,"max_estimated_payload_bytes":1<<26})).unwrap();
    b.path = KPath {
        points: Some(vec![
            vec![0.0, 0.0],
            vec![0.17, 0.09],
            vec![2.0 / 3.0, 1.0 / 3.0],
        ]),
        intervals_per_segment: None,
        ..Default::default()
    };
    c
}

#[test]
fn public_config_schema_roundtrip_and_projection_metadata_are_explicit() {
    let c = config();
    let restored = Config::from_toml(&c.to_toml().unwrap()).unwrap();
    assert_eq!(restored, c);
    assert!(schema().to_value().to_string().contains("metric_ball"));
    assert!(
        !serde_json::to_value(Config::default()).unwrap()["bands"]
            .as_object()
            .unwrap()
            .contains_key("discretization")
    );
    for pol in [Polarization::TM, Polarization::TE] {
        let mut c = c.clone();
        c.polarization = pol;
        let plan = Plan::new(c, Platform::Native).unwrap();
        let mut events = Vec::new();
        let out = execute(
            CpuBackend::<f64>::new(),
            "cpu",
            &plan.job(0).unwrap(),
            |e| events.push(e),
        )
        .unwrap();
        assert_eq!(out.arrays["frequencies"].shape, [3, 5]);
        assert_eq!(out.arrays["eigenvectors"].shape, [3, 5, 17, 17]);
        assert_eq!(
            out.metadata["certification"]["pencil"],
            "selected_metric_fourier_support"
        );
        assert_eq!(out.metadata["discretization"]["completed_points"], 3);
        for (i, r) in out.metadata["discretization"]["points"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_eq!(r["support"]["version"], "metric-plane-waves/1");
            assert_eq!(r["support"]["bit_encoding"], "hexadecimal_u64");
            for axis in 0..2 {
                assert_eq!(
                    u64::from_str_radix(r["support"]["k_bits"][axis].as_str().unwrap(), 16)
                        .unwrap(),
                    out.arrays["k_points"].data[2 * i + axis].to_bits()
                );
            }
            assert_eq!(r["selected"]["accepted"], true);
            assert_eq!(
                r["selected"]["projected"]["residuals"],
                json!(&out.arrays["residuals"].data[i * 5..(i + 1) * 5])
            );
            assert!(r["selected"]["full_rectangle"].is_object());
        }
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Progress { .. }))
                .count(),
            3
        );
        let restored: ResultRecord = serde_json::from_str(&out.to_json().unwrap()).unwrap();
        assert_eq!(out.arrays, restored.arrays);
    }
}

#[test]
fn unqualified_material_retains_same_support_and_reports_fallback() {
    let mut c = config();
    c.dielectric.smoothing = Smoothing::Analytic;
    c.bands.as_mut().unwrap().path.points = Some(vec![vec![0.0, 0.0]]);
    let plan = Plan::new(c, Platform::Native).unwrap();
    let out = execute(
        CpuBackend::<f64>::new(),
        "cpu",
        &plan.job(0).unwrap(),
        |_| {},
    )
    .unwrap();
    let point = &out.metadata["discretization"]["points"][0];
    assert_eq!(point["qualification"]["authorized"], false);
    assert!(
        point["qualification"]["rejection"]
            .as_str()
            .unwrap()
            .contains("material")
    );
    assert!(point["selected"]["coefficient_count"].as_u64().unwrap() < 17 * 17);
    assert_eq!(
        out.metadata["symmetry"]["points"][0]["solver_attempts"][1]["method"],
        "metric_support_unreduced"
    );
}

#[test]
fn failed_solve_exports_rectangular_partial_result_and_all_work() {
    let mut c = config();
    c.grid.resolution = Resolution::Uniform(33);
    c.eigensolver.max_iterations = Some(1);
    let b = c.bands.as_mut().unwrap();
    b.count = 8;
    if let Discretization::MetricBall { angular_cutoff, .. } = &mut b.discretization {
        *angular_cutoff = std::f64::consts::TAU * 7.0;
    }
    let plan = Plan::new(c, Platform::Native).unwrap();
    let failure = execute(
        CpuBackend::<f64>::new(),
        "cpu",
        &plan.job(0).unwrap(),
        |_| {},
    )
    .unwrap_err();
    assert_eq!(failure.diagnostic.code, "metric_support");
    let out = failure.partial_result.unwrap();
    out.to_json().unwrap();
    assert_eq!(out.metadata["converged"], false);
    assert_eq!(out.arrays["frequencies"].shape, [1, 8]);
    assert_eq!(out.arrays["k_points"].shape, [1, 2]);
    assert_eq!(out.metadata["discretization"]["requested_points"], 3);
    assert_eq!(out.metadata["failed_point"]["k_index"], 0);
    let ledger = &out.metadata["failed_point"]["symmetry"];
    let sum: u64 = ledger["solver_attempts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["iterations"].as_u64().unwrap())
        .sum();
    assert_eq!(ledger["attempted_iterations"], sum);
    assert!(sum > ledger["retained_iterations"].as_u64().unwrap());
}

#[test]
fn incompatible_requests_and_clipped_supports_are_rejected_at_configuration() {
    let original = config();
    for change in [
        "precision",
        "reuse",
        "pt",
        "storage",
        "traversal",
        "external",
        "rectangle_budget",
        "cutoff",
    ] {
        let mut c = original.clone();
        match change {
            "precision" => c.eigensolver.precision = Precision::F32,
            "reuse" => c.bands.as_mut().unwrap().symmetry.k_reuse = KReuse::Identical,
            "pt" => c.bands.as_mut().unwrap().symmetry.arithmetic = SymmetryArithmetic::PtReal,
            "storage" => {
                c.bands.as_mut().unwrap().symmetry.sector_storage = SectorStorage::FullGrid
            }
            "traversal" => c.bands.as_mut().unwrap().symmetry.k_traversal = KTraversal::Nearest,
            "external" => c.dielectric.source = DielectricSource::External,
            "rectangle_budget" => c.bands.as_mut().unwrap().symmetry.max_sector_workspace_bytes=1,
            "cutoff" => {
                if let Discretization::MetricBall { angular_cutoff, .. } =
                    &mut c.bands.as_mut().unwrap().discretization
                {
                    *angular_cutoff = 1e3;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            Plan::new(c, Platform::Native).is_err(),
            "{change} must not be silently ignored"
        );
    }
    let mut c = original;
    c.bands.as_mut().unwrap().count = 100;
    assert!(Plan::new(c, Platform::Native).is_err());
}

#[test]
fn planning_reserves_growth_budget_reports_and_full_output_independently() {
    let mut c = config();
    c.results.eigenvectors = false;
    let a = Plan::new(c.clone(), Platform::Native).unwrap();
    assert!(a.summary.estimated_peak_bytes > 1 << 26);
    if let Discretization::MetricBall {
        max_estimated_payload_bytes,
        ..
    } = &mut c.bands.as_mut().unwrap().discretization
    {
        *max_estimated_payload_bytes += 1 << 20;
    }
    let b = Plan::new(c.clone(), Platform::Native).unwrap();
    assert_eq!(
        b.summary.estimated_peak_bytes - a.summary.estimated_peak_bytes,
        1 << 20
    );
    c.results.eigenvectors = true;
    let exported = Plan::new(c, Platform::Native).unwrap();
    assert_eq!(
        exported.summary.estimated_peak_bytes - b.summary.estimated_peak_bytes,
        3 * 5 * 17 * 17 * 32
    );
    assert!(
        exported
            .summary
            .memory_estimate_scope
            .contains("admission budget")
    );
}

#[test]
fn exhausted_driver_budget_returns_an_empty_well_formed_partial_record() {
    let mut c = config();
    if let Discretization::MetricBall {
        max_estimated_payload_bytes,
        ..
    } = &mut c.bands.as_mut().unwrap().discretization
    {
        *max_estimated_payload_bytes = 1;
    }
    let plan = Plan::new(c, Platform::Native).unwrap();
    let mut progress = 0;
    let failure = execute(
        CpuBackend::<f64>::new(),
        "cpu",
        &plan.job(0).unwrap(),
        |e| {
            if matches!(e, Event::Progress { .. }) {
                progress += 1;
            }
        },
    )
    .unwrap_err();
    assert!(failure.diagnostic.message.contains("payload allowance"));
    let out = failure.partial_result.unwrap();
    out.to_json().unwrap();
    assert_eq!(out.arrays["frequencies"].shape, [0, 5]);
    assert_eq!(out.arrays["eigenvectors"].shape, [0, 5, 17, 17]);
    assert_eq!(out.metadata["discretization"]["completed_points"], 0);
    assert_eq!(out.metadata["converged"], false);
    assert_eq!(progress, 0);
}
