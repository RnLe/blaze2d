use blaze2d_interface::*;
use serde_json::json;

#[test]
fn public_example_round_trips_and_resolves_defaults() {
    let source = include_str!("fixtures/bands.toml");
    let config = Config::from_toml(source).unwrap();
    let plan = Plan::new(config.clone(), Platform::Native).unwrap();
    let resolved = plan.job(0).unwrap().resolved;
    assert_eq!(resolved.resolution, [32, 32]);
    assert_eq!(resolved.k_points_fractional.len(), 46);
    assert_eq!(resolved.k_points_fractional.first(), resolved.k_points_fractional.last());
    assert_eq!(resolved.k_label_indices, [0, 15, 30, 45]);
    assert_eq!(resolved.config.eigensolver.tolerance, Some(1e-6));
    let reparsed = Config::from_toml(&config.to_toml().unwrap()).unwrap();
    assert_eq!(config, reparsed);
    let from_json = Config::from_value(serde_json::to_value(config).unwrap()).unwrap();
    assert_eq!(from_json.resolve().unwrap(), resolved);
}

#[test]
fn reciprocal_coordinates_use_the_full_oblique_metric() {
    let mut config = Config::default();
    config.geometry.lattice = Lattice { kind: LatticeKind::Custom,
        vectors: Some(vec![vec![2.0, 0.0], vec![0.7, 1.5]]), a: None, b: None, angle_deg: None };
    config.grid.resolution = Resolution::Axes(vec![48, 32]);
    config.bands.as_mut().unwrap().path = KPath {
        points: Some(vec![vec![0.0, 0.0], vec![0.5, 0.0]]), intervals_per_segment: None,
        ..KPath::default()
    };
    let r = config.resolve().unwrap();
    assert_eq!(r.resolution, [48, 32]);
    let k = r.k_points_cartesian[1];
    assert!((k[0] - std::f64::consts::PI / 2.0).abs() < 1e-12);
    assert!((k[1] + 0.7 * std::f64::consts::PI / 3.0).abs() < 1e-12);
    assert!((r.distances[1] - k[0].hypot(k[1])).abs() < 1e-12);
    let q = cartesian_to_fractional(k, r.lattice_vectors);
    assert!((q[0] - 0.5).abs() < 1e-12 && q[1].abs() < 1e-12);
}

#[test]
fn triangular_k_is_equidistant_from_three_reciprocal_sites() {
    let mut config = Config::default();
    config.geometry.lattice.kind = LatticeKind::Triangular;
    let r = config.resolve().unwrap();
    let k = r.k_points_cartesian[r.k_label_indices[2]];
    let b = reciprocal_vectors(r.lattice_vectors);
    let norm = k[0].hypot(k[1]);
    assert!((norm - 4.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
    for site in [b[0], [b[0][0] + b[1][0], b[0][1] + b[1][1]]] {
        assert!(((k[0] - site[0]).hypot(k[1] - site[1]) - norm).abs() < 1e-12);
    }
}

#[test]
fn sweep_jobs_match_individual_calculations() {
    let mut config = Config::default();
    config.sweeps = vec![
        Sweep { name: "radius".into(), target: "geometry.objects.rod.radius".into(), values: None,
            linspace: Some(Linspace { start: 0.3, stop: 0.1, count: 3 }) },
        Sweep { name: "pol".into(), target: "polarization".into(),
            values: Some(vec![json!("TM"), json!("TE")]), linspace: None },
    ];
    let plan = Plan::new(config, Platform::Native).unwrap();
    assert_eq!(plan.summary.jobs, 6);
    for i in 0..6 {
        let job = plan.job(i).unwrap();
        assert_eq!(job.multi_index, vec![i / 2, i % 2]);
        let mut single = Config::default();
        single.geometry.objects[0].radius = job.sweep["radius"].as_f64().unwrap();
        single.polarization = if i % 2 == 0 { Polarization::TM } else { Polarization::TE };
        assert_eq!(job.resolved, single.resolve().unwrap());
    }
    assert_eq!(plan.job(5).unwrap().resolved.config.geometry.objects[0].radius, 0.1);
}

#[test]
fn linspace_planning_does_not_expand_a_large_study() {
    let mut config = Config::default();
    config.sweeps.push(Sweep { name: "radius".into(), target: "geometry.objects.rod.radius".into(), values: None,
        linspace: Some(Linspace { start: 0.1, stop: 0.3, count: 1_000_000_000 }) });
    let plan = Plan::new(config, Platform::Native).unwrap();
    assert_eq!(plan.summary.jobs, 1_000_000_000);
    assert_eq!(plan.job(999_999_999).unwrap().resolved.config.geometry.objects[0].radius, 0.3);
}

#[test]
fn memory_preflight_covers_joint_grid_and_solver_block_sweeps() {
    let mut config = Config::default();
    config.sweeps = vec![
        Sweep { name: "grid".into(), target: "grid.resolution".into(), linspace: None,
            values: Some(vec![json!([32, 32]), json!([128, 96])]) },
        Sweep { name: "block".into(), target: "eigensolver.block_size".into(), linspace: None,
            values: Some(vec![json!(8), json!(64)]) },
    ];
    let plan = Plan::new(config, Platform::Browser).unwrap();
    for index in 0..plan.summary.jobs {
        let mut single = plan.job(index).unwrap().resolved.config;
        single.sweeps.clear();
        let one = Plan::new(single, Platform::Browser).unwrap();
        assert!(plan.summary.estimated_peak_bytes >= one.summary.estimated_peak_bytes);
    }
}

#[test]
fn invalid_combinations_are_rejected_before_running() {
    let source = include_str!("fixtures/bands.toml");
    assert!(Config::from_toml(&source.replace("count = 8", "coutn = 8")).is_err());
    let mut config = Config::default();
    config.bands.as_mut().unwrap().path.points = Some(vec![vec![0., 0.]]);
    assert!(config.resolve().is_err());
    let mut config = Config::default();
    config.eigensolver.precision = Precision::F32;
    assert_eq!(Plan::new(config.clone(), Platform::Browser).unwrap_err().code, "unsupported_precision");
    assert!(Plan::new(config, Platform::Native).is_ok());
    let mut config = Config::default();
    config.dimension = 3;
    assert_eq!(config.resolve().unwrap_err().code, "unsupported_dimension");
}

#[test]
fn operator_registry_and_stencil_counts_include_the_full_window() {
    let config = Config::from_toml(include_str!("fixtures/operators.toml")).unwrap();
    let plan = Plan::new(config, Platform::Browser).unwrap();
    assert_eq!(plan.summary.jobs, 4);
    assert_eq!(plan.summary.solves, 36);
    assert_eq!(plan.summary.solved_bands, 7);
    let job = plan.job(3).unwrap();
    assert_eq!(job.registry, [0.25, 0.25]);
    assert_eq!(job.registry_index, Some(1));
    assert_eq!(job.multi_index, [1]);
    assert_eq!(job.resolved.config.geometry.objects[0].center, [0.0, 0.0]);
    assert_eq!(job.resolved.config.eigensolver.tolerance, Some(1e-8));
}

#[test]
fn coupled_band_grid_and_block_axes_are_validated_as_complete_jobs() {
    let mut config = Config::default();
    config.eigensolver.block_size = 8;
    config.sweeps = vec![
        Sweep { name: "bands".into(), target: "bands.count".into(), linspace: None, values: Some(vec![json!(12), json!(16)]) },
        Sweep { name: "block".into(), target: "eigensolver.block_size".into(), linspace: None, values: Some(vec![json!(16), json!(24)]) },
    ];
    let plan = Plan::new(config.clone(), Platform::Native).unwrap();
    for i in 0..4 { assert!(plan.job(i).is_ok()); }
    config.sweeps[1].values = Some(vec![json!(8), json!(24)]);
    assert!(Plan::new(config.clone(), Platform::Native).is_err());
    config.bands.as_mut().unwrap().count = 32;
    config.eigensolver.block_size = 32;
    config.sweeps[0].values = Some(vec![json!(1), json!(2)]);
    config.sweeps[1].values = Some(vec![json!(3), json!(4)]);
    config.sweeps.push(Sweep { name: "grid".into(), target: "grid.resolution".into(), linspace: None, values: Some(vec![json!([4, 4])]) });
    let plan = Plan::new(config, Platform::Native).unwrap();
    for i in 0..4 { assert!(plan.job(i).is_ok()); }
}
