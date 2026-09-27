use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;

fn config(pol: Polarization) -> Config {
    let mut c = Config::default();
    c.polarization = pol;
    c.grid.resolution = Resolution::Uniform(13);
    c.eigensolver.tolerance = Some(1e-14);
    c.eigensolver.max_iterations = Some(500);
    c.results.eigenvectors = true;
    let b = c.bands.as_mut().unwrap();
    b.count = 6;
    b.tracking = false;
    b.path = KPath {
        points: Some(vec![vec![0., 0.], vec![0.17, 0.], vec![0., 0.]]),
        intervals_per_segment: None,
        ..Default::default()
    };
    b.symmetry.coarse_start = Some(CoarseStart {
        resolution: [7, 7],
        max_seed_field_bytes: 1 << 20,
    });
    c
}

fn run(c: Config) -> ResultRecord {
    let plan = Plan::new(c, Platform::Native).unwrap();
    execute(
        CpuBackend::<f64>::new(),
        "cpu",
        &plan.job(0).unwrap(),
        |_| {},
    )
    .unwrap()
}

fn certified(out: &ResultRecord) {
    assert_eq!(out.metadata["converged"], true);
    assert!(
        out.metadata["certification"]["b_orthogonality_defect"]
            .as_f64()
            .unwrap()
            < 1e-8
    );
    for ((&f, &relative), &absolute) in out.arrays["frequencies"]
        .data
        .iter()
        .zip(&out.arrays["residuals"].data)
        .zip(&out.arrays["absolute_residuals"].data)
    {
        assert!(if f.abs() < 1e-8 {
            absolute < 1e-7
        } else {
            relative < 1e-5
        });
    }
    assert_eq!(out.arrays["eigenvectors"].shape, [3, 6, 13, 13]);
    assert!(
        out.arrays["eigenvectors"]
            .data
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
fn coarse_first_point_preserves_certified_spectra_with_pt_and_sectors() {
    for pol in [Polarization::TM, Polarization::TE] {
        let mut control = config(pol);
        control.bands.as_mut().unwrap().symmetry.coarse_start = None;
        let reference = run(control);
        certified(&reference);
        for pt in [false, true] {
            for sectors in [false, true] {
                let mut c = config(pol);
                let symmetry = &mut c.bands.as_mut().unwrap().symmetry;
                if pt {
                    symmetry.arithmetic = SymmetryArithmetic::PtReal;
                }
                if sectors {
                    symmetry.sectors = SectorPolicy::AdaptiveGroup;
                    symmetry.plane_group = Some("p4m".into());
                    symmetry.sector_guard_budget = Some(4);
                }
                assert_eq!(Config::from_toml(&c.to_toml().unwrap()).unwrap(), c);
                let out = run(c);
                certified(&out);
                for (&a, &b) in out.arrays["frequencies"]
                    .data
                    .iter()
                    .zip(&reference.arrays["frequencies"].data)
                {
                    assert!((a - b).abs() < 1e-6 * a.abs().max(b.abs()).max(1.));
                }
                let points = out.metadata["symmetry"]["points"].as_array().unwrap();
                assert_eq!(points[0]["coarse_start"]["used_as_seed"], true);
                assert!(points[0]["coarse_start"]["iterations"].as_u64().unwrap() > 0);
                assert!(points[1..].iter().all(|p| p.get("coarse_start").is_none()));
                for (i, p) in points.iter().enumerate() {
                    let attempts: u64 = p["solver_attempts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|a| a["iterations"].as_u64().unwrap())
                        .sum();
                    assert_eq!(p["attempted_iterations"].as_u64().unwrap(), attempts);
                    assert_eq!(out.metadata["iterations"][i].as_u64().unwrap(), attempts);
                }
            }
        }
    }
}

#[test]
fn zero_coarse_payload_budget_preserves_the_unrestricted_result_exactly() {
    for pol in [Polarization::TM, Polarization::TE] {
        let mut cold = config(pol);
        cold.bands.as_mut().unwrap().symmetry.coarse_start = None;
        let reference = run(cold);
        let mut c = config(pol);
        c.bands
            .as_mut()
            .unwrap()
            .symmetry
            .coarse_start
            .as_mut()
            .unwrap()
            .max_seed_field_bytes = 0;
        let out = run(c);
        assert_eq!(out.arrays, reference.arrays);
        assert_eq!(out.metadata["iterations"], reference.metadata["iterations"]);
        let report = &out.metadata["symmetry"]["points"][0]["coarse_start"];
        assert_eq!(report["used_as_seed"], false);
        assert_eq!(report["iterations"], 0);
        assert!(
            report["rejection"]
                .as_str()
                .unwrap()
                .contains("payload budget")
        );
    }
}

#[test]
fn invalid_coarse_settings_reject_before_solving() {
    for resolution in [[0, 7], [3, 3], [14, 7], [13, 13]] {
        let mut c = config(Polarization::TM);
        c.bands
            .as_mut()
            .unwrap()
            .symmetry
            .coarse_start
            .as_mut()
            .unwrap()
            .resolution = resolution;
        if resolution == [3, 3] {
            c.eigensolver.block_size = 10;
        }
        assert!(Plan::new(c, Platform::Native).is_err());
    }
    let mut c = config(Polarization::TM);
    c.eigensolver.precision = Precision::F32;
    assert!(Plan::new(c, Platform::Native).is_err());
    let mut c = config(Polarization::TM);
    c.dielectric.source = DielectricSource::External;
    assert!(Plan::new(c, Platform::Native).is_err());
}

#[test]
fn planning_reserves_coarse_and_rank_workspaces_on_native_and_browser() {
    for platform in [Platform::Native, Platform::Browser] {
        let c = config(Polarization::TE);
        let enabled = Plan::new(c.clone(), platform)
            .unwrap()
            .summary
            .estimated_peak_bytes;
        let mut cold = c.clone();
        cold.bands.as_mut().unwrap().symmetry.coarse_start = None;
        let baseline = Plan::new(cold, platform)
            .unwrap()
            .summary
            .estimated_peak_bytes;
        assert!(enabled > baseline);
        let mut refused = c;
        refused
            .bands
            .as_mut()
            .unwrap()
            .symmetry
            .coarse_start
            .as_mut()
            .unwrap()
            .max_seed_field_bytes = 0;
        assert_eq!(
            Plan::new(refused, platform)
                .unwrap()
                .summary
                .estimated_peak_bytes,
            baseline
        );
    }
}
