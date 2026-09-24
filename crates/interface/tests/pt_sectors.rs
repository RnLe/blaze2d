use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;

fn config(pol: Polarization, group: &str) -> Config {
    let mut c = Config::default();
    c.polarization = pol;
    c.grid.resolution = Resolution::Uniform(13);
    c.eigensolver.tolerance = Some(1e-14);
    c.eigensolver.max_iterations = Some(500);
    c.results.eigenvectors = true;
    let b = c.bands.as_mut().unwrap();
    b.count = 6;
    b.path = KPath {
        points: Some(vec![vec![0.0, 0.0], vec![0.17, 0.0], vec![0.0, 0.0]]),
        intervals_per_segment: None,
        ..Default::default()
    };
    b.symmetry.sectors = SectorPolicy::AdaptiveGroup;
    b.symmetry.plane_group = Some(group.into());
    b.symmetry.arithmetic = SymmetryArithmetic::PtReal;
    b.symmetry.sector_guard_budget = Some(4);
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
    assert_eq!(
        out.metadata["certification"]["source"],
        "fresh_operator_application"
    );
    assert!(
        out.metadata["certification"]["b_orthogonality_defect"]
            .as_f64()
            .unwrap()
            < 1e-7
    );
    for ((&f, &relative), &absolute) in out.arrays["frequencies"]
        .data
        .iter()
        .zip(&out.arrays["residuals"].data)
        .zip(&out.arrays["absolute_residuals"].data)
    {
        assert!(
            if f.abs() < 1e-8 {
                absolute < 1e-8
            } else {
                relative < 1e-6
            },
            "frequency {f}, relative residual {relative}, absolute residual {absolute}"
        );
    }
    let restored: ResultRecord = serde_json::from_str(&out.to_json().unwrap()).unwrap();
    assert_eq!(out.arrays, restored.arrays);
}

#[test]
fn public_pt_sectors_preserve_output_contract_and_report_each_representation() {
    for pol in [Polarization::TM, Polarization::TE] {
        for group in ["p4", "p4m"] {
            let c = config(pol, group);
            assert_eq!(Config::from_toml(&c.to_toml().unwrap()).unwrap(), c);
            let mut control = c.clone();
            control.bands.as_mut().unwrap().symmetry.arithmetic =
                SymmetryArithmetic::SpatialComplex;
            let control = run(control);
            let out = run(c);
            certified(&out);
            assert_eq!(out.arrays["eigenvectors"].shape, [3, 6, 13, 13]);
            assert_eq!(out.arrays["eigenvectors"].dtype, DType::Complex128);
            // Tracking may choose a different basis/order inside a degenerate
            // cluster. Compare complete sorted point spectra, not field phases.
            for (a, b) in out.arrays["frequencies"]
                .data
                .chunks(6)
                .zip(control.arrays["frequencies"].data.chunks(6))
            {
                let mut a = a.to_vec();
                let mut b = b.to_vec();
                a.sort_by(f64::total_cmp);
                b.sort_by(f64::total_cmp);
                for (x, y) in a.iter().zip(&b) {
                    assert!((x - y).abs() < 1e-7);
                }
            }
            for (index, point) in out.metadata["symmetry"]["points"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                if group == "p4" && index == 1 {
                    assert!(
                        point["sectors"].is_null(),
                        "generic point has no nontrivial C4 little group"
                    );
                    let attempts = point["solver_attempts"].as_array().unwrap();
                    assert_eq!(attempts[0]["method"], "fourier");
                    if point["coordinates"].is_null() {
                        // A portable TM continuation currently fails the PT
                        // certificate. Preserve its failed work and require the
                        // certified unrestricted fallback, not silent omission.
                        assert_eq!(attempts.len(), 2);
                        assert_eq!(attempts[0]["retained"], false);
                        assert!(attempts[0]["iterations"].as_u64().unwrap() > 0);
                        assert!(
                            attempts[0]["rejection"]
                                .as_str()
                                .unwrap()
                                .contains("fresh physical quality gates")
                        );
                        assert_eq!(attempts[1]["method"], "unrestricted");
                        assert_eq!(attempts[1]["retained"], true);
                    } else {
                        assert_eq!(point["coordinates"]["representation"], "pt_real_te");
                        assert_eq!(attempts.len(), 1);
                        assert_eq!(attempts[0]["retained"], true);
                    }
                    assert_eq!(
                        point["attempted_iterations"].as_u64().unwrap(),
                        attempts
                            .iter()
                            .map(|a| a["iterations"].as_u64().unwrap())
                            .sum::<u64>()
                    );
                    continue;
                }
                let sectors = point["sectors"]["sectors"].as_array().unwrap_or_else(|| {
                    panic!(
                        "missing {pol:?} {group} point {index} sectors: {}",
                        point["rejected"]
                    )
                });
                assert!(!point["rejected"].to_string().contains("not combined"));
                assert_eq!(point["sectors"]["kernel_owners"], usize::from(index != 1));
                assert!(
                    sectors
                        .iter()
                        .any(|s| s["coordinates"]["representation"] == "pt_real_te")
                );
                for sector in sectors {
                    assert!(
                        sector["coordinates"]["final_sector_projection_defect"]
                            .as_f64()
                            .unwrap()
                            .is_finite()
                    );
                    if sector["coordinate_rejection"].is_null() {
                        assert_eq!(sector["coordinates"]["representation"], "pt_real_te");
                        assert_eq!(sector["vector_column_bytes"], 85 * 16);
                    } else {
                        assert_eq!(sector["coordinates"]["representation"], "fourier_complex");
                        assert_eq!(sector["vector_column_bytes"], 169 * 16);
                        assert!(sector["pt_projector_defect"].as_f64().unwrap() > 1e-3);
                    }
                }
                if group == "p4" && index != 1 {
                    assert_eq!(
                        sectors
                            .iter()
                            .filter(|s| !s["coordinate_rejection"].is_null())
                            .count(),
                        2
                    );
                }
                let attempts = point["solver_attempts"].as_array().unwrap();
                assert_eq!(attempts.len(), 1);
                assert_eq!(attempts[0]["method"], "sectors");
                assert_eq!(attempts[0]["retained"], true);
                assert_eq!(point["attempted_iterations"], attempts[0]["iterations"]);
            }
        }
    }
}

#[test]
fn rejected_sector_budget_keeps_exact_unrestricted_pt_output_and_counts_attempt() {
    for pol in [Polarization::TM, Polarization::TE] {
        let mut c = config(pol, "p4m");
        c.bands
            .as_mut()
            .unwrap()
            .symmetry
            .max_sector_workspace_bytes = 0;
        let mut control = c.clone();
        control.bands.as_mut().unwrap().symmetry.sectors = SectorPolicy::Off;
        control.bands.as_mut().unwrap().symmetry.sector_guard_budget = None;
        let a = Plan::new(c.clone(), Platform::Native).unwrap();
        let b = Plan::new(control.clone(), Platform::Native).unwrap();
        assert_eq!(
            a.summary.estimated_peak_bytes,
            b.summary.estimated_peak_bytes
        );
        let fallback = run(c);
        let control = run(control);
        certified(&fallback);
        assert_eq!(fallback.arrays, control.arrays);
        assert_eq!(
            fallback.metadata["iterations"],
            control.metadata["iterations"]
        );
        for point in fallback.metadata["symmetry"]["points"].as_array().unwrap() {
            assert!(point["sectors"].is_null());
            assert_eq!(point["coordinates"]["representation"], "pt_real_te");
            let attempts = point["solver_attempts"].as_array().unwrap();
            assert_eq!(attempts.len(), 2);
            assert_eq!(attempts[0]["method"], "sectors");
            assert_eq!(attempts[0]["iterations"], 0);
            assert_eq!(attempts[0]["retained"], false);
            assert!(
                attempts[0]["rejection"]
                    .as_str()
                    .unwrap()
                    .contains("above budget 0")
            );
            assert_eq!(attempts[1]["method"], "fourier");
            assert_eq!(point["attempted_iterations"], attempts[1]["iterations"]);
        }
    }
}
