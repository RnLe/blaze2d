//! Matched native/portable interface workload for benchmarks/symmetry/wasm_case.mjs.
//! Construction, validation, solving, certificates and requested field retention
//! are timed together. Optional JSON export includes allocation and checksum.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::profiler;
use blaze2d_interface as api;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{hint::black_box, time::Instant};

const DEFAULT_TOML: &str = include_str!("../../../examples/calculations/square-rods.toml");

fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

fn rows(data: &[f64], width: usize) -> Vec<Vec<f64>> {
    assert_eq!(data.len() % width, 0);
    data.chunks_exact(width).map(<[f64]>::to_vec).collect()
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let case = args.first().map(String::as_str).unwrap_or("square-path");
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(31);
    let bands: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(8);
    let pol = args.get(3).map(String::as_str).unwrap_or("TM");
    let mode = args.get(4).map(String::as_str).unwrap_or("off");
    let tolerance: f64 = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(1e-14);
    assert!(n > 0 && bands > 0 && tolerance.is_finite() && tolerance > 0.0);
    assert!(matches!(pol, "TM" | "TE"));
    assert!(matches!(
        mode,
        "off" | "pt" | "unitary" | "pt-unitary" | "coarse" | "pt-coarse"
    ));
    let audit_fields = args.get(6).is_some_and(|value| value == "audit");
    let export = case.ends_with("-export");
    let without_export = case.strip_suffix("-export").unwrap_or(case);
    let field_output = !without_export.ends_with("-values");
    let base_case = without_export
        .strip_suffix("-values")
        .unwrap_or(without_export);
    // Both adapters use the browser's validation limits and the same normalized
    // default input. Preparing that primitive input is outside both timers.
    let default = api::Plan::new(
        api::Config::from_toml(DEFAULT_TOML).unwrap(),
        api::Platform::Browser,
    )
    .unwrap();
    let mut cfg = serde_json::to_value(default.config).unwrap();
    cfg["grid"]["resolution"] = json!([n, n]);
    cfg["bands"]["count"] = json!(bands);
    cfg["bands"]["tracking"] = json!(false);
    cfg["polarization"] = json!(pol);
    cfg["eigensolver"]["precision"] = json!("f64");
    cfg["eigensolver"]["tolerance"] = json!(tolerance);
    cfg["eigensolver"]["max_iterations"] = json!(500);
    cfg["eigensolver"]["block_size"] = json!(0);
    cfg["bands"]["path"] = match base_case {
        "square-path" => json!({"preset":"square", "intervals_per_segment":3,
            "basis":"reciprocal_fractional"}),
        "square-star" => json!({"points":[[0.13,0.21],[-0.21,0.13],[-0.13,-0.21],
            [0.21,-0.13],[0.13,-0.21],[0.21,0.13],[-0.13,0.21],[-0.21,-0.13]],
            "basis":"reciprocal_fractional"}),
        "repeated" => json!({"points":[[0.13,0.21],[0.27,0.09],[0.13,0.21],[0.27,0.09]],
            "basis":"reciprocal_fractional"}),
        "square-gamma" => json!({"points":[[0.0,0.0]],"basis":"reciprocal_fractional"}),
        "square-generic" => json!({"points":[[0.13,0.17]],"basis":"reciprocal_fractional"}),
        "square-return" => json!({"points":[[0.0,0.0],[0.17,0.0],[0.0,0.0]],
            "basis":"reciprocal_fractional"}),
        _ => panic!("unknown case: {case}"),
    };
    let reuse = if mode.contains("unitary") {
        "unitary"
    } else {
        "off"
    };
    cfg["bands"]["symmetry"]["k_reuse"] = json!(reuse);
    cfg["bands"]["symmetry"]["arithmetic"] = json!(if mode.starts_with("pt") {
        "pt_real"
    } else {
        "spatial_complex"
    });
    if mode.ends_with("coarse") {
        // Largest odd grid at most half the fine width, fixed before sampling.
        assert!(
            n >= 7,
            "coarse benchmark requires a fine grid of at least seven"
        );
        let half = (n - 1) / 2;
        let coarse = half - usize::from(half % 2 == 0);
        cfg["bands"]["symmetry"]["coarse_start"] = json!({
            "resolution":[coarse,coarse],"max_seed_field_bytes":67108864
        });
    }
    cfg["results"]["eigenvectors"] = json!(field_output);
    let source = api::edit::apply_config(
        DEFAULT_TOML,
        api::Config::from_value(cfg.clone()).unwrap(),
        api::Platform::Browser,
    )
    .unwrap();
    let backend_name = if cfg!(feature = "native-linalg") {
        "native / faer"
    } else {
        "native / nalgebra"
    };
    profiler::reset();
    let begin = Instant::now();
    let plan = api::Plan::new(
        api::Config::from_toml(&source).unwrap(),
        api::Platform::Browser,
    )
    .unwrap();
    let job = plan.job(0).unwrap();
    let record = api::execute(CpuBackend::<f64>::new(), backend_name, &job, |_| {})
        .unwrap_or_else(|e| panic!("{}", serde_json::to_string(&e).unwrap()));
    drop(job);
    drop(plan);
    let interface_seconds = begin.elapsed().as_secs_f64();
    let encode_begin = Instant::now();
    let (serialized_bytes, serialized_sha256) = if export {
        let serialized = record.to_json().unwrap();
        let digest = format!("{:x}", Sha256::digest(serialized.as_bytes()));
        let size = serialized.len();
        // Include destruction of the transient export buffer, retaining the
        // physical result exactly as the JavaScript adapter does.
        drop(serialized);
        (size, Some(digest))
    } else {
        (0, None)
    };
    let serialization_seconds = if export {
        encode_begin.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let seconds = begin.elapsed().as_secs_f64();
    let rss = peak_rss_kib();
    // Diagnostic-only export runs after the primary clock and RSS snapshot.
    // It is never requested by the latency harness.
    let audit_material = if audit_fields {
        let plan = api::Plan::new(
            api::Config::from_toml(&source).unwrap(),
            api::Platform::Browser,
        )
        .unwrap();
        let job = plan.job(0).unwrap();
        let material = blaze2d_core::dielectric::Dielectric2D::from_geometry(
            &job.geometry(),
            job.grid(),
            &job.dielectric(),
        );
        json!({"resolution":job.resolved.resolution,
            "lattice_vectors":job.resolved.lattice_vectors,"sample_origin":[0.0,0.0],
            "epsilon":material.eps(),"inverse_epsilon":material.inv_eps(),
            "inverse_tensors":material.inv_eps_tensors()})
    } else {
        Value::Null
    };
    let arr = &record.arrays;
    let meta = &record.metadata;
    let nk = arr["frequencies"].shape[0];
    let ks = rows(&arr["k_points"].data, 2);
    let residuals = rows(&arr["residuals"].data, bands);
    let absolute = rows(&arr["absolute_residuals"].data, bands);
    let spectrum: Vec<_> = rows(&arr["frequencies"].data, bands)
        .into_iter()
        .map(|r| {
            r.into_iter()
                .map(|x| x * std::f64::consts::TAU)
                .collect::<Vec<_>>()
        })
        .collect();
    let sym = meta["symmetry"]["points"].as_array().unwrap();
    let field_count = arr.get("eigenvectors").map_or(0, |a| a.shape[1]);
    let points: Vec<_> = (0..nk)
        .map(|i| {
            json!({
                "k":ks[i], "iterations":meta["iterations"][i],
                "converged":meta["converged_per_point"][i],
                "residuals":residuals[i], "absolute_residuals":absolute[i],
                "b_orthogonality_defect":meta["certification"]["b_orthogonality_per_point"][i],
                "field_count":field_count, "symmetry":sym[i]
            })
        })
        .collect();
    let reused_points = sym.iter().filter(|s| !s["source_index"].is_null()).count();
    let path_success = sym.iter().all(|s| s["requested"] == reuse)
        && (if mode.starts_with("pt") {
            sym.iter().all(|s| {
                s["coordinates"]["representation"] == "pt_real_te" || !s["source_index"].is_null()
            })
        } else {
            sym.iter().all(|s| s["coordinates"].is_null())
        })
        && (!mode.ends_with("coarse")
            || (sym[0]["coarse_start"]["used_as_seed"] == true
                && sym.iter().skip(1).all(|s| s["coarse_start"].is_null())))
        && (!(mode.contains("unitary") && matches!(base_case, "square-star" | "repeated"))
            || reused_points > 0);
    let counters = profiler::get_all_counters();
    let stages = profiler::get_all_stage_counters();
    let timings: Value = serde_json::from_str(&profiler::get_profile_json()).unwrap();
    let arrays: Value = arr
        .iter()
        .map(|(name, a)| {
            (
                name.clone(),
                json!({
                    "dtype":a.dtype,"shape":a.shape,"payload_bytes":a.data.len()*8
                }),
            )
        })
        .collect();
    println!(
        "{}",
        json!({"case":case,"mode":mode,"grid":[n,n],"bands":bands,
            "polarization":pol,"precision":"f64","tolerance":tolerance,
            "seconds":seconds,"peak_rss_kib":rss,"expected_points":nk,
            "requested_path_success":path_success,"field_output":field_output,
            "arrays":arrays,"spectrum":spectrum,"points":points,
            "audit_arrays":if audit_fields {serde_json::to_value(arr).unwrap()} else {Value::Null},
            "audit_material":audit_material,
            "counters_available":!counters.is_empty(),"counters":counters,"stage_counters":stages,"timings":timings,
            "backend":backend_name,"configuration":cfg,"reused_points":reused_points,"build":api::build_info(),
            "interface_seconds":interface_seconds,"core_seconds":meta["elapsed_seconds"],
            "serialization_seconds":serialization_seconds,"serialized_bytes":serialized_bytes,
            "serialized_sha256":serialized_sha256,"scope":"complete native interface call under the common browser validation contract; requested physical field retention; optional JSON export with checksum; process startup separate"
        })
    );
    black_box(record);
}
