//! Fixed-pencil PT/sector interaction study with complete physical outputs.
//! This harness also builds on the frozen parent; path checks live in Python.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    drivers::bandstructure::{BandStructureJob, RunOptions, run_with_k_streaming},
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup, cache::SnapshotIdentity, fourier::Arithmetic, reuse::SymmetryOptions,
        sectors::SectorPolicy,
    },
};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 7, "case grid bands TM|TE mode tolerance");
    let case = args[1].as_str();
    let n: usize = args[2].parse().unwrap();
    let bands: usize = args[3].parse().unwrap();
    let pol = match args[4].as_str() {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("unknown polarization"),
    };
    let mode = args[5].as_str();
    let tolerance: f64 = args[6].parse().unwrap();
    assert!(n >= 5 && bands > 0 && bands < n * n);
    assert!(tolerance.is_finite() && tolerance > 0.0);
    let (group, path, partners) = match case {
        "gamma" => (PlaneGroup::P4m, vec![[0.0; 2]], true),
        "rotation" => (PlaneGroup::P4, vec![[0.0; 2]], false),
        "mirror" => (
            PlaneGroup::Pm,
            vec![[0.0; 2], [0.125, 0.0], [0.25, 0.0], [0.5, 0.0], [0.0; 2]],
            false,
        ),
        _ => panic!("unknown case"),
    };
    let (arithmetic, sectors) = match mode {
        "off" => (Arithmetic::SpatialComplex, false),
        "pt" => (Arithmetic::PtRealTe, false),
        "sectors" => (Arithmetic::SpatialComplex, true),
        "fourier-sectors" => (Arithmetic::FourierComplex, true),
        "pt-sectors" => (Arithmetic::PtRealTe, true),
        _ => panic!("unknown mode"),
    };
    let lattice = Lattice2D::square(1.0);
    let job = BandStructureJob {
        geom: Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid: Grid2D::new(n, n, 1.0, 1.0),
        pol,
        k_path: path,
        eigensolver: EigensolverConfig {
            n_bands: bands,
            tol: tolerance,
            max_iter: 500,
            ..Default::default()
        },
        dielectric: DielectricOptions::default(),
    };
    let symmetry = SymmetryOptions {
        arithmetic,
        sectors: if sectors {
            SectorPolicy::AdaptiveGroup
        } else {
            SectorPolicy::Off
        },
        plane_group: Some(group),
        reconstruct_partners: sectors && partners,
        sector_guard_budget: sectors.then_some(2),
        ..Default::default()
    };
    let mut physical = json!({
        "geometry": job.geom, "grid": [n,n], "grid_centered": false,
        "polarization": format!("{pol:?}"), "k_path": job.k_path,
        "eigensolver": job.eigensolver, "dielectric": job.dielectric,
        "precision": "f64", "output": "full_spatial_complex_fields",
        "band_tracking": false, "legacy_gamma_reuse": false,
    });
    let policies = json!({
        "arithmetic": symmetry.arithmetic, "sectors": symmetry.sectors,
        "plane_group": group, "reconstruct_partners": symmetry.reconstruct_partners,
        "sector_guard_budget": symmetry.sector_guard_budget,
        "sector_storage": symmetry.sector_storage, "sector_kernel": symmetry.sector_kernel,
        "correction_projection_interval": symmetry.correction_projection_interval,
        "qualification_tolerance": symmetry.qualification_tolerance,
        "relative_rank_tolerance": symmetry.relative_rank_tolerance,
        "max_relative_residual": symmetry.max_relative_residual,
        "max_zero_mode_absolute_residual": symmetry.max_zero_mode_absolute_residual,
        "max_orthogonality_defect": symmetry.max_orthogonality_defect,
        "max_sector_workspace_bytes": symmetry.max_sector_workspace_bytes,
        "k_reuse": symmetry.k_reuse, "share_sector_workspace": symmetry.share_sector_workspace,
    });
    let options = RunOptions {
        symmetry,
        reuse_gamma: false,
        disable_band_tracking: true,
        retain_eigenvectors: true,
        ..Default::default()
    };
    physical["run_options"] = json!({
        "preconditioner": options.precond_type,
        "resolved_preconditioner": options.precond_type.resolve_for_polarization(pol),
        "use_subspace_prediction": options.use_subspace_prediction,
        "use_extrapolation": options.use_extrapolation,
        "use_band_window_shift": options.use_band_window_shift,
        "band_window_blend": options.band_window_blend,
        "band_window_scale": options.band_window_scale,
    });
    let mut points = Vec::new();
    let start = std::time::Instant::now();
    let result = run_with_k_streaming(CpuBackend::<f64>::new(), &job, options, |p| points.push(p));
    let seconds = start.elapsed().as_secs_f64();
    // Read process peak before constructing duplicate identity/JSON buffers.
    let peak_rss_kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        });
    let counters = blaze2d_core::profiler::get_all_counters();
    let stage_counters = blaze2d_core::profiler::get_all_stage_counters();
    let profile: serde_json::Value =
        serde_json::from_str(&blaze2d_core::profiler::get_profile_json()).unwrap();
    let material = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);
    let points: Vec<_> = points.into_iter().map(|p| {
        let theta = ThetaOperator::new(CpuBackend::<f64>::new(), material.clone(), pol,
            lattice.reciprocal().fractional_to_cartesian(p.k_point));
        json!({
            "k": p.k_point, "symmetry": p.symmetry, "iterations": p.iterations,
            "converged": p.converged, "residuals": p.residuals,
            "absolute_residuals": p.absolute_residuals,
            "b_orthogonality_defect": p.b_orthogonality_defect,
            "field_count": p.eigenvectors.as_ref().map_or(0, Vec::len),
            "field_bits": p.eigenvectors.as_ref().map(|v| v.iter().map(|f|
                f.as_slice().iter().map(|z| [z.re.to_bits(),z.im.to_bits()]).collect::<Vec<_>>()).collect::<Vec<_>>()),
            "operator_identity": SnapshotIdentity::from_operator(&theta),
        })
    }).collect();
    println!(
        "{}",
        json!({
            "case": case, "mode": mode, "grid": [n,n], "bands": bands,
            "polarization": format!("{pol:?}"), "precision": "f64", "tolerance": tolerance,
            "max_iterations": 500, "physical_configuration": physical, "policies": policies,
            "expected_points": job.k_path.len(), "seconds": seconds, "peak_rss_kib": peak_rss_kib,
            "spectrum": result.bands, "points": points, "counters": counters,
            "stage_counters": stage_counters, "profile": profile,
        })
    );
}
