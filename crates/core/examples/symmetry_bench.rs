//! Reproducible native symmetry experiments. Timing excludes JSON serialization.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    brillouin::{BrillouinPath, generate_path},
    dielectric::DielectricOptions,
    drivers::bandstructure::{BandStructureJob, RunOptions, run_with_k_streaming},
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
};
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("square-path");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(24);
    let bands: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8);
    let tolerance:f64=args.get(6).map(|s|s.parse().unwrap()).unwrap_or(1e-12);
    assert!(tolerance.is_finite() && tolerance>0.0);
    let pol = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TE" => Polarization::TE,
        "TM" => Polarization::TM,
        _ => panic!("expected TE or TM"),
    };
    let lattice = if case.starts_with("hex") { Lattice2D::hexagonal(1.0) } else { Lattice2D::square(1.0) };
    let path = match case {
        "gamma" => vec![[0.0;2]],
        "mirror-line" => vec![[0.0;2],[0.125,0.0],[0.25,0.0],[0.5,0.0],[0.0;2]],
        "shuffled-grid" => {
            let order=[0,24,4,20,12,2,22,10,14,6,18,8,16,1,23,5,19,9,15,3,21,7,17,11,13];
            order.iter().map(|&i|[0.025+0.065*(i%5) as f64,0.025+0.065*(i/5) as f64]).collect()
        },
        "square-path" => generate_path(&BrillouinPath::Square, 3),
        "hex-path" => generate_path(&BrillouinPath::Hexagonal, 3),
        "square-star" | "broken-star" => vec![[0.13,0.21],[-0.21,0.13],[-0.13,-0.21],[0.21,-0.13],[0.13,-0.21],[0.21,0.13],[-0.13,0.21],[-0.21,-0.13]],
        "repeated" => vec![[0.13,0.21],[0.27,0.09],[0.13,0.21],[0.27,0.09]],
        _ => panic!("unknown case"),
    };
    let geom = Geometry2D {
        lattice, eps_bg: 1.0,
        atoms: vec![BasisAtom { pos: if case == "broken-star" { [0.13, 0.07] } else { [0.0,0.0] }, radius: 0.2, eps_inside: 8.9 }],
    };
    let job = BandStructureJob {
        geom, grid: Grid2D::new(n,n,1.0,1.0), pol, k_path: path,
        eigensolver: EigensolverConfig { n_bands: bands, max_iter: 500, tol: tolerance, block_size: 0, record_diagnostics: false, k_index: None },
        dielectric: DielectricOptions::default(),
    };
    let mode = args.get(5).map(String::as_str).unwrap_or("off");
    let reuse = match mode { "off" | "rank" | "mirror" | "adaptive" | "guards" | "group" | "partners" | "workspace" | "projection2" | "projection4" | "fourier" | "pt_complex" | "pt_storage" | "pt_vectors" | "pt_gram" | "pt_dense" | "pt_tm" | "pt_te" | "pt_reference_vectors" | "orbit" | "orbit_mirror" | "mirror_tm" | "mirror_te" => blaze2d_core::symmetry::reuse::KReuse::Off, "identical" => blaze2d_core::symmetry::reuse::KReuse::Identical, "unitary" => blaze2d_core::symmetry::reuse::KReuse::Unitary, "time_reversal" | "route" => blaze2d_core::symmetry::reuse::KReuse::TimeReversal, _ => panic!("unknown reuse mode") };
    let requested_mode=mode;
    let share_workspace=matches!(mode,"workspace"|"projection2"|"projection4");
    let projection_interval=match mode {"projection2"=>2,"projection4"=>4,_=>1};
    let mode=if share_workspace {"partners"} else {mode};
    let options = RunOptions { symmetry: blaze2d_core::symmetry::reuse::SymmetryOptions { k_reuse: reuse, k_traversal:if mode=="route" {blaze2d_core::symmetry::traversal::KTraversal::Nearest} else {Default::default()}, share_sector_workspace:share_workspace, correction_projection_interval:projection_interval, sector_kernel:if mode=="mirror_te" {blaze2d_core::symmetry::reduced::ReducedKernel::MirrorTmTe} else if mode=="mirror_tm" {blaze2d_core::symmetry::reduced::ReducedKernel::MirrorTm} else {Default::default()}, sector_storage:if matches!(mode,"orbit"|"orbit_mirror"|"mirror_tm"|"mirror_te") {blaze2d_core::symmetry::reduced::SectorStorage::Orbit} else {Default::default()}, arithmetic:if mode=="pt_reference_vectors" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealReferenceVectors} else if mode=="pt_te" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealTe} else if mode=="pt_tm" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealTm} else if mode=="pt_dense" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealDense} else if mode=="pt_gram" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealGram} else if mode=="pt_vectors" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealVectors} else if mode=="pt_complex" {blaze2d_core::symmetry::fourier::Arithmetic::PtComplex} else if mode=="pt_storage" {blaze2d_core::symmetry::fourier::Arithmetic::PtRealStorage} else if mode=="fourier" {blaze2d_core::symmetry::fourier::Arithmetic::FourierComplex} else {Default::default()}, plane_group:if matches!(mode,"orbit_mirror"|"mirror_tm"|"mirror_te") {Some(blaze2d_core::symmetry::affine::PlaneGroup::Pm)} else if mode=="group" || mode=="partners" {Some(blaze2d_core::symmetry::affine::PlaneGroup::P4m)} else {None}, reconstruct_partners:mode=="partners", sector_guard_budget:if matches!(mode,"guards"|"orbit"|"orbit_mirror"|"mirror_tm"|"mirror_te") {Some(2)} else {None}, sectors:if matches!(mode,"orbit_mirror"|"mirror_tm"|"mirror_te") {blaze2d_core::symmetry::sectors::SectorPolicy::AdaptiveGroup} else if mode=="group" || mode=="partners" {blaze2d_core::symmetry::sectors::SectorPolicy::ConservativeGroup} else if mode=="adaptive" || (mode=="guards" || mode=="orbit") {blaze2d_core::symmetry::sectors::SectorPolicy::AdaptiveMirror} else if mode=="mirror" {blaze2d_core::symmetry::sectors::SectorPolicy::ConservativeMirror} else {Default::default()}, seed_transfer: if mode=="rank" {blaze2d_core::symmetry::seed::SeedTransfer::RankAware} else {Default::default()}, ..Default::default() }, reuse_gamma: false, disable_band_tracking: true, retain_eigenvectors: true, ..Default::default() };
    let mut points = Vec::new();
    let start = std::time::Instant::now();
    let result = run_with_k_streaming(CpuBackend::<f64>::new(), &job, options, |p| {
        points.push(json!({"k":p.k_point,"symmetry":p.symmetry,"iterations":p.iterations,"converged":p.converged,"residuals":p.residuals,"absolute_residuals":p.absolute_residuals,"b_orthogonality_defect":p.b_orthogonality_defect,"field_count":p.eigenvectors.as_ref().map_or(0,Vec::len)}));
    });
    let elapsed = start.elapsed().as_secs_f64();
    let rss_kib = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l|l.starts_with("VmHWM:")).and_then(|l|l.split_whitespace().nth(1)).and_then(|v|v.parse::<u64>().ok()));
    println!("{}",json!({"case":case,"mode":requested_mode,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),"precision":"f64","tolerance":tolerance,"max_iterations":500,"seconds":elapsed,"peak_rss_kib":rss_kib,"legacy_gamma_reuse":false,"spectrum":result.bands,"points":points,"counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),"profile":serde_json::from_str::<serde_json::Value>(&blaze2d_core::profiler::get_profile_json()).unwrap()}));
}
