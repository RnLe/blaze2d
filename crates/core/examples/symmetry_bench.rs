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
    let pol = match args.get(4).map(String::as_str).unwrap_or("TM") {
        "TE" => Polarization::TE,
        "TM" => Polarization::TM,
        _ => panic!("expected TE or TM"),
    };
    let lattice = if case.starts_with("hex") { Lattice2D::hexagonal(1.0) } else { Lattice2D::square(1.0) };
    let path = match case {
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
        eigensolver: EigensolverConfig { n_bands: bands, max_iter: 500, tol: 1e-12, block_size: 0, record_diagnostics: false, k_index: None },
        dielectric: DielectricOptions::default(),
    };
    let options = RunOptions { reuse_gamma: false, disable_band_tracking: true, retain_eigenvectors: true, ..Default::default() };
    let mut points = Vec::new();
    let start = std::time::Instant::now();
    let result = run_with_k_streaming(CpuBackend::<f64>::new(), &job, options, |p| {
        points.push(json!({"k":p.k_point,"iterations":p.iterations,"converged":p.converged,"residuals":p.residuals,"absolute_residuals":p.absolute_residuals,"b_orthogonality_defect":p.b_orthogonality_defect,"field_count":p.eigenvectors.as_ref().map_or(0,Vec::len)}));
    });
    let elapsed = start.elapsed().as_secs_f64();
    let rss_kib = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l|l.starts_with("VmHWM:")).and_then(|l|l.split_whitespace().nth(1)).and_then(|v|v.parse::<u64>().ok()));
    println!("{}",json!({"case":case,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),"precision":"f64","tolerance":1e-12,"max_iterations":500,"seconds":elapsed,"peak_rss_kib":rss_kib,"legacy_gamma_reuse":false,"spectrum":result.bands,"points":points,"counters":blaze2d_core::profiler::get_counters(),"stage_counters":blaze2d_core::profiler::get_stage_counters(),"profile":serde_json::from_str::<serde_json::Value>(&blaze2d_core::profiler::get_profile_json()).unwrap()}));
}
