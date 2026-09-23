//! Fixed-trial comparisons for shared sector Gram and Rayleigh-Ritz algebra.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        trial::{GramMode, RitzMode, SharedSectorTrial},
    },
};
use num_complex::Complex64 as C;
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(31);
    let columns: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let pol = match args.get(3).map(String::as_str).unwrap_or("TE") {
        "TE" => Polarization::TE,
        "TM" => Polarization::TM,
        _ => panic!("expected TE or TM"),
    };
    let group = match args.get(4).map(String::as_str).unwrap_or("p4m") {
        "pm" => PlaneGroup::Pm,
        "p4" => PlaneGroup::P4,
        "p4m" => PlaneGroup::P4m,
        _ => panic!("expected pm, p4 or p4m"),
    };
    let stage = args.get(5).map(String::as_str).unwrap_or("gram");
    assert!(matches!(stage, "gram" | "ritz"));
    let mode = args.get(6).map(String::as_str).unwrap_or("full");
    assert!(matches!(mode, "full" | "blocks"));
    let repetitions: usize = args.get(7).map(|s| s.parse().unwrap()).unwrap_or(100);
    assert!(repetitions > 0);
    let setup = std::time::Instant::now();
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let backend = CpuBackend::<f64>::new();
    let dielectric = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid,
        &Default::default(),
    );
    let mut op = ThetaOperator::new(backend.clone(), dielectric, pol, [0.0; 2]);
    let candidates: Vec<_> = (0..columns)
        .map(|j| {
            Field2D::from_vec(
                grid,
                (0..grid.len())
                    .map(|i| {
                        let a = (i + 1) as f64;
                        let b = (j + 1) as f64;
                        C::new((a * b * 0.723).sin(), (a * (b + 0.317) * 1.31).cos())
                    })
                    .collect(),
            )
        })
        .collect();
    let trial = SharedSectorTrial::prepare(
        &mut op,
        &lattice,
        [0.0; 2],
        group.group(),
        &candidates,
        columns,
        &Default::default(),
    )
    .unwrap();
    let full = trial.project(&backend, GramMode::Full, false);
    let block = trial.project(&backend, GramMode::Blocks, false);
    let norm = full.matrix().iter().map(C::norm_sqr).sum::<f64>().sqrt();
    let gram_error = full
        .matrix()
        .iter()
        .zip(block.matrix())
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / norm;
    let reference = block.solve(RitzMode::Full).unwrap();
    let independent = block.solve(RitzMode::Blocks).unwrap();
    let spectral_error = reference
        .eigenvalues
        .iter()
        .zip(&independent.eigenvalues)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0, f64::max);
    let dim = reference.dim;
    let cluster_error = (0..dim)
        .map(|j| {
            let overlap: f64 = (0..dim)
                .filter(|&i| {
                    (reference.eigenvalues[j] - independent.eigenvalues[i]).abs()
                        < 1e-8 * reference.eigenvalues[j].abs().max(1.0)
                })
                .map(|i| {
                    (0..dim)
                        .map(|r| {
                            reference.eigenvectors[r + dim * j].conj()
                                * independent.eigenvectors[r + dim * i]
                        })
                        .sum::<C>()
                        .norm_sqr()
                })
                .sum();
            (overlap - 1.0).abs()
        })
        .fold(0.0, f64::max);
    let setup_seconds = setup.elapsed().as_secs_f64();
    let gram_mode = if mode == "full" {
        GramMode::Full
    } else {
        GramMode::Blocks
    };
    let ritz_mode = if mode == "full" {
        RitzMode::Full
    } else {
        RitzMode::Blocks
    };
    let apply = || {
        if stage == "gram" {
            std::hint::black_box(trial.project(&backend, gram_mode, false));
            std::hint::black_box(trial.project(&backend, gram_mode, true));
        } else {
            std::hint::black_box(block.solve(ritz_mode).unwrap());
        }
    };
    for _ in 0..10 {
        apply();
    }
    blaze2d_core::profiler::reset();
    let start = std::time::Instant::now();
    for _ in 0..repetitions {
        apply();
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"stage":stage,"mode":mode,"grid":[n,n],"polarization":format!("{pol:?}"),"group":format!("{group:?}"),
        "repetitions":repetitions,"setup_seconds":setup_seconds,"seconds":seconds,"trial":trial.report,
        "gram_relative_error":gram_error,"spectral_relative_error":spectral_error,"cluster_span_error":cluster_error,
        "full_cross_block_defect":full.cross_block_defect(),"ritz_values":reference.eigenvalues,
        "valid":gram_error<2e-11 && spectral_error<1e-9 && cluster_error<1e-8,
        "counters":blaze2d_core::profiler::get_counters()})
    );
}
