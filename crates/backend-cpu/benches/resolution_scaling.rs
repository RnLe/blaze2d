//! Resolution scaling benchmark for CPU and GPU backends.
//!
//! This benchmark measures the performance of the full band structure computation
//! for a square lattice configuration at various grid resolutions (24, 32, 48, 64, 128).
//! Both TM and TE polarizations are tested.
//!
//! The benchmark uses the standard square lattice with ε=13 background and r/a=0.3 air holes.
//!
//! Run with GPU backend:
//!   cargo bench --bench resolution_scaling --features cuda
//! Run CPU only:
//!   cargo bench --bench resolution_scaling

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{
    bandstructure::{self, BandStructureJob, RunOptions, Verbosity},
    dielectric::DielectricOptions,
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{PathType, standard_path},
};

#[cfg(feature = "cuda")]
use mpb2d_backend_cuda::CudaBackend;

/// Create a standard square lattice geometry with eps=13 background and r/a=0.3 air holes.
fn create_square_geometry() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        atoms: vec![BasisAtom {
            pos: [0.5, 0.5],
            radius: 0.3,
            eps_inside: 1.0,
        }],
        eps_bg: 13.0,
    }
}

/// Create a band structure job for the given resolution and polarization.
fn create_job(resolution: usize, polarization: Polarization) -> BandStructureJob {
    let geometry = create_square_geometry();
    
    // Generate k-path: Γ → X → M → Γ with 4 segments per leg
    let k_path = standard_path(&geometry.lattice, PathType::Square, 4);
    
    BandStructureJob {
        geom: geometry,
        grid: Grid2D::new(resolution, resolution, 1.0, 1.0),
        pol: polarization,
        k_path,
        eigensolver: EigensolverConfig {
            n_bands: 8,
            max_iter: 1000,
            tol: 1e-6,
            block_size: 0, // Auto
            record_diagnostics: false,
            k_index: None,
        },
        dielectric: DielectricOptions::default(),
    }
}

fn bench_resolution_scaling(c: &mut Criterion) {
    let resolutions = [24usize, 32, 48, 64, 128];
    let polarizations = [
        ("TM", Polarization::TM),
        ("TE", Polarization::TE),
    ];
    
    let mut group = c.benchmark_group("resolution_scaling");
    
    // Lean benchmark: minimal iterations for quick ballpark results
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(100));
    group.measurement_time(std::time::Duration::from_secs(1));
    
    // CPU benchmarks
    for &resolution in &resolutions {
        for (pol_name, polarization) in &polarizations {
            let job = create_job(resolution, *polarization);
            let backend = CpuBackend::new();
            
            let bench_name = format!("CPU/{}", pol_name);
            group.bench_function(BenchmarkId::new(&bench_name, resolution), |b| {
                b.iter(|| {
                    let result = bandstructure::run_with_options(
                        backend.clone(),
                        &job,
                        Verbosity::Quiet,
                        RunOptions::default(),
                    );
                    black_box(result)
                });
            });
        }
    }
    
    // GPU benchmarks (only when cuda feature is enabled)
    #[cfg(feature = "cuda")]
    {
        let cuda_backend = CudaBackend::new();
        
        for &resolution in &resolutions {
            for (pol_name, polarization) in &polarizations {
                let job = create_job(resolution, *polarization);
                let backend = cuda_backend.clone();
                
                let bench_name = format!("GPU/{}", pol_name);
                group.bench_function(BenchmarkId::new(&bench_name, resolution), |b| {
                    b.iter(|| {
                        let result = bandstructure::run_with_options(
                            backend.clone(),
                            &job,
                            Verbosity::Quiet,
                            RunOptions::default(),
                        );
                        black_box(result)
                    });
                });
            }
        }
    }
    
    group.finish();
}

criterion_group!(resolution_benches, bench_resolution_scaling);
criterion_main!(resolution_benches);
