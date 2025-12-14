use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{
    bandstructure::BandStructureJob,
    dielectric::Dielectric2D,
    eigensolver::{EigenOptions, GammaContext, PreconditionerKind, solve_lowest_eigenpairs},
    io::JobConfig,
    operator::ThetaOperator,
    polarization::Polarization,
};

const HEX_TM_RES24: &str = include_str!("../../../examples/hex_eps13_r0p3_tm_res24.toml");
const HEX_TM_RES32: &str = include_str!("../../../examples/hex_eps13_r0p3_tm.toml");
const HEX_TE_RES32: &str = include_str!("../../../examples/hex_eps13_r0p3_te.toml");

struct EigenBenchmarkScenario {
    name: &'static str,
    polarization: Polarization,
    dielectric: Dielectric2D,
    eigensolver: EigenOptions,
    k_points: Vec<KPointSample>,
}

struct KPointSample {
    label: &'static str,
    coords: [f64; 2],
}

fn scenario_from_example(
    name: &'static str,
    toml_src: &'static str,
    node_labels: [&'static str; 3],
) -> EigenBenchmarkScenario {
    let config: JobConfig = toml::from_str(toml_src)
        .unwrap_or_else(|err| panic!("failed to parse benchmark config {name}: {err}"));
    let segments = config
        .path
        .as_ref()
        .map(|spec| spec.segments_per_leg.max(1))
        .unwrap_or(1);
    let job: BandStructureJob = config.clone().into();
    assert!(
        job.k_path.len() > segments * 2,
        "k-path must cover Gamma-M-K"
    );
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid);
    let samples = [0usize, segments, segments * 2];
    let k_points = samples
        .into_iter()
        .zip(node_labels)
        .map(|(idx, label)| KPointSample {
            label,
            coords: job.k_path[idx],
        })
        .collect();
    EigenBenchmarkScenario {
        name,
        polarization: job.pol,
        dielectric,
        eigensolver: job.eigensolver,
        k_points,
    }
}

fn bloch_from_fraction(coords: [f64; 2]) -> [f64; 2] {
    [coords[0] * 2.0 * PI, coords[1] * 2.0 * PI]
}

fn bench_cpu_eigensolver_real(c: &mut Criterion) {
    let scenarios = vec![
        scenario_from_example("hex_tm_res24", HEX_TM_RES24, ["Gamma", "M", "K"]),
        scenario_from_example("hex_tm_res32", HEX_TM_RES32, ["Gamma", "M", "K"]),
        scenario_from_example("hex_te_res32", HEX_TE_RES32, ["Gamma", "M", "K"]),
    ];
    let backend = CpuBackend::new();
    let mut group = c.benchmark_group("cpu_eigensolver_real_data");
    group.sample_size(20);
    for scenario in &scenarios {
        let base_opts = scenario.eigensolver.clone();
        let jacobi_opts = EigenOptions {
            preconditioner: PreconditionerKind::RealSpaceJacobi,
            ..base_opts.clone()
        };
        let variants = [("baseline", base_opts), ("jacobi", jacobi_opts)];
        for sample in &scenario.k_points {
            let bloch = bloch_from_fraction(sample.coords);
            for (variant, eigen_opts) in variants.clone() {
                let bench_id =
                    BenchmarkId::new(format!("{}-{}", scenario.name, variant), sample.label);
                group.bench_function(bench_id, |b| {
                    b.iter(|| {
                        let mut theta = ThetaOperator::new(
                            backend.clone(),
                            scenario.dielectric.clone(),
                            scenario.polarization,
                            bloch,
                        );
                        let bloch_norm = (bloch[0] * bloch[0] + bloch[1] * bloch[1]).sqrt();
                        let gamma_context =
                            GammaContext::new(eigen_opts.gamma.should_deflate(bloch_norm));
                        let result = match eigen_opts.preconditioner {
                            PreconditionerKind::None => solve_lowest_eigenpairs(
                                &mut theta,
                                &eigen_opts,
                                None,
                                gamma_context,
                            ),
                            PreconditionerKind::RealSpaceJacobi => {
                                let mut preconditioner =
                                    theta.build_real_space_jacobi_preconditioner();
                                solve_lowest_eigenpairs(
                                    &mut theta,
                                    &eigen_opts,
                                    Some(&mut preconditioner),
                                    gamma_context,
                                )
                            }
                        };
                        black_box(result.iterations);
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(eigensolver_benches, bench_cpu_eigensolver_real);
criterion_main!(eigensolver_benches);
