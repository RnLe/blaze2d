use std::{f64::consts::PI, hint::black_box, sync::Arc};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mpb2d_backend_cpu::CpuBackend;
use mpb2d_core::{
    bandstructure::BandStructureJob,
    dielectric::Dielectric2D,
    eigensolver::{
        DeflationWorkspace, EigenOptions, GammaContext, PreconditionerKind,
        build_deflation_workspace, solve_lowest_eigenpairs,
    },
    field::Field2D,
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
    let dielectric = Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric);
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
    group.sample_size(10);
    for scenario in &scenarios {
        let base_opts = scenario.eigensolver.clone();
        let structured_opts = EigenOptions {
            preconditioner: PreconditionerKind::StructuredDiagonal,
            ..base_opts.clone()
        };
        let homogeneous_opts = EigenOptions {
            preconditioner: PreconditionerKind::HomogeneousJacobi,
            ..base_opts.clone()
        };
        let none_opts = EigenOptions {
            preconditioner: PreconditionerKind::None,
            ..base_opts.clone()
        };
        let deflation_variants = [("nodefl", false), ("defl", true)];
        for sample in &scenario.k_points {
            let bloch = bloch_from_fraction(sample.coords);
            let bloch_norm = (bloch[0] * bloch[0] + bloch[1] * bloch[1]).sqrt();
            for &(defl_label, defl_enabled) in &deflation_variants {
                let variants = [
                    ("structured", structured_opts.clone()),
                    ("homog", homogeneous_opts.clone()),
                    ("none", none_opts.clone()),
                ];
                for (variant, mut eigen_opts) in variants {
                    eigen_opts.deflation.enabled = defl_enabled;
                    let bench_id = BenchmarkId::new(
                        format!("{}-{}-{}", scenario.name, variant, defl_label),
                        sample.label,
                    );
                    group.bench_function(bench_id, |b| {
                        let gamma_context =
                            GammaContext::new(eigen_opts.gamma.should_deflate(bloch_norm));
                        b.iter(|| {
                            let mut theta = ThetaOperator::new(
                                backend.clone(),
                                scenario.dielectric.clone(),
                                scenario.polarization,
                                bloch,
                            );
                            let result = match eigen_opts.preconditioner {
                                PreconditionerKind::None => solve_lowest_eigenpairs(
                                    &mut theta,
                                    &eigen_opts,
                                    None,
                                    gamma_context,
                                    None,
                                    None,
                                    None,
                                    None,
                                ),
                                PreconditionerKind::HomogeneousJacobi => {
                                    let mut preconditioner =
                                        theta.build_homogeneous_preconditioner();
                                    solve_lowest_eigenpairs(
                                        &mut theta,
                                        &eigen_opts,
                                        Some(&mut preconditioner),
                                        gamma_context,
                                        None,
                                        None,
                                        None,
                                        None,
                                    )
                                }
                                PreconditionerKind::StructuredDiagonal => {
                                    let mut preconditioner =
                                        theta.build_structured_preconditioner();
                                    solve_lowest_eigenpairs(
                                        &mut theta,
                                        &eigen_opts,
                                        Some(&mut preconditioner),
                                        gamma_context,
                                        None,
                                        None,
                                        None,
                                        None,
                                    )
                                }
                            };
                            black_box(result.iterations);
                        });
                    });
                }
            }

            let mut everything_opts = scenario.eigensolver.clone();
            everything_opts.preconditioner = PreconditionerKind::StructuredDiagonal;
            everything_opts.deflation.enabled = true;
            everything_opts.deflation.max_vectors = everything_opts.n_bands;
            everything_opts.warm_start.enabled = true;
            everything_opts.warm_start.max_vectors = everything_opts.n_bands;
            let everything_gamma =
                GammaContext::new(everything_opts.gamma.should_deflate(bloch_norm));
            let warm_limit = everything_opts
                .warm_start
                .effective_limit(everything_opts.n_bands);
            let warm_modes: Arc<Vec<Field2D>> = {
                let mut theta = ThetaOperator::new(
                    backend.clone(),
                    scenario.dielectric.clone(),
                    scenario.polarization,
                    bloch,
                );
                let mut preconditioner = theta.build_structured_preconditioner();
                let primer = solve_lowest_eigenpairs(
                    &mut theta,
                    &everything_opts,
                    Some(&mut preconditioner),
                    everything_gamma,
                    None,
                    None,
                    None,
                    None,
                );
                Arc::new(primer.modes.into_iter().take(warm_limit).collect())
            };

            let bench_id = BenchmarkId::new(format!("{}-everything", scenario.name), sample.label);
            let dielectric = scenario.dielectric.clone();
            let polarization = scenario.polarization;
            let backend_clone = backend.clone();
            let opts = everything_opts.clone();
            let warm_modes_clone = warm_modes.clone();
            group.bench_function(bench_id, move |b| {
                let warm_modes = warm_modes_clone.clone();
                b.iter(|| {
                    let mut theta = ThetaOperator::new(
                        backend_clone.clone(),
                        dielectric.clone(),
                        polarization,
                        bloch,
                    );
                    let warm_slice: &[Field2D] = warm_modes.as_slice();
                    let deflation_workspace: Option<DeflationWorkspace<CpuBackend>> = {
                        let limit = opts.deflation.effective_limit(opts.n_bands);
                        if limit == 0 {
                            None
                        } else {
                            let refs: Vec<&Field2D> = warm_slice.iter().take(limit).collect();
                            if refs.is_empty() {
                                None
                            } else {
                                let workspace = build_deflation_workspace(&mut theta, refs);
                                if workspace.is_empty() {
                                    None
                                } else {
                                    Some(workspace)
                                }
                            }
                        }
                    };
                    let workspace_ref = deflation_workspace.as_ref();
                    let mut preconditioner = theta.build_fourier_diagonal_preconditioner();
                    let result = solve_lowest_eigenpairs(
                        &mut theta,
                        &opts,
                        Some(&mut preconditioner),
                        everything_gamma,
                        Some(warm_slice),
                        workspace_ref,
                        None,
                        None,
                    );
                    black_box((result.iterations, result.gamma_deflated));
                });
            });
        }
    }
    group.finish();
}

criterion_group!(eigensolver_benches, bench_cpu_eigensolver_real);
criterion_main!(eigensolver_benches);
