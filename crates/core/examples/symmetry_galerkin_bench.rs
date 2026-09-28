//! New-discretization convergence and covariance diagnostics. Individual times
//! here are descriptive; comparing a different support is not a symmetry gain.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    drivers::single_solve::{SingleSolveJob, solve},
    eigensolver::{EigensolverConfig, refine::certify_without_refinement},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        galerkin::{
            GalerkinApplication as Application, GalerkinOperator, dense_pencil, solve_dense,
            solve_iterative,
        },
        support::{CoefficientAction, FourierSupport},
    },
};
use num_complex::Complex64 as C;
use serde_json::json;
use std::time::Instant;

fn common_guarded_prefix(reference: &[f64], candidate: &[f64], limit: usize) -> usize {
    let mut prefix = limit
        .min(reference.len().saturating_sub(1))
        .min(candidate.len().saturating_sub(1));
    while prefix > 0
        && [reference, candidate].iter().any(|values| {
            values[prefix] - values[prefix - 1] <= 1e-6 * values[prefix].abs().max(1.0)
        })
    {
        prefix -= 1;
    }
    prefix
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |i, default| args.get(i).map(String::as_str).unwrap_or(default);
    let case = get(1, "hex-k");
    let n: usize = get(2, "31").parse().unwrap();
    let bands: usize = get(3, "8").parse().unwrap();
    let pol = match get(4, "TM") {
        "TM" => Polarization::TM,
        "TE" => Polarization::TE,
        _ => panic!("invalid polarization"),
    };
    let method = get(5, "iterative");
    let cutoff: f64 = get(6, "5.0").parse().unwrap();
    let material = get(7, "analytic");
    let probe = get(8, "no-probe") == "probe";
    let tolerance: f64 = get(9, "1e-14").parse().unwrap();
    let lattice = if case.starts_with("hex-") {
        Lattice2D::hexagonal(1.0)
    } else {
        Lattice2D::square(1.0)
    };
    let k = match case {
        "hex-k" => [2.0 / 3.0, 1.0 / 3.0],
        "hex-gamma" | "square-gamma" => [0.0; 2],
        "hex-generic" => [0.17, 0.11],
        "square-m" => [0.5, 0.5],
        _ => panic!("invalid case"),
    };
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: vec![BasisAtom {
            pos: [0.0; 2],
            radius: 0.2,
            eps_inside: 8.9,
        }],
    };
    let mut options = DielectricOptions::default();
    match material {
        "analytic" => {}
        "voronoi" => options.smoothing.method = blaze2d_core::dielectric::SmoothingMethod::Voronoi,
        "raw" => options.smoothing.mesh_size = 1,
        _ => panic!("invalid material sampling"),
    }
    let config = EigensolverConfig {
        n_bands: bands,
        max_iter: 1000,
        tol: tolerance,
        ..Default::default()
    };
    let start = Instant::now();
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &options);
    let mut op = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        dielectric,
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    );
    let mut preconditioner = op.build_homogeneous_preconditioner_adaptive();
    let (spectrum, report, identity, fields) = if method == "full" {
        let result = solve(
            &mut op,
            Some(&mut preconditioner),
            &SingleSolveJob::new(bands)
                .with_tolerance(tolerance)
                .with_max_iterations(1000),
        );
        let cert = certify_without_refinement(&mut op, &result.eigenvalues, &result.eigenvectors);
        let accepted = result.converged
            && result.eigenvalues.len() == bands
            && result.eigenvectors.len() == bands
            && cert.b_orthogonality_defect.is_finite()
            && cert.b_orthogonality_defect <= 1e-8
            && cert
                .residuals
                .iter()
                .zip(&cert.absolute_residuals)
                .enumerate()
                .all(|(j, (&r, &a))| {
                    r.is_finite()
                        && a.is_finite()
                        && if k == [0.0; 2] && j == 0 {
                            a <= 1e-7
                        } else {
                            r <= 1e-5
                        }
                });
        let report = json!({"accepted":accepted,"solver_converged":result.converged,"iterations":result.iterations,
            "projected":null,"full_rectangle":{"residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,"b_orthogonality_defect":cert.b_orthogonality_defect}});
        (result.eigenvalues, report, None, result.eigenvectors)
    } else {
        let space = FourierSupport::metric_ball(
            grid,
            lattice,
            k,
            std::f64::consts::TAU * cutoff,
            blaze2d_core::symmetry::support::SupportBudget { max_coefficients: 32768, ..Default::default() },
        )
        .unwrap();
        let result = match method {
            "dense" => solve_dense(
                &mut op,
                &space,
                &config,
                Application::DirectFourier,
                Default::default(),
            ),
            "iterative" | "reference" => solve_iterative(
                &mut op,
                &space,
                &config,
                None,
                Some(&mut preconditioner),
                if method == "reference" {
                    Application::SpatialReference
                } else {
                    Application::DirectFourier
                },
                Default::default(),
            ),
            _ => panic!("invalid solve method"),
        }
        .unwrap();
        (
            result.eigenvalues,
            serde_json::to_value(result.report).unwrap(),
            Some(result.support_identity),
            result.fields,
        )
    };
    // Physical fields remain available for diagnostics; operator workspaces
    // are released before the clock. Serialization and probes are separate.
    drop(preconditioner);
    drop(op);
    let seconds = start.elapsed().as_secs_f64();
    let field_payload = fields.len() * grid.len() * 16;
    let counters = blaze2d_core::profiler::get_counters();
    let timings =
        serde_json::from_str::<serde_json::Value>(&blaze2d_core::profiler::get_profile_json())
            .unwrap();
    let peak_rss_kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u64>().ok())
        });
    // Supplementary complete-matrix covariance probe, outside the solve clock
    // and work counts. It does not grant structural/material qualification.
    let covariance = if probe && method != "full" {
        let space = FourierSupport::metric_ball(
            grid,
            lattice,
            k,
            std::f64::consts::TAU * cutoff,
            blaze2d_core::symmetry::support::SupportBudget { max_coefficients: 32768, ..Default::default() },
        )
        .unwrap();
        let d = Dielectric2D::from_geometry(&geom, grid, &options);
        let mut op = ThetaOperator::new(
            CpuBackend::<f64>::new(),
            d,
            pol,
            lattice.reciprocal().fractional_to_cartesian(k),
        );
        let mut projected =
            GalerkinOperator::new(&mut op, &space, Application::DirectFourier, 512 << 20).unwrap();
        let (a, b) = dense_pencil(&mut projected, 512 << 20).unwrap();
        let group = if case.starts_with("hex-") {
            PlaneGroup::P6m
        } else {
            PlaneGroup::P4m
        };
        let little = group.group().little_group(k, 1e-12).unwrap();
        let m = space.len();
        let mut defects = [0.0f64; 2];
        for op in little.operations {
            let action = CoefficientAction::new(op, &space, &space, pol, 1e-12).unwrap();
            for (kind, matrix) in [&a, &b].iter().enumerate() {
                let mut error = 0.0;
                for j in 0..m {
                    for i in 0..m {
                        let changed = action.phases()[i].conj()
                            * matrix[action.permutation()[i] + m * action.permutation()[j]]
                            * action.phases()[j];
                        error += (changed - matrix[i + m * j]).norm_sqr();
                    }
                }
                defects[kind] = defects[kind].max(
                    error.sqrt()
                        / matrix
                            .iter()
                            .map(C::norm_sqr)
                            .sum::<f64>()
                            .sqrt()
                            .max(1e-300),
                );
            }
        }
        json!({"projected_A_relative_frobenius":defects[0],"projected_B_relative_frobenius":defects[1],"authorizes_symmetry":false})
    } else {
        serde_json::Value::Null
    };
    let physical_reference = if get(10, "no-reference") == "full-reference" && method != "full" {
        let d = Dielectric2D::from_geometry(&geom, grid, &options);
        let mut full = ThetaOperator::new(
            CpuBackend::<f64>::new(),
            d,
            pol,
            lattice.reciprocal().fractional_to_cartesian(k),
        );
        let mut p = full.build_homogeneous_preconditioner_adaptive();
        let reference_bands = bands.saturating_add(8).min(grid.len());
        let reference = solve(
            &mut full,
            Some(&mut p),
            &SingleSolveJob::new(reference_bands)
                .with_tolerance(tolerance)
                .with_max_iterations(1000),
        );
        let cert =
            certify_without_refinement(&mut full, &reference.eigenvalues, &reference.eigenvectors);
        let accepted = reference.converged
            && reference.eigenvalues.len() == reference_bands
            && reference.eigenvectors.len() == reference_bands
            && cert.b_orthogonality_defect.is_finite()
            && cert.b_orthogonality_defect <= 1e-8
            && cert
                .residuals
                .iter()
                .zip(&cert.absolute_residuals)
                .enumerate()
                .all(|(j, (&r, &a))| {
                    r.is_finite()
                        && a.is_finite()
                        && if k == [0.0; 2] && j == 0 {
                            a <= 1e-7
                        } else {
                            r <= 1e-5
                        }
                });
        let reference_spectrum = &reference.eigenvalues;
        let frequency_errors: Vec<_> = spectrum
            .iter()
            .zip(reference_spectrum)
            .map(|(&a, &b)| {
                let x = a.max(0.0).sqrt();
                let y = b.max(0.0).sqrt();
                (x - y).abs() / y.max(1.0)
            })
            .collect();
        let gap_errors: Vec<_> = spectrum
            .windows(2)
            .zip(reference_spectrum.windows(2))
            .map(|(a, b)| {
                let x = a[1].max(0.0).sqrt() - a[0].max(0.0).sqrt();
                let y = b[1].max(0.0).sqrt() - b[0].max(0.0).sqrt();
                (x - y).abs()
            })
            .collect();
        // Reserve the last mode as a guard. Move the prefix edge below a
        // cluster in EITHER pencil rather than comparing arbitrary slices of a
        // degenerate multiplet. All requested frequencies are still compared.
        let limit = bands
            .saturating_sub(1)
            .min(fields.len())
            .min(reference.eigenvectors.len());
        let prefix = common_guarded_prefix(reference_spectrum, &spectrum, limit);
        let subspace = if accepted && report["accepted"] == true && prefix > 0 {
            let mut overlap = vec![C::ZERO; prefix * prefix];
            for j in 0..prefix {
                let mut bv = full.alloc_field();
                full.apply_mass(&fields[j], &mut bv);
                for i in 0..prefix {
                    overlap[i + prefix * j] = reference.eigenvectors[i]
                        .as_slice()
                        .iter()
                        .zip(bv.as_slice())
                        .map(|(x, y)| x.conj() * y)
                        .sum();
                }
            }
            let mut gram = vec![C::ZERO; prefix * prefix];
            for j in 0..prefix {
                for i in 0..prefix {
                    gram[i + prefix * j] = (0..prefix)
                        .map(|r| overlap[r + prefix * i].conj() * overlap[r + prefix * j])
                        .sum();
                }
            }
            let squared =
                blaze2d_core::eigensolver::solve_hermitian_eigen(&gram, prefix).eigenvalues;
            let minimum = squared.iter().copied().fold(f64::INFINITY, f64::min);
            json!({"prefix":prefix,"metric":"unchanged full sampled B; principal angles of the guarded prefix",
                "max_sin_angle":(1.0-minimum.clamp(0.0,1.0)).sqrt(),"squared_singular_values":squared,
                "reference_boundary_gap_lambda":reference_spectrum[prefix]-reference_spectrum[prefix-1],
                "candidate_boundary_gap_lambda":spectrum[prefix]-spectrum[prefix-1]})
        } else {
            serde_json::Value::Null
        };
        json!({"accepted":accepted,"bands":reference_bands,"iterations":reference.iterations,"spectrum":reference_spectrum,
            "residuals":cert.residuals,"absolute_residuals":cert.absolute_residuals,"b_orthogonality_defect":cert.b_orthogonality_defect,
            "scaled_frequency_errors":frequency_errors,"angular_gap_absolute_errors":gap_errors,"field_subspace":subspace,
            "scope":"supplementary same-grid/material full-rectangle solve, outside candidate clock and counters"})
    } else {
        serde_json::Value::Null
    };
    println!(
        "{}",
        json!({"case":case,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),"method":method,
        "material":options,"material_label":material,"lattice":lattice,"k":k,"cutoff_in_2pi_units":cutoff,
        "eigensolver":config,"precision":"f64","discretization":if method=="full" {"full_fft_rectangle"} else {"metric_galerkin"},
        "support_identity":identity,"spectrum":spectrum,"report":report,"field_payload_bytes":field_payload,
            "seconds":seconds,"peak_rss_kib":peak_rss_kib,"counters":counters,"profile":timings,"covariance":covariance,"physical_reference":physical_reference,
            "scope":"new-discretization diagnostic; complete solve plus physical output production/retention and operator teardown; same declared material within a cutoff study; no symmetry speedup claim"})
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_candidate_doublet_is_not_compared_as_one_arbitrary_vector() {
        // The finite rectangle splits a doublet that is exact in the closed
        // support. Rotating that candidate doublet is physically immaterial.
        let reference = [0.0, 1.0, 2.0, 2.1];
        let candidate = [0.0, 1.0, 2.0, 2.0];
        let prefix = common_guarded_prefix(&reference, &candidate, 3);
        assert_eq!(prefix, 2);
        // A single vector of the doublet would spuriously show a 45° angle.
        let rotated_overlap_squared = 0.5;
        assert!((1.0f64 - rotated_overlap_squared).sqrt() > 0.7);
        // The common complete prefix is identical for every such rotation.
        let gram = [C::ONE, C::ZERO, C::ZERO, C::ONE];
        let eigen = blaze2d_core::eigensolver::solve_hermitian_eigen(&gram, prefix);
        assert!(eigen.eigenvalues.iter().all(|x| (x - 1.0).abs() < 1e-14));
        assert_eq!(common_guarded_prefix(&candidate, &reference, 3), 2);
    }
}
