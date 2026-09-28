//! Native trial spaces. Predictions are initializers, never coverage proofs.
use super::{
    point::Method,
    snapshot::{FiberKey, NativeSnapshot},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{
        deflation::DeflationSubspace,
        initialization::{self, BlockEntry, InitializationConfig},
        subspace_prediction::{OverlapMatrix, polar_decomposition},
    },
    field::Real,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    preconditioners::OperatorPreconditioner,
};
use num_complex::Complex64 as C;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone, Default, Serialize)]
pub struct PredictionReport {
    pub input_columns: usize,
    pub independent_columns: Option<usize>,
    pub retained_columns: usize,
    pub secant_ratio: Option<f64>,
    pub minimum_overlap: Option<f64>,
    pub estimated_peak_payload_bytes: usize,
    pub parent_count: usize,
    pub fallback: Option<String>,
}
pub struct Prediction<B: SpectralBackend> {
    pub vectors: Vec<B::Buffer>,
    pub actions: Option<Vec<B::Buffer>>,
    pub report: PredictionReport,
}

/// A positive scalar multiple of the *same signed parameter direction* is
/// required. A traversal corner or reversal has no first-order secant model.
pub fn secant_ratio(older: &[f64], current: &[f64], target: &[f64]) -> Option<f64> {
    if older.len() != current.len() || current.len() != target.len() || current.is_empty() {
        return None;
    }
    let steps: Vec<_> = older
        .iter()
        .zip(current)
        .zip(target)
        .map(|((&a, &b), &c)| {
            let scale = a.abs().max(b.abs()).max(c.abs()).max(1.);
            ((b - a) / scale, (c - b) / scale)
        })
        .collect();
    let norm = steps.iter().map(|(a, _)| a * a).sum::<f64>();
    let next = steps.iter().map(|(_, b)| b * b).sum::<f64>();
    if !norm.is_finite() || !next.is_finite() || norm == 0. || next == 0. {
        return None;
    }
    let alpha = steps.iter().map(|(a, b)| a * b).sum::<f64>() / norm;
    let error = steps
        .iter()
        .map(|(a, b)| (b - alpha * a).powi(2))
        .sum::<f64>();
    (alpha > 0. && alpha <= 2. && error <= 1e-16 * next).then_some(alpha)
}

fn imported<B: SpectralBackend>(
    theta: &ThetaOperator<B>,
    source: &NativeSnapshot<B::Real>,
    isometry: bool,
) -> Vec<B::Buffer> {
    source
        .fields()
        .iter()
        .map(|f| {
            let mut out = theta.alloc_field();
            for (i, (dst, src)) in out.as_mut_slice().iter_mut().zip(f.as_slice()).enumerate() {
                let factor = if isometry && theta.polarization() == Polarization::TM {
                    (source.material.eps()[i] / theta.dielectric().eps()[i]).sqrt()
                } else {
                    1.
                };
                *dst =
                    crate::field::cscalar(src.re.to_accum() * factor, src.im.to_accum() * factor);
            }
            crate::profiler::count(
                "geometry.predictor_import_bytes",
                f.len() * std::mem::size_of::<num_complex::Complex<B::Real>>(),
            );
            out
        })
        .collect()
}

struct MetricVector<B: SpectralBackend> {
    vector: B::Buffer,
    mass: B::Buffer,
}
fn metric_frame<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    fields: Vec<B::Buffer>,
    count: usize,
) -> Vec<MetricVector<B>> {
    use crate::eigensolver::normalization::{normalize_to_unit_b_norm_with_tol, project_out};
    let kernel = theta
        .analytic_kernel_available()
        .then(|| initialization::create_gamma_mode(theta));
    let tolerance = 1e-10_f64.max(32. * <B::Real as num_traits::Float>::epsilon().to_accum());
    let mut entries: Vec<MetricVector<B>> = Vec::new();
    for mut vector in fields {
        if entries.len() == count {
            break;
        }
        let mut mass = theta.alloc_field();
        theta.apply_mass(&vector, &mut mass);
        let incoming = theta.backend().dot(&vector, &mass).re.sqrt();
        if !incoming.is_finite() || incoming <= 0. {
            continue;
        }
        for _ in 0..2 {
            if let Some((q, bq, _)) = &kernel {
                project_out(theta.backend(), &mut vector, &mut mass, q, bq);
            }
            for e in &entries {
                project_out(theta.backend(), &mut vector, &mut mass, &e.vector, &e.mass);
            }
            theta.apply_mass(&vector, &mut mass);
        }
        if normalize_to_unit_b_norm_with_tol(
            theta.backend(),
            &mut vector,
            &mut mass,
            tolerance * incoming,
        ) == 0.
        {
            continue;
        }
        theta.apply_mass(&vector, &mut mass);
        entries.push(MetricVector { vector, mass });
    }
    entries
}

/// No random completion: numerical rank loss remains visible to selection.
pub(crate) fn repair<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    fields: Vec<B::Buffer>,
    limit: usize,
) -> Vec<BlockEntry<B>> {
    repair_with_actions(theta, fields, None, limit)
}
fn repair_with_actions<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    fields: Vec<B::Buffer>,
    actions: Option<Vec<B::Buffer>>,
    limit: usize,
) -> Vec<BlockEntry<B>> {
    let mut locked = DeflationSubspace::new();
    if theta.analytic_kernel_available() {
        let (q, bq, l) = initialization::create_gamma_mode(theta);
        locked.add_vector(theta.backend(), &q, &bq, l, 0);
    }
    let config = InitializationConfig {
        block_size: fields
            .len()
            .min(limit)
            .min(theta.grid().len() - locked.len()),
        max_random_attempts: 0,
        zero_tolerance: 1e-10,
    };
    initialization::initialize_native_block(theta, &config, fields, actions, Some(&locked)).0
}

#[allow(clippy::too_many_arguments)]
pub fn prepare<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    donors: &[Arc<NativeSnapshot<B::Real>>],
    parameters: &[f64],
    method: Method,
    retained: usize,
    max_columns: usize,
    max_bytes: usize,
    reuse_actions: bool,
    mut preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
) -> Result<Prediction<B>, String> {
    crate::profile_scope!("geometry.predictor");
    let latest = donors.first().ok_or("no certified donor for prediction")?;
    let key = FiberKey::from_operator(theta);
    if donors.iter().any(|d| d.key() != &key) {
        return Err("prediction donors do not share the target physical fiber".into());
    }
    let n = theta.grid().len();
    let limit = max_columns.min(n - usize::from(theta.analytic_kernel_available()));
    if limit < retained {
        return Err("predictor column budget cannot hold the retained block".into());
    }
    let payload = (n as u128)
        * (limit as u128 + 1)
        * std::mem::size_of::<num_complex::Complex<B::Real>>() as u128
        * 8
        + (limit as u128).pow(2) * 16 * 8;
    if payload > max_bytes as u128 {
        return Err("predictor preparation exceeds its workspace payload budget".into());
    }
    let mut report = PredictionReport {
        estimated_peak_payload_bytes: payload as usize,
        parent_count: 1,
        ..Default::default()
    };
    let mut entries = match method {
        Method::Metric => {
            let mut vectors = imported(theta, latest, true);
            vectors.truncate(retained);
            report.input_columns = vectors.len();
            report.retained_columns = vectors.len();
            return Ok(Prediction {
                vectors,
                actions: None,
                report,
            });
        }
        Method::Secant => {
            let old = donors
                .get(1)
                .ok_or("secant requires two distinct accepted geometries")?;
            let alpha = secant_ratio(&old.parameters, &latest.parameters, parameters).ok_or(
                "parameter direction changed, reversed or exceeded the bounded secant step",
            )?;
            let current_fields = imported(theta, latest, true);
            let current = metric_frame(theta, current_fields, retained);
            let older_fields = imported(theta, old, true);
            let older = metric_frame(theta, older_fields, retained);
            if current.len() != retained || older.len() != retained {
                return Err("aligned secant lost retained target-metric rank".into());
            }
            let mut overlap = OverlapMatrix::new(retained);
            for j in 0..retained {
                for i in 0..retained {
                    overlap.set(
                        i,
                        j,
                        theta.backend().dot(&current[i].vector, &older[j].mass),
                    );
                }
            }
            let (rotation, sigma) = polar_decomposition(&overlap);
            if !sigma.is_finite() || sigma < 0.5 {
                return Err(
                    "aligned secant overlap is too small for a shared retained subspace".into(),
                );
            }
            let coefficients: Vec<_> = (0..retained)
                .flat_map(|j| (0..retained).map(move |i| (i, j)))
                .map(|(i, j)| rotation.get(j, i).conj())
                .collect();
            let old_vectors: Vec<_> = older.into_iter().map(|e| e.vector).collect();
            let aligned =
                theta
                    .backend()
                    .linear_combinations(&old_vectors, &coefficients, retained);
            let fields: Vec<_> = current
                .into_iter()
                .zip(aligned)
                .map(|(entry, old)| {
                    let mut field = entry.vector;
                    theta.backend().scale(C::new(1. + alpha, 0.), &mut field);
                    theta.backend().axpy(C::new(-alpha, 0.), &old, &mut field);
                    field
                })
                .collect();
            report.parent_count = 2;
            report.secant_ratio = Some(alpha);
            report.minimum_overlap = Some(sigma);
            report.input_columns = fields.len();
            report.retained_columns = fields.len();
            return Ok(Prediction {
                vectors: fields,
                actions: None,
                report,
            });
        }
        Method::Residual => {
            if limit <= retained {
                return Err("residual enrichment needs room beyond the retained block".into());
            }
            let fields = imported(theta, latest, false);
            let actions = latest
                .applied_for(&key)
                .filter(|_| reuse_actions)
                .map(|fields| {
                    fields
                        .iter()
                        .map(|f| {
                            let mut a = theta.alloc_field();
                            a.as_mut_slice().copy_from_slice(f.as_slice());
                            a
                        })
                        .collect()
                });
            let entries = repair_with_actions(theta, fields, actions, retained);
            if entries.len() < retained {
                return Err("residual enrichment lost source rank".into());
            }
            let (entries, values) = initialization::ritz_block(theta.backend(), entries);
            let mut residuals = Vec::new();
            for (entry, &lambda) in entries.iter().zip(&values).take(limit - retained) {
                let mut residual = entry.applied.clone();
                theta
                    .backend()
                    .axpy(C::new(-lambda, 0.), &entry.mass, &mut residual);
                if let Some(p) = preconditioner.as_deref_mut() {
                    p.apply(theta.backend(), &mut residual);
                }
                residuals.push(residual);
            }
            let (mut vectors, actions): (Vec<_>, Vec<_>) =
                entries.into_iter().map(|e| (e.vector, e.applied)).unzip();
            vectors.extend(residuals);
            report.input_columns = vectors.len();
            // The action prefix belongs to the unchanged base columns. Newly
            // appended residual directions receive fresh target applications.
            repair_with_actions(theta, vectors, Some(actions), limit)
        }
        Method::History => {
            let mut fields = Vec::new();
            report.parent_count = 0;
            for source in donors {
                if fields.len() >= limit {
                    break;
                }
                let mut block = imported(theta, source, true);
                block.truncate(limit - fields.len());
                fields.extend(block);
                report.parent_count += 1;
            }
            report.input_columns = fields.len();
            repair(theta, fields, limit)
        }
        _ => return Err("method does not construct a geometry trial space".into()),
    };
    report.independent_columns = Some(entries.len());
    if entries.len() < retained {
        return Err("prediction lost the requested positive rank".into());
    }
    // Compression retains the lowest target Ritz block, not the oldest basis
    // directions or arbitrary column positions. All preparation work is charged.
    entries = initialization::ritz_block(theta.backend(), entries).0;
    entries.truncate(retained);
    report.retained_columns = entries.len();
    let (vectors, actions) = entries.into_iter().map(|e| (e.vector, e.applied)).unzip();
    Ok(Prediction {
        vectors,
        actions: Some(actions),
        report,
    })
}

/// Target-constrained Rayleigh–Ritz on a supplied native trial space. This does
/// not claim low-band coverage or eigensolver convergence.
pub fn rayleigh_ritz_trial<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    vectors: Vec<B::Buffer>,
    retained: usize,
    max_bytes: usize,
) -> Result<Prediction<B>, String> {
    let columns = vectors.len();
    let bytes = (columns as u128 + 1)
        * theta.grid().len() as u128
        * std::mem::size_of::<num_complex::Complex<B::Real>>() as u128
        * 8
        + (columns as u128).pow(2) * 16 * 8;
    if bytes > max_bytes as u128 {
        return Err("target trial exceeds its payload budget".into());
    }
    let entries = repair(theta, vectors, columns);
    let rank = entries.len();
    if rank < retained {
        return Err("target trial lost the retained rank".into());
    }
    let (mut entries, _) = initialization::ritz_block(theta.backend(), entries);
    entries.truncate(retained);
    let (vectors, actions) = entries.into_iter().map(|e| (e.vector, e.applied)).unzip();
    Ok(Prediction {
        vectors,
        actions: Some(actions),
        report: PredictionReport {
            input_columns: columns,
            independent_columns: Some(rank),
            retained_columns: retained,
            estimated_peak_payload_bytes: bytes as usize,
            ..Default::default()
        },
    })
}
