//! Bounded, worker-owned geometry continuation at one physical Bloch fiber.
use super::{
    cache::{BankReport, NativeBank},
    snapshot::{BasisGeneration, FiberKey, NativeCertificate, NativeSnapshot, Quality},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{Eigensolver, EigensolverConfig, InitialAcceptance},
    field::{Field2D, Real},
    operators::{LinearOperator, ThetaOperator},
    preconditioners::OperatorPreconditioner,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Cold,
    Copy,
    Metric,
    Secant,
    History,
    Residual,
    Response,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Gamma,
    Anchors,
    Path,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DonorSelection {
    SameK,
    PreviousK,
    Screened,
}
#[derive(Debug, Clone, Serialize)]
pub struct SelectionReport {
    pub policy: DonorSelection,
    pub geometry_available: bool,
    pub path_available: bool,
    pub geometry_distance: Option<f64>,
    pub path_distance: Option<f64>,
    pub selected: &'static str,
    pub reason: &'static str,
}
#[derive(Debug, Clone)]
pub struct Options {
    pub method: Method,
    pub scope: Scope,
    /// Additional exact reciprocal-fractional points; Gamma is always included.
    pub anchors: Vec<[f64; 2]>,
    pub donor_selection: DonorSelection,
    pub guards: usize,
    pub max_guards: Option<usize>,
    pub history: usize,
    pub innovation_columns: usize,
    pub innovation_relative_tolerance: f64,
    pub max_predictor_columns: usize,
    pub max_preparation_bytes: usize,
    pub max_cache_bytes: usize,
    pub retain_tm_actions: bool,
    pub quality: Quality,
    pub response_iterations: usize,
    pub response_tolerance: f64,
}
pub struct Context<R: Real> {
    pub bank: NativeBank<R>,
    pub job_index: usize,
    pub path: super::path::PathHistory<R>,
}
impl<R: Real> Context<R> {
    pub fn new(limit: usize) -> Self {
        Self {
            bank: NativeBank::new(limit, 2),
            job_index: 0,
            path: super::path::PathHistory::new(false, false),
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct PointReport {
    pub method: Method,
    pub geometry_anchor: bool,
    pub path: Option<super::path::PathReport>,
    pub path_payload_bytes: usize,
    pub selection: SelectionReport,
    pub source: Option<BasisGeneration>,
    pub parents: Vec<BasisGeneration>,
    pub prediction: Option<super::predict::PredictionReport>,
    pub response: Option<super::response::NativeResponseReport>,
    pub response_failure: Option<crate::symmetry::response::ResponseError>,
    pub compression: Option<super::innovation::CompressionReport>,
    pub compression_failure: Option<String>,
    pub preparation_actions_reused: usize,
    pub cold_root: bool,
    pub solved_guards: usize,
    pub guard_attempts: Vec<GuardAttempt>,
    pub certified_guards: usize,
    pub initial_warm_columns: usize,
    pub initial_random_columns: usize,
    pub initial_rejected_columns: usize,
    pub iterations: usize,
    pub discarded_iterations: usize,
    pub initial_acceptance: bool,
    pub tm_actions_reused: usize,
    pub fallback: Option<String>,
    pub insertion: BankReport,
    pub leased_payload_bytes: usize,
    pub certificate: Option<NativeCertificate>,
}
#[derive(Debug, Clone, Serialize)]
pub struct GuardAttempt {
    pub guards: usize,
    pub iterations: usize,
    pub lookahead: usize,
    pub boundary_gap: Option<f64>,
    pub boundary_uncertainty: Option<f64>,
    pub gap_threshold: f64,
    pub separated: bool,
}
pub struct PointSolution<R: Real> {
    pub values: Vec<f64>,
    pub fields: Vec<Field2D>,
    pub converged: bool,
    pub report: PointReport,
    pub native_frame: Option<std::sync::Arc<NativeSnapshot<R>>>,
}

#[allow(clippy::too_many_arguments)]
pub fn solve<B: SpectralBackend + Clone>(
    theta: &mut ThetaOperator<B>,
    config: &EigensolverConfig,
    mut preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    context: &mut Context<B::Real>,
    options: &Options,
    parameters: Vec<f64>,
    k_index: usize,
    path_distance: Option<f64>,
    k_fractional: [f64; 2],
    geometry_anchor: bool,
) -> PointSolution<B::Real> {
    crate::profile_scope!("geometry.point");
    let key = FiberKey::from_operator(theta);
    let mut donors = if options.method == Method::Cold || !geometry_anchor {
        Vec::new()
    } else {
        context.bank.donors(
            &key,
            match options.method {
                Method::History => options.history,
                Method::Secant => 2,
                _ => 1,
            },
        )
    };
    let geometry_distance = donors.first().and_then(|d| {
        (d.parameters.len() == parameters.len() && !parameters.is_empty()).then(|| {
            (d.parameters
                .iter()
                .zip(&parameters)
                .map(|(&a, &b)| ((b - a) / a.abs().max(b.abs()).max(1.)).powi(2))
                .sum::<f64>()
                / parameters.len() as f64)
                .sqrt()
        })
    });
    let path_available = context.path.available(theta);
    let use_path = path_available
        && (donors.is_empty()
            || options.donor_selection == DonorSelection::PreviousK
            || (options.donor_selection == DonorSelection::Screened
                && geometry_distance
                    .zip(path_distance)
                    .is_some_and(|(g, k)| g > k)));
    let selection = SelectionReport {
        policy: options.donor_selection,
        geometry_available: !donors.is_empty(),
        path_available,
        geometry_distance,
        path_distance,
        selected: if use_path {
            "previous_k"
        } else if donors.is_empty() {
            "cold"
        } else {
            "same_k_geometry"
        },
        reason: if donors.is_empty() {
            "no compatible geometry donor"
        } else if !path_available {
            "no previous-k candidate"
        } else if options.donor_selection == DonorSelection::Screened {
            "dimensionless parameter-step screening; no extra target actions"
        } else {
            "explicit donor policy"
        },
    };
    let compression_donor = donors
        .first()
        .filter(|_| options.innovation_columns > 0)
        .cloned();
    if use_path {
        donors.clear();
    }
    let donor = donors.first().cloned();
    let innovations = donor
        .as_ref()
        .filter(|_| options.innovation_columns > 0)
        .and_then(|s| context.bank.innovations_for(s));
    let mut report = PointReport {
        method: options.method,
        geometry_anchor,
        path: None,
        path_payload_bytes: context.path.payload_bytes(),
        selection,
        source: donor.as_ref().map(|s| s.basis_generation()),
        parents: donor.iter().map(|s| s.basis_generation()).collect(),
        prediction: None,
        response: None,
        response_failure: None,
        compression: None,
        compression_failure: None,
        preparation_actions_reused: 0,
        cold_root: donor.is_none() && !path_available,
        solved_guards: 0,
        guard_attempts: Vec::new(),
        certified_guards: 0,
        initial_warm_columns: 0,
        initial_random_columns: 0,
        initial_rejected_columns: 0,
        iterations: 0,
        discarded_iterations: 0,
        initial_acceptance: false,
        tm_actions_reused: 0,
        fallback: None,
        insertion: BankReport::default(),
        leased_payload_bytes: donors.iter().map(|s| s.payload_bytes()).sum::<usize>()
            + innovations.as_ref().map_or(0, |i| i.payload_bytes())
            + compression_donor
                .as_ref()
                .filter(|_| donors.is_empty())
                .map_or(0, |s| s.payload_bytes()),
        certificate: None,
    };
    let base_guards = if geometry_anchor { options.guards } else { 0 };
    let max_guards = if geometry_anchor { options
        .max_guards
        .unwrap_or(base_guards)
        .max(base_guards)
        .min((key.shape[0] * key.shape[1]).saturating_sub(config.n_bands)) } else { 0 };
    let adaptive = max_guards > base_guards;
    let mut guards = donor
        .as_ref()
        .filter(|_| adaptive)
        .map_or(base_guards, |d| {
            d.guard_count().max(base_guards).min(max_guards)
        });
    let mut cfg = config.clone();
    cfg.n_bands += guards;
    cfg.k_index = Some(k_index);
    if cfg.block_size != 0 {
        cfg.block_size = cfg.block_size.max(cfg.n_bands);
    }
    let mut seeds = None;
    let mut actions = None;
    let mut source_actions = false;
    let mut preparation_actions = false;
    if donor.is_some()
        && matches!(
            options.method,
            Method::Metric | Method::Secant | Method::History | Method::Residual
        )
    {
        let retained = cfg.n_bands - usize::from(key.is_gamma());
        let columns = if options.max_predictor_columns == 0 {
            2 * retained
        } else {
            options.max_predictor_columns
        };
        match super::predict::prepare(
            theta,
            &donors,
            &parameters,
            options.method,
            retained,
            columns,
            options.max_preparation_bytes,
            options.retain_tm_actions,
            preconditioner
                .as_mut()
                .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>),
            innovations.as_deref(),
        ) {
            Ok(prepared) => {
                preparation_actions = prepared.actions.is_some();
                report.parents = donors
                    .iter()
                    .take(prepared.report.parent_count)
                    .map(|s| s.basis_generation())
                    .collect();
                for &generation in &prepared.report.innovation_parents {
                    if !report.parents.contains(&generation) {
                        report.parents.push(generation);
                    }
                }
                seeds = Some(prepared.vectors);
                actions = prepared.actions;
                report.prediction = Some(prepared.report);
            }
            Err(reason) => {
                report.prediction = Some(super::predict::PredictionReport {
                    fallback: Some(reason),
                    ..Default::default()
                })
            }
        }
    }
    if let Some(source) = donor
        .as_ref()
        .filter(|_| options.method == Method::Response)
    {
        let config = crate::symmetry::response::ResponseConfig {
            max_inner_iterations: options.response_iterations,
            inner_relative_tolerance: options.response_tolerance,
        };
        match super::response::predict(
            theta,
            source,
            config,
            options.quality,
            options.retain_tm_actions,
            options.max_preparation_bytes,
        ) {
            Ok(response) => {
                preparation_actions = response.actions.is_some();
                seeds = Some(response.vectors);
                actions = response.actions;
                report.response = Some(response.report);
            }
            Err(error) => report.response_failure = Some(error),
        }
    }
    if seeds.is_none() {
        seeds = donor.as_ref().map(|s| {
            s.fields()
                .iter()
                .map(|f| {
                    let mut out = theta.alloc_field();
                    out.as_mut_slice().copy_from_slice(f.as_slice());
                    crate::profiler::count(
                        "geometry.native_seed_copy_bytes",
                        f.len() * std::mem::size_of::<num_complex::Complex<B::Real>>(),
                    );
                    out
                })
                .collect::<Vec<_>>()
        });
        actions = donor
            .as_ref()
            .filter(|_| options.retain_tm_actions)
            .and_then(|s| s.applied_for(&key))
            .map(|fields| {
                fields
                    .iter()
                    .map(|f| {
                        let mut out = theta.alloc_field();
                        out.as_mut_slice().copy_from_slice(f.as_slice());
                        out
                    })
                    .collect::<Vec<_>>()
            });
        source_actions = actions.is_some();
    }
    if seeds.is_none() && use_path {
        if let Some((fields, path)) = context.path.predict(theta, k_fractional) {
            report.parents = path.parents.clone();
            report.path = Some(path);
            seeds = Some(fields);
        }
    }
    // An identical discrete pencil retains the source's lowest-window coverage.
    // A changed material cannot obtain coverage merely by passing a residual test.
    let identical = !adaptive
        && donor.as_ref().is_some_and(|s| {
            s.material.eps() == theta.dielectric().eps()
                && s.material.inv_eps() == theta.dielectric().inv_eps()
                && s.material.inv_eps_tensors() == theta.dielectric().inv_eps_tensors()
                && s.requested_bands + s.guard_count >= cfg.n_bands
        });
    let (result, pairs, separated) = loop {
        let lookahead =
            usize::from(adaptive && config.n_bands + guards < key.shape[0] * key.shape[1]);
        cfg.n_bands = config.n_bands + guards + lookahead;
        if cfg.block_size != 0 {
            cfg.block_size = cfg.block_size.max(cfg.n_bands);
        }
        let p = preconditioner
            .as_mut()
            .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>);
        let mut solver = Eigensolver::new(theta, cfg.clone(), p, None);
        if let Some(seeds) = seeds.take() {
            solver = if let Some(actions) = actions.take() {
                solver.with_native_seed_actions(seeds, actions)
            } else {
                solver.with_native_warm_start(seeds)
            };
        }
        if identical {
            solver.set_initial_acceptance(InitialAcceptance {
                certified_prefix: config.n_bands + guards,
                relative_residual: options.quality.relative_residual,
                absolute_residual: options.quality.absolute_residual,
                b_orthogonality: options.quality.orthogonality,
            });
        }
        let result = solver.solve();
        report.initial_acceptance = result.iterations == 0;
        let hits = solver
            .initialization_report()
            .map_or(0, |r| r.warm_start_hits);
        if let Some(init) = solver.initialization_report() {
            report.initial_warm_columns += init.warm_start_hits;
            report.initial_random_columns += init.random_vectors;
            report.initial_rejected_columns += init.rejected_warm_vectors;
        }
        if source_actions {
            report.tm_actions_reused += hits;
        }
        if preparation_actions {
            report.preparation_actions_reused += hits;
        }
        let mut pairs = solver.into_native_eigenpairs();
        let retained = config.n_bands + guards;
        let gap = if lookahead > 0 && pairs.len() > retained {
            Some(pairs[retained].0 - pairs[retained - 1].0)
        } else {
            None
        };
        let uncertainty = gap.map(|_| {
            pairs[retained - 1..=retained]
                .iter()
                .map(|(lambda, x)| boundary_residual(theta, *lambda, x))
                .sum::<f64>()
        });
        let scale = pairs
            .iter()
            .skip(retained.saturating_sub(1))
            .map(|(l, _)| l.abs())
            .fold(0., f64::max);
        let threshold =
            8. * (options.quality.absolute_residual + options.quality.relative_residual * scale);
        let separated = !adaptive
            || lookahead == 0
            || gap
                .zip(uncertainty)
                .is_some_and(|(g, u)| u.is_finite() && g > threshold + u);
        report.iterations += result.iterations;
        report.guard_attempts.push(GuardAttempt {
            guards,
            iterations: result.iterations,
            lookahead,
            boundary_gap: gap,
            boundary_uncertainty: uncertainty,
            gap_threshold: threshold,
            separated,
        });
        if adaptive && !separated && result.converged && guards < max_guards {
            report.discarded_iterations += result.iterations;
            guards = (guards + 2).min(max_guards);
            seeds = Some(pairs.into_iter().map(|(_, x)| x).collect());
            actions = None;
            source_actions = false;
            preparation_actions = false;
            continue;
        }
        report.solved_guards = guards;
        pairs.truncate(retained);
        break (result, pairs, separated);
    };
    let mut values: Vec<_> = pairs.iter().take(config.n_bands).map(|(l, _)| *l).collect();
    let mut fields = export_fields::<B>(&pairs, config.n_bands);
    let generation = context.bank.next_generation(context.job_index, k_index);
    let captured = NativeSnapshot::capture(
        theta,
        pairs,
        config.n_bands,
        generation,
        report.parents.clone(),
        parameters.clone(),
        result.converged,
        options.retain_tm_actions,
    );
    let accepted = captured.as_ref().is_ok_and(|s| s.admitted(options.quality));
    let mut converged = result.converged;
    let mut native_frame = None;
    if !accepted && (donor.is_some() || guards > 0 || adaptive || use_path) {
        report.fallback = Some(
            "guarded block failed convergence or direct certification; ordinary path retried"
                .into(),
        );
        report.discarded_iterations += result.iterations;
        let mut cfg = config.clone();
        cfg.k_index = Some(k_index);
        let p = preconditioner
            .as_mut()
            .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>);
        // Geometry failure falls back to the current path; a failed path seed
        // itself receives one cold retry. The previous accepted bank survives.
        let retry_seed = if donor.is_some() { context.path.predict(theta, k_fractional) } else { None };
        let mut solver = Eigensolver::new(theta, cfg, p, None);
        if let Some((seed, path)) = retry_seed {
            report.path = Some(path);
            solver = solver.with_native_warm_start(seed);
        }
        let retry = solver.solve();
        if let Some(init) = solver.initialization_report() {
            report.initial_warm_columns += init.warm_start_hits;
            report.initial_random_columns += init.random_vectors;
            report.initial_rejected_columns += init.rejected_warm_vectors;
        }
        let pairs = solver.into_native_eigenpairs();
        values = pairs.iter().map(|(value,_)| *value).collect();
        fields = export_fields::<B>(&pairs, config.n_bands);
        converged = retry.converged;
        report.iterations += retry.iterations;
        if let Ok(snapshot) = NativeSnapshot::capture(theta, pairs, config.n_bands,
            context.bank.next_generation(context.job_index, k_index), Vec::new(), parameters,
            retry.converged, false) {
            report.certificate = Some(snapshot.certificate().clone());
            if snapshot.admitted(options.quality) { native_frame = Some(std::sync::Arc::new(snapshot)); }
        }
        report.insertion.reason =
            Some("last accepted donor retained after failed candidate".into());
    } else {
        match captured {
            Ok(snapshot) => {
                let snapshot = std::sync::Arc::new(snapshot);
                if accepted { native_frame = Some(snapshot.clone()); }
                report.certified_guards = if accepted { snapshot.guard_count() } else { 0 };
                let mut certificate = snapshot.certificate().clone();
                certificate.relative.truncate(config.n_bands);
                certificate.absolute.truncate(config.n_bands);
                certificate.dual.truncate(config.n_bands);
                certificate.dual_relative.truncate(config.n_bands);
                report.certificate = Some(certificate);
                if !geometry_anchor {
                    report.insertion.reason = Some("point belongs only to native path history".into());
                } else if separated {
                    let identical_frame = compression_donor.as_ref().is_some_and(|old| {
                        old.parameters == snapshot.parameters
                            && old.material.eps() == snapshot.material.eps()
                            && old.material.inv_eps() == snapshot.material.inv_eps()
                            && old.material.inv_eps_tensors() == snapshot.material.inv_eps_tensors()
                            && old.fields().len() >= snapshot.fields().len()
                    });
                    if identical_frame {
                        report.insertion.reason =
                            Some("identical accepted frame and its innovations retained".into());
                    } else {
                        let compressed =
                            compression_donor
                                .as_ref()
                                .filter(|_| accepted)
                                .and_then(|previous| {
                                    match super::innovation::compress(
                                        theta,
                                        &snapshot,
                                        previous,
                                        options.innovation_columns,
                                        options.innovation_relative_tolerance,
                                        options.max_preparation_bytes,
                                    ) {
                                        Ok(innovation) => {
                                            report.compression = Some(innovation.report.clone());
                                            Some(innovation)
                                        }
                                        Err(reason) => {
                                            report.compression_failure = Some(reason);
                                            None
                                        }
                                    }
                                });
                        report.insertion = context.bank.insert_shared(
                            snapshot,
                            options.quality,
                            compressed,
                        );
                    }
                } else {
                    report.insertion.reason=Some("guard boundary remains unresolved at its expansion limit; last donor retained".into());
                }
            }
            Err(reason) => report.insertion.reason = Some(reason),
        }
    }
    report.insertion.entries = context.bank.len();
    report.insertion.payload_bytes = context.bank.payload_bytes();
    (
        report.insertion.innovation_columns,
        report.insertion.innovation_payload_bytes,
    ) = context.bank.innovation_payload();
    PointSolution {
        values,
        fields,
        converged,
        report,
        native_frame,
    }
}

fn export_fields<B: SpectralBackend>(pairs: &[(f64, B::Buffer)], count: usize) -> Vec<Field2D> {
    pairs.iter().take(count).map(|(_, x)| {
        crate::field::record_public_export::<B::Real>(x.len());
        Field2D::from_vec(x.grid(), x.as_slice().iter()
            .map(|z| num_complex::Complex64::new(z.re.to_accum(), z.im.to_accum())).collect())
    }).collect()
}

/// Direct source-mass-whitened residual bound in eigenvalue units. These two
/// probes qualify an estimated boundary gap, not a global eigenvalue count.
fn boundary_residual<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    lambda: f64,
    x: &B::Buffer,
) -> f64 {
    let mut ax = theta.alloc_field();
    let mut bx = theta.alloc_field();
    theta.apply(x, &mut ax);
    theta.apply_mass(x, &mut bx);
    let norm = theta.backend().dot(x, &bx).re;
    theta
        .backend()
        .axpy(num_complex::Complex64::new(-lambda, 0.), &bx, &mut ax);
    let tm = theta.polarization() == crate::polarization::Polarization::TM;
    let residual = ax
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, z)| {
            (z.re.to_accum().powi(2) + z.im.to_accum().powi(2))
                / if tm { theta.dielectric().eps()[i] } else { 1. }
        })
        .sum::<f64>();
    crate::profiler::count("geometry.guard_probe_columns", 1);
    if norm > 0. {
        (residual / norm).sqrt()
    } else {
        f64::INFINITY
    }
}
