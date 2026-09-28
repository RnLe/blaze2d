//! Bounded, worker-owned geometry continuation at one physical Bloch fiber.
use super::{
    cache::{BankReport, NativeBank},
    snapshot::{BasisGeneration, FiberKey, NativeCertificate, NativeSnapshot, Quality},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{Eigensolver, EigensolverConfig, InitialAcceptance},
    field::{Field2D, Real},
    operators::ThetaOperator,
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
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Gamma,
    Path,
}
#[derive(Debug, Clone)]
pub struct Options {
    pub method: Method,
    pub scope: Scope,
    pub guards: usize,
    pub history: usize,
    pub max_predictor_columns: usize,
    pub max_preparation_bytes: usize,
    pub max_cache_bytes: usize,
    pub retain_tm_actions: bool,
    pub quality: Quality,
}
pub struct Context<R: Real> {
    pub bank: NativeBank<R>,
    pub job_index: usize,
}
impl<R: Real> Context<R> {
    pub fn new(limit: usize) -> Self {
        Self {
            bank: NativeBank::new(limit, 2),
            job_index: 0,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct PointReport {
    pub method: Method,
    pub source: Option<BasisGeneration>,
    pub parents: Vec<BasisGeneration>,
    pub prediction: Option<super::predict::PredictionReport>,
    pub preparation_actions_reused: usize,
    pub cold_root: bool,
    pub solved_guards: usize,
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
pub struct PointSolution {
    pub values: Vec<f64>,
    pub fields: Vec<Field2D>,
    pub converged: bool,
    pub report: PointReport,
}

#[allow(clippy::too_many_arguments)]
pub fn solve<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    config: &EigensolverConfig,
    mut preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    path_seed: Option<&[Field2D]>,
    context: &mut Context<B::Real>,
    options: &Options,
    parameters: Vec<f64>,
    k_index: usize,
) -> PointSolution {
    crate::profile_scope!("geometry.point");
    let key = FiberKey::from_operator(theta);
    let donors = if options.method == Method::Cold {
        Vec::new()
    } else {
        context.bank.donors(&key, match options.method { Method::History=>options.history,Method::Secant=>2,_=>1 })
    };
    let donor = donors.first().cloned();
    let mut report = PointReport {
        method: options.method,
        source: donor.as_ref().map(|s| s.basis_generation()),
        parents: donor.iter().map(|s| s.basis_generation()).collect(),
        prediction: None,
        preparation_actions_reused: 0,
        cold_root: donor.is_none(),
        solved_guards: options.guards,
        certified_guards: 0,
        initial_warm_columns:0,initial_random_columns:0,initial_rejected_columns:0,
        iterations: 0,
        discarded_iterations: 0,
        initial_acceptance: false,
        tm_actions_reused: 0,
        fallback: None,
        insertion: BankReport::default(),
        leased_payload_bytes: donors.iter().map(|s| s.payload_bytes()).sum(),
        certificate: None,
    };
    let mut cfg = config.clone();
    cfg.n_bands += options.guards;
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
            Method::Metric | Method::Secant | Method::History
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
        ) {
            Ok(prepared) => {
                preparation_actions = prepared.actions.is_some();
                report.parents = donors
                    .iter()
                    .take(prepared.report.parent_count)
                    .map(|s| s.basis_generation())
                    .collect();
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
    // An identical discrete pencil retains the source's lowest-window coverage.
    // A changed material cannot obtain coverage merely by passing a residual test.
    let identical = donor.as_ref().is_some_and(|s| {
        s.material.eps() == theta.dielectric().eps()
            && s.material.inv_eps() == theta.dielectric().inv_eps()
            && s.material.inv_eps_tensors() == theta.dielectric().inv_eps_tensors()
            && s.requested_bands + s.guard_count >= cfg.n_bands
    });
    let (result, pairs) = {
        let p = preconditioner
            .as_mut()
            .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>);
        let mut solver = Eigensolver::new(theta, cfg, p, path_seed);
        if let Some(seeds) = seeds.take() {
            solver = if let Some(actions) = actions.take() {
                solver.with_native_seed_actions(seeds, actions)
            } else {
                solver.with_native_warm_start(seeds)
            };
        }
        if identical {
            solver.set_initial_acceptance(InitialAcceptance {
                certified_prefix: config.n_bands + options.guards,
                relative_residual: options.quality.relative_residual,
                absolute_residual: options.quality.absolute_residual,
                b_orthogonality: options.quality.orthogonality,
            });
        }
        let result = solver.solve();
        report.initial_acceptance = result.iterations == 0;
        let hits = solver.initialization_report().map_or(0, |r| r.warm_start_hits);
        if let Some(init)=solver.initialization_report() {
            report.initial_warm_columns=init.warm_start_hits;report.initial_random_columns=init.random_vectors;
            report.initial_rejected_columns=init.rejected_warm_vectors;
        }
        if source_actions {
            report.tm_actions_reused = hits;
        }
        if preparation_actions {
            report.preparation_actions_reused = hits;
        }
        (result, solver.into_native_eigenpairs())
    };
    report.iterations = result.iterations;
    let mut values: Vec<_> = pairs.iter().take(config.n_bands).map(|(l, _)| *l).collect();
    let mut fields: Vec<_> = pairs
        .iter()
        .take(config.n_bands)
        .map(|(_, x)| {
            crate::field::record_public_export::<B::Real>(x.len());
            Field2D::from_vec(
                x.grid(),
                x.as_slice()
                    .iter()
                    .map(|z| num_complex::Complex64::new(z.re.to_accum(), z.im.to_accum()))
                    .collect(),
            )
        })
        .collect();
    let generation = context.bank.next_generation(context.job_index, k_index);
    let captured = NativeSnapshot::capture(
        theta,
        pairs,
        config.n_bands,
        generation,
        report.parents.clone(),
        parameters,
        result.converged,
        options.retain_tm_actions,
    );
    let accepted = captured.as_ref().is_ok_and(|s| s.admitted(options.quality));
    let mut converged = result.converged;
    if !accepted && (donor.is_some() || options.guards > 0) {
        report.fallback = Some(
            "guarded block failed convergence or direct certification; ordinary path retried"
                .into(),
        );
        report.discarded_iterations = result.iterations;
        let mut cfg = config.clone();
        cfg.k_index = Some(k_index);
        let p = preconditioner
            .as_mut()
            .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>);
        let mut solver = Eigensolver::new(theta, cfg, p, path_seed);
        let retry = solver.solve();
        fields = solver.all_eigenvectors();
        values = retry.eigenvalues;
        converged = retry.converged;
        report.iterations += retry.iterations;
        report.insertion.reason =
            Some("last accepted donor retained after failed candidate".into());
    } else {
        match captured {
            Ok(snapshot) => {
                report.certified_guards = if accepted { snapshot.guard_count() } else { 0 };
                let mut certificate = snapshot.certificate().clone();
                certificate.relative.truncate(config.n_bands);
                certificate.absolute.truncate(config.n_bands);
                certificate.dual.truncate(config.n_bands);
                report.certificate = Some(certificate);
                report.insertion = context.bank.insert(snapshot, options.quality);
            }
            Err(reason) => report.insertion.reason = Some(reason),
        }
    }
    report.insertion.entries = context.bank.len();
    report.insertion.payload_bytes = context.bank.payload_bytes();
    PointSolution {
        values,
        fields,
        converged,
        report,
    }
}
