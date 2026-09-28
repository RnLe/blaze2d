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
    pub cold_root: bool,
    pub solved_guards: usize,
    pub certified_guards: usize,
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
    let donor = (options.method != Method::Cold)
        .then(|| context.bank.donors(&key, 1).pop())
        .flatten();
    let mut report = PointReport {
        method: options.method,
        source: donor.as_ref().map(|s| s.basis_generation()),
        cold_root: donor.is_none(),
        solved_guards: options.guards,
        certified_guards: 0,
        iterations: 0,
        discarded_iterations: 0,
        initial_acceptance: false,
        tm_actions_reused: 0,
        fallback: None,
        insertion: BankReport::default(),
        leased_payload_bytes: donor.as_ref().map_or(0, |s| s.payload_bytes()),
        certificate: None,
    };
    let mut cfg = config.clone();
    cfg.n_bands += options.guards;
    cfg.k_index = Some(k_index);
    if cfg.block_size != 0 {
        cfg.block_size = cfg.block_size.max(cfg.n_bands);
    }
    let mut seeds = donor.as_ref().map(|s| {
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
    let mut actions = donor.as_ref().filter(|_|options.retain_tm_actions)
        .and_then(|s|s.applied_for(&key)).map(|fields|fields.iter().map(|f|{
            let mut out=theta.alloc_field();out.as_mut_slice().copy_from_slice(f.as_slice());out
        }).collect::<Vec<_>>());
    let has_actions=actions.is_some();
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
            solver = if let Some(actions)=actions.take() {solver.with_native_seed_actions(seeds,actions)}
                else {solver.with_native_warm_start(seeds)};
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
        if has_actions {report.tm_actions_reused=solver.initialization_report().map_or(0,|r|r.warm_start_hits);}
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
        report.source.into_iter().collect(),
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
