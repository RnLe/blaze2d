//! Qualification tied to the actual Galerkin pencil, independently of the
//! original rectangular FFT support. No material averaging or repair occurs.
use std::collections::HashSet;

use crate::field::{Real, cscalar};
use num_complex::Complex;
use serde::Serialize;

use super::{
    action::sampled_node_permutation,
    affine::{AffineGroup, AffineOp},
    cache::SnapshotIdentity,
    galerkin::{GalerkinApplication, GalerkinOperator, validate_operator},
    representation::RepresentationPlan,
    support::{CoefficientAction, FourierSupport, SupportIdentity},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SupportQualificationOptions {
    pub tolerance: f64,
    pub max_group_order: usize,
    /// Numerical payload admission estimate, excluding the existing operator,
    /// backend FFT plans and allocator/library overhead. Not a hard RSS limit.
    pub max_estimated_payload_bytes: usize,
    pub complete_probe_dimension: usize,
    pub deterministic_probes: usize,
}
impl Default for SupportQualificationOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_group_order: 64,
            max_estimated_payload_bytes: 256 * 1024 * 1024,
            complete_probe_dimension: 64,
            deterministic_probes: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportActionReport {
    pub operation_index: usize,
    pub sample_permutation_exact: bool,
    pub support_closed: bool,
    pub metric_defect: f64,
    pub material_defect: f64,
    pub tensor_defect: f64,
    pub operator_defect: f64,
    pub mass_defect: f64,
    pub probe_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportQualificationReport {
    pub authorized: bool,
    pub tolerance: f64,
    pub estimated_payload_bytes: usize,
    pub maximum_multiplier_defect: f64,
    pub complete_column_probes: bool,
    pub actions: Vec<SupportActionReport>,
    pub rejection: Option<String>,
    pub evidence: &'static str,
}

/// Material and coefficient-space identities are both retained losslessly.
/// Each solver must call `validate_for` before using this authorization.
#[derive(Debug, Clone)]
pub struct QualifiedSupportGroup {
    support_identity: SupportIdentity,
    operator_identity: SnapshotIdentity,
    representation: RepresentationPlan,
    actions: Vec<CoefficientAction>,
    report: SupportQualificationReport,
}
impl QualifiedSupportGroup {
    pub fn report(&self) -> &SupportQualificationReport {
        &self.report
    }
    pub fn representation(&self) -> &RepresentationPlan {
        &self.representation
    }
    pub fn actions(&self) -> &[CoefficientAction] {
        &self.actions
    }
    pub fn support_identity(&self) -> &SupportIdentity {
        &self.support_identity
    }
    pub fn validate_for<B: SpectralBackend>(
        &self,
        full: &ThetaOperator<B>,
        support: &FourierSupport,
    ) -> Result<(), String> {
        validate_operator(full, support)?;
        if self.support_identity != support.identity()
            || self.operator_identity != SnapshotIdentity::from_operator(full)
        {
            return Err("qualified support/material/fiber identity changed".into());
        }
        Ok(())
    }

    pub fn qualify<B: SpectralBackend>(
        full: &mut ThetaOperator<B>,
        support: &FourierSupport,
        group: AffineGroup,
        options: SupportQualificationOptions,
    ) -> Result<Self, SupportQualificationReport> {
        crate::profile_scope!("symmetry.support.qualify");
        let mut report = SupportQualificationReport {
            authorized: false,
            tolerance: options.tolerance,
            estimated_payload_bytes: 0,
            maximum_multiplier_defect: 0.0,
            complete_column_probes: support.len() <= options.complete_probe_dimension,
            actions: Vec::new(),
            rejection: None,
            evidence: "exact coefficient and sample permutations, metric/material covariance, multiplier laws and supplementary projected A/B probes",
        };
        let attempt = (|| -> Result<_, String> {
            if !options.tolerance.is_finite()
                || options.tolerance <= 0.0
                || options.tolerance >= 1e-3
                || options.max_group_order == 0
                || options.max_group_order > 64
                || options.deterministic_probes == 0
                || options.deterministic_probes > 64
            {
                return Err("invalid support qualification options".into());
            }
            validate_group(&group, options.max_group_order)?;
            validate_operator(full, support)?;
            let n = support.grid().len() as u128;
            let m = support.len() as u128;
            let h = group.order() as u128;
            // Includes exact identities, coefficient actions and temporary
            // maps, sample indices, five probe fields and Galerkin buffers.
            // Small-group representation construction has a cubic allowance.
            let bytes = 256 * n + (24 * h + 160) * m + 64 * h * h * h + 256 * h * h;
            report.estimated_payload_bytes =
                usize::try_from(bytes).map_err(|_| "qualification payload overflow")?;
            if report.estimated_payload_bytes > options.max_estimated_payload_bytes {
                return Err("support qualification exceeds its payload allowance".into());
            }
            let mut actions = Vec::with_capacity(group.order());
            for (index, &operation) in group.operations.iter().enumerate() {
                let entry = SupportActionReport {
                    operation_index: index,
                    sample_permutation_exact: false,
                    support_closed: false,
                    metric_defect: operation.metric_defect(&support.lattice()),
                    material_defect: 0.0,
                    tensor_defect: 0.0,
                    operator_defect: 0.0,
                    mass_defect: 0.0,
                    probe_count: 0,
                };
                report.actions.push(entry);
                let entry = report.actions.last_mut().unwrap();
                if !entry.metric_defect.is_finite() || entry.metric_defect > options.tolerance {
                    return Err(format!("operation {index} breaks the Cartesian metric"));
                }
                let nodes = sampled_node_permutation(operation, support.grid())
                    .map_err(|e| format!("operation {index} sample nodes: {e}"))?;
                entry.sample_permutation_exact = true;
                let action = CoefficientAction::new(
                    operation,
                    support,
                    support,
                    full.polarization(),
                    options.tolerance.min(1e-12),
                )?;
                entry.support_closed = true;
                material_covariance(full, support, operation, &nodes, entry)?;
                if entry.material_defect > options.tolerance {
                    return Err(format!("operation {index} breaks sampled scalar material"));
                }
                if entry.tensor_defect > options.tolerance {
                    return Err(format!(
                        "operation {index} breaks sampled tensor covariance"
                    ));
                }
                actions.push(action);
            }
            let representation = RepresentationPlan::ordinary(group.clone(), support.k())
                .or_else(|_| RepresentationPlan::projective(group, support.k()))
                .map_err(|e| e.to_string())?;
            let h = actions.len();
            for g in 0..h {
                for v in 0..h {
                    let product = representation.group.product_index[g][v];
                    let omega = representation.multiplier[g][v];
                    for i in 0..support.len() {
                        let intermediate = actions[v].permutation()[i];
                        if actions[g].permutation()[intermediate]
                            != actions[product].permutation()[i]
                        {
                            return Err(
                                "coefficient permutations violate group multiplication".into()
                            );
                        }
                        let defect = (actions[g].phases()[intermediate] * actions[v].phases()[i]
                            - omega * actions[product].phases()[i])
                            .norm();
                        if !defect.is_finite() {
                            return Err("nonfinite coefficient multiplier law".into());
                        }
                        report.maximum_multiplier_defect =
                            report.maximum_multiplier_defect.max(defect);
                    }
                }
            }
            if report.maximum_multiplier_defect > options.tolerance {
                return Err("coefficient phases violate the little-group multiplier".into());
            }
            let mut op = GalerkinOperator::new(
                full,
                support,
                GalerkinApplication::DirectFourier,
                options.max_estimated_payload_bytes,
            )?;
            let mut x = op.alloc_field();
            let mut tx = op.alloc_field();
            let mut ax = op.alloc_field();
            let mut tax = op.alloc_field();
            let mut atx = op.alloc_field();
            let probes = if report.complete_column_probes {
                support.len()
            } else {
                options.deterministic_probes
            };
            for (action, entry) in actions.iter().zip(&mut report.actions) {
                for seed in 0..probes {
                    if report.complete_column_probes {
                        x.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
                        x.as_mut_slice()[seed] = cscalar::<B::Real>(1.0, 0.0);
                    } else {
                        for (i, z) in x.as_mut_slice().iter_mut().enumerate() {
                            let t = (i + 1 + seed * 19) as f64;
                            *z = cscalar::<B::Real>(
                                (t * 1.61803398875).sin(),
                                (t * 0.754877666).cos(),
                            );
                        }
                    }
                    action.apply(x.as_slice(), tx.as_mut_slice());
                    op.apply(&x, &mut ax);
                    action.apply(ax.as_slice(), tax.as_mut_slice());
                    op.apply(&tx, &mut atx);
                    entry.operator_defect = entry
                        .operator_defect
                        .max(relative_defect(tax.as_slice(), atx.as_slice()));
                    op.apply_mass(&x, &mut ax);
                    action.apply(ax.as_slice(), tax.as_mut_slice());
                    op.apply_mass(&tx, &mut atx);
                    entry.mass_defect = entry
                        .mass_defect
                        .max(relative_defect(tax.as_slice(), atx.as_slice()));
                    entry.probe_count += 1;
                    crate::profiler::count("symmetry.support.qualify.probes", 1);
                }
                if !entry.operator_defect.is_finite()
                    || entry.operator_defect > options.tolerance
                    || !entry.mass_defect.is_finite()
                    || entry.mass_defect > options.tolerance
                {
                    return Err(format!(
                        "operation {} fails projected A/B covariance",
                        entry.operation_index
                    ));
                }
            }
            Ok((representation, actions))
        })();
        match attempt {
            Ok((representation, actions)) => {
                report.authorized = true;
                Ok(Self {
                    support_identity: support.identity(),
                    operator_identity: SnapshotIdentity::from_operator(full),
                    representation,
                    actions,
                    report,
                })
            }
            Err(reason) => {
                report.rejection = Some(reason);
                Err(report)
            }
        }
    }
}

fn validate_group(group: &AffineGroup, limit: usize) -> Result<(), String> {
    let h = group.order();
    if h == 0
        || h > limit
        || group.operations[0] != AffineOp::IDENTITY
        || group.operations.iter().any(|g| g.conjugates())
        || group.operations.iter().collect::<HashSet<_>>().len() != h
        || group.product_index.len() != h
        || group.translation_carry.len() != h
        || group
            .product_index
            .iter()
            .any(|r| r.len() != h || r.iter().any(|&i| i >= h))
        || group.translation_carry.iter().any(|r| r.len() != h)
    {
        return Err("invalid or unsupported unitary support group".into());
    }
    for g in 0..h {
        for v in 0..h {
            let (product, carry) = group.operations[g]
                .compose(group.operations[v])
                .map_err(|e| e.to_string())?;
            if product != group.operations[group.product_index[g][v]]
                || carry != group.translation_carry[g][v]
            {
                return Err("affine multiplication table is inconsistent".into());
            }
        }
    }
    Ok(())
}

fn material_covariance<B: SpectralBackend>(
    full: &ThetaOperator<B>,
    support: &FourierSupport,
    operation: AffineOp,
    nodes: &[usize],
    report: &mut SupportActionReport,
) -> Result<(), String> {
    let lattice = support.lattice();
    let w = operation.matrix();
    let mut rotation = [[0.0; 2]; 2];
    for j in 0..2 {
        let mut unit = [0.0; 2];
        unit[j] = 1.0;
        let f = lattice.cartesian_to_fractional(unit);
        let cart = lattice.fractional_to_cartesian([
            w[0][0] as f64 * f[0] + w[0][1] as f64 * f[1],
            w[1][0] as f64 * f[0] + w[1][1] as f64 * f[1],
        ]);
        for i in 0..2 {
            rotation[i][j] = cart[i];
        }
    }
    let d = full.dielectric();
    let tensor = |j: usize| {
        d.inv_eps_tensors()
            .map_or([d.inv_eps()[j], 0.0, 0.0, d.inv_eps()[j]], |t| t[j])
    };
    for (j, &i) in nodes.iter().enumerate() {
        let (a, b) = (d.eps()[i], d.eps()[j]);
        report.material_defect = report
            .material_defect
            .max((a - b).abs() / a.abs().max(b.abs()));
        if full.polarization() == Polarization::TE {
            let (a, b) = (tensor(i), tensor(j));
            let scale = a
                .iter()
                .chain(&b)
                .map(|x| x.abs())
                .fold(f64::MIN_POSITIVE, f64::max);
            for row in 0..2 {
                for col in 0..2 {
                    let transformed: f64 = (0..2)
                        .flat_map(|u| {
                            (0..2).map(move |v| rotation[row][u] * a[2 * u + v] * rotation[col][v])
                        })
                        .sum();
                    let defect = (transformed - b[2 * row + col]).abs() / scale;
                    if !defect.is_finite() {
                        return Err("nonfinite transformed material tensor".into());
                    }
                    report.tensor_defect = report.tensor_defect.max(defect);
                }
            }
        }
    }
    Ok(())
}

fn relative_defect<R: Real>(a: &[Complex<R>], b: &[Complex<R>]) -> f64 {
    let mut error = 0.0;
    let mut scale = 0.0;
    for (x, y) in a.iter().zip(b) {
        error += (*x - *y).norm_sqr().to_accum();
        scale += x.norm_sqr().to_accum() + y.norm_sqr().to_accum();
    }
    if !error.is_finite() || !scale.is_finite() {
        return f64::INFINITY;
    }
    // A common analytic zero column gives exactly zero, not NaN.
    (2.0 * error / scale.max(f64::MIN_POSITIVE)).sqrt()
}

/// Qualify a star transport between two explicit supports of one sampled
/// material. Exact node/index bijections and material covariance establish the
/// candidate; independent A/B probes and a target residual check remain required.
pub(crate) fn qualify_transport<B: SpectralBackend + Clone>(
    target: &mut ThetaOperator<B>, source_support: &FourierSupport, target_support: &FourierSupport,
    operation: AffineOp, tolerance: f64, payload_limit: usize,
) -> Result<(CoefficientAction, SupportActionReport), String> {
    crate::profile_scope!("symmetry.support.transport_qualification");
    let estimated = 512_u128 * target_support.grid().len() as u128 + 256 * target_support.len() as u128;
    if estimated > payload_limit as u128 { return Err("support transport qualification exceeds its payload allowance".into()); }
    validate_operator(target, target_support)?;
    let mut report = SupportActionReport { operation_index: 0, sample_permutation_exact: false,
        support_closed: false, metric_defect: operation.metric_defect(&target_support.lattice()),
        material_defect: 0.0, tensor_defect: 0.0, operator_defect: 0.0, mass_defect: 0.0, probe_count: 0 };
    if !report.metric_defect.is_finite() || report.metric_defect > tolerance { return Err("star operation breaks the Cartesian metric".into()); }
    let nodes = sampled_node_permutation(operation, target_support.grid()).map_err(|e| e.to_string())?;
    report.sample_permutation_exact = true;
    let action = CoefficientAction::new(operation, source_support, target_support, target.polarization(), 8.0*f64::EPSILON)?;
    report.support_closed = true;
    material_covariance(target, target_support, operation, &nodes, &mut report)?;
    if report.material_defect > tolerance || report.tensor_defect > tolerance { return Err("star operation breaks sampled scalar/tensor covariance".into()); }
    let mut source = ThetaOperator::new(target.backend().clone(), target.dielectric().clone(), target.polarization(),
        source_support.lattice().reciprocal().fractional_to_cartesian(source_support.k()));
    let mut source = GalerkinOperator::new(&mut source, source_support, GalerkinApplication::DirectFourier, payload_limit)?;
    let mut target = GalerkinOperator::new(target, target_support, GalerkinApplication::DirectFourier, payload_limit)?;
    let mut x = source.alloc_field(); let mut ax = source.alloc_field();
    let mut tx = target.alloc_field(); let mut tax = target.alloc_field(); let mut atx = target.alloc_field();
    let complete = source_support.len() <= 64;
    for seed in 0..if complete { source_support.len() } else { 4 } {
        if complete { x.as_mut_slice().fill(cscalar::<B::Real>(0.0,0.0)); x.as_mut_slice()[seed] = cscalar::<B::Real>(1.0,0.0); }
        else { for (i,z) in x.as_mut_slice().iter_mut().enumerate() {
            let t = (i+1+seed*19) as f64; *z = cscalar::<B::Real>((t*1.61803398875).sin(),(t*0.754877666).cos());
        } }
        action.apply(x.as_slice(),tx.as_mut_slice());
        source.apply(&x,&mut ax); action.apply(ax.as_slice(),tax.as_mut_slice()); target.apply(&tx,&mut atx);
        report.operator_defect = report.operator_defect.max(relative_defect(tax.as_slice(),atx.as_slice()));
        source.apply_mass(&x,&mut ax); action.apply(ax.as_slice(),tax.as_mut_slice()); target.apply_mass(&tx,&mut atx);
        report.mass_defect = report.mass_defect.max(relative_defect(tax.as_slice(),atx.as_slice()));
        report.probe_count += 1;
        crate::profiler::count("symmetry.support.qualify.probes",1);
    }
    if report.operator_defect > tolerance || report.mass_defect > tolerance {
        return Err(format!("star operator/mass covariance failed: {:e}/{:e}",report.operator_defect,report.mass_defect));
    }
    Ok((action,report))
}
