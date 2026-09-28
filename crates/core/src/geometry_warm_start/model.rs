//! Reusable source-anchor tangents and complementary response pencils.
//! Models are bounded trial generators; target solves and certificates remain mandatory.
use super::{
    difference::DiscreteDifference,
    predict::{Prediction, rayleigh_ritz_trial},
    response::NativeResponseReport,
    snapshot::{BasisGeneration, FiberKey, NativeSnapshot, Quality},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::dense::solve_hermitian_eigen,
    field::{Field2D, Real},
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    preconditioners::OperatorPreconditioner,
    symmetry::response::{ResponseConfig, ResponseWork, multiply_real, project_out},
};
use num_complex::Complex64 as C;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Tangent,
    Reduced,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    Fourier,
    History,
    Residual,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Options {
    pub basis: Basis,
    pub columns: usize,
    pub training_iterations: usize,
    pub training_tolerance: f64,
    pub max_reuses: usize,
    pub max_step: f64,
    pub max_relative_residual: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            basis: Basis::Fourier,
            columns: 8,
            training_iterations: 128,
            training_tolerance: 1e-4,
            max_reuses: 4,
            max_step: 4.,
            max_relative_residual: 0.1,
        }
    }
}
impl Options {
    pub fn valid(&self) -> bool {
        (1..=64).contains(&self.columns)
            && (1..=1024).contains(&self.training_iterations)
            && self.training_tolerance.is_finite()
            && self.training_tolerance > 0.
            && self.training_tolerance < 1.
            && self.max_reuses <= 64
            && self.max_step.is_finite()
            && (1. ..=16.).contains(&self.max_step)
            && self.max_relative_residual.is_finite()
            && self.max_relative_residual > 0.
            && self.max_relative_residual < 1.
    }
}

enum Products<R: Real> {
    Tangent {
        increments: Vec<Field2D<R>>,
    },
    Reduced {
        basis: Vec<Field2D<R>>,
        applied: Vec<Field2D<R>>,
        stiffness: Vec<C>,
        mass: Vec<C>,
    },
}
pub struct ResponseModel<R: Real> {
    anchor: Arc<NativeSnapshot<R>>,
    training_parameters: Vec<f64>,
    options: Options,
    kind: Kind,
    gap: f64,
    products: Products<R>,
}
#[derive(Clone)]
pub struct Lease<R: Real> {
    pub model: Arc<ResponseModel<R>>,
    pub uses: usize,
}
impl<R: Real> ResponseModel<R> {
    pub fn key(&self) -> &FiberKey {
        self.anchor.key()
    }
    pub fn anchor(&self) -> BasisGeneration {
        self.anchor.basis_generation()
    }
    pub fn columns(&self) -> usize {
        match &self.products {
            Products::Tangent { increments } => increments.len(),
            Products::Reduced { basis, .. } => basis.len(),
        }
    }
    pub fn payload_bytes(&self) -> usize {
        let fields = match &self.products {
            Products::Tangent { increments } => increments.len(),
            Products::Reduced { basis, .. } => 2 * basis.len(),
        };
        let r = self.columns();
        // Charge the full leased anchor even when another bank entry shares it.
        self.anchor.payload_bytes()
            + std::mem::size_of::<Self>()
            + fields
                * (self.key().shape[0]
                    * self.key().shape[1]
                    * std::mem::size_of::<num_complex::Complex<R>>()
                    + std::mem::size_of::<Field2D<R>>())
            + self.training_parameters.capacity() * 8
            + 2 * r * r * 16
    }
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct Work {
    pub response: ResponseWork,
    pub basis_a_columns: usize,
    pub basis_b_columns: usize,
    pub target_preparation_columns: usize,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct Attempt {
    pub anchor: Option<BasisGeneration>,
    pub reused: bool,
    pub uses: usize,
    pub step_ratio: Option<f64>,
    pub basis_columns: usize,
    pub preferred_columns: usize,
    pub fourier_completion_columns: usize,
    pub source_guard_gap: Option<f64>,
    pub projected_relative_residuals: Vec<f64>,
    pub full_relative_residuals: Vec<f64>,
    pub target_max_relative_residual: Option<f64>,
    pub target_max_absolute_residual: Option<f64>,
    pub training: Option<NativeResponseReport>,
    pub work: Work,
    pub workspace_payload_bytes: usize,
    pub payload_bytes: usize,
    pub rejection: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub kind: Kind,
    pub attempts: Vec<Attempt>,
    pub accepted: bool,
}

fn import<B: SpectralBackend>(op: &ThetaOperator<B>, field: &Field2D<B::Real>) -> B::Buffer {
    let mut x = op.alloc_field();
    x.as_mut_slice().copy_from_slice(field.as_slice());
    x
}
fn source_frame<B: SpectralBackend>(
    op: &ThetaOperator<B>,
    source: &NativeSnapshot<B::Real>,
) -> (Vec<f64>, Vec<f64>, Vec<B::Buffer>) {
    let root: Vec<_> = (0..op.grid().len())
        .map(|i| {
            if op.polarization() == Polarization::TM {
                source.material.eps()[i].sqrt()
            } else {
                1.
            }
        })
        .collect();
    let inverse = root.iter().map(|v| v.recip()).collect();
    let mut excluded = Vec::new();
    for f in source.kernel().into_iter().chain(source.fields()) {
        let mut z = import(op, f);
        multiply_real::<B>(&mut z, &root);
        excluded.push(z);
    }
    (root, inverse, excluded)
}
fn hermitian(a: &mut [C], n: usize) {
    for j in 0..n {
        for i in 0..=j {
            let value = 0.5 * (a[i + j * n] + a[j + i * n].conj());
            a[i + j * n] = value;
            a[j + i * n] = value.conj();
        }
    }
}
fn displacement(anchor: &[f64], end: &[f64], target: &[f64], directional: bool) -> Option<f64> {
    if anchor.len() != end.len() || end.len() != target.len() || anchor.is_empty() {
        return None;
    }
    let mut aa = 0.;
    let mut bb = 0.;
    let mut ab = 0.;
    for ((&a, &e), &t) in anchor.iter().zip(end).zip(target) {
        let scale = a.abs().max(e.abs()).max(t.abs()).max(1.);
        let u = (e - a) / scale;
        let v = (t - a) / scale;
        aa += u * u;
        bb += v * v;
        ab += u * v;
    }
    if !aa.is_finite() || !bb.is_finite() || aa == 0. || bb == 0. {
        return None;
    }
    if directional {
        let alpha = ab / aa;
        let error = anchor
            .iter()
            .zip(end)
            .zip(target)
            .map(|((&a, &e), &t)| {
                let scale = a.abs().max(e.abs()).max(t.abs()).max(1.);
                ((t - a - alpha * (e - a)) / scale).powi(2)
            })
            .sum::<f64>();
        (alpha > 0. && error <= 1e-16 * bb).then_some(alpha)
    } else {
        Some((bb / aa).sqrt())
    }
}
fn gap<R: Real>(source: &NativeSnapshot<R>, quality: Quality) -> Result<f64, String> {
    let m = source.positive_requested();
    let p = source.fields().len();
    if m == 0 || p <= m {
        return Err(
            "reusable response needs a solved guard beyond the requested positive window".into(),
        );
    }
    let g = source.eigenvalues()[p - 1] - source.eigenvalues()[m - 1];
    let scale = source.eigenvalues()[p - 1].abs();
    if !g.is_finite() || g <= 8. * (quality.absolute_residual + quality.relative_residual * scale) {
        return Err("source guard window does not separate the response shifts".into());
    }
    Ok(g)
}
fn reuse_ratio<R: Real>(
    lease: &Lease<R>,
    latest: &NativeSnapshot<R>,
    parameters: &[f64],
    kind: Kind,
    options: Options,
    quality: Quality,
) -> Result<f64, String> {
    let model = &lease.model;
    if model.kind != kind
        || model.options != options
        || model.key() != latest.key()
        || model.anchor.requested_bands != latest.requested_bands
        || !model.anchor.admitted(quality)
    {
        return Err("response model source, options or physical fiber changed".into());
    }
    if lease.uses >= options.max_reuses {
        return Err("response model reached its reuse limit".into());
    }
    let ratio = displacement(
        &model.anchor.parameters,
        &model.training_parameters,
        parameters,
        kind == Kind::Tangent,
    )
    .ok_or("response model direction changed, reversed or vanished")?;
    if ratio > options.max_step {
        return Err("response model reached its displacement limit".into());
    }
    let m = model.anchor.positive_requested();
    if latest.eigenvalues().len() < m
        || gap(latest, quality).is_err()
        || model.anchor.eigenvalues()[..m]
            .iter()
            .zip(latest.eigenvalues())
            .any(|(a, b)| (a - b).abs() > 0.5 * model.gap)
    {
        return Err("response model source spectrum moved beyond its guarded trust region".into());
    }
    Ok(ratio)
}

/// Build a fixed source model. All products belong to the immutable anchor;
/// an unsuccessful build has no cache side effects.
#[allow(clippy::too_many_arguments)]
pub fn build<B: SpectralBackend + Clone>(
    target: &mut ThetaOperator<B>,
    anchor: Arc<NativeSnapshot<B::Real>>,
    previous: Option<&NativeSnapshot<B::Real>>,
    parameters: &[f64],
    kind: Kind,
    options: Options,
    quality: Quality,
    max_bytes: usize,
    report: &mut Attempt,
) -> Result<Arc<ResponseModel<B::Real>>, String> {
    crate::profile_scope!("geometry.model_build");
    if !options.valid()
        || !anchor.admitted(quality)
        || anchor.key() != &FiberKey::from_operator(target)
    {
        return Err("response model needs valid limits and an admitted same-fiber source".into());
    }
    if displacement(&anchor.parameters, parameters, parameters, true).is_none() {
        return Err("response model needs a nonzero finite parameter step".into());
    }
    let separation = gap(&anchor, quality)?;
    let n = target.grid().len();
    let p = anchor.fields().len();
    let r = options.columns;
    let bytes = (8 * p as u128 + 10 * r as u128 + 60)
        * n as u128
        * std::mem::size_of::<num_complex::Complex<B::Real>>() as u128
        + 16 * 8 * (p as u128 + r as u128).pow(2)
        + n as u128 * 8 * 16;
    if bytes > max_bytes as u128 {
        return Err("response model construction exceeds its preparation budget".into());
    }
    report.workspace_payload_bytes = bytes as usize;
    report.anchor = Some(anchor.basis_generation());
    report.source_guard_gap = Some(separation);
    let products = if kind == Kind::Tangent {
        let response = super::response::predict(
            target,
            &anchor,
            ResponseConfig {
                max_inner_iterations: options.training_iterations,
                inner_relative_tolerance: options.training_tolerance,
            },
            quality,
            false,
            max_bytes,
        )
        .map_err(|e| {
            report.work.response = e.work;
            e.reason
        })?;
        report.work.response = response.report.work.clone();
        let qualified = response
            .report
            .inner_columns
            .iter()
            .all(|c| c.inner_converged);
        report.full_relative_residuals = response
            .report
            .inner_columns
            .iter()
            .map(|c| c.inner_relative_residual)
            .collect();
        report.training = Some(response.report);
        if !qualified {
            return Err("tangent training did not meet its direct inner-residual contract".into());
        }
        let increments = response
            .vectors
            .into_iter()
            .zip(anchor.fields())
            .take(anchor.positive_requested())
            .map(|(mut v, x)| {
                for (a, &b) in v.as_mut_slice().iter_mut().zip(x.as_slice()) {
                    *a -= b;
                }
                v.into_field()
            })
            .collect();
        Products::Tangent { increments }
    } else {
        let mut source = ThetaOperator::new(
            target.backend().clone(),
            anchor.material.clone(),
            target.polarization(),
            target.bloch(),
        );
        let (root, inverse, excluded) = source_frame(&source, &anchor);
        let rank = options.columns.min(n - excluded.len());
        let mut preferred = Vec::new();
        match options.basis {
            Basis::Fourier => {}
            Basis::History => {
                if let Some(old) = previous.filter(|s| s.key() == anchor.key()) {
                    preferred = super::predict::imported(&source, old, true);
                }
            }
            Basis::Residual => {
                let delta =
                    DiscreteDifference::from_material(&anchor.material, anchor.key(), target)?;
                let mut pc = source.build_homogeneous_preconditioner_adaptive();
                let mut da = target.alloc_field();
                let mut db = target.alloc_field();
                for j in 0..anchor.positive_requested() {
                    let x = import(&source, &anchor.fields()[j]);
                    delta.apply(target, &x, &mut da, &mut db)?;
                    if target.polarization() == Polarization::TM {
                        report.work.response.difference_b_columns += 1;
                    } else {
                        report.work.response.difference_a_columns += 1;
                    }
                    target
                        .backend()
                        .axpy(C::new(-anchor.eigenvalues()[j], 0.), &db, &mut da);
                    let mut v = da.clone();
                    pc.apply(source.backend(), &mut v);
                    preferred.push(v);
                }
            }
        }
        let tolerance = 1e-10_f64.max(64. * <B::Real as num_traits::Float>::epsilon().to_accum());
        let mut frame: Vec<B::Buffer> = Vec::new();
        let admit = |mut v: B::Buffer, frame: &mut Vec<B::Buffer>| {
            multiply_real::<B>(&mut v, &root);
            let original = source.backend().dot(&v, &v).re.sqrt();
            project_out(source.backend(), &excluded, &mut v);
            project_out(source.backend(), frame, &mut v);
            let norm = source.backend().dot(&v, &v).re.sqrt();
            if norm.is_finite() && norm > tolerance * original && norm > 0. {
                source.backend().scale(C::new(norm.recip(), 0.), &mut v);
                frame.push(v);
                true
            } else {
                false
            }
        };
        for v in preferred {
            if frame.len() == rank {
                break;
            }
            if admit(v, &mut frame) {
                report.preferred_columns += 1;
            }
        }
        let grid = source.grid();
        let b1 = source.dielectric().reciprocal_b1();
        let b2 = source.dielectric().reciprocal_b2();
        let signed = |i: usize, n: usize| {
            if i <= n / 2 {
                i as isize
            } else {
                i as isize - n as isize
            }
        };
        let mut modes: Vec<_> = (0..n)
            .map(|i| {
                let a = signed(i % grid.nx, grid.nx) as f64;
                let b = signed(i / grid.nx, grid.ny) as f64;
                let k = source.bloch();
                let norm =
                    (a * b1[0] + b * b2[0] + k[0]).powi(2) + (a * b1[1] + b * b2[1] + k[1]).powi(2);
                (norm, i, a, b)
            })
            .collect();
        modes.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, _, a, b) in modes {
            if frame.len() == rank {
                break;
            }
            let mut v = source.alloc_field();
            for (i, z) in v.as_mut_slice().iter_mut().enumerate() {
                let x = grid.cartesian_coords(i % grid.nx, i / grid.nx);
                let angle = std::f64::consts::TAU * (a * x[0] / grid.lx + b * x[1] / grid.ly);
                *z = crate::field::cscalar(angle.cos(), angle.sin());
            }
            if admit(v, &mut frame) {
                report.fourier_completion_columns += 1;
            }
        }
        if frame.is_empty() {
            return Err("response complement has no independent columns".into());
        }
        let basis: Vec<_> = frame
            .into_iter()
            .map(|mut z| {
                multiply_real::<B>(&mut z, &inverse);
                z
            })
            .collect();
        let r = basis.len();
        let mut h = vec![C::ZERO; r * r];
        let mut mass = h.clone();
        let mut applied = Vec::new();
        let mut bx = source.alloc_field();
        for j in 0..r {
            let mut ax = source.alloc_field();
            source.apply(&basis[j], &mut ax);
            source.apply_mass(&basis[j], &mut bx);
            report.work.basis_a_columns += 1;
            report.work.basis_b_columns += 1;
            for i in 0..r {
                h[i + j * r] = source.backend().dot(&basis[i], &ax);
                mass[i + j * r] = source.backend().dot(&basis[i], &bx);
            }
            applied.push(ax.into_field());
        }
        hermitian(&mut h, r);
        hermitian(&mut mass, r);
        Products::Reduced {
            basis: basis.into_iter().map(SpectralBuffer::into_field).collect(),
            applied,
            stiffness: h,
            mass,
        }
    };
    let model = Arc::new(ResponseModel {
        anchor,
        training_parameters: parameters.to_vec(),
        options,
        kind,
        gap: separation,
        products,
    });
    report.basis_columns = model.columns();
    report.payload_bytes = model.payload_bytes();
    Ok(model)
}

/// Evaluate the source response only. Full target constraints and Ritz repair
/// are deliberately separate so dense oracles can inspect the actual equation.
pub fn evaluate<B: SpectralBackend + Clone>(
    target: &mut ThetaOperator<B>,
    model: &ResponseModel<B::Real>,
    parameters: &[f64],
    max_bytes: usize,
    report: &mut Attempt,
) -> Result<Vec<B::Buffer>, String> {
    crate::profile_scope!("geometry.model_evaluate");
    if model.key() != &FiberKey::from_operator(target) {
        return Err("response model fiber changed".into());
    }
    report.step_ratio = Some(
        displacement(
            &model.anchor.parameters,
            &model.training_parameters,
            parameters,
            model.kind == Kind::Tangent,
        )
        .filter(|&ratio| ratio <= model.options.max_step)
        .ok_or("response model direction or displacement is outside its trained range")?,
    );
    let n = target.grid().len();
    let p = model.anchor.fields().len();
    let r = model.columns();
    let bytes = (8 * p as u128 + 4 * r as u128 + 32)
        * n as u128
        * std::mem::size_of::<num_complex::Complex<B::Real>>() as u128
        + 16 * 8 * (p as u128 + r as u128).pow(2);
    if bytes > max_bytes as u128 {
        return Err("response model evaluation exceeds its preparation budget".into());
    }
    report.workspace_payload_bytes = report.workspace_payload_bytes.max(bytes as usize);
    let anchor = &model.anchor;
    let mut predicted: Vec<_> = anchor.fields().iter().map(|f| import(target, f)).collect();
    match &model.products {
        Products::Tangent { increments } => {
            let ratio = displacement(
                &anchor.parameters,
                &model.training_parameters,
                parameters,
                true,
            )
            .filter(|&s| s <= model.options.max_step)
            .ok_or("tangent direction or displacement is outside its trained range")?;
            report.step_ratio = Some(ratio);
            for (v, increment) in predicted.iter_mut().zip(increments) {
                for (a, &b) in v.as_mut_slice().iter_mut().zip(increment.as_slice()) {
                    let value = C::new(a.re.to_accum(), a.im.to_accum())
                        + ratio * C::new(b.re.to_accum(), b.im.to_accum());
                    *a = crate::field::cscalar(value.re, value.im);
                }
            }
        }
        Products::Reduced {
            basis,
            applied,
            stiffness,
            mass,
        } => {
            let delta = DiscreteDifference::from_material(&anchor.material, anchor.key(), target)?;
            let source = ThetaOperator::new(
                target.backend().clone(),
                anchor.material.clone(),
                target.polarization(),
                target.bloch(),
            );
            let (root, inverse, excluded) = source_frame(&source, anchor);
            let x: Vec<_> = anchor.fields().iter().map(|f| import(&source, f)).collect();
            let c: Vec<_> = basis.iter().map(|f| import(&source, f)).collect();
            let ac: Vec<_> = applied.iter().map(|f| import(&source, f)).collect();
            let r = basis.len();
            let mut da = target.alloc_field();
            let mut db = target.alloc_field();
            for j in 0..anchor.positive_requested() {
                delta.apply(target, &x[j], &mut da, &mut db)?;
                if target.polarization() == Polarization::TM {
                    report.work.response.difference_b_columns += 1;
                } else {
                    report.work.response.difference_a_columns += 1;
                }
                let normalization: Vec<_> =
                    x.iter().map(|v| source.backend().dot(v, &db)).collect();
                let lambda = anchor.eigenvalues()[j];
                source.backend().axpy(C::new(-lambda, 0.), &db, &mut da);
                let rhs: Vec<_> = c.iter().map(|v| -source.backend().dot(v, &da)).collect();
                let matrix: Vec<_> = stiffness
                    .iter()
                    .zip(mass)
                    .map(|(&a, &b)| a - lambda * b)
                    .collect();
                if matrix
                    .iter()
                    .chain(&rhs)
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
                {
                    return Err("nonfinite reduced response pencil".into());
                }
                let eig = solve_hermitian_eigen(&matrix, r);
                let scale = eig.eigenvalues.iter().copied().fold(0., f64::max);
                if eig
                    .eigenvalues
                    .iter()
                    .any(|&v| !v.is_finite() || v <= 1e-12 * scale.max(1.))
                {
                    return Err("reduced response complement lost positive separation".into());
                }
                let mut coefficients = vec![C::ZERO; r];
                for l in 0..r {
                    let amplitude = (0..r)
                        .map(|i| eig.eigenvectors[i + l * r].conj() * rhs[i])
                        .sum::<C>()
                        / eig.eigenvalues[l];
                    for i in 0..r {
                        coefficients[i] += eig.eigenvectors[i + l * r] * amplitude;
                    }
                }
                let small_error = (0..r)
                    .map(|i| {
                        ((0..r)
                            .map(|k| matrix[i + k * r] * coefficients[k])
                            .sum::<C>()
                            - rhs[i])
                            .norm_sqr()
                    })
                    .sum::<f64>()
                    .sqrt();
                let small_norm = rhs.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                report
                    .projected_relative_residuals
                    .push(if small_norm > 0. {
                        small_error / small_norm
                    } else {
                        small_error
                    });
                let mut correction = source
                    .backend()
                    .linear_combinations(&c, &coefficients, 1)
                    .remove(0);
                let mut action = source
                    .backend()
                    .linear_combinations(&ac, &coefficients, 1)
                    .remove(0);
                let mut forcing = da.clone();
                multiply_real::<B>(&mut forcing, &inverse);
                project_out(source.backend(), &excluded, &mut forcing);
                let initial = source.backend().dot(&forcing, &forcing).re.max(0.).sqrt();
                multiply_real::<B>(&mut action, &inverse);
                source.backend().axpy(C::ONE, &forcing, &mut action);
                let mut z = correction.clone();
                multiply_real::<B>(&mut z, &root);
                source.backend().axpy(C::new(-lambda, 0.), &z, &mut action);
                project_out(source.backend(), &excluded, &mut action);
                let error = source.backend().dot(&action, &action).re.max(0.).sqrt();
                report.full_relative_residuals.push(if initial > 0. {
                    error / initial
                } else {
                    error
                });
                for i in 0..p {
                    source
                        .backend()
                        .axpy(-0.5 * normalization[i], &x[i], &mut correction);
                }
                source
                    .backend()
                    .axpy(C::ONE, &correction, &mut predicted[j]);
            }
        }
    }
    if predicted.iter().any(|v| {
        v.as_slice()
            .iter()
            .any(|z| !z.re.to_accum().is_finite() || !z.im.to_accum().is_finite())
    }) {
        return Err("response model produced nonfinite fields".into());
    }
    Ok(predicted)
}

#[allow(clippy::too_many_arguments)]
pub fn prepare<B: SpectralBackend + Clone>(
    target: &mut ThetaOperator<B>,
    source: Arc<NativeSnapshot<B::Real>>,
    previous: Option<&NativeSnapshot<B::Real>>,
    cached: Option<Lease<B::Real>>,
    parameters: &[f64],
    kind: Kind,
    options: Options,
    quality: Quality,
    retained: usize,
    max_bytes: usize,
) -> (Option<Prediction<B>>, Option<Lease<B::Real>>, Report) {
    crate::profile_scope!("geometry.model_prepare");
    let mut report = Report {
        kind,
        attempts: Vec::new(),
        accepted: false,
    };
    let mut cached = cached;
    if let Some(lease) = &cached {
        if let Err(reason) = reuse_ratio(lease, &source, parameters, kind, options, quality) {
            report.attempts.push(Attempt {
                anchor: Some(lease.model.anchor()),
                reused: true,
                uses: lease.uses,
                rejection: Some(reason),
                ..Default::default()
            });
            cached = None;
        }
    }
    for reuse in [true, false] {
        if reuse && cached.is_none() {
            continue;
        }
        let mut attempt = Attempt {
            reused: reuse,
            ..Default::default()
        };
        let lease = if let Some(lease) = cached.take().filter(|_| reuse) {
            lease
        } else {
            match build(
                target,
                source.clone(),
                previous,
                parameters,
                kind,
                options,
                quality,
                max_bytes,
                &mut attempt,
            ) {
                Ok(model) => Lease { model, uses: 0 },
                Err(reason) => {
                    attempt.rejection = Some(reason);
                    report.attempts.push(attempt);
                    break;
                }
            }
        };
        attempt.anchor = Some(lease.model.anchor());
        attempt.uses = lease.uses;
        attempt.basis_columns = lease.model.columns();
        attempt.payload_bytes = lease.model.payload_bytes();
        attempt.source_guard_gap = Some(lease.model.gap);
        let trial = evaluate(target, &lease.model, parameters, max_bytes, &mut attempt).and_then(
            |vectors| {
                attempt.work.target_preparation_columns = vectors.len();
                rayleigh_ritz_trial(target, vectors, retained, max_bytes)
            },
        );
        match trial {
            Ok(prediction) => {
                let mut bx = target.alloc_field();
                let mut max_relative = 0_f64;
                let mut max_absolute = 0_f64;
                let mut accepted = true;
                for (x, ax) in prediction
                    .vectors
                    .iter()
                    .zip(prediction.actions.as_ref().unwrap())
                {
                    target.apply_mass(x, &mut bx);
                    let norm = target.backend().dot(x, &bx).re;
                    let lambda = target.backend().dot(x, ax).re / norm;
                    let mut residual = ax.clone();
                    target
                        .backend()
                        .axpy(C::new(-lambda, 0.), &bx, &mut residual);
                    let rr = target.backend().dot(&residual, &residual).re.max(0.).sqrt();
                    let denominator = target.backend().dot(ax, ax).re.sqrt()
                        + lambda.abs() * target.backend().dot(&bx, &bx).re.sqrt();
                    let relative = if denominator > 0. {
                        rr / denominator
                    } else {
                        rr
                    };
                    let absolute = rr / target.backend().dot(x, x).re.sqrt();
                    accepted &= relative.is_finite()
                        && absolute.is_finite()
                        && (relative <= options.max_relative_residual
                            || absolute <= quality.absolute_residual);
                    max_relative = max_relative.max(relative);
                    max_absolute = max_absolute.max(absolute);
                }
                attempt.target_max_relative_residual =
                    max_relative.is_finite().then_some(max_relative);
                attempt.target_max_absolute_residual =
                    max_absolute.is_finite().then_some(max_absolute);
                if accepted {
                    report.attempts.push(attempt);
                    report.accepted = true;
                    return (
                        Some(prediction),
                        Some(Lease {
                            model: lease.model,
                            uses: lease.uses + usize::from(reuse),
                        }),
                        report,
                    );
                }
                attempt.rejection =
                    Some("target Ritz prediction exceeds the model residual limit".into());
            }
            Err(reason) => attempt.rejection = Some(reason),
        }
        report.attempts.push(attempt);
    }
    (None, None, report)
}
