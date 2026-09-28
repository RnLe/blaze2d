//! Budgeted source response with a B0-whitened, orthogonal excluded space.
use super::{
    difference::{DifferenceReport, DiscreteDifference},
    snapshot::{NativeSnapshot, Quality},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    field::Real,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::response::{
        ResponseColumnReport, ResponseConfig, ResponseError, ResponseWork, multiply_real,
        project_out, solve_correction_products,
    },
};
use num_complex::Complex64 as C;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct NativeResponseReport {
    pub difference: DifferenceReport,
    pub source_relative_residuals: Vec<f64>,
    pub source_dual_residuals: Vec<f64>,
    pub source_dual_relative_residuals: Vec<f64>,
    pub excluded_columns: usize,
    pub includes_gamma_kernel: bool,
    pub response_columns: usize,
    pub copied_guards: usize,
    pub inner_columns: Vec<ResponseColumnReport>,
    pub work: ResponseWork,
    pub reused_source_actions: usize,
    pub reused_correction_actions: usize,
    pub estimated_workspace_payload_bytes: usize,
}
pub struct NativeResponse<B: SpectralBackend> {
    pub vectors: Vec<B::Buffer>,
    pub actions: Option<Vec<B::Buffer>>,
    pub report: NativeResponseReport,
}

pub fn predict<B: SpectralBackend + Clone>(
    target: &mut ThetaOperator<B>,
    snapshot: &NativeSnapshot<B::Real>,
    config: ResponseConfig,
    quality: Quality,
    reuse_actions: bool,
    max_bytes: usize,
) -> Result<NativeResponse<B>, ResponseError> {
    let mut work = ResponseWork::default();
    predict_impl(
        target,
        snapshot,
        config,
        quality,
        reuse_actions,
        max_bytes,
        &mut work,
    )
    .map_err(|reason| ResponseError { reason, work })
}
fn predict_impl<B: SpectralBackend + Clone>(
    target: &mut ThetaOperator<B>,
    snapshot: &NativeSnapshot<B::Real>,
    config: ResponseConfig,
    quality: Quality,
    reuse_actions: bool,
    max_bytes: usize,
    work: &mut ResponseWork,
) -> Result<NativeResponse<B>, String> {
    crate::profile_scope!("geometry.source_response");
    if config.max_inner_iterations == 0
        || config.max_inner_iterations > 4096
        || !config.inner_relative_tolerance.is_finite()
        || config.inner_relative_tolerance <= 0.
        || config.inner_relative_tolerance >= 1.
    {
        return Err("invalid bounded source-response configuration".into());
    }
    if !snapshot.admitted(quality) {
        return Err("source response requires a directly admitted native block".into());
    }
    let n = target.grid().len();
    let p = snapshot.fields().len();
    let columns = snapshot.positive_requested();
    let bytes = (8 * p as u128 + 32)
        * n as u128
        * std::mem::size_of::<num_complex::Complex<B::Real>>() as u128
        + n as u128 * 8 * 16
        + (p as u128).pow(2) * 16 * 4;
    if bytes > max_bytes as u128 {
        return Err("source response exceeds its workspace payload budget".into());
    }
    let difference = DiscreteDifference::from_material(&snapshot.material, snapshot.key(), target)?;
    let mut source = ThetaOperator::new(
        target.backend().clone(),
        snapshot.material.clone(),
        target.polarization(),
        target.bloch(),
    );
    let root: Vec<_> = (0..n)
        .map(|i| {
            if source.polarization() == Polarization::TM {
                snapshot.material.eps()[i].sqrt()
            } else {
                1.
            }
        })
        .collect();
    let inverse: Vec<_> = root.iter().map(|x| x.recip()).collect();
    let mut x = Vec::with_capacity(p);
    let mut excluded = Vec::with_capacity(p + 1);
    if let Some(kernel) = snapshot.kernel() {
        let mut q = source.alloc_field();
        q.as_mut_slice().copy_from_slice(kernel.as_slice());
        multiply_real::<B>(&mut q, &root);
        excluded.push(q);
    }
    for field in snapshot.fields() {
        let mut v = source.alloc_field();
        v.as_mut_slice().copy_from_slice(field.as_slice());
        let mut z = v.clone();
        multiply_real::<B>(&mut z, &root);
        excluded.push(z);
        x.push(v);
    }
    // All retained positives and actual guards enter the excluded space. Only
    // requested positives receive a response solve; guard tangents are not
    // needed for first-order coverage of the lower target Ritz window.
    let cached = if reuse_actions {
        snapshot.applied_for(snapshot.key())
    } else {
        None
    };
    let mut predicted = Vec::with_capacity(p);
    let mut predicted_actions = cached.map(|_| Vec::with_capacity(p));
    let mut reports = Vec::new();
    let mut da = target.alloc_field();
    let mut db = target.alloc_field();
    for j in 0..p {
        if j >= columns {
            predicted.push(x[j].clone());
            if let Some(actions) = &mut predicted_actions {
                let mut ax = source.alloc_field();
                ax.as_mut_slice()
                    .copy_from_slice(cached.unwrap()[j].as_slice());
                actions.push(ax);
            }
            continue;
        }
        difference.apply(target, &x[j], &mut da, &mut db)?;
        if target.polarization() == Polarization::TM {
            work.difference_b_columns += 1;
        } else {
            work.difference_a_columns += 1;
        }
        let normalization: Vec<_> = x.iter().map(|v| source.backend().dot(v, &db)).collect();
        source
            .backend()
            .axpy(C::new(-snapshot.eigenvalues()[j], 0.), &db, &mut da);
        multiply_real::<B>(&mut da, &inverse);
        source.backend().scale(-C::ONE, &mut da);
        project_out(source.backend(), &excluded, &mut da);
        let correction = solve_correction_products(
            &mut source,
            &excluded,
            &inverse,
            snapshot.eigenvalues()[j],
            &da,
            config,
            work,
        )?;
        let mut field = x[j].clone();
        source
            .backend()
            .axpy(C::ONE, &correction.physical, &mut field);
        for i in 0..p {
            source
                .backend()
                .axpy(-0.5 * normalization[i], &x[i], &mut field);
        }
        if let Some(actions) = &mut predicted_actions {
            let mut ax = correction.applied;
            for i in 0..p {
                let coefficient = (if i == j { C::ONE } else { C::ZERO }) - 0.5 * normalization[i];
                // Native immutable source products remain on the same TM A.
                for (dst, &value) in ax
                    .as_mut_slice()
                    .iter_mut()
                    .zip(cached.unwrap()[i].as_slice())
                {
                    let a = C::new(dst.re.to_accum(), dst.im.to_accum())
                        + coefficient * C::new(value.re.to_accum(), value.im.to_accum());
                    *dst = crate::field::cscalar(a.re, a.im);
                }
            }
            actions.push(ax);
        }
        predicted.push(field);
        reports.push(correction.report);
    }
    if predicted.iter().any(|v| {
        v.as_slice()
            .iter()
            .any(|z| !z.re.to_accum().is_finite() || !z.im.to_accum().is_finite())
    }) {
        return Err("source response produced a nonfinite trial field".into());
    }
    Ok(NativeResponse {
        vectors: predicted,
        actions: predicted_actions,
        report: NativeResponseReport {
            difference: difference.report,
            source_relative_residuals: snapshot.certificate().relative.clone(),
            source_dual_residuals: snapshot.certificate().dual.clone(),
            source_dual_relative_residuals: snapshot.certificate().dual_relative.clone(),
            excluded_columns: excluded.len(),
            includes_gamma_kernel: snapshot.kernel().is_some(),
            response_columns: columns,
            copied_guards: snapshot.guard_count(),
            inner_columns: reports,
            work: work.clone(),
            reused_source_actions: if cached.is_some() { p } else { 0 },
            reused_correction_actions: if cached.is_some() { columns } else { 0 },
            estimated_workspace_payload_bytes: bytes as usize,
        },
    })
}
