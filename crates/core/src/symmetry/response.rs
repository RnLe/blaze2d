//! Fixed-budget geometry response in the B0-whitened excluded subspace.
//!
//! This is an initializer for the unchanged target problem. Positive curvature
//! requires a lowest retained cluster separated from the omitted spectrum.
//! Internal cluster degeneracy is allowed. A truncated inner solve is reported
//! as approximate; it never replaces the target eigensolve or its certificate.
pub use super::response_selection::SelectionReport;
use super::{
    affine::AffineGroup,
    reduced::OrbitBasis,
    response_selection::{ResponseSelection, difference_norm},
};
use super::{
    cache::SnapshotIdentity,
    reuse::{CachedSpectrum, SymmetryOptions},
    seed::{SeedReport, rank_aware_seed},
};
use crate::lattice::Lattice2D;
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::refine::certify_without_refinement,
    field::{Field2D, Real, cscalar},
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
};
use num_complex::Complex64 as C;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ResponseConfig {
    pub max_inner_iterations: usize,
    pub inner_relative_tolerance: f64,
}
impl Default for ResponseConfig {
    fn default() -> Self {
        Self {
            max_inner_iterations: 16,
            inner_relative_tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseColumnReport {
    pub coordinate_rows: usize,
    pub deflation_columns: usize,
    /// Four live CG vectors plus two physical operator buffers. Does not
    /// include the caller's RHS, retained modes, orbit basis or FFT workspace.
    pub cg_workspace_payload_bytes: usize,
    pub iterations: usize,
    pub rhs_norm: f64,
    pub inner_relative_residual: f64,
    pub recursive_relative_residual: f64,
    pub inner_converged: bool,
    pub correction_mass_norm: f64,
    pub excluded_space_defect: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseReport {
    pub config: ResponseConfig,
    pub source_columns: usize,
    pub source_residuals: Vec<f64>,
    pub inner_iterations: usize,
    pub columns: Vec<ResponseColumnReport>,
    pub target_seed: SeedReport,
    pub work: ResponseWork,
    pub selection: Option<SelectionReport>,
}
pub struct ResponseSeed {
    pub fields: Vec<Field2D>,
    pub report: ResponseReport,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ResponseWork {
    pub qualification_a_columns: usize,
    pub qualification_b_columns: usize,
    pub source_certificate_columns: usize,
    pub difference_a_columns: usize,
    pub difference_b_columns: usize,
    pub inner_a_columns: usize,
    pub target_preparation_columns: usize,
}
#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[error("{reason}")]
pub struct ResponseError {
    pub reason: String,
    pub work: ResponseWork,
}

/// Solve the first-order subspace equation using actual finite operator
/// differences, then apply the within-span normalization term -X S/2 and
/// perform target B-normalization/Ritz extraction. All A/B work is profiled.
pub fn predict_response<B: SpectralBackend + Clone>(
    source: &mut ThetaOperator<B>,
    target: &mut ThetaOperator<B>,
    old: &CachedSpectrum,
    config: ResponseConfig,
    quality: &SymmetryOptions,
) -> Result<ResponseSeed, ResponseError> {
    let mut work = ResponseWork::default();
    predict_impl(source, target, old, config, quality, &mut work, None)
        .map_err(|reason| ResponseError { reason, work })
}

/// Restrict response equations to a common qualified unitary irrep. Modes
/// mixing inequivalent irreps must first be separated; they are rejected.
/// A/B still use the full physical FFT grid. Only inner coordinates and
/// forbidden deflation/normalization couplings are reduced.
#[allow(clippy::too_many_arguments)]
pub fn predict_response_symmetric<B: SpectralBackend + Clone>(
    source: &mut ThetaOperator<B>,
    target: &mut ThetaOperator<B>,
    old: &CachedSpectrum,
    config: ResponseConfig,
    quality: &SymmetryOptions,
    lattice: &Lattice2D,
    group: AffineGroup,
) -> Result<ResponseSeed, ResponseError> {
    let mut work = ResponseWork::default();
    predict_impl(
        source,
        target,
        old,
        config,
        quality,
        &mut work,
        Some((lattice, group)),
    )
    .map_err(|reason| ResponseError { reason, work })
}

#[allow(clippy::too_many_arguments)]
fn predict_impl<B: SpectralBackend + Clone>(
    source: &mut ThetaOperator<B>,
    target: &mut ThetaOperator<B>,
    old: &CachedSpectrum,
    config: ResponseConfig,
    quality: &SymmetryOptions,
    work: &mut ResponseWork,
    selection_request: Option<(&Lattice2D, AffineGroup)>,
) -> Result<ResponseSeed, String> {
    crate::profile_scope!("initialization.geometry_response");
    if std::mem::size_of::<B::Real>() != 8 {
        return Err("geometry response currently requires f64".into());
    }
    if config.max_inner_iterations == 0
        || !config.inner_relative_tolerance.is_finite()
        || config.inner_relative_tolerance <= 0.0
        || config.inner_relative_tolerance >= 1.0
    {
        return Err("invalid response iteration budget or tolerance".into());
    }
    let a = SnapshotIdentity::from_operator(source);
    let b = SnapshotIdentity::from_operator(target);
    if (a.nx, a.ny, a.centered, a.storage_bits) != (b.nx, b.ny, b.centered, b.storage_bits)
        || a.operator_bits[..9] != b.operator_bits[..9]
    {
        return Err(
            "geometry response requires the same grid, lattice, Bloch gauge and polarization"
                .into(),
        );
    }
    let n = source.grid().len();
    let p = old.fields.len();
    if p == 0
        || p > n
        || old.eigenvalues.len() != p
        || old.fields.iter().any(|f| {
            f.len() != n
                || (
                    f.grid().nx,
                    f.grid().ny,
                    f.grid().centered,
                    f.grid().lx.to_bits(),
                    f.grid().ly.to_bits(),
                ) != (
                    a.nx,
                    a.ny,
                    a.centered,
                    a.operator_bits[0],
                    a.operator_bits[1],
                )
                || f.as_slice()
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
        })
    {
        return Err("invalid response source fields".into());
    }
    for axis in 0..2 {
        let cart = old.k[0] * source.dielectric().reciprocal_b1()[axis]
            + old.k[1] * source.dielectric().reciprocal_b2()[axis];
        if !cart.is_finite()
            || (cart - source.bloch()[axis]).abs() > 8.0 * f64::EPSILON * cart.abs().max(1.0)
        {
            return Err("response source k metadata differs from its operator".into());
        }
    }
    let mut verified = old.clone();
    work.source_certificate_columns = p;
    let certificate = certify_without_refinement(source, &old.eigenvalues, &old.fields);
    verified.residuals = certificate.residuals;
    verified.absolute_residuals = certificate.absolute_residuals;
    verified.orthogonality_defect = certificate.b_orthogonality_defect;
    if !verified.passes(quality) {
        return Err("response source failed fresh certification".into());
    }
    let mass: Vec<_> = if source.polarization() == Polarization::TM {
        source.dielectric().eps().to_vec()
    } else {
        vec![1.0; n]
    };
    if mass.iter().any(|m| !m.is_finite() || *m <= 0.0)
        || (target.polarization() == Polarization::TM
            && target
                .dielectric()
                .eps()
                .iter()
                .any(|m| !m.is_finite() || *m <= 0.0))
    {
        return Err("response requires positive scalar mass".into());
    }
    let root: Vec<_> = mass.iter().map(|m| m.sqrt()).collect();
    let inverse: Vec<_> = root.iter().map(|m| m.recip()).collect();
    let mut x = Vec::with_capacity(p);
    let mut z = Vec::with_capacity(p);
    for field in &old.fields {
        let mut v = source.alloc_field();
        for (out, a) in v.as_mut_slice().iter_mut().zip(field.as_slice()) {
            *out = cscalar::<B::Real>(a.re, a.im);
        }
        let mut whitened = v.clone();
        multiply_real::<B>(&mut whitened, &root);
        x.push(v);
        z.push(whitened);
    }
    let mut selection = selection_request
        .map(|(lattice, group)| {
            ResponseSelection::new(source, target, &z, lattice, old.k, group, quality, work)
        })
        .transpose()?;
    let mut delta_mass = Vec::with_capacity(p);
    let mut rhs = Vec::with_capacity(p);
    let mut old_applied = source.alloc_field();
    let mut new_applied = target.alloc_field();
    let mut old_mass = source.alloc_field();
    for j in 0..p {
        source.apply(&x[j], &mut old_applied);
        target.apply(&x[j], &mut new_applied);
        work.difference_a_columns += 2;
        source
            .backend()
            .axpy(-C::ONE, &old_applied, &mut new_applied);
        source.apply_mass(&x[j], &mut old_mass);
        let mut change = target.alloc_field();
        target.apply_mass(&x[j], &mut change);
        source.backend().axpy(-C::ONE, &old_mass, &mut change);
        work.difference_b_columns += 2;
        source
            .backend()
            .axpy(C::new(-old.eigenvalues[j], 0.0), &change, &mut new_applied);
        let mut forcing = new_applied.clone();
        multiply_real::<B>(&mut forcing, &inverse);
        source.backend().scale(-C::ONE, &mut forcing);
        if selection.is_none() {
            project_out(source.backend(), &z, &mut forcing);
        }
        delta_mass.push(change);
        rhs.push(forcing);
    }
    let mut s = vec![C::ZERO; p * p];
    for j in 0..p {
        for i in 0..p {
            if selection
                .as_ref()
                .is_some_and(|s| s.labels[i] != s.labels[j])
            {
                continue;
            }
            s[i + p * j] = source.backend().dot(&x[i], &delta_mass[j]);
            crate::profiler::count("response.normalization_dot_terms", n);
        }
    }
    for j in 0..p {
        for i in 0..=j {
            let value = 0.5 * (s[i + p * j] + s[j + p * i].conj());
            s[i + p * j] = value;
            s[j + p * i] = value.conj();
        }
    }
    let mut correction = Vec::with_capacity(p);
    let mut reports = Vec::with_capacity(p);
    for j in 0..p {
        let (mut y, report) = if p == n || (old.k == [0.0; 2] && old.eigenvalues[j] == 0.0) {
            (
                source.alloc_field(),
                ResponseColumnReport {
                    coordinate_rows: selection
                        .as_ref()
                        .map_or(n, |s| s.bases[s.labels[j]].rank()),
                    deflation_columns: selection
                        .as_ref()
                        .map_or(p, |s| s.deflation[s.labels[j]].len()),
                    cg_workspace_payload_bytes: 0,
                    iterations: 0,
                    rhs_norm: source.backend().dot(&rhs[j], &rhs[j]).re.sqrt(),
                    inner_relative_residual: 0.0,
                    recursive_relative_residual: 0.0,
                    inner_converged: true,
                    correction_mass_norm: 0.0,
                    excluded_space_defect: 0.0,
                },
            )
        } else if let Some(selection) = &mut selection {
            let label = selection.labels[j];
            let basis = &selection.bases[label];
            let mut small_rhs = source.backend().alloc_field(basis.coordinate_grid());
            basis.restrict(&rhs[j], &mut small_rhs);
            let mut projected = source.alloc_field();
            basis.expand(&small_rhs, &mut projected);
            let norm = source.backend().dot(&rhs[j], &rhs[j]).re.sqrt();
            let defect = difference_norm(&rhs[j], &projected) / norm.max(1.0);
            selection.report.maximum_rhs_sector_defect =
                selection.report.maximum_rhs_sector_defect.max(defect);
            if !defect.is_finite() || defect > quality.qualification_tolerance {
                return Err("response forcing leaks outside its qualified irrep".into());
            }
            project_out(
                source.backend(),
                &selection.deflation[label],
                &mut small_rhs,
            );
            let (small, report) = solve_correction(
                source,
                &selection.deflation[label],
                &inverse,
                old.eigenvalues[j],
                &small_rhs,
                config,
                work,
                Some(basis),
            )?;
            basis.expand(&small, &mut projected);
            (projected, report)
        } else {
            solve_correction(
                source,
                &z,
                &inverse,
                old.eigenvalues[j],
                &rhs[j],
                config,
                work,
                None,
            )?
        };
        multiply_real::<B>(&mut y, &inverse);
        correction.push(y);
        reports.push(report);
    }
    let mut predicted = Vec::with_capacity(p);
    for j in 0..p {
        let mut field = x[j].clone();
        if old.k == [0.0; 2] && old.eigenvalues[j] == 0.0 {
            // This physical kernel is geometry independent. Keep it exactly
            // instead of introducing an O(delta^2) gauge-mixing error.
            source.fill_gamma_kernel(&mut field);
            predicted.push(promote::<B>(&field));
            continue;
        }
        source.backend().axpy(C::ONE, &correction[j], &mut field);
        for i in 0..p {
            source
                .backend()
                .axpy(-0.5 * s[i + p * j], &x[i], &mut field);
        }
        predicted.push(promote::<B>(&field));
    }
    work.target_preparation_columns = predicted.len();
    let seed = rank_aware_seed(target, &predicted, quality.relative_rank_tolerance)
        .map_err(|e| e.to_string())?;
    if seed.fields.len() != p {
        return Err("geometry response lost target seed rank".into());
    }
    Ok(ResponseSeed {
        fields: seed.fields,
        report: ResponseReport {
            config,
            source_columns: p,
            source_residuals: verified.residuals,
            inner_iterations: reports.iter().map(|r| r.iterations).sum(),
            columns: reports,
            target_seed: seed.report,
            work: work.clone(),
            selection: selection.map(|s| s.report),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_correction<B: SpectralBackend>(
    operator: &mut ThetaOperator<B>,
    z: &[B::Buffer],
    inverse: &[f64],
    lambda: f64,
    rhs: &B::Buffer,
    config: ResponseConfig,
    work: &mut ResponseWork,
    basis: Option<&OrbitBasis>,
) -> Result<(B::Buffer, ResponseColumnReport), String> {
    let result = solve_correction_impl(operator, z, inverse, lambda, rhs, config, work, basis)?;
    Ok((result.whitened, result.report))
}

pub(crate) struct CorrectionProducts<B: SpectralBackend> {
    pub whitened: B::Buffer,
    pub physical: B::Buffer,
    pub applied: B::Buffer,
    pub report: ResponseColumnReport,
}
/// Native physical correction and its fresh source action from the final inner
/// certificate. Coordinate-space callers retain the original wrapper above.
pub(crate) fn solve_correction_products<B: SpectralBackend>(
    operator: &mut ThetaOperator<B>,
    z: &[B::Buffer],
    inverse: &[f64],
    lambda: f64,
    rhs: &B::Buffer,
    config: ResponseConfig,
    work: &mut ResponseWork,
) -> Result<CorrectionProducts<B>, String> {
    solve_correction_impl(operator, z, inverse, lambda, rhs, config, work, None)
}

#[allow(clippy::too_many_arguments)]
fn solve_correction_impl<B: SpectralBackend>(
    operator: &mut ThetaOperator<B>,
    z: &[B::Buffer],
    inverse: &[f64],
    lambda: f64,
    rhs: &B::Buffer,
    config: ResponseConfig,
    work: &mut ResponseWork,
    basis: Option<&OrbitBasis>,
) -> Result<CorrectionProducts<B>, String> {
    let mut solution = operator.backend().alloc_field(rhs.grid());
    let mut residual = rhs.clone();
    let mut direction = rhs.clone();
    let mut applied = operator.backend().alloc_field(rhs.grid());
    let mut physical = operator.alloc_field();
    let mut aphysical = operator.alloc_field();
    let initial = operator.backend().dot(rhs, rhs).re;
    if !initial.is_finite() || initial < 0.0 {
        return Err("nonfinite response right-hand side".into());
    }
    let mut rr = initial;
    let mut iterations = 0;
    for step in 0..config.max_inner_iterations {
        if rr <= config.inner_relative_tolerance.powi(2) * initial {
            break;
        }
        apply_correction_operator(
            operator,
            z,
            inverse,
            lambda,
            &direction,
            &mut applied,
            &mut physical,
            &mut aphysical,
            work,
            basis,
        );
        let curvature = operator.backend().dot(&direction, &applied);
        let scale = (operator.backend().dot(&direction, &direction).re
            * operator.backend().dot(&applied, &applied).re)
            .sqrt();
        if !curvature.re.is_finite()
            || !curvature.im.is_finite()
            || curvature.re <= 0.0
            || curvature.im.abs()
                > 1e-9_f64.max(64.0 * <B::Real as num_traits::Float>::epsilon().to_accum()) * scale
        {
            return Err("response complement lost Hermitian positive curvature; check retained-cluster coverage and separation".into());
        }
        let alpha = rr / curvature.re;
        operator
            .backend()
            .axpy(C::new(alpha, 0.0), &direction, &mut solution);
        operator
            .backend()
            .axpy(C::new(-alpha, 0.0), &applied, &mut residual);
        project_out(operator.backend(), z, &mut residual);
        let next = operator.backend().dot(&residual, &residual).re;
        if !next.is_finite() || next < 0.0 {
            return Err("response residual became nonfinite".into());
        }
        iterations = step + 1;
        operator
            .backend()
            .scale(C::new(next / rr, 0.0), &mut direction);
        operator.backend().axpy(C::ONE, &residual, &mut direction);
        rr = next;
    }
    project_out(operator.backend(), z, &mut solution);
    let recursive_relative = if initial == 0.0 {
        0.0
    } else {
        (rr / initial).sqrt()
    };
    // Certify the returned increment with a fresh operator application. The
    // recurrence residual alone can become optimistic near roundoff.
    let relative = if initial == 0.0 {
        0.0
    } else {
        apply_correction_operator(
            operator,
            z,
            inverse,
            lambda,
            &solution,
            &mut applied,
            &mut physical,
            &mut aphysical,
            work,
            basis,
        );
        operator.backend().axpy(-C::ONE, rhs, &mut applied);
        (operator.backend().dot(&applied, &applied).re / initial).sqrt()
    };
    let norm = operator
        .backend()
        .dot(&solution, &solution)
        .re
        .max(0.0)
        .sqrt();
    let defect = z
        .iter()
        .map(|column| operator.backend().dot(column, &solution).norm())
        .fold(0.0, f64::max)
        / norm.max(f64::MIN_POSITIVE);
    Ok(CorrectionProducts {
        whitened: solution,
        physical,
        applied: aphysical,
        report: ResponseColumnReport {
            coordinate_rows: rhs.len(),
            deflation_columns: z.len(),
            cg_workspace_payload_bytes: (4 * rhs.len() + 2 * operator.grid().len())
                * std::mem::size_of::<num_complex::Complex<B::Real>>(),
            iterations,
            rhs_norm: initial.sqrt(),
            inner_relative_residual: relative,
            recursive_relative_residual: recursive_relative,
            inner_converged: relative <= config.inner_relative_tolerance,
            correction_mass_norm: norm,
            excluded_space_defect: defect,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_correction_operator<B: SpectralBackend>(
    operator: &mut ThetaOperator<B>,
    z: &[B::Buffer],
    inverse: &[f64],
    lambda: f64,
    input: &B::Buffer,
    output: &mut B::Buffer,
    physical: &mut B::Buffer,
    applied: &mut B::Buffer,
    work: &mut ResponseWork,
    basis: Option<&OrbitBasis>,
) {
    if let Some(basis) = basis {
        output.as_mut_slice().copy_from_slice(input.as_slice());
        project_out(operator.backend(), z, output);
        basis.expand(output, physical);
    } else {
        physical.as_mut_slice().copy_from_slice(input.as_slice());
        project_out(operator.backend(), z, physical);
    }
    multiply_real::<B>(physical, inverse);
    operator.apply(physical, applied);
    work.inner_a_columns += 1;
    if let Some(basis) = basis {
        for ((dst, &src), &factor) in physical
            .as_mut_slice()
            .iter_mut()
            .zip(applied.as_slice())
            .zip(inverse)
        {
            *dst = src * B::Real::from_accum(factor);
        }
        basis.restrict(physical, output);
    } else {
        for ((dst, &src), &factor) in output
            .as_mut_slice()
            .iter_mut()
            .zip(applied.as_slice())
            .zip(inverse)
        {
            *dst = src * B::Real::from_accum(factor);
        }
    }
    operator.backend().axpy(C::new(-lambda, 0.0), input, output);
    project_out(operator.backend(), z, output);
    crate::profiler::count("response.inner_operator_columns", 1);
}

pub(crate) fn project_out<B: SpectralBackend>(
    backend: &B,
    z: &[B::Buffer],
    vector: &mut B::Buffer,
) {
    for _ in 0..2 {
        for column in z {
            let overlap = backend.dot(column, vector);
            backend.axpy(-overlap, column, vector);
        }
    }
    crate::profiler::count("response.deflation_dot_terms", 2 * z.len() * vector.len());
}
pub(crate) fn multiply_real<B: SpectralBackend>(vector: &mut B::Buffer, factors: &[f64]) {
    for (z, &a) in vector.as_mut_slice().iter_mut().zip(factors) {
        *z *= B::Real::from_accum(a);
    }
}
fn promote<B: SpectralBackend>(field: &B::Buffer) -> Field2D {
    Field2D::from_vec(
        field.grid(),
        field
            .as_slice()
            .iter()
            .map(|z| C::new(z.re.to_accum(), z.im.to_accum()))
            .collect(),
    )
}
