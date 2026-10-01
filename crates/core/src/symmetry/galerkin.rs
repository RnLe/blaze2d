//! Maxwell pencils on an explicitly different finite plane-wave space.
//!
//! A_S = S† A S and B_S = S† B S, using `FourierSupport`'s isometric sampled
//! plane waves. Projected residuals certify this pencil; residuals against the
//! original FFT rectangle measure discretization error and do not gate it.
//! No symmetry reduction is implied by choosing a smaller support.
use num_complex::Complex64 as C;
use serde::Serialize;

use super::support::{FourierSupport, SupportIdentity};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{
        BlockCertification, Eigensolver, EigensolverConfig,
        refine::{certify_without_refinement, rayleigh_ritz_certify},
    },
    field::{Field2D, Real, cscalar, from_accum, to_accum},
    grid::Grid2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    preconditioners::OperatorPreconditioner,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GalerkinApplication {
    SpatialReference,
    DirectFourier,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct GalerkinBudget {
    /// Admission limit for the declared numerical workspace model, not RSS.
    /// Excludes the caller's existing operator, FFT plans, allocator overhead,
    /// and library-dependent dense scratch. These need separate measurement.
    pub max_estimated_payload_bytes: usize,
    pub max_dense_dimension: usize,
}
impl Default for GalerkinBudget {
    fn default() -> Self {
        Self {
            max_estimated_payload_bytes: 512 * 1024 * 1024,
            max_dense_dimension: 512,
        }
    }
}

/// Conservative numerical payload model for admission before solver allocation.
/// Includes wrapper/preconditioner fields, solver and certification columns,
/// dense matrices, coefficient results and all requested physical outputs.
pub fn estimated_payload_bytes(
    support: &FourierSupport,
    config: &EigensolverConfig,
    dense: bool,
) -> Result<usize, String> {
    estimated_payload_bytes_for_dimensions(support.grid().len(), support.len(), config, dense)
}

/// Dimension-only form for interface planning across geometry/grid sweeps.
/// An upper bound on coefficient count yields a conservative payload model;
/// the actual support and solver configuration are still admitted at execution.
pub fn estimated_payload_bytes_for_dimensions(
    embedding_points: usize,
    coefficients: usize,
    config: &EigensolverConfig,
    dense: bool,
) -> Result<usize, String> {
    let n = embedding_points as u128;
    let m = coefficients as u128;
    let b = config.effective_block_size().min(coefficients) as u128;
    let r = config.n_bands as u128;
    let product = |terms: &[u128]| terms.iter().try_fold(1_u128, |a, &b| a.checked_mul(b));
    let terms = if dense {
        [
            product(&[8, n]),
            product(&[24, m, m]),
            product(&[16, m, r]),
            product(&[4 * r + 2, n]),
        ]
    } else {
        [
            product(&[8, n]),
            product(&[64, m, b]),
            product(&[128, b, b]),
            product(&[4 * r + 2, n]),
        ]
    };
    let bytes = terms
        .into_iter()
        .try_fold(0_u128, |sum, term| sum.checked_add(term?))
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(m * (16 + std::mem::size_of::<usize>()) as u128))
        .ok_or("Galerkin workspace estimate overflow")?;
    usize::try_from(bytes).map_err(|_| "Galerkin workspace estimate overflow".into())
}

fn same_grid(a: Grid2D, b: Grid2D) -> bool {
    (a.nx, a.ny, a.lx, a.ly, a.centered) == (b.nx, b.ny, b.lx, b.ly, b.centered)
}

pub(super) fn validate_operator<B: SpectralBackend>(
    full: &ThetaOperator<B>,
    support: &FourierSupport,
) -> Result<(), String> {
    if std::any::TypeId::of::<B::Real>() != std::any::TypeId::of::<f64>() {
        return Err(
            "Galerkin solves require an f64 backend; precision is never silently promoted".into(),
        );
    }
    let reciprocal = support.lattice().reciprocal();
    let d = full.dielectric();
    if !same_grid(full.grid(), support.grid())
        || reciprocal.b1 != d.reciprocal_b1()
        || reciprocal.b2 != d.reciprocal_b2()
        || reciprocal.fractional_to_cartesian(support.k()) != full.bloch()
    {
        return Err(
            "support and operator must have identical grid, reciprocal basis and Bloch vector"
                .into(),
        );
    }
    if d.eps().iter().any(|x| !x.is_finite() || *x <= 0.0)
        || d.inv_eps().iter().any(|x| !x.is_finite() || *x <= 0.0)
        || full.k_plus_g_squares().iter().any(|x| !x.is_finite())
    {
        return Err(
            "Galerkin operator requires finite positive material and wavevector tables".into(),
        );
    }
    if full.polarization() == Polarization::TE {
        if let Some(tensors) = d.inv_eps_tensors() {
            for &[xx, xy, yx, yy] in tensors {
                let scale = xx.abs().max(xy.abs()).max(yx.abs()).max(yy.abs());
                if !scale.is_finite()
                    || scale == 0.0
                    || [xx, xy, yx, yy].iter().any(|x| !x.is_finite())
                    || (xy - yx).abs() > 64.0 * f64::EPSILON * scale
                    || xx <= 0.0
                    || yy <= 0.0
                    || (xx / scale) * (yy / scale) <= (xy / scale) * (yx / scale)
                {
                    return Err("Galerkin TE requires a positive Hermitian sampled tensor".into());
                }
            }
        }
    }
    Ok(())
}

/// Public coefficient-space operator, useful for independently testing its
/// congruence and for subsequent qualified reduction on exactly this pencil.
/// Generic callers may pass any backend; construction explicitly rejects
/// storage other than f64 before allocating the projected workspace.
pub struct GalerkinOperator<'a, B: SpectralBackend> {
    full: &'a mut ThetaOperator<B>,
    support: &'a FourierSupport,
    application: GalerkinApplication,
    input: B::Buffer,
    output: B::Buffer,
    scratch: B::Buffer,
    kernel_index: Option<usize>,
}
impl<'a, B: SpectralBackend> GalerkinOperator<'a, B> {
    /// The workspace limit here bounds only the three newly allocated full
    /// fields. Solver entry points apply their larger payload model first.
    pub fn new(
        full: &'a mut ThetaOperator<B>,
        support: &'a FourierSupport,
        application: GalerkinApplication,
        max_workspace_bytes: usize,
    ) -> Result<Self, String> {
        validate_operator(full, support)?;
        let bytes = support
            .grid()
            .len()
            .checked_mul(48)
            .ok_or("Galerkin wrapper size overflow")?;
        if bytes > max_workspace_bytes {
            return Err("Galerkin wrapper exceeds its workspace allowance".into());
        }
        let kernel_index = support
            .embedding_indices()
            .iter()
            .position(|&i| full.k_plus_g_squares()[i] == 0.0);
        let input = full.alloc_field();
        let output = full.alloc_field();
        let scratch = full.alloc_field();
        Ok(Self {
            full,
            support,
            application,
            input,
            output,
            scratch,
            kernel_index,
        })
    }

    pub fn support(&self) -> &FourierSupport {
        self.support
    }

    pub(super) fn project_physical(
        &mut self,
        fields: &[Field2D],
        limit: usize,
    ) -> Result<Vec<Field2D>, String> {
        let mut result = Vec::new();
        for field in fields.iter().take(limit) {
            if !same_grid(field.grid(), self.support.grid())
                || field.as_slice().iter().any(|z| !finite(*z))
            {
                return Err(
                    "Galerkin warm fields must be finite and use the same physical grid".into(),
                );
            }
            for (dst, &src) in self.input.as_mut_slice().iter_mut().zip(field.as_slice()) {
                *dst = from_accum(src);
            }
            let mut small = Field2D::zeros(self.grid());
            self.support.restrict(
                self.full.backend(),
                &self.input,
                &mut self.scratch,
                small.as_mut_slice(),
            );
            result.push(small);
        }
        Ok(result)
    }

    pub(super) fn expand_fields(&mut self, coefficients: &[Field2D]) -> Vec<Field2D> {
        coefficients
            .iter()
            .map(|field| {
                self.support
                    .expand(self.full.backend(), field.as_slice(), &mut self.input);
                Field2D::from_vec(
                    self.support.grid(),
                    self.input
                        .as_slice()
                        .iter()
                        .copied()
                        .map(to_accum)
                        .collect(),
                )
            })
            .collect()
    }
}

impl<B: SpectralBackend> LinearOperator<B> for GalerkinOperator<'_, B> {
    fn apply(&mut self, x: &B::Buffer, y: &mut B::Buffer) {
        assert_eq!(x.len(), self.support.len());
        assert_eq!(y.len(), self.support.len());
        crate::profile_scope!("symmetry.galerkin.apply");
        crate::profiler::count("symmetry.galerkin.A_coefficients", self.support.len());
        match (self.application, self.full.polarization()) {
            (GalerkinApplication::DirectFourier, Polarization::TM) => {
                crate::profiler::count("operator_A.columns", 1);
                crate::profiler::count("operator_A.elements", self.support.len());
                for ((dst, &src), &i) in y
                    .as_mut_slice()
                    .iter_mut()
                    .zip(x.as_slice())
                    .zip(self.support.embedding_indices())
                {
                    *dst = src * B::Real::from_accum(self.full.k_plus_g_squares()[i]);
                }
            }
            (GalerkinApplication::DirectFourier, Polarization::TE) => {
                self.input.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
                for (&i, &value) in self.support.embedding_indices().iter().zip(x.as_slice()) {
                    self.input.as_mut_slice()[i] = value;
                }
                self.full.apply_fourier(&self.input, &mut self.output);
                for (&i, value) in self
                    .support
                    .embedding_indices()
                    .iter()
                    .zip(y.as_mut_slice())
                {
                    *value = self.output.as_slice()[i];
                }
            }
            (GalerkinApplication::SpatialReference, _) => {
                self.support
                    .expand(self.full.backend(), x.as_slice(), &mut self.input);
                self.full.apply(&self.input, &mut self.output);
                self.support.restrict(
                    self.full.backend(),
                    &self.output,
                    &mut self.scratch,
                    y.as_mut_slice(),
                );
            }
        }
    }
    fn apply_mass(&mut self, x: &B::Buffer, y: &mut B::Buffer) {
        assert_eq!(x.len(), self.support.len());
        assert_eq!(y.len(), self.support.len());
        crate::profile_scope!("symmetry.galerkin.mass");
        if self.application == GalerkinApplication::DirectFourier
            && self.full.polarization() == Polarization::TE
        {
            crate::profiler::count("operator_B.columns", 1);
            crate::profiler::count("operator_B.elements", self.support.len());
            y.as_mut_slice().copy_from_slice(x.as_slice());
        } else {
            self.support
                .expand(self.full.backend(), x.as_slice(), &mut self.input);
            self.full.apply_mass(&self.input, &mut self.output);
            self.support.restrict(
                self.full.backend(),
                &self.output,
                &mut self.scratch,
                y.as_mut_slice(),
            );
        }
    }
    fn alloc_field(&self) -> B::Buffer {
        self.full.backend().alloc_field(self.grid())
    }
    fn backend(&self) -> &B {
        self.full.backend()
    }
    fn backend_mut(&mut self) -> &mut B {
        self.full.backend_mut()
    }
    fn grid(&self) -> Grid2D {
        self.support.coordinate_grid()
    }
    fn bloch(&self) -> [f64; 2] {
        self.full.bloch()
    }
    fn subspace_dimension(&self) -> usize {
        self.support.len()
    }
    fn owns_gamma_kernel(&self) -> bool {
        self.kernel_index.is_some()
    }
    fn analytic_kernel_available(&self) -> bool {
        self.kernel_index.is_some()
    }
    fn fill_gamma_kernel(&self, vector: &mut B::Buffer) {
        vector.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
        if let Some(i) = self.kernel_index {
            vector.as_mut_slice()[i] = cscalar::<B::Real>(1.0, 0.0);
        }
    }
    fn seed_initial_vector(&mut self, vector: &mut B::Buffer, phase: f64) {
        // Seed the same physical field before restriction, independent of
        // whether direct or literal spatial application is requested.
        crate::eigensolver::initialization::seed_random_vector(self.input.as_mut_slice(), phase);
        self.support.restrict(
            self.full.backend(),
            &self.input,
            &mut self.scratch,
            vector.as_mut_slice(),
        );
    }
}

pub(super) struct GalerkinPreconditioner<'a, B: SpectralBackend> {
    inner: &'a mut dyn OperatorPreconditioner<B>,
    support: &'a FourierSupport,
    application: GalerkinApplication,
    input: B::Buffer,
    scratch: B::Buffer,
}
impl<'a, B: SpectralBackend> GalerkinPreconditioner<'a, B> {
    pub(super) fn new(
        inner: &'a mut dyn OperatorPreconditioner<B>,
        full: &ThetaOperator<B>,
        support: &'a FourierSupport,
        application: GalerkinApplication,
    ) -> Result<Self, String> {
        if inner.fourier_diagonal().is_some_and(|d| {
            d.len() != support.grid().len()
                || d.iter().any(|x| !x.is_finite())
                || support
                    .embedding_indices()
                    .iter()
                    .any(|&i| d[i] < 0.0 || (d[i] == 0.0 && full.k_plus_g_squares()[i] != 0.0))
        }) {
            return Err("Galerkin preconditioner diagonal is invalid for the embedding".into());
        }
        Ok(Self {
            inner,
            support,
            application,
            input: full.alloc_field(),
            scratch: full.alloc_field(),
        })
    }
}
impl<B: SpectralBackend> OperatorPreconditioner<B> for GalerkinPreconditioner<'_, B> {
    fn apply(&mut self, backend: &B, vector: &mut B::Buffer) {
        crate::profile_scope!("symmetry.galerkin.preconditioner");
        if self.application == GalerkinApplication::DirectFourier {
            if let Some(diagonal) = self.inner.fourier_diagonal() {
                crate::profiler::count("symmetry.galerkin.diagonal_columns", 1);
                for (value, &i) in vector
                    .as_mut_slice()
                    .iter_mut()
                    .zip(self.support.embedding_indices())
                {
                    *value *= B::Real::from_accum(diagonal[i]);
                }
                return;
            }
            self.input.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
            for (&i, &value) in self
                .support
                .embedding_indices()
                .iter()
                .zip(vector.as_slice())
            {
                self.input.as_mut_slice()[i] = value;
            }
            if self.inner.apply_fourier(backend, &mut self.input) {
                for (&i, value) in self
                    .support
                    .embedding_indices()
                    .iter()
                    .zip(vector.as_mut_slice())
                {
                    *value = self.input.as_slice()[i];
                }
                return;
            }
        }
        self.support
            .expand(backend, vector.as_slice(), &mut self.input);
        self.inner.apply(backend, &mut self.input);
        self.support.restrict(
            backend,
            &self.input,
            &mut self.scratch,
            vector.as_mut_slice(),
        );
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GalerkinCertificate {
    pub residuals: Vec<f64>,
    pub absolute_residuals: Vec<f64>,
    pub b_orthogonality_defect: f64,
}
impl From<BlockCertification> for GalerkinCertificate {
    fn from(c: BlockCertification) -> Self {
        Self {
            residuals: c.residuals,
            absolute_residuals: c.absolute_residuals,
            b_orthogonality_defect: c.b_orthogonality_defect,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GalerkinReport {
    pub method: &'static str,
    pub application: GalerkinApplication,
    pub embedding_points: usize,
    pub coefficient_count: usize,
    pub estimated_payload_bytes: usize,
    pub iterations: usize,
    pub solver_converged: bool,
    pub analytic_kernel: bool,
    pub accepted: bool,
    pub rejection: Option<String>,
    pub projected: Option<GalerkinCertificate>,
    /// Discretization diagnostic only: does NOT certify the new finite pencil.
    pub full_rectangle: Option<GalerkinCertificate>,
}

#[derive(Debug, Clone)]
pub struct GalerkinResult {
    pub support_identity: SupportIdentity,
    pub eigenvalues: Vec<f64>,
    pub coefficients: Vec<Field2D>,
    pub fields: Vec<Field2D>,
    pub report: GalerkinReport,
}

fn finite(z: C) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

pub(super) fn admit(
    support: &FourierSupport,
    config: &EigensolverConfig,
    dense: bool,
    budget: GalerkinBudget,
) -> Result<usize, String> {
    if config.n_bands == 0
        || config.n_bands > support.len()
        || config.max_iter == 0
        || !config.tol.is_finite()
        || config.tol <= 0.0
        || config.tol >= 1.0
        || config.record_diagnostics
    {
        return Err("invalid Galerkin solve configuration; diagnostic history is outside this workspace model".into());
    }
    if dense && support.len() > budget.max_dense_dimension {
        return Err("Galerkin dense reference exceeds its dimension limit".into());
    }
    let bytes = estimated_payload_bytes(support, config, dense)?;
    if bytes > budget.max_estimated_payload_bytes {
        return Err(format!(
            "Galerkin payload estimate {bytes} exceeds budget {}",
            budget.max_estimated_payload_bytes
        ));
    }
    Ok(bytes)
}

fn finish<B: SpectralBackend>(
    operator: &mut GalerkinOperator<'_, B>,
    eigenvalues: Vec<f64>,
    coefficients: Vec<Field2D>,
    config: &EigensolverConfig,
    report: GalerkinReport,
) -> GalerkinResult {
    finish_impl(operator, eigenvalues, coefficients, config, report, true)
}

/// Certify a union of already refined sectors without rotating their labels.
pub(super) fn finish_preserving_labels<B: SpectralBackend>(
    operator: &mut GalerkinOperator<'_, B>,
    eigenvalues: Vec<f64>,
    coefficients: Vec<Field2D>,
    config: &EigensolverConfig,
    report: GalerkinReport,
) -> GalerkinResult {
    finish_impl(operator, eigenvalues, coefficients, config, report, false)
}

fn finish_impl<B: SpectralBackend>(
    operator: &mut GalerkinOperator<'_, B>,
    mut eigenvalues: Vec<f64>,
    mut coefficients: Vec<Field2D>,
    config: &EigensolverConfig,
    mut report: GalerkinReport,
    refine: bool,
) -> GalerkinResult {
    eigenvalues.truncate(config.n_bands);
    coefficients.truncate(config.n_bands);
    let finite_block = eigenvalues.len() == config.n_bands
        && coefficients.len() == config.n_bands
        && eigenvalues.iter().all(|x| x.is_finite())
        && coefficients
            .iter()
            .all(|v| v.as_slice().iter().all(|&z| finite(z)));
    if finite_block {
        let projected = if refine {
            rayleigh_ritz_certify(operator, &mut eigenvalues, &mut coefficients)
        } else {
            certify_without_refinement(operator, &eigenvalues, &coefficients)
        };
        report.projected = Some(projected.into());
    }
    let fields = operator.expand_fields(&coefficients);
    if finite_block {
        report.full_rectangle =
            Some(certify_without_refinement(operator.full, &eigenvalues, &fields).into());
    }
    let valid_certificate = report
        .projected
        .as_ref()
        .is_some_and(|c| certificate_passes(c, &eigenvalues, report.analytic_kernel));
    report.accepted = report.solver_converged && finite_block && valid_certificate;
    if !report.accepted {
        report.rejection = Some(
            if !report.solver_converged {
                "Galerkin eigensolver did not converge"
            } else {
                "Galerkin projected eigenblock failed certification"
            }
            .into(),
        );
    }
    GalerkinResult {
        support_identity: operator.support.identity(),
        eigenvalues,
        coefficients,
        fields,
        report,
    }
}

pub(super) fn certificate_passes(
    c: &GalerkinCertificate,
    eigenvalues: &[f64],
    analytic_kernel: bool,
) -> bool {
    !eigenvalues.is_empty()
        && c.b_orthogonality_defect.is_finite()
        && c.b_orthogonality_defect <= 1e-8
        && c.residuals.len() == eigenvalues.len()
        && c.absolute_residuals.len() == eigenvalues.len()
        && c.residuals
            .iter()
            .zip(&c.absolute_residuals)
            .enumerate()
            .all(|(j, (&r, &a))| {
                r.is_finite()
                    && a.is_finite()
                    && eigenvalues[j].is_finite()
                    && if analytic_kernel && j == 0 {
                        eigenvalues[j].abs() <= 1e-7 && a <= 1e-7
                    } else {
                        eigenvalues[j] >= 0.0 && r <= 1e-5
                    }
            })
}

pub(super) fn report<B: SpectralBackend>(
    operator: &GalerkinOperator<'_, B>,
    bytes: usize,
    method: &'static str,
) -> GalerkinReport {
    GalerkinReport {
        method,
        application: operator.application,
        embedding_points: operator.support.grid().len(),
        coefficient_count: operator.support.len(),
        estimated_payload_bytes: bytes,
        iterations: 0,
        solver_converged: false,
        analytic_kernel: operator.kernel_index.is_some(),
        accepted: false,
        rejection: None,
        projected: None,
        full_rectangle: None,
    }
}

/// LOBPCG on the new pencil. Nonconverged attempts retain fields, certificates,
/// iterations and an explicit rejection; they are never silently discarded.
/// Only f64 backend storage is admitted; generic dispatch never promotes f32.
pub fn solve_iterative<B: SpectralBackend>(
    full: &mut ThetaOperator<B>,
    support: &FourierSupport,
    config: &EigensolverConfig,
    warm_physical: Option<&[Field2D]>,
    preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    application: GalerkinApplication,
    budget: GalerkinBudget,
) -> Result<GalerkinResult, String> {
    crate::profile_scope!("symmetry.galerkin.iterative");
    let bytes = admit(support, config, false, budget)?;
    validate_operator(full, support)?;
    let mut operator = GalerkinOperator::new(
        full,
        support,
        application,
        budget.max_estimated_payload_bytes,
    )?;
    let warm = warm_physical
        .map(|v| operator.project_physical(v, config.effective_block_size().min(support.len())))
        .transpose()?;
    let mut preconditioner = preconditioner
        .map(|inner| GalerkinPreconditioner::new(inner, operator.full, support, application))
        .transpose()?;
    let mut solver = Eigensolver::new(
        &mut operator,
        config.clone(),
        preconditioner
            .as_mut()
            .map(|p| p as &mut dyn OperatorPreconditioner<B>),
        warm.as_deref(),
    );
    let result = solver.solve();
    let coefficients = solver.all_eigenvectors();
    drop(solver);
    let mut report = report(&operator, bytes, "lobpcg");
    report.iterations = result.iterations;
    report.solver_converged = result.converged;
    Ok(finish(
        &mut operator,
        result.eigenvalues,
        coefficients,
        config,
        report,
    ))
}

/// Assemble the complete coefficient pencil for a bounded independent dense
/// reference. Caller must budget two M² matrices before invoking this method.
pub fn dense_pencil<B: SpectralBackend>(
    operator: &mut GalerkinOperator<'_, B>,
    max_matrix_bytes: usize,
) -> Result<(Vec<C>, Vec<C>), String> {
    dense_operator_pencil(operator, max_matrix_bytes)
}

fn dense_operator_pencil<B: SpectralBackend, O: LinearOperator<B>>(
    operator: &mut O,
    max_matrix_bytes: usize,
) -> Result<(Vec<C>, Vec<C>), String> {
    let m = operator.subspace_dimension();
    if m == 0 || operator.grid().len() != m {
        return Err("dense coordinates must span the complete declared space".into());
    }
    let bytes = m
        .checked_mul(m)
        .and_then(|x| x.checked_mul(32))
        .ok_or("dense pencil size overflow")?;
    if bytes > max_matrix_bytes {
        return Err("dense pencil exceeds matrix budget".into());
    }
    let mut a = vec![C::ZERO; m * m];
    let mut b = a.clone();
    let mut x = operator.alloc_field();
    let mut y = operator.alloc_field();
    for j in 0..m {
        x.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
        x.as_mut_slice()[j] = cscalar::<B::Real>(1.0, 0.0);
        operator.apply(&x, &mut y);
        for (dst, &src) in a[j * m..(j + 1) * m].iter_mut().zip(y.as_slice()) {
            *dst = to_accum(src);
        }
        operator.apply_mass(&x, &mut y);
        for (dst, &src) in b[j * m..(j + 1) * m].iter_mut().zip(y.as_slice()) {
            *dst = to_accum(src);
        }
    }
    hermitize_checked(&mut a, m)?;
    hermitize_checked(&mut b, m)?;
    Ok((a, b))
}

fn hermitize_checked(a: &mut [C], n: usize) -> Result<(), String> {
    if a.iter().any(|&z| !finite(z)) {
        return Err("nonfinite Galerkin dense matrix".into());
    }
    let scale = a.iter().map(|z| z.norm()).fold(1.0, f64::max);
    for j in 0..n {
        for i in 0..=j {
            if (a[i + n * j] - a[j + n * i].conj()).norm() > 1e-10 * scale {
                return Err("Galerkin dense matrix is not Hermitian".into());
            }
            let z = (a[i + n * j] + a[j + n * i].conj()) * 0.5;
            a[i + n * j] = z;
            a[j + n * i] = z.conj();
        }
    }
    Ok(())
}

/// Complete generalized dense reference. Whitening retains every mass mode;
/// an unresolved/nonpositive mass eigenvalue rejects rather than dropping rank.
pub fn solve_dense<B: SpectralBackend>(
    full: &mut ThetaOperator<B>,
    support: &FourierSupport,
    config: &EigensolverConfig,
    application: GalerkinApplication,
    budget: GalerkinBudget,
) -> Result<GalerkinResult, String> {
    crate::profile_scope!("symmetry.galerkin.dense");
    let bytes = admit(support, config, true, budget)?;
    let mut operator = GalerkinOperator::new(
        full,
        support,
        application,
        budget.max_estimated_payload_bytes,
    )?;
    let (eigenvalues, coefficients) = dense_generalized(
        &mut operator,
        config.n_bands,
        budget.max_estimated_payload_bytes,
    )?;
    let mut report = report(&operator, bytes, "dense_generalized");
    report.solver_converged = true;
    Ok(finish(
        &mut operator,
        eigenvalues,
        coefficients,
        config,
        report,
    ))
}

/// Shared complete-coordinate reference. Callers admit the full dense
/// workspace model before invoking it; no mass rank is dropped.
pub(super) fn dense_generalized<B: SpectralBackend, O: LinearOperator<B>>(
    operator: &mut O,
    requested: usize,
    max_matrix_bytes: usize,
) -> Result<(Vec<f64>, Vec<Field2D>), String> {
    let m = operator.subspace_dimension();
    if requested == 0 || requested > m {
        return Err("invalid dense generalized band count".into());
    }
    let (a, b) = dense_operator_pencil(operator, max_matrix_bytes)?;
    let mass = operator.backend().dense_eigen(&b, m);
    let max_mass = mass.eigenvalues.iter().copied().fold(0.0, f64::max);
    if mass
        .eigenvalues
        .iter()
        .any(|x| !x.is_finite() || *x <= 64.0 * f64::EPSILON * m as f64 * max_mass)
    {
        return Err(
            "Galerkin dense mass is not numerically positive definite; no rank was dropped".into(),
        );
    }
    // W = Q diag(beta^-1/2); W† B W = I (W need not be Hermitian).
    let mut w = mass.eigenvectors;
    for (j, &beta) in mass.eigenvalues.iter().enumerate() {
        for z in &mut w[j * m..(j + 1) * m] {
            *z /= beta.sqrt();
        }
    }
    let mut aw = vec![C::ZERO; m * m];
    for j in 0..m {
        for k in 0..m {
            for i in 0..m {
                aw[i + m * j] += a[i + m * k] * w[k + m * j];
            }
        }
    }
    let mut h = vec![C::ZERO; m * m];
    for j in 0..m {
        for i in 0..m {
            h[i + m * j] = (0..m).map(|k| w[k + m * i].conj() * aw[k + m * j]).sum();
        }
    }
    hermitize_checked(&mut h, m)?;
    let eig = operator.backend().dense_eigen(&h, m);
    let mut coefficients = Vec::new();
    for j in 0..requested {
        let mut v = Field2D::zeros(operator.grid());
        for i in 0..m {
            v.as_mut_slice()[i] = (0..m)
                .map(|k| w[i + m * k] * eig.eigenvectors[k + m * j])
                .sum();
        }
        coefficients.push(v);
    }
    Ok((eig.eigenvalues[..requested].to_vec(), coefficients))
}

/// Fresh target-pencil certificate after an independently qualified transport.
pub(crate) fn certify_coefficients<B: SpectralBackend>(
    full: &mut ThetaOperator<B>, support: &FourierSupport, config: &EigensolverConfig,
    eigenvalues: Vec<f64>, coefficients: Vec<Field2D>, budget: GalerkinBudget, method: &'static str,
) -> Result<GalerkinResult, String> {
    let bytes = admit(support, config, false, budget)?;
    let mut op = GalerkinOperator::new(full, support, GalerkinApplication::DirectFourier, budget.max_estimated_payload_bytes)?;
    let mut r = report(&op, bytes, method);
    r.solver_converged = true;
    let result = finish_preserving_labels(&mut op, eigenvalues, coefficients, config, r);
    if !result.report.accepted { return Err(result.report.rejection.unwrap_or_else(|| "transported projected spectrum failed certification".into())); }
    Ok(result)
}

/// Prepare a physical initializer in the selected pencil's actual mass metric.
pub(crate) fn prepare_physical_seed<B: SpectralBackend>(
    full: &mut ThetaOperator<B>, support: &FourierSupport, fields: &[Field2D],
    columns: usize, tolerance: f64, payload_limit: usize,
) -> Result<super::seed::RankAwareSeed, String> {
    let mut op = GalerkinOperator::new(full, support, GalerkinApplication::DirectFourier, payload_limit)?;
    let coefficients = op.project_physical(fields, fields.len())?;
    let mut seed = super::seed::rank_aware_seed(&mut op, &coefficients, tolerance).map_err(|e| e.to_string())?;
    seed.fields.truncate(columns);
    seed.fields = op.expand_fields(&seed.fields);
    Ok(seed)
}
