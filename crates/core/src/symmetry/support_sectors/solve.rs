use num_complex::Complex64 as C;
use serde::Serialize;

use super::{CoefficientBasis, Sector, SupportSectorDescriptor, SupportSectorPlan};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{
        Eigensolver, EigensolverConfig,
        refine::{certify_without_refinement, rayleigh_ritz_certify},
    },
    field::{Field2D, cscalar, from_accum},
    grid::Grid2D,
    operators::{LinearOperator, ThetaOperator},
    preconditioners::OperatorPreconditioner,
    symmetry::{
        affine::AffineGroup,
        galerkin::{
            self, GalerkinApplication, GalerkinBudget, GalerkinCertificate, GalerkinOperator,
            GalerkinPreconditioner, GalerkinResult,
        },
        seed::rank_aware_seed,
        support::FourierSupport,
        support_qualification::{
            QualifiedSupportGroup, SupportQualificationOptions, SupportQualificationReport,
        },
    },
};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SupportSectorOptions {
    pub adaptive_prefixes: bool,
    pub reconstruct_partners: bool,
    /// None retains two guard columns per active sector. Some distributes the
    /// given total by prefix width; an explicit config.block_size is a floor.
    pub total_guard_columns: Option<usize>,
    /// Bounded complete spectra for validation, not a large production path.
    pub complete_dense: bool,
    pub max_allocation_rounds: usize,
    pub budget: GalerkinBudget,
    pub qualification: SupportQualificationOptions,
}
impl Default for SupportSectorOptions {
    fn default() -> Self {
        Self {
            adaptive_prefixes: true,
            reconstruct_partners: true,
            total_guard_columns: Some(8),
            complete_dense: false,
            max_allocation_rounds: 32,
            budget: GalerkinBudget::default(),
            qualification: SupportQualificationOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportSectorAttempt {
    pub sector: usize,
    pub round: usize,
    pub prefix: usize,
    pub working_column_limit: usize,
    pub dense_complete: bool,
    pub seed_rank: usize,
    pub iterations: usize,
    pub solver_converged: bool,
    pub accepted: bool,
    pub reconstructed_partners: usize,
    pub max_leakage: f64,
    pub projected: Option<GalerkinCertificate>,
    pub rejection: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportModeLabel {
    pub sector: usize,
    pub irrep: usize,
    /// None means a complete isotypic basis without an assigned row label.
    pub row: Option<usize>,
    pub local_index: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportSectorReport {
    pub qualification: SupportQualificationReport,
    pub descriptors: Vec<SupportSectorDescriptor>,
    pub attempts: Vec<SupportSectorAttempt>,
    pub finite_dimension: usize,
    pub sum_sector_dimensions: usize,
    pub kernel_owners: usize,
    pub allocation_rounds: usize,
    pub estimated_payload_bytes: usize,
    pub computed_with_multiplicity: usize,
    pub output_labels: Vec<SupportModeLabel>,
    pub output_boundary: &'static str,
    pub coverage: &'static str,
    pub accepted: bool,
    pub rejection: Option<String>,
}

pub struct SupportSectorResult {
    pub spectrum: GalerkinResult,
    pub report: SupportSectorReport,
}

struct ReducedOperator<'a, 'b, B: SpectralBackend> {
    full: &'a mut GalerkinOperator<'b, B>,
    sector: &'a Sector,
    input: B::Buffer,
    output: B::Buffer,
}
impl<B: SpectralBackend> LinearOperator<B> for ReducedOperator<'_, '_, B> {
    fn apply(&mut self, x: &B::Buffer, y: &mut B::Buffer) {
        self.sector
            .basis
            .expand(x.as_slice(), self.input.as_mut_slice());
        self.full.apply(&self.input, &mut self.output);
        self.sector
            .basis
            .restrict(self.output.as_slice(), y.as_mut_slice());
    }
    fn apply_mass(&mut self, x: &B::Buffer, y: &mut B::Buffer) {
        self.sector
            .basis
            .expand(x.as_slice(), self.input.as_mut_slice());
        self.full.apply_mass(&self.input, &mut self.output);
        self.sector
            .basis
            .restrict(self.output.as_slice(), y.as_mut_slice());
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
        self.sector.basis.coordinate_grid()
    }
    fn bloch(&self) -> [f64; 2] {
        self.full.bloch()
    }
    fn subspace_dimension(&self) -> usize {
        self.sector.basis.rank()
    }
    fn owns_gamma_kernel(&self) -> bool {
        self.sector.kernel.is_some()
    }
    fn analytic_kernel_available(&self) -> bool {
        self.sector.kernel.is_some()
    }
    fn fill_gamma_kernel(&self, vector: &mut B::Buffer) {
        vector.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
        if let Some(kernel) = &self.sector.kernel {
            for (dst, &src) in vector.as_mut_slice().iter_mut().zip(kernel) {
                *dst = from_accum(src);
            }
        }
    }
    fn seed_initial_vector(&mut self, vector: &mut B::Buffer, phase: f64) {
        self.full.seed_initial_vector(&mut self.input, phase);
        self.sector
            .basis
            .restrict(self.input.as_slice(), vector.as_mut_slice());
    }
}

struct ReducedPreconditioner<'a, B: SpectralBackend> {
    inner: &'a mut dyn OperatorPreconditioner<B>,
    basis: &'a CoefficientBasis,
    scratch: B::Buffer,
}
impl<B: SpectralBackend> OperatorPreconditioner<B> for ReducedPreconditioner<'_, B> {
    fn apply(&mut self, backend: &B, vector: &mut B::Buffer) {
        self.basis
            .expand(vector.as_slice(), self.scratch.as_mut_slice());
        self.inner.apply(backend, &mut self.scratch);
        self.basis
            .restrict(self.scratch.as_slice(), vector.as_mut_slice());
    }
}

struct Slot {
    values: Vec<f64>,
    coordinates: Vec<Field2D>,
    report: SupportSectorAttempt,
}

fn validate_config(
    config: &EigensolverConfig,
    support: &FourierSupport,
    options: SupportSectorOptions,
) -> Result<(), String> {
    if config.n_bands == 0
        || config.n_bands > support.len()
        || config.max_iter == 0
        || !config.tol.is_finite()
        || config.tol <= 0.0
        || config.tol >= 1.0
        || config.record_diagnostics
        || options.max_allocation_rounds == 0
    {
        return Err("invalid support sector configuration; diagnostic history is outside this workspace model".into());
    }
    Ok(())
}

/// Includes qualification, orbit construction, every attempted prefix and
/// partner certificate. Choosing a smaller Fourier support is a separate step.
#[allow(clippy::too_many_arguments)]
pub fn solve<B: SpectralBackend>(
    full: &mut ThetaOperator<B>,
    support: &FourierSupport,
    group: AffineGroup,
    config: &EigensolverConfig,
    warm_physical: Option<&[Field2D]>,
    preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    options: SupportSectorOptions,
) -> Result<SupportSectorResult, String> {
    crate::profile_scope!("symmetry.support.sectors");
    validate_config(config, support, options)?;
    let mut qualification = options.qualification;
    qualification.max_estimated_payload_bytes = qualification
        .max_estimated_payload_bytes
        .min(options.budget.max_estimated_payload_bytes);
    let qualified = QualifiedSupportGroup::qualify(full, support, group, qualification)
        .map_err(|r| format!("Galerkin sector qualification rejected: {:?}", r))?;
    let plan = SupportSectorPlan::new(
        qualified,
        full,
        support,
        options.reconstruct_partners,
        options.budget.max_estimated_payload_bytes,
    )?;
    solve_with_plan(
        full,
        support,
        &plan,
        config,
        warm_physical,
        preconditioner,
        options,
    )
}

/// Reusing a plan is permitted only for an exactly identical support/material/
/// fiber. Nonconverged solves and failed growth admissions return their work
/// ledger and rejected partial spectrum, rather than discarding the attempts.
#[allow(clippy::too_many_arguments)]
pub fn solve_with_plan<B: SpectralBackend>(
    full: &mut ThetaOperator<B>,
    support: &FourierSupport,
    plan: &SupportSectorPlan,
    config: &EigensolverConfig,
    warm_physical: Option<&[Field2D]>,
    preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    options: SupportSectorOptions,
) -> Result<SupportSectorResult, String> {
    validate_config(config, support, options)?;
    if options.reconstruct_partners != plan.reconstruct_partners {
        return Err("partner policy differs from the retained sector plan".into());
    }
    let active = plan.sectors.iter().filter(|s| s.basis.rank() > 0).count();
    let initial = if options.adaptive_prefixes {
        config.n_bands.div_ceil(active) + 2
    } else {
        config.n_bands
    };
    let mut prefixes: Vec<_> = plan
        .sectors
        .iter()
        .map(|s| {
            if options.complete_dense {
                s.basis.rank()
            } else {
                initial
                    .div_ceil(s.descriptor.partner_rows)
                    .min(s.basis.rank())
            }
        })
        .collect();
    let mut guards = allocate_guards(&prefixes, options.total_guard_columns, support.len());
    let warm_width = warm_physical.map_or(0, |v| {
        v.len()
            .min(config.effective_block_size())
            .min(support.len())
    });
    let mut peak_estimate = admit(
        plan, support, config, &prefixes, &guards, warm_width, options,
    )?;
    plan.validate_for(full, support)?;
    let mut preconditioner = preconditioner
        .map(|p| GalerkinPreconditioner::new(p, full, support, GalerkinApplication::DirectFourier))
        .transpose()?;
    let mut operator = GalerkinOperator::new(
        full,
        support,
        GalerkinApplication::DirectFourier,
        options.budget.max_estimated_payload_bytes,
    )?;
    let warm = warm_physical
        .map(|v| operator.project_physical(v, config.effective_block_size().min(support.len())))
        .transpose()?;
    let mut slots: Vec<Option<Slot>> = (0..plan.sectors.len()).map(|_| None).collect();
    let mut attempts = Vec::new();
    let mut failure = None;
    let mut rounds = 0;
    loop {
        rounds += 1;
        for (i, sector) in plan.sectors.iter().enumerate() {
            if sector.basis.rank() == 0
                || slots[i]
                    .as_ref()
                    .is_some_and(|s| s.report.prefix == prefixes[i])
            {
                continue;
            }
            let slot = solve_prefix(
                &mut operator,
                plan,
                i,
                config,
                prefixes[i],
                guards[i],
                rounds,
                slots[i].as_ref().map(|s| s.coordinates.as_slice()),
                warm.as_deref(),
                preconditioner
                    .as_mut()
                    .map(|p| p as &mut dyn OperatorPreconditioner<B>),
                options,
            );
            let accepted = slot.report.accepted;
            if !accepted {
                failure = slot.report.rejection.clone();
            }
            attempts.push(slot.report.clone());
            slots[i] = Some(slot);
            if !accepted {
                break;
            }
        }
        if failure.is_some() {
            break;
        }
        let mut values: Vec<_> = slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
            .flat_map(|(i, s)| {
                s.values.iter().flat_map(move |&x| {
                    std::iter::repeat_n(x, plan.sectors[i].descriptor.partner_rows)
                })
            })
            .collect();
        values.sort_by(f64::total_cmp);
        let cutoff = values
            .get(config.n_bands - 1)
            .copied()
            .unwrap_or(f64::INFINITY);
        let mut grow = false;
        if options.adaptive_prefixes && !options.complete_dense {
            for (i, slot) in slots.iter().enumerate() {
                if let Some(slot) = slot {
                    let rank = plan.sectors[i].basis.rank();
                    if prefixes[i] < rank
                        && slot
                            .values
                            .last()
                            .is_some_and(|&v| v <= cutoff + 1e-7 * cutoff.abs().max(1.0))
                    {
                        prefixes[i] = prefixes[i]
                            .saturating_add(prefixes[i].div_ceil(2).max(2))
                            .min(rank);
                        grow = true;
                    }
                }
            }
        }
        if !grow {
            break;
        }
        if rounds >= options.max_allocation_rounds {
            failure = Some("support sector allocation round limit reached".into());
            break;
        }
        guards = allocate_guards(&prefixes, options.total_guard_columns, support.len());
        match admit(
            plan, support, config, &prefixes, &guards, warm_width, options,
        ) {
            Ok(bytes) => peak_estimate = peak_estimate.max(bytes),
            Err(reason) => {
                failure = Some(reason);
                break;
            }
        }
    }
    let iterations = attempts.iter().map(|a| a.iterations).sum();
    let mut ordered = Vec::new();
    for (i, slot) in slots.iter().enumerate() {
        if let Some(slot) = slot {
            for (j, &value) in slot.values.iter().take(slot.coordinates.len()).enumerate() {
                for row in 0..plan.sectors[i].descriptor.partner_rows {
                    ordered.push((value, i, j, row));
                }
            }
        }
    }
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
    let computed = ordered.len();
    let boundary = if computed == support.len() && config.n_bands == computed {
        "complete finite spectrum"
    } else if computed > config.n_bands {
        if (ordered[config.n_bands].0 - ordered[config.n_bands - 1].0).abs()
            <= 1e-7 * ordered[config.n_bands].0.abs().max(1.0)
        {
            "requested output cuts a computed cluster"
        } else {
            "separated in computed prefixes"
        }
    } else {
        "boundary not certified"
    };
    ordered.truncate(config.n_bands);
    if ordered.len() != config.n_bands && failure.is_none() {
        failure = Some("insufficient modes in support sector output".into());
    }
    let mut labels = Vec::new();
    let mut eigenvalues = Vec::new();
    let mut coefficients = Vec::new();
    let mut input = vec![C::ZERO; support.len()];
    let mut scratch = input.clone();
    for (value, s, j, row) in ordered {
        let slot = slots[s].as_ref().unwrap();
        let sector = &plan.sectors[s];
        sector
            .basis
            .expand(slot.coordinates[j].as_slice(), &mut input);
        let mut field = Field2D::zeros(support.coordinate_grid());
        plan.reconstruct_partner(s, row, &input, field.as_mut_slice(), &mut scratch)?;
        eigenvalues.push(value);
        coefficients.push(field);
        labels.push(SupportModeLabel {
            sector: s,
            irrep: sector.descriptor.irrep,
            local_index: j,
            row: if !plan.reconstruct_partners
                && plan.group.representation().irreps[sector.descriptor.irrep].dimension > 1
            {
                None
            } else {
                Some(row)
            },
        });
    }
    let all_dense = slots.iter().flatten().all(|s| s.report.dense_complete);
    let mut basic = galerkin::report(&operator, peak_estimate, "symmetry_sectors");
    basic.iterations = iterations;
    basic.solver_converged = failure.is_none();
    let mut spectrum =
        galerkin::finish_preserving_labels(&mut operator, eigenvalues, coefficients, config, basic);
    if !spectrum.report.accepted && failure.is_none() {
        failure = spectrum.report.rejection.clone();
    }
    let accepted = failure.is_none() && spectrum.report.accepted;
    if let Some(reason) = &failure {
        spectrum.report.rejection = Some(reason.clone());
    }
    let report = SupportSectorReport {
        qualification: plan.group.report().clone(),
        descriptors: plan.descriptors().cloned().collect(),
        attempts,
        finite_dimension: support.len(),
        sum_sector_dimensions: plan.descriptors().map(|s| s.rank * s.partner_rows).sum(),
        kernel_owners: plan
            .descriptors()
            .filter(|s| s.owns_analytic_kernel)
            .count(),
        allocation_rounds: rounds,
        estimated_payload_bytes: peak_estimate,
        computed_with_multiplicity: computed,
        output_labels: labels,
        output_boundary: boundary,
        accepted,
        rejection: failure,
        coverage: if !accepted {
            "rejected; no spectrum coverage claim"
        } else if all_dense {
            "complete dense sector diagonalizations with sufficient retained prefixes"
        } else if options.adaptive_prefixes {
            "adaptive lowest prefixes with separated converged tails; empirical coverage, not inertia"
        } else {
            "conservative lowest sector prefixes; empirical iterative coverage, not inertia"
        },
    };
    Ok(SupportSectorResult { spectrum, report })
}

fn working_columns(
    config: &EigensolverConfig,
    sector: &Sector,
    prefix: usize,
    guard: usize,
) -> usize {
    config
        .block_size
        .max(prefix.saturating_add(guard))
        .min(sector.basis.rank() - usize::from(sector.kernel.is_some()))
}

fn admit(
    plan: &SupportSectorPlan,
    support: &FourierSupport,
    config: &EigensolverConfig,
    prefixes: &[usize],
    guards: &[usize],
    warm_width: usize,
    options: SupportSectorOptions,
) -> Result<usize, String> {
    let (n, m, q) = (
        support.grid().len() as u128,
        support.len() as u128,
        config.n_bands as u128,
    );
    let mut scalar = (6 * q + 16) * n + (6 * q + 12 + warm_width as u128) * m;
    let mut scratch = 0;
    for (i, s) in plan.sectors.iter().enumerate() {
        let (rank, p) = (s.basis.rank() as u128, prefixes[i] as u128);
        if rank == 0 {
            continue;
        }
        scalar += 2 * rank * p; // Retained old and new coordinates during growth.
        let dense = options.complete_dense || s.basis.rank() <= prefixes[i].saturating_add(2);
        if dense && s.basis.rank() > options.budget.max_dense_dimension {
            return Err("complete coefficient sector exceeds the dense dimension limit".into());
        }
        let b = working_columns(config, s, prefixes[i], guards[i]) as u128;
        let partners = s.descriptor.partner_rows as u128;
        let solver = if dense {
            24 * rank * rank
        } else {
            64 * rank * b + 128 * b * b
        };
        // Seed rank extraction forms a Gram matrix at the original supplied
        // width, which can exceed the retained target-sector rank.
        let seed_width = if dense { 0 } else { p.max(warm_width as u128) };
        scratch = scratch.max(
            solver
                + 32 * rank * seed_width
                + 32 * seed_width * seed_width
                + 32 * m * p * partners
                + 16 * rank * p
                + 8 * (p * partners) * (p * partners),
        );
    }
    let bytes =
        usize::try_from(plan.estimated_plan_payload_bytes as u128 + 16 * (scalar + scratch))
            .map_err(|_| "coefficient sector workspace estimate overflow")?;
    if bytes > options.budget.max_estimated_payload_bytes {
        return Err(format!(
            "coefficient sector payload estimate {bytes} exceeds budget {}",
            options.budget.max_estimated_payload_bytes
        ));
    }
    Ok(bytes)
}

fn allocate_guards(prefixes: &[usize], budget: Option<usize>, dimension: usize) -> Vec<usize> {
    let mut result: Vec<_> = prefixes
        .iter()
        .map(|&p| if p > 0 { 2 } else { 0 })
        .collect();
    if let Some(budget) = budget {
        result.fill(0);
        let budget = budget.min(dimension);
        let total: usize = prefixes.iter().sum();
        let mut remainders = Vec::new();
        let mut used = 0;
        for (i, &p) in prefixes.iter().enumerate() {
            if p == 0 {
                continue;
            }
            let scaled = (budget as u128) * (p as u128);
            result[i] = (scaled / total as u128) as usize;
            remainders.push(((scaled % total as u128) as usize, i));
            used += result[i];
        }
        remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, i) in remainders.iter().take(budget - used) {
            result[i] += 1;
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn solve_prefix<B: SpectralBackend>(
    operator: &mut GalerkinOperator<'_, B>,
    plan: &SupportSectorPlan,
    index: usize,
    config: &EigensolverConfig,
    prefix: usize,
    guard: usize,
    round: usize,
    previous: Option<&[Field2D]>,
    warm_coefficients: Option<&[Field2D]>,
    preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    options: SupportSectorOptions,
) -> Slot {
    crate::profile_scope!("symmetry.support.prefix");
    let sector = &plan.sectors[index];
    let dense = options.complete_dense || sector.basis.rank() <= prefix.saturating_add(2);
    let mut report = SupportSectorAttempt {
        sector: index,
        round,
        prefix,
        working_column_limit: if dense {
            sector.basis.rank()
        } else {
            working_columns(config, sector, prefix, guard)
        },
        dense_complete: dense,
        seed_rank: 0,
        iterations: 0,
        solver_converged: false,
        accepted: false,
        reconstructed_partners: 0,
        max_leakage: 0.0,
        projected: None,
        rejection: None,
    };
    let mut values = Vec::new();
    let mut coordinates = Vec::new();
    let attempt = (|| -> Result<(), String> {
        let input = operator.alloc_field();
        let output = operator.alloc_field();
        let mut preconditioner = preconditioner.map(|inner| ReducedPreconditioner {
            inner,
            basis: &sector.basis,
            scratch: operator.alloc_field(),
        });
        let mut reduced = ReducedOperator {
            full: operator,
            sector,
            input,
            output,
        };
        if dense {
            (values, coordinates) = galerkin::dense_generalized(
                &mut reduced,
                sector.basis.rank(),
                options.budget.max_estimated_payload_bytes,
            )?;
            report.solver_converged = true;
        } else {
            let projected = if let Some(previous) = previous {
                Some(previous.to_vec())
            } else {
                warm_coefficients.map(|fields| {
                    fields
                        .iter()
                        .map(|f| {
                            let mut result = Field2D::zeros(sector.basis.coordinate_grid());
                            sector.basis.restrict(f.as_slice(), result.as_mut_slice());
                            result
                        })
                        .collect::<Vec<_>>()
                })
            };
            let warm = projected
                .as_ref()
                .filter(|fields| !fields.is_empty())
                .map(|fields| rank_aware_seed(&mut reduced, fields, 1e-10))
                .transpose()
                .map_err(|e| e.to_string())?;
            report.seed_rank = warm.as_ref().map_or(0, |w| w.fields.len());
            let mut cfg = config.clone();
            cfg.n_bands = prefix;
            cfg.block_size = config.block_size.max(prefix.saturating_add(guard));
            let mut solver = Eigensolver::new(
                &mut reduced,
                cfg,
                preconditioner
                    .as_mut()
                    .map(|p| p as &mut dyn OperatorPreconditioner<B>),
                warm.as_ref().map(|w| w.fields.as_slice()),
            );
            let result = solver.solve();
            coordinates = solver.all_eigenvectors();
            values = result.eigenvalues;
            report.iterations = result.iterations;
            report.solver_converged = result.converged;
        }
        values.truncate(prefix);
        coordinates.truncate(prefix);
        if values.len() != prefix
            || coordinates.len() != prefix
            || values.iter().any(|v| !v.is_finite())
            || coordinates.iter().any(|v| {
                v.as_slice()
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
            })
        {
            return Err("sector attempt returned an incomplete or nonfinite block".into());
        }
        rayleigh_ritz_certify(&mut reduced, &mut values, &mut coordinates);
        drop(reduced);
        // Certify all computed partner rows together on the unreduced D04
        // operator, including cross-row B orthogonality and row leakage.
        let m = operator.support().len();
        let mut source = vec![C::ZERO; m];
        let mut scratch = source.clone();
        let mut projected = source.clone();
        let mut all = Vec::new();
        let mut all_values = Vec::new();
        for row in 0..sector.descriptor.partner_rows {
            let weights = if row == 0 {
                sector.weights.clone()
            } else {
                plan.group
                    .representation()
                    .matrix_unit_weights(sector.descriptor.irrep, row, row)
            };
            for (&value, field) in values.iter().zip(&coordinates) {
                sector.basis.expand(field.as_slice(), &mut source);
                let mut target = Field2D::zeros(operator.support().coordinate_grid());
                plan.reconstruct_partner(index, row, &source, target.as_mut_slice(), &mut scratch)?;
                plan.apply_weights(&weights, target.as_slice(), &mut projected, &mut scratch);
                let error = projected
                    .iter()
                    .zip(target.as_slice())
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>();
                let norm = target.as_slice().iter().map(C::norm_sqr).sum::<f64>();
                let leakage = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
                if !leakage.is_finite() {
                    return Err("nonfinite partner leakage".into());
                }
                report.max_leakage = report.max_leakage.max(leakage);
                report.reconstructed_partners += usize::from(row > 0);
                all_values.push(value);
                all.push(target);
            }
        }
        let certificate: GalerkinCertificate =
            certify_without_refinement(operator, &all_values, &all).into();
        let valid =
            galerkin::certificate_passes(&certificate, &all_values, sector.kernel.is_some());
        report.projected = Some(certificate);
        if !report.solver_converged {
            return Err("coefficient sector eigensolver did not converge".into());
        }
        if !valid || report.max_leakage > 1e-8 {
            return Err("coefficient sector or partner failed fresh certification".into());
        }
        Ok(())
    })();
    report.accepted = attempt.is_ok();
    if let Err(reason) = attempt {
        report.rejection = Some(reason);
    }
    Slot {
        values,
        coordinates,
        report,
    }
}
