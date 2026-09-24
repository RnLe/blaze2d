//! Checked band workflow for an explicitly different Fourier discretization.
//!
//! Qualification failure releases the sector constraint on the same support.
//! It never changes the material, cutoff, precision or full FFT embedding.
use serde::Serialize;

use super::bandstructure::{
    BandStructureJob, BandStructureResult, KPointResult, RunOptions, compute_k_path_distances,
};
use crate::{
    backend::SpectralBackend,
    band_tracking::track_bands_with_frequencies,
    diagnostics::PreconditionerType,
    dielectric::Dielectric2D,
    eigensolver::{SubspaceHistory, iteration_total},
    field::Field2D,
    operators::ThetaOperator,
    polarization::Polarization,
    preconditioners::OperatorPreconditioner,
    symmetry::{
        fourier::Arithmetic,
        galerkin::{self, GalerkinApplication, GalerkinBudget, GalerkinReport},
        reduced::{ReducedKernel, SectorStorage},
        reuse::{KReuse, ReuseReport, SolveAttempt},
        sectors::SectorPolicy,
        seed::SeedTransfer,
        support::{FourierSupport, SupportBudget, SupportIdentity},
        support_qualification::{QualifiedSupportGroup, SupportQualificationReport},
        support_sectors::{
            self, SupportModeLabel, SupportSectorOptions, SupportSectorPlan, SupportSectorReport,
        },
        traversal::KTraversal,
    },
};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct MetricBallOptions {
    /// Cartesian angular wavevector cutoff, in inverse reference length.
    pub angular_cutoff: f64,
    pub support_budget: SupportBudget,
    /// Numerical payload allowance for driver state plus the selected solve.
    /// FFT plans, allocator overhead and library scratch are not RSS-bounded.
    pub budget: GalerkinBudget,
    pub max_allocation_rounds: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportPointReport {
    pub angular_cutoff: f64,
    /// JSON bit patterns use hexadecimal strings so JavaScript cannot round
    /// 64-bit identities while copying a result through the browser boundary.
    #[serde(serialize_with = "serialize_support_identity")]
    pub support: Option<SupportIdentity>,
    pub driver_payload_bytes: usize,
    pub solver_payload_limit: usize,
    pub qualification: Option<SupportQualificationReport>,
    /// The sector attempt retains its own ascending-eigenvalue output order,
    /// even when rejected and followed by an unrestricted projected solve.
    pub sector_attempt: Option<SupportSectorReport>,
    /// Certificates in the returned (possibly tracked) band order.
    pub selected: Option<GalerkinReport>,
    /// Present only when the returned spectrum comes from the sector attempt.
    pub returned_labels: Option<Vec<SupportModeLabel>>,
    /// Returned index -> index in the selected ascending-eigenvalue spectrum.
    pub returned_permutation: Vec<usize>,
}

fn serialize_support_identity<S: serde::Serializer>(
    identity: &Option<SupportIdentity>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct PortableIdentity<'a> {
        version: &'a str,
        bit_encoding: &'static str,
        grid: [usize; 2],
        grid_length_bits: [String; 2],
        lattice_bits: [[String; 2]; 2],
        k_bits: [String; 2],
        modes: &'a [[i64; 2]],
    }
    identity
        .as_ref()
        .map(|id| PortableIdentity {
            version: id.version,
            bit_encoding: "hexadecimal_u64",
            grid: id.grid,
            grid_length_bits: id.grid_length_bits.map(|b| format!("{b:016x}")),
            lattice_bits: id.lattice_bits.map(|row| row.map(|b| format!("{b:016x}"))),
            k_bits: id.k_bits.map(|b| format!("{b:016x}")),
            modes: &id.modes,
        })
        .serialize(serializer)
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{reason}")]
pub struct SupportBandFailure {
    pub k_index: Option<usize>,
    pub reason: String,
    /// Contains only completed, rectangular points, including a rejected final
    /// point if it has the requested width. No uncomputed rows are fabricated.
    pub partial_result: BandStructureResult,
    pub symmetry: ReuseReport,
    pub discretization: Option<SupportPointReport>,
}

/// Validate policies independently of geometry and backend allocation. More
/// combinations can be added only with their own support identity/qualification.
pub fn validate_options(options: &RunOptions, selected: MetricBallOptions) -> Result<(), String> {
    let s = &options.symmetry;
    if s.max_sector_workspace_bytes != crate::symmetry::sectors::memory::DEFAULT_LIMIT_BYTES {
        return Err("metric support uses its discretization payload budget; the rectangular-sector workspace budget is not applicable".into());
    }
    if !selected.angular_cutoff.is_finite()
        || selected.angular_cutoff <= 0.0
        || selected.max_allocation_rounds == 0
        || selected.budget.max_estimated_payload_bytes == 0
    {
        return Err(
            "metric support requires a positive finite angular cutoff and nonzero budgets".into(),
        );
    }
    if s.k_reuse != KReuse::Off || s.k_traversal != KTraversal::Given || options.reuse_gamma {
        return Err("metric support currently requires supplied-order solves with k and legacy Gamma reuse disabled".into());
    }
    if s.arithmetic == Arithmetic::PtRealTe || s.pt_operation.is_some() {
        return Err(
            "PT packing on metric support is not implemented; select complex arithmetic explicitly"
                .into(),
        );
    }
    if s.seed_transfer != SeedTransfer::Copy {
        return Err("metric support currently accepts physical copy/prediction seeds; sector projection performs its own target-metric rank extraction".into());
    }
    if !matches!(
        s.sectors,
        SectorPolicy::Off | SectorPolicy::ConservativeGroup | SectorPolicy::AdaptiveGroup
    ) {
        return Err(
            "metric support requires an explicit group sector policy or sectors off".into(),
        );
    }
    if s.sectors != SectorPolicy::Off
        && (s.plane_group.is_none()
            || s.sector_storage != SectorStorage::Orbit
            || s.arithmetic != Arithmetic::FourierComplex)
    {
        return Err("metric group sectors require a plane group, orbit storage and complex Fourier application".into());
    }
    if s.sectors == SectorPolicy::Off
        && (s.sector_storage != SectorStorage::FullGrid
            || s.reconstruct_partners
            || s.sector_guard_budget.is_some())
    {
        return Err(
            "metric sector storage, guards and partners require an enabled group sector policy"
                .into(),
        );
    }
    if s.sector_kernel != ReducedKernel::FullGrid
        || s.share_sector_workspace
        || s.correction_projection_interval != 1
    {
        return Err("metric support uses its own serial coefficient workspace and full embedding FFT kernel".into());
    }
    if options.precond_type == PreconditionerType::Multigrid {
        return Err("multigrid preconditioning is not implemented".into());
    }
    Ok(())
}

/// Caller-owned continuation/prediction fields plus material/operator tables.
/// The solve budget separately includes temporary vectors, output fields and
/// certificates. Caller-retained streamed output belongs in its own preflight.
pub fn driver_payload_bytes(
    points: usize,
    bands: usize,
    path_points: usize,
) -> Result<usize, String> {
    let bytes = (points as u128)
        .checked_mul(16)
        .and_then(|n| n.checked_mul(32 + 6 * bands as u128))
        .and_then(|n| {
            (path_points as u128)
                .checked_mul(24 + 8 * bands as u128)?
                .checked_add(n)
        })
        .ok_or("metric band driver payload overflow")?;
    usize::try_from(bytes).map_err(|_| "metric band driver payload overflow".into())
}

/// Returns structured partial results on admission or certification failure.
/// Every attempted eigensolver iteration is included in the streamed ledger.
pub fn run_with_k_streaming<B: SpectralBackend + Clone>(
    backend: B,
    job: &BandStructureJob,
    options: RunOptions,
    selected: MetricBallOptions,
    mut on_k_point: impl FnMut(KPointResult),
) -> Result<BandStructureResult, SupportBandFailure> {
    crate::profile_scope!("run.support_bands");
    let mut result = BandStructureResult {
        k_path: Vec::new(),
        distances: Vec::new(),
        bands: Vec::new(),
    };
    let fail = |reason: String,
                partial_result: &BandStructureResult,
                k_index,
                symmetry: ReuseReport,
                discretization| {
        SupportBandFailure {
            reason,
            partial_result: partial_result.clone(),
            k_index,
            symmetry,
            discretization,
        }
    };
    let preflight = (|| -> Result<usize, String> {
        validate_options(&options, selected)?;
        if std::any::TypeId::of::<B::Real>() != std::any::TypeId::of::<f64>() {
            return Err("metric band solves require an f64 backend".into());
        }
        if job.eigensolver.record_diagnostics || job.k_path.is_empty()
            || job.eigensolver.n_bands == 0 || job.eigensolver.max_iter == 0
            || !job.eigensolver.tol.is_finite() || job.eigensolver.tol <= 0.0 || job.eigensolver.tol >= 1.0
        {
            return Err("metric bands require a nonempty path and solve; per-iteration diagnostic history is unsupported".into());
        }
        let points = job.grid.nx.checked_mul(job.grid.ny).ok_or("embedding size overflow")?;
        let bytes = driver_payload_bytes(points, job.eigensolver.n_bands, job.k_path.len())?;
        if bytes >= selected.budget.max_estimated_payload_bytes {
            return Err("metric band driver state exceeds its numerical payload allowance".into());
        }
        Ok(bytes)
    })().map_err(|reason| fail(reason, &result, None, ReuseReport::default(), None))?;
    let budget = GalerkinBudget {
        max_estimated_payload_bytes: selected.budget.max_estimated_payload_bytes - preflight,
        ..selected.budget
    };
    let mut first_support = Some(
        FourierSupport::metric_ball(
            job.grid,
            job.geom.lattice,
            job.k_path[0],
            selected.angular_cutoff,
            selected.support_budget,
        )
        .map_err(|reason| fail(reason, &result, Some(0), ReuseReport::default(), None))?,
    );
    if first_support.as_ref().unwrap().len() < job.eigensolver.n_bands {
        return Err(fail(
            "requested bands exceed the selected Fourier support".into(),
            &result,
            Some(0),
            ReuseReport::default(),
            None,
        ));
    }
    let reciprocal = job.geom.lattice.reciprocal();
    let dielectric = {
        crate::profile_scope!("run.setup");
        Dielectric2D::from_geometry(&job.geom, job.grid, &job.dielectric)
    };
    let eps = (job.pol == Polarization::TM).then(|| dielectric.eps().to_vec());
    let distances = compute_k_path_distances(&job.k_path);
    let mut history = options.use_subspace_prediction.then(|| {
        let mut h = SubspaceHistory::new(job.eigensolver.n_bands);
        if options.use_extrapolation {
            h.enable_extrapolation();
        }
        h
    });
    let mut previous: Option<Vec<Field2D>> = None;
    let mut previous_omegas: Option<Vec<f64>> = None;
    let mut previous_values: Option<Vec<f64>> = None;
    for (index, &k) in job.k_path.iter().enumerate() {
        let mut ledger = ReuseReport::default();
        let mut report = SupportPointReport {
            angular_cutoff: selected.angular_cutoff,
            support: None,
            driver_payload_bytes: preflight,
            solver_payload_limit: budget.max_estimated_payload_bytes,
            qualification: None,
            sector_attempt: None,
            selected: None,
            returned_labels: None,
            returned_permutation: Vec::new(),
        };
        let support = if let Some(support) = first_support.take() {
            support
        } else {
            FourierSupport::metric_ball(
                job.grid,
                job.geom.lattice,
                k,
                selected.angular_cutoff,
                selected.support_budget,
            )
            .map_err(|reason| {
                fail(
                    reason,
                    &result,
                    Some(index),
                    ledger.clone(),
                    Some(report.clone()),
                )
            })?
        };
        report.support = Some(support.identity());
        let prediction = history.as_mut().and_then(|h| {
            let p = h.predict(k, eps.as_deref());
            p.has_prediction().then_some(p.predicted_vectors)
        });
        let warm = if history.is_some() {
            prediction.as_deref()
        } else {
            previous.as_deref()
        };
        let mut theta = ThetaOperator::new(
            backend.clone(),
            dielectric.clone(),
            job.pol,
            reciprocal.fractional_to_cartesian(k),
        );
        let mut preconditioner =
            build_preconditioner(&theta, &options, job.pol, previous_values.as_deref());
        let start = iteration_total();
        let mut spectrum = None;
        if options.symmetry.sectors != SectorPolicy::Off {
            let sector_start = iteration_total();
            let sector = (|| -> Result<_, String> {
                let group = options
                    .symmetry
                    .plane_group
                    .unwrap()
                    .group()
                    .little_group(k, 1e-12)
                    .map_err(|e| e.to_string())?;
                ledger.candidate_operation_count = group.order();
                if group.order() <= 1 {
                    return Err("metric support little group is trivial at this point".into());
                }
                let mut settings = SupportSectorOptions {
                    adaptive_prefixes: options.symmetry.sectors == SectorPolicy::AdaptiveGroup,
                    reconstruct_partners: options.symmetry.reconstruct_partners,
                    total_guard_columns: options.symmetry.sector_guard_budget,
                    max_allocation_rounds: selected.max_allocation_rounds,
                    budget,
                    ..Default::default()
                };
                settings.qualification.tolerance = options.symmetry.qualification_tolerance;
                settings.qualification.max_estimated_payload_bytes = settings
                    .qualification
                    .max_estimated_payload_bytes
                    .min(budget.max_estimated_payload_bytes);
                ledger.qualification_attempts += 1;
                let qualified = match QualifiedSupportGroup::qualify(
                    &mut theta,
                    &support,
                    group,
                    settings.qualification,
                ) {
                    Ok(q) => {
                        report.qualification = Some(q.report().clone());
                        q
                    }
                    Err(q) => {
                        let reason = q
                            .rejection
                            .clone()
                            .unwrap_or_else(|| "support qualification rejected".into());
                        report.qualification = Some(q);
                        return Err(reason);
                    }
                };
                let plan = SupportSectorPlan::new(
                    qualified,
                    &theta,
                    &support,
                    settings.reconstruct_partners,
                    budget.max_estimated_payload_bytes,
                )?;
                support_sectors::solve_with_plan(
                    &mut theta,
                    &support,
                    &plan,
                    &job.eigensolver,
                    warm,
                    preconditioner
                        .as_mut()
                        .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>),
                    settings,
                )
            })();
            let rejection = match sector {
                Ok(r) => {
                    let accepted = r.report.accepted && r.spectrum.report.accepted;
                    let reason = r.report.rejection.clone();
                    if accepted {
                        report.returned_labels = Some(r.report.output_labels.clone());
                        spectrum = Some(r.spectrum);
                    }
                    report.sector_attempt = Some(r.report);
                    if accepted {
                        None
                    } else {
                        Some(reason.unwrap_or_else(|| "metric sector solve rejected".into()))
                    }
                }
                Err(reason) => Some(reason),
            };
            if let Some(reason) = &rejection {
                ledger.rejected.push(reason.clone());
            }
            ledger.solver_attempts.push(SolveAttempt {
                method: "metric_support_sectors",
                iterations: iteration_total().wrapping_sub(sector_start),
                retained: spectrum.is_some(),
                rejection,
            });
        }
        if spectrum.is_none() {
            let attempt_start = iteration_total();
            let application = if options.symmetry.arithmetic == Arithmetic::SpatialComplex {
                GalerkinApplication::SpatialReference
            } else {
                GalerkinApplication::DirectFourier
            };
            let attempt = galerkin::solve_iterative(
                &mut theta,
                &support,
                &job.eigensolver,
                warm,
                preconditioner
                    .as_mut()
                    .map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>),
                application,
                budget,
            );
            ledger.solver_attempts.push(SolveAttempt {
                method: "metric_support_unreduced",
                iterations: iteration_total().wrapping_sub(attempt_start),
                retained: attempt.is_ok(),
                rejection: attempt.as_ref().err().cloned().or_else(|| {
                    attempt
                        .as_ref()
                        .ok()
                        .and_then(|r| r.report.rejection.clone())
                }),
            });
            ledger.attempted_iterations = iteration_total().wrapping_sub(start);
            spectrum = Some(attempt.map_err(|reason| {
                fail(
                    reason,
                    &result,
                    Some(index),
                    ledger.clone(),
                    Some(report.clone()),
                )
            })?);
        }
        let mut spectrum = spectrum.unwrap();
        ledger.attempted_iterations = iteration_total().wrapping_sub(start);
        ledger.retained_iterations = spectrum.report.iterations;
        report.returned_permutation = (0..spectrum.fields.len()).collect();
        let mut omegas: Vec<_> = spectrum
            .eigenvalues
            .iter()
            .map(|&v| v.max(0.0).sqrt())
            .collect();
        if spectrum.report.accepted && !options.disable_band_tracking {
            crate::profile_scope!("run.band_tracking");
            if let (Some(fields), Some(frequencies)) = (&previous, &previous_omegas) {
                let tracking = track_bands_with_frequencies(
                    fields,
                    &spectrum.fields,
                    frequencies,
                    &omegas,
                    eps.as_deref(),
                );
                if tracking.had_swaps {
                    let p = &tracking.permutation;
                    reorder(&mut omegas, p);
                    reorder(&mut spectrum.eigenvalues, p);
                    reorder(&mut spectrum.fields, p);
                    reorder(&mut spectrum.coefficients, p);
                    for c in [
                        &mut spectrum.report.projected,
                        &mut spectrum.report.full_rectangle,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        reorder(&mut c.residuals, p);
                        reorder(&mut c.absolute_residuals, p);
                    }
                    if let Some(labels) = &mut report.returned_labels {
                        reorder(labels, p);
                    }
                    report.returned_permutation = p.clone();
                }
            }
        }
        report.selected = Some(spectrum.report.clone());
        let certificate = spectrum.report.projected.as_ref();
        let rectangular = spectrum.fields.len() == job.eigensolver.n_bands
            && omegas.len() == job.eigensolver.n_bands
            && certificate.is_some_and(|c| {
                c.residuals.len() == omegas.len() && c.absolute_residuals.len() == omegas.len()
            });
        if !rectangular {
            return Err(fail(
                "metric support produced an incomplete certified point".into(),
                &result,
                Some(index),
                ledger,
                Some(report),
            ));
        }
        let certificate = certificate.unwrap();
        result.k_path.push(k);
        result.distances.push(distances[index]);
        result.bands.push(omegas.clone());
        on_k_point(KPointResult {
            discretization: Some(report.clone()),
            symmetry: ledger.clone(),
            k_index: index,
            total_k_points: job.k_path.len(),
            k_point: k,
            distance: distances[index],
            omegas: omegas.clone(),
            iterations: ledger.attempted_iterations,
            converged: spectrum.report.accepted,
            residuals: certificate.residuals.clone(),
            absolute_residuals: certificate.absolute_residuals.clone(),
            b_orthogonality_defect: certificate.b_orthogonality_defect,
            eigenvectors: options.retain_eigenvectors.then(|| spectrum.fields.clone()),
            is_gamma: spectrum.report.analytic_kernel,
        });
        if !spectrum.report.accepted {
            return Err(fail(
                spectrum
                    .report
                    .rejection
                    .clone()
                    .unwrap_or_else(|| "metric solve rejected".into()),
                &result,
                Some(index),
                ledger,
                Some(report),
            ));
        }
        if let Some(h) = &mut history {
            h.update(k, &spectrum.fields);
        }
        previous = Some(spectrum.fields);
        previous_omegas = Some(omegas);
        previous_values = Some(spectrum.eigenvalues);
    }
    Ok(result)
}

fn reorder<T>(values: &mut Vec<T>, permutation: &[usize]) {
    let mut original: Vec<_> = std::mem::take(values).into_iter().map(Some).collect();
    values.extend(permutation.iter().map(|&i| {
        original[i]
            .take()
            .expect("tracking permutation must be bijective")
    }));
}

fn build_preconditioner<'a, B: SpectralBackend + 'a>(
    theta: &ThetaOperator<B>,
    options: &RunOptions,
    pol: Polarization,
    previous: Option<&[f64]>,
) -> Option<Box<dyn OperatorPreconditioner<B> + 'a>> {
    let previous = previous.filter(|_| options.use_band_window_shift);
    match options.precond_type.resolve_for_polarization(pol) {
        PreconditionerType::None => None,
        PreconditionerType::FourierDiagonalKernelCompensated => Some(match previous {
            Some(v) => Box::new(theta.build_homogeneous_preconditioner_band_window(
                v,
                Some(options.band_window_blend),
                Some(options.band_window_scale),
            )),
            None => Box::new(theta.build_homogeneous_preconditioner_adaptive()),
        }),
        PreconditionerType::TransverseProjection => Some(match previous {
            Some(v) => Box::new(
                theta.build_transverse_projection_preconditioner_band_window(
                    v,
                    Some(options.band_window_blend),
                    Some(options.band_window_scale),
                ),
            ),
            None => Box::new(theta.build_transverse_projection_preconditioner_adaptive()),
        }),
        PreconditionerType::Auto | PreconditionerType::Multigrid => {
            unreachable!("validated and resolved preconditioner")
        }
    }
}
