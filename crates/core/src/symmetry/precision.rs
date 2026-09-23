//! Explicit f32 initialization followed by an unchanged f64 target solve.
//!
//! This is separate from exact real symmetry storage. Low precision is never
//! used to authorize an exact symmetry cache hit or a final f64 certificate.
use super::seed::{SeedReport, rank_aware_seed};
use crate::{
    backend::SpectralBackend,
    drivers::single_solve::{SingleSolveJob, solve},
    eigensolver::refine::certify_without_refinement,
    field::Field2D,
    operators::{LinearOperator, ThetaOperator},
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct F32SeedConfig {
    pub max_iterations: usize,
    pub eigenvalue_tolerance: f64,
    pub target_relative_rank_tolerance: f64,
}
impl Default for F32SeedConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            eigenvalue_tolerance: 1e-6,
            target_relative_rank_tolerance: 1e-9,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct F32SeedReport {
    pub config: F32SeedConfig,
    pub low_precision_iterations: usize,
    pub low_precision_converged: bool,
    pub low_precision_residuals: Vec<f64>,
    pub promoted_physical_residuals: Vec<f64>,
    pub promoted_absolute_residuals: Vec<f64>,
    pub promoted_mass_orthogonality_defect: f64,
    pub low_field_column_bytes: usize,
    pub target_field_column_bytes: usize,
    pub target_seed: SeedReport,
}
pub struct F32Seed {
    pub fields: Vec<Field2D>,
    pub report: F32SeedReport,
}

/// Construct the same finite problem with f32 field/FFT storage and f64 dense
/// accumulation, then promote and prepare its fields in the target f64 mass
/// metric. The low solve may exhaust its budget: finite rank-valid output is
/// still an approximate seed. Final target correction/certification is the
/// caller's responsibility. No resolution, material or kernel is changed.
pub fn prepare_f32_seed<H, L>(
    target: &mut ThetaOperator<H>,
    low_backend: L,
    target_job: &SingleSolveJob,
    config: F32SeedConfig,
) -> Result<F32Seed, String>
where
    H: SpectralBackend<Real = f64>,
    L: SpectralBackend<Real = f32>,
{
    crate::profile_scope!("initialization.f32_seed");
    if config.max_iterations == 0
        || !config.eigenvalue_tolerance.is_finite()
        || config.eigenvalue_tolerance <= 0.0
        || config.eigenvalue_tolerance >= 1.0
        || !config.target_relative_rank_tolerance.is_finite()
        || config.target_relative_rank_tolerance <= 0.0
        || config.target_relative_rank_tolerance >= 1.0
        || target_job.n_bands == 0
        || target_job.n_bands >= target.grid().len()
    {
        return Err("invalid low-precision initializer configuration".into());
    }
    let mut low_operator = ThetaOperator::new(
        low_backend,
        target.dielectric().clone(),
        target.polarization(),
        target.bloch(),
    );
    let mut low_preconditioner = low_operator.build_homogeneous_preconditioner_adaptive();
    let mut low_job = target_job.clone();
    low_job.tolerance = config.eigenvalue_tolerance;
    low_job.max_iterations = config.max_iterations;
    low_job.record_diagnostics = false;
    let low = solve(&mut low_operator, Some(&mut low_preconditioner), &low_job);
    drop(low_preconditioner);
    drop(low_operator);
    if low.eigenvectors.len() != target_job.n_bands
        || low.eigenvalues.len() != target_job.n_bands
        || low.eigenvalues.iter().any(|v| !v.is_finite())
        || low.eigenvectors.iter().any(|f| {
            f.as_slice()
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
        })
    {
        return Err("low-precision solve did not return a finite requested seed block".into());
    }
    // Test the promoted fields against actual f64 A/B, before normalization
    // or Ritz rotation could hide the low-precision orthogonality floor.
    let certificate = certify_without_refinement(target, &low.eigenvalues, &low.eigenvectors);
    let prepared = rank_aware_seed(
        target,
        &low.eigenvectors,
        config.target_relative_rank_tolerance,
    )
    .map_err(|e| e.to_string())?;
    Ok(F32Seed {
        fields: prepared.fields,
        report: F32SeedReport {
            config,
            low_precision_iterations: low.iterations,
            low_precision_converged: low.converged,
            low_precision_residuals: low.final_residuals,
            promoted_physical_residuals: certificate.residuals,
            promoted_absolute_residuals: certificate.absolute_residuals,
            promoted_mass_orthogonality_defect: certificate.b_orthogonality_defect,
            low_field_column_bytes: target.grid().len() * 8,
            target_field_column_bytes: target.grid().len() * 16,
            target_seed: prepared.report,
        },
    })
}
