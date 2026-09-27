//! Fourier prolongation for a coarse-grid initializer of the unchanged fine
//! problem. This changes only the seed; the fine solver still needs complete
//! physical residual and spectrum validation.
use super::action::fft_indices;
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    field::{Field2D, Real, cscalar},
    grid::Grid2D,
};
use serde::Serialize;

/// Optional first-point initialization. The fine operator and its final quality
/// requirements are unchanged. This is a seed policy, not exact solve reuse.
#[derive(Debug, Clone, Copy)]
pub struct CoarseStartOptions {
    pub resolution: [usize; 2],
    /// Admission for simultaneously retained coarse, lifted and prepared seed
    /// fields. Solver/FFT/rank-extraction scratch and allocator overhead are extra.
    pub max_seed_field_bytes: usize,
}

impl CoarseStartOptions {
    pub fn seed_field_bytes(
        &self,
        fine: Grid2D,
        config: &crate::eigensolver::EigensolverConfig,
    ) -> Result<usize, String> {
        let [nx, ny] = self.resolution;
        if fine.centered
            || nx == 0
            || ny == 0
            || nx > fine.nx
            || ny > fine.ny
            || (nx == fine.nx && ny == fine.ny)
        {
            return Err(
                "coarse initialization requires a smaller nested node-centered grid".into(),
            );
        }
        let coarse_points = nx.checked_mul(ny).ok_or("coarse grid size overflows")?;
        if config.n_bands == 0 || config.effective_block_size() >= coarse_points {
            return Err("coarse grid must have more points than the effective solver block".into());
        }
        fine.nx
            .checked_mul(fine.ny)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(coarse_points))
            .and_then(|n| n.checked_mul(config.n_bands))
            .and_then(|n| n.checked_mul(std::mem::size_of::<num_complex::Complex64>()))
            .ok_or_else(|| "coarse seed field payload overflows".into())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CoarseStartReport {
    pub resolution: [usize; 2],
    pub reserved_seed_field_bytes: Option<usize>,
    pub max_seed_field_bytes: usize,
    pub payload_scope: &'static str,
    pub iterations: usize,
    pub converged: bool,
    pub residuals: Vec<f64>,
    pub b_orthogonality_defect: Option<f64>,
    pub prepared: Option<super::seed::SeedReport>,
    pub used_as_seed: bool,
    pub rejection: Option<String>,
    pub seconds: f64,
}

/// Run the complete coarse initializer, returning every attempted iteration even
/// if validation or target preparation fails. Nonconverged coarse vectors may
/// supply an approximate seed if finite and independent in the fine mass metric;
/// the returned report never labels them a converged fine solution.
#[allow(clippy::too_many_arguments)]
pub fn prepare_initializer<B: SpectralBackend + Clone>(
    target: &mut crate::operators::ThetaOperator<B>,
    geometry: &crate::geometry::Geometry2D,
    dielectric: &crate::dielectric::DielectricOptions,
    k: [f64; 2],
    config: &crate::eigensolver::EigensolverConfig,
    options: CoarseStartOptions,
    rank_tolerance: f64,
) -> (Option<super::seed::RankAwareSeed>, CoarseStartReport) {
    use crate::drivers::single_solve::{SingleSolveJob, solve};
    crate::profile_scope!("initialization.coarse");
    let timer = crate::timing::Timer::start();
    let first_iteration = crate::eigensolver::iteration_total();
    let mut report = CoarseStartReport {
        resolution: options.resolution,
        reserved_seed_field_bytes: None,
        max_seed_field_bytes: options.max_seed_field_bytes,
        payload_scope: "coarse output plus lifted and prepared physical seed fields; excludes solver, FFT and rank-extraction scratch",
        iterations: 0,
        converged: false,
        residuals: Vec::new(),
        b_orthogonality_defect: None,
        prepared: None,
        used_as_seed: false,
        rejection: None,
        seconds: 0.0,
    };
    let attempt = (|| -> Result<super::seed::RankAwareSeed, String> {
        if std::mem::size_of::<B::Real>() != 8 {
            return Err("coarse initialization requires f64 field storage".into());
        }
        if !rank_tolerance.is_finite()
            || rank_tolerance <= 0.0
            || rank_tolerance >= 1.0
            || k.iter().any(|v| !v.is_finite())
        {
            return Err("invalid coarse initializer wavevector or rank tolerance".into());
        }
        let fine = target.grid();
        let reserved = options.seed_field_bytes(fine, config)?;
        report.reserved_seed_field_bytes = Some(reserved);
        if reserved > options.max_seed_field_bytes {
            return Err("coarse seed fields exceed the declared payload budget".into());
        }
        let coarse = Grid2D::new(
            options.resolution[0],
            options.resolution[1],
            fine.lx,
            fine.ly,
        );
        let backend = target.backend().clone();
        let mut operator = crate::operators::ThetaOperator::new(
            backend.clone(),
            crate::dielectric::Dielectric2D::from_geometry(geometry, coarse, dielectric),
            target.polarization(),
            geometry.lattice.reciprocal().fractional_to_cartesian(k),
        );
        let mut preconditioner = operator.build_homogeneous_preconditioner_adaptive();
        let source_job = SingleSolveJob::new(config.n_bands)
            .with_tolerance(config.tol)
            .with_max_iterations(config.max_iter)
            .with_block_size(config.block_size);
        let source = solve(&mut operator, Some(&mut preconditioner), &source_job);
        report.converged = source.converged;
        report.residuals = source.final_residuals;
        report.b_orthogonality_defect = Some(source.b_orthogonality_defect);
        let lifted = prolongate_fourier(&backend, &source.eigenvectors, fine)?;
        let prepared = super::seed::rank_aware_seed(target, &lifted, rank_tolerance)
            .map_err(|error| error.to_string())?;
        if prepared.fields.is_empty() {
            return Err("coarse initialization supplied no independent fine-grid direction".into());
        }
        report.prepared = Some(prepared.report.clone());
        report.used_as_seed = true;
        Ok(prepared)
    })();
    report.iterations = crate::eigensolver::iteration_total().wrapping_sub(first_iteration);
    report.seconds = timer.elapsed_secs();
    let seed = match attempt {
        Ok(seed) => Some(seed),
        Err(reason) => {
            report.rejection = Some(reason);
            None
        }
    };
    (seed, report)
}

/// Interpolate periodic complex fields by zero-padding the stored signed
/// Fourier support. Blaze stores an even axis's Nyquist index as positive;
/// that single mode is preserved, without splitting it into ± partners.
/// A real coarse Nyquist sample is therefore not promised a real interpolant.
/// Target B-normalization and Rayleigh-Ritz preparation are separate steps.
pub fn prolongate_fourier<B: SpectralBackend>(
    backend: &B,
    fields: &[Field2D],
    fine: Grid2D,
) -> Result<Vec<Field2D>, String> {
    crate::profile_scope!("initialization.coarse_prolongation");
    if std::mem::size_of::<B::Real>() != 8 {
        return Err("coarse Fourier prolongation currently requires f64".into());
    }
    if fine.nx == 0
        || fine.ny == 0
        || fine.centered
        || !fine.lx.is_finite()
        || !fine.ly.is_finite()
        || fine.lx <= 0.0
        || fine.ly <= 0.0
    {
        return Err("fine grid must be finite, positive and node-centered".into());
    }
    let Some(first) = fields.first() else {
        return Ok(Vec::new());
    };
    let coarse = first.grid();
    if coarse.nx == 0
        || coarse.ny == 0
        || coarse.centered
        || coarse.nx > fine.nx
        || coarse.ny > fine.ny
        || coarse.lx.to_bits() != fine.lx.to_bits()
        || coarse.ly.to_bits() != fine.ly.to_bits()
    {
        return Err(
            "coarse and fine grids must share the same node-centered cell with nested support"
                .into(),
        );
    }
    if fields.iter().any(|f| {
        let g = f.grid();
        (g.nx, g.ny, g.centered, g.lx.to_bits(), g.ly.to_bits())
            != (
                coarse.nx,
                coarse.ny,
                coarse.centered,
                coarse.lx.to_bits(),
                coarse.ly.to_bits(),
            )
            || f.as_slice()
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
    }) {
        return Err("coarse fields have inconsistent grids or nonfinite values".into());
    }
    let axis_map = |source: usize, target: usize| -> Result<Vec<usize>, String> {
        let indices = fft_indices(target);
        fft_indices(source)
            .into_iter()
            .map(|mode| {
                indices
                    .iter()
                    .position(|&m| m == mode)
                    .ok_or_else(|| "coarse Fourier support is not nested".into())
            })
            .collect()
    };
    let x_map = axis_map(coarse.nx, fine.nx)?;
    let y_map = axis_map(coarse.ny, fine.ny)?;
    let mut input = backend.alloc_field(coarse);
    let mut output = backend.alloc_field(fine);
    let scale = fine.len() as f64 / coarse.len() as f64;
    let mut result = Vec::with_capacity(fields.len());
    for field in fields {
        for (z, a) in input.as_mut_slice().iter_mut().zip(field.as_slice()) {
            *z = cscalar::<B::Real>(a.re, a.im);
        }
        backend.forward_fft_2d(&mut input);
        output.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
        for (j, &y) in y_map.iter().enumerate() {
            for (i, &x) in x_map.iter().enumerate() {
                output.as_mut_slice()[fine.idx(x, y)] =
                    input.as_slice()[coarse.idx(i, j)] * B::Real::from_accum(scale);
            }
        }
        backend.inverse_fft_2d(&mut output);
        result.push(Field2D::from_vec(
            fine,
            output
                .as_slice()
                .iter()
                .map(|z| num_complex::Complex64::new(z.re.to_accum(), z.im.to_accum()))
                .collect(),
        ));
        crate::profiler::count("coarse.prolongated_columns", 1);
    }
    Ok(result)
}
