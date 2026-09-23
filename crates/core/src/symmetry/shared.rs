//! Shared sector trial-space iteration. Full algebra remains a reference mode.
//!
//! This is a separate conservative block-Davidson engine, not an assertion that
//! replacing independent LOBPCG loops alone accelerates them. Gram/Ritz switches
//! must be compared inside this engine, with its numerical policy held fixed.
use super::{
    affine::AffineGroup,
    reuse::{CachedSpectrum, SymmetryOptions},
    sectors::SectorPlan,
    trial::{GramMode, RitzMode},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{
        dense::solve_hermitian_eigen,
        refine::{certify_without_refinement, rayleigh_ritz_certify},
    },
    field::{Field2D, Real, cscalar},
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    preconditioners::OperatorPreconditioner,
};
use num_complex::Complex64 as C;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SharedSolverConfig {
    pub n_bands: usize,
    pub guards_per_sector: usize,
    pub max_iterations: usize,
    pub residual_tolerance: f64,
    pub trial_multiplier: usize,
    /// Cap on a conservative field/dense workspace payload estimate, not RSS.
    pub max_workspace_bytes: usize,
    pub gram: GramMode,
    pub ritz: RitzMode,
}
impl Default for SharedSolverConfig {
    fn default() -> Self {
        Self {
            n_bands: 8,
            guards_per_sector: 2,
            max_iterations: 500,
            residual_tolerance: 1e-8,
            trial_multiplier: 3,
            max_workspace_bytes: 512 * 1024 * 1024,
            gram: GramMode::Full,
            ritz: RitzMode::Full,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct SharedSolverReport {
    pub config: SharedSolverConfig,
    pub rounds: usize,
    pub sector_ranks: Vec<usize>,
    pub prefixes: Vec<usize>,
    pub working_columns: Vec<usize>,
    pub active_sector_rounds: Vec<usize>,
    pub restarts: Vec<usize>,
    pub maximum_trial_columns: usize,
    /// Measured by the full Gram reference; None when those entries are skipped.
    pub maximum_cross_block_defect: Option<f64>,
    pub maximum_mass_defect: f64,
    /// Frobenius ratio measured on actual search matrices before Ritz rotation.
    pub maximum_within_block_offdiagonal_ratio: f64,
    pub maximum_leakage: f64,
    pub conservative_workspace_payload_bytes: usize,
    pub coverage: String,
}
pub struct SharedSolverResult {
    pub spectrum: CachedSpectrum,
    pub report: SharedSolverReport,
}
struct Basis<B: SpectralBackend> {
    q: Vec<B::Buffer>,
    bq: Vec<B::Buffer>,
}
struct RitzBlock<B: SpectralBackend> {
    values: Vec<f64>,
    x: Vec<B::Buffer>,
    ax: Vec<B::Buffer>,
    bx: Vec<B::Buffer>,
}

/// All sectors use full physical vectors, a common operator and one iteration
/// schedule. No row-partner, reduced-kernel, precision or arithmetic change is
/// hidden in the Gram/Ritz experiment. Conservative per-sector prefixes retain
/// empirical coverage, never an inertia certificate.
#[allow(clippy::too_many_arguments)]
pub fn solve_shared<B: SpectralBackend + Clone>(
    op: &mut ThetaOperator<B>,
    lattice: &Lattice2D,
    k: [f64; 2],
    group: AffineGroup,
    config: &SharedSolverConfig,
    options: &SymmetryOptions,
    warm: Option<&[Field2D]>,
    mut preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
) -> Result<SharedSolverResult, String> {
    crate::profile_scope!("symmetry.shared_solve");
    crate::init_faer_sequential();
    if std::mem::size_of::<B::Real>() != 8
        || config.n_bands == 0
        || config.n_bands > op.grid().len()
        || config.max_iterations == 0
        || !config.residual_tolerance.is_finite()
        || config.residual_tolerance <= 0.0
        || config.residual_tolerance > options.max_relative_residual
        || config.trial_multiplier < 2
        || config.trial_multiplier > 8
        || options.reconstruct_partners
        || options.sector_storage != super::reduced::SectorStorage::FullGrid
        || options.arithmetic != super::fourier::Arithmetic::SpatialComplex
        || options.correction_projection_interval != 1
    {
        return Err("unsupported shared solver configuration".into());
    }
    let plan = SectorPlan::qualify(op, lattice, k, group, options)?;
    let ranks = plan.ranks();
    let prefixes: Vec<_> = ranks.iter().map(|&r| r.min(config.n_bands)).collect();
    let working: Vec<_> = ranks
        .iter()
        .map(|&r| r.min(config.n_bands.saturating_add(config.guards_per_sector)))
        .collect();
    let limits: Vec<_> = ranks
        .iter()
        .zip(&working)
        .map(|(&r, &w)| r.min(w.saturating_mul(config.trial_multiplier)))
        .collect();
    let trial_cap = limits
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .ok_or("shared trial count overflow")?;
    let working_sum: usize = working.iter().sum();
    let payload = op
        .grid()
        .len()
        .checked_mul(
            trial_cap
                .checked_mul(5)
                .and_then(|v| working_sum.checked_mul(8)?.checked_add(v))
                .and_then(|v| v.checked_add(16))
                .ok_or("workspace overflow")?,
        )
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| {
            trial_cap
                .checked_mul(trial_cap)?
                .checked_mul(128)?
                .checked_add(v)
        })
        .ok_or("workspace overflow")?;
    if payload > config.max_workspace_bytes {
        return Err(format!(
            "shared solver workspace estimate {payload} exceeds cap {}",
            config.max_workspace_bytes
        ));
    }
    let mut projected = op.alloc_field();
    let mut scratch = op.alloc_field();
    let mut kernel = op.alloc_field();
    let mut bkernel = op.alloc_field();
    let kernel_owner = if k == [0.0; 2] {
        op.fill_gamma_kernel(&mut kernel);
        op.apply_mass(&kernel, &mut bkernel);
        let norm = op.backend().dot(&kernel, &bkernel).re.sqrt();
        op.backend().scale(C::new(norm.recip(), 0.0), &mut kernel);
        op.backend().scale(C::new(norm.recip(), 0.0), &mut bkernel);
        let mut owners = Vec::new();
        for i in 0..ranks.len() {
            plan.project(i, &kernel, &mut projected, &mut scratch);
            if op.backend().dot(&projected, &projected).re
                > 0.5 * op.backend().dot(&kernel, &kernel).re
            {
                owners.push(i);
            }
        }
        if owners.len() != 1 {
            return Err("shared solver cannot assign Gamma kernel".into());
        }
        Some(owners[0])
    } else {
        None
    };
    let mut bases = Vec::new();
    for (i, &count) in working.iter().enumerate() {
        let mut basis = Basis {
            q: Vec::new(),
            bq: Vec::new(),
        };
        if count > 0 && kernel_owner == Some(i) {
            basis.q.push(kernel.clone());
            basis.bq.push(bkernel.clone());
        }
        if let Some(fields) = warm {
            for field in fields {
                if basis.q.len() >= count {
                    break;
                }
                if (
                    field.grid().nx,
                    field.grid().ny,
                    field.grid().centered,
                    field.grid().lx,
                    field.grid().ly,
                ) != (
                    op.grid().nx,
                    op.grid().ny,
                    op.grid().centered,
                    op.grid().lx,
                    op.grid().ly,
                ) || field
                    .as_slice()
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
                {
                    return Err("invalid shared warm field".into());
                }
                let mut v = op.alloc_field();
                for (a, b) in v.as_mut_slice().iter_mut().zip(field.as_slice()) {
                    *a = cscalar::<B::Real>(b.re, b.im);
                }
                plan.project(i, &v, &mut projected, &mut scratch);
                append_basis(op, &plan, i, &mut basis, projected.clone());
            }
        }
        for attempt in 0..count.saturating_mul(16).saturating_add(32) {
            if basis.q.len() >= count {
                break;
            }
            let mut v = op.alloc_field();
            crate::eigensolver::initialization::seed_random_vector(
                v.as_mut_slice(),
                (1 + attempt + 997 * i) as f64,
            );
            plan.project(i, &v, &mut projected, &mut scratch);
            append_basis(op, &plan, i, &mut basis, projected.clone());
        }
        if basis.q.len() != count {
            return Err(format!(
                "shared sector {i} initialization rank {} below {count}",
                basis.q.len()
            ));
        }
        bases.push(basis);
    }
    let mut report = SharedSolverReport {
        config: config.clone(),
        rounds: 0,
        sector_ranks: ranks.clone(),
        prefixes: prefixes.clone(),
        working_columns: working.clone(),
        active_sector_rounds: vec![0; ranks.len()],
        restarts: vec![0; ranks.len()],
        maximum_trial_columns: 0,
        maximum_cross_block_defect: (config.gram == GramMode::Full).then_some(0.0),
        maximum_mass_defect: 0.0,
        maximum_within_block_offdiagonal_ratio: 0.0,
        maximum_leakage: 0.0,
        conservative_workspace_payload_bytes: payload,
        coverage:
            "conservative per-sector lowest prefixes; empirical iterative coverage, not inertia"
                .into(),
    };
    for round in 0..config.max_iterations {
        crate::eigensolver::record_iteration();
        crate::profiler::count("shared.iterations", 1);
        report.rounds = round + 1;
        let mut ranges = Vec::new();
        let mut q = Vec::new();
        let mut bq = Vec::new();
        for basis in &bases {
            let start = q.len();
            q.extend(basis.q.iter().cloned());
            bq.extend(basis.bq.iter().cloned());
            ranges.push(start..q.len());
        }
        let dim = q.len();
        report.maximum_trial_columns = report.maximum_trial_columns.max(dim);
        // Structural qualification is not permission to ignore numerical
        // drift. Check the actual trial columns in B norm before dropping
        // forbidden Gram entries; otherwise that shortcut would hide leakage.
        let mut mass_error = op.alloc_field();
        for (sector, range) in ranges.iter().enumerate() {
            for j in range.clone() {
                plan.project(sector, &q[j], &mut projected, &mut scratch);
                op.backend().axpy(-C::ONE, &q[j], &mut projected);
                op.apply_mass(&projected, &mut mass_error);
                let leakage = (op.backend().dot(&projected, &mass_error).re.max(0.0)
                    / op.backend().dot(&q[j], &bq[j]).re)
                    .sqrt();
                report.maximum_leakage = report.maximum_leakage.max(leakage);
                if !leakage.is_finite() || leakage > 1e-10 {
                    return Err(format!(
                        "shared trial sector {sector} leaked by {leakage:e}"
                    ));
                }
            }
        }
        let mut aq: Vec<_> = q.iter().map(|_| op.alloc_field()).collect();
        op.batch_apply(&q, &mut aq);
        let mass = project_matrix(op.backend(), &q, &bq, &ranges, config.gram);
        let mass_defect = mass
            .iter()
            .enumerate()
            .map(|(i, z)| (*z - if i % dim == i / dim { C::ONE } else { C::ZERO }).norm())
            .fold(0.0, f64::max);
        report.maximum_mass_defect = report.maximum_mass_defect.max(mass_defect);
        if !mass_defect.is_finite() || mass_defect > 1e-8 {
            return Err(format!("shared trial mass defect {mass_defect:e}"));
        }
        let h = project_matrix(op.backend(), &q, &aq, &ranges, config.gram);
        let mut off_diagonal = 0.0;
        let mut block_norm = 0.0;
        for range in &ranges {
            for j in range.clone() {
                for i in range.clone() {
                    let v = h[i + dim * j].norm_sqr();
                    block_norm += v;
                    if i != j {
                        off_diagonal += v;
                    }
                }
            }
        }
        report.maximum_within_block_offdiagonal_ratio = report
            .maximum_within_block_offdiagonal_ratio
            .max((off_diagonal / block_norm.max(f64::MIN_POSITIVE)).sqrt());
        if config.gram == GramMode::Full {
            let cross = cross_defect(&h, dim, &ranges);
            report.maximum_cross_block_defect =
                Some(report.maximum_cross_block_defect.unwrap().max(cross));
            if !cross.is_finite() || cross > 1e-9 {
                return Err(format!("shared trial forbidden coupling {cross:e}"));
            }
        }
        let coeff = sector_ritz(&h, dim, &ranges, config.ritz)?;
        let mut blocks: Vec<RitzBlock<B>> = Vec::new();
        let mut all_done = true;
        let mut corrections = Vec::new();
        for i in 0..ranks.len() {
            let mut block: RitzBlock<B> = RitzBlock {
                values: Vec::new(),
                x: Vec::new(),
                ax: Vec::new(),
                bx: Vec::new(),
            };
            for (j, (lambda, c)) in coeff[i].iter().take(working[i]).enumerate() {
                let mut x = combine(op.backend(), &q, c);
                let mut a = combine(op.backend(), &aq, c);
                let mut b = combine(op.backend(), &bq, c);
                let mut value = *lambda;
                if j == 0 && kernel_owner == Some(i) {
                    x = kernel.clone();
                    b = bkernel.clone();
                    op.apply(&x, &mut a);
                    value = 0.0;
                }
                block.values.push(value);
                block.x.push(x);
                block.ax.push(a);
                block.bx.push(b);
            }
            if block.values.len() != working[i] {
                return Err(format!("shared Ritz lost sector {i} columns"));
            }
            let mut residuals = Vec::new();
            let mut done = true;
            for j in 0..block.values.len() {
                let mut r = block.ax[j].clone();
                op.backend()
                    .axpy(C::new(-block.values[j], 0.0), &block.bx[j], &mut r);
                let nr = op.backend().dot(&r, &r).re.max(0.0).sqrt();
                let den = op
                    .backend()
                    .dot(&block.ax[j], &block.ax[j])
                    .re
                    .max(0.0)
                    .sqrt()
                    + block.values[j].abs()
                        * op.backend()
                            .dot(&block.bx[j], &block.bx[j])
                            .re
                            .max(0.0)
                            .sqrt();
                let converged = if kernel_owner == Some(i) && j == 0 {
                    nr / op.backend().dot(&block.x[j], &block.x[j]).re.sqrt()
                        <= options.max_zero_mode_absolute_residual
                } else {
                    nr / den.max(f64::MIN_POSITIVE) <= config.residual_tolerance
                };
                if j < prefixes[i] {
                    done &= converged;
                }
                if !converged {
                    residuals.push(r);
                }
            }
            if !done {
                report.active_sector_rounds[i] += 1;
                all_done = false;
            }
            corrections.push(if done { Vec::new() } else { residuals });
            blocks.push(block);
        }
        if all_done {
            let mut all = Vec::new();
            for i in 0..ranks.len() {
                let block = &blocks[i];
                let mut values = block.values[..prefixes[i]].to_vec();
                let mut fields: Vec<_> = block.x[..prefixes[i]].iter().map(promote::<B>).collect();
                rayleigh_ritz_certify(op, &mut values, &mut fields);
                if kernel_owner == Some(i) && !fields.is_empty() {
                    values[0] = 0.0;
                    fields[0] = promote::<B>(&kernel);
                }
                for f in &fields {
                    let mut v = op.alloc_field();
                    for (a, b) in v.as_mut_slice().iter_mut().zip(f.as_slice()) {
                        *a = cscalar::<B::Real>(b.re, b.im);
                    }
                    plan.project(i, &v, &mut projected, &mut scratch);
                    op.backend().axpy(-C::ONE, &v, &mut projected);
                    let leakage = (op.backend().dot(&projected, &projected).re
                        / op.backend().dot(&v, &v).re)
                        .sqrt();
                    report.maximum_leakage = report.maximum_leakage.max(leakage);
                }
                all.extend(values.into_iter().zip(fields));
            }
            if !report.maximum_leakage.is_finite() || report.maximum_leakage > 1e-8 {
                return Err("shared final fields leaked".into());
            }
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            all.truncate(config.n_bands);
            let (values, fields): (Vec<_>, Vec<_>) = all.into_iter().unzip();
            let cert = certify_without_refinement(op, &values, &fields);
            let spectrum = CachedSpectrum {
                source_index: 0,
                k,
                eigenvalues: values,
                fields,
                residuals: cert.residuals,
                absolute_residuals: cert.absolute_residuals,
                orthogonality_defect: cert.b_orthogonality_defect,
            };
            if spectrum.fields.len() != config.n_bands || !spectrum.passes(options) {
                return Err("shared final output failed physical quality gates".into());
            }
            return Ok(SharedSolverResult { spectrum, report });
        }
        drop(q);
        drop(aq);
        drop(bq);
        for i in 0..bases.len() {
            if corrections[i].is_empty() {
                continue;
            }
            if bases[i].q.len() + corrections[i].len() > limits[i] {
                bases[i] = Basis {
                    q: Vec::new(),
                    bq: Vec::new(),
                };
                report.restarts[i] += 1;
                for x in &blocks[i].x {
                    if !append_basis(op, &plan, i, &mut bases[i], x.clone()) {
                        return Err("shared restart lost active rank".into());
                    }
                }
            }
            let before = bases[i].q.len();
            for mut r in corrections[i].drain(..) {
                if bases[i].q.len() >= limits[i] {
                    break;
                }
                if let Some(p) = preconditioner.as_mut() {
                    p.apply(op.backend(), &mut r);
                }
                plan.project(i, &r, &mut projected, &mut scratch);
                append_basis(op, &plan, i, &mut bases[i], projected.clone());
            }
            if bases[i].q.len() == before {
                return Err(format!(
                    "shared sector {i} stagnated without a new independent correction"
                ));
            }
        }
    }
    Err(format!(
        "shared solver exceeded {} rounds",
        config.max_iterations
    ))
}

fn append_basis<B: SpectralBackend>(
    op: &mut ThetaOperator<B>,
    plan: &SectorPlan,
    sector: usize,
    basis: &mut Basis<B>,
    mut v: B::Buffer,
) -> bool {
    let mut b = op.alloc_field();
    op.apply_mass(&v, &mut b);
    let original = op.backend().dot(&v, &b).re;
    if !original.is_finite() || original <= 1e-30 {
        return false;
    }
    let scale = C::new(original.sqrt().recip(), 0.0);
    op.backend().scale(scale, &mut v);
    op.backend().scale(scale, &mut b);
    for _ in 0..2 {
        for (q, bq) in basis.q.iter().zip(&basis.bq) {
            let a = op.backend().dot(q, &b);
            op.backend().axpy(-a, q, &mut v);
            op.backend().axpy(-a, bq, &mut b);
        }
    }
    op.apply_mass(&v, &mut b);
    let norm = op.backend().dot(&v, &b).re;
    if !norm.is_finite() || norm <= 1e-18 {
        return false;
    }
    let scale = C::new(norm.sqrt().recip(), 0.0);
    op.backend().scale(scale, &mut v);
    op.backend().scale(scale, &mut b);
    // Near dependence can amplify projector roundoff during the first MGS.
    // Reproject the normalized surviving direction, then restore B orthogonality.
    // This is unconditional in every Gram/Ritz control, never hidden by zeroing
    // forbidden matrix entries. A numerical ghost that loses rank is discarded.
    let mut projected = op.alloc_field();
    let mut scratch = op.alloc_field();
    plan.project(sector, &v, &mut projected, &mut scratch);
    v = projected;
    op.apply_mass(&v, &mut b);
    for _ in 0..2 {
        for (q, bq) in basis.q.iter().zip(&basis.bq) {
            let a = op.backend().dot(q, &b);
            op.backend().axpy(-a, q, &mut v);
            op.backend().axpy(-a, bq, &mut b);
        }
    }
    op.apply_mass(&v, &mut b);
    let retained = op.backend().dot(&v, &b).re;
    if !retained.is_finite() || retained <= 1e-12 {
        return false;
    }
    let scale = C::new(retained.sqrt().recip(), 0.0);
    op.backend().scale(scale, &mut v);
    op.backend().scale(scale, &mut b);
    basis.q.push(v);
    basis.bq.push(b);
    true
}
fn combine<B: SpectralBackend>(backend: &B, fields: &[B::Buffer], coeff: &[C]) -> B::Buffer {
    let mut v = backend.alloc_field(fields[0].grid());
    for (f, &c) in fields.iter().zip(coeff) {
        if c != C::ZERO {
            backend.axpy(c, f, &mut v);
        }
    }
    v
}
fn promote<B: SpectralBackend>(f: &B::Buffer) -> Field2D {
    Field2D::from_vec(
        f.grid(),
        f.as_slice()
            .iter()
            .map(|z| C::new(z.re.to_accum(), z.im.to_accum()))
            .collect(),
    )
}
fn project_matrix<B: SpectralBackend>(
    backend: &B,
    left: &[B::Buffer],
    right: &[B::Buffer],
    ranges: &[std::ops::Range<usize>],
    mode: GramMode,
) -> Vec<C> {
    crate::profile_scope!("symmetry.shared_gram");
    let n = left.len();
    let mut h = vec![C::ZERO; n * n];
    let all = vec![0..n];
    let selected = if mode == GramMode::Full { &all } else { ranges };
    for range in selected {
        let p = range.len();
        if p == 0 {
            continue;
        }
        let gram = backend.gram_matrix(&left[range.clone()], &right[range.clone()]);
        crate::profiler::count("shared_gram.multiply_terms", left[0].len() * p * p);
        for (i, row) in range.clone().enumerate() {
            for (j, col) in range.clone().enumerate() {
                h[row + n * col] = 0.5 * (gram[i * p + j] + gram[j * p + i].conj());
            }
        }
    }
    h
}
fn cross_defect(h: &[C], n: usize, ranges: &[std::ops::Range<usize>]) -> f64 {
    let mut labels = vec![0; n];
    for (i, r) in ranges.iter().enumerate() {
        labels[r.clone()].fill(i);
    }
    let mut error = 0.0;
    let mut norm = 0.0;
    for j in 0..n {
        for i in 0..n {
            let v = h[i + n * j].norm_sqr();
            norm += v;
            if labels[i] != labels[j] {
                error += v;
            }
        }
    }
    (error / norm.max(f64::MIN_POSITIVE)).sqrt()
}
fn sector_ritz(
    h: &[C],
    n: usize,
    ranges: &[std::ops::Range<usize>],
    mode: RitzMode,
) -> Result<Vec<Vec<(f64, Vec<C>)>>, String> {
    crate::profile_scope!("symmetry.shared_ritz");
    let mut result: Vec<Vec<(f64, Vec<C>)>> = (0..ranges.len()).map(|_| Vec::new()).collect();
    if mode == RitzMode::Blocks {
        for (i, range) in ranges.iter().enumerate() {
            let p = range.len();
            if p == 0 {
                continue;
            }
            let mut block = vec![C::ZERO; p * p];
            for (a, row) in range.clone().enumerate() {
                for (b, col) in range.clone().enumerate() {
                    block[a + p * b] = h[row + n * col];
                }
            }
            crate::profiler::count("shared_ritz.cubic_dimension", p * p * p);
            let eig = solve_hermitian_eigen(&block, p);
            for j in 0..p {
                let mut c = vec![C::ZERO; n];
                c[range.clone()].copy_from_slice(&eig.eigenvectors[j * p..(j + 1) * p]);
                result[i].push((eig.eigenvalues[j], c));
            }
        }
        return Ok(result);
    }
    crate::profiler::count("shared_ritz.cubic_dimension", n * n * n);
    let eig = solve_hermitian_eigen(h, n);
    // A full dense solve may rotate a degenerate cluster between sectors.
    // Split the complete coefficient eigenspace by the qualified row ranges,
    // not by assigning an individual mixed eigenvector to its largest overlap.
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n
            && (eig.eigenvalues[end] - eig.eigenvalues[start]).abs()
                <= 1e-11 * eig.eigenvalues[start].abs().max(1.0)
        {
            end += 1;
        }
        for (i, range) in ranges.iter().enumerate() {
            let candidates: Vec<Vec<C>> = (start..end)
                .map(|j| {
                    let mut c = vec![C::ZERO; n];
                    c[range.clone()]
                        .copy_from_slice(&eig.eigenvectors[j * n + range.start..j * n + range.end]);
                    c
                })
                .collect();
            for c in cluster_basis(candidates, range.len())? {
                let mut hc = vec![C::ZERO; n];
                for row in 0..n {
                    for col in 0..n {
                        hc[row] += h[row + n * col] * c[col];
                    }
                }
                let lambda = c
                    .iter()
                    .zip(&hc)
                    .map(|(a, b)| (a.conj() * b).re)
                    .sum::<f64>();
                let error = c
                    .iter()
                    .zip(&hc)
                    .map(|(a, b)| (b - lambda * a).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / lambda.abs().max(1.0);
                if !error.is_finite() || error > 1e-8 {
                    return Err(
                        "full shared Ritz cluster could not be separated into qualified sectors"
                            .into(),
                    );
                }
                result[i].push((lambda, c));
            }
        }
        start = end;
    }
    for (sector, range) in result.iter_mut().zip(ranges) {
        sector.sort_by(|a, b| a.0.total_cmp(&b.0));
        if sector.len() != range.len() {
            return Err("full shared Ritz label recovery lost rank".into());
        }
    }
    Ok(result)
}

/// Recover the range of a projected *complete* coefficient cluster. Its trace
/// gives the integer rank; pivoted two-pass QR recovers that many directions.
/// A tiny projection from a nearby eigenvalue must not become an extra mode
/// merely because normalizing it produces a unit vector.
fn cluster_basis(
    mut candidates: Vec<Vec<C>>,
    sector_dimension: usize,
) -> Result<Vec<Vec<C>>, String> {
    let trace: f64 = candidates.iter().flatten().map(C::norm_sqr).sum();
    if !trace.is_finite() || (trace - trace.round()).abs() > 1e-7 {
        return Err("ambiguous sector rank in full Ritz cluster".into());
    }
    let rank = trace.round() as usize;
    if rank > sector_dimension || rank > candidates.len() {
        return Err("invalid sector rank in full Ritz cluster".into());
    }
    let mut basis = Vec::with_capacity(rank);
    for _ in 0..rank {
        let norms: Vec<f64> = candidates
            .iter()
            .map(|c| c.iter().map(C::norm_sqr).sum())
            .collect();
        let pivot = (0..norms.len())
            .max_by(|&a, &b| norms[a].total_cmp(&norms[b]))
            .unwrap();
        if !norms[pivot].is_finite() || norms[pivot] <= 1e-12 {
            return Err("projected full Ritz cluster lost rank".into());
        }
        let mut q = candidates[pivot].clone();
        for v in &mut q {
            *v /= norms[pivot].sqrt();
        }
        for c in &mut candidates {
            for _ in 0..2 {
                let a: C = q.iter().zip(c.iter()).map(|(x, y)| x.conj() * y).sum();
                for (v, x) in c.iter_mut().zip(&q) {
                    *v -= a * x;
                }
            }
        }
        basis.push(q);
    }
    if candidates.iter().flatten().map(C::norm_sqr).sum::<f64>() > 1e-7 {
        return Err("full Ritz cluster has unresolved projected energy".into());
    }
    Ok(basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_sector_eigenvalues_do_not_create_ghost_modes() {
        let h = vec![
            C::ONE,
            C::new(1e-13, 0.0),
            C::new(1e-13, 0.0),
            C::new(1.0 + 1e-7, 0.0),
        ];
        let result = sector_ritz(&h, 2, &[0..1, 1..2], RitzMode::Full).unwrap();
        assert_eq!(result.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 1]);
        assert!((result[0][0].0 - 1.0).abs() < 1e-14);
        assert!((result[1][0].0 - 1.0 - 1e-7).abs() < 1e-14);
    }

    #[test]
    fn full_ritz_recovers_sectors_from_degenerate_nondiagonal_blocks() {
        let ranges = vec![0..3, 3..6];
        let n = 6;
        let mut h = vec![C::ZERO; n * n];
        // Identical 1 and 4 eigenvalues in different sectors force label
        // recovery to handle an entire degenerate eigenspace. Rotation makes
        // the input genuinely nondiagonal, unlike a prepared Ritz basis.
        for (i, values) in [[1.0, 1.0, 4.0], [1.0, 2.0, 4.0]].iter().enumerate() {
            let t: f64 = 0.71 + 0.2 * i as f64;
            let u = [
                [t.cos(), 0.0, -t.sin()],
                [0.0, 1.0, 0.0],
                [t.sin(), 0.0, t.cos()],
            ];
            for a in 0..3 {
                for b in 0..3 {
                    h[(a + 3 * i) + n * (b + 3 * i)] =
                        C::new((0..3).map(|j| u[a][j] * values[j] * u[b][j]).sum(), 0.0);
                }
            }
        }
        assert!(h[2].norm() > 0.1);
        let full = sector_ritz(&h, n, &ranges, RitzMode::Full).unwrap();
        let blocks = sector_ritz(&h, n, &ranges, RitzMode::Blocks).unwrap();
        for (i, (a, b)) in full.iter().zip(&blocks).enumerate() {
            assert_eq!(a.len(), 3);
            for ((x, c), (y, _)) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-11);
                assert!(
                    c.iter()
                        .enumerate()
                        .filter(|(j, _)| !ranges[i].contains(j))
                        .all(|(_, z)| *z == C::ZERO)
                );
            }
            for (j, (_, x)) in a.iter().enumerate() {
                for (l, (_, y)) in a.iter().enumerate() {
                    let dot: C = x.iter().zip(y).map(|(u, v)| u.conj() * v).sum();
                    assert!((dot - if j == l { C::ONE } else { C::ZERO }).norm() < 1e-10);
                }
            }
        }
    }
}
