//! Shared rectangular-sector numerical payload model.
//!
//! Admission covers qualification/action/orbit tables, private solve and seed
//! work, retained spectra and partner reconstruction. This is not a process-RSS
//! bound: FFT plans, container/allocator overhead and library scratch are extra.
use super::super::reduced::{ReducedKernel, SectorStorage};
use serde::Serialize;

pub const DEFAULT_LIMIT_BYTES: usize = 512 * 1024 * 1024;
pub const MODEL: &str = "rectangular-sector-payload/1";

fn product(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| "sector payload overflow".into())
}
pub(crate) fn sum(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .ok_or_else(|| "sector payload overflow".into())
}

/// Bound the plan before constructing representation, actions or orbit bases.
/// The orbit coefficient count is at most group_order * rows across sectors.
pub fn setup_payload(
    rows: usize,
    group_order: usize,
    storage: SectorStorage,
) -> Result<usize, String> {
    let actions = product(&[rows, group_order, 2 * std::mem::size_of::<usize>() + 32])?;
    let orbits = if storage == SectorStorage::Orbit {
        sum(&[
            product(&[rows, group_order, std::mem::size_of::<usize>() + 16])?,
            product(&[rows, 64])?,
        ])?
    } else {
        0
    };
    sum(&[
        product(&[rows, 16, 32])?,
        actions,
        orbits,
        product(&[256, group_order, group_order, group_order])?,
    ])
}

/// Full-grid bound also covers expand/apply/restrict paths. Count original warm
/// columns before rank extraction: a dependent seed still forms a p-by-p Gram.
pub fn worker_payload(rows: usize, columns: usize, warm_columns: usize) -> Result<usize, String> {
    let vectors = sum(&[product(&[24, columns])?, product(&[10, warm_columns])?, 32])?;
    sum(&[
        product(&[16, rows, vectors])?,
        product(&[1024, columns, columns])?,
        product(&[512, warm_columns, warm_columns])?,
    ])
}

/// Both mirror kernels use normal-axis dense matrices; ny^2 cannot be bounded
/// by nx*ny on a thin, tall embedding. Keep that term explicit.
pub fn mirror_payload(nx: usize, ny: usize, kernel: ReducedKernel) -> Result<usize, String> {
    if kernel == ReducedKernel::FullGrid {
        return Ok(0);
    }
    sum(&[product(&[256, nx, ny])?, product(&[32, ny, ny])?])
}

pub fn spectrum_payload(rows: usize, columns: usize) -> Result<usize, String> {
    product(&[sum(&[product(&[16, rows])?, 24])?, columns])
}

pub(crate) fn merge_payload(
    fixed: usize,
    rows: usize,
    requested: usize,
    reconstructed: usize,
) -> Result<usize, String> {
    sum(&[fixed, reconstructed, product(&[rows, requested, 64])?])
}

/// Planning reservation: the configured cap or a conservative fully exhausted
/// finite-space bound, whichever is smaller. No assumption of q-sized prefixes.
pub fn reservation(
    nx: usize,
    ny: usize,
    group_order: usize,
    requested: usize,
    storage: SectorStorage,
    kernel: ReducedKernel,
    limit: usize,
) -> Result<usize, String> {
    let rows = product(&[nx, ny])?;
    // The fully exhausted hypothetical workspace can exceed usize on wasm32
    // even for an ordinary 65² grid. Positive payload overflow means its bound
    // exceeds every representable cap, not that the requested prefix is invalid.
    // Execution still checks its actual setup/prefix before each allocation.
    let finite = (|| -> Result<usize, String> {
        let fixed = sum(&[
            setup_payload(rows, group_order, storage)?,
            spectrum_payload(rows, requested)?,
        ])?;
        let retained = spectrum_payload(rows, rows)?;
        let solve = sum(&[
            fixed,
            retained,
            worker_payload(rows, rows, rows.max(requested))?,
            mirror_payload(nx, ny, kernel)?,
        ])?;
        let merge = merge_payload(fixed, rows, requested, retained)?;
        Ok(solve.max(merge))
    })();
    Ok(finite.map_or(limit, |bound| limit.min(bound)))
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceReport {
    pub model: &'static str,
    pub budget_bytes: usize,
    pub setup_estimated_payload_bytes: usize,
    pub first_round_peak_estimated_payload_bytes: usize,
    pub peak_estimated_payload_bytes: usize,
    pub peak_retained_spectrum_payload_bytes: usize,
}
impl WorkspaceReport {
    pub(crate) fn new(budget_bytes: usize, setup: usize) -> Self {
        Self {
            model: MODEL,
            budget_bytes,
            setup_estimated_payload_bytes: setup,
            first_round_peak_estimated_payload_bytes: 0,
            peak_estimated_payload_bytes: setup,
            peak_retained_spectrum_payload_bytes: 0,
        }
    }
    pub(crate) fn admit(&mut self, stage: &str, round: usize, bytes: usize) -> Result<(), String> {
        self.peak_estimated_payload_bytes = self.peak_estimated_payload_bytes.max(bytes);
        if round == 1 {
            self.first_round_peak_estimated_payload_bytes =
                self.first_round_peak_estimated_payload_bytes.max(bytes);
        }
        if bytes > self.budget_bytes {
            return Err(format!(
                "{stage} at allocation round {round} needs estimated sector payload {bytes}, above budget {} ({MODEL})",
                self.budget_bytes
            ));
        }
        Ok(())
    }
}
