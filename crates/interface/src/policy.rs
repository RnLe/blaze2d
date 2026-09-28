//! Conservative policy selection from measured workflows. Eligibility remains
//! the numerical driver's responsibility; a selected policy can still fall back.
use std::collections::HashSet;
use serde::Serialize;
use crate::{ContinuationPolicy, KReuse, PlannedJob, Precision, Symmetry, SymmetryArithmetic, SymmetryPolicy};

pub const MODEL_VERSION: &str = "public-sweep-v1";

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub model_version: &'static str,
    pub backend: String,
    pub arithmetic: SymmetryArithmetic,
    pub k_reuse: KReuse,
    pub continuation: ContinuationPolicy,
    pub reasons: Vec<&'static str>,
    pub qualification: &'static str,
}

pub fn resolve(job: &PlannedJob, backend: &str) -> (Symmetry, Option<Decision>) {
    let bands = job.resolved.config.bands.as_ref().expect("band policy");
    let mut selected = bands.symmetry.clone();
    if selected.policy == SymmetryPolicy::Manual { return (selected, None); }
    let mut reasons = Vec::new();
    let [nx, ny] = job.resolved.resolution;
    let native_envelope = backend == "cpu" && job.resolved.config.eigensolver.precision == Precision::F64
        && bands.discretization.is_fft_rectangle() && nx == ny
        && (31..=65).contains(&nx) && (31..=65).contains(&ny)
        && (8..=24).contains(&bands.count);
    if native_envelope {
        selected.arithmetic = SymmetryArithmetic::PtReal;
        reasons.push("PT arithmetic is selected within the measured native f64 size/band envelope");
    } else {
        selected.arithmetic = SymmetryArithmetic::SpatialComplex;
        reasons.push("the backend, precision, support or problem size is outside the measured PT policy envelope");
    }
    let mut seen = HashSet::new();
    let duplicates = job.resolved.k_points_fractional.iter().any(|k|
        !seen.insert(k.map(|x| if x == 0.0 { 0 } else { x.to_bits() })));
    if duplicates && selected.max_cache_bytes > 0 && bands.discretization.is_fft_rectangle() {
        selected.k_reuse = KReuse::Identical;
        reasons.push("the requested path contains identical k points; only certified repeated solves are eliminated");
    } else {
        selected.k_reuse = KReuse::Off;
        reasons.push("no admitted repeated-k cache is available for this path");
    }
    selected.continuation = ContinuationPolicy::Off;
    reasons.push("unconditional cross-geometry seeds did not improve the matched public sweep controls");
    reasons.push("sector decomposition, coarse starts and precision changes have no general public-workflow gain");
    let decision = Decision {
        model_version: MODEL_VERSION, backend: backend.into(), arithmetic: selected.arithmetic,
        k_reuse: selected.k_reuse, continuation: selected.continuation, reasons,
        qualification: "each executed symmetry action and representation still requires the driver's finite-operator and fresh residual checks",
    };
    (selected, Some(decision))
}
