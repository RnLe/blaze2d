//! Conservative policy selection from measured workflows. Eligibility remains
//! the numerical driver's responsibility; a selected policy can still fall back.
use crate::{
    ContinuationPolicy, KReuse, PlannedJob, Precision, Symmetry, SymmetryArithmetic, SymmetryPolicy,
};
use serde::Serialize;
use std::collections::HashSet;

pub const MODEL_VERSION: &str = "public-sweep-v3";

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub model_version: &'static str,
    pub evidence: [&'static str; 3],
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
    if selected.policy == SymmetryPolicy::Manual {
        return (selected, None);
    }
    let mut reasons = Vec::new();
    let [nx, ny] = job.resolved.resolution;
    let measured_lattice = matches!(
        job.resolved.config.geometry.lattice.kind,
        crate::LatticeKind::Square | crate::LatticeKind::Triangular
    );
    let support_envelope = match &bands.discretization {
        crate::Discretization::FftRectangle => true,
        crate::Discretization::MetricBall { angular_cutoff, .. } => {
            let a = job.resolved.lattice_vectors[0][0].hypot(job.resolved.lattice_vectors[0][1]);
            let ratio = angular_cutoff * a / (std::f64::consts::TAU * nx as f64);
            job.resolved.config.geometry.lattice.kind == crate::LatticeKind::Triangular
                && (0.40..=0.46).contains(&ratio)
        }
    };
    let native_envelope = backend == "cpu"
        && job.resolved.config.eigensolver.precision == Precision::F64
        && measured_lattice
        && support_envelope
        && nx == ny
        && (31..=65).contains(&nx)
        && (8..=16).contains(&bands.count);
    // At the ordinary 1e-6 stopping tolerance the first PT attempt consistently
    // failed the physical certificate, adding a discarded solve per geometry.
    let calibrated_stopping = job.resolved.config.eigensolver.tolerance.is_some_and(|t| t <= 1e-13);
    if native_envelope && calibrated_stopping {
        selected.arithmetic = SymmetryArithmetic::PtReal;
        reasons.push("PT arithmetic is selected within the calibrated native f64 lattice/size/band/support envelope");
    } else {
        selected.arithmetic = SymmetryArithmetic::SpatialComplex;
        reasons.push(if !calibrated_stopping {
            "PT is not selected at stopping tolerances above 1e-13: default-tolerance sweeps paid for rejected PT solves without retaining PT output"
        } else {
            "the backend, precision, support or problem size is outside the measured PT policy envelope"
        });
    }
    let mut seen = HashSet::new();
    let duplicates = job
        .resolved
        .k_points_fractional
        .iter()
        .any(|k| !seen.insert(k.map(|x| if x == 0.0 { 0 } else { x.to_bits() })));
    if duplicates && selected.max_cache_bytes > 0 {
        selected.k_reuse = KReuse::Identical;
        reasons.push("the requested path contains identical k points; only certified repeated solves are eliminated");
    } else {
        selected.k_reuse = KReuse::Off;
        reasons.push("no admitted repeated-k cache is available for this path");
    }
    selected.continuation = ContinuationPolicy::Off;
    reasons.push("full-path continuation did not improve matched sweeps; first-point continuation also regressed and can trigger costly PT fallback");
    reasons.push("sector/coarse/continuation combinations regressed in the matched triangular sweeps; precision changes remain explicit experiments");
    let decision = Decision {
        model_version: MODEL_VERSION, evidence: ["benchmarks/symmetry/public-sweep-auto-v1", "benchmarks/symmetry/public-sweep-metric-v2", "benchmarks/symmetry/production-policy-v2"], backend: backend.into(), arithmetic: selected.arithmetic,
        k_reuse: selected.k_reuse, continuation: selected.continuation, reasons,
        qualification: "each executed symmetry action and representation still requires the driver's finite-operator and fresh residual checks",
    };
    (selected, Some(decision))
}
