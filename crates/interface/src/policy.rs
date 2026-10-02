//! Conservative policy selection from measured workflows. Eligibility remains
//! the numerical driver's responsibility; a selected policy can still fall back.
use crate::{
    ContinuationPolicy, KReuse, PlannedJob, Precision, ResolvedConfig, Symmetry, SymmetryArithmetic, SymmetryPolicy,
};
use serde::Serialize;
use std::collections::HashSet;

pub const MODEL_VERSION: &str = "residual-pt-v1";

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub model_version: &'static str,
    pub evidence: &'static [&'static str],
    pub backend: String,
    pub arithmetic: SymmetryArithmetic,
    pub k_reuse: KReuse,
    pub continuation: ContinuationPolicy,
    pub reasons: Vec<&'static str>,
    pub qualification: &'static str,
}

/// Residual-stopping gate, then the calibration kept for eigenvalue-change stopping.
const EVIDENCE: &[&str] = &["crates/interface/tests/pt_envelope.rs", "benchmarks/preconditioner/pt_gate.py",
    "benchmarks/symmetry/public-sweep-auto-v1", "benchmarks/symmetry/public-sweep-metric-v2",
    "benchmarks/symmetry/production-policy-v2"];

/// Residual targets for which the PT gate certified automatic PT arithmetic.
const RESIDUAL_PT_TOLERANCES: std::ops::RangeInclusive<f64> = 1e-8..=1e-6;

fn k_key(k: &[f64; 2]) -> [u64; 2] { k.map(|x| if x == 0.0 { 0 } else { x.to_bits() }) }

/// True when the path visits some k point twice (the repeated-k cache can help).
pub(crate) fn repeated_k_points(points: &[[f64; 2]]) -> bool {
    let mut seen = HashSet::new();
    points.iter().any(|k| !seen.insert(k_key(k)))
}

/// Distinct k points that occur more than once, the most identical reuse caches at once.
pub(crate) fn recurring_k_points(points: &[[f64; 2]]) -> usize {
    let mut counts = std::collections::HashMap::new();
    for k in points { *counts.entry(k_key(k)).or_insert(0usize) += 1; }
    counts.values().filter(|&&n| n > 1).count()
}

/// Pre-check for PT arithmetic: every object has a partner with the same radius and
/// permittivity at the inverted center. Overlaps can break the symmetry of the sampled
/// material behind a symmetric list (the driver's qualification then falls back), and
/// an unpaired object hidden under others makes a symmetric material keep complex
/// arithmetic.
fn inversion_symmetric(objects: &[crate::Object]) -> bool {
    let distance = |x: f64| { let d = x.rem_euclid(1.0); d.min(1.0 - d) };
    objects.iter().all(|o| objects.iter().any(|p| p.kind == o.kind
        && (p.radius - o.radius).abs() <= 1e-12 * o.radius.abs().max(1.0)
        && (p.epsilon - o.epsilon).abs() <= 1e-12 * o.epsilon.abs().max(1.0)
        && o.center.iter().zip(&p.center).all(|(a, b)| distance(a + b) <= 1e-9)))
}

/// Arithmetic and repeated-k reuse that the automatic policy selects for a resolved
/// band configuration on `backend`, with the reasons it reports.
pub(crate) fn select(resolved: &ResolvedConfig, backend: &str) -> (SymmetryArithmetic, KReuse, Vec<&'static str>) {
    let bands = resolved.config.bands.as_ref().expect("band policy");
    let mut reasons = Vec::new();
    let [nx, ny] = resolved.resolution;
    let lattice = resolved.config.geometry.lattice.kind;
    let measured_lattice = matches!(lattice, crate::LatticeKind::Square | crate::LatticeKind::Triangular);
    let fft_rectangle = bands.discretization == crate::Discretization::FftRectangle;
    let support_envelope = match &bands.discretization {
        crate::Discretization::FftRectangle => true,
        crate::Discretization::MetricBall { angular_cutoff, .. } => {
            let a = resolved.lattice_vectors[0][0].hypot(resolved.lattice_vectors[0][1]);
            let ratio = angular_cutoff * a / (std::f64::consts::TAU * nx as f64);
            lattice == crate::LatticeKind::Triangular && (0.40..=0.46).contains(&ratio)
        }
    };
    let eigensolver = &resolved.config.eigensolver;
    let native = backend == "cpu" && eigensolver.precision == Precision::F64 && measured_lattice
        && support_envelope && nx == ny && (8..=16).contains(&bands.count);
    let inversion = inversion_symmetric(&resolved.config.geometry.objects);
    // Residual stopping certifies every attempt directly; the PT gate passed the
    // full 16-128 grid range on FFT rectangles within this tolerance window. Tighter
    // targets approach the PT roundoff floor and end in a discarded PT solve.
    // Eigenvalue-change stopping keeps the narrower calibration: at 1e-6 the first
    // PT attempt consistently failed the certificate.
    let (tolerance_ok, size_ok) = match eigensolver.stopping {
        crate::Stopping::Residual => (
            eigensolver.residual_tolerance.is_some_and(|t| RESIDUAL_PT_TOLERANCES.contains(&t)),
            (16..=128).contains(&nx) && fft_rectangle,
        ),
        _ => (eigensolver.tolerance.is_some_and(|t| t <= 1e-13), (31..=65).contains(&nx)),
    };
    let arithmetic = if native && inversion && tolerance_ok && size_ok {
        reasons.push("PT arithmetic is selected within the calibrated native f64 lattice/size/band/support envelope");
        SymmetryArithmetic::PtReal
    } else {
        reasons.push(if !native {
            "the backend, precision, support or problem size is outside the measured PT policy envelope"
        } else if !inversion {
            "the objects are not inversion-symmetric about the cell origin, so PT arithmetic does not apply"
        } else if !tolerance_ok {
            "PT is not selected at these stopping tolerances: it needs residual stopping between 1e-8 and 1e-6 or eigenvalue-change stopping at 1e-13 or tighter"
        } else if eigensolver.stopping == crate::Stopping::Residual {
            "residual-stopped PT was measured on FFT rectangles with 16-128 samples per axis"
        } else {
            "eigenvalue-change PT was measured on grids with 31-65 samples per axis"
        });
        SymmetryArithmetic::SpatialComplex
    };
    // The metric-support driver reserves the whole cache allowance in its payload budget.
    let cache_fits = bands.discretization.metric_options().is_none_or(|metric| {
        blaze2d_core::drivers::support_bands::driver_payload_bytes(nx * ny, resolved.solved_bands, resolved.k_points_fractional.len())
            .ok().and_then(|bytes| bytes.checked_add(bands.symmetry.max_cache_bytes))
            .is_some_and(|bytes| bytes < metric.budget.max_estimated_payload_bytes)
    });
    let k_reuse = if repeated_k_points(&resolved.k_points_fractional) && bands.symmetry.max_cache_bytes > 0 && cache_fits {
        reasons.push("the requested path contains identical k points; only certified repeated solves are eliminated");
        KReuse::Identical
    } else {
        reasons.push(if cache_fits { "no admitted repeated-k cache is available for this path" }
            else { "the repeated-k cache does not fit the metric-support payload allowance" });
        KReuse::Off
    };
    (arithmetic, k_reuse, reasons)
}

pub fn resolve(job: &PlannedJob, backend: &str) -> (Symmetry, Option<Decision>) {
    let bands = job.resolved.config.bands.as_ref().expect("band policy");
    let mut selected = bands.symmetry.clone();
    if selected.policy != Some(SymmetryPolicy::Auto) {
        return (selected, None);
    }
    let (arithmetic, k_reuse, mut reasons) = select(&job.resolved, backend);
    selected.arithmetic = arithmetic;
    selected.k_reuse = k_reuse;
    selected.continuation = ContinuationPolicy::Off;
    reasons.push("full-path continuation did not improve matched sweeps; first-point continuation also regressed and can trigger costly PT fallback");
    reasons.push("sector/coarse/continuation combinations regressed in the matched triangular sweeps; precision changes remain explicit experiments");
    let decision = Decision {
        model_version: MODEL_VERSION, evidence: EVIDENCE, backend: backend.into(), arithmetic: selected.arithmetic,
        k_reuse: selected.k_reuse, continuation: selected.continuation, reasons,
        qualification: "each executed symmetry action and representation still requires the driver's finite-operator and fresh residual checks",
    };
    (selected, Some(decision))
}
