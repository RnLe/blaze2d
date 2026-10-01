//! One acceptance contract for final band outputs, including fallback results.

use crate::{BandAccuracy, Diagnostic, ResultRecord};
use serde_json::json;

/// Record the acceptance decision without changing fields, ordering, solver
/// tolerances or the finite pencil. Call after validating the exported arrays.
pub fn assess(result: &mut ResultRecord, policy: Option<&BandAccuracy>) -> Option<Diagnostic> {
    let Some(policy) = policy else {
        result.metadata["accuracy"] = json!({"status":"not_requested", "accepted":null});
        return None;
    };
    let mut first = None;
    let mut failed_residuals = 0;
    let mut failed_blocks = 0;
    let mut absolute_acceptances = 0;
    let mut relative_acceptances = 0;
    let mut fail = |point: Option<usize>, band: Option<usize>, reason: &str| {
        if first.is_none() {
            first = Some(json!({"k_index":point, "band_index":band, "reason":reason}));
        }
    };
    if result.metadata["converged"] != true {
        let reason = if result.metadata["stopping_criterion"] == "relative_residual" {
            "residual stopping did not converge at every requested point"
        } else {
            "eigenvalue-change stopping did not converge at every requested point"
        };
        fail(None, None, reason);
    }
    let frequencies = result.arrays.get("frequencies");
    let relative = result.arrays.get("residuals");
    let absolute = result.arrays.get("absolute_residuals");
    match (frequencies, relative, absolute) {
        (Some(f), Some(r), Some(a))
            if f.shape.len() == 2
                && f.shape.iter().all(|&n| n > 0)
                && r.shape == f.shape
                && a.shape == f.shape
                && f.shape[0].checked_mul(f.shape[1]) == Some(r.data.len())
                && a.data.len() == r.data.len() =>
        {
            for (index, (&r, &a)) in r.data.iter().zip(&a.data).enumerate() {
                let relative_ok = r.is_finite() && r >= 0.0 && r <= policy.max_relative_residual;
                let absolute_ok = a.is_finite() && a >= 0.0 && a <= policy.max_absolute_residual;
                if !r.is_finite()
                    || !a.is_finite()
                    || r < 0.0
                    || a < 0.0
                    || !(relative_ok || absolute_ok)
                {
                    failed_residuals += 1;
                    fail(
                        Some(index / f.shape[1]),
                        Some(index % f.shape[1]),
                        "neither the absolute nor the relative residual meets the acceptance bound",
                    );
                } else if relative_ok {
                    relative_acceptances += 1;
                } else {
                    absolute_acceptances += 1;
                }
            }
            match result.metadata["certification"]["b_orthogonality_per_point"].as_array() {
                Some(values) if values.len() == f.shape[0] => {
                    for (point, value) in values.iter().enumerate() {
                        if !value.as_f64().is_some_and(|v| {
                            v.is_finite() && v >= 0.0 && v <= policy.max_orthogonality_defect
                        }) {
                            failed_blocks += 1;
                            fail(
                                Some(point),
                                None,
                                "B-orthogonality does not meet the acceptance bound",
                            );
                        }
                    }
                }
                _ => fail(None, None, "missing per-point B-orthogonality certificate"),
            }
        }
        _ => fail(
            None,
            None,
            "missing or incompatible physical residual arrays",
        ),
    }
    let accepted = first.is_none();
    result.metadata["accuracy"] = json!({
        "status":if accepted {"accepted"} else {"rejected"}, "accepted":accepted,
        "policy":policy, "residual_rule":"absolute <= max_absolute_residual OR relative <= max_relative_residual",
        "absolute_residual_unit":"inverse_reference_length_squared",
        "requires_convergence":true, "failed_residuals":failed_residuals,
        "failed_orthogonality_blocks":failed_blocks, "absolute_only_acceptances":absolute_acceptances,
        "relative_acceptances":relative_acceptances, "first_violation":first,
        "scope":result.metadata["certification"].get("pencil").and_then(|v|v.as_str()).unwrap_or("full_fft_rectangle"),
        "interpretation":"acceptance of the selected finite pencil; not a continuum error or spectral completeness bound"
    });
    first.map(|violation| Diagnostic::new("accuracy_gate", "bands.accuracy",
        format!("Band accuracy acceptance failed: {} (k index {}, band index {}); see metadata.accuracy",
                violation["reason"].as_str().unwrap_or("invalid certificate"),
                violation["k_index"], violation["band_index"])))
}
