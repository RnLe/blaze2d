#!/usr/bin/env python3
"""Independently audit D04 gates, variational order and profile observations."""
import hashlib
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def certificate(c, values, gamma):
    if not c or not finite(c.get("b_orthogonality_defect")) or c["b_orthogonality_defect"] > 1e-8:
        return False
    residuals, absolute = c.get("residuals", []), c.get("absolute_residuals", [])
    if len(residuals) != len(values) or len(absolute) != len(values):
        return False
    for i, (value, r, a) in enumerate(zip(values, residuals, absolute)):
        if not all(finite(x) for x in (value, r, a)) or min(r, a) < 0:
            return False
        if gamma and i == 0:
            if abs(value) > 1e-7 or a > 1e-7:
                return False
        elif value < 0 or r > 1e-5:
            return False
    return True


def key(r):
    return (r["case"], r["grid"][0], r["bands"], r["polarization"],
            r["method"], r["cutoff_in_2pi_units"], r["material_label"])


def signature(r):
    return {k: r[k] for k in ("spectrum", "report", "support_identity", "physical_reference",
                             "field_payload_bytes", "k", "lattice", "material", "eigensolver")}


def audit_corrections(data):
    """Verify declared replacements against retained attempts, not their flags."""
    names = ("D04-initial-prefix-convergence.json", "D04-subspace-corrections.json",
             "D04-subspace-correction-manifest.json")
    original, corrected, manifest = [json.loads((ROOT / name).read_text()) for name in names]
    issues = []
    old_rows, new_rows = original["records"], data["records"]
    expected = []
    for i, row in enumerate(old_rows):
        r = row["result"]
        s = r["physical_reference"]["field_subspace"]
        if s["candidate_boundary_gap_lambda"] <= 1e-6*max(1, abs(r["spectrum"][s["prefix"]])):
            expected.append(i)
    observed = [row["original_index"] for row in corrected["records"]]
    if (len(old_rows) != len(new_rows) or len(expected) != 41
            or expected != observed or observed != manifest["indices"] or corrected["issues"]):
        issues.append(["correction manifest/selection mismatch"])
        return issues, {}
    replacements = {row["original_index"]: row for row in corrected["records"]}
    def numerical(r):
        ref = copy.deepcopy(r["physical_reference"])
        ref.pop("field_subspace")
        return {"spectrum": r["spectrum"], "report": r["report"],
                "support_identity": r["support_identity"], "physical_reference": ref}
    for i, (old, new) in enumerate(zip(old_rows, new_rows)):
        if i in replacements:
            if (new != replacements[i] or new["source_revision"] != manifest["source_revision"]
                    or numerical(new["result"]) != numerical(old["result"])):
                issues.append([i, "undeclared numerical change in diagnostic correction"])
        else:
            expected_row = dict(old, source_revision=original["source_revision"])
            if new != expected_row:
                issues.append([i, "undeclared replacement of unaffected record"])
    provenance = {"verified_replacements": len(replacements),
        "primary_replacements": sum(not r["profiled"] for r in replacements.values()),
        "profile_replacements": sum(r["profiled"] for r in replacements.values()),
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    return issues, provenance


def main():
    path = ROOT / "D04-convergence.json"
    data = json.loads(path.read_text())
    records = data["records"]
    issues, correction_provenance = audit_corrections(data)
    rows, primaries = [], {}
    profiles = []
    for index, rec in enumerate(records):
        if rec["returncode"] != 0 or "result" not in rec:
            issues.append([index, "process failure", rec.get("stderr")])
            continue
        r = rec["result"]
        k = key(r)
        report, reference = r["report"], r["physical_reference"]
        gamma = r["k"] == [0, 0]
        valid = (report["accepted"] is True and report["solver_converged"] is True
                 and report["analytic_kernel"] is gamma
                 and len(r["spectrum"]) == r["bands"]
                 and certificate(report.get("projected"), r["spectrum"], gamma)
                 and reference["accepted"] is True
                 and len(reference["spectrum"]) == reference["bands"] == r["bands"] + 8
                 and certificate(reference, reference["spectrum"], gamma)
                 and r["field_payload_bytes"] == r["bands"] * r["grid"][0] * r["grid"][1] * 16
                 and report["coefficient_count"] == len(r["support_identity"]["modes"])
                 and report["embedding_points"] == r["grid"][0] * r["grid"][1]
                 and r["support_identity"]["grid"] == r["grid"]
                 and r["discretization"] == "metric_galerkin"
                 and r["eigensolver"]["tol"] == 1e-14 and r["eigensolver"]["max_iter"] == 1000
                 and finite(r["seconds"]) and r["seconds"] > 0)
        if not valid:
            issues.append([index, list(k), "independent quality/shape/configuration gate"])
        space = reference.get("field_subspace")
        guarded = False
        if space:
            prefix = space["prefix"]
            squared = space["squared_singular_values"]
            guarded = (0 < prefix < len(r["spectrum"]) and prefix < len(reference["spectrum"])
                and len(squared) == prefix
                and all(finite(x) and -1e-8 <= x <= 1+1e-8 for x in squared)
                and space["reference_boundary_gap_lambda"] > 1e-6*max(1, abs(reference["spectrum"][prefix]))
                and space["candidate_boundary_gap_lambda"] > 1e-6*max(1, abs(r["spectrum"][prefix])))
        if not guarded:
            issues.append([index, list(k), "subspace prefix cuts a cluster or has invalid overlap"])
        if rec["profiled"]:
            profiles.append((k, r))
            continue
        if k in primaries:
            issues.append([index, "duplicate primary", list(k)])
        primaries[k] = r
        errors = reference["scaled_frequency_errors"]
        max_error = max(errors) if len(errors) == r["bands"] and all(finite(x) for x in errors) else None
        angle = space.get("max_sin_angle") if space else None
        if max_error is None or not finite(angle):
            issues.append([index, list(k), "missing/nonfinite convergence diagnostic"])
        rows.append({"key": list(k), "study": rec["study"], "quality_pass": valid,
                     "coefficients": report["coefficient_count"], "iterations": report["iterations"],
                     "seconds_descriptive": r["seconds"], "peak_rss_kib": r["peak_rss_kib"],
                     "max_scaled_frequency_error": max_error,
                     "max_angular_gap_error": max(reference["angular_gap_absolute_errors"]),
                     "subspace_prefix": space["prefix"] if space else None, "max_subspace_sine": angle,
                     "approximation_target_met": valid and guarded and max_error is not None and max_error <= 1e-3
                        and finite(angle) and angle <= .1,
                     "max_nonzero_full_rectangle_residual": max(report["full_rectangle"]["residuals"][1:] if gamma
                                                                else report["full_rectangle"]["residuals"]),
                     "covariance": r["covariance"]})
    if len(primaries) != 184 or len(profiles) != 16:
        issues.append(["incomplete design", len(primaries), len(profiles)])

    # Full references repeat independently with eight extra guard bands; they
    # must not depend on which projected cutoff/control preceded them.
    references = {}
    for k, r in primaries.items():
        rk = k[:4] + k[-1:]
        sig = {name: r["physical_reference"][name] for name in (
            "spectrum", "iterations", "residuals", "absolute_residuals", "b_orthogonality_defect")}
        if rk in references and references[rk] != sig:
            issues.append([list(rk), "full reference changed across projected methods/cutoffs"])
        references[rk] = sig

    variational = []
    groups = {}
    for k, r in primaries.items():
        if k[4] != "iterative":
            continue
        gk = k[:4] + k[-1:]
        groups.setdefault(gk, []).append(r)
    for gk, rs in groups.items():
        rs.sort(key=lambda r: r["cutoff_in_2pi_units"])
        previous = None
        for r in rs:
            values, control = r["spectrum"], r["physical_reference"]["spectrum"]
            if any(a < b-1e-7*max(1, abs(b)) for a, b in zip(values, control)):
                variational.append([list(gk), r["cutoff_in_2pi_units"], "below full reference"])
            if previous and any(a > b+1e-7*max(1, abs(b)) for a, b in zip(values, previous["spectrum"])):
                variational.append([list(gk), r["cutoff_in_2pi_units"], "nested eigenvalue increased"])
            if previous and not {tuple(g) for g in previous["support_identity"]["modes"]}.issubset(
                    {tuple(g) for g in r["support_identity"]["modes"]}):
                variational.append([list(gk), "supports are not nested"])
            previous = r
    issues.extend(variational)

    controls = []
    for k, r in primaries.items():
        if k[4] not in ("dense", "reference"):
            continue
        parent = primaries[k[:4] + ("iterative",) + k[5:]]
        error = max(abs(x-y)/max(1,abs(x),abs(y)) for x,y in zip(r["spectrum"],parent["spectrum"]))
        same = r["support_identity"] == parent["support_identity"]
        controls.append({"key": list(k), "relative_eigenvalue_error": error, "same_support": same})
        if not same or error > 1e-8:
            issues.append([list(k), "direct/literal/dense disagreement", error])

    observations = []
    for k, r in profiles:
        same = signature(r) == signature(primaries[k])
        c = r["counters"]
        counts = c.get("iterations") == r["report"]["iterations"] and not c.get("profiler.unbalanced_timers", 0)
        observations.append({"key": list(k), "same_signature": same, "iteration_accounting": counts,
            "iterations": c.get("iterations"), "A_columns": c.get("operator_A.columns"),
            "B_columns": c.get("operator_B.columns"),
            "fft_vectors": c.get("fft.forward.vectors", 0) + c.get("fft.inverse.vectors", 0)})
        if not same or not counts:
            issues.append([list(k), "profile numerical/accounting mismatch"])

    result = {"source_revisions": data.get("source_revisions", [data.get("source_revision")]),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "primary_count": len(primaries), "profile_count": len(profiles),
        "correction_provenance": correction_provenance,
        "rows": rows, "method_controls": controls, "profiles": observations,
        "variational_violations": variational, "issues": issues,
        "scope": "Convergence and work diagnostics; no paired latency acceptance or continuum-order claim"}
    (ROOT / "D04-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary": len(primaries), "profiles": len(profiles),
                      "approximation_targets_met": sum(r["approximation_target_met"] for r in rows), "issues": issues}))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
