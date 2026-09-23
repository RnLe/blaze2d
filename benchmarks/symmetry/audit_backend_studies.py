#!/usr/bin/env python3
"""Audit the complete L08 primary matrix against its separate work profiles."""
import hashlib
import json
from pathlib import Path
import statistics

from interface_contract import same_interface_jobs
from run_pairs import quality_passes

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(result):
    # Field values are not retained in these JSON records. Compare all stored
    # numerical certificates and decisions, not a claim of bitwise field identity.
    keys = ("grid", "bands", "polarization", "precision", "tolerance", "mode",
            "configuration", "expected_points", "field_output", "arrays",
            "spectrum", "points", "reused_points", "requested_path_success")
    return {k: result[k] for k in keys}


def main():
    manifest = json.loads((ROOT / "L08-study-manifest.json").read_text())
    issues, studies = [], []
    references, primary_times = {}, {}
    primary_count = warmup_count = 0
    sources = {}

    def check(ok, detail):
        if not ok:
            issues.append(detail)

    def record(backend, sample, name):
        nonlocal primary_count, warmup_count
        r = sample.get("result", {})
        check(sample.get("valid") is True, [name, "invalid sample"])
        check(quality_passes(r, 8, r.get("field_output", True)), [name, "physical/output gate"])
        check(r.get("requested_path_success") is True, [name, "requested path"])
        check(r.get("build", {}).get("source_revision") == manifest["source_revision"],
              [name, "numerical source revision"])
        if not r:
            return
        key = backend, r["case"], r["polarization"], r["mode"]
        sig = signature(r)
        if key in references:
            check(sig == references[key], [name, list(key), "numerical repeat changed"])
        else:
            references[key] = sig
        if sample["warmup"]:
            warmup_count += 1
        else:
            primary_count += 1
            primary_times.setdefault(key, []).append(r["seconds"])

    for a, b, case, pol, am, bm, name in manifest["paired_studies"]:
        path = ROOT / name
        d = json.loads(path.read_text())
        sources[name] = digest(path)
        s = d["summary"]
        check(s["valid_pairs"] == 20 and s["failed_pairs"] == 0, [name, "pair count"])
        check(d["manifest"]["harness_sha256"] == digest(ROOT / "run_pairs.py"),
              [name, "paired harness changed"])
        by_pair = {}
        for sample in d["samples"]:
            record(a if sample["label"] == "baseline" else b, sample, name)
            if not sample["warmup"]:
                by_pair.setdefault(sample["pair"], []).append(sample)
        for pair, samples in by_pair.items():
            check(len(samples) == 2 and same_interface_jobs(samples[0]["result"], samples[1]["result"]),
                  [name, pair, "interface jobs"])
        studies.append({"file": name, "summary": s})

    for backend, case, pol, name in manifest["factorial_studies"]:
        path = ROOT / name
        d = json.loads(path.read_text())
        sources[name] = digest(path)
        s = d["summary"]
        check(s["valid_blocks"] == 20 and s["failed_blocks"] == 0, [name, "block count"])
        check(d["manifest"]["harness_sha256"] == digest(ROOT / "run_factorial.py"),
              [name, "factorial harness changed"])
        by_block = {}
        for sample in d["samples"]:
            record(backend, sample, name)
            if not sample["warmup"]:
                by_block.setdefault(sample["block"], []).append(sample)
        for block, samples in by_block.items():
            check(len(samples) == 4 and {x["mode"] for x in samples} == {"off", "pt", "unitary", "pt-unitary"},
                  [name, block, "factorial cells"])
            check(all(same_interface_jobs(samples[0]["result"], x["result"]) for x in samples),
                  [name, block, "interface jobs"])
        studies.append({"file": name, "summary": s})

    profiles_file = ROOT / "L08-work-counters.json"
    sources[profiles_file.name] = digest(profiles_file)
    profiles, observed = [], set()
    for rec in json.loads(profiles_file.read_text())["records"]:
        r = rec.get("result", {})
        check(rec["returncode"] == 0, [rec["command"], "profile process"])
        if not r:
            continue
        key = rec["backend"], r["case"], r["polarization"], r["mode"]
        check(key not in observed, [list(key), "duplicate profile"])
        observed.add(key)
        same = references.get(key) == signature(r)
        check(same, [list(key), "profile numerical observer effect"])
        counters = r["counters"]
        iterations = sum(p["symmetry"]["attempted_iterations"] for p in r["points"])
        check(counters.get("iterations") == iterations, [list(key), "iteration accounting"])
        check(r["counters_available"] and not counters.get("profiler.unbalanced_timers", 0),
              [list(key), "missing/unbalanced profile"])
        check(quality_passes(r, 8, r["field_output"]), [list(key), "profile quality"])
        profiles.append({"backend": key[0], "case": key[1], "polarization": key[2], "mode": key[3],
            "same_numerical_signature": same, "iterations": iterations,
            "solves": counters.get("eigensolve.calls", 0), "reused_points": r["reused_points"],
            "A_columns": counters.get("operator_A.columns", 0),
            "B_columns": counters.get("operator_B.columns", 0),
            "fft_vectors": counters.get("fft.forward.vectors", 0) + counters.get("fft.inverse.vectors", 0),
            "profile_seconds": r["seconds"],
            "unprofiled_median_seconds": statistics.median(primary_times[key]),
            "profile_to_primary_median_time": r["seconds"] / statistics.median(primary_times[key])})
    check(observed == set(references), ["profile coverage", len(observed), len(references)])
    check(primary_count == 1760 and warmup_count == 88, ["sample totals", primary_count, warmup_count])
    result = {"scope": "Post-run audit, no new numerical runs; profile latency is descriptive and unpaired",
        "source_revision": manifest["source_revision"], "primary_samples": primary_count,
        "warmup_samples": warmup_count, "paired_studies": len(manifest["paired_studies"]),
        "factorial_studies": len(manifest["factorial_studies"]), "profiles": profiles,
        "studies": studies, "dataset_sha256": sources, "issues": issues}
    (ROOT / "L08-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary_samples": primary_count, "warmups": warmup_count,
                      "profiles": len(profiles), "issues": issues}))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
