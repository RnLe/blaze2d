#!/usr/bin/env python3
"""Randomized paired process-cold benchmarks; all samples, including failures, survive."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

def file_digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def spectrum_passes(result, bands):
    rows = result.get("spectrum")
    points = result.get("points")
    return (isinstance(rows, list) and isinstance(points, list) and len(rows) == len(points) > 0
            and all(isinstance(row, list) and len(row) == bands
                    and all(finite_number(v) and v >= 0 for v in row) for row in rows))

def quality_passes(result, bands):
    if not spectrum_passes(result, bands):
        return False
    points = result["points"]
    if len(points) != result.get("expected_points", len(points)):
        return False
    for point in points:
        residuals = point.get("residuals")
        absolute = point.get("absolute_residuals")
        k = point.get("k")
        orthogonality = point.get("b_orthogonality_defect")
        if (point.get("converged") is not True or point.get("field_count") != bands
                or not isinstance(k, list) or len(k) != 2 or not all(finite_number(v) for v in k)
                or not isinstance(residuals, list) or len(residuals) != bands
                or not all(finite_number(v) and v >= 0 for v in residuals)
                or not finite_number(orthogonality) or not 0 <= orthogonality <= 1e-8):
            return False
        gamma = k == [0.0, 0.0]
        if any(v > 1e-5 for v in residuals[int(gamma):]):
            return False
        # Frozen non-Gamma seed snapshots do not store an absolute-residual
        # array in this harness. Gamma always requires the absolute certificate.
        if absolute != [] or gamma:
            if (not isinstance(absolute, list) or len(absolute) != bands
                    or not all(finite_number(v) and v >= 0 for v in absolute)
                    or (gamma and absolute[0] > 1e-7)):
                return False
    return True

def parameter_job_identity(result):
    """Ordered physical parameter-job labels, or None for malformed output."""
    points = result.get("points")
    if not isinstance(points, list) or not points:
        return None
    identity, ids = [], set()
    for index, point in enumerate(points):
        job = point.get("job", {})
        if not isinstance(job, dict):
            return None
        label, parameters = job.get("id"), job.get("parameters")
        if (not isinstance(label, str) or not label or label in ids
                or type(job.get("original_index")) is not int or job["original_index"] != index
                or not isinstance(parameters, list) or not parameters
                or not all(finite_number(v) for v in parameters)):
            return None
        ids.add(label)
        identity.append((label, parameters))
    return identity

def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--baseline-mode", default="off")
    p.add_argument("--candidate-mode", default="off")
    p.add_argument("--case", default="square-path")
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--polarization", choices=["TE", "TM"], default="TM")
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--cpus", help="Common Linux CPU affinity list for both controls, e.g. 0-3; overrides --cpu")
    p.add_argument("--tolerance", type=float, help="Common stopping tolerance; requires a binary supporting the seventh argument")
    p.add_argument("--stop-on-quality-failure", action="store_true", help="Stop after a failed completed pair; retain it and make no acceptance claim")
    p.add_argument("--require-sector-success", action="store_true", help="Require sector output at every point; a correct unrestricted fallback is not a sector-stage acceptance")
    p.add_argument("--require-requested-path", action="store_true", help="Require an explicit requested_path_success report; fallback output alone cannot accept the feature")
    p.add_argument("--require-job-identity", action="store_true", help="Require identical ordered job IDs and parameter coordinates in paired sweep outputs")
    p.add_argument("--snapshot", help="Use a frozen seed input instead of a k-path")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.tolerance is not None and (args.snapshot or not math.isfinite(args.tolerance) or args.tolerance<=0):
        p.error("a finite positive --tolerance is supported only for path runs")
    rng = random.Random(20260923)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    manifest = vars(args).copy()
    manifest["output"] = str(args.output)
    manifest.update(seed=20260923, timestamp=time.time(), cold="numerical and process; OS/hardware caches uncontrolled", field_output=True,
                    load_average=os.getloadavg(), harness_sha256=file_digest(__file__),
                    baseline_sha256=file_digest(args.baseline),
                    candidate_sha256=file_digest(args.candidate))
    if args.snapshot:
        manifest["snapshot_sha256"] = file_digest(args.snapshot)
    samples = []

    def run(label, warmup=False):
        binary = getattr(args, label)
        mode = getattr(args, label + "_mode")
        cmd = ["taskset", "-c", args.cpus or str(args.cpu), binary, args.case, str(args.grid), str(args.bands), args.polarization, mode]
        if args.tolerance is not None:
            cmd.append(str(args.tolerance))
        if args.snapshot:
            cmd = ["taskset", "-c", args.cpus or str(args.cpu), binary, mode, args.snapshot, str(args.grid), str(args.bands), args.polarization]
        t = time.perf_counter()
        load_before = os.getloadavg()
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=600)
            record = dict(label=label, warmup=warmup, command=cmd, process_seconds=time.perf_counter()-t, returncode=result.returncode,
                          load_average_before=load_before, load_average_after=os.getloadavg())
            if result.returncode:
                record.update(error=result.stderr[-4000:], valid=False)
            else:
                record["result"] = json.loads(result.stdout.strip().splitlines()[-1])
                r = record["result"]
                if args.tolerance is not None and r.get("tolerance") != args.tolerance:
                    raise ValueError("binary did not report the requested common stopping tolerance")
                if args.snapshot:
                    r["spectrum"] = [r["eigenvalues"]]
                    r["points"] = [dict(k=r["target_k"], residuals=r["residuals"], absolute_residuals=[], b_orthogonality_defect=r["orthogonality"], converged=r["converged"], field_count=len(r["eigenvalues"]))]
                record["spectrum_valid"] = spectrum_passes(r, args.bands)
                record["valid"] = quality_passes(r, args.bands)
                record["output_valid"] = record["valid"]
                if args.require_job_identity:
                    record["job_identity_valid"] = parameter_job_identity(r) is not None
                    record["valid"] &= record["job_identity_valid"]
                if args.require_requested_path:
                    record["requested_path_valid"] = r.get("requested_path_success") is True
                    record["valid"] &= record["requested_path_valid"]
                if args.require_sector_success:
                    record["requested_path_valid"] = all(x.get("symmetry", {}).get("sectors") is not None for x in r["points"])
                    record["valid"] &= record["requested_path_valid"]
        except (subprocess.TimeoutExpired, ValueError) as e:
            record = dict(label=label, warmup=warmup, valid=False, error=str(e))
        return record

    for label in ("baseline", "candidate"):
        samples.append(dict(pair=-1, **run(label, warmup=True)))
    for pair in range(args.pairs):
        labels = ["baseline", "candidate"]
        rng.shuffle(labels)
        records = [dict(pair=pair, **run(label)) for label in labels]
        if args.require_job_identity:
            identities = [parameter_job_identity(r.get("result", {})) for r in records]
            same_jobs = identities[0] is not None and identities[0] == identities[1]
            for r in records:
                r["same_ordered_jobs"] = same_jobs
                r["valid"] &= same_jobs
        if all(r.get("spectrum_valid") for r in records):
            a, b = [r["result"]["spectrum"] for r in records]
            error = max((abs(x-y)/max(1.0,abs(x),abs(y)) for ra,rb in zip(a,b) for x,y in zip(sorted(ra),sorted(rb))), default=math.inf)
            for r in records:
                r.update(spectral_relative_error=error)
                r["valid"] &= error <= 1e-6 and len(a) == len(b) and all(len(row) == args.bands for row in a+b)
        samples.extend(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(manifest=manifest,samples=samples),indent=2)+"\n")
        if args.stop_on_quality_failure and not all(r["valid"] for r in records):
            break

    completed = len([r for r in samples if not r["warmup"]]) // 2
    pairs = [[r for r in samples if r["pair"] == i] for i in range(completed)]
    valid = [pair for pair in pairs if len(pair) == 2 and all(r["valid"] for r in pair)]
    logs = []
    for pair in valid:
        d = {r["label"]: r["result"]["seconds"] for r in pair}
        logs.append(math.log(d["baseline"]/d["candidate"]))
    summary = dict(valid_pairs=len(valid),failed_pairs=len(pairs)-len(valid),declared_pairs=args.pairs,completed_pairs=completed,early_stopped=completed<args.pairs,minimum_practical_speedup=1.02,
                   quality_decision="pass" if len(valid)==args.pairs else "fail" if len(valid)<completed else "incomplete")
    if logs:
        boot = sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
        summary.update(geometric_speedup=math.exp(statistics.mean(logs)), ci95=[boot[250],boot[9749]])
        summary["latency_decision"] = "quality_failure" if len(valid)<completed else "incomplete" if completed<args.pairs else "accepted" if boot[250]>1.02 else "regression" if boot[9749]<1 else "inconclusive"
    else:
        summary["latency_decision"] = "quality_failure"
    args.output.write_text(json.dumps(dict(manifest=manifest,summary=summary,samples=samples),indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
