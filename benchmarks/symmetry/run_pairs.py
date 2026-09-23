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
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = random.Random(20260923)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    manifest = vars(args).copy()
    manifest["output"] = str(args.output)
    manifest.update(seed=20260923, timestamp=time.time(), cold="numerical and process; OS/hardware caches uncontrolled", field_output=True,
                    baseline_sha256=hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
                    candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest())
    samples = []

    def run(label, warmup=False):
        binary = getattr(args, label)
        mode = getattr(args, label + "_mode")
        cmd = ["taskset", "-c", str(args.cpu), binary, args.case, str(args.grid), str(args.bands), args.polarization, mode]
        t = time.perf_counter()
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=600)
            record = dict(label=label, warmup=warmup, command=cmd, process_seconds=time.perf_counter()-t, returncode=result.returncode)
            if result.returncode:
                record.update(error=result.stderr[-4000:], valid=False)
            else:
                record["result"] = json.loads(result.stdout.strip().splitlines()[-1])
                r = record["result"]
                residuals = [v for x in r["points"] for v in x["residuals"][1 if x["k"] == [0.0,0.0] else 0:]]
                absolute = [v for x in r["points"] for v in x["absolute_residuals"]]
                record["valid"] = (all(x["converged"] and x["field_count"] == args.bands for x in r["points"])
                    and all(math.isfinite(v) for row in r["spectrum"] for v in row)
                    and max(residuals, default=0) <= 1e-3
                    and max(absolute, default=0) <= 1e-2
                    and max(x["b_orthogonality_defect"] for x in r["points"]) <= 1e-6)
        except (subprocess.TimeoutExpired, ValueError) as e:
            record = dict(label=label, warmup=warmup, valid=False, error=str(e))
        return record

    for label in ("baseline", "candidate"):
        samples.append(dict(pair=-1, **run(label, warmup=True)))
    for pair in range(args.pairs):
        labels = ["baseline", "candidate"]
        rng.shuffle(labels)
        records = [dict(pair=pair, **run(label)) for label in labels]
        if all("result" in r for r in records):
            a, b = [r["result"]["spectrum"] for r in records]
            error = max((abs(x-y)/max(1.0,abs(x),abs(y)) for ra,rb in zip(a,b) for x,y in zip(sorted(ra),sorted(rb))), default=math.inf)
            for r in records:
                r.update(spectral_relative_error=error)
                r["valid"] &= error <= 1e-4 and len(a) == len(b) and all(len(row) == args.bands for row in a+b)
        samples.extend(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(manifest=manifest,samples=samples),indent=2)+"\n")

    pairs = [[r for r in samples if r["pair"] == i] for i in range(args.pairs)]
    valid = [pair for pair in pairs if len(pair) == 2 and all(r["valid"] for r in pair)]
    logs = []
    for pair in valid:
        d = {r["label"]: r["result"]["seconds"] for r in pair}
        logs.append(math.log(d["baseline"]/d["candidate"]))
    summary = dict(valid_pairs=len(valid),failed_pairs=len(pairs)-len(valid),declared_pairs=args.pairs,minimum_practical_speedup=1.02)
    if logs:
        boot = sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
        summary.update(geometric_speedup=math.exp(statistics.mean(logs)), ci95=[boot[250],boot[9749]])
        summary["latency_decision"] = "accepted" if len(valid)==len(pairs) and boot[250]>1.02 else "regression" if boot[9749]<1 else "inconclusive"
    args.output.write_text(json.dumps(dict(manifest=manifest,summary=summary,samples=samples),indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
