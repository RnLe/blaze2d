#!/usr/bin/env python3
"""Fixed-trial Gram/Ritz comparison; setup and validation are outside the timer."""
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


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--grid", type=int, default=31)
    parser.add_argument("--polarization", choices=["TE", "TM"], default="TM")
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--group", choices=["pm", "p4", "p4m"], default="p4m")
    parser.add_argument("--stage", choices=["gram", "ritz"], default="gram")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = dict(vars(args), output=str(args.output), seed=20260923,
                    timestamp=time.time(), binary_sha256=digest(args.binary),
                    harness_sha256=digest(__file__), load_average=os.getloadavg(),
                    scope="fixed-trial Gram/Ritz; setup and validation excluded", cpu=0)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    rng = random.Random(20260923)
    samples = []
    logs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pair in range(args.pairs):
        order = ["full", "blocks"]
        rng.shuffle(order)
        records = []
        for mode in order:
            command = ["taskset", "-c", "0", args.binary, str(args.grid), str(args.columns), args.polarization, args.group,
                       args.stage, mode, str(args.repetitions)]
            record = dict(pair=pair, mode=mode, command=command, load_average_before=os.getloadavg())
            try:
                process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
                record["returncode"] = process.returncode
                if process.returncode:
                    record.update(valid=False, error=process.stderr[-4000:])
                else:
                    result = json.loads(process.stdout.strip().splitlines()[-1])
                    record.update(result=result, valid=result["valid"] and result["gram_relative_error"] < 2e-11
                                  and result["spectral_relative_error"] < 1e-9 and result["cluster_span_error"] < 1e-8
                                  and math.isfinite(result["seconds"]) and result["seconds"] > 0)
            except (ValueError, subprocess.TimeoutExpired) as error:
                record.update(valid=False, error=str(error))
            record["load_average_after"] = os.getloadavg()
            samples.append(record)
            records.append(record)
        if all(record["valid"] for record in records):
            times = {record["mode"]: record["result"]["seconds"] for record in records}
            logs.append(math.log(times["full"] / times["blocks"]))
        args.output.write_text(json.dumps(dict(manifest=manifest, samples=samples), indent=2) + "\n")
    summary = dict(valid_pairs=len(logs), failed_pairs=args.pairs-len(logs), minimum_practical_speedup=1.02)
    if logs:
        boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(10000))
        summary.update(geometric_speedup=math.exp(statistics.mean(logs)), ci95=[boot[250], boot[9749]],
                       decision="quality_failure" if len(logs) != args.pairs else "accepted" if boot[250] > 1.02
                       else "regression" if boot[9749] < 1 else "inconclusive")
    else:
        summary["decision"] = "quality_failure"
    args.output.write_text(json.dumps(dict(manifest=manifest, summary=summary, samples=samples), indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
