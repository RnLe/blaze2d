#!/usr/bin/env python3
"""Blocked PT × unitary-reuse experiment on one immutable interface executable."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from interface_contract import physical_configuration, same_interface_jobs
from run_pairs import file_digest, finite_number, quality_passes

MODES = ("off", "pt", "unitary", "pt-unitary")


def summarize(blocks, declared, resamples=10000):
    valid = [b for b in blocks if len(b) == 4 and all(s.get("valid") for s in b)]
    result = {"completed_blocks": len(blocks), "declared_blocks": declared,
              "valid_blocks": len(valid), "failed_blocks": len(blocks)-len(valid),
              "minimum_inference_blocks": 20,
              "quality_decision": "pass" if len(valid) == declared else "fail"}
    if not valid:
        return result
    vectors = []
    for block in valid:
        times = {r["mode"]: math.log(r["result"]["seconds"]) for r in block}
        a, b, c, d = (times[m] for m in MODES)
        vectors.append([a-b, a-c, a-d, b-d, c-d, d-b-c+a])
    names = ["pt_speedup", "reuse_speedup", "combined_speedup",
             "reuse_after_pt_speedup", "pt_after_reuse_speedup", "interaction_time_factor"]
    rng = random.Random(20260923)
    boots = [[] for _ in names]
    for _ in range(resamples):
        selected = rng.choices(vectors, k=len(vectors))
        for j in range(len(names)):
            boots[j].append(math.exp(statistics.mean(v[j] for v in selected)))
    for j, name in enumerate(names):
        ordered = sorted(boots[j])
        lo, hi = ordered[int(.025*resamples)], ordered[min(resamples-1, int(.975*resamples))]
        value = math.exp(statistics.mean(v[j] for v in vectors))
        result[name] = {"geometric_value": value, "ci95": [lo, hi] if len(valid) >= 20 else None}
        if name != "interaction_time_factor":
            result[name]["latency_decision"] = (
                "quality_failure" if result["quality_decision"] != "pass" else
                "insufficient_samples" if len(valid) < 20 else
                "accepted" if lo > 1.02 else "regression" if hi < 1 else "inconclusive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--case", default="square-star")
    parser.add_argument("--grid", type=int, default=31)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--polarization", choices=["TM", "TE"], default="TM")
    parser.add_argument("--tolerance", type=float, default=1e-14)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--output-contract", choices=["fields", "values"], default="fields")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (min(args.grid, args.bands, args.blocks) <= 0 or args.cpu < 0
            or not math.isfinite(args.tolerance) or args.tolerance <= 0):
        parser.error("positive dimensions, block count and finite tolerance are required")
    manifest = vars(args).copy()
    manifest["output"] = str(args.output)
    manifest.update(timestamp=time.time(), seed=20260923, modes=MODES,
                    binary_sha256=file_digest(args.binary), harness_sha256=file_digest(__file__),
                    contract_sha256=file_digest(Path(__file__).with_name("interface_contract.py")),
                    quality_harness_sha256=file_digest(Path(__file__).with_name("run_pairs.py")),
                    cold="new process/module each sample; OS/hardware caches uncontrolled",
                    scope="same physical jobs and output; randomized four-cell blocks; complete interface timing",
                    interaction="T11*T00/(T10*T01); >1 means slower combined time than multiplying isolated speedups predicts")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    samples, blocks = [], []
    rng = random.Random(20260923)

    def run(mode, index, warmup=False):
        cmd = ["taskset", "-c", str(args.cpu), args.binary, args.case, str(args.grid),
               str(args.bands), args.polarization, mode, str(args.tolerance)]
        rec = {"mode": mode, "block": index, "warmup": warmup, "command": cmd,
               "load_average_before": os.getloadavg(), "valid": False}
        begin = time.perf_counter()
        try:
            process = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=600)
            rec.update(returncode=process.returncode, error=process.stderr[-4000:])
            if process.returncode == 0:
                r = json.loads(process.stdout.strip().splitlines()[-1])
                rec["result"] = r
                fields = args.output_contract == "fields"
                rec["valid"] = (quality_passes(r, args.bands, fields)
                    and physical_configuration(r) is not None
                    and r.get("requested_path_success") is True
                    and r.get("field_output") is fields
                    and r.get("grid") == [args.grid, args.grid]
                    and r.get("bands") == args.bands and r.get("mode") == mode
                    and r.get("polarization") == args.polarization and r.get("tolerance") == args.tolerance
                    and finite_number(r.get("seconds")) and r["seconds"] > 0)
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            rec["error"] = str(e)
        rec.update(process_seconds=time.perf_counter()-begin, load_average_after=os.getloadavg())
        return rec

    def save(summary=None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"manifest": manifest, "samples": samples}
        if summary is not None:
            payload["summary"] = summary
        args.output.write_text(json.dumps(payload, indent=2)+"\n")

    for mode in MODES:
        samples.append(run(mode, -1, warmup=True))
    for index in range(args.blocks):
        modes = list(MODES)
        rng.shuffle(modes)
        block = [run(m, index) for m in modes]
        reference_record = next(r for r in block if r["mode"] == "off")
        reference = reference_record.get("result", {})
        reference_valid = reference_record["valid"]
        for r in block:
            result = r.get("result", {})
            r["same_interface_jobs"] = same_interface_jobs(reference, result)
            r["output_valid"] = r["valid"]
            r["valid"] &= r["same_interface_jobs"] and reference_valid
            if r["valid"]:
                error = max(abs(x-y)/max(1.0, abs(x), abs(y))
                            for a, b in zip(reference["spectrum"], result["spectrum"])
                            for x, y in zip(a, b))
                r["spectral_relative_error"] = error
                r["valid"] &= error <= 1e-6
        samples.extend(block)
        blocks.append(block)
        save()
        if not all(r["valid"] for r in block):
            break
    summary = summarize(blocks, args.blocks)
    summary["failed_warmups"] = sum(not r["valid"] for r in samples if r["warmup"])
    if summary["failed_warmups"]:
        summary["quality_decision"] = "fail"
        for name, value in summary.items():
            if isinstance(value, dict) and "latency_decision" in value:
                value["latency_decision"] = "quality_failure"
    save(summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
