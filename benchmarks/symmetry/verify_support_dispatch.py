#!/usr/bin/env python3
"""Replay exact D05 fields and compare frozen binaries after generic dispatch."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "benchmarks/symmetry/D05"
SELECTED = [f"heldout-{case}-{pol}" for case in ("hex-gamma-c6v", "glide-pgg") for pol in ("TM", "TE")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "diff", "HEAD", "--exit-code"], cwd=ROOT, capture_output=True, check=True)
    parent = json.loads((PARENT / "manifest.json").read_text())
    build = json.loads(args.build_manifest.read_text())
    assert build["source_revision"] == revision
    digests = {key: sha(getattr(args, key)) for key in ("baseline", "candidate")}
    assert digests["baseline"] == parent["builds"]["artifacts"]["unprofiled"]["sha256"]
    assert digests["candidate"] == build["artifacts"]["unprofiled"]["sha256"]
    archived = [json.loads(line) for line in (PARENT / "records.jsonl").open()]
    signatures, timed = {}, {}
    for row in archived:
        if row["phase"] not in ("primary", "negative"):
            continue
        key = tuple(row["command"][4:])
        if key in signatures:
            assert signatures[key]["numerics_sha256"] == row["numerics_sha256"]
        signatures[key] = row
        if row["study"] in SELECTED and row["label"] == "candidate":
            timed[row["study"]] = row
    assert set(timed) == set(SELECTED)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "objects").mkdir()
    manifest = dict(schema="blaze2d/support-dispatch/1", source_revision=revision,
                    builds=build, binary_sha256=digests, parent="../D05", pairs=20,
                    parent_manifest_sha256=sha(PARENT / "manifest.json"),
                    parent_records_sha256=sha(PARENT / "records.jsonl"),
                    protocol_sha256=sha(ROOT / "benchmarks/symmetry/dispatch-protocol.md"),
                    runner_sha256=sha(Path(__file__)), seed=2026092401, cpu=0,
                    selected=SELECTED, started=time.time())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    manifest["thread_environment"] = {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []

    def run(parent_row, label, pair, phase):
        command = ["taskset", "-c", "0", str(getattr(args, label).resolve()), *parent_row["command"][4:]]
        row = dict(study=parent_row["study"], label=label, pair=pair, phase=phase, command=command,
                   expected_numerics_sha256=parent_row["numerics_sha256"], timestamp=time.time(), load_before=os.getloadavg())
        start = time.perf_counter()
        try:
            p = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, timeout=600)
            row.update(returncode=p.returncode, stderr=p.stderr.decode(errors="replace"))
            if p.returncode:
                row["stdout"] = p.stdout.decode(errors="replace")
            else:
                parsed = json.loads(p.stdout)
                data = canonical(parsed["numerics"])
                digest = hashlib.sha256(data).hexdigest()
                row.update(numerics_sha256=digest, exact=digest == row["expected_numerics_sha256"],
                           measurement=parsed["measurement"])
                if not (PARENT / "objects" / (digest + ".json.gz")).exists():
                    (args.output / "objects" / (digest + ".json.gz")).write_bytes(gzip.compress(data, mtime=0))
                    row["object_source"] = "objects"
                else:
                    row["object_source"] = "../D05/objects"
        except (subprocess.TimeoutExpired, ValueError, KeyError) as error:
            row.update(returncode=-1, error=str(error))
            if isinstance(error, subprocess.TimeoutExpired):
                row.update(stdout=(error.stdout or b"").decode(errors="replace"), stderr=(error.stderr or b"").decode(errors="replace"))
        row.update(process_seconds=time.perf_counter() - start, load_after=os.getloadavg())
        with (args.output / "records.jsonl").open("a") as f:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        records.append(row)

    rng = random.Random(manifest["seed"])
    for name in SELECTED:
        for pair in range(-1, 20):
            labels = ["baseline", "candidate"]
            rng.shuffle(labels)
            for label in labels:
                run(timed[name], label, pair, "warmup" if pair < 0 else "primary")
        print("PRIMARY", name, len(records), flush=True)
    for row in signatures.values():
        run(row, "candidate", 0, "replay")
    issues = [dict(study=r["study"], phase=r["phase"], label=r["label"], pair=r["pair"])
              for r in records if not r.get("exact", False)]
    for row in records:
        m = row.get("measurement", {})
        if m and (m["counters"] or m["profile"] or m["stage_counters"] or m["references"] is not None
                  or not np.isfinite(m["seconds"]) or m["seconds"] <= 0):
            issues.append(dict(study=row["study"], issue="invalid primary measurement contract"))
    comparisons = []
    bootstrap = np.random.default_rng(2026092402)
    for name in SELECTED:
        rows = [r for r in records if r["study"] == name and r["phase"] == "primary"]
        if len(rows) != 40 or any(not r.get("exact", False) for r in rows):
            continue
        pairs = [{r["label"]: r["measurement"] for r in rows if r["pair"] == i} for i in range(20)]
        log = np.log([p["baseline"]["seconds"] / p["candidate"]["seconds"] for p in pairs])
        lo, hi = np.exp(np.quantile(bootstrap.choice(log, (20000, 20)).mean(axis=1), [0.025, 0.975]))
        decision = "practical regression" if hi < 0.98 else "practical equivalence" if lo >= 0.98 and hi <= 1.02 else "equivalence unresolved"
        comparisons.append(dict(study=name, geometric_speedup=float(np.exp(log.mean())), ci95=[float(lo), float(hi)], decision=decision,
                                median_seconds={k: float(np.median([p[k]["seconds"] for p in pairs])) for k in ("baseline", "candidate")},
                                median_peak_rss_kib={k: float(np.median([p[k]["peak_rss_kib"] for p in pairs])) for k in ("baseline", "candidate")}))
    assert len(records) == 168 + len(signatures)
    manifest.update(finished=time.time(), record_count=len(records), distinct_replayed_configurations=len(signatures),
                    final_binary_sha256={k: sha(getattr(args, k)) for k in ("baseline", "candidate")})
    if manifest["final_binary_sha256"] != digests:
        issues.append("executable changed during collection")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = dict(record_count=len(records), distinct_replayed_configurations=len(signatures), exact_records=sum(r.get("exact", False) for r in records), issues=issues, comparisons=comparisons)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main())
