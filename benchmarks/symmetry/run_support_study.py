#!/usr/bin/env python3
"""Freeze and collect the declared D05 process-cold paired study.

Build both executables beforehand. Numerical payloads are stored losslessly in
content-addressed gzip objects; equal values are deduplicated, not discarded.
No FFT/linear-algebra audit runs concurrently with primary sampling.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def design():
    studies = []
    for pol in ("TM", "TE"):
        for case in ("hex-gamma-c3", "hex-gamma-c6", "hex-gamma-c6v", "hex-k", "chiral-c4", "glide-pg", "glide-pgg"):
            studies.append(dict(name=f"standard-{case}-{pol}", case=case, grid=32 if case.startswith("glide") else 33,
                                radius=7, bands=8, polarization=pol, baseline="unreduced", candidate="adaptive"))
        for case in ("hex-gamma-c6v", "hex-k", "chiral-c4", "glide-pgg"):
            studies.append(dict(name=f"heldout-{case}-{pol}", case=case, grid=64 if case.startswith("glide") else 65,
                                radius=14, bands=16, polarization=pol, baseline="unreduced", candidate="adaptive"))
        for mode in ("conservative", "isotypic", "fixed-guards"):
            studies.append(dict(name=f"mechanism-{mode}-{pol}", case="hex-gamma-c6v", grid=33,
                                radius=7, bands=8, polarization=pol, baseline=mode, candidate="adaptive"))
    assert len(studies) == 28
    return studies


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def store(root, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    sha = hashlib.sha256(data).hexdigest()
    path = root / "objects" / (sha + ".json.gz")
    if not path.exists():
        path.write_bytes(gzip.compress(data, mtime=0))
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unprofiled", type=Path, required=True)
    parser.add_argument("--profiled", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--study", action="append", help="Exact study names; omitted for all 28")
    args = parser.parse_args()
    if args.pairs < 1 or args.pairs > 1000:
        parser.error("pairs must be between 1 and 1000")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"objects").mkdir()
    revision = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git","diff","HEAD","--exit-code"], cwd=ROOT, check=True, capture_output=True)
    builds = json.loads(args.build_manifest.read_text())
    if builds["source_revision"] != revision:
        raise ValueError("build manifest source differs from clean checkout")
    for label in ("unprofiled","profiled"):
        if digest(getattr(args,label)) != builds["artifacts"][label]["sha256"]:
            raise ValueError("frozen executable digest mismatch")
    specs = design()
    if args.study:
        specs = [s for s in specs if s["name"] in args.study]
        if len(specs) != len(set(args.study)):
            raise ValueError("unknown study selection")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    manifest = dict(schema="blaze2d/support-study/1", source_revision=revision, pairs=args.pairs,
                    design=specs, seed=args.seed, cpu=args.cpu, started=time.time(), builds=builds,
                    files={path:digest(ROOT/path) for path in (
                        "crates/core/examples/symmetry_support_bench.rs", "benchmarks/symmetry/D05-protocol.md",
                        "benchmarks/symmetry/run_support_study.py", "benchmarks/symmetry/support_contract.py",
                        "benchmarks/symmetry/audit_support_study.py")},
                    thread_environment={key:env[key] for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","RAYON_NUM_THREADS")},
                    cold="fresh process and numerical state; hardware/OS caches uncontrolled",
                    scope="primary numerical timer; full physical outputs; export separate; references only after primary sampling")
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rng = random.Random(args.seed)
    count = 0
    def run(spec, label, pair, phase, negative=False):
        nonlocal count
        binary = args.profiled if phase == "profile" else args.unprofiled
        cmd = ["taskset","-c",str(args.cpu),str(binary.resolve()),spec["case"],str(spec["grid"]),str(spec["bands"]),
               spec["polarization"],spec[label],str(spec["radius"]),
               "references" if phase=="profile" and not negative else "no-references","1e-14","1000"]
        row = dict(study=spec["name"], label=label, pair=pair, phase=phase, negative_control=negative,
                   command=cmd, timestamp=time.time(), load_before=os.getloadavg())
        start = time.perf_counter()
        try:
            result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=600)
            row.update(returncode=result.returncode, stderr=result.stderr.decode(errors="replace"),
                       process_seconds=time.perf_counter()-start)
            if result.returncode == 0:
                parsed = json.loads(result.stdout)
                row["numerics_sha256"] = store(args.output, parsed["numerics"])
                row["measurement"] = parsed["measurement"]
                references = row["measurement"].pop("references")
                if references is not None:
                    row["references_sha256"] = store(args.output, references)
            else:
                row["failed_stdout_sha256"] = store(args.output, dict(stdout=result.stdout.decode(errors="replace")))
        except (subprocess.TimeoutExpired, ValueError, KeyError) as e:
            row.update(returncode=-1, error=str(e), process_seconds=time.perf_counter()-start)
            if isinstance(e, subprocess.TimeoutExpired):
                row["timeout_output_sha256"] = store(args.output, dict(stdout=(e.stdout or b"").decode(errors="replace"),stderr=(e.stderr or b"").decode(errors="replace")))
            elif "result" in locals():
                row["failed_stdout_sha256"] = store(args.output, dict(stdout=result.stdout.decode(errors="replace")))
        row.update(collection_seconds=time.perf_counter()-start, load_after=os.getloadavg())
        with (args.output/"records.jsonl").open("a") as stream:
            stream.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
            stream.flush()
        count += 1
    for index,spec in enumerate(specs):
        for pair in range(-1,args.pairs):
            labels=["baseline","candidate"]; rng.shuffle(labels)
            for label in labels:
                run(spec,label,pair,"warmup" if pair<0 else "primary")
        print(f"PRIMARY {index+1}/{len(specs)} {spec['name']} records={count}",flush=True)
    # Both variants of every declared condition receive independent references.
    # These executions and their counters cannot enter primary speed estimates.
    for index,spec in enumerate(specs):
        for label in ("baseline","candidate"):
            run(spec,label,0,"profile")
        print(f"PROFILE {index+1}/{len(specs)} {spec['name']} records={count}",flush=True)
    for case,n in (("hex-gamma-analytic",33),("glide-pg",33)):
        for pol in ("TM","TE"):
            spec=dict(name=f"negative-{case}-{pol}",case=case,grid=n,bands=8,radius=7,polarization=pol,candidate="adaptive")
            for phase in ("negative","profile"):
                run(spec,"candidate",-1,phase,negative=True)
    # Frozen executables must remain unchanged through the whole collection.
    manifest["finished"] = time.time()
    manifest["record_count"] = count
    manifest["final_binary_digests"] = {label:digest(getattr(args,label)) for label in ("unprofiled","profiled")}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"COMPLETE {count} records",flush=True)


if __name__ == "__main__":
    main()
