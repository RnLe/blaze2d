#!/usr/bin/env python3
"""Pinned single-threaded timing of the operators task, C3 binary against the current one.

Operator calculations take no preconditioner settings, so the comparison is between
binaries: before C5 the operators task used the legacy Fourier diagonal for both
polarizations, afterwards the band calculations' default. Every repetition runs the
two binaries on each example in seeded random order on one pinned CPU; records hold
wall seconds, the solver's elapsed time and iterations, and the load average around
each run (no loadmon admission, so run it on a quiet host).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np

import cases
import runlib

EXAMPLES = ('examples/calculations/operator-point.toml', 'examples/calculations/registry-stencil.toml',
            'crates/interface/tests/fixtures/operators.toml')


def run(binary: Path, config: dict, cpu: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        toml, out = Path(tmp) / 'case.toml', Path(tmp) / 'out.json'
        toml.write_text(cases.toml_text(config))
        load = os.getloadavg()[0]
        start = time.perf_counter()
        subprocess.run(['taskset', '-c', str(cpu), str(binary), 'run', str(toml), '--threads', '1', '--quiet', '-o', str(out)],
                       check=True, capture_output=True)
        wall = time.perf_counter() - start
        results = json.loads(out.read_text())['results']
    return dict(wall_seconds=wall, load_before=load, load_after=os.getloadavg()[0],
                solver_seconds=sum(r['metadata'].get('elapsed_seconds') or 0 for r in results),
                iterations=int(sum(np.sum(r['metadata'].get('iterations') or 0) for r in results)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', default='c5-operators-timing')
    parser.add_argument('--before', type=Path, default=runlib.REPO / 'benchmarks/preconditioner/bin/55dd68f5/blaze2d')
    parser.add_argument('--after', type=Path, default=runlib.REPO / 'benchmarks/preconditioner/bin/ffcd56d7/blaze2d')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    root = runlib.RESULTS / args.study
    root.mkdir(parents=True, exist_ok=True)
    binaries = {'before': runlib.binary_info(args.before), 'after': runlib.binary_info(args.after)}
    runlib.write_json(root / 'manifest.json', runlib.manifest(args.study, 'ml', 'ml', 'operators',
                      {'operators': dict(description=__doc__.splitlines()[0])}, binaries, dict(reps=args.reps, cpu=args.cpu)))
    writer = runlib.JsonlWriter(root / 'records.jsonl')
    rng = random.Random(args.seed)
    jobs = [(Path(e).stem, pol, dict(tomllib.loads((runlib.REPO / e).read_text()), polarization=pol))
            for e in EXAMPLES for pol in ('TE', 'TM')]
    times: dict[tuple, dict[str, list[float]]] = {}
    for rep in range(args.reps):
        for name, pol, config in jobs:
            for which in rng.sample(['before', 'after'], 2):
                r = run(Path(binaries[which]['path']), config, args.cpu)
                times.setdefault((name, pol), {}).setdefault(which, []).append(r['wall_seconds'])
                writer.write(dict(tag='operator-timing', variant='operators', binary=binaries[which], which=which,
                                  example=name, polarization=pol, rep=rep,
                                  validation=dict(completed=True, ok=True, flags=[]), **r))
    for (name, pol), t in times.items():
        b, a = statistics.median(t['before']), statistics.median(t['after'])
        print(f'{name:18s} {pol}  before {b:6.3f}s  after {a:6.3f}s  speed-up {b / a:.2f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
