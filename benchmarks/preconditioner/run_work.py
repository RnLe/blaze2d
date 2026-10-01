#!/usr/bin/env python3
"""Deterministic work sweep: subset x variants, one pinned single-threaded worker per CPU.

Records append to results/<study>/records.jsonl; a rerun skips every case and variant
already recorded with the same binary. Wall times from the parallel pool describe
throughput, not latency. --determinism repeats a few cases serially, in the pool, with
a profiling binary and on a second CPU, and reports any change in the frequencies,
per-k iterations or work counters.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

import cases
import runlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', required=True)
    parser.add_argument('--subset', choices=cases.SUBSETS, default='dev96')
    parser.add_argument('--variant', action='append', required=True)
    parser.add_argument('--binary', help="override each variant's binary: frozen, current or a path")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cpus', help='CPU list such as 16-23 (default: one per virtual core from 16)')
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--nice', type=int, default=19)
    parser.add_argument('--match', help='glob on case labels')
    parser.add_argument('--case', action='append', default=[], help='case id or label')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--work-file', type=Path, help='records.jsonl choosing timing48 roles')
    parser.add_argument('--work-variant')
    parser.add_argument('--stage', default='baseline', help='lab stage this study belongs to')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--determinism', action='store_true')
    parser.add_argument('--repeats', type=int, default=3, help='serial repeats in --determinism')
    parser.add_argument('--profiling-binary', help='frozen, current or a path (--determinism)')
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    return parser.parse_args()


def setup(args: argparse.Namespace, kind: str, profiling: bool = False):
    known = cases.load_variants()
    missing = [name for name in args.variant if name not in known]
    if missing:
        raise SystemExit(f'unknown variants {missing}; known: {", ".join(known)}')
    variants = {name: known[name] for name in args.variant}
    binaries = {name: runlib.variant_binary(variants[name], args.binary, profiling) for name in variants}
    work = cases.work_from_args(args)
    selected = cases.select(args.subset, match=args.match, ids=args.case, limit=args.limit, work=work)
    if not selected:
        raise SystemExit('no cases selected')
    root = args.results / args.study
    root.mkdir(parents=True, exist_ok=True)
    invocation = dict(started=runlib.utc_now(), argv=sys.argv[1:], cases=len(selected))
    path = root / 'manifest.json'
    if path.exists():
        meta = json.loads(path.read_text())
        if meta['kind'] != kind:
            raise SystemExit(f'{root} is a {meta["kind"]} study')
        for name, binary in binaries.items():
            old = meta['binaries'].get(name)
            if old and old['sha256'] != binary['sha256']:
                raise SystemExit(f'{name}: binary changed since this study started; use a new study')
        meta['variants'].update(variants)
        meta['binaries'].update(binaries)
        meta.update(status='running', ended=None, pid=os.getpid())
    else:
        meta = runlib.manifest(args.study, kind, args.stage, args.subset, variants, binaries,
                               {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    meta.setdefault('invocations', []).append(invocation)
    runlib.write_json(path, meta)
    return root, meta, variants, binaries, selected


def finish(root: Path, meta: dict[str, Any], status: str, records: list[dict[str, Any]]) -> None:
    meta.update(status=status, ended=runlib.utc_now())
    meta['invocations'][-1]['ended'] = meta['ended']
    for name, binary in meta['binaries'].items():
        if Path(binary['path']).is_file() and runlib.file_digest(Path(binary['path'])) != binary['sha256']:
            meta.setdefault('warnings', []).append(f'{name}: {binary["path"]} changed during the study')
        revisions = {r['source_revision'] for r in records if r['variant'] == name and r.get('source_revision')}
        if len(revisions) == 1 and not binary.get('revision'):
            binary['revision'] = revisions.pop()
    meta['counts'] = dict(records=len(records), failed=sum(not r['validation']['completed'] for r in records),
                          uncertified=sum(not r['validation']['ok'] for r in records))
    runlib.write_json(root / 'manifest.json', meta)
    runlib.remove_scratch(root)


def latest(records: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {(r['case_id'], r['variant'], r['binary']['sha256'], r.get('tag', 'work')): r for r in records}


def sweep(args: argparse.Namespace) -> None:
    root, meta, variants, binaries, selected = setup(args, 'work')
    records_path = root / 'records.jsonl'
    done = latest(runlib.read_jsonl(records_path))
    tasks = []
    for case in sorted(selected, key=lambda c: (-c.resolution, c.label)):
        for name in variants:
            previous = done.get((case.id, name, binaries[name]['sha256'], 'work'))
            if previous is None or (args.retry_failed and not previous['validation']['ok']):
                tasks.append((case, name))
    total = len(selected) * len(variants)
    print(f'{args.study}: {len(tasks)} of {total} runs to do', flush=True)
    writer = runlib.JsonlWriter(records_path)
    counter, lock = itertools.count(1), threading.Lock()

    def work(task: tuple[cases.Case, str], cpu: int) -> None:
        case, name = task
        record, _, _ = runlib.execute_case(case, name, variants[name], binaries[name], root / 'tmp',
                                           root / 'objects', cpu=cpu, nice=args.nice, timeout=args.timeout)
        if runlib.INTERRUPTED.is_set():
            return
        writer.write(record)
        with lock:
            index = next(counter)
        flags = ','.join(record['validation']['flags']) or 'ok'
        print(f"[{index}/{len(tasks)}] cpu{cpu} {case.label} {name} it={record['iterations_total']} "
              f"wall={record['wall_seconds']:.3f}s {flags}", flush=True)

    cpus = runlib.cpu_list(args.cpus, args.workers)[:args.workers]
    try:
        runlib.run_pool(tasks, cpus, work)
    except KeyboardInterrupt:
        finish(root, meta, 'interrupted', runlib.read_jsonl(records_path))
        raise SystemExit(130)
    records = list(latest(runlib.read_jsonl(records_path)).values())
    covered = {(r['case_id'], r['variant']) for r in records if r.get('tag', 'work') == 'work'}
    complete = all((c.id, n) in covered for c in selected for n in variants)
    finish(root, meta, 'complete' if complete else 'partial', records)


def signature(record: dict[str, Any]) -> dict[str, Any]:
    return dict(returncode=record['returncode'], frequencies_sha256=record.get('frequencies_sha256'),
                iterations=record.get('iterations'), work=record.get('optional', {}).get('work'),
                attempted_iterations=record.get('execution', {}).get('attempted_iterations'),
                retained_iterations=record.get('execution', {}).get('retained_iterations'))


def compare(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mismatches of every repeat against the first serial run of its case and variant."""
    mismatches = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record['case_id'], record['variant']), []).append(record)
    for runs in groups.values():
        reference = next((r for r in runs if r['tag'] == 'det-serial-0'), runs[0])
        expected = signature(reference)
        for record in runs:
            actual = signature(record)
            fields = [k for k in expected if actual[k] != expected[k]]
            if fields:
                mismatches.append(dict(label=record['label'], variant=record['variant'],
                                       tag=record['tag'], cpu=record['cpu'], fields=fields,
                                       binary=record['binary']['name']))
    return mismatches


def determinism(args: argparse.Namespace) -> None:
    if args.limit is None and not args.case:
        args.limit = 2
    root, meta, variants, binaries, selected = setup(args, 'determinism')
    cpus = runlib.cpu_list(args.cpus, max(args.workers, 2))
    first, second = cpus[0], cpus[1]
    profiling = (runlib.binary_info(runlib.resolve_binary(args.profiling_binary, True), args.profiling_binary)
                 if args.profiling_binary else None)
    if profiling:
        meta['binaries']['profiling'] = profiling
    writer, records, lock = runlib.JsonlWriter(root / 'records.jsonl'), [], threading.Lock()

    def run(case: cases.Case, name: str, binary: dict[str, Any], cpu: int, tag: str) -> None:
        record, _, _ = runlib.execute_case(case, name, variants[name], binary, root / 'tmp',
                                           root / 'objects', cpu=cpu, nice=args.nice,
                                           timeout=args.timeout, tag=tag)
        if runlib.INTERRUPTED.is_set():
            return
        writer.write(record)
        with lock:
            records.append(record)
        print(f"{tag:>14} cpu{cpu} {case.label} {name} it={record['iterations_total']} "
              f"freq={str(record.get('frequencies_sha256'))[:12]} wall={record['wall_seconds']:.3f}s",
              flush=True)

    pairs = [(case, name) for case in selected for name in variants]
    try:
        for repeat in range(args.repeats):
            for case, name in pairs:
                run(case, name, binaries[name], first, f'det-serial-{repeat}')
        copies = math.ceil(len(cpus) / len(pairs))
        tasks = [(case, name, copy) for copy in range(copies) for case, name in pairs]
        runlib.run_pool(tasks, cpus, lambda t, cpu: run(t[0], t[1], binaries[t[1]], cpu, f'det-pool-{t[2]}'))
        if profiling:
            for case, name in pairs:
                run(case, name, profiling, first, 'det-profiling')
        for case, name in pairs:
            run(case, name, binaries[name], second, f'det-cpu{second}')
    except KeyboardInterrupt:
        finish(root, meta, 'interrupted', records)
        raise SystemExit(130)
    mismatches = compare(records)
    report = dict(schema='blaze2d-lab/preconditioner/determinism/1', study=args.study,
                  cases=[c.label for c in selected], variants=list(variants),
                  cpus=dict(serial=first, second=second, pool=cpus),
                  profiling_binary=profiling and profiling['name'], runs=len(records),
                  failed_runs=sum(r['returncode'] != 0 for r in records),
                  mismatches=mismatches, deterministic=not mismatches and all(r['returncode'] == 0 for r in records))
    runlib.write_json(root / 'determinism.json', report)
    finish(root, meta, 'complete', records)
    print(json.dumps({k: report[k] for k in ('runs', 'failed_runs', 'deterministic', 'mismatches')}, indent=1))


def main() -> None:
    args = parse_args()
    determinism(args) if args.determinism else sweep(args)


if __name__ == '__main__':
    main()
