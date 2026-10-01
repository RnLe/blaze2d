#!/usr/bin/env python3
"""Explanatory per-stage time shares from the profiling binary; not latency evidence.

Cases run on pinned single-threaded workers with --profile-output. Timer exclusive
times (adjusted clock) are grouped into solver stages and reported as shares per
variant x resolution x polarization, so the clock scale cancels.
"""
from __future__ import annotations

import argparse
import itertools
import threading
from pathlib import Path
from typing import Any

import cases
import runlib
from run_work import latest, setup

CATEGORIES = ('fft', 'operator_apply', 'preconditioner', 'svqb_gemm', 'rayleigh_ritz',
              'ritz_update', 'residuals', 'other')
SUMMARY_SCHEMA = 'blaze2d-lab/preconditioner/profile-summary/1'


def category(timer: str) -> str:
    name = timer.lower()
    if 'precondition' in name:
        return 'preconditioner'
    if 'fft' in name:
        return 'fft'
    if name.startswith(('apply_', 'batch_apply_')) or name in ('compute_aq_block', 'symmetry.galerkin.apply'):
        return 'operator_apply'
    if name.startswith('svqb') or any(s in name for s in ('orthonormal', 'gram', 'linear_combinations')):
        return 'svqb_gemm'
    if 'update_ritz' in name:
        return 'ritz_update'
    if 'ritz' in name or name == 'project_operator':
        return 'rayleigh_ritz'
    if 'residual' in name or 'certification' in name:
        return 'residuals'
    return 'other'


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    timers = {name: stats['exclusive_nanos'] / 1e9 for name, stats in profile.get('timings', {}).items()}
    seconds = dict.fromkeys(CATEGORIES, 0.0)
    for name, value in timers.items():
        seconds[category(name)] += value
    counters = profile.get('counters', {})
    return dict(category_seconds=seconds, timer_seconds=timers, counters=counters,
                unbalanced_timers=counters.get('profiler.unbalanced_timers', 0),
                object_sha256=profile.get('object_sha256'))


def summarize(study: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple, dict[str, Any]] = {}
    timers: dict[str, set[str]] = {c: set() for c in CATEGORIES}
    for record in records:
        profile = record.get('profile')
        if not profile or not record['validation']['completed']:
            continue
        key = (record['variant'], record['params']['resolution'], record['params']['polarization'])
        group = groups.setdefault(key, dict(variant=key[0], resolution=key[1], polarization=key[2],
                                            cases=0, seconds=dict.fromkeys(CATEGORIES, 0.0)))
        group['cases'] += 1
        for name, value in profile['category_seconds'].items():
            group['seconds'][name] += value
        for name in profile['timer_seconds']:
            timers[category(name)].add(name)
    rows = []
    for key in sorted(groups):
        group = groups[key]
        total = sum(group['seconds'].values())
        rows.append(dict(group, total_exclusive_seconds=total,
                         shares={c: v / total if total else None for c, v in group['seconds'].items()}))
    return dict(schema=SUMMARY_SCHEMA, study=study, categories=list(CATEGORIES),
                clock='profiler timers use the adjusted monotonic clock; shares only',
                groups=rows, timers={c: sorted(v) for c, v in timers.items()},
                unbalanced_runs=sum(1 for r in records if (r.get('profile') or {}).get('unbalanced_timers')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', required=True)
    parser.add_argument('--subset', choices=cases.SUBSETS, default='screen32')
    parser.add_argument('--variant', action='append', required=True)
    parser.add_argument('--binary', help='profiling binary: frozen, current or a path')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpus')
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--nice', type=int, default=19)
    parser.add_argument('--match')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--work-file', type=Path)
    parser.add_argument('--work-variant')
    parser.add_argument('--stage', default='baseline')
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    root, meta, variants, resolved, selected = setup(args, 'profile', profiling=True)
    records_path = root / 'profile_records.jsonl'
    done = latest(runlib.read_jsonl(records_path))
    tasks = [(case, name) for case in sorted(selected, key=lambda c: (-c.resolution, c.label))
             for name in variants if (case.id, name, resolved[name]['sha256'], 'profile') not in done]
    writer, counter, lock = runlib.JsonlWriter(records_path), itertools.count(1), threading.Lock()
    print(f'{args.study}: {len(tasks)} profiles to run', flush=True)

    def work(task: tuple[cases.Case, str], cpu: int) -> None:
        case, name = task
        record, _, profile = runlib.execute_case(case, name, variants[name], resolved[name], root / 'tmp',
                                                 root / 'objects', cpu=cpu, nice=args.nice,
                                                 timeout=args.timeout, profile=True,
                                                 include_frequencies=False, tag='profile')
        if runlib.INTERRUPTED.is_set():
            return
        record['profile'] = profile_summary(profile) if profile else None
        writer.write(record)
        with lock:
            index = next(counter)
        print(f'[{index}/{len(tasks)}] cpu{cpu} {case.label} {name} profile={"yes" if profile else "missing"}',
              flush=True)

    status = 'complete'
    try:
        runlib.run_pool(tasks, runlib.cpu_list(args.cpus, args.workers)[:args.workers], work)
    except KeyboardInterrupt:
        status = 'interrupted'
    records = list(latest(runlib.read_jsonl(records_path)).values())
    summary = summarize(args.study, records)
    runlib.write_json(root / 'profile_summary.json', summary)
    meta.update(status=status, ended=runlib.utc_now(),
                counts=dict(records=len(records), failed=sum(not r['validation']['completed'] for r in records)))
    runlib.write_json(root / 'manifest.json', meta)
    runlib.remove_scratch(root)
    for group in summary['groups']:
        shares = ' '.join(f'{c}={100 * v:.1f}%' for c, v in group['shares'].items() if v)
        print(f"{group['variant']} N={group['resolution']} {group['polarization']}: {shares}")


if __name__ == '__main__':
    main()
