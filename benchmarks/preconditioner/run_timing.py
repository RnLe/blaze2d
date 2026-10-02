#!/usr/bin/env python3
"""Serial, pinned, single-threaded timing of variant pairs (or one variant, or blocks) under load monitoring.

Each case gets adjacent A/B pairs in seeded random order while the loadmon sampler runs.
With more than two variants every repetition is a block that runs all of them in seeded
random order, admitted as a whole like a pair; ratios are then seconds(first variant) /
seconds(variant) for every other variant.
A pair whose members are not both clean reruns as a whole (at most three attempts, ten
seconds apart); the first clean attempt wins, otherwise the first paired-matched attempt
(only background-load reasons, Linux background <= 1.5 cores in both runs and differing
by <= 0.5). Solver seconds are the job timer rescaled to CLOCK_MONOTONIC_RAW through
normalized_timing; process seconds are raw wall time including startup and output.
Ratios are seconds(A) / seconds(B), so values above one mean B is faster.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import cases
import loadmon
import runlib
from normalized_timing import normalize
from quiet_host import QuietHost, elapsed_clocks

SEED = 20261001
RESAMPLES = 10000
PRACTICAL = 1.02
MATCHED_BACKGROUND = 1.5
MATCHED_DIFFERENCE = 0.5
SENSITIVITY = 0.10
SUMMARY_SCHEMA = 'blaze2d-lab/preconditioner/timing-summary/1'


def geometric_mean(values) -> float:
    values = list(values)
    return math.exp(statistics.fmean(map(math.log, values)))


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[math.ceil(0.975 * len(ordered)) - 1]]


def decision(ci: list[float]) -> str:
    low, high = ci
    if low > PRACTICAL:
        return 'improvement'
    if high < 1 / PRACTICAL:
        return 'regression'
    return 'equivalent' if low >= 1 / PRACTICAL and high <= PRACTICAL else 'inconclusive'


def ratio_statistics(per_case: dict[str, tuple[list[float], list[float]]], seed: int = SEED,
                     resamples: int = RESAMPLES) -> dict[str, Any] | None:
    """Per-case geometric pair ratio; case-bootstrap CIs of the geometric and time-weighted ratios."""
    ratios, means = {}, {}
    for case_id, (a, b) in per_case.items():
        if a and len(a) == len(b):
            ratios[case_id] = geometric_mean(x / y for x, y in zip(a, b))
            means[case_id] = (statistics.fmean(a), statistics.fmean(b))
    if not ratios:
        return None
    ids = sorted(ratios)
    logs = {c: math.log(ratios[c]) for c in ids}
    rng = random.Random(seed)
    geometric, weighted = [], []
    for _ in range(resamples):
        draw = rng.choices(ids, k=len(ids))
        geometric.append(math.exp(statistics.fmean(logs[c] for c in draw)))
        weighted.append(sum(means[c][0] for c in draw) / sum(means[c][1] for c in draw))
    ci = interval(geometric)
    return dict(cases=len(ids), pairs=sum(len(per_case[c][0]) for c in ids),
                geometric_mean=geometric_mean(ratios.values()), median_case=statistics.median(ratios.values()),
                time_weighted=sum(m[0] for m in means.values()) / sum(m[1] for m in means.values()),
                ci95_geometric=ci, ci95_time_weighted=interval(weighted), decision=decision(ci),
                per_case=ratios)


def level_statistics(per_case: dict[str, list[float]]) -> dict[str, Any] | None:
    medians = {c: statistics.median(v) for c, v in per_case.items() if v}
    if not medians:
        return None
    return dict(cases=len(medians), runs=sum(len(v) for v in per_case.values()),
                median_case_seconds=statistics.median(medians.values()),
                geometric_mean_seconds=geometric_mean(medians.values()),
                total_seconds=sum(medians.values()), per_case=medians)


def matched(runs: list[dict[str, Any]]) -> bool:
    if not all(loadmon.background_only(r['load']) for r in runs):
        return False
    metrics = [r['load']['metrics'] for r in runs]
    backgrounds = [m.get('linux_background_mean') for m in metrics]
    if any(b is None or b > MATCHED_BACKGROUND for b in backgrounds):
        return False
    if any((m.get('windows_background_mean') or 0) > loadmon.LIMITS['windows_peak'] for m in metrics):
        return False
    return max(backgrounds) - min(backgrounds) <= MATCHED_DIFFERENCE


class Session:
    def __init__(self, args: argparse.Namespace, root: Path, variants: dict, binaries: dict,
                 host: QuietHost, sampler_pid: int, reader: loadmon.SampleReader):
        self.args, self.root, self.variants, self.binaries = args, root, variants, binaries
        self.host, self.sampler_pid, self.reader = host, sampler_pid, reader
        self.writer = runlib.JsonlWriter(root / 'runs.jsonl')

    def timed(self, case: cases.Case, name: str, tag: str) -> dict[str, Any]:
        record, run, _ = runlib.execute_case(
            case, name, self.variants[name], self.binaries[name], self.root / 'tmp', self.root / 'objects',
            cpu=self.args.cpu, nice=self.args.nice, timeout=self.args.timeout, host=self.host,
            owned_pids=[self.sampler_pid], include_frequencies=False, tag=tag)
        try:
            clocks = elapsed_clocks(*run['snapshots'])
        except ValueError as error:
            clocks, record['clock_issue'] = None, str(error)
        record['clock'], record['snapshots'] = clocks, run['snapshots']
        job = record.get('job_seconds')
        record['solver_seconds'] = job * clocks['adjusted_to_raw_scale'] if clocks and job else None
        record['child_cpu_seconds'] = record['rusage']['user_seconds'] + record['rusage']['system_seconds']
        return record

    def judge(self, runs: list[dict[str, Any]]) -> None:
        self.reader.wait_for('windows', runs[-1]['raw_end'], 5)
        samples = self.reader.poll()
        for run in runs:
            run['load'] = loadmon.verdict(samples, run['raw_start'], run['raw_end'], run['child_cpu_seconds'],
                                          [self.args.cpu], run['snapshots'])
        self.reader.prune(runs[-1]['raw_end'] - 600)

    def entry(self, case: cases.Case, rep: int, order: list[str]) -> dict[str, Any]:
        key, attempts, chosen = f'{case.id}:{rep}', [], {}
        for attempt in range(self.args.attempts):
            if attempt:
                time.sleep(self.args.pause)
            runs = [self.timed(case, name, 'timing') for name in order]
            self.judge(runs)
            for run in runs:
                run.update(key=key, rep=rep, attempt=attempt, run_id=f'{key}:{attempt}:{run["variant"]}')
                self.writer.write(run)
            quality = all(r['validation']['completed'] for r in runs)
            strict = all(r['load']['verdict'] == 'clean' for r in runs)
            attempts.append(dict(attempt=attempt, quality=quality, strict=strict, matched=matched(runs),
                                 verdicts={r['variant']: r['load']['verdict'] for r in runs},
                                 reasons={r['variant']: r['load']['reasons'] for r in runs}))
            chosen[attempt] = runs
            if strict or not quality or (self.args.accept_matched and attempts[-1]['matched']):
                break
        winner = next((a['attempt'] for a in attempts if a['quality'] and a['strict']), None)
        tier = 'strict' if winner is not None else None
        if winner is None:
            winner = next((a['attempt'] for a in attempts if a['quality'] and a['matched']), None)
            tier = 'matched' if winner is not None else None
        runs = chosen.get(winner, [])
        return dict(key=key, case_id=case.id, label=case.label, params=case.params, rep=rep, order=order,
                    attempts=attempts, winner=winner, tier=tier,
                    quality=all(a['quality'] for a in attempts),
                    run_ids={r['variant']: r['run_id'] for r in runs},
                    solver_seconds={r['variant']: r['solver_seconds'] for r in runs},
                    process_seconds={r['variant']: r['wall_seconds'] for r in runs},
                    background={r['variant']: r['load']['metrics'].get('linux_background_mean') for r in runs})


def summarize(study: str, variants: list[str], entries: list[dict[str, Any]],
              runs: list[dict[str, Any]], seed: int = SEED, resamples: int = RESAMPLES) -> dict[str, Any]:
    by_id = {r['run_id']: r for r in runs if r.get('job_seconds') and 'run_id' in r}
    admitted = [e for e in entries if e['tier'] and all(i in by_id for i in e['run_ids'].values())]
    chosen = [by_id[i] for e in admitted for i in e['run_ids'].values() if i in by_id]
    clock: dict[str, Any] = dict(clock_qualified=False, issues=['no admitted runs'], scale_relative_spread=None)
    solver: dict[str, float] = {}
    if chosen:
        normalized = normalize(dict(samples=[dict(result=dict(seconds=r['job_seconds'])) for r in chosen],
                                    summary={}, manifest=dict(seed=seed, pairs=0)),
                               [dict(start=r['snapshots'][0], end=r['snapshots'][1]) for r in chosen], 'run_pairs')
        clock = {k: normalized[k] for k in ('clock_qualified', 'issues', 'scale_relative_spread', 'method')}
        if normalized['clock_qualified']:
            solver = {r['run_id']: s['result']['seconds'] for r, s in zip(chosen, normalized['samples'])}
    metrics = {'solver': solver, 'process': {r['run_id']: r['wall_seconds'] for r in chosen}}
    pair_mode = len(variants) == 2
    mode = 'pair' if pair_mode else 'block' if len(variants) > 2 else 'single'
    tiers: dict[str, dict[str, Any]] = {}
    levels: dict[str, dict[str, Any]] = {}
    for tier, allowed in (('strict', {'strict'}), ('matched', {'strict', 'matched'})):
        tiers[tier] = {}
        for metric, seconds in metrics.items():
            if not seconds:
                tiers[tier][metric] = None
                continue
            grouped: dict[str, list[list[float]]] = {}
            for e in admitted:
                if e['tier'] in allowed:
                    values = grouped.setdefault(e['case_id'], [[] for _ in variants])
                    for index, name in enumerate(variants):
                        values[index].append(seconds[e['run_ids'][name]])
            if mode == 'pair':
                tiers[tier][metric] = ratio_statistics({c: (v[0], v[1]) for c, v in grouped.items()}, seed, resamples)
            elif mode == 'block':
                tiers[tier][metric] = {name: ratio_statistics({c: (v[0], v[i]) for c, v in grouped.items()}, seed, resamples)
                                       for i, name in enumerate(variants) if i}
                levels.setdefault(tier, {})[metric] = {name: level_statistics({c: v[i] for c, v in grouped.items()})
                                                       for i, name in enumerate(variants)}
            else:
                tiers[tier][metric] = level_statistics({c: v[0] for c, v in grouped.items()})
    primary = 'solver' if solver else 'process'
    key = 'median_case_seconds' if mode == 'single' else 'geometric_mean'
    strict, loose = tiers['strict'][primary], tiers['matched'][primary]
    if mode == 'block':
        sensitive = bool(strict and loose and any(strict.get(n) and loose.get(n) and abs(loose[n][key] / strict[n][key] - 1)
                                                  > SENSITIVITY for n in variants[1:]))
    else:
        sensitive = bool(strict and loose and abs(loose[key] / strict[key] - 1) > SENSITIVITY)
    labels = {e['case_id']: (e['label'], e['params']) for e in entries}
    primary_seconds = metrics[primary]
    per_case = []
    for case_id, (label, params) in sorted(labels.items(), key=lambda item: item[1][0]):
        row = dict(case_id=case_id, label=label, params=params)
        for tier in tiers:
            stats = tiers[tier][primary]
            if mode == 'block':
                row[tier] = {n: s['per_case'].get(case_id) for n, s in (stats or {}).items() if s} or None
            else:
                row[tier] = stats['per_case'].get(case_id) if stats else None
        row['seconds'] = {}
        for name in variants:
            values = [primary_seconds[e['run_ids'][name]] for e in admitted
                      if e['case_id'] == case_id and e['run_ids'].get(name) in primary_seconds]
            row['seconds'][name] = statistics.median(values) if values else None
        per_case.append(row)
    verdicts = [r['load']['verdict'] for r in runs if r.get('tag') == 'timing']
    return dict(schema=SUMMARY_SCHEMA, study=study, mode=mode, variants=variants, levels=levels,
                orientation=dict(pair='seconds(A) / seconds(B); above one means B is faster',
                                 block='seconds(first variant) / seconds(variant); above one means the variant is faster',
                                 single='seconds')[mode],
                seed=seed, resamples=resamples, primary_metric=primary, clock=clock, tiers=tiers,
                sensitive_to_environment=sensitive, per_case=per_case,
                counts=dict(entries=len(entries), strict=sum(e['tier'] == 'strict' for e in entries),
                            matched=sum(e['tier'] == 'matched' for e in entries),
                            unqualified=sum(e['tier'] is None for e in entries),
                            quality_failures=sum(not e['quality'] for e in entries), runs=len(verdicts),
                            clean_runs=verdicts.count('clean'), contaminated_runs=verdicts.count('contaminated'),
                            unknown_runs=verdicts.count('unknown')))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', required=True)
    parser.add_argument('--subset', choices=cases.SUBSETS, default='timing48')
    parser.add_argument('--variant', action='append', required=True,
                        help='A, then optionally B, or three or more variants (the first is the reference)')
    parser.add_argument('--binary', action='append', default=[], help='VARIANT=frozen|current|PATH')
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--sampler-cpu', type=int, default=30)
    parser.add_argument('--pairs', type=int, default=3, help='pairs (runs, blocks) per case')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--pause', type=float, default=10)
    parser.add_argument('--accept-matched', action='store_true',
                        help='stop retrying a pair at its first paired-matched attempt (hosts without strict-clean windows)')
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--nice', type=int, default=0)
    parser.add_argument('--wait', type=float, default=120, help='preflight polling budget in seconds')
    parser.add_argument('--preflight-linux', type=float, default=0.25, help='preflight Linux background cores')
    parser.add_argument('--preflight-windows', type=float, default=1.5, help='preflight Windows background cores')
    parser.add_argument('--match')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--work-file', type=Path)
    parser.add_argument('--work-variant')
    parser.add_argument('--stage', default='baseline')
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    if len(set(args.variant)) != len(args.variant):
        parser.error('variants must be distinct')
    return args


def main() -> None:
    args = parse_args()
    known = cases.load_variants()
    if missing := [v for v in args.variant if v not in known]:
        raise SystemExit(f'unknown variants {missing}')
    variants = {name: known[name] for name in args.variant}
    overrides = dict(item.split('=', 1) for item in args.binary)
    binaries = {name: runlib.variant_binary(variants[name], overrides.get(name)) for name in variants}
    work = cases.work_from_args(args)
    selected = cases.select(args.subset, match=args.match, ids=args.case, limit=args.limit, work=work)
    if not selected:
        raise SystemExit('no cases selected')
    root = args.results / args.study
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'manifest.json'
    if path.exists():
        meta = json.loads(path.read_text())
        if meta['kind'] != 'timing' or list(meta['variants']) != args.variant or any(
                meta['binaries'][n]['sha256'] != b['sha256'] for n, b in binaries.items()):
            raise SystemExit(f'{root} holds a different timing study; choose a new --study')
        meta.update(status='running', ended=None)
        meta.setdefault('protocol', {})['accept_matched'] = args.accept_matched
    else:
        meta = runlib.manifest(args.study, 'timing', args.stage, args.subset, variants, binaries,
                               {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        meta['protocol'] = dict(mode='pair' if len(variants) == 2 else 'block' if len(variants) > 2 else 'single', cpu=args.cpu,
                                sampler_cpu=args.sampler_cpu, pairs_per_case=args.pairs, seed=args.seed,
                                attempts=args.attempts, pause_seconds=args.pause, warmups_per_variant=1,
                                accept_matched=args.accept_matched,
                                limits=loadmon.LIMITS, matched=dict(background=MATCHED_BACKGROUND,
                                                                    difference=MATCHED_DIFFERENCE),
                                practical_ratio=PRACTICAL, resamples=RESAMPLES)
    runlib.write_json(path, meta)
    entries = json.loads((root / 'pairs.json').read_text()) if (root / 'pairs.json').exists() else []
    done = {e['key'] for e in entries}
    rng = random.Random(args.seed)
    schedule = [(case, rep) for case in selected for rep in range(args.pairs)]
    rng.shuffle(schedule)
    orders = [rng.sample(args.variant, len(args.variant)) for _ in schedule]
    sampler = loadmon.start_sampler(root / 'loadmon.jsonl', args.sampler_cpu)
    reader = loadmon.SampleReader(root / 'loadmon.jsonl')
    status = 'complete'
    try:
        if not reader.wait_for('windows', runlib.raw_clock(), 60):
            raise SystemExit('load sampler produced no Windows reading within 60 s')
        with QuietHost() as host:
            session = Session(args, root, variants, binaries, host, sampler.pid, reader)
            check = loadmon.preflight(host, args.wait, linux=args.preflight_linux,
                                      windows=args.preflight_windows, owned_pids=[sampler.pid])
            meta.setdefault('preflight', []).append(dict(check, when=runlib.utc_now()))
            runlib.write_json(path, meta)
            print(f"preflight passed={check['passed']} after {check['attempts']} windows", flush=True)
            if not check['passed']:
                status = 'preflight_failed'
                return
            for name in args.variant:
                warmup = session.timed(schedule[0][0], name, 'warmup')
                session.writer.write(dict(warmup, run_id=f'warmup:{name}:{runlib.utc_now()}'))
            for index, ((case, rep), order) in enumerate(zip(schedule, orders), 1):
                if f'{case.id}:{rep}' in done:
                    continue
                entry = session.entry(case, rep, order)
                entries.append(entry)
                runlib.write_json(root / 'pairs.json', entries)
                outcome = (' '.join(f'{n}={entry["process_seconds"][n]:.3f}s' for n in order) if entry['tier']
                           else ' '.join(f'{n}:{v}' for n, v in entry['attempts'][-1]['verdicts'].items()))
                print(f'[{index}/{len(schedule)}] {case.label} rep{rep} tier={entry["tier"]} '
                      f'attempts={len(entry["attempts"])} {outcome}', flush=True)
            if any(e['tier'] is None for e in entries):
                status = 'partial'
    except KeyboardInterrupt:
        status = 'interrupted'
    finally:
        loadmon.stop_sampler(sampler)
        summary = summarize(args.study, args.variant, entries, runlib.read_jsonl(root / 'runs.jsonl'), args.seed)
        runlib.write_json(root / 'summary.json', summary)
        meta.update(status=status, ended=runlib.utc_now(), counts=summary['counts'])
        runlib.write_json(path, meta)
        runlib.remove_scratch(root)
        print(json.dumps(dict(status=status, counts=summary['counts'], clock=summary['clock'],
                              sensitive_to_environment=summary['sensitive_to_environment'],
                              matched=headline(summary)), indent=1))


def headline(summary: dict[str, Any]) -> dict[str, Any]:
    """Primary-metric ratios (or levels) of the matched tier without per-case detail."""
    stats = summary['tiers']['matched'][summary['primary_metric']]
    if not stats:
        return {}
    if summary['mode'] == 'block':
        return {n: s and {k: s[k] for k in ('cases', 'geometric_mean', 'ci95_geometric', 'time_weighted')}
                for n, s in stats.items()}
    return {k: v for k, v in stats.items() if k != 'per_case'}


if __name__ == '__main__':
    main()
