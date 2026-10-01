#!/usr/bin/env python3
"""Build the preconditioner lab pages (schema blaze2d-lab/preconditioner/1) from results/.

Every page is written whether or not data exists; a page without data is "empty" and
its notes name the study that would fill it. Work counts come from work sweeps,
latency only from admitted timing studies, and time shares from profiles. A study
belongs to the stage named in its manifest (--stage of the run scripts).
"""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import functools
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable

import cases
import runlib
from run_timing import PRACTICAL, ratio_statistics

SCHEMA = 'blaze2d-lab/preconditioner/1'
OUT = runlib.REPO / 'web' / 'public' / 'lab-data' / 'preconditioner'
HREF = '/lab/preconditioner/{}'
PRIMARY_VARIANTS = ('baseline', 'frozen-pt-tight', 'frozen-legacy')
# Order and labels follow the lab's tab bar (web/app/(docs)/lab/preconditioner/_lab/pages.ts).
STAGES = (
    ('baseline', 'Baseline', 'Work, latency and stage time shares of the current solver'),
    ('conditioning', 'Conditioning', 'Spectral conditioning of the preconditioned operator'),
    ('ablation', 'History ablation', 'Eigensolver history modes under the baseline and best preconditioner'),
    ('tuning', 'Tuning', 'Preconditioner parameter sweeps against the baseline'),
    ('multigrid', 'Multigrid', 'Multilevel preconditioners against the baseline'),
    ('ml', 'ML-ready', 'Parametric preconditioner export, offline harness and proof of concept'),
)
SKIPPED: set[str] = set()
TITLES = dict(index='Preconditioner lab', runs='Runs', **{s: label for s, label, _ in STAGES})
# Stage-specific additions: EXTENSIONS[page](studies, content) may add tiles, charts, tables or notes.
EXTENSIONS: dict[str, Callable[[list['Study'], 'Content'], None]] = {}


def _alive(pid: int | None) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return bool(pid)


class Study:
    def __init__(self, root: Path):
        self.root, self.name = root, root.name
        self.manifest = json.loads((root / 'manifest.json').read_text())
        self.kind, self.stage = self.manifest.get('kind'), self.manifest.get('stage')
        status = self.manifest.get('status')
        self.status = 'interrupted' if status == 'running' and not _alive(self.manifest.get('pid')) else status
        self.started = self.manifest.get('started') or ''
        self.variants = list(self.manifest.get('variants', {}))

    def _json(self, name: str) -> Any:
        path = self.root / name
        return json.loads(path.read_text()) if path.exists() else None

    @functools.cached_property
    def records(self) -> list[dict[str, Any]]:
        """Latest record per case, variant, tag and binary; every attempt for timing studies."""
        if self.kind == 'timing':
            return runlib.read_jsonl(self.root / 'runs.jsonl')
        name = 'profile_records.jsonl' if self.kind == 'profile' else 'records.jsonl'
        latest = {(r['case_id'], r['variant'], r.get('tag'), r['binary']['sha256']): r
                  for r in runlib.read_jsonl(self.root / name)}
        return list(latest.values())

    @functools.cached_property
    def timing(self) -> dict[str, Any] | None:
        return self._json('summary.json') if self.kind == 'timing' else None

    @functools.cached_property
    def profile(self) -> dict[str, Any] | None:
        return self._json('profile_summary.json') if self.kind == 'profile' else None

    @functools.cached_property
    def determinism(self) -> dict[str, Any] | None:
        return self._json('determinism.json') if self.kind == 'determinism' else None


def load_studies(results: Path, exclude: list[str]) -> list[Study]:
    roots = sorted(p.parent for p in results.glob('*/manifest.json'))
    studies = [Study(r) for r in roots if not any(fnmatch.fnmatchcase(r.name, g) for g in exclude)]
    return sorted(studies, key=lambda s: (s.started, s.name))


@dataclasses.dataclass
class Content:
    tiles: list = dataclasses.field(default_factory=list)
    charts: list = dataclasses.field(default_factory=list)
    tables: list = dataclasses.field(default_factory=list)
    notes: list = dataclasses.field(default_factory=list)

    def __iadd__(self, other: 'Content') -> 'Content':
        for field in dataclasses.fields(self):
            getattr(self, field.name).extend(getattr(other, field.name))
        return self


def finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def tile(label: str, value: Any, unit: str | None = None, *, delta: float | None = None,
         delta_label: str | None = None, ci: list[float] | None = None, tone: str | None = None,
         note: str | None = None) -> dict[str, Any]:
    return dict(label=label, value=value, unit=unit, delta=delta, delta_label=delta_label, ci=ci,
                tone=tone, note=note)


def column(key: str, label: str, fmt: str = 'text') -> dict[str, str]:
    return dict(key=key, label=label, format=fmt)


def table(id: str, title: str, columns: list[dict], rows: list[dict], caption: str | None = None) -> dict:
    return dict(id=id, title=title, columns=columns, rows=rows, caption=caption)


def bar(label: str, value: float, ci: list[float] | None = None, flag: str | None = None) -> dict:
    return dict(label=label, value=value, ci=ci, flag=flag)


def chart(kind: str, id: str, title: str, caption: str, **fields: Any) -> dict:
    """Chart dictionary of one renderer kind; callers pass that kind's remaining keys."""
    defaults = dict(ratio_bars=dict(reference=1.0, log_y=True), bars=dict(log_y=False, stacked=False),
                    lines=dict(log_x=True, log_y=True), heatmap=dict(log_color=False),
                    scatter=dict(log_x=False, log_y=False))[kind]
    return dict(id=id, kind=kind, title=title, caption=caption, **(defaults | fields))


def source(studies: list[Study]) -> dict[str, Any]:
    binaries = {}
    for study in studies:
        for binary in study.manifest.get('binaries', {}).values():
            binaries[binary['sha256']] = dict(name=binary['name'], revision=binary.get('revision') or 'unknown',
                                              sha256=binary['sha256'])
    return dict(runlib.git_state(), binaries=list(binaries.values()))


def page(name: str, studies: list[Study], status: str, content: Content, subtitle: str | None = None,
         **extra: Any) -> dict[str, Any]:
    if name in EXTENSIONS:
        EXTENSIONS[name](studies, content)
    return finite(dict(schema=SCHEMA, page=name, title=TITLES[name], subtitle=subtitle,
                       generated_at=runlib.utc_now(), source=source(studies), status=status,
                       **dataclasses.asdict(content), **extra))


def per_k(record: dict[str, Any]) -> float | None:
    return record['iterations_total'] / record['points'] if record.get('points') else None


def contrast(params: dict[str, Any]) -> float:
    return max(params['eps_rod'], params['eps_bg']) / min(params['eps_rod'], params['eps_bg'])


def group_key(params: dict[str, Any]) -> tuple[str, int]:
    return params['polarization'], params['resolution']


def geometric(values: list[float]) -> float:
    return math.exp(statistics.fmean(map(math.log, values)))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def tone(ci: list[float]) -> str:
    return 'good' if ci[0] > PRACTICAL else 'bad' if ci[1] < 1 / PRACTICAL else 'neutral'


def median_lines(rows: list[tuple[dict, float]]) -> list[dict]:
    """One series per lattice x polarization: median over cases at each N, min-max band."""
    grouped: dict[tuple, dict[int, list[float]]] = {}
    for params, value in rows:
        key = ('sq' if params['lattice'] == 'square' else 'tri', params['polarization'])
        grouped.setdefault(key, {}).setdefault(params['resolution'], []).append(value)
    return [dict(label=f'{lattice} {polarization}', dashed=polarization == 'TM',
                 points=[dict(x=n, y=statistics.median(v), lo=min(v), hi=max(v)) for n, v in sorted(by_n.items())])
            for (lattice, polarization), by_n in sorted(grouped.items())]


def timing_comparison(study: Study) -> Content:
    summary, out = study.timing, Content()
    a, b = summary['variants']
    metric = summary['primary_metric']
    strict, loose = summary['tiers']['strict'][metric], summary['tiers']['matched'][metric]
    if not strict and not loose:
        out.notes.append(f"{study.name}: no admitted pair yet ({summary['counts']['contaminated_runs']} "
                         'contaminated runs); no speed-up is reported.')
        return out
    if strict:
        # The renderer reads a tile delta as a signed change, so the time-weighted ratio goes in the note.
        out.tiles.append(tile(f'Speed-up {b} over {a}', strict['geometric_mean'], 'x', ci=strict['ci95_geometric'],
                              tone=tone(strict['ci95_geometric']),
                              note=f"{strict['cases']} cases, strict-clean tier, {metric} seconds; "
                                   f"time-weighted {strict['time_weighted']:.3g}x"))
    flag = 'contaminated' if summary['sensitive_to_environment'] else None
    groups: dict[tuple, dict[str, list[float]]] = {}
    for row in summary['per_case']:
        for tier in ('strict', 'matched'):
            if row.get(tier):
                groups.setdefault(group_key(row['params']), {}).setdefault(tier, []).append(row[tier])
    chart_groups = [dict(label=f'{p} N={n}', bars=[bar(t, geometric(v), flag=flag if t == 'matched' else None)
                                                   for t, v in tiers.items()])
                    for (p, n), tiers in sorted(groups.items())]
    overall = [bar(t, s['geometric_mean'], s['ci95_geometric'], flag if t == 'matched' else None)
               for t, s in (('strict', strict), ('matched', loose)) if s]
    if overall:
        chart_groups.append(dict(label='all cases', bars=overall))
    out.charts.append(chart('ratio_bars', f'{study.name}-speedup', f'{b} vs {a}: speed-up',
                            f'seconds({a}) / seconds({b}) per case, geometric mean per group; all-cases bars '
                            'carry the case-bootstrap 95% interval', y_label='speed-up', groups=chart_groups))
    resolutions = sorted({r['params']['resolution'] for r in summary['per_case']})

    def seconds_at(variant: str, n: int) -> float | None:
        return median([r['seconds'][variant] for r in summary['per_case']
                       if r['seconds'].get(variant) and r['params']['resolution'] == n])
    out.charts.append(chart('bars', f'{study.name}-seconds', f'{study.name}: seconds per diagram',
                            f'median admitted {metric} seconds per diagram at each resolution',
                            y_label='seconds', log_y=True, categories=[f'N={n}' for n in resolutions],
                            series=[dict(label=v, values=[seconds_at(v, n) for n in resolutions]) for v in (a, b)]))
    rows = [dict(label=r['label'], strict=r.get('strict'), matched=r.get('matched'),
                 seconds_a=r['seconds'].get(a), seconds_b=r['seconds'].get(b)) for r in summary['per_case']]
    out.tables.append(table(f'{study.name}-cases', f'{study.name}: per-case timing',
                            [column('label', 'Case'), column('strict', 'Strict ratio', 'float3'),
                             column('matched', 'Matched ratio', 'float3'),
                             column('seconds_a', f'{a} s', 'seconds'), column('seconds_b', f'{b} s', 'seconds')],
                            rows, f"{summary['counts']['contaminated_runs']} contaminated runs"))
    if summary['sensitive_to_environment']:
        out.notes.append(f'{study.name}: strict and paired-matched tiers differ by more than 10%; '
                         'the result is sensitive to the environment.')
    if not summary['clock']['clock_qualified']:
        out.notes.append(f'{study.name}: solver timer not clock-qualified; ratios use raw process seconds.')
    return out


def work_comparison(study: Study) -> Content:
    """Iteration ratios of every later variant against the study's first variant."""
    out = Content()
    reference, others = study.variants[0], study.variants[1:]
    by_case: dict[str, dict[str, dict]] = {}
    for record in study.records:
        if record['validation']['completed'] and record['iterations_total']:
            by_case.setdefault(record['case_id'], {})[record['variant']] = record
    for other in others:
        groups: dict[tuple, dict[str, tuple[list, list]]] = {}
        for case_id, runs in by_case.items():
            if reference in runs and other in runs:
                groups.setdefault(group_key(runs[reference]['params']), {})[case_id] = (
                    [runs[reference]['iterations_total']], [runs[other]['iterations_total']])
        bars = []
        for (p, n), data in sorted(groups.items()):
            stats = ratio_statistics(data, resamples=2000)
            bars.append(dict(label=f'{p} N={n}', bars=[bar(other, stats['geometric_mean'], stats['ci95_geometric'])]))
        if bars:
            out.charts.append(chart('ratio_bars', f'{study.name}-{other}-work', f'{other} vs {reference}: iterations',
                                    f'iterations({reference}) / iterations({other}) per case; geometric mean per '
                                    'group with case-bootstrap 95% interval', y_label='iteration ratio', groups=bars))
    resolutions = sorted({r['params']['resolution'] for runs in by_case.values() for r in runs.values()})

    def iterations_at(variant: str, n: int) -> float | None:
        return median([runs[variant]['iterations_total'] for runs in by_case.values()
                       if variant in runs and runs[variant]['params']['resolution'] == n])
    out.charts.append(chart('bars', f'{study.name}-iterations', f'{study.name}: iterations per diagram',
                            'median total iterations per diagram at each resolution', y_label='iterations',
                            categories=[f'N={n}' for n in resolutions],
                            series=[dict(label=v, values=[iterations_at(v, n) for n in resolutions])
                                    for v in study.variants]))
    rows = []
    for runs in sorted(by_case.values(), key=lambda runs: next(iter(runs.values()))['label']):
        row = dict(label=next(iter(runs.values()))['label'])
        row.update({f'it_{v}': runs[v]['iterations_total'] if v in runs else None for v in study.variants})
        row.update({f'ratio_{v}': runs[reference]['iterations_total'] / runs[v]['iterations_total']
                    if reference in runs and v in runs else None for v in others})
        rows.append(row)
    out.tables.append(table(f'{study.name}-work', f'{study.name}: per-case iterations',
                            [column('label', 'Case')] + [column(f'it_{v}', v, 'int') for v in study.variants]
                            + [column(f'ratio_{v}', f'{reference}/{v}', 'float3') for v in others], rows,
                            'iteration counts are deterministic and independent of host load'))
    return out


def comparisons(studies: list[Study], stage: str) -> Content:
    out = Content()
    for study in studies:
        if study.stage != stage:
            continue
        if study.kind == 'timing' and study.timing and study.timing['mode'] == 'pair':
            out += timing_comparison(study)
        elif study.kind == 'work' and len(study.variants) >= 2:
            out += work_comparison(study)
    return out


def work_records(studies: list[Study]) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest work record per (case, variant) across work studies, oldest study first."""
    return {(r['case_id'], r['variant']): r for s in studies if s.kind == 'work'
            for r in s.records if r.get('tag', 'work') == 'work'}


def primary_variant(records: dict[tuple[str, str], dict]) -> str | None:
    counts: dict[str, int] = {}
    for _, variant in records:
        counts[variant] = counts.get(variant, 0) + 1
    preferred = [v for v in PRIMARY_VARIANTS if v in counts]
    return preferred[0] if preferred else max(counts, key=counts.get, default=None)


def single_seconds(studies: list[Study]) -> dict[str, float]:
    """Median admitted seconds per case from single-variant timing (strict tier, else matched)."""
    seconds = {}
    for study in studies:
        if study.kind == 'timing' and study.timing and study.timing['mode'] == 'single':
            tiers, metric = study.timing['tiers'], study.timing['primary_metric']
            stats = tiers['strict'][metric] or tiers['matched'][metric]
            seconds.update(stats['per_case'] if stats else {})
    return seconds


def work_section(completed: list[dict], total: int, variant: str) -> Content:
    out = Content()
    out.tiles += [tile('Cases', len(completed), note=f'of {len(cases.full_matrix())} in full768; variant {variant}'),
                  tile('Median iterations per diagram', median([r['iterations_total'] for r in completed])),
                  tile('Total iterations', sum(r['iterations_total'] for r in completed)),
                  tile('Failed runs', total - len(completed), tone='bad' if total > len(completed) else 'good')]
    out.charts.append(chart('lines', 'work-vs-n', 'Iterations per diagram vs N', 'median over radii and permittivity '
                            'pairs; band spans min to max', x_label='N', y_label='total iterations',
                            series=median_lines([(r['params'], r['iterations_total']) for r in completed])))
    resolutions = {r['params']['resolution'] for r in completed}
    chosen = 32 if 32 in resolutions else min(resolutions)
    for polarization in ('TE', 'TM'):
        cells = [[mean([per_k(r) for r in completed if (r['params']['resolution'], r['params']['polarization'],
                                                        r['params']['eps_rod'], r['params']['eps_bg'])
                        == (chosen, polarization, rod, bg)]) for bg in cases.EPS_BG] for rod in cases.EPS_ROD]
        if any(v is not None for row in cells for v in row):
            out.charts.append(chart('heatmap', f'iterations-per-k-{polarization}',
                                    f'{polarization}: iterations per k-point at N={chosen}',
                                    'mean over lattices and radii; empty cells were not run',
                                    x_label='background permittivity', y_label='rod permittivity',
                                    x=[f'{bg:g}' for bg in cases.EPS_BG], y=[f'{rod:g}' for rod in cases.EPS_ROD],
                                    values=cells, value_label='iterations per k'))
    out.charts.append(chart('scatter', 'work-vs-contrast', 'Iterations per k-point vs contrast',
                            'each point is one case; contrast is max(eps)/min(eps)', x_label='contrast',
                            y_label='iterations per k', log_x=True,
                            series=[dict(label=p, points=[dict(x=contrast(r['params']), y=per_k(r), tag=r['label'])
                                                          for r in completed if r['params']['polarization'] == p])
                                    for p in ('TE', 'TM')]))
    return out


def profile_section(profiles: list[Study], variant: str | None) -> Content:
    groups = [g for s in profiles for g in s.profile['groups'] if g['variant'] == variant]
    groups = groups or profiles[-1].profile['groups']
    categories = profiles[-1].profile['categories']
    return Content(charts=[chart(
        'bars', 'profile-shares', 'Solver time shares', 'exclusive profiler time per stage summed over cases '
        '(instrumented build, shares only; residuals inside the iteration count as other)',
        y_label='share of timed time (%)', stacked=True,
        categories=[f"{g['variant']} {g['polarization']} N={g['resolution']}" for g in groups],
        series=[dict(label=c.replace('_', ' '), values=[100 * g['shares'][c] if g['shares'][c] is not None else None
                                                        for g in groups]) for c in categories])])


def case_table(work: list[dict], seconds: dict[str, float], variant: str) -> dict:
    rows = [dict(label=r['label'], lattice=r['params']['lattice'], polarization=r['params']['polarization'],
                 resolution=r['params']['resolution'], radius=r['params']['radius'],
                 eps_rod=r['params']['eps_rod'], eps_bg=r['params']['eps_bg'], iterations=r['iterations_total'],
                 per_k=per_k(r), residual=(r.get('residuals') or {}).get('max_relative_nonzero'),
                 seconds=seconds.get(r['case_id']), wall=r['wall_seconds'],
                 flags=', '.join(r['validation']['flags']) or 'ok') for r in work]
    return table('cases', f'Per-case baseline ({variant})',
                 [column('label', 'Case'), column('lattice', 'Lattice'), column('polarization', 'Pol.'),
                  column('resolution', 'N', 'int'), column('radius', 'r/a', 'float2'),
                  column('eps_rod', 'eps rod', 'float2'), column('eps_bg', 'eps bg', 'float2'),
                  column('iterations', 'Iterations', 'int'), column('per_k', 'Per k', 'float2'),
                  column('residual', 'Max residual', 'sci'), column('seconds', 'Timed s', 'seconds'),
                  column('wall', 'Sweep wall s', 'seconds'), column('flags', 'Flags')], rows,
                 'sweep wall time comes from the parallel work pool and is not latency evidence')


def baseline_page(studies: list[Study]) -> dict[str, Any]:
    stage = [s for s in studies if s.stage == 'baseline']
    records = work_records(stage)
    variant = primary_variant(records)
    work = [r for (_, v), r in sorted(records.items()) if v == variant]
    completed = [r for r in work if r['validation']['completed']]
    seconds = single_seconds(stage)
    profiles = [s for s in stage if s.kind == 'profile' and s.profile and s.profile['groups']]
    out = Content()
    if seconds:
        out.tiles.append(tile('Median seconds per diagram', median(list(seconds.values())), 's',
                              note=f'{len(seconds)} cases, single-threaded, admitted timing runs'))
        timed = [(r['params'], seconds[r['case_id']]) for r in work if r['case_id'] in seconds]
        if timed:
            out.charts.append(chart('lines', 'seconds-vs-n', 'Seconds per diagram vs N', 'admitted single-threaded '
                                    'timing; median over cases, band spans min to max', x_label='N',
                                    y_label='seconds', series=median_lines(timed)))
    else:
        out.notes.append('No admitted single-variant timing yet: run run_timing.py --stage baseline with one variant.')
    if completed:
        out += work_section(completed, len(work), variant)
    else:
        out.notes.append('No baseline work sweep yet: run run_work.py --stage baseline on dev96 or full768.')
    if profiles:
        out += profile_section(profiles, variant)
    else:
        out.notes.append('No profile yet: run run_profile.py --stage baseline with the profiling binary.')
    if work:
        out.tables.append(case_table(work, seconds, variant))
    extra = comparisons(studies, 'baseline')
    out += extra
    present = [bool(completed), bool(seconds), bool(profiles)]
    status = 'complete' if all(present) else 'partial' if any(present) or extra.charts else 'empty'
    return page('baseline', stage, status, out, subtitle=f'variant {variant}' if variant else None)


def comparison_page(name: str, studies: list[Study]) -> dict[str, Any]:
    stage = [s for s in studies if s.stage == name]
    out = comparisons(studies, name)
    if not out.charts:
        out.notes.append(f'No {name} comparison yet: run run_work.py or run_timing.py with --stage {name} '
                         'and a reference plus candidate variant.')
    status = 'empty' if not out.charts else 'complete' if all(s.status == 'complete' for s in stage) else 'partial'
    return page(name, stage, status, out, subtitle=dict((s, d) for s, _, d in STAGES)[name])


def run_rows(study: Study) -> list[dict[str, Any]]:
    """One runs-page row per variant (per variant pair for timing studies)."""
    binaries = study.manifest.get('binaries', {})
    if study.kind == 'timing':
        groups = {' vs '.join(study.variants): [r for r in study.records if r.get('tag') == 'timing']}
    else:
        groups = {v: [r for r in study.records if r['variant'] == v] for v in study.variants}
    rows = []
    for variant, records in groups.items():
        notes = []
        if study.kind == 'work':
            notes.append('work counts are load-independent; parallel wall time is not latency')
        if study.determinism:
            notes.append('deterministic' if study.determinism['deterministic']
                         else f"{len(study.determinism['mismatches'])} determinism mismatches")
        if failed := sum(not r['validation']['completed'] for r in records):
            notes.append(f'{failed} failed runs')
        verdicts = [(r.get('load') or {}).get('verdict') for r in records]
        timing = study.kind == 'timing'
        rows.append(dict(id=f'{study.name}:{variant}', study=study.name, started=study.started, variant=variant,
                         subset=study.manifest.get('subset', ''), cases=len({r['case_id'] for r in records}),
                         status='complete' if study.status == 'complete' else
                         'failed' if study.status == 'preflight_failed' else 'partial',
                         contaminated=sum(v in ('contaminated', 'unknown') for v in verdicts) if timing else 0,
                         clean=verdicts.count('clean') if timing else len(records),
                         wall_seconds=sum(r['wall_seconds'] for r in records) if records else None,
                         binary=(binaries.get(variant) or next(iter(binaries.values()), {})).get('name', ''),
                         notes='; '.join(notes) or None))
    return rows


def runs_page(studies: list[Study]) -> dict[str, Any]:
    runs = [row for study in studies for row in run_rows(study)]
    out = Content(tiles=[tile('Studies', len(studies)), tile('Runs recorded', sum(len(s.records) for s in studies)),
                         tile('Contaminated timing runs', sum(r['contaminated'] for r in runs), tone='neutral')])
    # The lab draws `runs` as its run log, so no table repeats it.
    if studies:
        out.notes.append('Clean counts load-admitted timing runs, or recorded runs for load-independent work '
                         'studies; wall time sums per-run process wall time.')
    else:
        out.notes.append('No studies under benchmarks/preconditioner/results yet.')
    status = 'empty' if not studies else 'complete' if all(r['status'] == 'complete' for r in runs) else 'partial'
    return page('runs', studies, status, out, runs=runs)


def stage_status(stage: str, studies: list[Study]) -> str:
    if stage in SKIPPED:
        return 'skipped'
    own = [s for s in studies if s.stage == stage]
    if any(s.status == 'running' for s in own):
        return 'running'
    return 'done' if any(s.status == 'complete' for s in own) else 'pending'


def index_page(studies: list[Study], pages: dict[str, dict]) -> dict[str, Any]:
    stages = []
    for stage, label, description in STAGES:
        kinds: dict[str, int] = {}
        for study in studies:
            if study.stage == stage:
                kinds[study.kind] = kinds.get(study.kind, 0) + 1
        stages.append(dict(id=stage, label=label, status=stage_status(stage, studies), href=HREF.format(stage),
                           summary=', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) or description))
    timeline = []
    for study in studies:
        detail = f"{study.kind} on {study.manifest.get('subset')} with {', '.join(study.variants)}"
        timeline.append(dict(when=study.started, label=f'{study.name} started', detail=detail))
        if study.manifest.get('ended'):
            timeline.append(dict(when=study.manifest['ended'], label=f'{study.name} {study.status}', detail=None))
    timeline.sort(key=lambda event: event['when'])
    # The lab shows the first tile as the overview's headline: the latest speed-up, else the baseline.
    out = Content()
    latest = next((s for s in reversed(studies) if s.kind == 'timing' and s.timing and s.timing['mode'] == 'pair'
                   and s.timing['tiers']['strict'][s.timing['primary_metric']]), None)
    if latest:
        out.tiles += timing_comparison(latest).tiles
    out.tiles += [t for t in pages['baseline']['tiles'] if t['label'] in
                  ('Median seconds per diagram', 'Median iterations per diagram', 'Total iterations')]
    contaminated = sum(r['contaminated'] for r in pages['runs']['runs'])
    out.tiles += [tile('Studies', len(studies)), tile('Contaminated timing runs', contaminated, tone='neutral')]
    out.charts = [c for name, _, _ in STAGES for c in pages[name]['charts'] if c['kind'] == 'ratio_bars'][:2]
    if not studies:
        out.notes.append('No studies yet. Stages fill in as work sweeps, timing and profiles complete.')
    statuses = [s['status'] for s in stages]
    status = 'empty' if not studies else 'complete' if all(s in ('done', 'skipped') for s in statuses) else 'partial'
    return page('index', studies, status, out, subtitle='LOBPCG preconditioner study for 2D photonic band diagrams',
                stages=stages, timeline=timeline)


def build(results: Path, exclude: list[str]) -> dict[str, dict[str, Any]]:
    studies = load_studies(results, exclude) if results.is_dir() else []
    pages = {'baseline': baseline_page(studies)}
    pages.update({stage: comparison_page(stage, studies) for stage, _, _ in STAGES[1:]})
    pages['runs'] = runs_page(studies)
    pages['index'] = index_page(studies, pages)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--lab', action='store_true', help='write the lab page JSON files')
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--exclude', action='append', default=[], help='glob of study names to skip')
    args = parser.parse_args()
    pages = build(args.results, args.exclude)
    if args.lab:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, document in pages.items():
            temporary = args.out / f'.{name}.json.tmp'
            temporary.write_text(json.dumps(document, indent=1, allow_nan=False, ensure_ascii=False) + '\n')
            temporary.replace(args.out / f'{name}.json')
    for name, document in pages.items():
        print(f"{name:13} {document['status']:9} tiles={len(document['tiles'])} charts={len(document['charts'])} "
              f"tables={len(document['tables'])}")
    if args.lab:
        print(f'wrote {len(pages)} pages to {args.out}')


if __name__ == '__main__':
    main()
