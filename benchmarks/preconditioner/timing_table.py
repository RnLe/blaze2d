#!/usr/bin/env python3
"""One comparison table for a block timing study: speed-up, seconds, accuracy, projection.

Per variant: speed-up against the reference (geometric mean over cases with its 95%
case-bootstrap interval, and time-weighted), median seconds per diagram at each grid
size, the largest fresh relative residual any admitted run reported, iterations and FFT
work relative to the reference, and the full-matrix serial hours of cost_model.json
(per-stratum projection). Prints Markdown; writes timing_table.json into the study.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

import runlib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', required=True)
    args = parser.parse_args()
    root = runlib.RESULTS / args.study
    summary = json.loads((root / 'summary.json').read_text())
    variants = summary['variants']
    reference = variants[0]
    metric = summary['primary_metric']
    ratios = summary['tiers']['matched'][metric] or {}
    model = json.loads((root / 'cost_model.json').read_text()) if (root / 'cost_model.json').exists() else {}
    runs = {r['run_id']: r for r in runlib.read_jsonl(root / 'runs.jsonl') if 'run_id' in r}
    entries = [e for e in json.loads((root / 'pairs.json').read_text()) if e['tier']]
    resolutions = sorted({e['params']['resolution'] for e in entries})
    rows = []
    for name in variants:
        mine = [runs[e['run_ids'][name]] for e in entries if e['run_ids'].get(name) in runs]
        ref = [runs[e['run_ids'][reference]] for e in entries if e['run_ids'].get(name) in runs]
        certificates = [(r.get('certification') or {}).get('max_residual') for r in mine]
        certificates = [c for c in certificates if c is not None]
        work = lambda r, k: ((r.get('optional') or {}).get('work') or {}).get(k)
        pairs = [(work(a, 'fft_vectors'), work(b, 'fft_vectors')) for a, b in zip(ref, mine)]
        pairs = [(a, b) for a, b in pairs if a and b]
        iterations = [(a['iterations_total'], b['iterations_total']) for a, b in zip(ref, mine)
                      if a.get('iterations_total') and b.get('iterations_total')]
        stats = ratios.get(name) if name != reference else None
        seconds = {n: statistics.median([r['wall_seconds'] for r in mine if r['params']['resolution'] == n] or [float('nan')])
                   for n in resolutions}
        rows.append(dict(
            variant=name, runs=len(mine),
            speedup=stats['geometric_mean'] if stats else 1.0,
            ci=stats['ci95_geometric'] if stats else None,
            time_weighted=stats['time_weighted'] if stats else 1.0,
            seconds=seconds,
            max_residual=max(certificates) if certificates else None,
            median_residual=statistics.median(certificates) if certificates else None,
            iteration_ratio=float(np.exp(np.mean([np.log(b / a) for a, b in iterations]))) if iterations else None,
            fft_ratio=float(np.exp(np.mean([np.log(b / a) for a, b in pairs]))) if pairs else None,
            hours=(model.get('stratified') or {}).get(name, {}).get('hours'),
            model=(model.get('by_variant') or {}).get(name)))
    runlib.write_json(root / 'timing_table.json', dict(study=args.study, reference=reference, metric=metric, rows=rows))
    head = ['variant', 'speed-up', '95% CI', 'time-wt'] + [f's@{n}' for n in resolutions] + \
           ['max resid', 'iters', 'FFT', 'full h', 'model']
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in sorted(rows, key=lambda r: -r['speedup']):
        ci = f"{r['ci'][0]:.3f}–{r['ci'][1]:.3f}" if r['ci'] else ''
        model_text = f"{r['model']['predicted']:.3f}x" if r['model'] else ''
        cells = [r['variant'], f"{r['speedup']:.3f}x", ci, f"{r['time_weighted']:.3f}x"] + \
                [f"{r['seconds'][n]:.2f}" for n in resolutions] + \
                [f"{r['max_residual']:.1e}" if r['max_residual'] is not None else '',
                 f"{r['iteration_ratio']:.2f}" if r['iteration_ratio'] else '',
                 f"{r['fft_ratio']:.2f}" if r['fft_ratio'] else '',
                 f"{r['hours']:.2f}" if r['hours'] else '', model_text]
        print('| ' + ' | '.join(cells) + ' |')
    return 0


if __name__ == '__main__':
    sys.exit(main())
