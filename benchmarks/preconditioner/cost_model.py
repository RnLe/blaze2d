#!/usr/bin/env python3
"""Cost model from admitted timing runs, projected over full work sweeps.

Per resolution N, seconds of the admitted runs of a timing study are fitted as
t = c_F(N) * FFT vectors + c_G(N) * GEMM terms + c_I(N) * iterations + c_0(N) with
non-negative coefficients. The iteration term carries block-wide work every iteration
does that neither counter sees (residual norms, vector updates, locking); without it the
model misses configurations whose iterations and FFTs diverge, e.g. no history by 23%. The model then turns the deterministic work counters of full sweeps
into projected serial seconds per variant, so a timing subset calibrates claims
about the whole matrix. Per variant it compares the measured speed-up against the
reference (first variant) with the speed-up the model predicts from the same runs' work
counters and with the bare FFT ratio, which exposes cost the counters do not see.
Writes cost_model.json into the timing study.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.optimize as so

import runlib


def admitted_runs(study: Path) -> list[dict]:
    runs = {r['run_id']: r for r in runlib.read_jsonl(study / 'runs.jsonl') if 'run_id' in r}
    entries = json.loads((study / 'pairs.json').read_text())
    return [dict(runs[i], block=e['key']) for e in entries if e['tier'] for i in e['run_ids'].values() if i in runs]


def seconds(record: dict) -> float | None:
    return record.get('wall_seconds') or record.get('solver_seconds') or record.get('job_seconds')


def predict(models: dict[int, dict], record: dict) -> float | None:
    m, w = models.get(record['params']['resolution']), work(record)
    return (m['fft_seconds'] * w[0] + m['gemm_seconds'] * w[1] + m['iteration_seconds'] * w[2] + m['constant']
            if m and w else None)


def by_variant(models: dict[int, dict], runs: list[dict], reference: str) -> dict[str, dict]:
    """Measured, model-predicted and FFT-ratio speed-ups against the reference, per variant."""
    blocks: dict[str, dict[str, dict]] = {}
    for r in runs:
        blocks.setdefault(r['block'], {})[r['variant']] = r
    out = {}
    for variant in sorted({r['variant'] for r in runs} - {reference}):
        measured, predicted, fft, residual = [], [], [], []
        for block in blocks.values():
            a, b = block.get(reference), block.get(variant)
            if not (a and b and work(a) and work(b) and seconds(a) and seconds(b)):
                continue
            measured.append(seconds(a) / seconds(b))
            predicted.append(predict(models, a) / predict(models, b))
            fft.append(work(a)[0] / work(b)[0])
            residual.append(seconds(b) / predict(models, b) - 1)
        if measured:
            gm = lambda values: float(np.exp(np.mean(np.log(values))))
            out[variant] = dict(blocks=len(measured), measured=gm(measured), predicted=gm(predicted), fft=gm(fft),
                                mean_residual=float(np.mean(residual)))
    return out


def work(record: dict) -> tuple[float, float, float] | None:
    w = (record.get('optional') or {}).get('work') or {}
    iterations = record.get('iterations_total') or w.get('iterations')
    return (w['fft_vectors'], w['gemm_terms'], iterations) if w.get('fft_vectors') and w.get('gemm_terms') and iterations else None


def fit(runs: list[dict]) -> dict[int, dict]:
    models = {}
    for n in sorted({r['params']['resolution'] for r in runs}):
        rows = [(work(r), seconds(r)) for r in runs if r['params']['resolution'] == n and work(r) and seconds(r)]
        x = np.array([[f, g, i, 1.0] for (f, g, i), _ in rows])
        t = np.array([s for _, s in rows])
        scale = x.max(axis=0)
        coef, _ = so.nnls(x / scale, t)
        coef = coef / scale
        predicted = x @ coef
        models[n] = dict(fft_seconds=coef[0], gemm_seconds=coef[1], iteration_seconds=coef[2], constant=coef[3], runs=len(rows),
                         median_relative_error=float(np.median(np.abs(predicted - t) / t)),
                         max_relative_error=float(np.max(np.abs(predicted - t) / t)),
                         fft_share=float(np.sum(coef[0] * x[:, 0]) / np.sum(predicted)),
                         gemm_share=float(np.sum(coef[1] * x[:, 1]) / np.sum(predicted)),
                         iteration_share=float(np.sum(coef[2] * x[:, 2]) / np.sum(predicted)))
    return models


def project(models: dict[int, dict], records: list[dict]) -> dict:
    seconds, cases = 0.0, 0
    for r in records:
        t = predict(models, r)
        if t:
            seconds += t
            cases += 1
    return dict(seconds=seconds, cases=cases)


def stratified(models: dict[int, dict], runs: list[dict], reference: str, records: list[dict]) -> dict[str, dict]:
    """Full-matrix serial hours per variant from measured time ratios per stratum.

    The reference's model-predicted seconds of every full-sweep diagram are scaled by the
    geometric-mean measured ratio seconds(variant) / seconds(reference) of the timing
    blocks in the same lattice x polarization x N stratum, so variants without work
    counters (or without full sweeps) are projected too.
    """
    blocks: dict[str, dict[str, dict]] = {}
    for r in runs:
        blocks.setdefault(r['block'], {})[r['variant']] = r
    key = lambda params: (params['lattice'], params['polarization'], params['resolution'])
    base = {}
    for r in records:
        t = predict(models, r)
        if t:
            base.setdefault(key(r['params']), []).append(t)
    out = {}
    for variant in sorted({r['variant'] for r in runs}):
        ratios: dict[tuple, list[float]] = {}
        for block in blocks.values():
            a, b = block.get(reference), block.get(variant)
            if a and b and seconds(a) and seconds(b):
                ratios.setdefault(key(a['params']), []).append(seconds(b) / seconds(a))
        if set(ratios) >= set(base):
            hours = sum(sum(times) * float(np.exp(np.mean(np.log(ratios[k])))) for k, times in base.items()) / 3600
            out[variant] = dict(hours=hours, strata=len(base), diagrams=sum(len(t) for t in base.values()))
    return out


def sweep(spec: str) -> tuple[str, list[dict]]:
    """LABEL=STUDY:VARIANT[,STUDY:VARIANT...]; later sources win per case."""
    label, sources = spec.split('=', 1)
    chosen: dict[str, dict] = {}
    for source in sources.split(','):
        study, variant = source.split(':')
        for r in runlib.read_jsonl(runlib.RESULTS / study / 'records.jsonl'):
            if r['variant'] == variant and r.get('tag', 'work') == 'work' and r['validation']['completed']:
                chosen[r['case_id']] = r
    return label, list(chosen.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', required=True, help='timing study to calibrate on')
    parser.add_argument('--sweep', action='append', default=[], help='LABEL=STUDY:VARIANT[,STUDY:VARIANT]')
    parser.add_argument('--fit-variant', action='append', help='fit only on these variants (default: all)')
    parser.add_argument('--stratify', help='STUDY:VARIANT full sweep of the reference for per-stratum projections')
    args = parser.parse_args()
    root = runlib.RESULTS / args.study
    runs = admitted_runs(root)
    models = fit([r for r in runs if not args.fit_variant or r['variant'] in args.fit_variant])
    for n, m in models.items():
        print(f"N={n:3d} runs={m['runs']:3d} c_F={m['fft_seconds']:.3e} s c_G={m['gemm_seconds']:.3e} s "
              f"c_I={m['iteration_seconds']:.3e} s c_0={m['constant']:.3f} s  FFT {100 * m['fft_share']:.0f}% "
              f"GEMM {100 * m['gemm_share']:.0f}% iterations {100 * m['iteration_share']:.0f}%  "
              f"error median {100 * m['median_relative_error']:.1f}% max {100 * m['max_relative_error']:.1f}%")
    variants = json.loads((root / 'manifest.json').read_text())['variants']
    comparison = by_variant(models, runs, next(iter(variants)))
    for name, c in comparison.items():
        print(f"{name:18s} blocks={c['blocks']:3d} measured {c['measured']:.3f}x predicted {c['predicted']:.3f}x "
              f"FFT {c['fft']:.3f}x  time vs model {100 * c['mean_residual']:+.1f}%")
    strata = {}
    if args.stratify:
        _, records = sweep('reference=' + args.stratify)
        strata = stratified(models, runs, next(iter(variants)), records)
        for name, p in sorted(strata.items(), key=lambda item: item[1]['hours']):
            print(f"{name:22s} full matrix {p['hours']:.2f} serial hours (per-stratum measured ratios)")
    projections = {}
    for spec in args.sweep:
        label, records = sweep(spec)
        projections[label] = project(models, records)
        p = projections[label]
        print(f"{label}: {p['cases']} diagrams, projected {p['seconds'] / 3600:.2f} serial hours")
    if len(projections) >= 2:
        (a, pa), (b, pb) = list(projections.items())[:2]
        print(f'projected speed-up {b} over {a}: {pa["seconds"] / pb["seconds"]:.3f}x')
    runlib.write_json(root / 'cost_model.json', dict(models={str(n): m for n, m in models.items()},
                                                     fit_variants=args.fit_variant, by_variant=comparison,
                                                     stratified=strata,
                                                     projections=projections, sweeps=args.sweep))
    return 0


if __name__ == '__main__':
    sys.exit(main())
