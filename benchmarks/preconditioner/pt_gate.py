"""Gate PT-real arithmetic against complex arithmetic on a work study.

For every case solved with both variants: no rejected solver attempt, every
non-reused point retained by the Fourier/PT solve, PT leakage within tolerance,
converged, fresh certificates within the acceptance gate, and frequencies equal
to the complex solve within a relative tolerance.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import runlib


def raw_output(root: Path, record: dict) -> dict | None:
    sha = record.get('output_sha256')
    if not sha:
        return None
    with gzip.open(root / 'objects' / f'{sha}.json.gz', 'rt') as f:
        return json.load(f)


def exceeds(value, limit: float) -> bool:
    """True unless value is a number within limit (NaN and missing values fail)."""
    return not (value is not None and value <= limit)


def pt_checks(document: dict) -> dict:
    result = document['results'][0]
    meta = result['metadata']
    points = meta['symmetry']['points']
    rejected = sum(1 for p in points for a in p.get('solver_attempts', []) if not a.get('retained'))
    not_pt = [i for i, p in enumerate(points) if p.get('source_index') is None
              and not any(a.get('retained') and a.get('method') == 'fourier' for a in p.get('solver_attempts', []))]
    leakage = max((p.get('coordinates') or {}).get('maximum_real_subspace_leakage', 0.0) or 0.0 for p in points)
    tolerance = (meta.get('eigensolver') or {}).get('residual_tolerance')
    arrays = result['arrays']
    shape = arrays['frequencies']['shape']
    frequencies = np.asarray(arrays['frequencies']['data'], float).reshape(shape)
    relative = np.asarray(arrays['residuals']['data'], float).reshape(shape)
    absolute = np.asarray(arrays['absolute_residuals']['data'], float).reshape(shape)
    zero = frequencies == 0.0
    max_relative = float(relative[~zero].max()) if (~zero).any() else 0.0
    max_zero_mode = float(absolute[zero].max()) if zero.any() else 0.0
    return {
        'converged': meta.get('converged') is True,
        'rejected_attempts': rejected,
        'non_pt_points': not_pt,
        'max_leakage': leakage,
        'max_residual': max_relative,
        'max_zero_mode_residual': max_zero_mode,
        'fresh_over_stopping': max_relative / tolerance if tolerance else None,
        'iterations': sum(meta['iterations']),
        'seconds': meta.get('elapsed_seconds'),
    }


def frequency_gap(a: dict, b: dict) -> float:
    fa = np.sort(np.asarray(a['results'][0]['arrays']['frequencies']['data'], float).reshape(-1, a['results'][0]['arrays']['frequencies']['shape'][-1]), axis=1)
    fb = np.sort(np.asarray(b['results'][0]['arrays']['frequencies']['data'], float).reshape(fa.shape), axis=1)
    mask = fa > 1e-6
    return float(np.max(np.abs(fa[mask] - fb[mask]) / fa[mask])) if mask.any() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', required=True)
    parser.add_argument('--pt', default='baseline')
    parser.add_argument('--ref', default='complex-baseline')
    parser.add_argument('--leakage', type=float, default=1e-10)
    parser.add_argument('--residual', type=float, default=1e-5)
    parser.add_argument('--zero-mode', type=float, default=1e-7)
    parser.add_argument('--frequency', type=float, default=1e-9)
    parser.add_argument('--drift', type=float, default=10.0)
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    root = args.results / args.study
    records = runlib.read_jsonl(root / 'records.jsonl')
    latest: dict[tuple, dict] = {}
    for r in records:
        latest[(r['case_id'], r['variant'])] = r
    rows, failures = [], []
    for (case, variant), record in sorted(latest.items()):
        if variant != args.pt:
            continue
        pt = raw_output(root, record)
        ref_record = latest.get((case, args.ref))
        ref = raw_output(root, ref_record) if ref_record else None
        if not (pt or {}).get('results'):
            failures.append({'case': record['label'], 'reasons': ['no PT output']})
            continue
        if not (ref or {}).get('results'):
            failures.append({'case': record['label'], 'reasons': [f'no {args.ref} reference output']})
            continue
        row = {'case': record['label'], **pt_checks(pt)}
        row['frequency_gap'] = frequency_gap(pt, ref)
        row['ref_iterations'] = sum(ref['results'][0]['metadata']['iterations'])
        row['ref_seconds'] = ref['results'][0]['metadata'].get('elapsed_seconds')
        reasons = []
        if not row['converged']:
            reasons.append('not converged')
        if row['rejected_attempts']:
            reasons.append(f"{row['rejected_attempts']} rejected attempts")
        if row['non_pt_points']:
            reasons.append(f"non-PT points {row['non_pt_points']}")
        if exceeds(row['max_leakage'], args.leakage):
            reasons.append(f"leakage {row['max_leakage']}")
        if exceeds(row['max_residual'], args.residual):
            reasons.append(f"residual {row['max_residual']}")
        if exceeds(row['max_zero_mode_residual'], args.zero_mode):
            reasons.append(f"zero-mode absolute residual {row['max_zero_mode_residual']}")
        if row['fresh_over_stopping'] is not None and exceeds(row['fresh_over_stopping'], args.drift):
            reasons.append(f"fresh/stopping {row['fresh_over_stopping']}")
        if exceeds(row['frequency_gap'], args.frequency):
            reasons.append(f"frequency gap {row['frequency_gap']}")
        row['reasons'] = reasons
        rows.append(row)
        if reasons:
            failures.append({'case': row['case'], 'reasons': reasons})
    summary = {
        'study': args.study, 'cases': len(rows), 'failures': failures,
        'max_leakage': max((r['max_leakage'] for r in rows), default=None),
        'max_residual': max((r['max_residual'] for r in rows), default=None),
        'max_frequency_gap': max((r['frequency_gap'] for r in rows), default=None),
        'max_fresh_over_stopping': max((r['fresh_over_stopping'] or 0.0 for r in rows), default=None),
        'iterations_pt': sum(r['iterations'] for r in rows),
        'iterations_ref': sum(r['ref_iterations'] or 0 for r in rows),
        'passed': not failures and bool(rows),
    }
    runlib.write_json(root / 'pt_gate.json', {'summary': summary, 'rows': rows})
    print(json.dumps(summary, indent=1))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
