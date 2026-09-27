"""Independent physical and subspace checks of public coarse-start WASM fields.

The FFT pencil is shared with the standalone PT-sector verifier, not the solver.
Smoothed TE is excluded from the operator audit because the geometry preview
does not export the smoothed inverse tensor. Raw TE and all TM are audited.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_wasm_pt_sectors import certificate, compare, passes


def audit(document, raw):
    issues = list(document['issues'])
    checks, excluded, comparisons, negatives = [], [], [], []
    records = {r['name']: r for r in document['records']}
    if len(records) != 24 or len(document['records']) != 24 or len(document['comparisons']) != 16:
        issues.append('incomplete coarse-start calculation/comparison matrix')
    if len(document['negatives']) != 3 or not all(n['rejected'] for n in document['negatives']):
        issues.append('incomplete configuration negatives')
    for name, item in records.items():
        if item['result'] is None or item['failure'] is not None:
            issues.append(name + ': failed calculation')
            continue
        for key, array in item['result']['arrays'].items():
            values = np.asarray(array['data'], dtype='<f8')
            if hashlib.sha256(values.tobytes()).hexdigest() != item['array_sha256'][key]:
                issues.append(name + '/' + key + ': array hash mismatch')
        if item['config']['polarization'] == 'TE' and item['config']['dielectric']['smoothing'] != 'none':
            excluded.append(dict(name=name, points=len(item['config']['bands']['path']['points']),
                reason='public preview omits inverse tensor; reported fresh certificates and subspace checks only'))
            continue
        for i in range(len(item['config']['bands']['path']['points'])):
            try:
                c = certificate(item, i)
                if not passes(c):
                    issues.append(f'{name}/{i}: independent physical gate failed')
                q = item['config']['bands']['count']
                arrays = item['result']['arrays']
                reported = np.asarray(arrays['residuals']['data']).reshape(-1, q)[i]
                absolute = np.asarray(arrays['absolute_residuals']['data']).reshape(-1, q)[i]
                nonzero = [j for j in range(q) if j not in c['zero_indices']]
                delta = max((abs(c['residuals'][j] - reported[j]) for j in nonzero), default=0)
                adelta = float(np.max(abs(np.asarray(c['absolute_residuals']) - absolute)))
                if delta > 1e-8 or adelta > 1e-7:
                    issues.append(f'{name}/{i}: independent/reported residual disagreement')
                checks.append(dict(name=name, point=i, certificate=c,
                    max_reported_relative_difference=delta, max_reported_absolute_difference=adelta))
            except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
                issues.append(f'{name}/{i}: {error}')
    for pair in document['comparisons']:
        a, b = records[pair['candidate']], records[pair['reference']]
        if a['result'] is None or b['result'] is None:
            continue
        ca, cb = copy.deepcopy(a['config']), copy.deepcopy(b['config'])
        for config in (ca, cb):
            for key in ('arithmetic', 'sectors', 'sector_guard_budget', 'coarse_start'):
                config['bands']['symmetry'].pop(key, None)
        if ca != cb or a['preview'] != b['preview']:
            issues.append(a['name'] + ': physical comparison contract changed')
        for i in range(len(a['config']['bands']['path']['points'])):
            c = compare(a, b, i)
            if c['max_scaled_frequency_error'] > 1e-7 or c['max_sin_angle'] > 1e-3:
                issues.append(f"{a['name']}/{i}: spectrum/subspace disagreement")
            comparisons.append(dict(candidate=a['name'], reference=b['name'], point=i, **c))
    for polarization in ('TM', 'TE'):
        item = copy.deepcopy(records[f'none/{polarization}/coarse/pt-false/sector-false'])
        if item['result'] is None:
            continue
        item['result']['arrays']['eigenvectors']['data'][0] += 0.25
        rejected = not passes(certificate(item, 0))
        if not rejected:
            issues.append(polarization + ': corrupted field escaped independent audit')
        negatives.append(dict(polarization=polarization, corrupted_field_rejected=rejected,
            reported_metadata_unchanged=True))
    return dict(schema='blaze2d/wasm-coarse-audit/1', input_sha256=hashlib.sha256(raw).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        operator_verifier_sha256=hashlib.sha256(Path(__file__).with_name('audit_wasm_pt_sectors.py').read_bytes()).hexdigest(),
        calculations=len(records), independent_points=len(checks), checks=checks,
        excluded=excluded, comparisons=comparisons, negative_controls=negatives, issues=issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = gzip.decompress(args.input.read_bytes()) if args.input.suffix == '.gz' else args.input.read_bytes()
    result = audit(json.loads(raw), raw)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('checks', 'comparisons')}, indent=2))
    raise SystemExit(bool(result['issues']))
