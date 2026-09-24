"""Independent FFT audit of physical fields exported by the WASM public API.

The public preview exports epsilon, not the smoothed inverse tensor. Raw TM/TE
and smoothed TM can therefore be audited independently here. Smoothed TE is
explicitly excluded from that check; its exact fallback/control arrays are
checked by the JS harness. This is correctness evidence, not a speedup study.
"""
import argparse
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2


def decode(bits):
    return np.array([struct.unpack('>d', bytes.fromhex(x))[0] for x in bits])


def certificate(item, index, projected):
    record = item['result']
    arrays = record['arrays']
    nx, ny = item['preview']['resolution']
    width = item['config']['bands']['count']
    raw = np.array(arrays['eigenvectors']['data']).reshape(-1, width, ny, nx, 2)[index]
    u = raw[..., 0] + 1j * raw[..., 1]
    values = (math.tau * np.array(arrays['frequencies']['data']).reshape(-1, width)[index])**2
    eps = np.array(item['preview']['epsilon']).reshape(ny, nx)
    lattice = np.array(item['preview']['lattice_vectors'])
    reciprocal = math.tau * np.linalg.inv(lattice).T
    k = np.array(arrays['k_points']['data']).reshape(-1, 2)[index]
    cart = k @ reciprocal
    gx = np.arange(nx); gx = np.where(gx <= nx // 2, gx, gx - nx)
    gy = np.arange(ny); gy = np.where(gy <= ny // 2, gy, gy - ny)
    kx = cart[0] + reciprocal[0, 0]*gx[None, :] + reciprocal[1, 0]*gy[:, None]
    ky = cart[1] + reciprocal[0, 1]*gx[None, :] + reciprocal[1, 1]*gy[:, None]
    f = fft2(u, norm='ortho', workers=1)
    if item['config']['polarization'] == 'TM':
        a = (kx*kx + ky*ky) * f
        b = fft2(eps*u, norm='ortho', workers=1)
    else:
        if item['config']['dielectric']['smoothing'] != 'none':
            raise ValueError('smoothed TE inverse tensors are absent from the public preview')
        a = sum(v * fft2(ifft2(v*f, norm='ortho', workers=1)/eps, norm='ortho', workers=1)
                for v in (kx, ky))
        b = f.copy()
    r = record['metadata']['discretization']['points'][index]
    support = r['support']
    modes = np.array(support['modes'], dtype=int)
    ix, iy = modes[:, 0] % nx, modes[:, 1] % ny
    mask = np.ones((ny, nx), dtype=bool); mask[iy, ix] = False
    leakage = np.max(np.linalg.norm(f[:, mask], axis=1)/np.linalg.norm(u.reshape(width, -1), axis=1))
    if projected:
        a, b, c = a[:, iy, ix], b[:, iy, ix], f[:, iy, ix]
    else:
        a, b, c = a.reshape(width, -1), b.reshape(width, -1), f.reshape(width, -1)
    numerator = np.linalg.norm(a-values[:, None]*b, axis=1)
    denominator = np.linalg.norm(a, axis=1)+values*np.linalg.norm(b, axis=1)
    residuals = np.divide(numerator, denominator, out=numerator.copy(), where=denominator > 0)
    absolute = numerator/np.linalg.norm(c, axis=1)
    orth = np.max(np.abs(c.conj() @ b.T-np.eye(width)))
    return dict(residuals=residuals.tolist(), absolute_residuals=absolute.tolist(),
                b_orthogonality_defect=float(orth), max_support_leakage=float(leakage),
                zero_indices=np.flatnonzero(values < 1e-10).tolist())


def run(document, input_bytes):
    issues = list(document['issues'])
    checks, excluded = [], []
    for item in document['records']:
        result = item['result']
        if result is None:
            issues.append(item['name'] + ': missing result')
            continue
        if item['config']['polarization'] == 'TE' and item['config']['dielectric']['smoothing'] != 'none':
            excluded.append(dict(name=item['name'], reason='public preview omits inverse tensor; exact fallback/control arrays checked separately'))
            continue
        for index, report in enumerate(result['metadata']['discretization']['points']):
            label = f"{item['name']}/{index}"
            try:
                support = report['support']
                k = np.array(result['arrays']['k_points']['data']).reshape(-1, 2)[index]
                lattice = np.array(item['preview']['lattice_vectors'])
                if (support['bit_encoding'] != 'hexadecimal_u64'
                        or not np.array_equal(k, decode(support['k_bits']))
                        or not np.array_equal(lattice, np.array([decode(row) for row in support['lattice_bits']]))):
                    issues.append(label + ': identity bits disagree')
                # Independent bounding box from |g_j+k_j| <= |a_j|*cutoff/(2pi).
                cutoff = report['angular_cutoff']
                bounds = np.ceil(np.linalg.norm(lattice, axis=1)*cutoff/math.tau + np.abs(k)).astype(int) + 1
                reciprocal = math.tau * np.linalg.inv(lattice).T
                expected = {(x, y) for x in range(-bounds[0], bounds[0]+1)
                            for y in range(-bounds[1], bounds[1]+1)
                            if np.linalg.norm((np.array([x, y])+k) @ reciprocal) <= cutoff*(1+1e-12)}
                if expected != set(map(tuple, support['modes'])):
                    issues.append(label + ': incomplete metric support')
                projected = certificate(item, index, True)
                full = certificate(item, index, False)
                success = not item['expected_failure']
                zero = set(projected['zero_indices'])
                nonzero = [x for j, x in enumerate(projected['residuals']) if j not in zero]
                failed_gate = (max(nonzero, default=0) > 1e-5
                               or max([projected['absolute_residuals'][j] for j in zero], default=0) > 1e-7
                               or projected['b_orthogonality_defect'] > 1e-8
                               or projected['max_support_leakage'] > 1e-10)
                if success and failed_gate:
                    issues.append(label + ': independent projected gate failed')
                if not success and not failed_gate:
                    issues.append(label + ': expected iteration-limit control passed the residual gate')
                discrepancies = {}
                for name, actual in [('projected', projected), ('full_rectangle', full)]:
                    stored = report['selected'][name]
                    # The relative residual of a numerical zero mode is ill-conditioned;
                    # compare its absolute residual instead, including at folded Gamma.
                    diff = max([abs(actual['residuals'][j]-stored['residuals'][j])
                                for j in range(len(actual['residuals'])) if j not in zero], default=0)
                    abs_diff = max(abs(np.array(actual['absolute_residuals'])-np.array(stored['absolute_residuals'])), default=0)
                    orth_diff = abs(actual['b_orthogonality_defect']-stored['b_orthogonality_defect'])
                    if diff > 1e-8 or abs_diff > 1e-7 or orth_diff > 1e-8:
                        issues.append(label + ': stored ' + name + ' certificate disagrees')
                    discrepancies[name] = dict(relative_residual=float(diff), absolute_residual=float(abs_diff), orthogonality=float(orth_diff))
                checks.append(dict(name=item['name'], point=index, expected_success=success,
                                   independent_projected=projected, independent_full_rectangle=full,
                                   certificate_discrepancies=discrepancies))
            except (ValueError, KeyError, IndexError, TypeError) as e:
                issues.append(label + ': ' + str(e))
    return dict(schema='blaze2d/wasm-support-audit/1',
                input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                complete_record_count=len(document['records']),
                configuration_negative_count=len(document['configuration_negatives']),
                independent_successful_points=sum(x['expected_success'] for x in checks),
                independent_failed_points=sum(not x['expected_success'] for x in checks),
                excluded=excluded, checks=checks, issues=issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = gzip.decompress(args.input.read_bytes()) if args.input.suffix == '.gz' else args.input.read_bytes()
    audit = run(json.loads(raw), raw)
    with args.output.open('x') as f:
        json.dump(audit, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}, indent=2))
    raise SystemExit(bool(audit['issues']))
