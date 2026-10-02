"""Condition numbers of the preconditioned Maxwell pencil for representative geometries.

For each case and k index this records kappa(T A) on the nonzero spectrum and,
for selected bands n of a block of size m, kappa_eff of T P^H (A - lambda_n B) P
with P = I - V_m V_m^H B, plus the gap ratio lambda_n / lambda_m. Dense spectra
use the symmetric form T^(1/2) K T^(1/2); larger grids use Lanczos for kappa(T A).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases  # noqa: E402
import runlib  # noqa: E402
from lab.rustops import RustOperators  # noqa: E402
from lab.spectra import lanczos_kappa_ta  # noqa: E402

PAIRS = ((1.0, 13.0), (10.0, 2.0), (7.0, 11.0))
K_INDICES = (0, 1, 20)


def psd_sqrt(t: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((t + t.conj().T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T


def positive_spread(h: np.ndarray, drop: int) -> tuple[float, float]:
    values = np.linalg.eigvalsh((h + h.conj().T) / 2)
    scale = np.max(np.abs(values))
    values = np.delete(values, np.argsort(np.abs(values))[:drop]) if drop else values
    values = values[values > 1e-10 * scale]
    return float(values[0]), float(values[-1])


def dense_metrics(ops: RustOperators, block: int, bands: list[int]) -> dict:
    a, b, t = ops.dense('A'), ops.dense('B'), ops.dense('T')
    w, v = sla.eigh((a + a.conj().T) / 2, (b + b.conj().T) / 2)
    s = psd_sqrt(t)
    lo, hi = positive_spread(s @ a @ s, 0)
    vm = v[:, :block]
    projector = np.eye(len(a)) - vm @ (vm.conj().T @ b)
    rows = []
    for n in bands:
        k = projector.conj().T @ (a - w[n] * b) @ projector
        klo, khi = positive_spread(s @ k @ s, block)
        rows.append({'band': n, 'lambda': float(w[n]), 'kappa_eff': khi / klo,
                     'gap': float(w[n] / w[block]) if w[block] > 0 else None})
    return {'ta_min': lo, 'ta_max': hi, 'kappa_ta': hi / lo, 'bands': rows,
            'lambda_block': float(w[block])}


def lanczos_metrics(ops: RustOperators, steps: int) -> dict:
    kernel = None
    if ops.point.get('is_gamma'):
        kernel = np.ones(ops.n, dtype=complex)
    lo, hi, kappa = lanczos_kappa_ta(lambda x: ops.apply('A', x[None])[0], lambda x: ops.apply('T', x[None])[0],
                                     ops.n, steps=steps, kernel=kernel)
    return {'ta_min': lo, 'ta_max': hi, 'kappa_ta': kappa, 'bands': [], 'method': f'lanczos-{steps}'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', default='e7-spectra')
    parser.add_argument('--binary', required=True, help='precond_lab executable')
    parser.add_argument('--resolutions', default='16,32,64')
    parser.add_argument('--lanczos-from', type=int, default=128, help='use Lanczos at and above this resolution')
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--spec', help='JSON preconditioner spec overriding the variant')
    parser.add_argument('--label', help='name of the preconditioner configuration (default: the variant; required with --spec)')
    parser.add_argument('--radius', type=float, default=0.30)
    parser.add_argument('--bands', default='0,5,11')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    if args.label is None:
        if args.spec:
            parser.error('--spec needs --label so its spectra do not replace the variant\'s')
        args.label = args.variant
    variant = cases.load_variants()[args.variant]
    root = args.results / args.study
    root.mkdir(parents=True, exist_ok=True)
    binary = runlib.binary_info(Path(args.binary))
    manifest = root / 'manifest.json'
    meta = json.loads(manifest.read_text()) if manifest.exists() else runlib.manifest(
        args.study, 'spectra', 'conditioning', 'conditioning12', {}, {},
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    meta['variants'][args.label] = dict(description=f'{args.variant} overlay', spec=json.loads(args.spec) if args.spec else None)
    meta['binaries'][args.label] = binary
    meta.setdefault('invocations', []).append(dict(started=runlib.utc_now(), argv=sys.argv[1:]))
    meta.update(status='running', ended=None, pid=os.getpid())
    runlib.write_json(manifest, meta)
    writer = runlib.JsonlWriter(root / 'records.jsonl')
    bands = [int(b) for b in args.bands.split(',')]
    block = cases.BANDS + 2
    for n in [int(r) for r in args.resolutions.split(',')]:
        for lattice in cases.LATTICES:
            for polarization in cases.POLARIZATIONS:
                for eps_rod, eps_bg in PAIRS:
                    case = cases.Case(lattice, polarization, n, args.radius, eps_rod, eps_bg)
                    config = cases.merge(case.config(), cases.overlay(variant))
                    with tempfile.NamedTemporaryFile('w', suffix='.toml', delete=False) as f:
                        f.write(cases.toml_text(config))
                        path = f.name
                    with RustOperators(args.binary, path) as ops:
                        if args.spec:
                            ops.set_spec(json.loads(args.spec))
                        for k in K_INDICES:
                            start = time.perf_counter()
                            point = ops.set_k(index=k)
                            metrics = lanczos_metrics(ops, args.steps) if n >= args.lanczos_from else dense_metrics(ops, block, bands)
                            record = {'label': case.label, 'case_id': case.id, 'params': case.params, 'k_index': k,
                                      'k': point['k'], 'configuration': args.label, 'variant': args.label,
                                      'tag': f'k{k}', 'binary': binary,
                                      'validation': {'completed': True, 'ok': True, 'flags': []},
                                      'preconditioner': point['preconditioner'], 'block': block,
                                      'contrast': case.contrast, **metrics}
                            record['seconds'] = record['wall_seconds'] = time.perf_counter() - start
                            writer.write(record)
                            print(f"{case.label} k{k:02d} kappa_TA={metrics['kappa_ta']:.2f} "
                                  + ' '.join(f"b{r['band']}:{r['kappa_eff']:.1f}" for r in metrics['bands'])
                                  + f" ({record['seconds']:.1f}s)", flush=True)
                    Path(path).unlink(missing_ok=True)
    meta.update(status='complete', ended=runlib.utc_now())
    runlib.write_json(manifest, meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
