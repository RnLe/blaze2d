"""E10 gates: dense spectral evidence for building a candidate preconditioner in Rust.

For each case and k index the top band's kappa_eff (see conditioning.py) is computed
for the solver's preconditioner and for numpy candidates built on the same Fourier
data: tensor-consistent transverse projection, Gamma-continuous shifts, the TE
Fourier diagonal, an additive two-level plane-wave correction with coarse spaces
of 1, 2 and 4 block sizes, Chebyshev polynomials of degree 1 and 2 in TK (K = A + shift B, spectral interval from the
dense spectrum, so an upper bound of what rigorous bounds can give), and the exact
inverse of K as the oracle; every family is evaluated with the solver's legacy shift
and with shift 0 (suffix -shift0). `cost_fft` counts FFT-equivalent work per preconditioned
column; speed-up estimates combine it with measured iteration cost shares.
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
from lab import candidates as cd  # noqa: E402
from lab.conditioning import positive_spread, psd_sqrt  # noqa: E402
from lab.rustops import RustOperators  # noqa: E402

K_INDICES = (0, 1, 20)


def kappa_eff(t: np.ndarray, k_n: np.ndarray, block: int) -> float:
    s = psd_sqrt(t)
    lo, hi = positive_spread(s @ k_n @ s, block)
    return hi / lo


def slow_energy(t: np.ndarray, k_n: np.ndarray, basis: np.ndarray, block: int, count: int = 16) -> float:
    """Mean fraction of the slowest preconditioned modes x = T^(1/2) y that lies in span(Z)."""
    s = psd_sqrt(t)
    h = s @ k_n @ s
    values, vectors = np.linalg.eigh((h + h.conj().T) / 2)
    scale = np.max(np.abs(values))
    order = np.argsort(np.abs(values))[block:]
    keep = [i for i in order if values[i] > 1e-10 * scale]
    keep = sorted(keep, key=lambda i: values[i])[:count]
    x = s @ vectors[:, keep]
    x /= np.linalg.norm(x, axis=0)
    q, _ = np.linalg.qr(basis)
    return float(np.mean(np.linalg.norm(q.conj().T @ x, axis=0) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', default='e10-gates')
    parser.add_argument('--binary', required=True, help='precond_lab executable')
    parser.add_argument('--subset', default='screen32')
    parser.add_argument('--match')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    variant = cases.load_variants()['baseline']
    selected = cases.select(args.subset, match=args.match, limit=args.limit)
    root = args.results / args.study
    root.mkdir(parents=True, exist_ok=True)
    binary = runlib.binary_info(Path(args.binary))
    manifest = root / 'manifest.json'
    meta = json.loads(manifest.read_text()) if manifest.exists() else runlib.manifest(
        args.study, 'spectra', 'multigrid', args.subset, {'gates': dict(description=__doc__.splitlines()[0])},
        {'gates': binary}, {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    meta.update(status='running', ended=None, pid=os.getpid())
    runlib.write_json(manifest, meta)
    writer = runlib.JsonlWriter(root / 'records.jsonl')
    block = cases.BANDS + 2
    band = cases.BANDS - 1
    for case in selected:
        config = cases.merge(case.config(), cases.overlay(variant))
        with tempfile.NamedTemporaryFile('w', suffix='.toml', delete=False) as f:
            f.write(cases.toml_text(config))
            path = f.name
        with RustOperators(args.binary, path) as ops:
            for k in K_INDICES:
                start = time.perf_counter()
                ops.set_k(index=k)
                p = cd.PointData.from_ops(ops)
                a, b, t_ref = ops.dense('A'), ops.dense('B'), ops.dense('T')
                a, b = (a + a.conj().T) / 2, (b + b.conj().T) / 2
                w, v = sla.eigh(a, b)
                vm = v[:, :block]
                projector = np.eye(p.n) - vm @ (vm.conj().T @ b)
                k_n = projector.conj().T @ (a - w[band] * b) @ projector
                shift = p.shift if p.shift > 0 else p.s_min
                te = p.polarization == 'TE'
                # FFT-equivalents per column: TP 6, Fourier diagonal 2, TE operator 6, TM operator 2.
                a_cost = 6.0 if te else 2.0
                results = {}

                def record(name, t, cost, **extra):
                    t = (t + t.conj().T) / 2
                    results[name] = dict(kappa_eff=kappa_eff(t, k_n, block), cost_fft=cost, **extra)

                record('reference', t_ref, 6.0 if te else 2.0)
                if te:
                    smoothers = {'': cd.tp(p, weight='tensor'), '-shift0': cd.tp(p, shift=0.0, weight='tensor')}
                    smoother_cost = 6.0
                    record('tp-tensor', cd.dense(smoothers[''], p.n), 6.0)
                    record('tp-tensor-shift0', cd.dense(smoothers['-shift0'], p.n), 6.0)
                    record('fd', cd.dense(cd.fd(p), p.n), 2.0)
                else:
                    smoothers = {'': cd.fd(p), '-shift0': cd.fd(p, shift=0.0)}
                    smoother_cost = 2.0
                    record('fd-shift0', cd.dense(smoothers['-shift0'], p.n), 2.0)
                    record('fd-nomass', cd.dense(cd.fd(p, tm_mass_fraction=0.0), p.n), 2.0)
                # Each family runs with the solver's legacy shift and with shift 0 (the
                # Gamma-continuous limit), so the oracle and the corrections are not tied
                # to one shift choice. K = A + shift B; at shift 0 its Gamma kernel is
                # excluded by the masked smoothers and the pseudo-inverse.
                for suffix, sigma in (('', shift), ('-shift0', 0.0)):
                    smoother = smoothers[suffix]
                    k_shift = a + sigma * b
                    t_s = cd.dense(smoother, p.n)
                    root_t = psd_sqrt(t_s)
                    tk = root_t @ k_shift @ root_t
                    values = np.linalg.eigvalsh((tk + tk.conj().T) / 2)
                    values = values[values > 1e-10 * values.max()]
                    interval = (float(values[0]), float(values[-1]))
                    for multiple in (1, 2, 4):
                        basis = cd.plane_waves(p, multiple * block)
                        m = basis.shape[1]
                        # Plane-wave restriction and prolongation fuse into the smoother's FFTs; the dense
                        # m x m solve costs 8 m^2 flops per column against 5 n log2 n per FFT.
                        cost = smoother_cost + 8.0 * m * m / (5.0 * p.n * np.log2(p.n))
                        record(f'two-level-{multiple}m{suffix}', cd.dense(cd.two_level(smoother, a, b, basis, sigma), p.n),
                               cost, coarse=m, slow_energy=slow_energy(t_s, k_n, basis, block))
                    for degree in (1, 2):
                        record(f'chebyshev-{degree}{suffix}', cd.dense(cd.chebyshev(smoother, k_shift, interval, degree), p.n),
                               (degree + 1) * smoother_cost + degree * a_cost)
                    k_herm = (k_shift + k_shift.conj().T) / 2
                    record(f'exact-inverse{suffix}', np.linalg.pinv(k_herm, rcond=1e-12, hermitian=True), float('nan'))
                reference = results['reference']['kappa_eff']
                for r in results.values():
                    r['ratio'] = reference / r['kappa_eff']
                row = dict(label=case.label, case_id=case.id, params=case.params, contrast=case.contrast, k_index=k,
                           variant='gates', tag=f'k{k}', binary=binary, shift=shift, interval=interval, a_cost_fft=a_cost,
                           candidates=results, validation=dict(completed=True, ok=True, flags=[]))
                row['seconds'] = row['wall_seconds'] = time.perf_counter() - start
                writer.write(row)
                best = max((n for n in results if n != 'exact-inverse'), key=lambda n: results[n]['ratio'])
                print(f"{case.label} k{k:02d} ref={reference:.2f} exact={results['exact-inverse']['kappa_eff']:.2f} "
                      f"best={best}:{results[best]['kappa_eff']:.2f} ({row['seconds']:.1f}s)", flush=True)
        Path(path).unlink(missing_ok=True)
    meta.update(status='complete', ended=runlib.utc_now())
    runlib.write_json(manifest, meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
