#!/usr/bin/env python3
"""C4 probe of the spectral profile and of the coarse-basis hook on a few cases.

parity  the Rust profile against W^(1/2) T W^(1/2) in numpy.
scan    simple profile shapes on the 16^2 training cases (screen pairs at radii 0.15 and
        0.45, k indices 0, 1 and 20), recording the surrogates S1 (kappa_eff of the top
        band, dense) and S2 (iterations of a cold solve of the real solver) and their
        rank correlation.
fit     profile knots per polarization minimizing the mean log of S1, S2 or S3 (FFT work
        of full band runs of the c3 variant, warm starts included) over the training
        cases; writes fit.json for the c4-fit variant.
oracle  cold solves with coarse bases built from exact and perturbed eigenvectors and from
        the slowest preconditioned modes, the best a learned subspace correction could
        do, beside the plane-wave correction.
variant add the study's fit.json as a variant (c3 plus per-polarization profiles) to
        variants.json.
compare work of full band runs (run_work.py records; FFT vectors, GEMM terms or
        iterations) per case against a reference variant, with geometric means per
        polarization and resolution.

Knots above s = 64, beyond the 16^2 spectrum, are tied to zero: a constant offset only
rescales T, which LOBPCG ignores, and at large s the base preconditioner already
matches the operator's symbol.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The dense matrices are small; threaded BLAS only oversubscribes the host.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.optimize as so  # noqa: E402
import scipy.stats as ss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases  # noqa: E402
import runlib  # noqa: E402
from lab import candidates as cd  # noqa: E402
from lab.conditioning import positive_spread, psd_sqrt  # noqa: E402
from lab.rustops import RustOperators  # noqa: E402
from ml import profile  # noqa: E402

LAB = runlib.REPO / 'target' / 'release' / 'examples' / 'precond_lab'
CLI = runlib.REPO / 'target' / 'release' / 'blaze2d'
K_INDICES = (0, 1, 20)
FREE = 6  # knots s = 1/16 ... 64
BLOCK = cases.BANDS + 2
BAND = cases.BANDS - 1


def knots(free) -> list[float]:
    return [float(x) for x in free] + [0.0] * (len(profile.KNOTS) - FREE)


def shapes() -> dict[str, list[float]]:
    s = profile.KNOTS[:FREE]
    out = {'zero': knots(np.zeros(FREE))}
    for beta in (-0.3, -0.15, 0.15, 0.3):
        out[f'tilt{beta:+.2f}'] = knots(beta * np.log(s / s[-1]))
    for gamma in (-1.0, -0.5, 0.5, 1.0):
        out[f'bump{gamma:+.1f}'] = knots([gamma, gamma, gamma, gamma / 2, 0.0, 0.0])
    return out


def training_cases(polarization: str, match: str | None = None) -> list[cases.Case]:
    selected = [cases.Case(lattice, polarization, 16, radius, *pair) for lattice in cases.LATTICES
                for radius in (0.15, 0.45) for pair in cases.SCREEN_PAIRS]
    return [c for c in selected if match is None or fnmatch.fnmatchcase(c.label, match)]


def open_case(case: cases.Case, binary: Path) -> RustOperators:
    with tempfile.NamedTemporaryFile('w', suffix='.toml', delete=False) as f:
        f.write(cases.render(case, {}))
    ops = RustOperators(binary, f.name)
    Path(f.name).unlink()
    return ops


def b2min(ops: RustOperators) -> float:
    return profile.shortest_reciprocal_sq(*ops.info['reciprocal'])


def unitary_dft(ny: int, nx: int) -> np.ndarray:
    n = nx * ny
    return np.fft.fft2(np.eye(n).reshape(n, ny, nx)).reshape(n, n).T / np.sqrt(n)


@dataclasses.dataclass
class Point:
    """Dense data of one case and k for S1: eig(T' K_n) = eig(P^H D Kh D P) with D = exp(tau/2)."""
    label: str
    k_index: int
    s: np.ndarray
    p: np.ndarray
    kh: np.ndarray

    def kappa(self, tau) -> float:
        d = np.exp(0.5 * profile.tau(self.s, tau))
        h = (self.p.conj().T * d) @ (self.kh * np.outer(d, d)) @ (d[:, None] * self.p)
        lo, hi = positive_spread(h, BLOCK)
        return hi / lo


def dense_point(ops: RustOperators, case: cases.Case, k: int) -> Point:
    ops.set_spec(with_profile(ops.info['spec'], None))
    ops.set_k(index=k)
    a, b, t = ops.dense('A'), ops.dense('B'), ops.dense('T')
    a, b = (a + a.conj().T) / 2, (b + b.conj().T) / 2
    w, v = sla.eigh(a, b)
    vm = v[:, :BLOCK]
    projector = np.eye(ops.n) - vm @ (vm.conj().T @ b)
    k_n = projector.conj().T @ (a - w[BAND] * b) @ projector
    u = unitary_dft(ops.ny, ops.nx)
    q2 = cd.PointData.from_ops(ops).q2.ravel()
    return Point(case.label, k, q2 / b2min(ops), u @ psd_sqrt(t), u @ k_n @ u.conj().T)


def fractional_iterations(trace: list[dict], tolerance: float) -> float:
    """Iterations until every requested residual is below tolerance, log-interpolated."""
    worst = [max(entry['residuals']) for entry in trace if entry['residuals']]
    for i, r in enumerate(worst):
        if r <= tolerance:
            if i == 0:
                return 0.0
            prev = worst[i - 1]
            return i - 1 + np.log(prev / tolerance) / np.log(prev / r)
    return float(len(worst))


def cold_solve(ops: RustOperators) -> dict:
    reply, _ = ops.solve(trace=True)
    tolerance = ops.info['eigensolver']['residual_tol']
    return dict(iterations=reply['iterations'], converged=reply['converged'],
                fractional=fractional_iterations(reply['trace'], tolerance), work=reply['work'])


def with_profile(spec: dict, tau) -> dict:
    """The spec with knot values tau, or with no profile for None."""
    return dict(spec, profile='none' if tau is None else {'knots': [float(x) for x in tau]})


def path_work(case: cases.Case, tau, workdir: Path) -> dict:
    """FFT work and certificate of a full band run of the c3 variant with profile tau."""
    variant = cases.load_variants()['c3']
    if tau is not None:
        variant = cases.merge(variant, {'eigensolver': {'preconditioner': {'profile': [float(x) for x in tau]}}})
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        toml, out = Path(tmp) / 'case.toml', Path(tmp) / 'out.json'
        toml.write_text(cases.render(case, variant))
        subprocess.run([str(CLI), 'run', str(toml), '--threads', '1', '--quiet', '-o', str(out)],
                       check=True, capture_output=True)
        reduced = runlib.reduce_output(json.loads(out.read_text()), False)
    return dict(fft=reduced['optional']['work']['fft_vectors'], iterations=reduced['iterations_total'],
                certificate=reduced['certification']['max_residual'], converged=reduced['converged'])


class Training:
    """Open operator servers and dense points of one polarization's training cases."""

    def __init__(self, polarization: str, binary: Path, dense: bool, workers: int = 8, workdir: Path | None = None,
                 match: str | None = None):
        self.cases = training_cases(polarization, match)
        self.workers, self.workdir = workers, workdir
        self.ops = [open_case(case, binary) for case in self.cases]
        self.points = [dense_point(ops, case, k) for ops, case in zip(self.ops, self.cases)
                       for k in K_INDICES] if dense else []

    def s1(self, tau) -> list[float]:
        return [p.kappa(tau) for p in self.points]

    def s2(self, tau) -> list[dict]:
        out = []
        for ops in self.ops:
            ops.set_spec(with_profile(ops.info['spec'], tau))
            for k in K_INDICES:
                ops.set_k(index=k)
                out.append(cold_solve(ops))
        return out

    def s3(self, tau) -> list[dict]:
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda case: path_work(case, tau, self.workdir), self.cases))

    def close(self) -> None:
        for ops in self.ops:
            ops.close()


def study(args: argparse.Namespace, binary: dict) -> tuple[Path, runlib.JsonlWriter]:
    root = args.results / args.study
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.json'
    meta = json.loads(manifest.read_text()) if manifest.exists() else runlib.manifest(
        args.study, 'ml', 'ml', 'train32', {'probe': dict(description=__doc__.splitlines()[0])},
        {'probe': binary}, {})
    meta.setdefault('invocations', []).append(dict(started=runlib.utc_now(), argv=sys.argv[1:]))
    meta['binaries']['probe'] = binary
    runlib.write_json(manifest, meta)
    return root, runlib.JsonlWriter(root / 'records.jsonl')


def record(writer: runlib.JsonlWriter, binary: dict, tag: str, **fields) -> None:
    writer.write(dict(tag=tag, variant='probe', binary=binary,
                      validation=dict(completed=True, ok=True, flags=[]), **fields))


def run_parity(args, binary, writer) -> None:
    rng = np.random.default_rng(7)
    tau = rng.uniform(-1.5, 1.5, len(profile.KNOTS))
    worst = 0.0
    for polarization in cases.POLARIZATIONS:
        case = cases.Case('triangular', polarization, 16, 0.30, 1.0, 13.0)
        with open_case(case, args.binary) as ops:
            spec = ops.info['spec']
            for k in (0, 1):
                ops.set_k(index=k)
                ops.set_spec(with_profile(spec, None))
                point = cd.PointData.from_ops(ops)
                expected = profile.dense_profiled(ops.dense('T'), point, b2min(ops), tau)
                ops.set_spec(with_profile(spec, tau))
                actual = ops.dense('T')
                error = float(np.abs(actual - expected).max() / np.abs(expected).max())
                worst = max(worst, error)
                record(writer, binary, 'parity', label=case.label, k_index=k, relative_error=error)
                print(f'parity {case.label} k{k}: {error:.2e}')
    if worst > 1e-12:
        raise SystemExit(f'profile parity failed: {worst:.2e}')


def run_scan(args, binary, writer) -> None:
    for polarization in cases.POLARIZATIONS:
        training = Training(polarization, args.binary, dense=True)
        try:
            rows = {}
            for name, tau in shapes().items():
                start = time.perf_counter()
                s1, s2 = training.s1(tau), training.s2(tau)
                rows[name] = (np.array(s1), np.array([r['fractional'] for r in s2]),
                              np.array([r['work']['fft_vectors'] for r in s2]))
                record(writer, binary, 'scan', polarization=polarization, shape=name, knots=tau,
                       points=[dict(label=p.label, k_index=p.k_index, kappa_eff=x, **r)
                               for p, x, r in zip(training.points, s1, s2)],
                       seconds=time.perf_counter() - start)
            base = rows['zero']
            print(f'\n{polarization}: shape, kappa_eff ratio, cold iterations ratio, FFT ratio (geometric means vs zero)')
            changes = []
            for name, (s1, s2, fft) in rows.items():
                g = lambda x, y: float(np.exp(np.mean(np.log(x / y))))
                print(f'  {name:12s} {g(s1, base[0]):6.3f} {g(s2, base[1]):6.3f} {g(fft, base[2]):6.3f}')
                if name != 'zero':
                    changes.append((np.log(s1 / base[0]), np.log(s2 / base[1])))
            pooled = ss.spearmanr(np.concatenate([c[0] for c in changes]), np.concatenate([c[1] for c in changes]))
            means = ss.spearmanr([c[0].mean() for c in changes], [c[1].mean() for c in changes])
            print(f'  spearman(S1, S2): pointwise {pooled.statistic:.2f}, shape means {means.statistic:.2f}')
            record(writer, binary, 'scan-summary', polarization=polarization,
                   spearman_pointwise=float(pooled.statistic), spearman_shapes=float(means.statistic))
        finally:
            training.close()


def run_fit(args, binary, writer, root: Path) -> None:
    fits = json.loads((root / 'fit.json').read_text()) if (root / 'fit.json').exists() else {}
    for polarization in args.polarization:
        training = Training(polarization, args.binary, dense=args.objective == 's1',
                            workers=args.workers, workdir=root, match=args.match)
        history = []

        def objective(free):
            tau = knots(free)
            if args.objective == 's1':
                value = float(np.mean(np.log(training.s1(tau))))
            elif args.objective == 's3':
                value = float(np.mean(np.log([r['fft'] for r in training.s3(tau)])))
            else:
                value = float(np.mean(np.log([r['fractional'] for r in training.s2(tau)])))
            history.append((value, [float(x) for x in free]))
            return value

        try:
            start = time.perf_counter()
            zero = objective(np.zeros(FREE))
            result = so.minimize(objective, np.zeros(FREE), method='Powell', bounds=[(-3.0, 3.0)] * FREE,
                                 options=dict(xtol=0.02, ftol=1e-4, maxfev=args.maxfev))
            tau = knots(result.x)
            check = training.s2(tau)
            base = training.s2(None)
            ratio = float(np.exp(np.mean(np.log([b['fractional'] / c['fractional'] for b, c in zip(base, check)]))))
            fft = float(sum(b['work']['fft_vectors'] for b in base) / sum(c['work']['fft_vectors'] for c in check))
            paths, base_paths = training.s3(tau), training.s3(None)
            path_fft = float(np.exp(np.mean(np.log([c['fft'] / b['fft'] for b, c in zip(base_paths, paths)]))))
            fits[polarization] = dict(objective=args.objective, match=args.match, knots=tau, value=float(result.fun), zero=zero,
                                      evaluations=len(history), train_iteration_ratio=ratio, train_fft_ratio=fft,
                                      train_path_fft=path_fft)
            record(writer, binary, 'fit', polarization=polarization, objective=args.objective, knots=tau,
                   value=float(result.fun), zero=zero, evaluations=len(history), history=history,
                   train_iteration_ratio=ratio, train_fft_ratio=fft, train_path_fft=path_fft,
                   seconds=time.perf_counter() - start)
            print(f'{polarization} fit ({args.objective}, {len(history)} evaluations, {time.perf_counter() - start:.0f}s): '
                  f'objective {zero:.4f} -> {result.fun:.4f}; cold iterations faster x{ratio:.3f}, '
                  f'cold FFTs x{1 / fft:.3f}, path FFTs x{path_fft:.3f} on training')
            print('  knots', ' '.join(f'{x:+.2f}' for x in tau))
        finally:
            training.close()
    runlib.write_json(root / 'fit.json', fits)


def run_oracle(args, binary, writer) -> None:
    rng = np.random.default_rng(11)
    for lattice in cases.LATTICES:
        for polarization in cases.POLARIZATIONS:
            case = cases.Case(lattice, polarization, args.resolution, 0.30, 1.0, 11.0)
            with open_case(case, args.binary) as ops:
                spec = ops.info['spec']
                for k in (1, 20):
                    ops.set_spec(dict(spec, profile='none', coarse_modes=0))
                    ops.set_k(index=k)
                    a, b, t = ops.dense('A'), ops.dense('B'), ops.dense('T')
                    a, b = (a + a.conj().T) / 2, (b + b.conj().T) / 2
                    w, v = sla.eigh(a, b)
                    bases = {'none': None, 'next-block': v[:, BLOCK:2 * BLOCK], 'target': v[:, :BLOCK]}
                    # Slowest modes x = T^(1/2) y of the top band's preconditioned pencil T^(1/2) K_n T^(1/2).
                    vm = v[:, :BLOCK]
                    projector = np.eye(ops.n) - vm @ (vm.conj().T @ b)
                    root = psd_sqrt(t)
                    h = root @ (projector.conj().T @ (a - w[BAND] * b) @ projector) @ root
                    values, vectors = np.linalg.eigh((h + h.conj().T) / 2)
                    order = [i for i in np.argsort(values) if values[i] > 1e-10 * np.abs(values).max()]
                    for multiple in (1, 2, 4):
                        bases[f'slow-{multiple}m'] = root @ vectors[:, order[:multiple * BLOCK]]
                    for noise in (0.03, 0.1, 0.3):
                        g = rng.standard_normal(v[:, :BLOCK].shape) + 1j * rng.standard_normal(v[:, :BLOCK].shape)
                        g *= noise * np.linalg.norm(v[:, :BLOCK], axis=0) / np.linalg.norm(g, axis=0)
                        bases[f'target+{noise:g}'] = v[:, :BLOCK] + g
                    results = {}
                    for name, basis in bases.items():
                        ops.set_spec(dict(spec, profile='none', coarse_modes=0))
                        if basis is not None:
                            ops.request({'op': 'set_basis', 'count': basis.shape[1]}, basis.T)
                        results[name] = cold_solve(ops)
                    ops.set_spec(dict(spec, profile='none', coarse_modes=2 * BLOCK))
                    results['plane-waves-2m'] = cold_solve(ops)
                    record(writer, binary, 'oracle', label=case.label, k_index=k, results=results)
                    print(f'{case.label} k{k:02d} ' + ' '.join(f"{n}:{r['iterations']}" for n, r in results.items()))


def run_variant(args) -> None:
    fits = json.loads((args.results / args.study / 'fit.json').read_text())
    variants = cases.load_variants()
    variant = json.loads(json.dumps(variants['c3']))
    variant['description'] = (f'New default preconditioner with spectral profiles fitted on 16^2 training cases '
                              f'({args.study}, objective ' + ', '.join(f"{p} {f['objective']}" for p, f in fits.items()) + ')')
    variant['requires'] = 'eigensolver.preconditioner.profile (binaries after 55dd68f5)'
    variant['by_polarization'] = {p: {'eigensolver': {'preconditioner': {'profile': [round(x, 4) + 0.0 for x in f['knots']]}}}
                                  for p, f in fits.items()}
    variants[args.name] = variant
    cases.VARIANTS.write_text(json.dumps(variants, indent=2) + '\n')
    print(f'{args.name}: ' + '; '.join(f"{p} " + ' '.join(f'{x:+.2f}' for x in f['knots']) for p, f in fits.items()))


def run_compare(args) -> None:
    def load(study: str) -> list[dict]:
        rows = [json.loads(line) for line in (args.results / study / 'records.jsonl').read_text().splitlines()]
        return [r for r in rows if r.get('tag', 'work') == 'work' and r['validation']['completed']]

    rows = load(args.study)
    if args.reference_study:
        labels = {r['label'] for r in rows}
        rows = [r for r in rows if r['variant'] != args.reference] + [
            r for r in load(args.reference_study) if r['variant'] == args.reference and r['label'] in labels]
    fft = {(r['label'], r['variant']): r['iterations_total'] if args.metric == 'iterations'
           else r['optional']['work'][args.metric] for r in rows}
    worst: dict[str, float] = {}
    for r in rows:
        worst[r['variant']] = max(worst.get(r['variant'], 0.0), r['certification']['max_residual'])
    variants = list(dict.fromkeys(r['variant'] for r in rows))
    others = [v for v in variants if v != args.reference]
    labels = sorted({label for label, _ in fft})
    print(f"{'case':24s}" + ''.join(f'{v:>14s}' for v in others) + f'   ({args.metric} / {args.reference})')
    ratios = {v: {} for v in others}
    for label in labels:
        if (label, args.reference) not in fft:
            continue
        cells = []
        for v in others:
            ratio = fft.get((label, v), float('nan')) / fft[label, args.reference]
            ratios[v][label] = ratio
            cells.append(f'{ratio:14.3f}')
        print(f'{label:24s}' + ''.join(cells))
    groups = sorted({(label.split('-')[1], label.split('-')[2]) for label in labels})
    for polarization, resolution in groups + [('all', '')]:
        cells = []
        for v in others:
            values = [r for label, r in ratios[v].items() if polarization == 'all'
                      or (label.split('-')[1], label.split('-')[2]) == (polarization, resolution)]
            cells.append(f'{float(np.exp(np.mean(np.log(values)))):14.3f}')
        print(f"{'geomean ' + polarization + ' ' + resolution:24s}" + ''.join(cells))
    print('max certificate ' + ', '.join(f'{v} {worst[v]:.1e}' for v in variants))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('parity', 'scan', 'fit', 'oracle', 'variant', 'compare'))
    parser.add_argument('--study', default='c4-probe')
    parser.add_argument('--binary', type=Path, default=LAB, help='precond_lab executable')
    parser.add_argument('--objective', choices=('s1', 's2', 's3'), default='s3')
    parser.add_argument('--workers', type=int, default=8, help='parallel band runs for s3')
    parser.add_argument('--polarization', action='append', choices=cases.POLARIZATIONS)
    parser.add_argument('--maxfev', type=int, default=400)
    parser.add_argument('--match', help='fit: glob restricting the training cases, e.g. one label')
    parser.add_argument('--resolution', type=int, default=32, help='oracle grid')
    parser.add_argument('--reference', default='c3', help='compare: reference variant')
    parser.add_argument('--reference-study', help='compare: take the reference variant from this study')
    parser.add_argument('--name', help='variant: name of the new variant')
    parser.add_argument('--metric', choices=('fft_vectors', 'gemm_terms', 'iterations'), default='fft_vectors')
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    args.polarization = args.polarization or list(cases.POLARIZATIONS)
    if args.command == 'compare':
        run_compare(args)
        return 0
    if args.command == 'variant':
        run_variant(args)
        return 0
    binary = runlib.binary_info(args.binary)
    root, writer = study(args, binary)
    if args.command == 'parity':
        run_parity(args, binary, writer)
    elif args.command == 'scan':
        run_scan(args, binary, writer)
    elif args.command == 'fit':
        run_fit(args, binary, writer, root)
    else:
        run_oracle(args, binary, writer)
    return 0


if __name__ == '__main__':
    sys.exit(main())
