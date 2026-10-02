#!/usr/bin/env python3
"""Generalization of the default TE profile beyond the study matrix.

Every scenario runs with the current defaults (residual stopping at 1e-7 on f64, the
automatic symmetry policy) twice, with profile = "none" and with "auto", and records
FFT work, convergence, fresh certificates and the largest relative difference of the
result arrays between the two. Scenarios vary what the 12-band study matrix held
fixed: band count, lattice, objects, resolution, f32 storage and eigenvalue-change
stopping (also checked against a tight f64 reference), and the operators task, which
takes no preconditioner settings and is compared against the C3 binary instead.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases  # noqa: E402
import runlib  # noqa: E402

CLI = runlib.REPO / 'target' / 'release' / 'blaze2d'
# The last binary without spectral profiles, for tasks that reject preconditioner settings.
UNPROFILED = runlib.REPO / 'benchmarks' / 'preconditioner' / 'bin' / '55dd68f5' / 'blaze2d'
LATTICES = {
    'square': ({'type': 'square', 'a': 1.0}, {'preset': 'square'}),
    'triangular': ({'type': 'triangular', 'a': 1.0}, {'preset': 'triangular'}),
    'rect-1.5': ({'type': 'rectangular', 'a': 1.0, 'b': 1.5}, {'preset': 'rectangular'}),
    'rect-0.5': ({'type': 'rectangular', 'a': 2.0, 'b': 1.0}, {'preset': 'rectangular'}),
    'oblique-70': ({'type': 'oblique', 'a': 1.0, 'b': 1.2, 'angle_deg': 70.0}, None),
    'oblique-40': ({'type': 'oblique', 'a': 1.0, 'b': 1.0, 'angle_deg': 40.0}, None),
    'oblique-15': ({'type': 'oblique', 'a': 1.0, 'b': 2.5, 'angle_deg': 15.0}, None),
}
OBLIQUE_PATH = {'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.0, 0.0]]}
HOLES, RODS = (1.0, 13.0), (4.0, 2.0)


def rod(eps: float, radius: float, center=(0.0, 0.0), name: str = 'rod') -> dict:
    return {'name': name, 'kind': 'circle', 'center': list(center), 'radius': radius, 'epsilon': eps}


def bands_config(lattice: str = 'square', bands: int = 12, objects: list | None = None, bg: float = 13.0,
                 resolution: Any = 32, polarization: str = 'TE', eigensolver: dict | None = None) -> dict:
    geometry, path = LATTICES[lattice]
    return {
        'schema': 'blaze2d/1', 'task': 'bands', 'polarization': polarization,
        'geometry': {'background_epsilon': bg, 'lattice': geometry,
                     'objects': objects if objects is not None else [rod(1.0, 0.3)]},
        'grid': {'resolution': resolution},
        'bands': {'count': bands, 'path': dict(path or OBLIQUE_PATH, intervals_per_segment=12)},
        'dielectric': {'smoothing': 'analytic'},
        'results': {'eigenvectors': False},
        'eigensolver': eigensolver or {'max_iterations': 600},
    }


def pair(eps: tuple[float, float], radius: float = 0.3) -> dict:
    return dict(objects=[rod(eps[0], radius)], bg=eps[1])


def scenarios() -> list[tuple[str, str, dict, bool]]:
    """(group, name, config, compare against a tight f64 reference)."""
    out = []
    for lattice in ('square', 'triangular'):
        for label, eps in (('holes', HOLES), ('rods', RODS)):
            for bands in (4, 8, 12, 24, 40):
                out.append(('bands', f'{lattice} {label} {bands} bands 32', bands_config(lattice, bands, **pair(eps)), False))
            out.append(('bands', f'{lattice} {label} 24 bands 64', bands_config(lattice, 24, resolution=64, **pair(eps)), False))
    for lattice in ('rect-1.5', 'rect-0.5', 'oblique-70', 'oblique-40', 'oblique-15'):
        resolution = {'rect-1.5': [32, 48], 'rect-0.5': [48, 24]}.get(lattice, 32)
        for label, eps in (('holes', HOLES), ('rods', RODS)):
            out.append(('lattice', f'{lattice} {label}', bands_config(lattice, resolution=resolution, **pair(eps, 0.2)), False))
    out += [
        ('geometry', 'square dimer rods', bands_config(objects=[rod(10.0, 0.12, (-0.2, 0.0), 'a'), rod(10.0, 0.12, (0.2, 0.0), 'b')], bg=1.0), False),
        ('geometry', 'square unequal pair (no inversion)', bands_config(objects=[rod(10.0, 0.15, (0.15, 0.0), 'a'), rod(5.0, 0.1, (-0.2, 0.1), 'b')], bg=1.0), False),
        ('geometry', 'triangular near-touching holes', bands_config('triangular', **pair(HOLES, 0.48)), False),
        ('geometry', 'square small rods', bands_config(objects=[rod(10.0, 0.08)], bg=1.0), False),
        ('geometry', 'square rod and hole', bands_config(objects=[rod(13.0, 0.3, (0.0, 0.0), 'a'), rod(1.0, 0.1, (0.5, 0.5), 'b')], bg=4.0), False),
        ('geometry', 'homogeneous', bands_config(objects=[], bg=4.0), False),
        ('geometry', 'square holes TM control', bands_config(polarization='TM', **pair(HOLES)), False),
    ]
    for n in (48, 96):
        out.append(('resolution', f'square holes {n}', bands_config(resolution=n, **pair(HOLES)), False))
        out.append(('resolution', f'triangular rods {n}', bands_config('triangular', resolution=n, **pair(RODS)), False))
    out.append(('resolution', 'square holes 256', bands_config(resolution=256, **pair(HOLES)), False))
    for lattice in ('square', 'triangular'):
        for label, eps in (('holes', HOLES), ('rods', RODS)):
            for n in (32, 64):
                f32 = {'precision': 'f32', 'stopping': 'eigenvalue_change', 'tolerance': 1e-6, 'max_iterations': 600}
                f64 = dict(f32, precision='f64')
                out.append(('f32', f'{lattice} {label} {n} f32 eigenvalue change', bands_config(lattice, resolution=n, eigensolver=f32, **pair(eps)), True))
                out.append(('eigenvalue-change', f'{lattice} {label} {n} f64 eigenvalue change', bands_config(lattice, resolution=n, eigensolver=f64, **pair(eps)), True))
    operators = tomllib.loads((runlib.REPO / 'examples/calculations/operator-point.toml').read_text())
    out.append(('operators', 'operator point (TE holes)', operators, False))
    registry = tomllib.loads((runlib.REPO / 'crates/interface/tests/fixtures/operators.toml').read_text())
    registry.pop('sweeps', None)
    out.append(('operators', 'registry derivatives (TE holes)', registry, False))
    return out


def with_profile(config: dict, profile: str) -> dict:
    return cases.merge(config, {'eigensolver': {'preconditioner': {'profile': profile}}})


def reference(config: dict) -> dict:
    """Tight f64 residual-stopping reference without a profile."""
    solver = {k: v for k, v in config.get('eigensolver', {}).items() if k not in ('precision', 'stopping', 'tolerance')}
    return cases.merge(dict(config, eigensolver=solver), {'eigensolver': {
        'stopping': 'residual', 'residual_tolerance': 1e-10, 'max_iterations': 1000, 'preconditioner': {'profile': 'none'}}})


def run(config: dict, workdir: Path, timeout: float, binary: Path = CLI) -> dict:
    try:
        return solve(config, workdir, timeout, binary)
    except Exception as error:  # A failed scenario is recorded, not fatal to the study.
        return dict(ok=False, error=repr(error), seconds=None)


def solve(config: dict, workdir: Path, timeout: float, binary: Path) -> dict:
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        toml, out = Path(tmp) / 'case.toml', Path(tmp) / 'out.json'
        toml.write_text(cases.toml_text(config))
        start = time.perf_counter()
        process = subprocess.run([str(binary), 'run', str(toml), '--threads', '1', '--quiet', '-o', str(out)],
                                 capture_output=True, text=True, timeout=timeout)
        seconds = time.perf_counter() - start
        document = json.loads(out.read_text()) if out.exists() else None
    if process.returncode or not document or not document.get('results'):
        return dict(ok=False, error=(process.stderr or '')[-800:], seconds=seconds)
    result = document['results'][0]
    metadata = result.get('metadata') or {}
    arrays = {k: v['data'] if isinstance(v, dict) and 'data' in v else v
              for k, v in (result.get('arrays') or {}).items() if isinstance(v, (list, dict))}
    points = (metadata.get('preconditioner') or {}).get('points') or []
    return dict(ok=True, seconds=seconds, converged=bool(metadata.get('converged', True)),
                fft=(metadata.get('work') or {}).get('fft_vectors'),
                iterations=sum(np.atleast_1d(metadata.get('iterations') or 0)) or None,
                certificate=(metadata.get('certification') or {}).get('max_residual'),
                profiled=any(p.get('profile') for p in points if isinstance(p, dict)),
                arithmetic=((metadata.get('execution') or {}).get('resolved') or {}).get('arithmetic'),
                arrays=arrays)


def numeric(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def max_relative(a: dict, b: dict, bands: int | None = None) -> float | None:
    """Largest |a - b| / max|b| over the result arrays both runs hold with equal shapes.

    Band frequencies and eigenvalues are tracked across k, so crossing bands may be
    ordered differently; they are compared sorted per k point.
    """
    worst = None
    for key in sorted(set(a) & set(b)):
        x, y = numeric(a[key]), numeric(b[key])
        if x is None or y is None or x.shape != y.shape or any(w in key for w in ('residual', 'seconds', 'iteration')):
            continue
        if bands and key in ('frequencies', 'eigenvalues') and x.size % bands == 0:
            x, y = np.sort(x.reshape(-1, bands), axis=1), np.sort(y.reshape(-1, bands), axis=1)
        scale = np.nanmax(np.abs(y)) or 1.0
        value = float(np.nanmax(np.abs(x - y)) / scale)
        worst = value if worst is None else max(worst, value)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--study', default='c4-general')
    parser.add_argument('--group', action='append', help='run only these groups')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--results', type=Path, default=runlib.RESULTS)
    args = parser.parse_args()
    root = args.results / args.study
    root.mkdir(parents=True, exist_ok=True)
    binary = runlib.binary_info(CLI)
    manifest = root / 'manifest.json'
    meta = json.loads(manifest.read_text()) if manifest.exists() else runlib.manifest(
        args.study, 'ml', 'ml', 'general', {'general': dict(description=__doc__.splitlines()[0])}, {'general': binary}, {})
    meta.setdefault('invocations', []).append(dict(started=runlib.utc_now(), argv=sys.argv[1:]))
    meta['binaries']['general'] = binary
    runlib.write_json(manifest, meta)
    writer = runlib.JsonlWriter(root / 'records.jsonl')
    selected = [s for s in scenarios() if not args.group or s[0] in args.group]
    jobs = []
    for i, (group, _, config, tight) in enumerate(selected):
        if config['task'] == 'operators':
            jobs += [(i, 'none', config, UNPROFILED), (i, 'auto', config, CLI)]
            continue
        jobs += [(i, 'none', with_profile(config, 'none'), CLI), (i, 'auto', with_profile(config, 'auto'), CLI)]
        if tight:
            jobs.append((i, 'reference', reference(config), CLI))
    jobs.sort(key=lambda job: -np.prod(np.atleast_1d(job[2]['grid']['resolution'])) * (job[2].get('bands') or {}).get('count', 8))
    runs: dict[int, dict[str, dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, cfg, root, args.timeout, binary): (i, kind) for i, kind, cfg, binary in jobs}
        for future in concurrent.futures.as_completed(futures):
            i, kind = futures[future]
            runs.setdefault(i, {})[kind] = future.result()
            group, name, config, tight = selected[i]
            bands = (config.get('bands') or {}).get('count')
            if len(runs[i]) < (3 if tight else 2):
                continue
            none, auto = runs[i]['none'], runs[i]['auto']
            row = dict(group=group, name=name, ok=none['ok'] and auto['ok'])
            if row['ok']:
                row.update(fft_ratio=auto['fft'] / none['fft'] if auto['fft'] and none['fft'] else None,
                           converged=none['converged'] and auto['converged'],
                           certificate={'none': none['certificate'], 'auto': auto['certificate']},
                           profiled=auto['profiled'], arithmetic=auto['arithmetic'],
                           difference=max_relative(auto['arrays'], none['arrays'], bands),
                           seconds={'none': none['seconds'], 'auto': auto['seconds']})
                if tight and runs[i]['reference']['ok']:
                    ref = runs[i]['reference']['arrays']
                    row['error'] = {'none': max_relative(none['arrays'], ref, bands), 'auto': max_relative(auto['arrays'], ref, bands)}
            else:
                row['errors'] = {k: r.get('error') for k, r in runs[i].items() if not r['ok']}
            writer.write(dict(tag='general', variant='general', binary=binary,
                              validation=dict(completed=row['ok'], ok=row['ok'], flags=[]), **row))
            ratio = f"{row['fft_ratio']:.3f}" if row.get('fft_ratio') else '-'
            extra = (f" error none {row['error']['none']:.1e} auto {row['error']['auto']:.1e}" if row.get('error') else '')
            print(f"{group:17s} {name:44s} FFT auto/none {ratio:>6s} profiled={row.get('profiled')} "
                  f"diff={row.get('difference') or 0:.1e}{extra}" if row['ok'] else f'{group} {name} FAILED {row["errors"]}',
                  flush=True)
    meta.update(status='complete', ended=runlib.utc_now())
    runlib.write_json(manifest, meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
