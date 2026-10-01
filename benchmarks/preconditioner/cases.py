#!/usr/bin/env python3
"""Preconditioner study case matrix, named subsets and single-job TOML rendering.

A case is one physical band diagram: a circular rod "rod" at the origin, analytic
dielectric smoothing, 12 bands on the lattice's preset k-path with 20 intervals per
segment (61 points) and no eigenvectors. Variants are overlays deep-merged into the
case configuration; the reserved keys below select binaries and CLI arguments.
"""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
VARIANTS = HERE / 'variants.json'
RESERVED = frozenset({'binary', 'cli_args', 'description', 'requires'})

LATTICES = ('square', 'triangular')
POLARIZATIONS = ('TE', 'TM')
RESOLUTIONS = (16, 32, 64, 128)
RADII = (0.15, 0.30, 0.45)
EPS_ROD = (1.0, 4.0, 7.0, 10.0)
EPS_BG = (13.0, 11.0, 5.0, 2.0)
SCREEN_PAIRS = ((1.0, 13.0), (10.0, 2.0), (7.0, 11.0), (4.0, 5.0))
BANDS = 12
INTERVALS = 20
SUBSETS = ('full768', 'dev96', 'screen32', 'timing48', 'pilot8')
ROLES = ('easy', 'median', 'hard')


@dataclasses.dataclass(frozen=True)
class Case:
    lattice: str
    polarization: str
    resolution: int
    radius: float
    eps_rod: float
    eps_bg: float

    @property
    def params(self) -> dict[str, Any]:
        return dict(dataclasses.asdict(self), bands=BANDS, intervals=INTERVALS)

    @property
    def stratum(self) -> tuple[str, str, int]:
        return self.lattice, self.polarization, self.resolution

    @property
    def contrast(self) -> float:
        return max(self.eps_rod, self.eps_bg) / min(self.eps_rod, self.eps_bg)

    @property
    def id(self) -> str:
        canonical = json.dumps(self.config(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        return (f"{'sq' if self.lattice == 'square' else 'tr'}-{self.polarization}-"
                f'{self.resolution:03d}-r{round(100 * self.radius):02d}-'
                f'e{self.eps_rod:02.0f}-{self.eps_bg:02.0f}')

    def config(self) -> dict[str, Any]:
        return {
            'schema': 'blaze2d/1', 'task': 'bands', 'polarization': self.polarization,
            'geometry': {
                'background_epsilon': self.eps_bg,
                'lattice': {'type': self.lattice, 'a': 1.0},
                'objects': [{'name': 'rod', 'kind': 'circle', 'center': [0.0, 0.0],
                             'radius': self.radius, 'epsilon': self.eps_rod}],
            },
            'grid': {'resolution': self.resolution},
            'bands': {'count': BANDS,
                      'path': {'preset': self.lattice, 'intervals_per_segment': INTERVALS}},
            'dielectric': {'smoothing': 'analytic'},
            'results': {'eigenvectors': False},
        }


def full_matrix() -> list[Case]:
    return [Case(lattice, polarization, resolution, radius, eps_rod, eps_bg)
            for lattice in LATTICES for polarization in POLARIZATIONS
            for resolution in RESOLUTIONS for radius in RADII
            for eps_rod in EPS_ROD for eps_bg in EPS_BG]


def _pair(case: Case) -> tuple[float, float]:
    return case.eps_rod, case.eps_bg


def timing_roles(work: dict[str, float] | None = None) -> dict[str, tuple[str, str]]:
    """Map case id to (role, basis): easy/median/hard per lattice x polarization x N stratum.

    With baseline work (case id -> total iterations) covering at least three cases of a
    stratum, roles are the minimum, lower median and maximum work; otherwise they are the
    lowest, lower-median and highest dielectric contrast at radius 0.30.
    """
    strata: dict[tuple, list[Case]] = {}
    for case in full_matrix():
        strata.setdefault(case.stratum, []).append(case)
    roles = {}
    for members in strata.values():
        measured = [c for c in members if work and c.id in work]
        if len(measured) >= 3:
            ranked, basis = sorted(measured, key=lambda c: (work[c.id], c.label)), 'work'
        else:
            ranked = sorted((c for c in members if c.radius == 0.30),
                            key=lambda c: (c.contrast, c.eps_rod, c.eps_bg))
            basis = 'contrast'
        chosen = (ranked[0], ranked[(len(ranked) - 1) // 2], ranked[-1])
        for role, case in zip(ROLES, chosen, strict=True):
            roles[case.id] = (role, basis)
    return roles


def subset(name: str, work: dict[str, float] | None = None) -> list[Case]:
    cases = full_matrix()
    if name == 'full768':
        return cases
    if name == 'dev96':
        return [c for c in cases if c.resolution in (16, 32) and _pair(c) in SCREEN_PAIRS]
    if name == 'screen32':
        return [c for c in cases if c.resolution == 32 and c.radius in (0.15, 0.45)
                and _pair(c) in SCREEN_PAIRS]
    if name == 'pilot8':
        return [c for c in cases if c.resolution in (64, 128) and c.radius == 0.30
                and _pair(c) == (1.0, 13.0)]
    if name == 'timing48':
        roles = timing_roles(work)
        return [c for c in cases if c.id in roles]
    raise ValueError(f'unknown subset {name!r}; expected one of {", ".join(SUBSETS)}')


def load_work(path: Path, variant: str | None = None) -> dict[str, float]:
    """Total iterations per case from a work records file (last completed record wins)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get('tag', 'work') == 'work' and r.get('validation', {}).get('completed')]
    names = sorted({r['variant'] for r in rows})
    if variant is None and len(names) > 1:
        raise ValueError(f'{path} holds variants {names}; choose one')
    return {r['case_id']: r['iterations_total'] for r in rows
            if variant is None or r['variant'] == variant}


def select(name: str, *, match: str | None = None, ids: Iterable[str] = (),
           limit: int | None = None, work: dict[str, float] | None = None) -> list[Case]:
    cases = subset(name, work)
    wanted = set(ids)
    if wanted:
        cases = [c for c in cases if c.id in wanted or c.label in wanted]
    if match:
        cases = [c for c in cases if fnmatch.fnmatchcase(c.label, match)]
    return cases[:limit] if limit is not None else cases


def load_variants(path: Path = VARIANTS) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text())


def overlay(variant: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in variant.items() if k not in RESERVED}


def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge patch into a copy of base; None deletes a key, lists replace."""
    result = dict(base)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value))
    return result


_BARE = re.compile(r'^[A-Za-z0-9_-]+$')


def _key(key: str) -> str:
    return key if _BARE.match(key) else json.dumps(key)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'nan'
        return repr(value) if math.isfinite(value) else ('inf' if value > 0 else '-inf')
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(map(_value, value)) + ']'
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{_key(k)} = {_value(v)}' for k, v in value.items()) + ' }'
    raise TypeError(f'cannot render {type(value).__name__} as TOML')


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _table(lines: list[str], table: dict[str, Any], path: tuple[str, ...]) -> None:
    for key, value in table.items():
        if not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f'{_key(key)} = {_value(value)}')
    for key, value in table.items():
        name = '.'.join(map(_key, path + (key,)))
        if isinstance(value, dict):
            lines.append(f'[{name}]')
            _table(lines, value, path + (key,))
        elif _is_table_array(value):
            for item in value:
                lines.append(f'[[{name}]]')
                _table(lines, item, path + (key,))


def toml_text(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _table(lines, data, ())
    return '\n'.join(lines) + '\n'


def render(case: Case, variant: dict[str, Any]) -> str:
    return toml_text(merge(case.config(), overlay(variant)))


def work_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    """Baseline work for timing48 roles from --work-file/--work-variant, if given."""
    try:
        return load_work(args.work_file, args.work_variant) if args.work_file else None
    except (OSError, ValueError) as error:
        raise SystemExit(f'--work-file: {error}') from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'render'):
        command = commands.add_parser(name)
        command.add_argument('--subset', choices=SUBSETS, default='full768')
        command.add_argument('--match', help='glob on case labels, e.g. "sq-T?-016-*"')
        command.add_argument('--case', action='append', default=[], help='case id or label')
        command.add_argument('--limit', type=int)
        command.add_argument('--work-file', type=Path, help='records.jsonl for timing48 roles')
        command.add_argument('--work-variant')
    commands.choices['list'].add_argument('--json', action='store_true')
    commands.choices['render'].add_argument('--variant', required=True)
    commands.choices['render'].add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    work = work_from_args(args)
    cases = select(args.subset, match=args.match, ids=args.case, limit=args.limit, work=work)
    roles = timing_roles(work) if args.subset == 'timing48' else {}
    if args.command == 'list':
        for case in cases:
            role = roles.get(case.id)
            row = dict(id=case.id, label=case.label, **case.params,
                       **({'role': role[0], 'basis': role[1]} if role else {}))
            print(json.dumps(row) if args.json else
                  f"{case.id}  {case.label}" + (f'  {role[0]} ({role[1]})' if role else ''))
        if not args.json:
            print(f'{len(cases)} cases', file=sys.stderr)
        return
    variants = load_variants()
    if args.variant not in variants:
        parser.error(f'unknown variant {args.variant!r}; known: {", ".join(variants)}')
    args.out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (args.out / f'{case.label}.toml').write_text(render(case, variants[args.variant]))
    print(f'rendered {len(cases)} cases with {args.variant} into {args.out}')


if __name__ == '__main__':
    main()
