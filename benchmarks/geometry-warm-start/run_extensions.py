"""M7/M8 paired experiments using the established public-workflow collector.

Case files are explicit JSON arrays, not a second benchmarking framework. Each
record selects geometry, precision, concurrency and two warm-start dictionaries.
All work, including failed training and cold worker roots, remains in the run.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import SimpleNamespace

from run_paths import configuration, compare_retained_fields
from run_native import native_work

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/symmetry'))
from quiet_host import QuietHost
from run_public_sweep import execute, estimate


def table(name, values):
    lines = [f'[{name}]']
    for key, value in values.items():
        if not isinstance(value, dict):
            lines.append(f'{key} = {json.dumps(value)}')
    for key, value in values.items():
        if isinstance(value, dict):
            lines += ['', table(f'{name}.{key}', value)]
    return '\n'.join(lines)


def work(path):
    result = native_work(path)
    result.update(model_builds=0, model_reuses=0, model_accepted=0, model_rejected=0,
                  model_training_iterations=0, model_training_a_columns=0, model_basis_a_columns=0,
                  model_target_columns=0, max_model_payload=0, transport_accepted=0,
                  transport_terms=0, transport_target_columns=0, transport_ffts=0)
    reasons = {}
    data = json.loads(gzip.decompress(path.read_bytes()))
    for job in data.get('results', []):
        for point in job['metadata'].get('geometry_warm_start', {}).get('points', []):
            if not point:
                continue
            model = point.get('model') or {}
            result['model_accepted'] += bool(model.get('accepted'))
            result['model_rejected'] += bool(model) and not model.get('accepted', False)
            result['max_model_payload'] = max(result['max_model_payload'], point['insertion'].get('model_payload_bytes', 0))
            for attempt in model.get('attempts', []):
                result['model_reuses' if attempt['reused'] else 'model_builds'] += 1
                training = attempt.get('training') or {}
                result['model_training_iterations'] += sum(c['iterations'] for c in training.get('inner_columns', []))
                result['model_training_a_columns'] += attempt['work']['response'].get('inner_a_columns', 0)
                result['model_basis_a_columns'] += attempt['work']['basis_a_columns']
                result['model_target_columns'] += attempt['work']['target_preparation_columns']
                if attempt.get('rejection'):
                    reasons[attempt['rejection']] = reasons.get(attempt['rejection'], 0) + 1
            transport = point.get('transport') or {}
            result['transport_accepted'] += bool(transport.get('accepted'))
            result['transport_terms'] += transport.get('interpolation_terms', 0)
            result['transport_target_columns'] += transport.get('target_preparation_columns', 0)
            result['transport_ffts'] += transport.get('forward_ffts', 0) + transport.get('inverse_ffts', 0)
            if transport.get('reason'):
                reasons[transport['reason']] = reasons.get(transport['reason'], 0) + 1
    result['reasons'] = reasons
    return result


def inputs(case, side):
    precision = case.get('precision', 'f64')
    count = case.get('geometries', 12)
    family = case.get('family', 'radius')
    # The existing generator supplies independently named contrast/radius axes.
    text, accuracy = configuration(case['lattice'], case['pol'], case['size'], precision,
        case.get('fields', False), 'radius' if family in ('rigid', 'local', 'epsilon') else family,
        'public', 'path', 'same_k', 2, case.get('bands', 8), count)
    if case.get('gamma'):
        line = next(line for line in text.splitlines() if line.startswith('path ='))
        text = text.replace(line, 'path = { points = [[0.0, 0.0]], basis = "reciprocal_fractional" }')
    if family == 'epsilon':
        # A single fixed radius and a genuine contrast chain, including model reuse.
        text = text.replace('values = [7.3, 11.7]', 'values = '+str([7.3+0.03*i for i in range(2*count)]))
        radius_values = text.split('name = "geometry"')[1].splitlines()[2]
        text = text.replace(radius_values, 'values = [0.193]')
    if family in ('rigid', 'local'):
        text = text.replace('target = "geometry.objects.rod.radius"', 'target = "geometry.objects.rod.center"')
        values = [[0.011 + 0.014*i/(count-1), 0.013] for i in range(count)]
        line = text.split('name = "geometry"')[1].splitlines()[2]
        assert line.startswith('values =')
        text = text.replace(line, f'values = {values}')
        if family == 'local':
            text = text.replace('[grid]', '[[geometry.objects]]\nname = "fixed"\nkind = "circle"\ncenter = [-0.3, -0.27]\nradius = 0.035\nepsilon = 3.1\n[grid]')
    if family == 'local' and 'fixed_center' in case:
        text = text.replace('center = [-0.3, -0.27]', 'center = '+str(case['fixed_center']))
    tolerance = 1e-6 if precision == 'f32' else 1e-13
    text = text.replace('tolerance = 1e-14', f'tolerance = {tolerance}')
    options = case[side]
    if options:
        text += '\n' + table('bands.warm_start', {'scope':'path', 'guards':2,
            'max_cache_bytes':64 << 20, **options}) + '\n'
    if case.get(f'{side}_auto'):
        text += '\n[bands.symmetry]\npolicy = "auto"\n'
    return text, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--filter', default='')
    args = parser.parse_args()
    rng = random.Random(2026092817)
    for case in json.loads(args.cases.read_text()):
        name = case['name']
        if args.filter not in name:
            continue
        out = args.output / name
        if (out / 'summary.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=False)
        for side in ['reference', 'candidate']:
            text, accuracy = inputs(case, side)
            (out / f'{side}.toml').write_text(text)
        cfg = SimpleNamespace(reference=(args.baseline if case.get('baseline') else args.binary).resolve(),
            candidate=args.binary.resolve(), output=out, threads=case.get('threads', 2),
            expected_jobs=2*case.get('geometries', 12), accuracy=accuracy)
        manifest = dict(case=case, source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            binaries={side: dict(path=str(getattr(cfg, side)), sha256=hashlib.sha256(getattr(cfg, side).read_bytes()).hexdigest()) for side in ['reference', 'candidate']},
            inputs={side: hashlib.sha256((out / f'{side}.toml').read_bytes()).hexdigest() for side in ['reference', 'candidate']},
            collectors={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'benchmarks/symmetry/run_public_sweep.py', ROOT/'benchmarks/geometry-warm-start/run_paths.py']},
            pairs=args.pairs, warmups=1, accuracy=accuracy, scope='complete CLI workflow including cold roots, training, correction, certification and requested output')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        blocks = []
        with QuietHost() as host:
            (out / 'preflight.json').write_text(json.dumps(host.interval(2, limit=2, windows_limit=4), indent=2)+'\n')
            for index in range(-1, args.pairs):
                label = 'warmup' if index < 0 else f'pair-{index:02d}'
                sides = ['reference', 'candidate']; rng.shuffle(sides)
                block = {side: execute(cfg, host, side, label) for side in sides}
                for side in sides:
                    block[side]['native_work'] = work(out / f'{label}-{side}.json.gz')
                block['valid'] = all(b['returncode'] == 0 and not b['validation']['errors'] for b in block.values())
                if block['valid']:
                    a, b = (block[side]['validation']['spectra'] for side in ['reference', 'candidate'])
                    block['max_frequency_difference'] = max(abs(x-y) for aa, bb in zip(a, b, strict=True) for ar, br in zip(aa, bb, strict=True) for x, y in zip(ar, br, strict=True))
                    block['valid'] &= block['max_frequency_difference'] < (2e-5 if case.get('precision') == 'f32' else 2e-7)
                    if case.get('fields'):
                        block['field_comparison'] = compare_retained_fields(out, label)
                        block['valid'] &= block['field_comparison']['maximum_principal_sine'] < (0.05 if case.get('precision') == 'f32' else 0.01)
                block['admitted'] = all(block[side]['host']['accepted'] for side in sides)
                for side in sides:
                    block[side]['validation'].pop('spectra', None)
                blocks.append(block)
                (out / 'blocks.json').write_text(json.dumps(blocks, indent=2)+'\n')
                print(name, label, 'valid', block['valid'], 'admitted', block['admitted'],
                      {side: round(block[side]['elapsed_seconds'], 3) for side in sides}, flush=True)
                if not block['valid']:
                    break
        ratios = [b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in blocks[1:] if b['valid'] and b['admitted']]
        descriptive = [b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in blocks[1:] if b['valid']]
        (out / 'summary.json').write_text(json.dumps(dict(numerical_pass=all(b['valid'] for b in blocks),
            ratio=estimate(ratios) if ratios else None, paired_median_ratio=statistics.median(ratios) if ratios else None,
            descriptive_median_ratio=statistics.median(descriptive) if descriptive else None, admitted_pairs=len(ratios)), indent=2)+'\n')


if __name__ == '__main__':
    main()
