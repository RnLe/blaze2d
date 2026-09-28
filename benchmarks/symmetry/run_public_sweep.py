#!/usr/bin/env python3
"""Paired public CLI studies; raw/QPC clocks, complete output and load evidence."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

from quiet_host import QuietHost, assess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(args, candidate):
    symmetry = {'off':'', 'identical':'k_reuse = "identical"',
                'pt':'arithmetic = "pt_real"\nk_reuse = "time_reversal"'}[args.control]
    if candidate:
        if args.policy == 'continuation':
            symmetry = 'continuation = "previous_geometry"'
        elif args.policy == 'pt_continuation':
            symmetry = 'continuation = "previous_geometry"\narithmetic = "pt_real"\nk_reuse = "time_reversal"'
        elif args.policy == 'pt':
            symmetry = 'arithmetic = "pt_real"\nk_reuse = "time_reversal"'
        elif args.policy == 'auto':
            symmetry = 'policy = "auto"'
    return f'''schema = "blaze2d/1"
task = "bands"
polarization = "{args.polarization}"
[geometry]
background_epsilon = 1.0
lattice = {{ type = "{args.lattice}", a = 1.0 }}
[[geometry.objects]]
name = "rod"
kind = "circle"
center = [0.0, 0.0]
radius = 0.2
epsilon = 8.9
[grid]
resolution = {args.size}
[bands]
count = {args.bands}
tracking = true
path = {{ preset = "{args.lattice}", intervals_per_segment = {args.intervals} }}
[bands.symmetry]
{symmetry}
[eigensolver]
tolerance = 1e-14
max_iterations = 600
[results]
eigenvectors = {str(args.fields).lower()}
[[sweeps]]
name = "contrast"
target = "geometry.objects.rod.epsilon"
values = [8.9, 12.0]
[[sweeps]]
name = "radius"
target = "geometry.objects.rod.radius"
values = [0.18, 0.20, 0.22]
'''


def inspect(study):
    errors = []
    work = dict(iterations=0, seeded_points=0, exact_geometry_hits=0,
                reused_points=0, continuation_fallbacks=0, max_cache_bytes=0)
    spectra = []
    if study['errors'] or study['statistics']['status'] != 'completed':
        errors.append('incomplete study')
    if [r['job_index'] for r in study['results']] != list(range(6)):
        errors.append('missing or reordered jobs')
    for result in study['results']:
        arrays, metadata = result['arrays'], result['metadata']
        if not metadata['converged']:
            errors.append(f"job {result['job_index']}: not converged")
        certificate = metadata['certification']
        if not math.isfinite(certificate['b_orthogonality_defect']) or certificate['b_orthogonality_defect'] > 1e-8:
            errors.append('mass orthogonality')
        frequencies = arrays['frequencies']['data']
        for frequency, relative, absolute in zip(frequencies, arrays['residuals']['data'], arrays['absolute_residuals']['data'], strict=True):
            value, limit = (absolute, 1e-7) if frequency == 0 else (relative, 1e-5)
            if not all(map(math.isfinite, [frequency, relative, absolute])) or value > limit:
                errors.append('fresh residual')
                break
        rows, bands = arrays['frequencies']['shape']
        if len(frequencies) != rows * bands:
            errors.append('frequency shape')
        spectra.append([sorted(frequencies[i*bands:(i+1)*bands]) for i in range(rows)])
        work['iterations'] += sum(metadata['iterations'])
        for point in metadata['symmetry']['points']:
            continuation = point.get('continuation') or {}
            work['seeded_points'] += bool(continuation.get('seed'))
            work['exact_geometry_hits'] += bool(continuation.get('exact_source'))
            work['continuation_fallbacks'] += bool(continuation.get('fallback'))
            work['reused_points'] += point.get('source_index') is not None
            work['max_cache_bytes'] = max(work['max_cache_bytes'], continuation.get('cache_payload_bytes', 0))
        for array in arrays.values():
            factor = 2 if array['dtype'] == 'complex128' else 1
            if len(array['data']) != math.prod(array['shape']) * factor or not all(map(math.isfinite, array['data'])):
                errors.append('array shape or finiteness')
    return dict(errors=sorted(set(errors)), work=work, spectra=spectra)


def execute(args, host, variant, name):
    binary = args.candidate if variant == 'candidate' else args.reference
    output = args.output / f'{name}-{variant}.json'
    rss = args.output / f'{name}-{variant}.resource.json'
    command = ['/usr/bin/time', '-f', '{"rss_kib":%M,"user_seconds":%U,"system_seconds":%S}',
               '-o', str(rss), str(binary), 'run', str(args.output / f'{variant}.toml'),
               '--threads', str(args.threads)]
    before = host.snapshot()
    start = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    with output.open('wb') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE)
    elapsed = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - start
    after = host.snapshot()
    record = dict(command=command, elapsed_seconds=elapsed, returncode=process.returncode,
                  stderr=process.stderr.decode(errors='replace'), start=before, end=after,
                  host=assess(before, after, limit=1.0, windows_limit=3.0),
                  resources=json.loads(rss.read_text()))
    try:
        record['validation'] = inspect(json.loads(output.read_text()))
    except (ValueError, KeyError, TypeError) as error:
        record['validation'] = dict(errors=[str(error)])
    data = output.read_bytes()
    record['output_sha256'] = hashlib.sha256(data).hexdigest()
    output.with_suffix('.json.gz').write_bytes(gzip.compress(data, mtime=0))
    output.unlink()
    (args.output / f'{name}-{variant}.measurement.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def estimate(ratios):
    if not ratios:
        return None
    logs = list(map(math.log, ratios))
    rng = random.Random(2026092802)
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(4000))
    return dict(ratio=math.exp(statistics.mean(logs)), interval=[bootstrap[100], bootstrap[3899]], pairs=len(logs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lattice', choices=['square', 'triangular'], default='square')
    parser.add_argument('--polarization', choices=['TM', 'TE'], default='TM')
    parser.add_argument('--policy', choices=['continuation', 'pt', 'pt_continuation', 'auto'], default='continuation')
    parser.add_argument('--control', choices=['off', 'identical', 'pt'], default='off')
    parser.add_argument('--size', type=int, default=31)
    parser.add_argument('--bands', type=int, default=8)
    parser.add_argument('--intervals', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=9)
    parser.add_argument('--fields', action='store_true')
    args = parser.parse_args()
    args.reference, args.candidate, args.output = args.reference.resolve(), args.candidate.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['reference', 'candidate']:
        (args.output / f'{name}.toml').write_text(configuration(args, name == 'candidate'))
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest['sha256'] = {str(p): digest(p) for p in [args.reference, args.candidate, Path(__file__), Path(__file__).with_name('quiet_host.py')]}
    (args.output / 'harness.py').write_bytes(Path(__file__).read_bytes())
    manifest['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', 'diff', '--binary'])
    (args.output / 'source.patch').write_bytes(patch)
    manifest['source_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    manifest['protocol'] = dict(warmups=1, pairs=args.pairs, linux_background_cores=1.0,
                                windows_background_cores=3.0, practical_band=0.05,
                                minimum_pairs=6, timing_scope='CLI startup, planning, solving and complete JSON output')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rng = random.Random(2026092802)
    blocks = []
    with QuietHost() as host:
        preflight = [host.interval(3, limit=1.0, windows_limit=3.0)]
        if not preflight[-1]['accepted']:
            preflight.extend(host.interval(3, limit=1.0, windows_limit=3.0) for _ in range(2))
        (args.output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
        if not any(sample['accepted'] for sample in preflight):
            raise SystemExit('Sustained competing load in preflight; no timed runs started: ' + str(preflight[-1]['issues']))
        for index in range(-1, args.pairs):
            name = 'warmup' if index < 0 else f'pair-{index:02d}'
            variants = ['reference', 'candidate']
            rng.shuffle(variants)
            block = {v: execute(args, host, v, name) for v in variants}
            valid = all(not r['validation']['errors'] and r['returncode'] == 0 for r in block.values())
            if valid:
                a, b = (block[v]['validation']['spectra'] for v in ['reference', 'candidate'])
                delta = max(abs(x-y) for ga, gb in zip(a,b,strict=True) for ra,rb in zip(ga,gb,strict=True) for x,y in zip(ra,rb,strict=True))
                block['max_frequency_difference'] = delta
                valid = delta <= 2e-7
            block['valid'] = valid
            block['admitted'] = all(block[v]['host']['accepted'] for v in variants)
            blocks.append(block)
            (args.output / 'blocks.json').write_text(json.dumps(blocks, indent=2) + '\n')
            print(name, 'valid=', valid, 'admitted=', block['admitted'],
                  'seconds=', {v: round(block[v]['elapsed_seconds'], 4) for v in variants}, flush=True)
    numerical_pass = all(b['valid'] for b in blocks)
    measured = blocks[1:]
    ratio = lambda b: b['reference']['elapsed_seconds'] / b['candidate']['elapsed_seconds']
    report = dict(numerical_pass=numerical_pass,
                  all_blocks=estimate([ratio(b) for b in measured]) if numerical_pass else None,
                  admitted=estimate([ratio(b) for b in measured if b['admitted']]) if numerical_pass else None,
                  admitted_pair_ids=[i for i,b in enumerate(measured) if b['admitted']],
                  excluded_pair_ids=[i for i,b in enumerate(measured) if not b['admitted']])
    admitted, all_blocks = report['admitted'], report['all_blocks']
    direction = lambda value: 'improvement' if value > 1.05 else ('regression' if value < 1 / 1.05 else 'small')
    report['timing_qualified'] = bool(admitted and admitted['pairs'] >= 6
        and abs(admitted['ratio'] / all_blocks['ratio'] - 1) <= .10
        and direction(admitted['ratio']) == direction(all_blocks['ratio']))
    report['conclusion'] = 'pilot or unresolved'
    if report['timing_qualified']:
        low, high = admitted['interval']
        report['conclusion'] = 'improvement' if low > 1.05 else ('regression' if high < 1 / 1.05 else 'small or inconclusive')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
