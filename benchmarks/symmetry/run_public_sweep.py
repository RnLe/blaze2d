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
                'pt':'arithmetic = "pt_real"\nk_reuse = "time_reversal"',
                'fourier':'arithmetic = "fourier_complex"', 'auto':'policy = "auto"'}[args.control]
    if candidate:
        if args.policy == 'continuation':
            symmetry = 'continuation = "previous_geometry"'
        elif args.policy == 'pt_continuation':
            symmetry = 'continuation = "previous_geometry"\narithmetic = "pt_real"\nk_reuse = "time_reversal"'
        elif args.policy == 'pt_first_point':
            symmetry = 'continuation = "first_point"\narithmetic = "pt_real"\nk_reuse = "time_reversal"'
        elif args.policy == 'pt':
            symmetry = 'arithmetic = "pt_real"\nk_reuse = "time_reversal"'
        elif args.policy in ('pt_coarse', 'pt_combined', 'pt_groups'):
            symmetry = 'arithmetic = "pt_real"\nk_reuse = "time_reversal"'
            if args.policy == 'pt_combined':
                symmetry += '\ncontinuation = "previous_geometry"'
            if args.policy in ('pt_coarse', 'pt_combined'):
                symmetry += f'\ncoarse_start = {{ resolution = [{args.size // 2},{args.size // 2}] }}'
            if args.policy in ('pt_groups', 'pt_combined'):
                symmetry += '\nsectors = "adaptive_group"\nplane_group = "p6m"\nsector_storage = "orbit"\nreconstruct_partners = true'
        elif args.policy == 'auto':
            symmetry = 'policy = "auto"'
    discretization = '' if args.support_cutoff is None else f'\n[bands.discretization]\nkind = "metric_ball"\nangular_cutoff = {math.tau * args.support_cutoff}\nmax_coefficients = 32768'
    radii = [.175, .215, .255] if args.heldout else [.18, .20, .22]
    contrasts = [6.5, 13.0] if args.heldout else [8.9, 12.0]
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
{symmetry}{discretization}
[dielectric]
smoothing = "{args.smoothing}"
[eigensolver]
tolerance = 1e-14
max_iterations = 600
[results]
eigenvectors = {str(args.fields).lower()}
[[sweeps]]
name = "contrast"
target = "geometry.objects.rod.epsilon"
values = {contrasts}
[[sweeps]]
name = "radius"
target = "geometry.objects.rod.radius"
values = {radii}
'''


def inspect(study, expected_jobs=6):
    errors = []
    work = dict(iterations=0, seeded_points=0, exact_geometry_hits=0,
                reused_points=0, continuation_fallbacks=0, max_cache_bytes=0,
                pt_points=0, sector_points=0, coarse_points=0, coarse_iterations=0,
                retained_iterations=0, discarded_iterations=0, qualification_attempts=0)
    spectra = []
    if study['errors'] or study['statistics']['status'] != 'completed':
        errors.append('incomplete study')
    if [r['job_index'] for r in study['results']] != list(range(expected_jobs)):
        errors.append('missing or reordered jobs')
    for result in study['results']:
        arrays, metadata = result['arrays'], result['metadata']
        if not metadata['converged']:
            errors.append(f"job {result['job_index']}: not converged")
        certificate = metadata['certification']
        if not math.isfinite(certificate['b_orthogonality_defect']) or certificate['b_orthogonality_defect'] > 1e-8:
            errors.append('mass orthogonality')
        frequencies = arrays['frequencies']['data']
        width = arrays['frequencies']['shape'][1]
        reports = (metadata.get('discretization') or {}).get('points', [])
        for offset, (frequency, relative, absolute) in enumerate(zip(frequencies, arrays['residuals']['data'], arrays['absolute_residuals']['data'], strict=True)):
            row, band = divmod(offset, width)
            analytic = reports[row]['selected']['analytic_kernel'] and reports[row]['returned_permutation'][band] == 0 if reports else frequency == 0
            if analytic and (math.tau*frequency)**2 > 1e-7:
                errors.append('analytic kernel eigenvalue')
            value, limit = (absolute, 1e-7) if analytic else (relative, 1e-5)
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
            coarse = point.get('coarse_start') or {}
            work['pt_points'] += bool(point.get('coordinates'))
            work['sector_points'] += any(a['retained'] and 'sector' in a['method'] for a in point['solver_attempts'])
            work['coarse_points'] += bool(coarse.get('used_as_seed'))
            work['coarse_iterations'] += coarse.get('iterations', 0)
            work['retained_iterations'] += point['retained_iterations']
            work['discarded_iterations'] += point['attempted_iterations'] - point['retained_iterations'] - coarse.get('iterations', 0)
            work['qualification_attempts'] += point['qualification_attempts']
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
                  host=assess(before, after, limit=2.0, windows_limit=4.0),
                  resources=json.loads(next(line for line in reversed(rss.read_text().splitlines()) if line.startswith('{'))))
    try:
        record['validation'] = inspect(json.loads(output.read_text()), expected_jobs=getattr(args,'expected_jobs',6))
    except (ValueError, KeyError, TypeError) as error:
        record['validation'] = dict(errors=[str(error)])
    data = output.read_bytes()
    record['output_sha256'] = hashlib.sha256(data).hexdigest()
    try: document = json.loads(data)
    except ValueError: document = {'failed_stdout': data.decode(errors='replace')}
    objects = args.output / 'objects'
    objects.mkdir(exist_ok=True)
    for result in document.get('results', []):
        arrays = result.pop('arrays')
        encoded = json.dumps(arrays, separators=(',', ':'), allow_nan=False).encode()
        sha = hashlib.sha256(encoded).hexdigest()
        destination = objects / f'{sha}.json.gz'
        if not destination.exists():
            destination.write_bytes(gzip.compress(encoded, mtime=0))
        result['arrays_object_sha256'] = sha
    record['archive_format'] = 'arrays_object_sha256 references lossless JSON arrays in objects/'
    output.with_suffix('.json.gz').write_bytes(gzip.compress(json.dumps(document, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    output.unlink()
    archived = dict(record, validation={k:v for k,v in record['validation'].items() if k != 'spectra'})
    (args.output / f'{name}-{variant}.measurement.json').write_text(json.dumps(archived, indent=2) + '\n')
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
    parser.add_argument('--policy', choices=['continuation', 'pt', 'pt_continuation', 'pt_first_point', 'pt_coarse', 'pt_combined', 'pt_groups', 'auto'], default='continuation')
    parser.add_argument('--control', choices=['off', 'identical', 'pt', 'fourier', 'auto'], default='off')
    parser.add_argument('--size', type=int, default=31)
    parser.add_argument('--bands', type=int, default=8)
    parser.add_argument('--intervals', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=9)
    parser.add_argument('--fields', action='store_true')
    parser.add_argument('--support-cutoff', type=float)
    parser.add_argument('--smoothing', choices=['analytic', 'none', 'voronoi'], default='analytic')
    parser.add_argument('--heldout', action='store_true')
    args = parser.parse_args()
    args.reference, args.candidate, args.output = args.reference.resolve(), args.candidate.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['reference', 'candidate']:
        (args.output / f'{name}.toml').write_text(configuration(args, name == 'candidate'))
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest['sha256'] = {str(p): digest(p) for p in [args.reference, args.candidate, Path(__file__), Path(__file__).with_name('quiet_host.py')]}
    manifest['executables'] = {}
    for label, binary in [('reference', args.reference), ('candidate', args.candidate)]:
        provenance = json.loads(binary.with_suffix('.provenance.json').read_text())
        if provenance['binary_sha256'] != digest(binary):
            raise SystemExit(f'{label} executable differs from its frozen provenance')
        manifest['executables'][label] = provenance
        source_patch = binary.parent / 'source.patch'
        if provenance.get('source_patch_sha256'):
            if digest(source_patch) != provenance['source_patch_sha256']:
                raise SystemExit(f'{label} source patch differs from its provenance')
            (args.output / f'{label}-source.patch').write_bytes(source_patch.read_bytes())
    (args.output / 'harness.py').write_bytes(Path(__file__).read_bytes())
    manifest['collector_source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', 'diff', '--binary'])
    (args.output / 'source.patch').write_bytes(patch)
    manifest['source_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    manifest['protocol'] = dict(warmups=1, pairs=args.pairs, linux_background_cores=2.0,
                                windows_background_cores=4.0, practical_band=0.05,
                                minimum_pairs=6, timing_scope='CLI startup, planning, solving and complete JSON output')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rng = random.Random(2026092802)
    blocks = []
    with QuietHost() as host:
        preflight = [host.interval(3, limit=2.0, windows_limit=4.0)]
        if not preflight[-1]['accepted']:
            preflight.extend(host.interval(3, limit=2.0, windows_limit=4.0) for _ in range(2))
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
            for variant in variants:
                block[variant]['validation'].pop('spectra', None)
            block['valid'] = valid
            block['admitted'] = all(block[v]['host']['accepted'] for v in variants)
            blocks.append(block)
            (args.output / 'blocks.json').write_text(json.dumps(blocks, indent=2) + '\n')
            print(name, 'valid=', valid, 'admitted=', block['admitted'],
                  'seconds=', {v: round(block[v]['elapsed_seconds'], 4) for v in variants}, flush=True)
            if not valid:
                print('Stopping after failed numerical/output comparison.', flush=True)
                break
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
