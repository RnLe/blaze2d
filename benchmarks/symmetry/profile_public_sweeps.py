#!/usr/bin/env python3
"""Explanatory public CLI profiles, separate from paired primary timings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from quiet_host import QuietHost, assess
from run_public_sweep import inspect


def archived_result(directory, name):
    value = json.loads(gzip.decompress((directory / name).read_bytes()))
    for result in value['results']:
        if 'arrays_object_sha256' in result:
            digest = result.pop('arrays_object_sha256')
            raw = gzip.decompress((directory / 'objects' / f'{digest}.json.gz').read_bytes())
            assert hashlib.sha256(raw).hexdigest() == digest
            result['arrays'] = json.loads(raw)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--comparison', action='append', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(args.binary.with_suffix('.provenance.json').read_text())
    digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    assert provenance['binary_sha256'] == digest and provenance['profiled']
    manifest = dict(binary=str(args.binary), provenance=provenance,
                    comparisons=[str(p.resolve()) for p in args.comparison],
                    purpose='work counters and explanatory instrumented timing; no primary speedup')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'harness.py').write_bytes(Path(__file__).read_bytes())
    records = []
    with QuietHost() as host:
        checks = [host.interval(3, limit=2.0, windows_limit=4.0)]
        if not checks[-1]['accepted']:
            checks.extend(host.interval(3, limit=2.0, windows_limit=4.0) for _ in range(2))
        (args.output / 'preflight.json').write_text(json.dumps(checks, indent=2) + '\n')
        if not any(c['accepted'] for c in checks):
            raise SystemExit('Competing load; no profiles started')
        for directory in args.comparison:
            design = json.loads((directory / 'manifest.json').read_text())
            for variant in ['reference', 'candidate']:
                stem = f'{directory.name}-{variant}'
                profile = args.output / f'{stem}.profile.json'
                resources = args.output / f'{stem}.resource.json'
                command = ['/usr/bin/time', '-f', '{"rss_kib":%M,"user_seconds":%U,"system_seconds":%S}',
                           '-o', str(resources), str(args.binary), 'run', str(directory / f'{variant}.toml'),
                           '--threads', str(design['threads']), '--profile-output', str(profile)]
                before = host.snapshot()
                start = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
                run = subprocess.run(command, capture_output=True, timeout=900)
                elapsed = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - start
                after = host.snapshot()
                row = dict(comparison=str(directory), variant=variant, command=command,
                           returncode=run.returncode, stderr=run.stderr.decode(), elapsed_seconds=elapsed,
                           start=before, end=after, host=assess(before, after, limit=2.0, windows_limit=4.0),
                           resources=json.loads(resources.read_text().splitlines()[-1]))
                (args.output / f'{stem}.json.gz').write_bytes(gzip.compress(run.stdout, mtime=0))
                result = json.loads(run.stdout)
                checked = inspect(result)
                expected = archived_result(directory, f'warmup-{variant}.json.gz')
                old = inspect(expected)
                row['validation'] = {k:v for k,v in checked.items() if k != 'spectra'}
                row['max_frequency_difference'] = max(abs(a-b)
                    for actual, reference in zip(checked['spectra'], old['spectra'], strict=True)
                    for x,y in zip(actual, reference, strict=True) for a,b in zip(x,y, strict=True))
                row['same_iterations'] = checked['work']['iterations'] == old['work']['iterations']
                row['arrays_identical'] = all(a['arrays'] == b['arrays']
                    for a,b in zip(result['results'], expected['results'], strict=True))
                data = json.loads(profile.read_text())
                row['balanced_profile'] = data['counters'].get('profiler.unbalanced_timers', 0) == 0
                row['complete_iteration_total'] = data['counters'].get('iterations', 0) == checked['work']['iterations']
                row['accepted'] = (run.returncode == 0 and not checked['errors'] and row['same_iterations']
                                   and row['max_frequency_difference'] < 2e-7 and row['balanced_profile']
                                   and row['complete_iteration_total'])
                (args.output / f'{stem}.measurement.json').write_text(json.dumps(row, indent=2) + '\n')
                records.append(row)
                print(stem, 'accepted=', row['accepted'], 'identical_arrays=', row['arrays_identical'],
                      'host=', row['host']['accepted'], flush=True)
                if not row['accepted']:
                    raise SystemExit('Profile disagrees with its unprofiled workflow; retain and investigate')
    assert hashlib.sha256(args.binary.read_bytes()).hexdigest() == digest
    (args.output / 'complete.json').write_text(json.dumps(dict(records=len(records), accepted=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
