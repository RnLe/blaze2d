#!/usr/bin/env python3
"""Repeat convergence, negative controls and profiles separately from paired timing."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

from quiet_host import QuietHost, assess, elapsed_clocks, MEASUREMENT_PROTOCOL
from replay_quiet import HERE, ROOT, digest, numerical, backend_dependencies
from replay_record_study import canonical, payload_sha, load_object, unpack, expected_measurement


def stable(value, new_profile=False):
    value = numerical(value)
    if new_profile:
        value = {k: v for k, v in value.items() if k not in ('counters', 'stage_counters', 'timings', 'profile', 'counters_available')}
    return value


def validate(value, old, family, original):
    """Timing may change; fields, certificates, work and rejection reasons may not."""
    if family in ('D04', 'coarse', 'root-profiles'):
        if stable(value, family == 'coarse') != stable(old['result'], family == 'coarse'):
            return ['Numerical output, certificate or work changed']
        if family == 'coarse' and (not value.get('counters') or value.get('counters_available') is not True):
            return ['Requested profile produced no numerical work counters']
        return []
    if family in ('D05', 'dispatch'):
        measurement = dict(value['measurement'])
        references = measurement.pop('references')
        issues = []
        if payload_sha(value['numerics']) != old['numerics_sha256']:
            issues.append('Complete numerical snapshot changed')
        expected = dict(old['measurement']); expected.pop('references', None)
        if stable(measurement) != stable(expected):
            issues.append('Profile work or non-timing metadata changed')
        reference_sha = old.get('references_sha256')
        if reference_sha is None:
            if references is not None:
                issues.append('Unexpected supplementary reference')
        elif references is None or stable(references) != stable(load_object(original, reference_sha)):
            issues.append('Supplementary reference spectrum, fields or certificate changed')
        return issues
    nvalue, measurement = unpack(value, family)
    issues = []
    if payload_sha(nvalue) != old['numerics_sha256']:
        issues.append('Complete numerical snapshot changed')
    if stable(measurement) != stable(expected_measurement(old, family)):
        issues.append('Profile work or non-timing metadata changed')
    return issues


def prepare(family):
    sources, artifacts = {}, {}
    def read(path):
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())
    if family == 'D04':
        original = HERE
        data = read(HERE/'D04-convergence.json')
        audit = read(HERE/'D04-audit.json')
        if audit['issues'] or audit['dataset_sha256'] != sources[str(HERE/'D04-convergence.json')]:
            raise ValueError('Consolidated D04 archive has no valid matching audit')
        for name in ('D04-study-manifest.json', 'D04-subspace-correction-manifest.json'):
            manifest = read(HERE/name)
            for artifact in manifest['artifacts'].values():
                artifacts[artifact['path']] = dict(sha256=artifact['sha256'], source_revision=manifest['source_revision'])
        rows = data['records']
        for row in rows:
            if artifacts[row['command'][3]]['source_revision'] != row['source_revision']:
                raise ValueError('D04 command uses a different diagnostic revision')
        env = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    elif family == 'coarse':
        original = HERE/'coarse-workflow-reference'
        manifest = read(original/'manifest.json')
        builds = read(original/'builds.json')
        audit = read(original/'audit.json')
        artifact = builds['artifacts']['profiled']
        artifacts[artifact['path']] = dict(sha256=artifact['sha256'], source_revision=builds['source_revision'])
        rows = []
        for path in sorted((original/'pilots').glob('*.json')):
            pilot = read(path); m = pilot['manifest']
            for sample in pilot['samples']:
                result = sample['result']
                command = ['taskset', '-c', str(m['cpu']), artifact['path'], m['case'], str(m['grid']),
                           str(m['bands']), m['polarization'], result['mode'], str(m['tolerance'])]
                rows.append(dict(study=path.stem, label=sample['label'], phase='profile', command=command,
                                 result=result, returncode=0))
        env = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    elif family == 'root-profiles':
        original = HERE
        rows = []
        archives = [name+'-work-counters.json' for name in
                    ['C08-shared', 'L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08']]
        archives.append('L01-heldout-TE-diagnostic.json')
        for filename in archives:
            path = HERE/filename
            family_name = filename.split('-')[0]
            data = read(path)
            revision = data.get('revision', data.get('source_revision'))
            if family_name == 'L08':
                build = read(HERE/'L08-build-manifest.json')
                study = read(HERE/'L08-study-manifest.json')
                if revision != build['source_revision'] or revision != study['source_revision']:
                    raise ValueError('L08 profile revision changed')
                for entry in study['wrappers'].values():
                    artifacts[entry['path']] = dict(sha256=entry['sha256'], source_revision=revision)
                for dependency in backend_dependencies(path):
                    sources[dependency['path']] = dependency['sha256']
            else:
                binaries = {row['command'][3] for row in data['records']}
                if len(binaries) != 1:
                    raise ValueError('Profile archive contains undeclared binaries')
                binary = binaries.pop()
                entry = dict(sha256=data['binary_sha256'], source_revision=revision)
                if binary in artifacts and artifacts[binary] != entry:
                    raise ValueError('Conflicting profile artifact provenance')
                artifacts[binary] = entry
            for index, row in enumerate(data['records']):
                if row['returncode'] != 0 or row.get('result') is None:
                    raise ValueError('Archive has no complete JSON diagnostic result')
                rows.append(dict(row, source_record_file=str(path), source_record_index=index))
        env = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    else:
        directory = dict(D05='D05', dispatch='dispatch', PT='pt-sector-measurements',
                         memory='memory-admission-verified')[family]
        original = HERE/directory
        manifest = read(original/'manifest.json')
        audit = read(original/'audit.json')
        if audit['issues']:
            raise ValueError('Original supplementary audit contains unresolved issues')
        records = original/'records.jsonl'; sources[str(records)] = digest(records)
        rows = [json.loads(line) for line in records.read_text().splitlines()]
        rows = [row for row in rows if row['phase'] not in ('primary', 'warmup', 'primary_regression', 'warmup_regression')]
        builds = manifest['builds']
        for name, artifact in builds['artifacts'].items():
            revision = builds.get('source_revision') or builds.get('candidate_revision' if name.startswith('candidate') else 'parent_revision')
            artifacts[artifact['path']] = dict(sha256=artifact['sha256'], source_revision=revision)
        if family == 'dispatch':
            parent = (original/manifest['parent']).resolve()
            for name, key in [('manifest.json', 'parent_manifest_sha256'), ('records.jsonl', 'parent_records_sha256')]:
                if digest(parent/name) != manifest[key]:
                    raise ValueError('Dispatch parent changed')
                sources[str(parent/name)] = digest(parent/name)
        if family == 'memory':
            if not builds.get('complete') or len({a['target_directory'] for a in builds['artifacts'].values()}) != 4:
                raise ValueError('Memory builds were not isolated')
            for kind in ('unprofiled', 'profiled'):
                if builds['artifacts']['baseline_'+kind]['sha256'] == builds['artifacts']['candidate_'+kind]['sha256']:
                    raise ValueError('Identical baseline and candidate executables')
        for row in rows:
            load_object(original, row['numerics_sha256'])
            if row.get('references_sha256'):
                load_object(original, row['references_sha256'])
        env = manifest['thread_environment']
    # One new diagnostic execution per distinct frozen command. Identical old
    # preflight/profile invocations remain mapped, never counted as new samples.
    selected = {}
    for index, row in enumerate(rows):
        command = row['command']
        if command[:2] != ['taskset', '-c'] or command[3] not in artifacts:
            raise ValueError('Undeclared executable/affinity')
        key = tuple(command)
        selected.setdefault(key, dict(command=command, original_indices=[], rows=[]))
        selected[key]['original_indices'].append(index)
        selected[key]['rows'].append(row)
        if 'source_record_file' in row:
            selected[key].setdefault('source_records', []).append(dict(
                file=row['source_record_file'], index=row['source_record_index']))
    used = {item['command'][3] for item in selected.values()}
    artifacts = {path: value for path, value in artifacts.items() if path in used}
    if any(digest(path) != value['sha256'] for path, value in artifacts.items()):
        raise ValueError('Frozen executable changed')
    return original, list(selected.values()), sources, artifacts, env, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['D04', 'D05', 'dispatch', 'PT', 'memory', 'coarse', 'root-profiles'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original, selected, sources, artifacts, env, old_count = prepare(args.family)
    out = args.output.resolve(); out.mkdir(exist_ok=False, parents=True)
    (out/'outputs').mkdir()
    harnesses = {str(path): digest(path) for path in [Path(__file__), HERE/'replay_quiet.py',
        HERE/'replay_record_study.py', HERE/'quiet_host.py', HERE/'host_clock.ps1',
        HERE/'diagnostic-replay-protocol.md']}
    metadata = dict(schema='blaze2d/diagnostic-replay/1', family=args.family,
        original=str(original), source_sha256=sources, artifacts=artifacts, harness_sha256=harnesses,
        measurement_protocol=MEASUREMENT_PROTOCOL, thread_environment=env, started=time.time(),
        scope='Serial convergence/work/negative-control replay; descriptive timing only, no paired latency claim',
        original_record_count=old_count, unique_commands=len(selected),
        mapping=[{k:v for k,v in item.items() if k != 'rows'} for item in selected])
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    records = []
    with QuietHost() as host:
        admission = [host.interval(limit=1.0, windows_limit=3.0) for _ in range(2)]
        (out/'admission.json').write_bytes(canonical(admission)+b'\n')
        # These are deterministic numerical/work reproductions, not primary
        # timing samples. Keep load observations without retrying a numerical
        # experiment merely to obtain a cleaner descriptive time.
        for index, item in enumerate(selected):
            row = dict(index=index, command=item['command'], original_indices=item['original_indices'], issues=[])
            start = host.snapshot()
            try:
                process = subprocess.run(item['command'], cwd=ROOT, env=dict(os.environ, **env), capture_output=True, timeout=1200)
            except subprocess.TimeoutExpired as error:
                process = None
                stdout, stderr = error.stdout or b'', error.stderr or b''
                row['issues'].append('Process timeout')
            end = host.snapshot()
            if process is not None:
                stdout, stderr = process.stdout, process.stderr
                row['returncode'] = process.returncode
            (out/'outputs'/f'{index:04d}.stdout.gz').write_bytes(gzip.compress(stdout, mtime=0))
            (out/'outputs'/f'{index:04d}.stderr.gz').write_bytes(gzip.compress(stderr, mtime=0))
            row['window'] = dict(start=start, end=end, assessment=assess(start, end))
            if process is not None:
                try:
                    if process.returncode != 0:
                        raise ValueError('Benchmark process failed')
                    value = json.loads(stdout)
                    row['issues'] += list(dict.fromkeys(issue for old in item['rows']
                        for issue in validate(value, old, args.family, original)))
                    row['full_output_sha256'] = payload_sha(value)
                    measurement = value['measurement'] if args.family in ('D05', 'dispatch') else value
                    row['reported_seconds'] = measurement['seconds']
                    row['peak_rss_kib'] = measurement['peak_rss_kib']
                    row['clock_projection'] = elapsed_clocks(start, end)
                    row['normalized_seconds_descriptive'] = measurement['seconds']*row['clock_projection']['adjusted_to_raw_scale']
                    row['counters'] = measurement.get('counters')
                    row['stage_counters'] = measurement.get('stage_counters')
                except (ValueError, KeyError, TypeError) as error:
                    row['issues'].append(str(error))
            records.append(row)
            with (out/'records.jsonl').open('ab') as stream:
                stream.write(canonical(row)+b'\n')
            if row['issues']:
                break
            if (index+1) % 10 == 0:
                print(json.dumps(dict(completed=index+1, declared=len(selected))), flush=True)
    unchanged = all(digest(path) == sha for path, sha in sources.items()) and all(
        digest(path) == value['sha256'] for path, value in artifacts.items()) and all(
        digest(path) == sha for path, sha in harnesses.items())
    audit = dict(schema='blaze2d/diagnostic-replay-audit/1', complete=len(records) == len(selected),
                 unchanged=unchanged, records=len(records), unique_commands=len(selected),
                 original_record_count=old_count, issues=[dict(index=r['index'], issues=r['issues']) for r in records if r['issues']],
                 records_sha256=digest(out/'records.jsonl'), primary_latency_qualified=False)
    audit['numerical_reproduction_passed'] = audit['complete'] and unchanged and not audit['issues']
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)
    return 0 if audit['numerical_reproduction_passed'] else 4


if __name__ == '__main__':
    raise SystemExit(main())
