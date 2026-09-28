#!/usr/bin/env python3
"""Replay frozen support, sector, dispatch and memory-admission comparisons.

Primary sampling and subsequent independent numerical audits are serial. The
original records, executables, order, outputs, tolerances and block units stay
fixed; only the explicitly versioned host/clock instrumentation is new.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from quiet_host import QuietHost, assess, elapsed_clocks, MEASUREMENT_PROTOCOL
from replay_quiet import digest, numerical, practical, interaction_decision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PT_MODES = ('off', 'pt', 'sectors', 'fourier-sectors', 'pt-sectors')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def payload_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_object(root, sha):
    path = root/'objects'/f'{sha}.json.gz'
    if not path.exists() and (root/'manifest.json').is_file():
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['schema'] == 'blaze2d/support-dispatch/1':
            path = (root/manifest['parent']).resolve()/'objects'/f'{sha}.json.gz'
    payload = gzip.decompress(path.read_bytes())
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError('Archived numerical object changed: ' + sha)
    return json.loads(payload)


def unpack(value, family):
    if family in ('D05', 'dispatch'):
        measurement = dict(value['measurement'])
        references = measurement.pop('references')
        if references is not None:
            raise ValueError('Primary replay unexpectedly ran supplementary references')
        return value['numerics'], measurement
    measurement = {k: value[k] for k in ('seconds', 'peak_rss_kib', 'counters', 'stage_counters', 'profile')}
    if family == 'memory':
        from check_memory_admission import numerics
        measurement['admission'] = [p['symmetry']['sectors'].get('workspace')
                                    if p['symmetry']['sectors'] else None for p in value['points']]
        return numerics(value), measurement
    return {k: v for k, v in value.items() if k not in measurement}, measurement


def expected_measurement(row, family):
    if family in ('D05', 'dispatch'):
        result = dict(row['measurement'])
        if result.pop('references', None) is not None:
            raise ValueError('Archived primary record contains supplementary references')
        return result
    result = {k: row[k] for k in ('seconds', 'peak_rss_kib', 'counters', 'stage_counters', 'profile')}
    if family == 'memory':
        result['admission'] = row['admission']
    return result


def verify_result(value, original, family):
    nvalue, measurement = unpack(value, family)
    issues = []
    if payload_sha(nvalue) != original['numerics_sha256']:
        issues.append('Full numerical snapshot changed')
    if numerical(measurement) != numerical(expected_measurement(original, family)):
        issues.append('Non-timing measurement metadata or work counters changed')
    if not math.isfinite(measurement['seconds']) or measurement['seconds'] <= 0:
        issues.append('Invalid solver duration')
    if type(measurement['peak_rss_kib']) is not int or measurement['peak_rss_kib'] <= 0:
        issues.append('Invalid peak RSS')
    if any(measurement[k] for k in ('profile', 'counters', 'stage_counters')):
        issues.append('Instrumented output in primary sampling')
    return nvalue, measurement, issues


def designs(manifest, family):
    if family == 'D05':
        return {s['name']: dict(kind='paired', spec=s, seed=20260924)
                for s in manifest['design']}
    if family == 'dispatch':
        return {name: dict(kind='paired', seed=2026092402, bootstrap_advance=i)
                for i, name in enumerate(manifest['selected'])}
    if family == 'memory':
        return {'-'.join(case): dict(kind='paired', case=case, seed=manifest['seed']+i)
                for i, case in enumerate(manifest['cases'])}
    result = {}
    for i, (case, mode) in enumerate(manifest['timed_regressions']):
        name = 'refactor-' + '-'.join(map(str, case)) + '-' + mode
        result[name] = dict(kind='refactor', case=case, mode=mode, seed=manifest['seed']+i)
    for i, case in enumerate(manifest['cases']):
        result['interaction-' + '-'.join(map(str, case))] = dict(
            kind='interaction', case=case, seed=manifest['seed']+100*i)
    return result


def select(rows, name, design, family):
    if family in ('D05', 'dispatch'):
        return [r for r in rows if r['study'] == name and r['phase'] in ('warmup', 'primary')]
    if family == 'memory':
        return [r for r in rows if r['case'] == design['case'] and r['phase'] in ('warmup', 'primary')]
    phases = ('warmup', 'primary') if design['kind'] == 'interaction' else ('warmup_regression', 'primary_regression')
    return [r for r in rows if r['case'] == design['case'] and r['phase'] in phases
            and (design['kind'] == 'interaction' or r['mode'] == design['mode'])]


def archived_physical_gate(audit, name, design, family):
    if audit['issues']:
        return False
    if family in ('D05', 'dispatch'):
        matches = [r for r in audit['comparisons'] if r['study'] == name]
        key = 'valid' if family == 'D05' else None
    elif family == 'memory':
        matches = [r for r in audit['summaries'] if r['case'] == design['case']]
        key = None
    else:
        kind = 'interactions' if design['kind'] == 'interaction' else 'regression'
        matches = [r for r in audit[kind] if r['case'] == design['case']
                   and (kind == 'interactions' or r['mode'] == design['mode'])]
        key = 'accepted'
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous archived physical decision')
    return True if key is None else matches[0][key] is True


def grouping(rows, design, family):
    blocks = {}
    for row in rows:
        block = row['index'] if family == 'PT' else row['pair']
        variant = row['mode'] if design['kind'] == 'interaction' else row['label']
        if variant in blocks.setdefault(block, {}):
            raise ValueError('Duplicate variant within a declared block')
        blocks[block][variant] = row
    wanted = set(PT_MODES if design['kind'] == 'interaction' else ('baseline', 'candidate'))
    if set(blocks) != set(range(-1, 20)) or any(set(b) != wanted for b in blocks.values()):
        raise ValueError('Archived selection does not contain one warmup and 20 complete blocks')
    return blocks


def summarize(rows, design, family):
    blocks = grouping(rows, design, family)
    # Preserve each original study's bootstrap generator and quantile convention.
    def interval(logs, seed):
        if family in ('D05', 'dispatch'):
            import numpy as np
            array = np.asarray(logs)
            rng = np.random.default_rng(seed)
            if family == 'dispatch':
                # The original dispatch audit shared one RNG across all four
                # comparisons. Reproduce its state for a standalone replay.
                for _ in range(design['bootstrap_advance']):
                    rng.choice(np.zeros(20), (20000, 20))
            resamples = 20000 if family == 'dispatch' else 10000
            boot = rng.choice(array, (resamples, len(array)), replace=True).mean(axis=1)
            lo, hi = np.exp(np.quantile(boot, [.025, .975]))
            return dict(geometric_ratio=float(np.exp(array.mean())), ci95=[float(lo), float(hi)])
        from run_pt_sectors import interval as old_interval
        return {k: v for k, v in old_interval(logs, seed).items() if k in ('geometric_ratio', 'ci95')}
    def ratios(left, right, seed):
        values = [math.log(blocks[i][left]['normalized_seconds']/blocks[i][right]['normalized_seconds'])
                  for i in range(20)]
        result = interval(values, seed)
        result['practical_decision'] = practical(result['ci95'])
        return result
    if design['kind'] != 'interaction':
        result = {'paired': ratios('baseline', 'candidate', design['seed'])}
        if family == 'dispatch':
            lo, hi = result['paired']['ci95']
            result['paired']['original_protocol_decision'] = (
                'practical regression' if hi < .98 else
                'practical equivalence' if lo >= .98 and hi <= 1.02 else 'equivalence unresolved')
    else:
        contrasts = [('pt', 'off', 'pt'), ('sectors', 'off', 'sectors'), ('combined', 'off', 'pt-sectors'),
                     ('sectors_after_pt', 'pt', 'pt-sectors'), ('pt_after_sectors', 'sectors', 'pt-sectors'),
                     ('fourier_projection', 'sectors', 'fourier-sectors'),
                     ('pt_after_fourier_sectors', 'fourier-sectors', 'pt-sectors')]
        result = {name: ratios(left, right, design['seed']+j) for j, (name, left, right) in enumerate(contrasts)}
        logs = [math.log(blocks[i]['pt-sectors']['normalized_seconds']*blocks[i]['off']['normalized_seconds'] /
                         (blocks[i]['pt']['normalized_seconds']*blocks[i]['sectors']['normalized_seconds'])) for i in range(20)]
        interaction = interval(logs, design['seed']+10)
        interaction['interpretation'] = interaction_decision(interaction['ci95'])
        result['interaction_time_factor'] = interaction
    result['median_seconds'] = {variant: statistics.median(blocks[i][variant]['normalized_seconds'] for i in range(20))
                                for variant in blocks[0]}
    result['median_rss_kib'] = {variant: statistics.median(blocks[i][variant]['measurement']['peak_rss_kib'] for i in range(20))
                                for variant in blocks[0]}
    return result


def current_physical_checks(original, all_rows, chosen, values, design, family):
    """Only called after the final primary process; no numerical audit overlaps it."""
    if family in ('D05', 'dispatch'):
        from support_contract import quality, pair_quality
        checks = {sha: quality(value) for sha, value in values.items()}
        pair = {r['label']: values[r['numerics_sha256']] for r in chosen if r['pair'] == 0}
        comparison = pair_quality(pair['baseline'], pair['candidate'])
        return dict(objects=checks, pair=comparison, passed=not comparison['issues'] and
                    all(not c['issues'] for c in checks.values()))
    if family == 'memory':
        from check_memory_admission import audit_fields
        from run_pairs import quality_passes
        checks = {sha: dict(independent=audit_fields(value),
                            quality=quality_passes(value, value['bands'])) for sha, value in values.items()}
        return dict(objects=checks, passed=all(c['quality'] and not c['independent']['issues']
                                               for c in checks.values()))
    from pt_sector_contract import path_issues, independent
    controls = [r for r in all_rows if r['case'] == design['case'] and r['mode'] == 'off' and r.get('numerics_sha256')]
    if not controls:
        raise ValueError('No independently audited unrestricted reference for this case')
    reference = load_object(original, controls[0]['numerics_sha256'])
    checks = {sha: dict(path_issues=path_issues(value), independent=independent(value, reference))
              for sha, value in values.items()}
    return dict(objects=checks, reference_sha256=controls[0]['numerics_sha256'],
                passed=all(not c['path_issues'] and not c['independent']['issues'] for c in checks.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--study')
    p.add_argument('--output', type=Path)
    p.add_argument('--list', action='store_true')
    p.add_argument('--desktop-preflight', action='store_true')
    args = p.parse_args()
    original = args.original.resolve()
    manifest = json.loads((original/'manifest.json').read_text())
    family = {'blaze2d/support-study/1': 'D05', 'blaze2d/pt-sector-study/1': 'PT',
              'blaze2d/support-dispatch/1': 'dispatch', 'blaze2d/memory-admission/1': 'memory'}[manifest['schema']]
    available = designs(manifest, family)
    if args.list:
        print(json.dumps(available, indent=2)); return 0
    if args.study not in available or args.output is None:
        p.error('Select one listed study and a new output directory')
    design = available[args.study]
    all_rows = [json.loads(line) for line in (original/'records.jsonl').read_text().splitlines()]
    chosen = select(all_rows, args.study, design, family)
    grouping(chosen, design, family)
    # A fresh immutable artifact check replaces the old runner's HEAD-equality
    # requirement. The implementation revision remains the archived build's.
    artifacts = manifest['builds']['artifacts']
    declared = {str(Path(a['path']).resolve()): a['sha256'] for a in artifacts.values()}
    additional_sources = {}
    if family == 'dispatch':
        parent = (original/manifest['parent']).resolve()
        for filename, key in [('manifest.json', 'parent_manifest_sha256'), ('records.jsonl', 'parent_records_sha256')]:
            path = parent/filename
            if digest(path) != manifest[key]:
                raise ValueError('Frozen dispatch parent evidence changed')
            additional_sources[str(path)] = manifest[key]
        parent_manifest = json.loads((parent/'manifest.json').read_text())
        declared.update({str(Path(a['path']).resolve()): a['sha256']
                         for a in parent_manifest['builds']['artifacts'].values()})
    if family == 'memory':
        if not manifest['builds'].get('complete') or len({a['target_directory'] for a in artifacts.values()}) != 4:
            raise ValueError('Missing isolated memory-admission builds')
        for kind in ('unprofiled', 'profiled'):
            if artifacts['baseline_'+kind]['sha256'] == artifacts['candidate_'+kind]['sha256']:
                raise ValueError('Identical parent and candidate artifacts are not a refactor comparison')
    used = {}
    for row in chosen:
        cmd = row['command']
        if cmd[:3] != ['taskset', '-c', str(manifest['cpu'])]:
            raise ValueError('Archived affinity command differs from declared CPU')
        binary = str(Path(cmd[3]).resolve())
        if binary not in declared:
            raise ValueError('Command is not one of the frozen artifacts')
        used[binary] = declared[binary]
    if any(digest(path) != sha for path, sha in used.items()):
        raise ValueError('Frozen executable changed')
    source_hashes = {name: digest(original/name) for name in ('manifest.json', 'records.jsonl', 'audit.json')}
    archived_ok = archived_physical_gate(json.loads((original/'audit.json').read_text()), args.study, design, family)
    for sha in {r['numerics_sha256'] for r in chosen}:
        load_object(original, sha)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'objects').mkdir(); (out/'failures').mkdir()
    contract_files = ['replay_record_study.py', 'replay_quiet.py', 'quiet_host.py', 'host_clock.ps1',
                      'quiet-host-protocol.md', 'record-replay-protocol.md', 'support_contract.py',
                      'pt_sector_contract.py', 'run_pt_sectors.py', 'run_pairs.py', 'check_memory_admission.py']
    hashes = {name: digest(HERE/name) for name in contract_files}
    if args.desktop_preflight:
        hashes['qualitative-protocol.md'] = digest(HERE/'qualitative-protocol.md')
    metadata = dict(schema='blaze2d/record-replay/1', measurement_protocol=MEASUREMENT_PROTOCOL,
                    original=str(original), original_files=source_hashes, original_manifest=manifest,
                    study=args.study, design=design, selected_records=len(chosen),
                    harness_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    harness_sha256=hashes, artifacts=used, timestamp=time.time(),
                    additional_sources=additional_sources,
                    verification_packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
                    archived_physical_qualified=archived_ok,
                    clock_projection='Internal adjusted timer multiplied by enclosing RAW/MONOTONIC ratio; maximum scale spread 0.1%',
                    numerical_gate='Exact archived complete numerical snapshots plus current independent physical/path checks')
    metadata['preflight_limits'] = dict(linux=1.0 if args.desktop_preflight else .25,
                                       windows=3.0 if args.desktop_preflight else 1.5)
    (out/'manifest.json').write_bytes(canonical(metadata)+b'\n')
    env = dict(os.environ, **manifest['thread_environment'])
    os.environ.update(manifest['thread_environment'])
    collected, values = [], {}
    with QuietHost() as host:
        admission = [host.interval(limit=metadata['preflight_limits']['linux'],
                                   windows_limit=metadata['preflight_limits']['windows']) for _ in range(2)]
        (out/'admission.json').write_bytes(canonical(admission)+b'\n')
        if not all(g['accepted'] for g in admission):
            audit = dict(environment_qualified=False, primary_latency_qualified=False, phase='preflight')
            (out/'audit.json').write_bytes(canonical(audit)+b'\n'); return 3
        for index, old in enumerate(chosen):
            row = {k: old[k] for k in ('command', 'label', 'phase')}
            row.update(pair=old.get('pair', old.get('index')), index=old.get('index', old.get('pair')),
                       mode=old.get('mode'), issues=[], timestamp=time.time())
            start = host.snapshot(); process = None
            try:
                process = subprocess.run(old['command'], cwd=ROOT, env=env, capture_output=True, timeout=600)
            except subprocess.TimeoutExpired as error:
                row['issues'].append('Benchmark timeout')
                (out/'failures'/f'{index:04d}.txt.gz').write_bytes(gzip.compress(
                    (error.stdout or b'')+b'\nSTDERR\n'+(error.stderr or b''), mtime=0))
            finally:
                end = host.snapshot()
            row['window'] = dict(start=start, end=end)
            try:
                clocks = elapsed_clocks(start, end)
                row['clock_projection'] = clocks
                if process is not None:
                    row.update(returncode=process.returncode, stderr=process.stderr.decode(errors='replace'))
                    if process.returncode:
                        raise ValueError('Benchmark process failed')
                    nvalue, measurement, issues = verify_result(json.loads(process.stdout), old, family)
                    sha = payload_sha(nvalue)
                    if sha not in values:
                        (out/'objects'/f'{sha}.json.gz').write_bytes(gzip.compress(canonical(nvalue), mtime=0))
                        values[sha] = nvalue
                    row.update(numerics_sha256=sha, measurement=measurement,
                               normalized_seconds=measurement['seconds']*clocks['adjusted_to_raw_scale'])
                    row['issues'] += issues
            except (ValueError, KeyError, TypeError) as error:
                row['issues'].append(str(error))
                if process is not None:
                    (out/'failures'/f'{index:04d}.txt.gz').write_bytes(gzip.compress(
                        process.stdout+b'\nSTDERR\n'+process.stderr, mtime=0))
            collected.append(row)
            with (out/'records.jsonl').open('ab') as stream:
                stream.write(canonical(row)+b'\n')
            if row['issues']:
                break  # Changed numerical output is investigated, never retried for success.
    blocks = {}
    for row in collected:
        blocks.setdefault(row['pair'], []).append(row)
    environment = [dict(block=block, **assess(rows[0]['window']['start'], rows[-1]['window']['end']))
                   for block, rows in blocks.items()]
    scales = [r['clock_projection']['adjusted_to_raw_scale'] for r in collected if 'clock_projection' in r]
    spread = max(scales)/min(scales)-1 if scales else math.inf
    unchanged = (all(digest(path) == sha for path, sha in used.items()) and
                 all(digest(original/name) == sha for name, sha in source_hashes.items()) and
                 all(digest(path) == sha for path, sha in additional_sources.items()) and
                 all(digest(HERE/name) == sha for name, sha in hashes.items()))
    complete = len(collected) == len(chosen)
    exact = complete and all(not r['issues'] for r in collected)
    try:
        checks = current_physical_checks(original, all_rows, chosen, values, design, family) if exact else {'passed': False}
    except Exception as error:
        checks = dict(passed=False, error='Independent physical audit failed: ' + str(error))
    env_ok = all(w['accepted'] for w in environment) and spread <= .001
    numerical_ok = exact and checks['passed'] and unchanged and archived_ok
    accepted = numerical_ok and env_ok
    audit = dict(schema='blaze2d/record-replay-audit/1', measurement_protocol=MEASUREMENT_PROTOCOL,
                 complete=complete, expected_records=len(chosen), records=len(collected),
                 artifacts_and_contracts_unchanged=unchanged, numerical_qualified=numerical_ok,
                 archived_physical_qualified=archived_ok,
                 environment_qualified=env_ok, clock_qualified=spread <= .001,
                 scale_relative_spread=spread if math.isfinite(spread) else None,
                 environment_blocks=environment, physical_checks=checks,
                 records_sha256=digest(out/'records.jsonl'),
                 primary_latency_qualified=accepted, summary=summarize(collected, design, family) if accepted else {},
                 scope='Monitored low-load desktop; resources are not exclusively reserved')
    (out/'audit.json').write_bytes(canonical(audit)+b'\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ('physical_checks', 'environment_blocks')}), flush=True)
    return 4 if not exact or not unchanged else 0 if env_ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
