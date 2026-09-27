#!/usr/bin/env python3
"""Replay one archived comparison with unchanged binaries and host admission."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from quiet_host import QuietHost, assess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME_KEYS = {'seconds', 'setup_seconds', 'replay_seconds', 'nanos', 'exclusive_nanos',
                'peak_rss_kib', 'peak_rss_bytes', 'serialized_sha256', 'serialized_bytes'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def numerical(value):
    if isinstance(value, dict):
        return {k: numerical(v) for k, v in value.items()
                if k not in RUNTIME_KEYS and not k.endswith('_seconds')}
    if isinstance(value, list):
        return [numerical(v) for v in value]
    return value


def signature(value):
    return hashlib.sha256(json.dumps(numerical(value), sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve(path, workspace, expected=None):
    candidates = [Path(path)] if Path(path).is_absolute() else [ROOT/path, workspace/path]
    for candidate in candidates:
        if candidate.is_file() and (expected is None or digest(candidate) == expected):
            return str(candidate.resolve())
    raise ValueError('Missing or mismatched frozen artifact: ' + str(path))


def legacy_command(manifest, workspace, output):
    m = manifest
    if 'baseline' in m:
        contract = m.get('comparison_contract')
        if contract not in (None, 'public_coarse'):
            raise ValueError('Unknown declared comparison contract')
        name = 'run_coarse_workflow_pairs' if contract == 'public_coarse' else 'run_pairs'
        argv = [resolve(m[k], workspace, m[k+'_sha256']) for k in ('baseline', 'candidate')]
        fields = ['baseline_mode', 'candidate_mode', 'case', 'grid', 'bands', 'polarization',
                  'pairs', 'cpu', 'cpus', 'tolerance', 'stop_on_quality_failure',
                  'require_sector_success', 'require_requested_path', 'require_job_identity',
                  'require_interface_contract', 'output_contract', 'snapshot']
    elif 'blocks' in m:
        name = 'run_factorial'
        argv = [resolve(m['binary'], workspace, m['binary_sha256'])]
        fields = ['case', 'grid', 'bands', 'polarization', 'tolerance', 'blocks', 'cpu', 'output_contract']
    else:
        name = 'run_trial_pairs' if 'columns' in m else 'run_kernel_pairs'
        argv = [resolve(m['binary'], workspace, m['binary_sha256'])]
        fields = ['grid', 'polarization', 'repetitions', 'pairs']
        fields += ['columns', 'group', 'stage'] if name == 'run_trial_pairs' else ['sector']
    for key in fields:
        value = m.get(key)
        if value is None or value is False:
            continue
        if key == 'snapshot':
            value = resolve(value, workspace, m.get('snapshot_sha256'))
        argv.append('--' + key.replace('_', '-'))
        if value is not True:
            argv.append(str(value))
    return name, argv + ['--output', str(output)]


def practical(ci):
    if not ci or len(ci) != 2 or not all(math.isfinite(v) for v in ci):
        return 'unavailable'
    lo, hi = ci
    return ('win' if lo > 1.02 else 'regression' if hi < 1/1.02 else
            'equivalent' if lo >= 1/1.02 and hi <= 1.02 else 'inconclusive')


def interaction_decision(ci):
    """T11*T00/(T10*T01) is a time factor, not a speedup."""
    decision = practical(ci)
    return {'win': 'above_multiplicative_time_prediction',
            'regression': 'below_multiplicative_time_prediction',
            'equivalent': 'within_two_percent_of_multiplicative_prediction'}.get(decision, decision)


def backend_dependencies(original):
    dependencies = []
    if original.name.startswith('L08-'):
        manifest_path = HERE/'L08-build-manifest.json'
        build = json.loads(manifest_path.read_text())
        study = json.loads((HERE/'L08-study-manifest.json').read_text())
        dependencies.append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
        for name, entry in build['builds'].items():
            if name.startswith('wasm-'):
                modules = list(Path(entry['path']).glob('*_bg.wasm'))
                if len(modules) != 1:
                    raise ValueError('Expected one frozen WASM module')
                module = modules[0]
                if digest(module) != entry['sha256']:
                    raise ValueError('Frozen WASM module changed: '+str(module))
                for p in [module, module.with_name(module.name.replace('_bg.wasm', '.js'))]:
                    dependencies.append(dict(path=str(p), sha256=digest(p)))
        adapter = Path(study['wrappers']["('wasm', False)"]['path']).with_name('wasm_case_l08.mjs')
        if digest(adapter) != study['wasm_adapter_sha256']:
            raise ValueError('Frozen WASM adapter changed')
        dependencies.append(dict(path=str(adapter), sha256=digest(adapter)))
        node = subprocess.check_output(['/usr/bin/node', '--version'], text=True).strip()
        if node != study['node_version']:
            raise ValueError('WASM Node runtime version changed')
        dependencies.append(dict(path='/usr/bin/node', sha256=digest('/usr/bin/node'), version=node))
    return dependencies


def audit_replay(original, current, windows):
    expected = {}
    for r in original['samples']:
        if 'result' in r:
            expected.setdefault(r.get('label', r.get('mode')), set()).add(signature(r['result']))
    mismatches = []
    for index, r in enumerate(current['samples']):
        if 'result' in r and signature(r['result']) not in expected.get(r.get('label', r.get('mode')), set()):
            mismatches.append(index)
    samples = current['samples']
    if len(samples) != len(windows):
        raise ValueError('Host windows do not cover every benchmark process')
    groups = {}
    for sample, window in zip(samples, windows):
        key = sample.get('pair', sample.get('block'))
        groups.setdefault(key, []).append(window)
    environment = []
    for key, records in groups.items():
        a, b = records[0]['start'], records[-1]['end']
        environment.append(dict(block=key, **assess(a, b)))
    env_ok = all(e['accepted'] for e in environment)
    quality_ok = all(r.get('valid') is True for r in samples)
    declared = current['manifest'].get('pairs', current['manifest'].get('blocks'))
    actual = sum(key is not None and key >= 0 for key in groups)
    complete = actual == declared
    instrumented = any(r.get('result', {}).get('profile') or
                       r.get('result', {}).get('timings') for r in samples)
    qualified = env_ok and quality_ok and complete and not mismatches
    summary = current.get('summary', {})
    decisions = ({'paired': practical(summary.get('ci95'))} if 'ci95' in summary else
                 {k: practical(v.get('ci95')) for k, v in summary.items()
                  if k != 'interaction_time_factor' and isinstance(v, dict) and 'ci95' in v})
    interaction = summary.get('interaction_time_factor')
    return dict(environment_qualified=env_ok, numerical_qualified=quality_ok,
                complete=complete, declared_blocks=declared, completed_blocks=actual,
                numerical_signature_mismatches=mismatches, environment_blocks=environment,
                timing_qualified=qualified,
                instrumented=bool(instrumented), primary_latency_qualified=qualified and not instrumented,
                practical_decisions=decisions if qualified else {},
                interaction_decision=interaction_decision(interaction.get('ci95'))
                    if qualified and isinstance(interaction, dict) else None,
                old_summary=original.get('summary'), new_summary=summary,
                scope='monitored low-load desktop; hardware resources are not exclusively reserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.original.read_text())
    legacy_name, argv = legacy_command(original['manifest'], args.workspace, out/'samples.json')
    original_run = subprocess.run
    windows = []
    dependencies = [Path(__file__), HERE/(legacy_name+'.py'),
                    HERE/'quiet_host.py', HERE/'host_clock.ps1', HERE/'quiet-host-protocol.md']
    if legacy_name == 'run_coarse_workflow_pairs':
        dependencies += [HERE/name for name in ['run_pairs.py', 'interface_contract.py',
                          'coarse_workflow_contract.py', 'coarse-workflow-protocol.md']]
    metadata = dict(schema='blaze2d/quiet-replay/1', original=str(args.original.resolve()),
                    original_sha256=digest(args.original), timestamp=time.time(),
                    harness_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    legacy_harness=legacy_name, arguments=argv,
                    harness_sha256={p.name: digest(p) for p in dependencies},
                    original_manifest=original['manifest'], runtime_signature_exclusions=sorted(RUNTIME_KEYS),
                    backend_dependencies=backend_dependencies(args.original),
                    additional_signature_exclusions='keys ending in _seconds')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with QuietHost() as host:
        gates = [host.interval(), host.interval()]
        (out/'admission.json').write_text(json.dumps(gates, indent=2)+'\n')
        if not all(g['accepted'] for g in gates):
            (out/'audit.json').write_text(json.dumps(dict(environment_qualified=False,
                timing_qualified=False, phase='preflight', issues=[i for g in gates for i in g['issues']]), indent=2)+'\n')
            print(json.dumps({'study': args.original.name, 'phase': 'preflight', 'environment_qualified': False}), flush=True)
            return 3

        def measured_run(command, *pos, **kw):
            if not command or command[0] != 'taskset':
                return original_run(command, *pos, **kw)
            start = host.snapshot()
            try:
                return original_run(command, *pos, **kw)
            finally:
                end = host.snapshot()
                record = dict(command=command, start=start, end=end, **assess(start, end))
                windows.append(record)
                with (out/'environment.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')

        try:
            subprocess.run = measured_run
            sys.argv = [legacy_name] + argv
            importlib.import_module(legacy_name).main()
        finally:
            subprocess.run = original_run
    current = json.loads((out/'samples.json').read_text())
    audit = audit_replay(original, current, windows)
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'study': args.original.name, **{k:v for k,v in audit.items()
          if k not in ['old_summary', 'new_summary', 'environment_blocks']}}), flush=True)
    return 0 if audit['environment_qualified'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
