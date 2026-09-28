#!/usr/bin/env python3
"""Continue the complete comparison inventory under the explicit desktop amendment."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from qualitative_timing import analyze
from quiet_host import MEASUREMENT_PROTOCOL
from replay_quiet import HERE, ROOT, digest, legacy_command
from replay_record_study import designs


def inventory(workspace):
    studies = []
    for source in sorted(HERE.glob('*.json')):
        value = json.loads(source.read_text())
        if not isinstance(value, dict) or 'samples' not in value:
            continue
        name = source.stem
        legacy_command(value['manifest'], workspace, Path('/unused/samples.json'))
        profiled = any(s.get('result', {}).get('profile') or s.get('result', {}).get('timings')
                       for s in value['samples'])
        priority = 0 if name.startswith(('R-heldout', 'R-combined')) else 1 if name.startswith('L0') else 2
        studies.append(dict(name=name, family='root', original=str(source),
                            source_sha256={source.name: digest(source)}, priority=priority,
                            instrumented=bool(profiled)))
    for directory, family in [('pt-sector-measurements', 'PT'), ('D05', 'D05'),
                               ('dispatch', 'dispatch'), ('memory-admission-verified', 'memory')]:
        source = HERE/directory
        manifest = json.loads((source/'manifest.json').read_text())
        hashes = {name: digest(source/name) for name in ('manifest.json', 'records.jsonl', 'audit.json')}
        for name in designs(manifest, family):
            studies.append(dict(name=family+'-'+name, study=name, family='record', original=str(source),
                                source_sha256=hashes, priority=3, instrumented=False))
    for source in sorted((HERE/'coarse-workflow-reference/pilots').glob('*.json')):
        studies.append(dict(name='public-coarse-'+source.stem, family='coarse', original=str(source),
                            source_sha256={source.name: digest(source)}, priority=-1, instrumented=False))
    return sorted(studies, key=lambda s: (s['instrumented'], s['priority'], s['name']))


def inputs_match(study, manifest):
    if study['family'] == 'record':
        return (manifest.get('study') == study['study'] and
                manifest.get('original_files') == study['source_sha256'])
    return manifest.get('original_sha256') == next(iter(study['source_sha256'].values()))


def attempts(study, roots):
    seen, found = set(), []
    pattern = re.compile(re.escape(study['name']) + r'(?:-attempt-[1-9][0-9]*)?\Z')
    for root in roots:
        for directory in Path(root).glob(study['name']+'*'):
            directory = directory.resolve()
            if directory in seen or not directory.is_dir() or not pattern.fullmatch(directory.name):
                continue
            seen.add(directory)
            if not all((directory/f).is_file() for f in ('manifest.json', 'audit.json')):
                raise ValueError('Incomplete prior attempt: '+str(directory))
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest.get('measurement_protocol') != MEASUREMENT_PROTOCOL:
                raise ValueError('Prior attempt has superseded clock accounting: '+str(directory))
            if not inputs_match(study, manifest):
                raise ValueError('Prior attempt uses different frozen inputs: '+str(directory))
            audit = json.loads((directory/'audit.json').read_text())
            if audit.get('numerical_signature_mismatches'):
                raise ValueError('Numerical output changed: '+str(directory))
            if study['family'] == 'record' and audit.get('phase') != 'preflight':
                if (not audit.get('complete') or not audit.get('artifacts_and_contracts_unchanged') or
                        digest(directory/'records.jsonl') != audit.get('records_sha256') or
                        any(json.loads(line)['issues'] for line in (directory/'records.jsonl').read_text().splitlines())):
                    raise ValueError('Incomplete or changed numerical records: '+str(directory))
            found.append((manifest['timestamp'], directory, audit))
    return sorted(found, key=lambda r: (r[0], str(r[1])))


def new_allowance_used(prior):
    return any(json.loads((directory/'manifest.json').read_text()).get('preflight_limits') ==
               dict(linux=1.0, windows=3.0) for _, directory, _ in prior)


def choose(study, prior, analysis_root):
    # Preserve previously published strict admission first; never replace it
    # with whichever secondary analysis has the most attractive timing.
    for _, directory, audit in prior:
        if audit.get('environment_qualified') and audit.get('numerical_qualified') is not None:
            state = ('primary qualified' if audit.get('primary_latency_qualified') else
                     'instrumented diagnostic' if audit.get('numerical_qualified') else 'numerical failure')
            return dict(state=state, evidence=str(directory), audit_sha256=digest(directory/'audit.json'))
    for _, directory, audit in prior:
        if audit.get('phase') != 'preflight' and audit.get('numerical_qualified') is False:
            return dict(state='numerical failure', evidence=str(directory), audit_sha256=digest(directory/'audit.json'))
        if not audit.get('complete') or not audit.get('clock_qualified'):
            continue
        analysis = analyze(directory)
        path = analysis_root/(directory.parent.name+'--'+directory.name+'.json')
        path.write_text(json.dumps(analysis, indent=2)+'\n')
        if analysis['qualitative_timing_qualified'] or (analysis.get('instrumented') and 'filtered' in analysis):
            return dict(state='qualitative qualified' if analysis['qualitative_timing_qualified'] else 'instrumented diagnostic',
                        evidence=str(directory), analysis=str(path), analysis_sha256=digest(path),
                        retained_blocks=len(analysis['retained_blocks']),
                        contrasts=analysis['filtered']['contrasts'], decisions=analysis['decisions'])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--completed-root', type=Path, action='append', default=[])
    p.add_argument('--only', nargs='+')
    p.add_argument('--analyze-only', action='store_true')
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(exist_ok=True, parents=True)
    analysis_root = out/'analyses'; analysis_root.mkdir(exist_ok=True)
    # Freeze the complete Python benchmark harness as well as both host policies.
    files = sorted(HERE.glob('*.py'))+[HERE/name for name in
        ['qualitative-protocol.md', 'quiet-host-protocol.md', 'host_clock.ps1',
         'record-replay-protocol.md', 'coarse-workflow-protocol.md']]
    hashes = {str(path): digest(path) for path in files}
    path = out/'inventory.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['harness_sha256'] != hashes:
            raise ValueError('Harness changed; freeze a new inventory and reference earlier attempts')
    else:
        manifest = dict(schema='blaze2d/qualitative-inventory/1', timestamp=time.time(),
                        harness_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        harness_sha256=hashes, studies=inventory(args.workspace),
                        completed_roots=[str(r.resolve()) for r in args.completed_root],
                        new_attempt_allowance=1, minimum_retained_blocks=12,
                        policy='Explicit user resumption; previous strict attempts remain in the ledger')
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.only and not set(args.only) <= {s['name'] for s in manifest['studies']}:
        p.error('Unknown study selection')
    def record(filename, value):
        with (out/filename).open('a') as stream:
            stream.write(json.dumps(value)+'\n')
    record('invocations.jsonl', dict(timestamp=time.time(), arguments=sys.argv[1:]))
    coverage = []
    for study in manifest['studies']:
        if args.only and study['name'] not in args.only:
            continue
        if (out/'STOP').exists():
            print('Stopped at study boundary', flush=True); break
        original = Path(study['original'])
        if any(digest((original if original.is_file() else original/name)) != sha
               for name, sha in study['source_sha256'].items()):
            raise ValueError('Original evidence changed: '+study['name'])
        prior = attempts(study, manifest['completed_roots']+[out])
        selected = choose(study, prior, analysis_root)
        if not selected and not args.analyze_only and not new_allowance_used(prior) and not (out/study['name']).exists():
            destination = out/study['name']
            script = 'replay_record_study.py' if study['family'] == 'record' else 'replay_quiet.py'
            command = [sys.executable, str(HERE/script), '--original', study['original'],
                       '--output', str(destination), '--desktop-preflight']
            command += ['--study', study['study']] if study['family'] == 'record' else ['--workspace', str(args.workspace)]
            print('Starting '+study['name'], flush=True)
            with (out/(study['name']+'.log')).open('w') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record('executions.jsonl', dict(study=study['name'], returncode=process.returncode,
                                           timestamp=time.time(), earlier_attempts=len(prior)))
            if process.returncode not in (0, 3):
                print('Investigation required: '+study['name'], flush=True); return process.returncode or 4
            prior = attempts(study, manifest['completed_roots']+[out])
            selected = choose(study, prior, analysis_root)
        row = dict(study=study['name'], attempts=len(prior), **(selected or dict(
            state='timing unresolved' if prior else 'unattempted')))
        record('progress.jsonl', row)
        coverage.append(row)
        (out/'coverage.json').write_text(json.dumps(dict(inventory_count=len(manifest['studies']),
                                                        studies=coverage), indent=2)+'\n')
        print(json.dumps(row), flush=True)
    if any(digest(path) != sha for path, sha in hashes.items()):
        raise ValueError('Harness changed during collection')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
