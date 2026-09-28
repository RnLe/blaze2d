#!/usr/bin/env python3
"""Bounded, resumable record replay queue; every attempt remains evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from quiet_host import MEASUREMENT_PROTOCOL
from replay_quiet import digest
from replay_record_study import HERE, ROOT, designs


def prior_attempts(study, roots):
    pattern = re.compile(re.escape(study['name']) + r'(?:-attempt-[1-9][0-9]*)?\Z')
    seen, attempts = set(), []
    for root in roots:
        for directory in Path(root).glob(study['name']+'*'):
            directory = directory.resolve()
            if not directory.is_dir() or not pattern.fullmatch(directory.name) or directory in seen:
                continue
            seen.add(directory)
            if not all((directory/name).is_file() for name in ('manifest.json', 'audit.json')):
                raise ValueError('Incomplete attempt needs investigation: ' + str(directory))
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['study'] != study['study'] or manifest['original_files'] != study['original_files']:
                raise ValueError('An existing attempt has different frozen inputs')
            if manifest['measurement_protocol'] != MEASUREMENT_PROTOCOL:
                raise ValueError('Prior attempt used superseded clock accounting')
            audit = json.loads((directory/'audit.json').read_text())
            if audit.get('phase') != 'preflight':
                if not audit['complete'] or not audit['artifacts_and_contracts_unchanged']:
                    raise ValueError('Incomplete/changed record replay needs investigation')
                if digest(directory/'records.jsonl') != audit['records_sha256']:
                    raise ValueError('Prior raw records changed')
                with (directory/'records.jsonl').open() as stream:
                    if any(json.loads(line)['issues'] for line in stream):
                        raise ValueError('Prior output mismatch needs investigation')
            if type(audit.get('environment_qualified')) is not bool:
                raise ValueError('Missing prior environmental decision')
            attempts.append((manifest['timestamp'], directory, audit))
    attempts.sort(key=lambda row: (row[0], str(row[1])))
    for _, directory, audit in attempts:
        if audit['environment_qualified']:
            return dict(study=study['name'], referenced_attempt=str(directory),
                        audit_sha256=digest(directory/'audit.json'),
                        numerical_qualified=audit['numerical_qualified'],
                        primary_latency_qualified=audit['primary_latency_qualified']), len(attempts)
    return None, len(attempts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--completed-root', type=Path, action='append', default=[])
    p.add_argument('--only', nargs='+')
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    files = ['rerun_record_studies.py', 'replay_record_study.py', 'replay_quiet.py',
             'quiet_host.py', 'host_clock.ps1', 'quiet-host-protocol.md', 'record-replay-protocol.md',
             'support_contract.py', 'pt_sector_contract.py', 'run_pt_sectors.py', 'run_pairs.py', 'check_memory_admission.py']
    frozen = {name: digest(HERE/name) for name in files}
    inventory_path = out/'inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        if inventory['harness_sha256'] != frozen or inventory['measurement_protocol'] != MEASUREMENT_PROTOCOL:
            raise ValueError('Queue contracts changed; create a new inventory and reference all prior attempts')
    else:
        studies = []
        for directory, family in [('pt-sector-measurements', 'PT'), ('D05', 'D05'),
                                   ('dispatch', 'dispatch'), ('memory-admission-verified', 'memory')]:
            source = HERE/directory
            manifest = json.loads((source/'manifest.json').read_text())
            hashes = {name: digest(source/name) for name in ('manifest.json', 'records.jsonl', 'audit.json')}
            for name, design in designs(manifest, family).items():
                studies.append(dict(name=family+'-'+name, study=name, original=str(source),
                                    original_files=hashes, design=design))
        inventory = dict(schema='blaze2d/record-replay-inventory/1', measurement_protocol=MEASUREMENT_PROTOCOL,
                         timestamp=time.time(), studies=studies, harness_sha256=frozen,
                         environment_attempt_limit=3, maximum_consecutive_exhausted_cases=3,
                         retry_settling_seconds=10, completed_roots=[str(r.resolve()) for r in args.completed_root])
        inventory_path.write_text(json.dumps(inventory, indent=2)+'\n')
    if args.only and not set(args.only) <= {s['name'] for s in inventory['studies']}:
        p.error('Unknown study selection')
    def record(name, value):
        with (out/name).open('a') as stream:
            stream.write(json.dumps(value)+'\n')
    record('invocations.jsonl', dict(timestamp=time.time(), arguments=sys.argv[1:], only=args.only))
    consecutive = 0
    for study in inventory['studies']:
        if args.only and study['name'] not in args.only:
            continue
        if (out/'STOP').exists():
            print('Stopped at requested study boundary', flush=True); return 0
        if any(digest(Path(study['original'])/name) != sha for name, sha in study['original_files'].items()):
            raise ValueError('Original evidence changed')
        prior, count = prior_attempts(study, inventory['completed_roots']+[out])
        if prior:
            record('references.jsonl', prior); print(json.dumps(prior), flush=True); continue
        if count >= inventory['environment_attempt_limit']:
            record('references.jsonl', dict(study=study['name'], phase='previous_environment_allowance_exhausted', attempts=count))
            continue
        for attempt in range(count+1, inventory['environment_attempt_limit']+1):
            if attempt > 1:
                time.sleep(inventory['retry_settling_seconds'])
            destination = out/(study['name'] if attempt == 1 else f"{study['name']}-attempt-{attempt}")
            if destination.exists():
                raise ValueError('Refusing to overwrite an existing attempt')
            print(f"Starting {study['name']}, attempt {attempt}", flush=True)
            command = [sys.executable, str(HERE/'replay_record_study.py'), '--original', study['original'],
                       '--study', study['study'], '--output', str(destination)]
            with (out/(destination.name+'.log')).open('w') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            audit = json.loads((destination/'audit.json').read_text()) if (destination/'audit.json').exists() else {}
            state = dict(study=study['name'], attempt=attempt, returncode=process.returncode, timestamp=time.time(),
                         environment_qualified=audit.get('environment_qualified'),
                         numerical_qualified=audit.get('numerical_qualified'),
                         primary_latency_qualified=audit.get('primary_latency_qualified'))
            record('progress.jsonl', state); print(json.dumps(state), flush=True)
            if process.returncode not in (0, 3):
                return process.returncode or 4
            if audit.get('environment_qualified'):
                consecutive = 0; break
        else:
            consecutive += 1
            record('progress.jsonl', dict(study=study['name'], phase='environment_allowance_exhausted',
                                         consecutive_exhausted_cases=consecutive))
            if consecutive >= inventory['maximum_consecutive_exhausted_cases']:
                return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
