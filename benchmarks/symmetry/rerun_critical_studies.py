#!/usr/bin/env python3
"""Serial, restartable inventory of every archived root comparison, without selection by result."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from replay_quiet import HERE, ROOT, digest, legacy_command
from quiet_host import MEASUREMENT_PROTOCOL


def previous_attempts(study, roots):
    """Count every frozen attempt, including suffixed retries, exactly once.

    Reuse the first environmentally admitted result, including a reproduced
    numerical failure. Neither its speed nor its numerical success selects it.
    """
    name = study['name']
    pattern = re.compile(re.escape(name) + r'(?:-attempt-[1-9][0-9]*)?\Z')
    seen, attempts = set(), []
    for root in roots:
        for candidate in Path(root).glob(name + '*'):
            if not candidate.is_dir() or not pattern.fullmatch(candidate.name):
                continue
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if not (candidate/'audit.json').is_file() or not (candidate/'manifest.json').is_file():
                raise ValueError('Incomplete prior attempt requires investigation: ' + str(candidate))
            manifest = json.loads((candidate/'manifest.json').read_text())
            if manifest['original_sha256'] != study['original_sha256']:
                continue
            if manifest.get('measurement_protocol') != MEASUREMENT_PROTOCOL:
                raise ValueError('Prior attempt used superseded clock accounting: ' + str(candidate))
            audit = json.loads((candidate/'audit.json').read_text())
            if audit.get('numerical_signature_mismatches'):
                raise ValueError('Numerical signature changed: ' + str(candidate))
            if audit.get('environment_qualified') not in (True, False):
                raise ValueError('Missing prior admission decision: ' + str(candidate))
            attempts.append((manifest['timestamp'], str(candidate), audit))
    attempts.sort(key=lambda item: (item[0], item[1]))
    for _, path, audit in attempts:
        if audit['environment_qualified']:
            return dict(study=name, referenced_attempt=path,
                        audit_sha256=digest(Path(path)/'audit.json'),
                        numerical_qualified=audit.get('numerical_qualified'),
                        timing_qualified=audit.get('timing_qualified'),
                        primary_latency_qualified=audit.get('primary_latency_qualified')), 0
    return None, len(attempts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', nargs='*', help='Explicit subset of frozen names; inventory still includes all studies')
    p.add_argument('--completed-root', type=Path, action='append', default=[], help='Prior admitted evidence to reference by input hash')
    p.add_argument('--continue-after-environment-exhaustion', action='store_true',
                   help='Continue independent cases; stop after three consecutive exhausted cases')
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'inventory.json'
    if not manifest_path.exists():
        studies = []
        for source in sorted(HERE.glob('*.json')):
            value = json.loads(source.read_text())
            if not isinstance(value, dict) or 'samples' not in value:
                continue
            name = source.stem
            harness, argv = legacy_command(value['manifest'], args.workspace, out/name/'samples.json')
            priority = (0 if name.startswith('R-heldout') or name.startswith('R-combined') else
                        1 if name.startswith('L08') and 'native' in name else
                        2 if name.startswith('L0') else 3)
            studies.append(dict(name=name, original=str(source), original_sha256=digest(source),
                                legacy_harness=harness, resolved_arguments=argv, priority=priority,
                                original_samples=len(value['samples']),
                                original_instrumented=any(bool(s.get('result',{}).get('profile') or
                                     s.get('result',{}).get('timings')) for s in value['samples'])))
        studies.sort(key=lambda s: (s['original_instrumented'], s['priority'], s['name']))
        manifest = dict(schema='blaze2d/critical-replay-inventory/1', timestamp=time.time(),
                        measurement_protocol=MEASUREMENT_PROTOCOL,
                        harness_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        queue_sha256=digest(__file__), replay_sha256=digest(HERE/'replay_quiet.py'),
                        admission_sha256={n:digest(HERE/n) for n in ['quiet_host.py','host_clock.ps1','quiet-host-protocol.md','normalized_timing.py']},
                        studies=studies, additional_required_studies=['D04 convergence/negative controls',
                        'D05 28 studies', 'dispatch overhead', 'valid memory admission', 'PT-sector interactions'],
                        presentation_work=False, environment_attempt_limit=3,
                        retry_settling_seconds=10, completed_roots=[str(p.resolve()) for p in args.completed_root])
        manifest['continue_after_environment_exhaustion'] = args.continue_after_environment_exhaustion
        manifest['maximum_consecutive_exhausted_cases'] = 3
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        manifest = json.loads(manifest_path.read_text())
        if (manifest['queue_sha256'] != digest(__file__) or
            manifest['replay_sha256'] != digest(HERE/'replay_quiet.py') or
            manifest['admission_sha256'] != {n:digest(HERE/n) for n in manifest['admission_sha256']}):
            raise ValueError('Replay harness changed; create a new inventory')
    names = {s['name'] for s in manifest['studies']}
    if args.only and not set(args.only) <= names:
        raise ValueError('Requested study is not in the complete inventory')
    with (out/'invocations.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(timestamp=time.time(), only=args.only,
                                   arguments=sys.argv[1:]))+'\n')
    exhausted_cases = 0
    for study in manifest['studies']:
        name=study['name']
        if args.only and name not in args.only:
            continue
        if (out/'STOP').exists():
            print('Stopped at requested study boundary', flush=True); return 0
        if digest(study['original']) != study['original_sha256']:
            raise ValueError('Original evidence changed: '+name)
        prior, prior_environment_failures = previous_attempts(study, manifest['completed_roots']+[out])
        if prior:
            with (out/'references.jsonl').open('a') as f:f.write(json.dumps(prior)+'\n')
            print(json.dumps(prior),flush=True)
            continue
        if prior_environment_failures >= manifest['environment_attempt_limit']:
            state = dict(study=name, phase='previous_environment_allowance_exhausted',
                         previous_attempts=prior_environment_failures)
            with (out/'references.jsonl').open('a') as stream:
                stream.write(json.dumps(state)+'\n')
            print(json.dumps(state), flush=True)
            continue
        for attempt in range(1+prior_environment_failures,manifest['environment_attempt_limit']+1):
            destination=out/(name if attempt==1 else f'{name}-attempt-{attempt}')
            if destination.exists():
                audit_path=destination/'audit.json'
                if not audit_path.is_file():
                    raise ValueError('Incomplete existing attempt requires investigation: '+str(destination))
                audit=json.loads(audit_path.read_text())
                if audit.get('numerical_signature_mismatches'):
                    raise ValueError('Numerical signature changed: '+str(destination))
                if audit.get('environment_qualified'):
                    exhausted_cases = 0
                    break
                continue
            if attempt>1:
                time.sleep(manifest['retry_settling_seconds'])
            command=[sys.executable,str(HERE/'replay_quiet.py'),'--original',study['original'],
                     '--workspace',str(args.workspace),'--output',str(destination)]
            print(f'Starting {name}, attempt {attempt}', flush=True)
            with (out/(destination.name+'.log')).open('w') as log:
                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            audit_path=destination/'audit.json'
            audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
            state=dict(study=name,attempt=attempt,returncode=process.returncode,timestamp=time.time(),
                       environment_qualified=audit.get('environment_qualified'),
                       numerical_qualified=audit.get('numerical_qualified'),
                       timing_qualified=audit.get('timing_qualified'),
                       primary_latency_qualified=audit.get('primary_latency_qualified'),
                       numerical_signature_mismatches=audit.get('numerical_signature_mismatches'),
                       practical_decisions=audit.get('practical_decisions'))
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
            print(json.dumps(state),flush=True)
            if process.returncode not in [0,3] or audit.get('numerical_signature_mismatches'):
                return process.returncode or 4
            if audit.get('environment_qualified'):
                exhausted_cases = 0
                break
        else:
            exhausted_cases += 1
            state = dict(study=name, phase='environment_allowance_exhausted',
                         consecutive_exhausted_cases=exhausted_cases)
            with (out/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(state)+'\n')
            print(json.dumps(state), flush=True)
            if (not manifest['continue_after_environment_exhaustion'] or
                    exhausted_cases >= manifest['maximum_consecutive_exhausted_cases']):
                return 3
    return 0


if __name__=='__main__':
    raise SystemExit(main())
