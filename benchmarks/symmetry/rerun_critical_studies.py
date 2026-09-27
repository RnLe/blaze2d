#!/usr/bin/env python3
"""Serial, restartable inventory of every archived root comparison, without selection by result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from replay_quiet import HERE, ROOT, digest, legacy_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', nargs='*', help='Explicit subset of frozen names; inventory still includes all studies')
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
                        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        queue_sha256=digest(__file__), replay_sha256=digest(HERE/'replay_quiet.py'),
                        admission_sha256={n:digest(HERE/n) for n in ['quiet_host.py','host_clock.ps1','quiet-host-protocol.md']},
                        studies=studies, additional_required_studies=['D04 convergence/negative controls',
                        'D05 28 studies', 'dispatch overhead', 'valid memory admission', 'PT-sector interactions'],
                        presentation_work=False)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        manifest = json.loads(manifest_path.read_text())
        if (manifest['replay_sha256'] != digest(HERE/'replay_quiet.py') or
            manifest['admission_sha256'] != {n:digest(HERE/n) for n in manifest['admission_sha256']}):
            raise ValueError('Replay harness changed; create a new inventory')
    names = {s['name'] for s in manifest['studies']}
    if args.only and not set(args.only) <= names:
        raise ValueError('Requested study is not in the complete inventory')
    for study in manifest['studies']:
        name=study['name']; destination=out/name
        if args.only and name not in args.only:
            continue
        if (out/'STOP').exists():
            print('Stopped at requested study boundary', flush=True); return 0
        if digest(study['original']) != study['original_sha256']:
            raise ValueError('Original evidence changed: '+name)
        if destination.exists():
            audit_path=destination/'audit.json'
            if not audit_path.exists():
                raise ValueError('Incomplete existing attempt requires investigation: '+name)
            audit=json.loads(audit_path.read_text())
            if not audit.get('environment_qualified') or audit.get('numerical_signature_mismatches'):
                raise ValueError('Existing unqualified attempt needs a separately identified rerun: '+name)
            continue
        command=[sys.executable,str(HERE/'replay_quiet.py'),'--original',study['original'],
                 '--workspace',str(args.workspace),'--output',str(destination)]
        print('Starting '+name, flush=True)
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        audit_path=destination/'audit.json'
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        state=dict(study=name,returncode=process.returncode,timestamp=time.time(),
                   environment_qualified=audit.get('environment_qualified'),
                   numerical_qualified=audit.get('numerical_qualified'),
                   timing_qualified=audit.get('timing_qualified'),
                   primary_latency_qualified=audit.get('primary_latency_qualified'),
                   numerical_signature_mismatches=audit.get('numerical_signature_mismatches'),
                   practical_decisions=audit.get('practical_decisions'))
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if process.returncode or audit.get('numerical_signature_mismatches'):
            return process.returncode or 4
    return 0


if __name__=='__main__':
    raise SystemExit(main())
