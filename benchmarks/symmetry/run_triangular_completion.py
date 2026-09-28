#!/usr/bin/env python3
"""Focused continuation of D04; convergence diagnostics, not speedup inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from quiet_host import QuietHost, assess


def design(edge=False):
    if edge: return [(point,129,'TE',64,'voronoi',False) for point in ['hex-gamma','hex-k','hex-generic']]
    cases = []
    for point in ['hex-gamma', 'hex-k', 'hex-generic']:
        for size, cutoff in [(33,7),(33,14),(65,14),(65,28),(129,28),(129,56)]:
            cases.append((point,size,'TE',cutoff,'voronoi',False))
    for point in ['hex-gamma', 'hex-k']:
        for size, cutoff in [(33,14),(65,28)]:
            cases.append((point,size,'TM',cutoff,'voronoi',False))
        for material in ['analytic','raw']:
            cases.append((point,33,'TE',14,material,False))
    for material in ['analytic','raw','voronoi']:
        for pol in ['TM','TE']:
            cases.append(('hex-gamma',17,pol,3.5,material,True))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--edge-refinement',action='store_true')
    p.add_argument('--tight-reference',action='store_true')
    p.add_argument('--repair-reference',action='store_true')
    args = p.parse_args()
    args.binary = args.binary.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    cases = [('hex-generic',129,'TE',r,'voronoi',False) for r in [28,56,64]] if args.tight_reference or args.repair_reference else design(args.edge_refinement)
    tolerance = '1e-16' if args.tight_reference else '1e-14'
    manifest = dict(binary=str(args.binary),sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        provenance=json.loads(args.binary.with_suffix('.provenance.json').read_text()),
        design=cases,tolerance=tolerance,purpose='numerical convergence; descriptive time only')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'harness.py').write_bytes(Path(__file__).read_bytes())
    records = []
    with QuietHost() as host:
        checks = [host.interval(3,limit=2.0,windows_limit=4.0)]
        if not checks[-1]['accepted']:
            checks.extend(host.interval(3,limit=2.0,windows_limit=4.0) for _ in range(2))
        (args.output/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not any(c['accepted'] for c in checks):
            raise SystemExit('Competing load; no study started')
        for i,(case,size,pol,cutoff,material,probe) in enumerate(cases):
            cmd = [str(args.binary),case,str(size),'8',pol,'iterative',str(cutoff),material,
                   'probe' if probe else 'no-probe',tolerance,'full-reference','repair-reference' if args.repair_reference else 'no-repair']
            before = host.snapshot()
            start = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
            run = subprocess.run(cmd,capture_output=True,timeout=900)
            elapsed = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)-start
            after = host.snapshot()
            row = dict(command=cmd,returncode=run.returncode,stderr=run.stderr.decode(),
                process_seconds=elapsed,host=assess(before,after,limit=2.0,windows_limit=4.0),start=before,end=after)
            try: row['result'] = json.loads(run.stdout)
            except ValueError: row['stdout'] = run.stdout.decode(errors='replace')
            (args.output/f'{i:02d}-{case}-{pol}-{size}-{cutoff}-{material}.json').write_text(json.dumps(row,indent=2)+'\n')
            records.append(row)
            result = row.get('result',{})
            print(i,case,pol,size,cutoff,material,'seconds=',round(elapsed,2),
                  'returncode=',run.returncode,flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(records=len(records),binary_unchanged=hashlib.sha256(args.binary.read_bytes()).hexdigest()==manifest['sha256']),indent=2)+'\n')

if __name__ == '__main__': main()
