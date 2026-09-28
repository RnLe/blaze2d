#!/usr/bin/env python3
"""Summarize complete public runs without pooling distinct comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(directory):
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    blocks = json.loads((directory/'blocks.json').read_text())[1:]
    admitted = [b for b in blocks if b['valid'] and b['admitted']]
    rows = admitted or [b for b in blocks if b['valid']]
    variants = {}
    for name in ['reference','candidate']:
        if not rows: continue
        variants[name] = dict(
            median_seconds=statistics.median(b[name]['elapsed_seconds'] for b in rows),
            median_rss_kib=statistics.median(b[name]['resources']['rss_kib'] for b in rows),
            work=rows[0][name]['validation']['work'],
            work_repeatable=all(b[name]['validation']['work']==rows[0][name]['validation']['work'] for b in rows),
        )
    return dict(name=directory.name,directory=str(directory),
        comparison={k:manifest.get(k) for k in ['lattice','polarization','size','bands','intervals','threads','policy','control','fields','smoothing','support_cutoff','heldout']},
        executables=manifest.get('executables'),collector_source_revision=manifest.get('collector_source_revision'),
        summary=summary,variants=variants,
        manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
        max_frequency_difference=max((b.get('max_frequency_difference',0) for b in blocks),default=None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',action='append',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profiles',type=Path)
    args=p.parse_args()
    records=[summarize(f.parent) for root in args.root for f in sorted(root.glob('*/summary.json'))]
    if args.profiles:
        for r in records:
            for name,variant in r['variants'].items():
                prefix=args.profiles/f"{r['name']}-{name}"
                measurement=prefix.with_suffix('.measurement.json')
                if not measurement.exists(): continue
                m=json.loads(measurement.read_text())
                if Path(m['comparison']).resolve()!=Path(r['directory']).resolve(): continue
                assert m['accepted'] and m['complete_iteration_total']
                profile=prefix.with_suffix('.profile.json')
                data=json.loads(profile.read_text())
                variant['profile']=dict(source=str(profile),sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                    counters=data['counters'],arrays_identical=m['arrays_identical'],
                    iteration_ledger_agrees=m['complete_iteration_total'],
                    role='instrumented work counters, not primary latency')
    args.output.write_text(json.dumps(records,indent=2)+'\n')
    for r in records:
        a=r['summary']['admitted']
        print(r['name'], 'quality',r['summary']['numerical_pass'],
              'ratio',None if not a else round(a['ratio'],3),
              'admitted',0 if not a else a['pairs'],'timing',r['summary'].get('timing_qualified'),
              'work',[(n,v['work']['iterations']) for n,v in r['variants'].items()])

if __name__=='__main__':main()
