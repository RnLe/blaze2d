#!/usr/bin/env python3
"""Retrospective UTC-clock diagnostic; cannot admit historical timings as RAW/QPC evidence."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from replay_quiet import digest


def diagnostic(window):
    a,b = window['start'], window['end']
    utc = lambda s: datetime.fromisoformat(s['windows']['utc'].replace('Z','+00:00'))
    host = (utc(b)-utc(a)).total_seconds()
    guest = b['monotonic']-a['monotonic']
    own = b['owned_cpu_seconds']-a['owned_cpu_seconds']
    collector = b['windows']['collector_cpu_seconds']-a['windows']['collector_cpu_seconds']
    background = (b['windows']['busy_seconds']-a['windows']['busy_seconds']-own-collector)/host
    return dict(utc_seconds=host, adjusted_seconds=guest, estimated_scale=host/guest,
                estimated_windows_background_cores=background)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('roots',type=Path,nargs='+')
    args=p.parse_args();rows=[]
    for root in args.roots:
        for path in sorted(root.glob('*/audit.json')):
            old=json.loads(path.read_text())
            if not old.get('environment_qualified'):
                continue
            parent=path.parent
            sample_path=parent/'samples.json';env_path=parent/'environment.jsonl'
            study=json.loads(sample_path.read_text())
            windows=[json.loads(s) for s in env_path.read_text().splitlines()]
            if len(study['samples']) != len(windows):
                raise ValueError('Incomplete process coverage')
            blocks={};times={};scales=[]
            for sample,window in zip(study['samples'],windows):
                key=sample.get('pair',sample.get('block'))
                d=diagnostic(window);scales.append(d['estimated_scale'])
                blocks.setdefault(key,[]).append(window)
                if key is not None and key>=0 and sample.get('valid'):
                    label=sample.get('label',sample.get('mode'))
                    times.setdefault(key,{})[label]=sample['result']['seconds']*d['estimated_scale']
            block_diag=[dict(block=key,**diagnostic({'start':w[0]['start'],'end':w[-1]['end']}))
                        for key,w in blocks.items()]
            preflight=[diagnostic(w) for w in json.loads((parent/'admission.json').read_text())]
            logs={}
            for t in times.values():
                pairs=([('paired','baseline','candidate')] if 'baseline' in t else
                       [('pt','off','pt'),('reuse','off','unitary'),('combined','off','pt-unitary')])
                for name,a,b in pairs:
                    if a in t and b in t:logs.setdefault(name,[]).append(math.log(t[a]/t[b]))
            rows.append(dict(study=str(parent),
                input_sha256={p.name:digest(p) for p in [path,sample_path,env_path,parent/'admission.json']},
                old_primary_latency_qualified=old.get('primary_latency_qualified'),
                numerical_qualified=old.get('numerical_qualified'),
                process_scale_range=[min(scales),max(scales)],
                estimated_ratios={k:math.exp(statistics.mean(v)) for k,v in logs.items()},
                windows_blocks_above_limit=[r for r in block_diag if r['estimated_windows_background_cores']>1.5],
                preflight_above_limit=[r for r in preflight if r['estimated_windows_background_cores']>1.5]))
    result=dict(schema='blaze2d/retrospective-clock-diagnostic/1',
        qualification='Diagnostic only; UTC timestamps are not independent monotonic RAW/QPC evidence',
        scope='Previously environmentally admitted replacement studies; all original files remain unchanged',
        studies=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(json.dumps({k:v for k,v in r.items() if k!='input_sha256'}))


if __name__=='__main__':main()
