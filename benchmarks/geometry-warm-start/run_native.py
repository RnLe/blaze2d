"""Paired native continuation decision gates on the complete CLI workflow."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/symmetry'))
from quiet_host import QuietHost
from run_public_sweep import execute,estimate


def native_work(path):
    data=json.loads(gzip.decompress(path.read_bytes()))
    result=dict(roots=0,seeded=0,guards=0,discarded_iterations=0,zero_correction=0,max_cache_payload=0,tm_actions_reused=0)
    for job in data.get('results',[]):
        for point in job['metadata'].get('geometry_warm_start',{}).get('points',[]):
            if not point: continue
            result['roots']+=point['cold_root']
            result['seeded']+=point['source'] is not None
            result['guards']+=point['certified_guards']
            result['discarded_iterations']+=point['discarded_iterations']
            result['zero_correction']+=point['initial_acceptance']
            result['max_cache_payload']=max(result['max_cache_payload'],point['insertion']['payload_bytes'])
            result['tm_actions_reused']+=point.get('tm_actions_reused',0)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--filter',default='')
    parser.add_argument('--actions',action='store_true')
    args=parser.parse_args()
    args.reference=args.reference.resolve();args.candidate=args.candidate.resolve()
    rng=random.Random(2026092805)
    for size in [32,64]:
      for pol in ['TM','TE']:
       for control in ['public','guarded']:
        name=f'square-{pol}-{size}-gamma-{control}'
        if args.filter not in name: continue
        out=args.output/name
        if (out/'summary.json').exists(): continue
        out.mkdir(parents=True,exist_ok=False)
        base=(ROOT/f'benchmarks/geometry-warm-start/baseline/public/square-{pol}-gamma-copy/reference.toml').read_text().replace('resolution = 32',f'resolution = {size}')
        for variant in ['reference','candidate']:
            text=base
            if variant=='candidate' or control=='guarded':
                method='copy' if variant=='candidate' else 'cold'
                text+=f'\n[bands.warm_start]\nmethod = "{method}"\nguards = 2\nretain_tm_actions = {str(args.actions).lower()}\n'
            (out/f'{variant}.toml').write_text(text)
        cfg=SimpleNamespace(reference=args.candidate if control=='guarded' else args.reference,
            candidate=args.candidate,output=out,threads=args.threads,fields=False)
        (out/'manifest.json').write_text(json.dumps({
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'binaries':{v:{'path':str(getattr(cfg,v)),'sha256':hashlib.sha256(getattr(cfg,v).read_bytes()).hexdigest()} for v in ['reference','candidate']},
            'inputs':{v:hashlib.sha256((out/f'{v}.toml').read_bytes()).hexdigest() for v in ['reference','candidate']},
            'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'threads':args.threads,'pairs':args.pairs,'warmups':1,'protocol':'benchmarks/geometry-warm-start/plan.md',
            'scope':'public control measures guard and certificate overhead; guarded control holds explicit solve-window size fixed'},indent=2)+'\n')
        blocks=[]
        with QuietHost() as host:
            (out/'preflight.json').write_text(json.dumps(host.interval(3,limit=2,windows_limit=4),indent=2)+'\n')
            for i in range(-1,args.pairs):
                label='warmup' if i<0 else f'pair-{i:02d}'
                variants=['reference','candidate'];rng.shuffle(variants)
                block={v:execute(cfg,host,v,label) for v in variants}
                for v in variants:block[v]['native_work']=native_work(out/f'{label}-{v}.json.gz')
                block['valid']=all(b['returncode']==0 and not b['validation']['errors'] for b in block.values())
                if block['valid']:
                    a,b=(block[v]['validation']['spectra'] for v in ['reference','candidate'])
                    block['max_frequency_difference']=max(abs(x-y) for aa,bb in zip(a,b,strict=True) for ar,br in zip(aa,bb,strict=True) for x,y in zip(ar,br,strict=True))
                    block['valid'] &= block['max_frequency_difference']<2e-7
                block['admitted']=all(block[v]['host']['accepted'] for v in variants)
                for v in variants:block[v]['validation'].pop('spectra',None)
                blocks.append(block)
                (out/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
                print(name,label,'valid',block['valid'],'admitted',block['admitted'],{v:round(block[v]['elapsed_seconds'],4) for v in variants},flush=True)
                if not block['valid']:break
        ratios=[b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in blocks[1:]]
        (out/'summary.json').write_text(json.dumps({'numerical_pass':all(b['valid'] for b in blocks),
            'ratio':estimate(ratios) if ratios else None,'paired_median_ratio':statistics.median(ratios) if ratios else None,
            'admitted_pairs':sum(b['admitted'] for b in blocks[1:]),'scope':'exploratory complete-workflow result; defaults unchanged'},indent=2)+'\n')

if __name__=='__main__':main()
