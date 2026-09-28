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
    result=dict(roots=0,seeded=0,guards=0,discarded_iterations=0,zero_correction=0,max_cache_payload=0,tm_actions_reused=0,
                preparation_actions_reused=0,response_points=0,response_failures=0,inner_iterations=0,inner_a_columns=0,
                difference_a_columns=0,difference_b_columns=0,approximate_response_columns=0)
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
            result['preparation_actions_reused']+=point.get('preparation_actions_reused',0)
            response=point.get('response') or {}
            failure=point.get('response_failure') or {}
            result['response_points']+=bool(response)
            result['response_failures']+=bool(failure)
            for report in [response,failure]:
                for counter in ['inner_a_columns','difference_a_columns','difference_b_columns']:
                    result[counter]+=report.get('work',{}).get(counter,0)
            for column in response.get('inner_columns',[]):
                result['inner_iterations']+=column['iterations']
                result['approximate_response_columns']+=not column['inner_converged']
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
    parser.add_argument('--ablation',choices=['copy','actions','block','predictors','responses','response-actions'],default='copy')
    parser.add_argument('--smoothing',choices=['none','analytic','voronoi'])
    args=parser.parse_args()
    args.reference=args.reference.resolve();args.candidate=args.candidate.resolve()
    rng=random.Random(2026092805)
    for size in [32,64]:
      for pol in ['TM','TE']:
       for control in (['public','guarded'] if args.ablation=='copy' else ['metric','secant','history'] if args.ablation=='predictors'
                       else ['secant','history','residual','response-1','response-2','response-4'] if args.ablation=='responses' else [args.ablation]):
        name=f'square-{pol}-{size}-gamma-{control}'
        if args.filter not in name: continue
        out=args.output/name
        if (out/'summary.json').exists(): continue
        out.mkdir(parents=True,exist_ok=False)
        base=(ROOT/f'benchmarks/geometry-warm-start/baseline/public/square-{pol}-gamma-copy/reference.toml').read_text().replace('resolution = 32',f'resolution = {size}')
        ladder=args.ablation in ('predictors','responses','response-actions')
        if args.smoothing:
            base=base.replace('smoothing = "analytic"',f'smoothing = "{args.smoothing}"')
        if ladder:
            base=base.replace('values = [0.18, 0.2, 0.22]', 'values = ['+', '.join(f'{0.18+0.002*i:.3f}' for i in range(12))+']')
            base=base.replace('[eigensolver]','[eigensolver]\nblock_size = 10')
        for variant in ['reference','candidate']:
            text=base
            if variant=='candidate' or control=='guarded' or args.ablation!='copy':
                method='copy' if variant=='candidate' or args.ablation!='copy' else 'cold'
                if args.ablation=='predictors':method=control if variant=='candidate' else 'copy'
                if args.ablation=='responses':method=control.split('-')[0] if variant=='candidate' else 'copy'
                if args.ablation=='response-actions':method='response'
                actions=args.actions or (args.ablation in ('actions','response-actions') and variant=='candidate')
                text+=f'\n[bands.warm_start]\nmethod = "{method}"\nguards = 2\nretain_tm_actions = {str(actions).lower()}\n'
                if method=='response':
                    budget=int(control.split('-')[-1]) if args.ablation=='responses' else 2
                    text+=f'response_iterations = {budget}\nresponse_tolerance = 0.25\n'
            if args.ablation=='block' and variant=='candidate':
                text=text.replace('[eigensolver]','[eigensolver]\nblock_size = 10')
            (out/f'{variant}.toml').write_text(text)
        cfg=SimpleNamespace(reference=args.candidate if control!='public' else args.reference,
            candidate=args.candidate,output=out,threads=args.threads,fields=False,expected_jobs=24 if ladder else 6)
        (out/'manifest.json').write_text(json.dumps({
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'binaries':{v:{'path':str(getattr(cfg,v)),'sha256':hashlib.sha256(getattr(cfg,v).read_bytes()).hexdigest()} for v in ['reference','candidate']},
            'inputs':{v:hashlib.sha256((out/f'{v}.toml').read_bytes()).hexdigest() for v in ['reference','candidate']},
            'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'threads':args.threads,'expected_jobs':cfg.expected_jobs,'pairs':args.pairs,'warmups':1,'protocol':'benchmarks/geometry-warm-start/plan.md',
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
        ratios=[b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in blocks[1:] if b['valid']]
        (out/'summary.json').write_text(json.dumps({'numerical_pass':all(b['valid'] for b in blocks),
            'ratio':estimate(ratios) if ratios else None,'paired_median_ratio':statistics.median(ratios) if ratios else None,
            'admitted_pairs':sum(b['admitted'] for b in blocks[1:]),'scope':'exploratory complete-workflow result; defaults unchanged'},indent=2)+'\n')

if __name__=='__main__':main()
