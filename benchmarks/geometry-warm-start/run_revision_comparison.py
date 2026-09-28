"""Paired whole-workflow ablation with identical inputs on two frozen revisions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/symmetry'))
from quiet_host import QuietHost
from run_public_sweep import execute, estimate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=ROOT/'benchmarks/geometry-warm-start/baseline/public')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--filter',default='')
    p.add_argument('--pairs',type=int,default=3)
    args=p.parse_args()
    args.reference=args.reference.resolve();args.candidate=args.candidate.resolve()
    rng=random.Random(2026092804)
    for source in sorted(args.inputs.glob('*/*.toml')):
        name=source.parent.name+'-'+source.stem
        if args.filter not in name: continue
        out=args.output/name
        if (out/'summary.json').exists():continue
        out.mkdir(parents=True,exist_ok=False)
        for v in ['reference','candidate']:(out/f'{v}.toml').write_bytes(source.read_bytes())
        cfg=SimpleNamespace(reference=args.reference,candidate=args.candidate,output=out,threads=2,fields=False)
        manifest={'input':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'binaries':{v:{'path':str(getattr(args,v)),'sha256':hashlib.sha256(getattr(args,v).read_bytes()).hexdigest()} for v in ['reference','candidate']},
            'threads':2,'pairs':args.pairs,'protocol':'benchmarks/geometry-warm-start/plan.md'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        blocks=[]
        with QuietHost() as host:
            (out/'preflight.json').write_text(json.dumps(host.interval(3,limit=2,windows_limit=4),indent=2)+'\n')
            for i in range(-1,args.pairs):
                label='warmup' if i<0 else f'pair-{i:02d}'
                variants=['reference','candidate'];rng.shuffle(variants)
                block={v:execute(cfg,host,v,label) for v in variants}
                block['valid']=all(x['returncode']==0 and not x['validation']['errors'] for x in block.values())
                if block['valid']:
                    a,b=(block[v]['validation']['spectra'] for v in ['reference','candidate'])
                    block['max_frequency_difference']=max(abs(x-y) for aa,bb in zip(a,b,strict=True) for ar,br in zip(aa,bb,strict=True) for x,y in zip(ar,br,strict=True))
                    block['valid'] &= block['max_frequency_difference'] < 2e-7
                block['admitted']=all(block[v]['host']['accepted'] for v in variants)
                for v in variants:block[v]['validation'].pop('spectra',None)
                blocks.append(block)
                (out/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
                print(name,label,'valid',block['valid'],'admitted',block['admitted'],{v:round(block[v]['elapsed_seconds'],4) for v in variants},flush=True)
                if not block['valid']:break
        pairs=blocks[1:]
        summary={'numerical_pass':all(b['valid'] for b in blocks),
            'ratio':estimate([b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in pairs]) if pairs else None,
            'admitted_pairs':sum(b['admitted'] for b in pairs),
            'scope':'qualitative full-workflow ratio at identical inputs and resources; no default promotion'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
