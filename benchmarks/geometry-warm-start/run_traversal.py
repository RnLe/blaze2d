"""Pair serial geometry-major execution with bounded material/path schedules."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from run_paths import configuration

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/symmetry'))
from quiet_host import QuietHost,assess
from run_public_sweep import estimate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=48);p.add_argument('--pairs',type=int,default=3)
    args=p.parse_args();args.binary=args.binary.resolve();rng=random.Random(2026092810)
    for lattice in ['square','triangular']:
      for pol in ['TM','TE']:
       for candidate in ['k','tile']:
        folder=args.output/f'{lattice}-{pol}-{args.size}-{candidate}'
        if (folder/'summary.json').exists():continue
        folder.mkdir(parents=True,exist_ok=False)
        cfg,_=configuration(lattice,pol,args.size,'f64',False,'radius','copy','path','same_k',2)
        source=folder/'input.toml';source.write_text(cfg)
        (folder/'manifest.json').write_text(json.dumps(dict(binary=str(args.binary),binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            threads=1,pairs=args.pairs,warmups=1,tile_geometries=2,tile_points=2,material_cache_capacity=2,
            scope='serial experimental schedules with the production point solver; includes material setup, certificates, canonical tracking and JSON output'),indent=2)+'\n')
        blocks=[]
        with QuietHost() as host:
            (folder/'preflight.json').write_text(json.dumps(host.interval(3,limit=2,windows_limit=4),indent=2)+'\n')
            for pair in range(-1,args.pairs):
                label='warmup' if pair<0 else f'pair-{pair:02d}'
                order=['geometry',candidate];rng.shuffle(order);block={};spectra={}
                for mode in order:
                    resource=folder/f'{label}-{mode}.resource.json'
                    command=['/usr/bin/time','-f','{"rss_kib":%M,"user_seconds":%U,"system_seconds":%S}','-o',str(resource),
                        str(args.binary),str(source),mode,'2','2']
                    before=host.snapshot();start=time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
                    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    elapsed=time.clock_gettime(time.CLOCK_MONOTONIC_RAW)-start;after=host.snapshot()
                    (folder/f'{label}-{mode}.json.gz').write_bytes(gzip.compress(run.stdout,mtime=0))
                    try:
                        result=json.loads(run.stdout);valid=result['pass'] and len(result['records'])==56
                        spectra[mode]=[sorted(r['frequencies']) for r in result.pop('records')]
                        result.pop('config')
                    except (ValueError,KeyError,TypeError):valid=False;result={}
                    block[mode]=dict(command=command,returncode=run.returncode,stderr=run.stderr.decode(errors='replace'),elapsed_seconds=elapsed,
                        host=assess(before,after,limit=2,windows_limit=4),start=before,end=after,
                        resources=json.loads(next(line for line in reversed(resource.read_text().splitlines()) if line.startswith('{'))),work=result,valid=valid and run.returncode==0)
                block['valid']=all(block[m]['valid'] for m in order)
                if block['valid']:
                    error=max(abs(x-y) for a,b in zip(spectra['geometry'],spectra[candidate],strict=True) for x,y in zip(a,b,strict=True))
                    block['max_frequency_difference']=error;block['valid'] &= error<2e-7
                block['admitted']=all(block[m]['host']['accepted'] for m in order)
                blocks.append(block);(folder/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
                print(folder.name,label,block['valid'],block['admitted'],{m:round(block[m]['elapsed_seconds'],3) for m in order},flush=True)
                if not block['valid']:break
        ratios=[b['geometry']['elapsed_seconds']/b[candidate]['elapsed_seconds'] for b in blocks[1:] if b['valid'] and b['admitted']]
        (folder/'summary.json').write_text(json.dumps(dict(numerical_pass=all(b['valid'] for b in blocks),ratio=estimate(ratios),
            paired_median_ratio=statistics.median(ratios) if ratios else None,admitted_pairs=len(ratios)),indent=2)+'\n')


if __name__=='__main__':main()
