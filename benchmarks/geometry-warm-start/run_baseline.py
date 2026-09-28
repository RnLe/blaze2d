"""Qualitative controls on the existing public workflow; keep all failed work."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/symmetry'))
from quiet_host import QuietHost
from run_public_sweep import configuration, execute, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--profile-binary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026092803)
    for polarization in ['TM', 'TE']:
        for case in ['gamma-copy', 'historical-pt-first-point']:
            name = f'square-{polarization}-{case}'
            directory = args.output / name
            if (directory/'summary.json').exists():
                print(name, 'already collected', flush=True)
                continue
            directory.mkdir(exist_ok=False)
            cfg = SimpleNamespace(control='pt' if case.startswith('historical') else 'off',
                policy='pt_first_point', support_cutoff=None, heldout=False,
                lattice='square', polarization=polarization, size=32, bands=8,
                intervals=3, fields=False, smoothing='analytic', reference=args.binary,
                candidate=args.binary, output=directory, threads=2)
            for variant in ['reference', 'candidate']:
                text = configuration(cfg, variant == 'candidate')
                if case == 'gamma-copy':
                    text = text.replace('path = { preset = "square", intervals_per_segment = 3 }',
                        'path = { points = [[0.0,0.0]], basis = "reciprocal_fractional" }')
                    text = text.replace('arithmetic = "pt_real"\nk_reuse = "time_reversal"', '')
                (directory/f'{variant}.toml').write_text(text+'\n[bands.accuracy]\n')
            manifest = {'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'binary':str(args.binary),'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'protocol':'benchmarks/geometry-warm-start/plan.md', 'pairs':args.pairs,
                'warmups':1,'threads':2,'scope':'complete CLI workflow; diagnostic qualitative timing, no default promotion'}
            (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            blocks=[]
            with QuietHost() as host:
                admission=host.interval(3,limit=2,windows_limit=4)
                (directory/'preflight.json').write_text(json.dumps(admission,indent=2)+'\n')
                if not admission['accepted']:
                    print(name,'background activity; retaining descriptive timing:',admission['issues'],flush=True)
                for index in range(-1,args.pairs):
                    label='warmup' if index<0 else f'pair-{index:02d}'
                    variants=['reference','candidate']; rng.shuffle(variants)
                    block={v:execute(cfg,host,v,label) for v in variants}
                    block['valid']=all(v['returncode']==0 and not v['validation']['errors'] for v in block.values())
                    if block['valid']:
                        x,y=(block[v]['validation']['spectra'] for v in ['reference','candidate'])
                        block['max_frequency_difference']=max(abs(a-b) for xx,yy in zip(x,y,strict=True)
                            for xr,yr in zip(xx,yy,strict=True) for a,b in zip(xr,yr,strict=True))
                        block['valid'] &= block['max_frequency_difference']<2e-7
                    for variant in variants:
                        block[variant]['validation'].pop('spectra',None)
                    block['admitted']=all(block[v]['host']['accepted'] for v in variants)
                    blocks.append(block)
                    (directory/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
                    print(name,label,'valid',block['valid'],'admitted',block['admitted'],
                        {v:round(block[v]['elapsed_seconds'],4) for v in variants},flush=True)
                    if not block['valid']: break
            measured=blocks[1:]
            summary={'numerical_pass':all(b['valid'] for b in blocks),
                'ratio':estimate([b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in measured]) if measured else None,
                'admitted_pairs':sum(b['admitted'] for b in measured),
                'paired_median_ratio':statistics.median([b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in measured]) if measured else None,
                'scope':'exploratory ratio and resampling interval; three pairs do not establish precise timing'}
            (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            if args.profile_binary:
                for variant in ['reference','candidate']:
                    command=[str(args.profile_binary.resolve()),'run',str(directory/f'{variant}.toml'),
                        '--threads','2','--profile-output',str(directory/f'{variant}-profile.json')]
                    run=subprocess.run(command,capture_output=True)
                    (directory/f'{variant}-profile-result.json.gz').write_bytes(gzip.compress(run.stdout,mtime=0))
                    (directory/f'{variant}-profile-run.json').write_text(json.dumps({'command':command,
                        'returncode':run.returncode,'stderr':run.stderr.decode(errors='replace'),
                        'binary_sha256':hashlib.sha256(args.profile_binary.read_bytes()).hexdigest(),
                        'scope':'explanatory instrumented replay, excluded from timing ratios'},indent=2)+'\n')


if __name__=='__main__': main()
