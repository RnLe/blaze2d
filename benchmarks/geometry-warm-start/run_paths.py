"""Matched full-path and held-out public workflow decision gates."""
import argparse
import hashlib
import gzip
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
from run_native import native_work


def compare_retained_fields(folder,label):
    """Gauge-invariant Euclidean span angles for complete interior clusters.

    The final public cluster is omitted because the public result does not
    expose the next eigenvalue needed to establish boundary completeness.
    """
    import numpy as np
    documents=[json.loads(gzip.decompress((folder/f'{label}-{v}.json.gz').read_bytes())) for v in ['reference','candidate']]
    maximum=0.;checked=0;omitted=0
    for a,b in zip(*(d['results'] for d in documents),strict=True):
        arrays=[json.loads(gzip.decompress((folder/'objects'/f'{r["arrays_object_sha256"]}.json.gz').read_bytes())) for r in [a,b]]
        fs=[np.array(x['frequencies']['data']).reshape(x['frequencies']['shape']) for x in arrays]
        xs=[]
        for x in arrays:
            raw=np.array(x['eigenvectors']['data']).reshape(-1,2)
            xs.append((raw[:,0]+1j*raw[:,1]).reshape(x['eigenvectors']['shape'][:2]+[-1]))
        for ki in range(len(fs[0])):
            order=[np.argsort(f[ki]) for f in fs]
            f=fs[0][ki,order[0]];ends=[0]+[j for j in range(1,len(f)) if abs(f[j]-f[j-1])>1e-5*max(1.,abs(f[j]))]+[len(f)]
            omitted+=ends[-1]-ends[-2]
            for lo,hi in zip(ends[:-2],ends[1:-1]):
                q=[np.linalg.qr(x[ki,ix[lo:hi]].T)[0] for x,ix in zip(xs,order)]
                sigma=np.linalg.svd(q[0].conj().T@q[1],compute_uv=False)[-1]
                maximum=max(maximum,float(np.sqrt(max(0.,1.-min(1.,sigma)**2))));checked+=hi-lo
    return dict(maximum_principal_sine=maximum,checked_columns=checked,unresolved_boundary_columns=omitted,
        metric='Euclidean span angle on identical target grids; independent of phases and internal basis rotations')


def configuration(lattice,pol,size,precision,fields,family,method,scope,selection,guards):
    accuracy=(5e-4,2e-3,2e-5) if precision=='f32' else (1e-5,1e-7,1e-8)
    middle=[0.5,0.5] if lattice=='square' else [2/3,1/3]
    points=[[0.,0.],[0.25,0.],[0.5,0.],middle,[0.17,0.13],[0.08,0.06],[0.,0.]]
    extra=''
    if family=='displacement':
        target='geometry.objects.rod.center'
        values=[[.011,.013],[.018,.013],[.025,.013],[.032,.013]]
        # A second object prevents a trivial rigid translation of the entire cell.
        extra='''[[geometry.objects]]
name = "fixed"
kind = "circle"
center = [-0.23, -0.17]
radius = 0.07
epsilon = 3.1
'''
    elif family=='events':
        target='geometry.objects.rod.radius';values=[.249,.25,.251,.27]
    else:
        target='geometry.objects.rod.radius';values=[.187,.191,.195,.199]
    smoothing='none' if family=='events' else 'voronoi'
    contrasts=[2.2,17.] if family in ('contrast','events') else [7.3,11.7]
    warm='' if method=='public' else f'''
[bands.warm_start]
method = "{method}"
scope = "{scope}"
anchors = [[0.5, 0.0], {middle}]
donor_selection = "{selection}"
guards = {guards}
max_cache_bytes = 67108864
'''
    return f'''schema = "blaze2d/1"
task = "bands"
polarization = "{pol}"
[geometry]
background_epsilon = 1.0
lattice = {{ type = "{lattice}", a = 1.0 }}
[[geometry.objects]]
name = "rod"
kind = "circle"
center = [0.0, 0.0]
radius = 0.193
epsilon = 8.9
{extra}[grid]
resolution = {size}
[bands]
count = 8
tracking = true
path = {{ points = {points}, basis = "reciprocal_fractional" }}
[bands.accuracy]
max_relative_residual = {accuracy[0]}
max_absolute_residual = {accuracy[1]}
max_orthogonality_defect = {accuracy[2]}
[dielectric]
smoothing = "{smoothing}"
[eigensolver]
precision = "{precision}"
tolerance = {1e-6 if precision=='f32' else 1e-14}
max_iterations = 600
[results]
eigenvectors = {str(fields).lower()}
[[sweeps]]
name = "contrast"
target = "geometry.objects.rod.epsilon"
values = {contrasts}
[[sweeps]]
name = "geometry"
target = "{target}"
values = {values}
{warm}''',accuracy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ablation',choices=['native-path','anchors','donors','public','guards','order'],required=True)
    p.add_argument('--size',type=int,default=48);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--pairs',type=int,default=3);p.add_argument('--precision',choices=['f64','f32'],default='f64')
    p.add_argument('--fields',action='store_true');p.add_argument('--family',choices=['radius','contrast','displacement','events'],default='radius')
    p.add_argument('--filter',default='')
    a=p.parse_args();a.reference=a.reference.resolve();a.candidate=a.candidate.resolve()
    rng=random.Random(2026092809)
    variants={'native-path':['gamma','path'],'anchors':['anchors','path'],'donors':['previous_k','screened'],
              'public':['gamma','path'],'guards':['0','4'],'order':['gamma','path']}[a.ablation]
    for lattice in ['square','triangular']:
      for pol in ['TM','TE']:
       for variant in variants:
        name=f'{lattice}-{pol}-{a.size}-{a.precision}-{a.family}-{variant}'
        if a.filter not in name:continue
        output=a.output/name
        if (output/'summary.json').exists():continue
        output.mkdir(parents=True,exist_ok=False)
        for side in ['reference','candidate']:
            method='secant' if a.ablation in ('public','order') else 'copy'
            if side=='reference' and a.ablation=='public':method='public'
            scope=variant if a.ablation in ('native-path','anchors','public','order') else 'path'
            if a.ablation=='anchors' and side=='reference':scope='gamma'
            selection=variant if side=='candidate' and a.ablation=='donors' else 'same_k'
            guards=int(variant) if side=='candidate' and a.ablation=='guards' else 2
            text,accuracy=configuration(lattice,pol,a.size,a.precision,a.fields,a.family,method,scope,selection,guards)
            if a.ablation=='order' and side=='candidate':text+='parameter_order = "serpentine"\n'
            (output/f'{side}.toml').write_text(text)
        cfg=SimpleNamespace(reference=a.reference,candidate=a.candidate,output=output,threads=a.threads,expected_jobs=8,accuracy=accuracy)
        (output/'manifest.json').write_text(json.dumps(dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            binaries={v:dict(path=str(getattr(cfg,v)),sha256=hashlib.sha256(getattr(cfg,v).read_bytes()).hexdigest()) for v in ['reference','candidate']},
            inputs={v:hashlib.sha256((output/f'{v}.toml').read_bytes()).hexdigest() for v in ['reference','candidate']},
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shared_collector_sha256=hashlib.sha256((ROOT/'benchmarks/symmetry/run_public_sweep.py').read_bytes()).hexdigest(),
            threads=a.threads,expected_jobs=8,pairs=a.pairs,warmups=1,accuracy=accuracy,
            scope='complete public CLI startup, planning, solving, certification and requested JSON output'),indent=2)+'\n')
        blocks=[]
        with QuietHost() as host:
            (output/'preflight.json').write_text(json.dumps(host.interval(3,limit=2,windows_limit=4),indent=2)+'\n')
            for i in range(-1,a.pairs):
                label='warmup' if i<0 else f'pair-{i:02d}'
                sides=['reference','candidate'];rng.shuffle(sides)
                block={s:execute(cfg,host,s,label) for s in sides}
                for s in sides:block[s]['native_work']=native_work(output/f'{label}-{s}.json.gz')
                block['valid']=all(b['returncode']==0 and not b['validation']['errors'] for b in block.values())
                if block['valid']:
                    x,y=(block[s]['validation']['spectra'] for s in ['reference','candidate'])
                    block['max_frequency_difference']=max(abs(x-y) for xx,yy in zip(x,y,strict=True) for xr,yr in zip(xx,yy,strict=True) for x,y in zip(xr,yr,strict=True))
                    def gaps(rows):
                        return [min(row[j+1] for row in rows)-max(row[j] for row in rows) for j in range(len(rows[0])-1)]
                    block['max_path_gap_difference']=max(abs(ga-gb) for xx,yy in zip(x,y,strict=True) for ga,gb in zip(gaps(xx),gaps(yy),strict=True))
                    block['valid'] &= block['max_frequency_difference'] < (2e-5 if a.precision=='f32' else 2e-7)
                    if a.fields:
                        block['field_comparison']=compare_retained_fields(output,label)
                        block['valid'] &= block['field_comparison']['maximum_principal_sine'] < (0.05 if a.precision=='f32' else 0.01)
                block['admitted']=all(block[s]['host']['accepted'] for s in sides)
                for s in sides:block[s]['validation'].pop('spectra',None)
                blocks.append(block);(output/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
                print(name,label,'valid',block['valid'],'admitted',block['admitted'],{s:round(block[s]['elapsed_seconds'],3) for s in sides},flush=True)
                if not block['valid']:break
        ratios=[b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds'] for b in blocks[1:] if b['valid'] and b['admitted']]
        (output/'summary.json').write_text(json.dumps(dict(numerical_pass=all(b['valid'] for b in blocks),
            ratio=estimate(ratios),paired_median_ratio=statistics.median(ratios) if ratios else None,admitted_pairs=len(ratios),
            scope='exploratory held-out full-workflow decision gate; no default promotion'),indent=2)+'\n')


if __name__=='__main__':main()
