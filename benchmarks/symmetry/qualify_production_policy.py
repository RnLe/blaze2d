"""Small predeclared default-setting qualification, retaining unsuccessful paths."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
from quiet_host import QuietHost
from run_public_sweep import configuration, execute, estimate


def arrays(directory, name, variant):
    study = json.loads(gzip.decompress((directory / f'{name}-{variant}.json.gz').read_bytes()))
    for result in study['results']:
        key = result['arrays_object_sha256']
        result['arrays'] = json.loads(gzip.decompress((directory / 'objects' / f'{key}.json.gz').read_bytes()))
    return study


def compare(actual, reference):
    if len(actual['results']) != 6 or len(reference['results']) != 6:
        return {'frequency_difference': None, 'field_overlap': None}
    frequency_difference, overlap = 0., None
    for a, b in zip(actual['results'], reference['results'], strict=True):
        x = np.array(a['arrays']['frequencies']['data']).reshape(a['arrays']['frequencies']['shape'])
        y = np.array(b['arrays']['frequencies']['data']).reshape(b['arrays']['frequencies']['shape'])
        ix, iy = np.argsort(x, axis=1), np.argsort(y, axis=1)
        x, y = np.take_along_axis(x, ix, axis=1), np.take_along_axis(y, iy, axis=1)
        frequency_difference = max(frequency_difference, float(np.max(abs(x-y))))
        if 'eigenvectors' in a['arrays']:
            def fields(r):
                v = r['arrays']['eigenvectors']
                raw = np.array(v['data']).reshape(-1, 2)
                return (raw[:,0]+1j*raw[:,1]).reshape(v['shape'])
            u, v = fields(a), fields(b)
            for row in range(len(x)):
                end = x.shape[1]-1
                while end > 0 and (abs(x[row,end-1]-x[row,-1])<2e-5 or abs(y[row,end-1]-y[row,-1])<2e-5):
                    end -= 1
                if not end:
                    raise ValueError('No guarded prefix')
                qa = np.linalg.qr(u[row,ix[row,:end]].reshape(end,-1).T)[0]
                qb = np.linalg.qr(v[row,iy[row,:end]].reshape(end,-1).T)[0]
                value = float(np.linalg.svd(qa.conj().T@qb,compute_uv=False).min())
                overlap = value if overlap is None else min(overlap,value)
    return {'frequency_difference': frequency_difference, 'field_overlap': overlap}


def accuracy(study):
    accepted, worst = True, 0.
    for result in study['results']:
        m, a = result['metadata'], result['arrays']
        r, s = np.array(a['residuals']['data']), np.array(a['absolute_residuals']['data'])
        accepted &= bool(m['converged'] and np.all((r<=1e-5)|(s<=1e-7)) and m['certification']['b_orthogonality_defect']<=1e-8)
        worst = max(worst,float(np.max(r)))
    return dict(accepted=accepted and len(study['results'])==6 and not study['errors'], max_relative_residual=worst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=6)
    p.add_argument('--case',help='Run a single condition by name')
    p.add_argument('--tight-references',type=Path,help='Reuse the previously archived, identical tight reference inputs')
    args = p.parse_args()
    args.binary=args.binary.resolve(); args.output=args.output.resolve()
    cases = [(f'{lattice}-{pol}-32', lattice, pol, 32, 8, 'rod') for lattice in ['square','triangular'] for pol in ['TM','TE']]
    cases += [(f'square-{pol}-31-ring','square',pol,31,8,'ring') for pol in ['TM','TE']]
    cases += [('square-TE-32-displaced','square','TE',32,8,'displaced'),
              ('triangular-TE-33-metric','triangular','TE',33,8,'metric'),
              ('square-TE-64','square','TE',64,16,'rod')]
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = dict(binary=str(args.binary),binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pairs=args.pairs,cases=cases,
        tight_references=str(args.tight_references) if args.tight_references else None,
        protocol='benchmarks/symmetry/production-policy-protocol.md')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,lattice,pol,size,bands,motif in cases:
        if args.case and name != args.case: continue
        directory = args.output/name
        directory.mkdir(exist_ok=False)
        cfg = SimpleNamespace(control='off',policy='auto',support_cutoff=14 if motif=='metric' else None,
            heldout=True,lattice=lattice,polarization=pol,size=size,bands=bands,intervals=3,
            fields=name=='square-TE-32',smoothing='voronoi' if motif=='metric' else 'analytic',
            reference=args.binary,candidate=args.binary,output=directory,threads=4)
        for variant in ['reference','candidate']:
            source = configuration(cfg,variant=='candidate').replace('tolerance = 1e-14','tolerance = 1e-6').replace('max_iterations = 600','max_iterations = 200')
            if motif=='displaced': source=source.replace('center = [0.0, 0.0]','center = [0.13, 0.07]')
            if motif=='ring':
                source=source.replace('[[geometry.objects]]', '[[geometry.objects]]\nname="shell"\nkind="circle"\ncenter=[0.0,0.0]\nradius=0.34\nepsilon=3.0\n[[geometry.objects]]',1)
            (directory/f'{variant}.toml').write_text(source)
        ordinary=(directory/'reference.toml').read_text()
        (directory/'reference.toml').write_text(ordinary.replace('tolerance = 1e-6','tolerance = 1e-14').replace('max_iterations = 200','max_iterations = 600'))
        blocks=[]; rng=random.Random(2026092802)
        with QuietHost() as host:
            preflight=[host.interval(3,limit=2,windows_limit=4)]
            for _ in range(2):
                if any(v['accepted'] for v in preflight): break
                preflight.append(host.interval(3,limit=2,windows_limit=4))
            (directory/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
            if not any(v['accepted'] for v in preflight): raise RuntimeError(f'{name}: sustained host load')
            if args.tight_references:
                origin=args.tight_references/name
                if (origin/'reference.toml').read_text()!=ordinary:
                    raise ValueError('Archived reference inputs differ')
                tight=arrays(origin,'tight','reference')
                tight_record=json.loads((origin/'tight-reference.measurement.json').read_text())
                (directory/'tight-reference-origin.json').write_text(json.dumps(dict(path=str(origin),
                    sha256=hashlib.sha256((origin/'tight-reference.json.gz').read_bytes()).hexdigest()),indent=2)+'\n')
            else:
                tight_record=execute(cfg,host,'reference','tight')
                tight=arrays(directory,'tight','reference')
            tight_ok=accuracy(tight)['accepted']
            (directory/'reference.toml').write_text(ordinary)
            for index in range(-1,args.pairs):
                label='warmup' if index<0 else f'pair-{index:02d}'
                variants=['reference','candidate'];rng.shuffle(variants)
                block={variant:execute(cfg,host,variant,label) for variant in variants}
                for variant in variants:
                    study=arrays(directory,label,variant)
                    block[variant]['acceptance']=accuracy(study)
                    block[variant]['tight_comparison']=compare(study,tight) if tight_ok else None
                    completed_and_partial=[*study['results'],*(e['partial_result'] for e in study['errors'] if e.get('partial_result'))]
                    block[variant]['complete_attempted_iterations']=sum(r['metadata'].get('execution',{}).get('attempted_iterations',0) for r in completed_and_partial)
                    block[variant]['validation'].pop('spectra',None)
                left,right=(arrays(directory,label,v) for v in ['reference','candidate'])
                block['completed_arrays_identical']=len(left['results'])==len(right['results'])==6 and all(a['arrays']==b['arrays'] for a,b in zip(left['results'],right['results'],strict=True))
                block['admitted']=all(block[v]['host']['accepted'] for v in variants)
                blocks.append(block)
                (directory/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
                print(name,label,{v:round(block[v]['elapsed_seconds'],3) for v in variants},'admitted',block['admitted'],
                    'accuracy',[block[v]['acceptance']['accepted'] for v in ['reference','candidate']],flush=True)
        measured=blocks[1:]
        ratio=lambda b:b['reference']['elapsed_seconds']/b['candidate']['elapsed_seconds']
        report=dict(tight_reference_accepted=tight_ok,tight_reference_errors=tight_record['validation']['errors'],
            admitted=estimate([ratio(b) for b in measured if b['admitted']]),
            all_pairs=estimate([ratio(b) for b in measured]),
            acceptance={v:all(b[v]['acceptance']['accepted'] for b in blocks) for v in ['reference','candidate']},
            diagnostics=blocks[0]['candidate']['tight_comparison'])
        (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,json.dumps(report),flush=True)


if __name__=='__main__':main()
