"""Independent FFT and guarded-subspace audit of public PT-sector WASM fields.

Smoothed TE is explicitly excluded: the existing preview omits inverse tensors.
Raw TE and all TM records use the full sampled FFT pencil, without projection.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2


def state(item, index):
    a = item['result']['arrays']; nx, ny = item['preview']['resolution']
    q = item['config']['bands']['count']
    raw = np.asarray(a['eigenvectors']['data']).reshape(-1,q,ny,nx,2)[index]
    u = raw[...,0] + 1j*raw[...,1]
    f = np.asarray(a['frequencies']['data']).reshape(-1,q)[index]
    return u, f


def certificate(item, index):
    nx,ny = item['preview']['resolution']; u, frequencies = state(item,index)
    q = len(frequencies); values = (math.tau*frequencies)**2
    eps = np.asarray(item['preview']['epsilon']).reshape(ny,nx)
    if not np.isfinite(eps).all() or np.min(eps)<=0 or not np.isfinite(u).all():
        raise ValueError('invalid material/fields')
    lattice = np.asarray(item['preview']['lattice_vectors'])
    reciprocal = math.tau*np.linalg.inv(lattice).T
    k = np.asarray(item['result']['arrays']['k_points']['data']).reshape(-1,2)[index]
    cart = k@reciprocal
    gx = np.arange(nx); gx = np.where(gx<=nx//2,gx,gx-nx)
    gy = np.arange(ny); gy = np.where(gy<=ny//2,gy,gy-ny)
    kx = cart[0]+reciprocal[0,0]*gx[None,:]+reciprocal[1,0]*gy[:,None]
    ky = cart[1]+reciprocal[0,1]*gx[None,:]+reciprocal[1,1]*gy[:,None]
    f = fft2(u,norm='ortho',workers=1)
    if item['config']['polarization']=='TM':
        a=(kx*kx+ky*ky)*f; b=fft2(eps*u,norm='ortho',workers=1)
    else:
        if item['config']['dielectric']['smoothing']!='none':
            raise ValueError('preview omits smoothed inverse tensor')
        a=sum(v*fft2(ifft2(v*f,norm='ortho',workers=1)/eps,norm='ortho',workers=1) for v in (kx,ky))
        b=f.copy()
    a,b,c = [v.reshape(q,-1) for v in (a,b,f)]
    numerator=np.linalg.norm(a-values[:,None]*b,axis=1)
    denominator=np.linalg.norm(a,axis=1)+np.abs(values)*np.linalg.norm(b,axis=1)
    relative=np.divide(numerator,denominator,out=numerator.copy(),where=denominator>0)
    absolute=numerator/np.linalg.norm(c,axis=1)
    return dict(residuals=relative.tolist(),absolute_residuals=absolute.tolist(),
        b_orthogonality_defect=float(np.max(np.abs(c.conj()@b.T-np.eye(q)))),
        zero_indices=np.flatnonzero(np.abs(frequencies)<1e-8).tolist())


def passes(c):
    zero=set(c['zero_indices'])
    return (all(np.isfinite(c[k]).all() for k in ('residuals','absolute_residuals'))
        and math.isfinite(c['b_orthogonality_defect']) and c['b_orthogonality_defect']<=1e-8
        and all(x<=1e-5 for j,x in enumerate(c['residuals']) if j not in zero)
        and all(c['absolute_residuals'][j]<=1e-7 for j in zero))


def compare(a,b,index):
    ua,fa=state(a,index); ub,fb=state(b,index)
    ia,ib=np.argsort(fa),np.argsort(fb)
    ua,fa=ua[ia].reshape(len(fa),-1),fa[ia]
    ub,fb=ub[ib].reshape(len(fb),-1),fb[ib]
    prefix=min(len(fa),len(fb))-1
    # Check the cluster boundary in both spectra before taking a common span.
    while prefix>0 and any(v[prefix]-v[prefix-1]<=1e-6*max(1,abs(v[prefix])) for v in (fa,fb)):
        prefix-=1
    if prefix<=0: raise ValueError('no guarded subspace prefix')
    mass=np.asarray(a['preview']['epsilon']) if a['config']['polarization']=='TM' else np.ones(ua.shape[1])
    qa=np.linalg.qr((ua[:prefix]*np.sqrt(mass)).T)[0]
    qb=np.linalg.qr((ub[:prefix]*np.sqrt(mass)).T)[0]
    smallest=float(np.min(np.linalg.svd(qa.conj().T@qb,compute_uv=False)))
    return dict(guarded_prefix=prefix,max_scaled_frequency_error=float(np.max(np.abs(fa-fb)/np.maximum(1,np.maximum(abs(fa),abs(fb))))),
        max_sin_angle=math.sqrt(max(0,1-min(1,smallest)**2)))


def audit(document,raw):
    issues=list(document['issues']); checks=[]; excluded=[]; comparisons=[]; negative=[]
    records={r['name']:r for r in document['records']}
    if len(records)!=24 or len(document['records'])!=24 or len(document['comparisons'])!=12:
        issues.append('incomplete calculation/comparison matrix')
    if len(document['planning'])!=3 or len(document['configuration_negatives'])!=3:
        issues.append('incomplete planning/negative matrix')
    for item in document['records']:
        name=item['name']
        if item['result'] is None or item['failure'] is not None:
            issues.append(name+': failed calculation'); continue
        for key,a in item['result']['arrays'].items():
            values=np.asarray(a['data'],dtype='<f8')
            if hashlib.sha256(values.tobytes()).hexdigest()!=item['array_sha256'][key]:
                issues.append(name+'/'+key+': array hash mismatch')
        if item['config']['polarization']=='TE' and item['config']['dielectric']['smoothing']!='none':
            excluded.append(dict(name=name,reason='public preview omits inverse tensor; reported fresh certificates and JS path/fallback checks only'))
            continue
        count=len(item['config']['bands']['path']['points'])
        for i in range(count):
            try:
                c=certificate(item,i)
                if not passes(c): issues.append(f'{name}/{i}: independent physical gate failed')
                q=item['config']['bands']['count']; arrays=item['result']['arrays']
                reported=np.asarray(arrays['residuals']['data']).reshape(-1,q)[i]
                absolute=np.asarray(arrays['absolute_residuals']['data']).reshape(-1,q)[i]
                nonzero=[j for j in range(q) if j not in c['zero_indices']]
                delta=max([abs(c['residuals'][j]-reported[j]) for j in nonzero],default=0)
                adelta=float(np.max(abs(np.asarray(c['absolute_residuals'])-absolute)))
                if delta>1e-8 or adelta>1e-7:
                    issues.append(f'{name}/{i}: independent/reported residual disagreement')
                checks.append(dict(name=name,point=i,certificate=c,max_reported_relative_difference=delta,max_reported_absolute_difference=adelta))
            except (ValueError,KeyError,TypeError,IndexError,np.linalg.LinAlgError) as e:
                issues.append(f'{name}/{i}: {e}')
    for pair in document['comparisons']:
        a,b=records[pair['candidate']],records[pair['reference']]
        if a['result'] is None or b['result'] is None: continue
        ca,cb=copy.deepcopy(a['config']),copy.deepcopy(b['config'])
        for c in (ca,cb):
            for key in ('arithmetic','sectors','sector_guard_budget'):
                c['bands']['symmetry'].pop(key,None)
        if ca!=cb or a['preview']!=b['preview']:
            issues.append(a['name']+': physical comparison contract changed')
        for i in range(len(a['config']['bands']['path']['points'])):
            c=compare(a,b,i)
            if c['max_scaled_frequency_error']>1e-7 or c['max_sin_angle']>1e-3:
                issues.append(f"{a['name']}/{i}: spectrum/subspace disagreement")
            comparisons.append(dict(candidate=a['name'],reference=b['name'],point=i,**c))
    for pol in ('TM','TE'):
        item=copy.deepcopy(records[f'none-{pol}-p4m-pt'])
        if not item['result']: continue
        item['result']['arrays']['eigenvectors']['data'][0]+=0.25
        rejected=not passes(certificate(item,0))
        if not rejected: issues.append(pol+': corrupted field escaped independent audit')
        negative.append(dict(polarization=pol,corrupted_field_rejected=rejected,reported_metadata_unchanged=True))
    return dict(schema='blaze2d/wasm-pt-sector-audit/1',input_sha256=hashlib.sha256(raw).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calculations=len(records),independent_points=len(checks),checks=checks,excluded=excluded,
        comparisons=comparisons,negative_controls=negative,issues=issues)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    raw=gzip.decompress(args.input.read_bytes()) if args.input.suffix=='.gz' else args.input.read_bytes()
    result=audit(json.loads(raw),raw)
    with args.output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','comparisons')},indent=2))
    raise SystemExit(bool(result['issues']))
