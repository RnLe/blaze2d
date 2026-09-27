"""Independent full-FFT pencil and guarded-subspace audit, including TE tensors."""
import copy
import math

import numpy as np
from scipy.fft import fft2, ifft2

from audit_wasm_pt_sectors import certificate as scalar_certificate, compare, passes, state
from coarse_workflow_contract import path_issues, same_jobs
from run_pairs import quality_passes


def item(result):
    return dict(config=result['configuration'], preview=result['audit_material'],
                result={'arrays': result['audit_arrays']})


def certificate(document, index):
    if document['config']['polarization'] == 'TM':
        return scalar_certificate(document, index)
    material = document['preview']
    nx, ny = material['resolution']
    u, frequencies = state(document, index)
    q = len(frequencies)
    values = (math.tau * frequencies)**2
    reciprocal = math.tau * np.linalg.inv(np.asarray(material['lattice_vectors'])).T
    k = np.asarray(document['result']['arrays']['k_points']['data']).reshape(-1, 2)[index]
    cart = k @ reciprocal
    gx, gy = np.arange(nx), np.arange(ny)
    gx, gy = np.where(gx <= nx//2, gx, gx-nx), np.where(gy <= ny//2, gy, gy-ny)
    kx = cart[0] + reciprocal[0, 0]*gx[None, :] + reciprocal[1, 0]*gy[:, None]
    ky = cart[1] + reciprocal[0, 1]*gx[None, :] + reciprocal[1, 1]*gy[:, None]
    f = fft2(u, norm='ortho', workers=1)
    gradients = [ifft2(v*f, norm='ortho', workers=1) for v in (kx, ky)]
    tensors = material['inverse_tensors']
    if tensors is None:
        inverse = np.asarray(material['inverse_epsilon']).reshape(ny, nx)
        if not np.isfinite(inverse).all() or np.min(inverse) <= 0:
            raise ValueError('invalid inverse scalar material')
        flux = [inverse*v for v in gradients]
    else:
        tensor = np.asarray(tensors).reshape(ny, nx, 2, 2)
        if (not np.isfinite(tensor).all() or np.max(abs(tensor[..., 0, 1]-tensor[..., 1, 0])) > 1e-12
                or np.min(np.linalg.eigvalsh(tensor)) <= 0):
            raise ValueError('inverse tensor is not finite symmetric positive definite')
        flux = [sum(tensor[..., i, j]*gradients[j] for j in range(2)) for i in range(2)]
    a = sum(v*fft2(g, norm='ortho', workers=1) for v, g in zip((kx, ky), flux)).reshape(q, -1)
    b = f.reshape(q, -1)
    numerator = np.linalg.norm(a-values[:, None]*b, axis=1)
    denominator = np.linalg.norm(a, axis=1)+abs(values)*np.linalg.norm(b, axis=1)
    relative = np.divide(numerator, denominator, out=numerator.copy(), where=denominator > 0)
    return dict(residuals=relative.tolist(), absolute_residuals=(numerator/np.linalg.norm(b, axis=1)).tolist(),
                b_orthogonality_defect=float(np.max(abs(b.conj()@b.T-np.eye(q)))),
                zero_indices=np.flatnonzero(abs(frequencies) < 1e-8).tolist())


def audit(result, reference=None):
    issues = path_issues(result)
    if not quality_passes(result, result['bands']):
        issues.append('reported physical certificate failed')
    checks, comparisons = [], []
    document = item(result)
    for index, point in enumerate(result['points']):
        try:
            c = certificate(document, index)
            if not passes(c):
                issues.append(f'{index}: independent full-pencil certificate failed')
            nonzero = [i for i in range(result['bands']) if i not in c['zero_indices']]
            difference = max((abs(c['residuals'][i]-point['residuals'][i]) for i in nonzero), default=0)
            absolute_difference = float(np.max(abs(np.asarray(c['absolute_residuals'])-point['absolute_residuals'])))
            if difference > 1e-8 or absolute_difference > 1e-7:
                issues.append(f'{index}: reported/independent certificate mismatch')
            checks.append(dict(point=index, certificate=c, relative_difference=difference,
                               absolute_difference=absolute_difference))
            if reference is not None:
                if not same_jobs(result, reference) or result['audit_material'] != reference['audit_material']:
                    issues.append(f'{index}: physical comparison contract changed')
                c = compare(document, item(reference), index)
                if c['max_scaled_frequency_error'] > 1e-6 or c['max_sin_angle'] > 1e-3:
                    issues.append(f'{index}: spectrum/subspace disagreement')
                comparisons.append(dict(point=index, **c))
        except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError) as error:
            issues.append(f'{index}: independent audit unavailable: {error}')
    damaged = copy.deepcopy(document)
    damaged['result']['arrays']['eigenvectors']['data'][0] += 0.25
    rejected = not passes(certificate(damaged, 0))
    if not rejected:
        issues.append('corrupted physical field escaped independent audit')
    return dict(checks=checks, comparisons=comparisons,
                corrupted_field_rejected=rejected, issues=issues)
