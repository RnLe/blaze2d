"""Optional reference alignment for visualization of complete degenerate blocks."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


def align_modes(result, reference_fields, reference_frequencies, *, bands: Sequence[int],
                k_index=0, weights=None, eigenvalue_atol=1e-10, eigenvalue_rtol=1e-8,
                orthogonality_tolerance=1e-5, minimum_overlap=0.99):
    """Return a separate visualization frame aligned to a reference basis.

    ``bands`` are zero-based indices in the returned result. Reference fields
    have shape (band, ny, nx), on the same physical cell, grid and k point;
    their frequencies use the same units as Blaze. Reference bands are matched
    by sorted frequency, allowing tracked source ordering. Both spectra must
    include a separated upper guard band beyond every selected cluster.

    Supply positive mass/quadrature ``weights`` of shape (ny, nx) for TM.
    TE defaults to Euclidean weights. Alignment uses unitary Procrustes within
    complete, mutually compatible eigenvalue clusters. It rejects mixed or
    truncated clusters and poorly matching spans. It changes neither result
    nor reference and does not certify the rotated fields as eigenpairs.
    The caller is responsible for a common physical coordinate convention.
    """
    if result.get('task') != 'bands' or result.get('metadata', {}).get('converged') is not True:
        raise ValueError('Alignment requires a converged band result')
    if any(not np.isfinite(x) or x <= 0 for x in [eigenvalue_atol, eigenvalue_rtol, orthogonality_tolerance]):
        raise ValueError('Alignment tolerances must be finite and positive')
    if not np.isfinite(minimum_overlap) or not 0 < minimum_overlap <= 1:
        raise ValueError('minimum_overlap must lie in (0, 1]')
    source = np.asarray(result.get('eigenvectors'), dtype=np.complex128)
    frequency = np.asarray(result.get('frequencies'), dtype=float)
    reference = np.asarray(reference_fields, dtype=np.complex128)
    reference_frequency = np.asarray(reference_frequencies, dtype=float)
    if source.ndim != 4 or frequency.shape != source.shape[:2] or not isinstance(k_index, (int, np.integer)) or not 0 <= k_index < len(source):
        raise ValueError('Expected retained (k_point, band, ny, nx) fields and a valid k_index')
    source, frequency = source[k_index], frequency[k_index]
    if reference.ndim != 3 or reference.shape[1:] != source.shape[1:] or reference_frequency.shape != (len(reference),):
        raise ValueError('Reference fields and frequencies must share the source grid')
    if any(not np.isfinite(v).all() for v in [source, frequency, reference, reference_frequency]) or np.any(frequency < 0) or np.any(reference_frequency < 0):
        raise ValueError('Fields and nonnegative frequencies must be finite')
    indices = list(bands)
    if not indices or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(source) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('bands must contain unique valid zero-based source indices')
    if weights is None:
        if result['metadata'].get('inner_product') != 'euclidean':
            raise ValueError('Supply the physical mass/quadrature weights for TM alignment')
        weights = np.ones(source.shape[1:])
    if np.iscomplexobj(weights):
        raise ValueError('weights must be real positive mass/quadrature weights')
    weights = np.asarray(weights, dtype=float)
    if weights.shape != source.shape[1:] or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('weights must be finite, positive and have shape (ny, nx)')
    sx, sy = np.argsort(frequency, kind='stable'), np.argsort(reference_frequency, kind='stable')
    lx, ly = (2*np.pi*frequency[sx])**2, (2*np.pi*reference_frequency[sy])**2

    def clusters(values):
        groups, start = [], 0
        for end in range(1,len(values)+1):
            # Bound the whole block's spread; transitive chains of nearby but
            # distinct eigenvalues do not authorize a large rotation.
            if end == len(values) or values[end]-values[start] > eigenvalue_atol+eigenvalue_rtol*max(abs(values[start]),abs(values[end])):
                groups.append(list(range(start,end))); start=end
        return groups

    gx, gy = clusters(lx), clusters(ly)
    positions = {int(index): position for position,index in enumerate(sx)}
    selected = {positions[i] for i in indices}
    transform = np.eye(len(indices), dtype=np.complex128)
    diagnostics = []
    root_weight = np.sqrt(weights).reshape(-1)
    for group in gx:
        if not selected.intersection(group): continue
        if not set(group).issubset(selected) or group not in gy:
            raise ValueError('Select a complete degenerate cluster present in both spectra')
        if group[-1] == len(lx)-1 or group[-1] == len(ly)-1:
            raise ValueError('Compute a separated upper guard band before aligning the final cluster')
        source_indices, target_indices = sx[group], sy[group]
        x = source[source_indices].reshape(len(group),-1).T * root_weight[:,None]
        y = reference[target_indices].reshape(len(group),-1).T * root_weight[:,None]
        nx, ny = np.linalg.norm(x,axis=0), np.linalg.norm(y,axis=0)
        if not np.isfinite(nx).all() or not np.isfinite(ny).all() or np.any(nx == 0) or np.any(ny == 0):
            raise ValueError('Modes must have finite positive norms')
        # Unit source norms within a block are necessary for a unitary rotation
        # of the original fields to preserve the physical mass Gram matrix.
        gram = x.conj().T@x / np.mean(nx**2)
        x, y = x/nx, y/ny
        if max(np.max(abs(gram-np.eye(len(group)))),np.max(abs(y.conj().T@y-np.eye(len(group))))) > orthogonality_tolerance:
            raise ValueError('Modes are not orthogonal with equal source norms in the supplied metric')
        left, singular, right = np.linalg.svd(x.conj().T@y)
        if singular.min() < minimum_overlap:
            raise ValueError('Reference and source subspaces do not meet minimum_overlap')
        rotation = (left@right).T
        output_indices = [indices.index(int(i)) for i in source_indices]
        transform[np.ix_(output_indices,output_indices)] = rotation
        mixing = np.diag(lx[group])@rotation.T-rotation.T@np.diag(lx[group])
        diagnostics.append(dict(bands=source_indices.tolist(),reference_bands=target_indices.tolist(),
            principal_overlaps=singular.tolist(),eigenvalue_spread=float(np.ptp(lx[group])),
            eigenvalue_commutator_norm=float(np.linalg.norm(mixing,2)),
            unitarity_defect=float(np.max(abs(rotation@rotation.conj().T-np.eye(len(group)))))))
    fields = (transform@source[indices].reshape(len(indices),-1)).reshape(len(indices),*source.shape[1:])
    return dict(fields=fields,transform=transform,band_indices=indices,clusters=diagnostics,
        source_frequencies=frequency[indices].copy(),
        scope='visualization frame; original solve gauge and certificates are unchanged; rotated fields require fresh certification for numerical reuse')
