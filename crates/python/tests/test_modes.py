"""Unitary field alignment must preserve the span and reject unsafe mixing."""
import numpy as np
import pytest
import blaze


def fixture():
    rng=np.random.default_rng(381)
    weights=np.linspace(1,8,16).reshape(4,4)
    q=np.linalg.qr(rng.normal(size=(16,5))+1j*rng.normal(size=(16,5)))[0]
    fields=(q.T/np.sqrt(weights).reshape(-1)).reshape(5,4,4)
    frequency=np.array([.1,.2,.2,.3,.4])
    unitary=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
    reference=fields.copy();reference[1:3]=(unitary@fields[1:3].reshape(2,-1)).reshape(2,4,4)
    reference[0]*=np.exp(.7j)
    result=dict(task='bands',metadata=dict(converged=True,inner_product='epsilon_weighted'),
        eigenvectors=fields[None],frequencies=frequency[None])
    return result,reference,frequency.copy(),weights


def test_reference_alignment_preserves_metric_and_pointwise_cluster_intensity():
    result,reference,frequency,weights=fixture()
    original=result['eigenvectors'].copy()
    aligned=blaze.align_modes(result,reference,frequency,bands=[0,1,2,3],weights=weights)
    np.testing.assert_allclose(aligned['fields'],reference[:4],atol=1e-14)
    np.testing.assert_array_equal(result['eigenvectors'],original)
    np.testing.assert_allclose(np.sum(abs(aligned['fields'][1:3])**2,axis=0),np.sum(abs(original[0,1:3])**2,axis=0),atol=1e-14)
    x=(aligned['fields']*np.sqrt(weights)).reshape(4,-1)
    np.testing.assert_allclose(x.conj()@x.T,np.eye(4),atol=1e-14)
    assert max(c['eigenvalue_commutator_norm'] for c in aligned['clusters']) < 1e-14


@pytest.mark.parametrize('failure',['partial','guard','split','metric','span','norms'])
def test_alignment_rejects_incomplete_or_incompatible_bases(failure):
    result,reference,frequency,weights=fixture()
    bands=[0,1,2]
    if failure=='partial': bands=[1]
    if failure=='guard': bands=[4]
    if failure=='split': frequency[2]+=.002
    if failure=='metric': weights=None
    if failure=='span': reference[[1,3]]=reference[[3,1]]
    if failure=='norms': result['eigenvectors'][0,1]*=2
    with pytest.raises(ValueError):
        blaze.align_modes(result,reference,frequency,bands=bands,weights=weights)
