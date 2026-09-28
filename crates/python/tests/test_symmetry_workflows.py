"""Exercise symmetry through installed Python and independently built native CLI APIs."""
from functools import lru_cache
import itertools
import json
import os
import subprocess

import blaze
import numpy as np
import pytest


def configuration(pol, pt=False, sectors=False, coarse=False):
    c = blaze.Config().to_dict()
    c['polarization'] = pol
    c['grid']['resolution'] = [13, 13]
    c['eigensolver'].update(precision='f64', tolerance=1e-14, max_iterations=500)
    c['results'] = {'eigenvectors': True}
    c['bands'] = {'count': 6, 'tracking': False,
                  'path': {'points': [[0., 0.], [.17, 0.], [0., 0.]]},
                  'symmetry': {}}
    s = c['bands']['symmetry']
    if pt:
        s['arithmetic'] = 'pt_real'
    if sectors:
        s.update(sectors='adaptive_group', plane_group='p4m', sector_guard_budget=4,
                 reconstruct_partners=True, share_sector_workspace=True)
    if coarse:
        s['coarse_start'] = {'resolution': [7, 7], 'max_seed_field_bytes': 1 << 20}
    return c


@lru_cache(None)
def solved(pol, pt=False, sectors=False, coarse=False):
    c = blaze.Config.from_dict(configuration(pol, pt, sectors, coarse))
    return c, blaze.solve(c)


def certified(result):
    fields = result['eigenvectors']
    assert fields.shape == (3, 6, 13, 13)
    assert fields.dtype == np.complex128 and fields.flags.c_contiguous
    assert np.isfinite(fields).all()
    assert result['frequencies'].shape == (3, 6)
    assert result['frequencies'].dtype == np.float64
    assert result['metadata']['converged'] is True
    assert result['metadata']['certification']['b_orthogonality_defect'] < 1e-8
    nonzero = np.abs(result['frequencies']) >= 1e-8
    assert np.all(result['residuals'][nonzero] < 1e-5)
    assert np.all(result['absolute_residuals'][~nonzero] < 1e-7)
    assert blaze.execution_summary(result) == result["metadata"]["execution"]["text"]
    assert result["metadata"]["execution"]["attempted_iterations"] == sum(result["metadata"]["iterations"])
    points = result['metadata']['symmetry']['points']
    for i, point in enumerate(points):
        iterations = sum(a['iterations'] for a in point['solver_attempts'])
        assert point['attempted_iterations'] == iterations
        assert result['metadata']['iterations'][i] == iterations


def same_guarded_span(a, b):
    """Exclude the trailing cluster; compare spans without choosing eigenvector phases."""
    for frequency, left, right in zip(a['frequencies'], a['eigenvectors'], b['eigenvectors']):
        end = len(frequency)-1
        while end > 0 and abs(frequency[end-1]-frequency[-1]) < 2e-6:
            end -= 1
        assert end > 0
        qa = np.linalg.qr(left[:end].reshape(end,-1).T)[0]
        qb = np.linalg.qr(right[:end].reshape(end,-1).T)[0]
        assert np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min() > 1-2e-6


@pytest.mark.parametrize('pol', ['TM', 'TE'])
@pytest.mark.parametrize('pt,sectors,coarse', list(itertools.product([False,True], repeat=3)))
def test_python_symmetry_configuration_fields_and_ledger(pol, pt, sectors, coarse, tmp_path):
    config, result = solved(pol, pt, sectors, coarse)
    assert blaze.Config.from_toml(config.to_toml()).to_dict() == config.to_dict()
    certified(result)
    reference = solved(pol)[1]
    np.testing.assert_allclose(result['frequencies'], reference['frequencies'], atol=1e-6, rtol=1e-6)
    same_guarded_span(result, reference)
    points = result['metadata']['symmetry']['points']
    if coarse:
        assert points[0]['coarse_start']['used_as_seed'] is True
        source = [a for a in points[0]['solver_attempts'] if a['method'] == 'coarse_initializer']
        assert len(source) == 1 and source[0]['retained'] is False
        assert source[0]['iterations'] == points[0]['coarse_start']['iterations'] > 0
        assert all('coarse_start' not in point for point in points[1:])
    restored = blaze.load(blaze.save(result, tmp_path/'fields.npz'))
    np.testing.assert_array_equal(restored['eigenvectors'], result['eigenvectors'])
    assert restored['metadata']['symmetry'] == result['metadata']['symmetry']


@pytest.mark.parametrize('pol', ['TM', 'TE'])
def test_zero_coarse_budget_preserves_cold_python_fields_exactly(pol):
    c = configuration(pol, coarse=True)
    c['bands']['symmetry']['coarse_start']['max_seed_field_bytes'] = 0
    result = blaze.solve(c)
    reference = solved(pol)[1]
    for name in ['frequencies', 'eigenvectors', 'residuals', 'absolute_residuals']:
        np.testing.assert_array_equal(result[name], reference[name])
    assert result['metadata']['iterations'] == reference['metadata']['iterations']
    report = result['metadata']['symmetry']['points'][0]['coarse_start']
    assert report['used_as_seed'] is False and report['iterations'] == 0
    assert 'payload budget' in report['rejection']


@pytest.mark.parametrize('pol', ['TM', 'TE'])
def test_rank_transfer_traversal_and_reuse_preserve_output_order(pol):
    c = configuration(pol, pt=True)
    c['bands']['symmetry'].update(k_reuse='unitary', k_traversal='nearest', seed_transfer='rank_aware')
    result = blaze.solve(c)
    certified(result)
    reference = solved(pol)[1]
    np.testing.assert_array_equal(result['k_points'], reference['k_points'])
    np.testing.assert_allclose(result['frequencies'], reference['frequencies'], atol=1e-6, rtol=1e-6)
    same_guarded_span(result, reference)


@pytest.mark.parametrize('change', ['f32', 'too_large', 'empty', 'external'])
def test_invalid_coarse_requests_fail_during_python_configuration(change):
    c = configuration('TE', coarse=True)
    if change == 'f32':
        c['eigensolver']['precision'] = 'f32'
    elif change == 'external':
        c['dielectric']['source'] = 'external'
    else:
        c['bands']['symmetry']['coarse_start']['resolution'] = [13,13] if change == 'too_large' else [0,7]
    with pytest.raises(blaze.ConfigurationError):
        blaze.Config.from_dict(c)


@pytest.mark.parametrize('pol', ['TM', 'TE'])
@pytest.mark.parametrize('pt,sectors', list(itertools.product([False,True], repeat=2)))
def test_native_cli_matches_installed_python_coarse_workflow(pol, pt, sectors, tmp_path):
    cli = os.environ.get('BLAZE_TEST_CLI')
    if not cli:
        pytest.skip('Set BLAZE_TEST_CLI to an independently built native executable')
    config, reference = solved(pol, pt, sectors, True)
    source = tmp_path/'calculation.toml'; source.write_text(config.to_toml())
    validated = subprocess.run([cli,'config','validate',str(source)], check=True, capture_output=True,text=True)
    assert json.loads(validated.stdout)['jobs'] == 1
    normalized = subprocess.run([cli,'config','normalize',str(source)], check=True,capture_output=True,text=True)
    assert blaze.Config.from_toml(normalized.stdout).to_dict() == config.to_dict()
    output = tmp_path/'native.json'
    subprocess.run([cli,'run',str(source),'-o',str(output),'--threads','1'], check=True,capture_output=True,text=True)
    study = blaze.load(output)
    assert study['statistics']['status'] == 'completed' and not study['errors']
    assert len(study['results']) == 1
    result = study['results'][0]
    certified(result)
    np.testing.assert_allclose(result['frequencies'], reference['frequencies'], atol=1e-10, rtol=1e-10)
    same_guarded_span(result, reference)
    assert result['metadata']['iterations'] == reference['metadata']['iterations']
    assert result['metadata']['symmetry']['points'][0]['coarse_start']['used_as_seed'] is True


def test_native_response_study_checkpoint_restarts_a_cold_chain(tmp_path):
    c = configuration('TE')
    c['bands']['accuracy'] = {}
    c['bands']['warm_start'] = {'method': 'response', 'scope': 'path', 'guards': 2,
                              'response_iterations': 2, 'max_cache_bytes': 1 << 22}
    c['sweeps'] = [{'name': 'radius', 'target': 'geometry.objects.rod.radius',
                    'values': [.191, .193, .195]}]
    normalized = blaze.Config.from_dict(c)
    assert blaze.Config.from_toml(normalized.to_toml()).to_dict() == normalized.to_dict()
    full = blaze.run(normalized, threads=1)
    assert not full['errors'] and full['statistics']['status'] == 'completed'
    points = full['results'][1]['metadata']['geometry_warm_start']['points']
    assert any(p and p['response'] for p in points)
    path = tmp_path / 'geometry.ndjson'

    def interrupt(event):
        if event['event'] == 'result':
            raise RuntimeError('interrupted after a stored result')

    with pytest.raises(RuntimeError, match='interrupted after'):
        blaze.run_checkpointed(normalized, path, threads=1, progress=interrupt)
    resumed = blaze.run_checkpointed(normalized, path, threads=1)
    assert resumed['statistics']['resumed'] == 1 and not resumed['errors']
    assert resumed['results'][1]['metadata']['geometry_warm_start']['points'][0]['cold_root']
    for expected, actual in zip(full['results'], resumed['results'], strict=True):
        assert actual['job_index'] == expected['job_index']
        assert actual['metadata']['accuracy']['accepted']
        np.testing.assert_allclose(np.sort(actual['frequencies'], axis=1),
                                   np.sort(expected['frequencies'], axis=1), atol=2e-7, rtol=0)
        same_guarded_span(actual, expected)
