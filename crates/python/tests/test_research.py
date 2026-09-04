import numpy as np
import pytest
import blaze
from test_contract import operator_config, band_config, compare_arrays


def test_reference_fields_use_full_band_window():
    c=operator_config();c["results"]={"eigenvectors":True}
    # Analytical plane waves make the expected eigenpairs explicit.
    c["geometry"]["background_epsilon"]=4.0
    c["geometry"]["objects"][0]["epsilon"]=4.0
    c["eigensolver"]={"tolerance":1e-10,"max_iterations":500}
    k=2*np.pi*np.array([0.19,0.13])
    modes=sorted((np.dot(2*np.pi*np.array([x,y])+k,2*np.pi*np.array([x,y])+k)/4,x,y)
                 for x in range(-2,3) for y in range(-2,3))[:5]
    y,x=np.meshgrid(np.arange(16)/16,np.arange(12)/12,indexing="ij")
    fields=np.array([np.exp(2j*np.pi*(mx*x+my*y))/np.sqrt(4*12*16) for _,mx,my in modes])
    c["operators"]["quantities"].append("overlap")
    c["operators"]["reference"]={"source":"external"}
    with pytest.raises(blaze.CalculationError): blaze.solve(c)
    result=blaze.OperatorDataExtractor.extract_with_reference(c,reference_eigenvectors=fields,warmstart_eigenvectors=fields)
    np.testing.assert_allclose(np.abs(result["overlap_matrix"]),np.eye(2),atol=1e-8)
    assert len(result["metadata"]["external_fields"]["reference_sha256"]) == 64
    np.testing.assert_allclose(result["eigenvalues"],[mode[0] for mode in modes],rtol=1e-8,atol=1e-8)
    assert np.max(result["residuals"]) < 1e-8
    with pytest.raises(ValueError):
        blaze.OperatorDataExtractor.extract_with_reference(c,reference_eigenvectors=fields[:2])


def test_external_dielectric_data_is_retained_with_certification():
    c=band_config();c["bands"]["path"]["points"]=[[0.19,0.13]]
    c["geometry"]["objects"]=[];c["dielectric"]={"source":"external"}
    c["results"]={"eigenvectors":True}
    epsilon=np.full((16,12),4.0)
    result=blaze.OperatorDataExtractor.solve_external_map(c,epsilon)
    assert result["eigenvectors"].shape == (1,3,16,12)
    np.testing.assert_array_equal(result["inputs.epsilon"],epsilon)
    k=result["k_points_cartesian"][0]
    assert result["eigenvalues"][0,0] == pytest.approx(np.dot(k,k)/4,abs=1e-8)
    assert result["metadata"]["certification"]["source"] == "fresh_rayleigh_ritz"
    with pytest.raises(ValueError): blaze.OperatorDataExtractor.solve_external_map(c,-epsilon)
    with pytest.raises(ValueError): blaze.OperatorDataExtractor.solve_external_map(c,epsilon.T)


def test_checkpoint_resume_preserves_completed_results_and_recovers_partial_tail(tmp_path):
    c=operator_config("TE")
    c["sweeps"]=[{"name":"radius","target":"geometry.objects.rod.radius","values":[0.18,0.2,0.22]}]
    path=tmp_path/"study.ndjson"
    def interrupt(event):
        if event["event"] == "result": raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError,match="interrupted"):
        blaze.run_checkpointed(c,path,threads=1,progress=interrupt)
    partial=blaze.load_checkpoint(path)
    assert len(partial["results"]) == 1
    with path.open("ab") as file: file.write(b'{"schema":')
    resumed=blaze.run_checkpointed(c,path,threads=1)
    assert resumed["statistics"]["resumed"] == 1
    assert resumed["statistics"]["completed"] == 3
    assert resumed["statistics"]["status"] == "completed"
    assert len(list(tmp_path.glob("*.interrupted-*"))) == 1
    full=blaze.run(c,threads=1)
    for a,b in zip(full["results"],resumed["results"]): compare_arrays(a,b)
    assert len(blaze.OperatorDataExtractor.load_checkpoint_row(path)) == 3
    c["operators"]["remote_bands"]=2
    with pytest.raises(ValueError,match="configuration differs"):
        blaze.run_checkpointed(c,path)


def test_checkpoint_with_missing_schema_remains_unchanged(tmp_path):
    path=tmp_path/"old.json";source='{"old_checkpoint":true}\n';path.write_text(source)
    with pytest.raises(ValueError,match="Unsupported checkpoint"):
        blaze.run_checkpointed(operator_config(),path)
    assert path.read_text(encoding="utf8") == source
