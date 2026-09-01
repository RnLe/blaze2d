import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import blaze

FIXTURES = Path(__file__).parents[2] / "interface" / "tests" / "fixtures"


def band_config():
    c = blaze.Config().to_dict()
    c["grid"]["resolution"] = [12,16]
    c["bands"] = {"count":3,"path":{"points":[[0.19,0.13],[0.21,0.17]]}}
    return c


def operator_config(pol="TM"):
    c = band_config()
    del c["bands"]
    c["task"] = "operators"
    c["polarization"] = pol
    c["operators"] = {"band_lo":2,"retained_bands":2,"remote_bands":1,
        "k_point":{"value":[0.19,0.13],"basis":"reciprocal_fractional"},"quantities":["velocity","mass_tensor"]}
    # Let the operator contract choose operator numerical defaults.
    c.pop("eigensolver")
    return c


def compare_arrays(a,b):
    for key in a.get("array_info",{}):
        assert a[key].dtype == b[key].dtype
        np.testing.assert_array_equal(a[key],b[key])
    for first,second in zip(a.get("samples",[]),b.get("samples",[])):
        compare_arrays(first,second)


def test_config_roundtrip_and_diagnostics():
    for path in FIXTURES.glob("*.toml"):
        c = blaze.Config.from_file(path)
        assert c.to_dict() == blaze.Config.from_toml(c.to_toml()).to_dict()
        assert c.to_dict() == blaze.Config.from_dict(c.to_dict()).to_dict()
        assert c.source == path.read_text()
    with pytest.raises(blaze.ConfigurationError) as error:
        blaze.Config.from_toml('schema = "blaze2d/1"\nmisspelled = 1')
    assert error.value.diagnostic["span"] is not None
    with pytest.raises(ValueError):
        blaze.solve(radius_atom=[0.1,0.2,0.3])


def test_numpy_arrays_and_scalar_configuration_agree():
    kwargs=dict(resolution=[12,16],n_bands=3,k_path={"points":[[0.19,0.13],[0.21,0.17]]})
    a=blaze.solve(**kwargs)
    b=blaze.solve(blaze.Config.from_dict(band_config()))
    assert type(a) is dict
    assert a["frequencies"].shape == (2,3)
    assert a["frequencies"].dtype == np.float64
    assert a["frequencies"].flags.c_contiguous
    compare_arrays(a,b)
    assert "scipy" not in sys.modules


def test_study_order_stream_and_single_element_equivalence():
    c=band_config()
    c["sweeps"]=[{"name":"radius","target":"geometry.objects.rod.radius","linspace":{"start":0.22,"stop":0.18,"count":3}},
                 {"name":"pol","target":"polarization","values":["TM","TE"]}]
    config=blaze.Config.from_dict(c)
    with pytest.raises(ValueError): blaze.solve(config)
    study=blaze.run(config,threads=2,queue_capacity=1)
    assert study["statistics"]["status"] == "completed"
    assert len(study["results"]) == 6
    assert study["results"][3]["metadata"]["multi_index"] == [1,1]
    one=band_config(); one["geometry"]["objects"][0]["radius"]=0.2; one["polarization"]="TE"
    compare_arrays(study["results"][3],blaze.solve(one))
    assert len(list(blaze.stream(config,threads=1))) == 6


def test_operator_dimensions_precision_and_field_retention():
    c=operator_config()
    c["results"]={"eigenvectors":True}
    result=blaze.solve(c)
    assert result["eigenvectors"].shape == (5,16,12)
    assert result["eigenvectors"].dtype == np.complex128
    assert result["velocity_matrices"].shape == (2,2,5)
    assert result["metadata"]["remote_band_indices"] == [0,1,4]
    c["results"]["eigenvectors"]=False
    c["eigensolver"]={"precision":"f32"}
    result=blaze.solve(c)
    assert "eigenvectors" not in result
    assert result["metadata"]["storage_precision"] == "f32"
    assert result["velocity_matrices"].dtype == np.complex128


def test_residual_failures_are_preserved():
    c=operator_config()
    c["operators"]["fail_on_residual"]=1e-30
    with pytest.raises(blaze.CalculationError) as error: blaze.solve(c)
    assert error.value.diagnostic["code"] == "residual_gate"
    assert "residuals" in error.value.partial_result
    study=blaze.run(c)
    assert study["statistics"]["status"] == "failed"
    assert len(study["errors"]) == 1
    assert study["results"] == []


