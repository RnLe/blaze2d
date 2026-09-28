"""Installed Python workflow checks for bounded native continuation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import blaze


records = []
for polarization in ['TM', 'TE']:
    source = f'''schema = "blaze2d/1"
task = "bands"
polarization = "{polarization}"
[geometry]
background_epsilon = 1.0
lattice = {{ type = "triangular", a = 1.0 }}
[[geometry.objects]]
name = "rod"
kind = "circle"
center = [0.0,0.0]
radius = 0.2
epsilon = 8.9
[grid]
resolution = 17
[dielectric]
smoothing = "voronoi"
[bands]
count = 5
tracking = false
path = {{ points = [[0.0,0.0],[0.17,0.09],[-0.17,-0.09],[0.0,0.0]] }}
[bands.discretization]
kind = "metric_ball"
angular_cutoff = {2*np.pi*3.1}
[bands.symmetry]
arithmetic = "fourier_complex"
[bands.accuracy]
[eigensolver]
precision = "f64"
tolerance = 1e-14
max_iterations = 600
[results]
eigenvectors = true
[[sweeps]]
name = "radius"
target = "geometry.objects.rod.radius"
values = [0.18,0.20,0.22]
'''
    candidate_source = source.replace('arithmetic = "fourier_complex"', '''arithmetic = "pt_real"
plane_group = "p6m"
k_reuse = "time_reversal"
continuation = "previous_geometry"
max_continuation_bytes = 1048576
coarse_start = { resolution = [9,9], max_seed_field_bytes = 1048576 }''')
    reference = blaze.run(blaze.Config.from_toml(source),threads=2)
    candidate = blaze.run(blaze.Config.from_toml(candidate_source),threads=2)
    assert not reference['errors'] and not candidate['errors']
    assert len(candidate['results']) == 3
    difference = 0.0
    seeds = reuses = coarse = 0
    for expected,actual in zip(reference['results'],candidate['results'],strict=True):
        assert actual['job_index'] == expected['job_index']
        a,b = actual, expected
        difference = max(difference,float(np.max(np.abs(np.sort(a['frequencies'],axis=1)-np.sort(b['frequencies'],axis=1)))))
        assert a['eigenvectors'].shape == (4,5,17,17)
        assert a['eigenvectors'].dtype == np.complex128 and np.isfinite(a['eigenvectors']).all()
        assert actual['metadata']['runner']['parallelism'] == 'geometry_chains'
        assert actual['metadata']['converged']
        assert actual['metadata']['accuracy']['accepted']
        assert actual['metadata']['execution']['attempted_iterations'] == sum(actual['metadata']['iterations'])
        for point in actual['metadata']['symmetry']['points']:
            assert point['continuation']['cache_payload_bytes'] <= 1048576//2
            assert point['attempted_iterations'] == sum(a['iterations'] for a in point['solver_attempts'])
            seeds += bool(point['continuation']['seed'])
            reuses += point['source_index'] is not None
            coarse += bool((point.get('coarse_start') or {}).get('used_as_seed'))
    assert difference < 2e-7 and (seeds,reuses,coarse) == (2,6,2)
    records.append(dict(polarization=polarization,max_frequency_difference=difference,seeded_points=seeds,reused_points=reuses,coarse_starts=coarse))
module = Path(blaze.__file__).resolve()
assert 'site-packages' in module.parts, module
from blaze import _native
native = Path(_native.__file__).resolve()
result = dict(package=str(module),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),records=records)
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
