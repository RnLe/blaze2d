# Blaze2D

Blaze2D computes photonic bands and projected Maxwell operators for two-dimensional periodic dielectric structures. It supports TE and TM polarization, circle geometries, positive scalar permittivity, and square, triangular, rectangular, or explicit oblique lattices. Independent configurations can run in parallel as ordered parameter studies.

The Rust solver is available through Python, a command-line interface, and the browser [Workbench](https://rnle.github.io/blaze2d/workbench). Each uses the same versioned configuration, numerical defaults, coordinate conventions, and result names. Native calculations support f32 or f64 storage; the browser uses f64. Three-dimensional computation, general geometry, and an MPB compatibility adapter are future work.

## Install

```bash
python -m pip install blaze2d
```

The package requires Python 3.10 or later and NumPy. Release wheels cover Linux x86_64, macOS x86_64 and arm64, and Windows x86_64. Plotting and terminal progress are optional:

```bash
python -m pip install 'blaze2d[plot,progress]'
```

## Calculate bands

```python
import blaze

result = blaze.solve(
    lattice_type="square",
    epsilon_background=1.0,
    epsilon_atoms=8.9,
    radius_atom=0.20,
    polarization="TM",
    resolution=32,
    n_bands=8,
)
frequencies = result["frequencies"]  # NumPy: (k_point, band)
print(frequencies.shape)
print(result["metadata"]["build"])
```

Results are ordinary dictionaries containing named NumPy arrays and scientific metadata. Frequencies use the common reference length recorded in that metadata. Arrays use C ordering; complex outputs have dtype `complex128`. Eigenvectors are retained only when requested. The solver stopping flag reports eigenvalue stabilization; band and operator results also include freshly evaluated residuals and orthogonality information.

## Symmetry and accuracy

Symmetry remains explicit. In a complete configuration, use `[bands.symmetry]`
with `policy = "auto"` for the measured native f64 path. PT selection currently
covers square/triangular equal grids 31–65, 8–16 bands and stopping tolerance at
most `1e-13`; runtime qualification and fallback still apply. Default-tolerance
PT attempts did not pay off, so auto preserves complex arithmetic there.

Add `[bands.accuracy]` to require final convergence, a relative residual at most
`1e-5` OR an absolute residual at most `1e-7` for every mode, and B-orthogonality
at most `1e-8`. These limits are configurable. The absolute criterion handles
zero/near-zero Gamma modes. It certifies the selected finite pencil, not
continuum accuracy. No stopping tolerance is silently changed.

`print(blaze.execution_summary(result))` shows actual methods, reuse, fallback
and acceptance separately from requested settings. The CLI reports this on
stderr by default (`--quiet` suppresses it); Python library calls remain quiet.
For a complete example see `examples/calculations/certified-symmetry.toml`.

`blaze.align_modes` optionally returns a separate reference-aligned visualization
frame using unitary rotations within complete degenerate clusters. Supply a
common physical grid/coordinate convention, an upper guard band, and positive
mass weights for TM. It preserves the original result and does not transfer its
certificates to the rotated fields. The [API guide](https://rnle.github.io/blaze2d/configuration)
describes the exact interface and the recommended, advanced and experimental options.

## Run a TOML study

Download an executable [example](https://rnle.github.io/blaze2d/examples) or export a configuration from the Workbench, then run:

```bash
blaze2d config validate calculation.toml
blaze2d run calculation.toml --output results.npz
```

The same study is available in Python:

```python
config = blaze.Config.from_file("calculation.toml")
study = blaze.run(config, threads=4)
for result in study["results"]:
    print(result["job_index"])
blaze.save(study, "results.npz")
restored = blaze.load("results.npz")
```

Use `blaze.stream(config)` when results should be consumed incrementally. Studies retain completed results and report failures explicitly. JSON, NDJSON, and NPZ exports preserve array shapes, complex values, calculation settings, and build provenance without pickle.

See the [API and TOML reference](https://rnle.github.io/blaze2d/configuration) for named sweep axes, registry studies, k-stencils, projected operators, external dielectric data, and checkpoints. Version 0.7 uses `schema = "blaze2d/1"`; earlier configuration dialects require migration.

Scientific context and benchmark conditions are documented in the [technical report](https://rnle.github.io/blaze2d/blaze) and [manuscript](https://rnle.github.io/blaze2d/paper). MPB also exposes fields, eigenvectors, and group velocities. Blaze focuses on integrated parameter studies and projected operator and derivative extraction.

[Source code](https://github.com/RnLe/blaze2d) · [Workbench guide](https://rnle.github.io/blaze2d/workbench-guide) · MIT license
