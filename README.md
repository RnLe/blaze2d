# Blaze2D

A Rust solver for two-dimensional photonic bands and projected Maxwell operators.
Blaze combines TE and TM calculations with ordered parameter studies, registry sampling,
and k-stencils. TOML, Python, the native CLI, and the browser use one configuration contract.

[![PyPI](https://img.shields.io/pypi/v/blaze2d)](https://pypi.org/project/blaze2d/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[Workbench](https://rnle.github.io/blaze2d/workbench) ·
[Examples](https://rnle.github.io/blaze2d/examples) ·
[API and TOML](https://rnle.github.io/blaze2d/configuration)

## Install and calculate

```bash
python -m pip install blaze2d
```

```python
import blaze

result = blaze.solve(resolution=32, n_bands=8)
print(result["frequencies"].shape)  # (k_point, band)
```

Results are dictionaries containing NumPy arrays, resolved parameters, coordinate
conventions, and convergence information. Native execution supports f32 and f64 storage;
the Workbench uses f64. Complex arrays use NumPy `complex128` in both cases.

## Reproducible studies

```bash
blaze2d config validate examples/calculations/radius-sweep.toml
blaze2d run examples/calculations/radius-sweep.toml --output results.npz
```

```python
config = blaze.Config.from_file("examples/calculations/radius-sweep.toml")
study = blaze.run(config, threads=4)
blaze.save(study, "results.npz")
```

Use `blaze.stream(config)` to consume jobs incrementally. JSON, NDJSON, and NPZ preserve
array dimensions, complex data, and provenance. The [configuration reference](https://rnle.github.io/blaze2d/configuration)
explains sweeps, operator quantities, advanced Python inputs, checkpoints, and migration
to `schema = "blaze2d/1"`.

## Build and verify

Rust 1.91.1 and Python 3.10 or later are required to build the Python extension.

```bash
cargo test -p blaze2d-core -p blaze2d-interface -p blaze2d-runner
cargo build --release -p blaze2d-cli
python -m pip install ./crates/python
```

The native executable is `target/release/blaze2d`. The website uses Node 22,
pnpm 10.10.0, and wasm-pack 0.13.1:

```bash
cd web
pnpm install --frozen-lockfile
pnpm build
pnpm test:browser
```

The site build regenerates its interface types and WASM from the current source.
Release workflows build portable wheels and test installation outside the repository.

## Scientific scope

This release supports 2D circle geometry and positive scalar dielectric materials.
It provides projected velocity, mass, registry derivatives, and the formulation-specific
research quantities documented in the API. General geometry, 3D, and an MPB adapter remain
future work. Current overlap rasterization is not guaranteed to follow MPB precedence.

The [technical report](https://rnle.github.io/blaze2d/blaze) records benchmark conditions
and comparisons. MPB exposes fields, eigenvectors, and group velocities; Blaze's research
focus is integrated parameter studies and projected operator extraction.
See [Optimization & Roadmap](https://rnle.github.io/blaze2d/roadmap) for planned work.

## License

MIT. See [LICENSE](LICENSE).
