# Blaze2D

A Rust-based 2D Maxwell solver designed for large-scale band diagram sweeps.
Outperforms [MIT's MPB](https://github.com/NanoComp/mpb) in TM polarizations and high-throughput workloads.

[![PyPI](https://img.shields.io/pypi/v/blaze2d)](https://pypi.org/project/blaze2d/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**[Try it in your browser](https://rnle.github.io/blaze2d/blaze/)** — no installation required (WebAssembly).

---

## Quick Start

```bash
# Clone and test
git clone https://github.com/RnLe/blaze2d.git
cd blaze2d
cargo test

# Run a simulation
cargo run --release -- --lattice square --eps-bg 13.0 --radius 0.3 --polarization TM --resolution 24

# Reproduce benchmarks (requires conda/mamba for MPB)
cd benchmarks
make setup-env   # one-time: creates mpb-reference environment
make quick       # ~2 min validation run
```

→ [Full Installation Guide](https://rnle.github.io/blaze2d/installation)

---

## Usage

Blaze2D offers two interfaces:

| Interface | Best For | Documentation |
|-----------|----------|---------------|
| **CLI flags** | Quick single runs, scripting | [CLI Reference](docs/usage-cli.md) |
| **TOML files** | Reproducible simulations, parameter sweeps | [TOML Reference](docs/usage-toml.md) |

### Quick Example

```bash
# Using CLI flags
blaze2d --lattice square --eps-bg 13.0 --radius 0.3 --polarization TM --resolution 24

# Using a TOML file
blaze2d run examples/square_eps13_r0p3_tm_res24.toml
```

---

## Web Demo

The solver compiles to WebAssembly, enabling browser-based simulations without installation:

**https://rnle.github.io/blaze2d/blaze/**

Features:
- Interactive band diagram visualization
- Real-time parameter adjustment
- Export results as CSV/JSON

---

## Benchmarks

Comparative benchmarks against MPB under matched conditions (same tolerance, resolution, band count).

→ [Technical Report with Full Results](https://rnle.github.io/blaze2d/blaze)

---

## Installation

### From PyPI (Recommended)

```bash
pip install blaze2d
```

### From Source

```bash
git clone https://github.com/RnLe/blaze2d.git
cd blaze2d
cargo build --release
```

The binary will be at `target/release/blaze2d`.

### Requirements

- Rust 1.91+ (for building from source)
- Python 3.9+ (for Python bindings)

---

## Optimization Potential

Blaze2D is currently CPU-bound with a straightforward LOBPCG implementation.
Significant performance gains are achievable through:

| Optimization | Expected Impact |
|--------------|-----------------|
| Optimized preconditioners | up to 2–3× fewer iterations |
| Dynamic subspace deflation | Reduced dense matrix ops; faster convergence |
| Advanced BLAS/LAPACK integration | Faster dense operations; speedup potential 1–3x |
| **GPU acceleration (CUDA/Metal)** | 10–100× for large grids; linear scaling (!) |

GPU support is expected to provide substantial speedups even at modest resolutions (64×64), where memory transfer overhead is amortized across many eigenvalue iterations.

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use Blaze2D in academic work, please cite this repository.
