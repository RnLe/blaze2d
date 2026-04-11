# Blaze Bulk Driver Examples

Example configurations for the bulk driver demonstrating both Maxwell (photonic crystals) and EA (envelope approximation) solver modes.

## Quick Start

```bash
# Run a sweep
blaze-bulk --config examples/bulk_simple_sweep.toml

# Preview job count without running
blaze-bulk --config examples/bulk_parameter_sweep.toml --dry-run

# Use all available threads
blaze-bulk --config examples/bulk_benchmark_100.toml -j -1 --verbose
```

## Configuration Format

### Ordered Sweeps (New)

The `[[sweeps]]` array defines parameters to sweep **in TOML order**. First sweep = outermost loop.

```toml
[bulk]
threads = 8

[solver]
type = "maxwell"

[defaults]
eps_bg = 12.0
resolution = 32

# Sweep order: radius (outer) → eps_bg → polarization (inner)
[[sweeps]]
parameter = "atom0.radius"
min = 0.2
max = 0.4
step = 0.05

[[sweeps]]
parameter = "eps_bg"
min = 10.0
max = 14.0
step = 1.0

[[sweeps]]
parameter = "polarization"
values = ["TM", "TE"]
```

### Sweep Parameters

| Parameter | Format | Description |
|-----------|--------|-------------|
| `eps_bg` | range | Background dielectric |
| `resolution` | range | Grid resolution (nx=ny) |
| `polarization` | values | `["TM", "TE"]` |
| `lattice_type` | values | `["square", "triangular", "hexagonal"]` |
| `atomN.radius` | range | Atom N radius |
| `atomN.pos_x` | range | Atom N x-position |
| `atomN.pos_y` | range | Atom N y-position |
| `atomN.eps_inside` | range | Atom N dielectric |

### Output Modes

| Mode | Description |
|------|-------------|
| `full` | One CSV per job with complete band structure |
| `selective` | Single merged CSV with specified k-points/bands |

## Examples

| File | Jobs | Description |
|------|------|-------------|
| `stream_config.toml` | 5 | Streaming mode with selective output |
| `two_atom_sweep.toml` | 243 | **Complete tutorial** - 2-atom basis with r, ε sweeps |
| `ea_config.toml` | 1 | EA solver for moiré lattices |

## Lattice Conventions

| Lattice | Vectors | High-Symmetry Path |
|---------|---------|-------------------|
| Square | a₁=[a,0], a₂=[0,a] | Γ → X → M → Γ |
| Rectangular | a₁=[a,0], a₂=[0,b] | Γ → X → S → Y → Γ |
| Triangular/Hex | a₁=[a,0], a₂=[a/2, a√3/2] | Γ → M → K → Γ |
