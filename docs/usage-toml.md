# TOML Configuration Reference

TOML files provide a readable, reproducible way to specify simulation parameters.

## Basic Structure

```toml
# Simulation metadata
polarization = "TM"  # or "TE"

[geometry]
eps_bg = 13.0        # Background dielectric constant

[geometry.lattice]
type = "square"      # Lattice type
a = 1.0              # Lattice constant

[[geometry.atoms]]   # Atoms in the unit cell
pos = [0.5, 0.5]     # Position (fractional coordinates)
radius = 0.3         # Radius (in units of a)
eps_inside = 1.0     # Dielectric constant inside atom

[grid]
nx = 24              # Grid points in x
ny = 24              # Grid points in y

[path]
preset = "square"    # k-path preset

[eigensolver]
n_bands = 8          # Number of bands
tol = 1e-6           # Convergence tolerance
```

## Complete Reference

### Root Level

| Key | Type | Description |
|-----|------|-------------|
| `polarization` | `"TM"` \| `"TE"` | Electromagnetic polarization |

### `[geometry]`

| Key | Type | Description |
|-----|------|-------------|
| `eps_bg` | float | Background dielectric constant |

### `[geometry.lattice]`

| Key | Type | Description |
|-----|------|-------------|
| `type` | string | Lattice type (see below) |
| `a` | float | Lattice constant |
| `b` | float | Second lattice constant (for rectangular) |
| `angle` | float | Lattice angle in degrees (for oblique) |

**Supported lattice types:**
- `square`
- `rectangular`
- `triangular` / `hexagonal`
- `oblique`

### `[[geometry.atoms]]`

Multiple atoms can be defined by repeating this section:

```toml
[[geometry.atoms]]
pos = [0.0, 0.0]
radius = 0.2
eps_inside = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.15
eps_inside = 4.0
```

| Key | Type | Description |
|-----|------|-------------|
| `pos` | [float, float] | Position in fractional coordinates |
| `radius` | float | Atom radius |
| `eps_inside` | float | Dielectric constant inside atom |

### `[grid]`

| Key | Type | Description |
|-----|------|-------------|
| `nx` | int | Grid points in x-direction |
| `ny` | int | Grid points in y-direction |
| `lx` | float | Physical size in x (default: 1.0) |
| `ly` | float | Physical size in y (default: 1.0) |

### `[path]`

Define the k-space path for band structure calculation.

**Using a preset:**

```toml
[path]
preset = "square"        # Γ → X → M → Γ
segments_per_leg = 10    # k-points per segment
```

**Available presets:**
- `square`: Γ → X → M → Γ
- `triangular` / `hexagonal`: Γ → M → K → Γ
- `rectangular`: Γ → X → S → Y → Γ

**Custom path:**

```toml
[path]
points = [
    [0.0, 0.0],    # Γ
    [0.5, 0.0],    # X
    [0.5, 0.5],    # M
    [0.0, 0.0],    # Γ
]
segments_per_leg = 10
```

### `[eigensolver]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_bands` | int | 8 | Number of eigenvalues to compute |
| `max_iter` | int | 1000 | Maximum LOBPCG iterations |
| `tol` | float | 1e-6 | Convergence tolerance |

## Bulk Sweep Configuration

For parameter sweeps, additional sections define the sweep space:

```toml
[bulk]
output_dir = "sweep_output"
parallel = true

[bulk.sweep]
# Sweep over radius from 0.1 to 0.4 in 10 steps
radius = { start = 0.1, end = 0.4, steps = 10 }

# Sweep over discrete epsilon values
eps_bg = [9.0, 11.0, 13.0]
```

## Example Files

See the `examples/` directory for complete configurations:

| File | Description |
|------|-------------|
| `square_eps13_r0p3_tm_res24.toml` | Basic square lattice, TM mode |
| `triangular_two_atom.toml` | Two-atom basis on triangular lattice |
| `bulk_parameter_sweep.toml` | Parameter sweep example |
| `bulk_stress_test.toml` | High-throughput benchmark |
