# BLAZE - Band-structure LOBPCG Accelerated Zone Eigensolver

High-performance 2D photonic crystal band structure solver with Python bindings.
Also supports Envelope Approximation (EA) eigenproblems for moiré lattice research.

## Installation

```bash
pip install blaze2d
```

## Quick Start

### Maxwell Mode (Photonic Crystals)

```python
from blaze2d import BulkDriver

driver = BulkDriver("config.toml")
print(f"Running {driver.job_count} jobs with {driver.solver_type} solver")

# Stream results as they complete
for result in driver.run_streaming():
    bands = result['bands']           # [k_index][band_index] frequencies
    sv = result['sweep_values']       # Current parameter values
    print(f"Job {result['job_index']}: r={sv.get('atom0.radius')}, bands at Γ: {bands[0][:4]}")

# Or collect all results
results, stats = driver.run_collect()
print(f"Completed in {stats['total_time_secs']:.2f}s")
```

### EA Mode (Envelope Approximation)

```python
from blaze2d import BulkDriver
import numpy as np

driver = BulkDriver("ea_config.toml")

for result in driver.run_streaming():
    eigenvalues = result['eigenvalues']
    eigenvectors = result['eigenvectors']  # List of (N,2) arrays [re, im]
    nx, ny = result['grid_dims']
    
    # Reshape eigenvector to 2D field
    psi = np.array(eigenvectors[0])
    psi_2d = (psi[:, 0] + 1j * psi[:, 1]).reshape((nx, ny))
```

## Configuration Reference

TOML configuration with ordered `[[sweeps]]` for parameter variations.

### Complete Maxwell Example

```toml
# Bulk sweep configuration
[bulk]
threads = 8                    # Number of threads (0 = all cores)
verbose = true                 # Print progress
dry_run = false                # Count jobs without running
skip_final_gamma = false       # Copy initial Γ instead of recalculating
disable_band_tracking = false  # Disable eigenvector-based band reordering

[solver]
type = "maxwell"               # "maxwell" or "ea"

# Base values for sweepable parameters
[defaults]
eps_bg = 12.0                  # Background dielectric
resolution = 32                # Grid resolution
polarization = "TM"            # "TM" or "TE"
lattice_type = "square"        # Lattice type

# Geometry definition
[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"                # square, rectangular, triangular, hexagonal, oblique
a = 1.0                        # Primary lattice constant
# b = 1.5                      # Required for rectangular/oblique
# angle = 60.0                 # Required for oblique (degrees)

[[geometry.atoms]]
pos = [0.5, 0.5]               # Fractional coordinates [0, 1)
radius = 0.2                   # In units of lattice constant
eps_inside = 1.0               # Dielectric inside atom

# Computational grid
[grid]
nx = 32
ny = 32
lx = 1.0                       # Supercell size
ly = 1.0

# K-path specification
[path]
preset = "square"              # square, rectangular, triangular, hexagonal
segments_per_leg = 10          # Points per segment (default: 8)

# Alternative: explicit k-path (use INSTEAD of preset)
# [path]
# k_path = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]]

# Eigensolver settings
[eigensolver]
n_bands = 8                    # Number of bands
tol = 1e-6                     # Convergence tolerance
max_iter = 200                 # Maximum iterations
block_size = 0                 # LOBPCG block size (0 = auto)
# record_diagnostics = false   # Per-iteration diagnostics

# Dielectric smoothing (MPB-style)
[dielectric.smoothing]
mesh_size = 3                  # Subgrid mesh (1 = disabled, default: 3)
method = "analytic"            # "analytic" (default) or "subgrid"
# interface_tolerance = 1e-6   # Interface detection tolerance

# Parameter sweeps (outer to inner)
[[sweeps]]
parameter = "atom0.radius"
min = 0.15
max = 0.35
step = 0.05

[[sweeps]]
parameter = "polarization"
values = ["TM", "TE"]

# Output configuration
[output]
mode = "full"                  # "full" (one CSV per job) or "selective"
directory = "./results"        # Output directory for full mode
prefix = "job"                 # Filename prefix for full mode
# filename = "results.csv"     # Filename for selective mode
# io_mode = "sync"             # "sync", "batch", or "stream"

# For selective mode
[output.selective]
k_indices = [0, 10, 15]        # K-point indices (0-based)
k_labels = ["Gamma", "X", "M"] # Or use high-symmetry labels
bands = [1, 2, 3, 4]           # Band indices (1-based)
```

### EA Example

```toml
[bulk]
threads = 4

[solver]
type = "ea"

[grid]
nx = 64
ny = 64
lx = 10.0
ly = 10.0

[ea]
potential = "data/potential.bin"
mass_inv = "data/mass_inv.bin"
eta = 0.1
domain_size = [10.0, 10.0]
periodic = true

[eigensolver]
n_bands = 12
tol = 1e-6
max_iter = 500

[output]
mode = "full"
directory = "./ea_output"
```

## API Reference

### BulkDriver

```python
driver = BulkDriver(config_path: str, threads: int = 0)
```

| Property | Description |
|----------|-------------|
| `job_count` | Number of jobs to execute |
| `solver_type` | `"maxwell"` or `"ea"` |

| Method | Description |
|--------|-------------|
| `run_streaming()` | Iterator yielding results as computed |
| `run_streaming_filtered(k_indices, band_indices)` | Filtered streaming (Maxwell only) |
| `run_collect()` | Returns `(results_list, stats_dict)` |
| `dry_run()` | Preview job count without executing |

### Result Dictionary

**Common fields:**

| Key | Type | Description |
|-----|------|-------------|
| `job_index` | int | Job number (completion order) |
| `result_type` | str | `"maxwell"` or `"ea"` |
| `params` | dict | Full configuration snapshot |
| `sweep_values` | dict | Current sweep parameter values |
| `sweep_order` | str | Parseable string `"param1=val1\|param2=val2"` |
| `num_bands` | int | Number of bands computed |

**Maxwell fields:**

| Key | Type | Description |
|-----|------|-------------|
| `k_path` | list | K-points as `(kx, ky)` tuples (fractional coordinates) |
| `distances` | list | Cumulative path distance (for x-axis plotting) |
| `bands` | list | 2D array `[k_index][band_index]` (ω/2π normalized) |
| `num_k_points` | int | Number of k-points |

**EA fields:**

| Key | Type | Description |
|-----|------|-------------|
| `eigenvalues` | list | Sorted eigenvalues |
| `eigenvectors` | list | Each as `(N, 2)` array `[re, im]` |
| `grid_dims` | list | `[nx, ny]` for reshaping |
| `converged` | bool | Solver convergence status |
| `n_iterations` | int | Number of iterations used |
| `num_eigenvalues` | int | Number of eigenvalues |

### Stats Dictionary (from `run_collect`)

| Key | Type | Description |
|-----|------|-------------|
| `total_jobs` | int | Total jobs executed |
| `total_time_secs` | float | Wall-clock time |
| `jobs_per_second` | float | Throughput |

## Sweep Parameters

| Parameter | Format | Description |
|-----------|--------|-------------|
| `eps_bg` | range | Background dielectric constant |
| `resolution` | range | Grid resolution (nx=ny) |
| `polarization` | values | `["TM", "TE"]` |
| `lattice_type` | values | `["square", "rectangular", "triangular", "hexagonal"]` |
| `atomN.radius` | range | Atom N radius (N = 0, 1, 2, ...) |
| `atomN.pos_x` | range | Atom N x-position |
| `atomN.pos_y` | range | Atom N y-position |
| `atomN.eps_inside` | range | Atom N dielectric |

**Range format:** `{ min = 0.1, max = 0.5, step = 0.1 }`

**Values format:** `["value1", "value2"]`

## K-Path Presets

| Preset | Path | Description |
|--------|------|-------------|
| `square` | Γ → X → M → Γ | Square lattice |
| `rectangular` | Γ → X → S → Y → Γ | Rectangular lattice |
| `triangular` | Γ → M → K → Γ | Triangular lattice |
| `hexagonal` | Γ → M → K → Γ | Hexagonal lattice (same as triangular) |

## Band Data Interpretation

Frequencies are normalized: `ω_norm = ωa/(2πc)`

Convert to physical frequency:
```python
a = 500e-9  # lattice constant in meters
c = 3e8     # speed of light
f_Hz = omega_norm * c / a
```

## Plotting Example

```python
import matplotlib.pyplot as plt
from blaze2d import BulkDriver

driver = BulkDriver("sweep.toml")

for result in driver.run_streaming():
    distances = result['distances']
    for band_idx in range(result['num_bands']):
        band = [result['bands'][k][band_idx] for k in range(result['num_k_points'])]
        plt.plot(distances, band)
    plt.xlabel('k-path')
    plt.ylabel('ω (normalized)')
    plt.title(f"Job {result['job_index']}")
    plt.show()
```

## Filtered Streaming

For large sweeps, use server-side filtering to reduce memory:

```python
# Stream only high-symmetry points and first 4 bands
for result in driver.run_streaming_filtered(
    k_indices=[0, 10, 15],      # Γ, X, M (0-based)
    band_indices=[0, 1, 2, 3]   # First 4 bands (0-based)
):
    assert result['num_k_points'] == 3
    assert result['num_bands'] == 4
```

Note: Filtering only applies to Maxwell results. EA results pass through unchanged.

## EA Input Data Format

EA mode requires binary input files (raw f64, little-endian, row-major):

- **Potential**: `Nx × Ny` values
- **Mass inverse**: `Nx × Ny × 4` values `[m_xx, m_xy, m_yx, m_yy]`
- **Group velocity** (optional): `Nx × Ny × 2` values `[vg_x, vg_y]`

```python
import numpy as np

nx, ny = 64, 64
Lx, Ly = 10.0, 10.0
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

V = 0.1 * (np.cos(2*np.pi*X/Lx) + np.cos(2*np.pi*Y/Ly))
V.astype(np.float64).tofile('potential.bin')
```

## License

MIT

