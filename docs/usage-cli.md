# CLI Reference

Blaze2D provides a command-line interface for quick simulations and scripting.

## Basic Usage

```bash
blaze2d [COMMAND] [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| `run <file.toml>` | Execute a simulation from a TOML configuration |
| `bulk <file.toml>` | Run a bulk parameter sweep |
| `--help` | Show help information |
| `--version` | Show version |

## Single Run Options

```bash
blaze2d run examples/square_eps13_r0p3_tm_res24.toml
```

### Override TOML Parameters

CLI flags override values specified in the TOML file:

```bash
blaze2d run config.toml --resolution 48 --n-bands 12
```

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--resolution <N>` | Grid resolution (N×N) | From TOML |
| `--n-bands <N>` | Number of bands to compute | 8 |
| `--polarization <TM\|TE>` | Polarization mode | TM |
| `--tol <float>` | Convergence tolerance | 1e-6 |
| `--max-iter <N>` | Maximum LOBPCG iterations | 1000 |
| `--threads <N>` | Number of parallel threads | All cores |
| `--output <path>` | Output file path | stdout |

## Bulk Driver

For parameter sweeps, use the bulk driver:

```bash
blaze2d bulk examples/bulk_parameter_sweep.toml --threads 16
```

The bulk driver distributes independent jobs across threads, maximizing throughput.

### Bulk-specific Options

| Flag | Description |
|------|-------------|
| `--output-dir <path>` | Directory for output files |
| `--benchmark` | Suppress I/O for timing measurements |
| `--dry-run` | Validate configuration without execution |

## Output Formats

```bash
# CSV output (default)
blaze2d run config.toml --output results.csv

# JSON output
blaze2d run config.toml --output results.json --format json
```

## Examples

### Compute 8 bands on a square lattice

```bash
blaze2d --lattice square \
        --eps-bg 13.0 \
        --radius 0.3 \
        --polarization TM \
        --resolution 24 \
        --n-bands 8
```

### High-resolution sweep with custom path

```bash
blaze2d run config.toml \
        --resolution 64 \
        --path-segments 20 \
        --output high_res_bands.csv
```

### Benchmark mode (no file I/O)

```bash
blaze2d bulk examples/bulk_benchmark_100.toml --benchmark --threads 16
```
