# blaze2d-bulk-driver

High-performance parallel parameter sweep driver for 2D photonic crystal band structure calculations.

## Features

- **Parameter Sweeps**: Define ranges for radius, epsilon, resolution, polarization, lattice type, and multi-atom basis positions
- **Parallel Execution**: Rayon-based thread pool with adaptive load balancing
- **Flexible I/O**: Three output modes for different use cases

## I/O Modes

### Sync (Default)

Traditional synchronous file writes. Simple but may bottleneck solver threads.

### Batch Mode

Decouples computation from I/O with a background writer thread:

- Buffers ~1000 results (~10 MB) in memory
- Background thread flushes to disk without blocking solvers
- Configurable buffer size and flush interval
- **Supports both Full and Selective output**

```rust
let stats = driver.run_batched(BatchConfig::default())?;
```

### Stream Mode

Real-time result emission for live consumers (plotting, web UIs):

- Subscriber pattern with multiple concurrent consumers
- Zero-copy broadcasting via `Arc<CompactBandResult>`
- Built-in backpressure handling
- **Always streams full data** (consumers filter as needed)

```rust
let (rx, handle) = driver.run_streaming()?;
for result in rx {
    plot(result.bands, result.distances);
}
```

## Mode Combinations

| I/O Mode | Output Mode | Behavior |
|----------|-------------|----------|
| Batch | Full | One CSV per job, buffered writes |
| Batch | Selective | Single merged CSV with specific k-points/bands |
| Stream | Full | Full `CompactBandResult` per job |
| Stream | Selective | **Server-side filtering** before broadcast |

### Filtered Streaming

Use `FilteredStreamChannel` or `run_streaming_filtered()` for efficient server-side filtering:

```rust
// Stream only Gamma (0), X (10), M (15) and first 4 bands
let filter = SelectiveFilter::new(
    vec![0, 10, 15],  // k-point indices
    vec![0, 1, 2, 3], // band indices (0-based)
);
let (rx, handle) = driver.run_streaming_filtered(filter)?;

for result in rx {
    assert_eq!(result.num_k_points(), 3);
    assert_eq!(result.num_bands(), 4);
}
```

Or from config:

```rust
let (rx, handle) = driver.run_streaming_selective(&config.output.selective)?;
```

## Output Formats

| Mode | Description |
|------|-------------|
| **Full** | One CSV per job with complete band data |
| **Selective** | Single merged CSV with specific k-points/bands |

## Configuration

TOML files with `[bulk]` section trigger bulk mode:

```toml
[bulk]
output_dir = "sweep_results"
output_mode = "full"  # or "selective"

[bulk.ranges]
radius = { min = 0.2, max = 0.4, step = 0.05 }
eps_inside = { values = [9.0, 12.0, 13.0] }
polarization = ["te", "tm"]

# For selective mode
[bulk.selective]
k_labels = ["Gamma", "X", "M"]
bands = [1, 2, 3, 4]
```

## Bindings

- **Python**: Iterator-based streaming via `BulkDriverPy`
- **WASM**: Callback-based streaming for React/JS via `WasmBulkDriver`

## Key Types

| Type | Purpose |
|------|---------|
| `BulkDriver` | Main driver, owns config and runs jobs |
| `BatchChannel` | Background writer with buffered I/O |
| `StreamChannel` | Real-time broadcast to subscribers |
| `FilteredStreamChannel` | Stream with server-side k/band filtering |
| `SelectiveFilter` | Efficient k-point and band filter |
| `CompactBandResult` | ~10 KB serialized band diagram |
| `Subscriber` | Trait for stream consumers |
| `SelectiveSpec` | Config-based filter spec |
