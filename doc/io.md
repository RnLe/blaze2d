# io

**Status:** Handles serde-friendly configuration ingestion plus basic validation hooks, including k-path presets.

## Responsibilities

- `JobConfig` mirrors `BandStructureJob` in a serializable form, including eigensolver settings and optional `path` presets.
- Implements `From<JobConfig>` to convert configs into runtime jobs (auto-generating `k_path` from presets when needed).

## Usage

```rust
use mpb2d_core::io::{JobConfig, PathSpec, PathPreset};
let cfg: JobConfig = toml::from_str(r#"
    geometry = { /* ... */ }
    polarization = "TM"
    [path]
    preset = "hexagonal"
    segments_per_leg = 12
"#)?;
let job = mpb2d_core::bandstructure::BandStructureJob::from(cfg);
assert!(!job.k_path.is_empty());
```

## Gaps / Next Steps

- Allow configs to declare symbolic k-node labels for documentation/plotting.
- Validate input ranges (positive radii, epsilons, etc.) before constructing jobs.
