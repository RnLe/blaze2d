# units

**Status:** Minimal constants module used across crates.

## Responsibilities

- Exposes `SPEED_OF_LIGHT` for normalization or reporting.
- Provides the `Units` struct to capture project-level scaling factors (currently just data storage).

## Usage

```rust
use mpb2d_core::units::SPEED_OF_LIGHT;
let c = SPEED_OF_LIGHT; // 299_792_458.0 m/s
```

## Gaps / Next Steps

- Flesh out helper methods on `Units` (e.g., converting between normalized and SI quantities).
- Add serialization support once units are configurable via TOML.
