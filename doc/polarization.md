# polarization

**Status:** Enumerates TE/TM cases with serde helpers.

## Responsibilities

- `Polarization` enum captures the current scalar problem type.
- Serde rename ensures configs can specify `"TE"` or `"TM"` directly.

## Usage

```rust
use mpb2d_core::polarization::Polarization;
let pol = Polarization::TM;
```

## Gaps / Next Steps

- Add helper methods (e.g., returning which field component is solved for).
- Consider supporting hybrid or polarized mixes in future phases.
