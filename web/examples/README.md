# Blaze2D WASM Examples

This folder contains examples and utilities for using the Blaze2D WASM module in web applications.

## Important: Required `[bulk]` Section

**All TOML configurations for the WASM module MUST include a `[bulk]` section!**

This is required because the WASM module uses the bulk driver internally, even for single calculations. Without the `[bulk]` section, you will get:

```
Error: Invalid config: TOML file does not contain [bulk] section - not a bulk configuration
```

## Contents

- **`square_lattice_tutorial.toml`** - Complete example with tutorial comments (ready to use!)
- **`blaze2d-wasm.d.ts`** - TypeScript type definitions for the WASM module
- **`useBandStructure.tsx`** - React hooks for band structure calculations

## Quick Start

### 1. Build the WASM Module

```bash
# From the blaze2d root directory
make wasm
```

This creates `wasm-dist/` containing:
- `blaze2d_backend_wasm.js` - JavaScript glue code
- `blaze2d_backend_wasm_bg.wasm` - WebAssembly binary
- `blaze2d_backend_wasm.d.ts` - TypeScript definitions

### 2. Copy to Your Project

For Next.js:
```bash
cp -r wasm-dist/* my-nextjs-app/public/wasm/
```

For other frameworks, copy to your static assets folder.

### 3. Use K-Point Streaming (RECOMMENDED)

K-point streaming provides **real-time updates** as each k-point is computed, enabling smooth progressive rendering of band diagrams:

```typescript
import init, { WasmBulkDriver } from '/wasm/blaze2d_backend_wasm.js';

async function computeBands() {
    await init();  // Initialize WASM (once per page load)
    
    // Note: configToml MUST include [bulk] section!
    const driver = new WasmBulkDriver(configToml);
    
    // Accumulate data progressively
    const distances: number[] = [];
    const bands: number[][] = [];
    
    // K-POINT STREAMING: callback fires after EACH k-point solve
    driver.runWithKPointStreaming((kResult) => {
        // kResult.k_index: which k-point (0 to total-1)
        // kResult.progress: 0.0 to 1.0
        // kResult.omegas: frequencies for all bands at this k
        
        distances.push(kResult.distance);
        bands.push([...kResult.omegas]);
        
        updatePlotProgressively(distances, bands);
        console.log(`Progress: ${(kResult.progress * 100).toFixed(0)}%`);
    });
}
```

## Streaming Modes

| Method | Granularity | Best For |
|--------|-------------|----------|
| `runWithKPointStreaming()` | Per k-point | **Real-time visualization** (recommended) |
| `runWithCallback()` | Per job | Parameter sweeps |
| `runCollect()` | All at once | Batch processing |

## Configuration Reference

See `square_lattice_tutorial.toml` for detailed comments on all configuration options.

### Minimal Valid Configuration

```toml
# REQUIRED: Marks this as a valid bulk configuration
[bulk]

# Optional: Specify solver type (defaults to "maxwell")
[solver]
type = "maxwell"

# Polarization: TE or TM
polarization = "TM"

[geometry]
eps_bg = 1.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.2
eps_inside = 12.0

[grid]
nx = 32
ny = 32
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 12

[eigensolver]
n_bands = 8
max_iter = 200
tol = 1e-8
```

### Configuration Sections

- **`[bulk]`** - REQUIRED. Marks this as a valid bulk config (can be empty)
- **`[solver]`** - Optional. Solver type: "maxwell" (default) or "ea"
- **Polarization**: `TE` or `TM` (top-level)
- **`[geometry]`**: Lattice type, atoms, permittivities
- **`[grid]`**: Resolution (nx, ny)
- **`[path]`**: K-point path through Brillouin zone
- **`[eigensolver]`**: Number of bands, convergence settings

## Result Data Formats

### K-Point Streaming Result (`runWithKPointStreaming`)

```typescript
{
    stream_type: 'k_point',           // Identifies streaming type
    job_index: 0,                     // Which job (for sweeps)
    k_index: 5,                       // Current k-point index
    total_k_points: 37,               // Total in path
    k_point: [0.5, 0],                // [kx, ky] fractional
    distance: 0.5,                    // Path distance
    omegas: [0.0, 0.15, 0.32, ...],   // Frequencies (ωa/2πc)
    bands: [0.0, 0.15, 0.32, ...],    // Same as omegas
    iterations: 45,                   // LOBPCG iterations
    is_gamma: false,                  // Γ-point flag
    progress: 0.162,                  // 0.0 to 1.0
    num_bands: 8,
    params: { ... }
}
```

### Job-Level Result (`runWithCallback`)

For Maxwell (photonic crystal) calculations:

```typescript
{
    result_type: 'maxwell',
    k_path: [[kx, ky], ...],      // K-points in reciprocal space
    distances: [0.0, 0.1, ...],   // Cumulative path distance
    bands: [[ω₀, ω₁, ...], ...],  // Frequencies: bands[k][band]
    num_k_points: 37,
    num_bands: 8,
    params: { eps_bg, resolution, polarization, atoms, ... }
}
```

Frequencies are normalized as ωa/2πc (dimensionless).

## React Integration

Copy `useBandStructure.tsx` to your project:

```tsx
import { useBandStructureStreaming } from '@/hooks/useBandStructure';

// IMPORTANT: Config must include [bulk] section!
const CONFIG = `
[bulk]

[solver]
type = "maxwell"

polarization = "TM"
# ... rest of config
`;

function MyComponent() {
    // Use K-POINT STREAMING for real-time updates
    const { data, progress, isLoading, compute } = useBandStructureStreaming();
    
    return (
        <div>
            <button onClick={() => compute(CONFIG)}>Compute</button>
            {isLoading && <div>Progress: {(progress * 100).toFixed(0)}%</div>}
            <BandPlot distances={data.distances} bands={data.bands} />
        </div>
    );
}
```

## Performance Tips

1. **Resolution**: Start with 32×32 for testing, use 64×64 or higher for production
2. **Bands**: Computing more bands increases time; only request what you need
3. **Filtering**: Use `runStreamingFiltered()` to only receive specific k-points/bands
4. **Collect mode**: Use `runCollect()` instead of streaming if you don't need real-time updates

## Browser Compatibility

Requires modern browsers with:
- WebAssembly (Chrome 57+, Firefox 52+, Safari 11+, Edge 16+)
- Bulk memory operations (Chrome 75+, Firefox 79+, Safari 15+)

All major browsers from 2020+ are supported.
