//! WASM-oriented backend and streaming interface for Blaze.
//!
//! This crate provides:
//! - A WebAssembly-compatible spectral backend
//! - Streaming interface for real-time band structure visualization
//! - Unified API matching Python bindings for cross-platform consistency
//!
//! # Solver Types
//!
//! The bulk driver supports two solver types:
//!
//! - **Maxwell**: Photonic crystal band structure (k_path, distances, bands)
//! - **EA**: Envelope Approximation for moiré lattices (eigenvalues only)
//!
//! # I/O Modes
//!
//! - **Stream**: Real-time emission for live visualization (primary focus)
//! - **Selective**: Filter specific k-points and bands before emission
//! - **Collect**: Gather all results at once
//!
//! # Streaming Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmBulkDriver } from 'blaze2d-wasm';
//!
//! await init();
//! const driver = new WasmBulkDriver(configToml);
//!
//! // Check solver type
//! console.log(`Solver: ${driver.solverType}`);  // "maxwell" or "ea"
//!
//! // Stream with callback
//! driver.runWithCallback((result) => {
//!     if (result.result_type === 'maxwell') {
//!         updatePlot(result.distances, result.bands);
//!     } else {  // EA
//!         displayEigenvalues(result.eigenvalues);
//!     }
//! });
//! ```
//!
//! # Selective Mode (Filtering)
//!
//! ```javascript
//! // Stream only Gamma, X, M points and first 4 bands
//! driver.runStreamingFiltered(
//!     [0, 10, 15],  // k_indices for Γ, X, M
//!     [0, 1, 2, 3], // band indices (0-based)
//!     (result) => {
//!         console.assert(result.num_k_points === 3);
//!         console.assert(result.num_bands === 4);
//!     }
//! );
//! ```
//!
//! # React Integration
//!
//! ```javascript
//! function BandPlot() {
//!     const [results, setResults] = useState([]);
//!     
//!     useEffect(() => {
//!         const driver = new WasmBulkDriver(config);
//!         driver.runWithCallback((result) => {
//!             setResults(prev => [...prev, result]);
//!         });
//!     }, []);
//!     
//!     return <Plot data={results} />;
//! }
//! ```

use blaze2d_core::backend::{SpectralBackend, SpectralBuffer};
use blaze2d_core::field::FieldScalar;
use blaze2d_core::grid::Grid2D;
use num_complex::Complex64;

#[cfg(feature = "bindings")]
use wasm_bindgen::prelude::*;

// Streaming module for JavaScript consumers
#[cfg(feature = "streaming")]
pub mod streaming;

// Re-export streaming types when feature is enabled
#[cfg(feature = "streaming")]
pub use streaming::*;

#[derive(Clone)]
#[cfg_attr(feature = "bindings", wasm_bindgen)]
pub struct WasmBackend;

#[derive(Clone)]
pub struct WasmField {
    grid: Grid2D,
    data: Vec<FieldScalar>,
}

impl SpectralBuffer for WasmField {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn grid(&self) -> Grid2D {
        self.grid
    }

    fn as_slice(&self) -> &[FieldScalar] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [FieldScalar] {
        &mut self.data
    }
}

impl SpectralBackend for WasmBackend {
    type Buffer = WasmField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        WasmField {
            grid,
            data: vec![FieldScalar::default(); grid.len()],
        }
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in &mut buffer.data {
            // Compute in f64, store in FieldScalar
            let v64 = Complex64::new(value.re as f64, value.im as f64);
            let result = v64 * alpha;
            #[cfg(not(feature = "mixed-precision"))]
            {
                *value = result;
            }
            #[cfg(feature = "mixed-precision")]
            {
                *value = FieldScalar::new(result.re as f32, result.im as f32);
            }
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.data.iter_mut().zip(&x.data) {
            // Compute in f64, store in FieldScalar
            let src64 = Complex64::new(src.re as f64, src.im as f64);
            let dst64 = Complex64::new(dst.re as f64, dst.im as f64);
            let result = dst64 + alpha * src64;
            #[cfg(not(feature = "mixed-precision"))]
            {
                *dst = result;
            }
            #[cfg(feature = "mixed-precision")]
            {
                *dst = FieldScalar::new(result.re as f32, result.im as f32);
            }
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        // Accumulate in f64 for numerical stability
        x.data
            .iter()
            .zip(&y.data)
            .map(|(a, b)| {
                let a64 = Complex64::new(a.re as f64, a.im as f64);
                let b64 = Complex64::new(b.re as f64, b.im as f64);
                a64.conj() * b64
            })
            .sum()
    }
}
