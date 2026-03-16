//! WASM-oriented backend and streaming interface for MPB2D.
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
//! import init, { WasmBulkDriver } from 'mpb2d-wasm';
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

use mpb2d_core::backend::{SpectralBackend, SpectralBuffer};
use mpb2d_core::grid::Grid2D;
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
    data: Vec<Complex64>,
}

impl SpectralBuffer for WasmField {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn grid(&self) -> Grid2D {
        self.grid
    }

    fn as_slice(&self) -> &[Complex64] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [Complex64] {
        &mut self.data
    }
}

impl SpectralBackend for WasmBackend {
    type Buffer = WasmField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        WasmField {
            grid,
            data: vec![Complex64::default(); grid.len()],
        }
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in &mut buffer.data {
            *value *= alpha;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.data.iter_mut().zip(&x.data) {
            *dst += alpha * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.data.iter().zip(&y.data).map(|(a, b)| a.conj() * b).sum()
    }
}
