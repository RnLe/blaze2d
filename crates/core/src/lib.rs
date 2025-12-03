//! Core math, physics, and APIs for the MPB-style 2D solver.

// ============================================================================
// faer parallelism configuration
// ============================================================================
//
// IMPORTANT: We disable faer's internal GEMM parallelism because for typical
// photonic crystal simulations (grids up to ~64×64), the thread synchronization
// overhead far exceeds any parallel speedup. Benchmarks showed:
//
//   - 24×24 grid, 8 bands: parallel GEMM is 10× SLOWER than sequential
//   - 32×32 grid, 10 bands: parallel GEMM is 5× SLOWER than sequential
//
// This does NOT affect the bulk driver's job-level parallelism (rayon), which
// still runs multiple independent simulations in parallel efficiently.
//
// TODO: For very large grids (128×128+), parallel GEMM may become beneficial.
// A future enhancement could add size-based dispatch:
//   - n * r < THRESHOLD → sequential GEMM (current default)
//   - n * r >= THRESHOLD → parallel GEMM with faer::Par::rayon(nthreads)
// Estimated threshold: ~50,000-100,000 elements (e.g., 128×128 grid with 3+ bands)
//
use std::sync::Once;

static INIT_FAER: Once = Once::new();

/// Initialize faer with sequential execution (no internal parallelism).
/// Safe to call multiple times - only the first call takes effect.
pub(crate) fn init_faer_sequential() {
    INIT_FAER.call_once(|| {
        faer::set_global_parallelism(faer::Par::Seq);
    });
}

// ============================================================================
// New strongly-typed lattice system
// ============================================================================
pub mod basis;
pub mod bravais;
pub mod brillouin;
pub mod crystal;

// ============================================================================
// Core modules
// ============================================================================
pub mod analytic_geometry;
pub mod backend;
pub mod band_tracking;
pub mod diagnostics;
pub mod dielectric;
pub mod eigensolver;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod io;
pub mod lattice;
pub mod metrics;
pub mod polarization;
pub mod profiler;
pub mod reference;
pub mod symmetry;
pub mod units;

// ============================================================================
// New modular architecture (Option B refactor)
// ============================================================================
pub mod drivers;
pub mod operators;
pub mod preconditioners;

// ============================================================================
// Convenience re-exports for common types
// ============================================================================

/// Re-export of the core operator trait and Maxwell operator.
pub use operators::{LinearOperator, ThetaOperator};

/// Re-export of the bandstructure driver for backwards compatibility.
pub use drivers::bandstructure;

/// Re-export of eigensolver progress reporting types.
pub use eigensolver::ProgressInfo;

#[cfg(test)]
mod _tests_backend;
#[cfg(test)]
mod _tests_bandstructure;
#[cfg(test)]
mod _tests_dielectric;
#[cfg(test)]
mod _tests_eigensolver;
#[cfg(test)]
mod _tests_field;
#[cfg(test)]
mod _tests_geometry;
#[cfg(test)]
mod _tests_grid;
#[cfg(test)]
mod _tests_io;
#[cfg(test)]
mod _tests_lattice;
#[cfg(test)]
mod _tests_operator;
#[cfg(test)]
mod _tests_polarization;
#[cfg(test)]
mod _tests_reference;
#[cfg(test)]
mod _tests_symmetry;
#[cfg(test)]
mod _tests_units;
