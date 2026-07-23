//! Configuration types for bulk parameter sweeps.
//!
//! This module re-exports types from `blaze2d-bulk-driver-core` and adds
//! native-specific functionality (e.g., CPU thread detection).
//!
//! # Solver Types
//!
//! - **Maxwell** (default): Photonic crystal band structure via the Maxwell
//!   eigenproblem. Requires geometry, k-path, and polarization.
//! - **OperatorData**: Operator-data extraction at a single (R, k₀) point.

// Re-export all types from core
pub use blaze2d_bulk_driver_core::config::*;

// ============================================================================
// Native-specific Extensions
// ============================================================================

/// Extension trait for BulkConfig with native-specific functionality.
pub trait BulkConfigNativeExt {
    /// Get the effective number of threads.
    /// Defaults to physical CPU cores (optimal for CPU-bound workloads).
    fn effective_threads(&self) -> usize;
}

impl BulkConfigNativeExt for BulkConfig {
    fn effective_threads(&self) -> usize {
        self.run.threads.unwrap_or_else(num_cpus::get_physical)
    }
}
