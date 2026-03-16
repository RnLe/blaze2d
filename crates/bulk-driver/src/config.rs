//! Configuration types for bulk parameter sweeps.
//!
//! This module re-exports types from `mpb2d-bulk-driver-core` and adds
//! native-specific functionality (e.g., CPU thread detection).
//!
//! # Solver Types
//!
//! The bulk driver supports two solver types:
//!
//! - **Maxwell** (default): Photonic crystal band structure calculations using the
//!   Maxwell eigenproblem. Requires geometry, k-path, and polarization.
//!
//! - **EA (Envelope Approximation)**: Moiré lattice eigenproblems using the effective
//!   Hamiltonian H = V(R) - (η²/2)∇·M⁻¹(R)∇. Requires input data files for potential,
//!   mass tensor, and optionally group velocity for drift terms.

// Re-export all types from core
pub use mpb2d_bulk_driver_core::config::*;

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
        self.bulk.threads.unwrap_or_else(num_cpus::get_physical)
    }
}