//! Platform-agnostic timing utilities.
//!
//! This module provides a thin abstraction over timing that works on both
//! native platforms (using `std::time::Instant`) and WASM (where timing
//! is not available and returns dummy values).
//!
//! # Usage
//!
//! ```ignore
//! use blaze2d_core::timing::Timer;
//!
//! let timer = Timer::start();
//! // ... do work ...
//! let elapsed_secs = timer.elapsed_secs();
//! ```
//!
//! # Features
//!
//! - `std-time` (default): Use `std::time::Instant` for real timing
//! - Without `std-time`: Return 0.0 for all timing (WASM-compatible)

/// A platform-agnostic timer.
///
/// On native platforms with `std-time` feature, this wraps `std::time::Instant`.
/// On WASM (without `std-time`), this is a no-op that returns dummy values.
#[derive(Debug, Clone, Copy)]
pub struct Timer {
    #[cfg(feature = "std-time")]
    start: std::time::Instant,
    #[cfg(not(feature = "std-time"))]
    _marker: (),
}

impl Timer {
    /// Start a new timer.
    #[inline]
    pub fn start() -> Self {
        #[cfg(feature = "std-time")]
        {
            Self {
                start: std::time::Instant::now(),
            }
        }
        #[cfg(not(feature = "std-time"))]
        {
            Self { _marker: () }
        }
    }

    /// Get elapsed time in seconds since the timer was started.
    ///
    /// Returns 0.0 when compiled without `std-time` feature.
    #[inline]
    pub fn elapsed_secs(&self) -> f64 {
        #[cfg(feature = "std-time")]
        {
            self.start.elapsed().as_secs_f64()
        }
        #[cfg(not(feature = "std-time"))]
        {
            0.0
        }
    }

    /// Get elapsed time in milliseconds since the timer was started.
    ///
    /// Returns 0 when compiled without `std-time` feature.
    #[inline]
    pub fn elapsed_millis(&self) -> u128 {
        #[cfg(feature = "std-time")]
        {
            self.start.elapsed().as_millis()
        }
        #[cfg(not(feature = "std-time"))]
        {
            0
        }
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::start()
    }
}
