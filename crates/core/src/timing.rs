//! Monotonic timing on native platforms and in browser workers.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct Timer { start: Instant }

impl Timer {
    pub fn start() -> Self { Self { start: Instant::now() } }
    pub fn elapsed_secs(&self) -> f64 { self.start.elapsed().as_secs_f64() }
    pub fn elapsed_millis(&self) -> u128 { self.start.elapsed().as_millis() }
}

impl Default for Timer { fn default() -> Self { Self::start() } }
