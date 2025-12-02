//! Simple profiling helper for performance analysis.
//!
//! **By default, profiling is disabled and has zero runtime cost.**
//! To enable profiling, compile with `--features profiling`:
//! ```bash
//! cargo build --release --features profiling
//! ```
//!
//! Usage:
//! ```
//! use mpb2d_core::profiler::{start_timer, stop_timer, print_profile};
//!
//! start_timer("my_function");
//! // ... do work ...
//! stop_timer("my_function");
//!
//! print_profile(); // Print summary table
//! ```

// ============================================================================
// PROFILING ENABLED: Full implementation
// ============================================================================
#[cfg(feature = "profiling")]
mod enabled {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};
    use std::time::{Duration, Instant};

    /// Profiling data for a single named section.
    #[derive(Debug, Clone, Default)]
    struct ProfileEntry {
        total_time: Duration,
        call_count: usize,
        active_start: Option<Instant>,
    }

    /// Global profiler state.
    static PROFILER: LazyLock<Mutex<HashMap<String, ProfileEntry>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    /// Start timing a named section.
    #[inline]
    pub fn start_timer(name: &str) {
        let mut profiler = PROFILER.lock().unwrap();
        let entry = profiler.entry(name.to_string()).or_default();
        entry.active_start = Some(Instant::now());
    }

    /// Stop timing a named section and accumulate.
    #[inline]
    pub fn stop_timer(name: &str) {
        let mut profiler = PROFILER.lock().unwrap();
        if let Some(entry) = profiler.get_mut(name) {
            if let Some(start) = entry.active_start.take() {
                entry.total_time += start.elapsed();
                entry.call_count += 1;
            }
        }
    }

    /// Reset all profiling data.
    pub fn reset_profile() {
        let mut profiler = PROFILER.lock().unwrap();
        profiler.clear();
    }

    /// Get profiling results as a sorted vector of (name, total_ms, call_count, avg_us).
    pub fn get_profile_data() -> Vec<(String, f64, usize, f64)> {
        let profiler = PROFILER.lock().unwrap();
        let mut entries: Vec<_> = profiler
            .iter()
            .map(|(name, entry)| {
                let total_ms = entry.total_time.as_secs_f64() * 1000.0;
                let avg_us = if entry.call_count > 0 {
                    entry.total_time.as_secs_f64() * 1_000_000.0 / entry.call_count as f64
                } else {
                    0.0
                };
                (name.clone(), total_ms, entry.call_count, avg_us)
            })
            .collect();

        // Sort by total time descending
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        entries
    }

    /// Print profiling results as a formatted table.
    pub fn print_profile() {
        let entries = get_profile_data();

        if entries.is_empty() {
            println!("No profiling data collected.");
            return;
        }

        let total_ms: f64 = entries.iter().map(|(_, t, _, _)| t).sum();

        println!();
        println!(
            "╭────────────────────────────────────────────────────────────────────────────────╮"
        );
        println!(
            "│                              PROFILING RESULTS                                 │"
        );
        println!(
            "├────────────────────────────────────┬──────────┬─────────┬──────────┬───────────┤"
        );
        println!(
            "│ Function                           │ Total ms │  Calls  │  Avg µs  │  % Time   │"
        );
        println!(
            "├────────────────────────────────────┼──────────┼─────────┼──────────┼───────────┤"
        );

        for (name, total, calls, avg_us) in &entries {
            let pct = if total_ms > 0.0 {
                total / total_ms * 100.0
            } else {
                0.0
            };
            println!(
                "│ {:<34} │ {:>8.2} │ {:>7} │ {:>8.1} │ {:>8.1}% │",
                truncate_name(name, 34),
                total,
                calls,
                avg_us,
                pct
            );
        }

        println!(
            "├────────────────────────────────────┼──────────┼─────────┼──────────┼───────────┤"
        );
        println!(
            "│ {:<34} │ {:>8.2} │         │          │   100.0%  │",
            "TOTAL (measured)", total_ms
        );
        println!(
            "╰────────────────────────────────────┴──────────┴─────────┴──────────┴───────────╯"
        );
        println!();
    }

    fn truncate_name(name: &str, max_len: usize) -> String {
        if name.len() <= max_len {
            name.to_string()
        } else {
            format!("{}...", &name[..max_len - 3])
        }
    }
}

#[cfg(feature = "profiling")]
pub use enabled::*;

// ============================================================================
// PROFILING DISABLED: No-op stubs with zero runtime cost
// ============================================================================
#[cfg(not(feature = "profiling"))]
mod disabled {
    /// No-op: profiling disabled.
    #[inline(always)]
    pub fn start_timer(_name: &str) {}

    /// No-op: profiling disabled.
    #[inline(always)]
    pub fn stop_timer(_name: &str) {}

    /// No-op: profiling disabled.
    #[inline(always)]
    pub fn reset_profile() {}

    /// No-op: profiling disabled. Returns empty vec.
    #[inline(always)]
    pub fn get_profile_data() -> Vec<(String, f64, usize, f64)> {
        Vec::new()
    }

    /// No-op: profiling disabled.
    #[inline(always)]
    pub fn print_profile() {}
}

#[cfg(not(feature = "profiling"))]
pub use disabled::*;

/// Convenience macro for timing a block.
/// When profiling is disabled, this compiles to just the block with no overhead.
#[macro_export]
macro_rules! profile_block {
    ($name:expr, $block:expr) => {{
        $crate::profiler::start_timer($name);
        let result = $block;
        $crate::profiler::stop_timer($name);
        result
    }};
}
