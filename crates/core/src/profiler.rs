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
//! use blaze2d_core::profiler::{start_timer, stop_timer, print_profile, get_profile_json};
//!
//! start_timer("my_function");
//! // ... do work ...
//! stop_timer("my_function");
//!
//! print_profile(); // Print summary table
//! get_profile_json(); // Get JSON for automated comparison
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
    #[derive(Debug, Clone)]
    struct ProfileEntry {
        total_time: Duration,
        call_count: usize,
        min_time: Duration,
        max_time: Duration,
        active_start: Option<Instant>,
    }

    impl Default for ProfileEntry {
        fn default() -> Self {
            Self {
                total_time: Duration::ZERO,
                call_count: 0,
                min_time: Duration::MAX,
                max_time: Duration::ZERO,
                active_start: None,
            }
        }
    }

    /// Global profiler state.
    static PROFILER: LazyLock<Mutex<ProfilerState>> =
        LazyLock::new(|| Mutex::new(ProfilerState::default()));

    #[derive(Debug, Default)]
    struct ProfilerState {
        entries: HashMap<String, ProfileEntry>,
        session_start: Option<Instant>,
    }

    /// Start timing a named section.
    #[inline]
    pub fn start_timer(name: &str) {
        let mut profiler = PROFILER.lock().unwrap();
        if profiler.session_start.is_none() {
            profiler.session_start = Some(Instant::now());
        }
        let entry = profiler.entries.entry(name.to_string()).or_default();
        entry.active_start = Some(Instant::now());
    }

    /// Stop timing a named section and accumulate.
    #[inline]
    pub fn stop_timer(name: &str) {
        let mut profiler = PROFILER.lock().unwrap();
        if let Some(entry) = profiler.entries.get_mut(name) {
            if let Some(start) = entry.active_start.take() {
                let elapsed = start.elapsed();
                entry.total_time += elapsed;
                entry.call_count += 1;
                entry.min_time = entry.min_time.min(elapsed);
                entry.max_time = entry.max_time.max(elapsed);
            }
        }
    }

    /// Reset all profiling data.
    pub fn reset_profile() {
        let mut profiler = PROFILER.lock().unwrap();
        profiler.entries.clear();
        profiler.session_start = Some(Instant::now());
    }

    /// Get profiling results as a sorted vector of (name, total_ms, call_count, avg_us).
    pub fn get_profile_data() -> Vec<(String, f64, usize, f64)> {
        let profiler = PROFILER.lock().unwrap();
        let mut entries: Vec<_> = profiler
            .entries
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

    /// Get profiling results as JSON string for automated comparison.
    pub fn get_profile_json() -> String {
        let profiler = PROFILER.lock().unwrap();

        let session_ms = profiler
            .session_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let mut entries: Vec<_> = profiler.entries.iter().collect();
        entries.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"session_total_ms\": {:.3},\n", session_ms));
        json.push_str(&format!("  \"timestamp\": \"{}\",\n", chrono_timestamp()));
        json.push_str("  \"methods\": {\n");

        for (i, (name, entry)) in entries.iter().enumerate() {
            let comma = if i < entries.len() - 1 { "," } else { "" };
            let total_ms = entry.total_time.as_secs_f64() * 1000.0;
            let avg_us = if entry.call_count > 0 {
                entry.total_time.as_secs_f64() * 1_000_000.0 / entry.call_count as f64
            } else {
                0.0
            };
            let min_us = if entry.min_time == Duration::MAX {
                0.0
            } else {
                entry.min_time.as_secs_f64() * 1_000_000.0
            };
            let max_us = entry.max_time.as_secs_f64() * 1_000_000.0;
            let pct = if session_ms > 0.0 {
                total_ms / session_ms * 100.0
            } else {
                0.0
            };

            json.push_str(&format!(
                "    \"{}\": {{ \"calls\": {}, \"total_ms\": {:.3}, \"avg_us\": {:.3}, \"min_us\": {:.3}, \"max_us\": {:.3}, \"pct\": {:.2} }}{}\n",
                name, entry.call_count, total_ms, avg_us, min_us, max_us, pct, comma
            ));
        }

        json.push_str("  }\n");
        json.push_str("}\n");
        json
    }

    fn chrono_timestamp() -> String {
        // Simple timestamp without chrono dependency
        use std::time::SystemTime;
        let duration = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        format!("{}", duration.as_secs())
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

    /// No-op: profiling disabled. Returns empty JSON.
    #[inline(always)]
    pub fn get_profile_json() -> String {
        String::from("{\"session_total_ms\": 0, \"methods\": {}}")
    }

    /// No-op: profiling disabled.
    #[inline(always)]
    pub fn print_profile() {}
}

#[cfg(not(feature = "profiling"))]
pub use disabled::*;
