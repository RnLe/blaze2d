//! Adaptive thread management with EWMA timing and hysteresis.
//!
//! This module provides intelligent thread count adjustment based on runtime
//! performance metrics. It uses exponentially weighted moving averages (EWMA)
//! to smooth out timing measurements and hysteresis to prevent oscillation.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Configuration for the adaptive thread manager.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Minimum number of threads (floor).
    pub min_threads: usize,
    /// Maximum number of threads (ceiling, defaults to physical CPU cores).
    /// Using physical cores (not hyperthreads) is optimal for CPU-bound workloads.
    pub max_threads: usize,
    /// EWMA decay factor (0.0-1.0, higher = more weight on recent).
    pub ewma_alpha: f64,
    /// Minimum improvement ratio to increase threads.
    pub improvement_threshold: f64,
    /// Maximum degradation ratio before decreasing threads.
    pub degradation_threshold: f64,
    /// Hysteresis: minimum samples before considering adjustment.
    pub min_samples_before_adjust: usize,
    /// Hysteresis: minimum time between adjustments.
    pub adjustment_cooldown: Duration,
    /// How many jobs to run before first evaluation.
    pub warmup_jobs: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        // Use physical cores, not logical (hyperthreads don't help CPU-bound workloads)
        let physical_cpus = num_cpus::get_physical();
        Self {
            min_threads: 1,
            max_threads: physical_cpus,
            // EWMA with moderate smoothing
            ewma_alpha: 0.3,
            // Require 10% improvement to increase threads
            improvement_threshold: 0.10,
            // Allow 15% degradation before decreasing (account for noise)
            degradation_threshold: 0.15,
            // Need at least 3 samples at current level before adjusting
            min_samples_before_adjust: 3,
            // Wait at least 500ms between adjustments
            adjustment_cooldown: Duration::from_millis(500),
            // Run a few jobs before starting to adapt
            warmup_jobs: 2,
        }
    }
}

/// Timing sample for a completed job.
#[derive(Debug, Clone, Copy)]
pub struct JobTiming {
    pub duration: Duration,
    pub thread_count: usize,
}

/// State for the adaptive thread manager.
struct AdaptiveState {
    /// Current thread count.
    current_threads: usize,
    /// EWMA of job throughput (jobs per second) at current thread count.
    current_throughput_ewma: Option<f64>,
    /// Best observed throughput and its thread count.
    best_throughput: Option<(f64, usize)>,
    /// Samples collected at current thread count since last adjustment.
    samples_at_current: usize,
    /// Last time we adjusted thread count.
    last_adjustment: Option<Instant>,
    /// Total jobs completed.
    jobs_completed: usize,
    /// Recent timings for analysis.
    recent_timings: Vec<JobTiming>,
    /// Direction of last adjustment (+1 = increased, -1 = decreased, 0 = none).
    last_direction: i8,
    /// Consecutive adjustments in same direction (for momentum).
    consecutive_same_direction: usize,
    /// Log of adjustments for debugging.
    adjustment_log: Vec<AdjustmentEvent>,
}

/// Record of a thread count adjustment.
#[derive(Debug, Clone)]
pub struct AdjustmentEvent {
    pub timestamp: Instant,
    pub from_threads: usize,
    pub to_threads: usize,
    pub reason: AdjustmentReason,
    pub throughput_before: Option<f64>,
    pub throughput_after: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustmentReason {
    Initial,
    Improvement,
    Degradation,
    Exploration,
    Cooldown,
}

impl std::fmt::Display for AdjustmentReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdjustmentReason::Initial => write!(f, "initial"),
            AdjustmentReason::Improvement => write!(f, "perf-up"),
            AdjustmentReason::Degradation => write!(f, "perf-down"),
            AdjustmentReason::Exploration => write!(f, "explore"),
            AdjustmentReason::Cooldown => write!(f, "cooldown"),
        }
    }
}

/// Adaptive thread manager that adjusts thread count based on performance.
pub struct AdaptiveThreadManager {
    config: AdaptiveConfig,
    state: Mutex<AdaptiveState>,
    /// Atomic for fast reads without lock.
    thread_count: AtomicUsize,
    /// Whether adaptation is enabled.
    enabled: AtomicBool,
}

impl AdaptiveThreadManager {
    /// Create a new adaptive thread manager.
    pub fn new(config: AdaptiveConfig) -> Self {
        // Start at physical core count (optimal for CPU-bound workloads).
        // The adaptive algorithm will explore nearby values if beneficial.
        let initial_threads = config.max_threads.max(config.min_threads);

        let state = AdaptiveState {
            current_threads: initial_threads,
            current_throughput_ewma: None,
            best_throughput: None,
            samples_at_current: 0,
            last_adjustment: None,
            jobs_completed: 0,
            recent_timings: Vec::with_capacity(32),
            last_direction: 0,
            consecutive_same_direction: 0,
            adjustment_log: Vec::new(),
        };

        // Log initial state
        let mut state = state;
        state.adjustment_log.push(AdjustmentEvent {
            timestamp: Instant::now(),
            from_threads: initial_threads,
            to_threads: initial_threads,
            reason: AdjustmentReason::Initial,
            throughput_before: None,
            throughput_after: None,
        });

        Self {
            thread_count: AtomicUsize::new(initial_threads),
            config,
            state: Mutex::new(state),
            enabled: AtomicBool::new(true),
        }
    }

    /// Create with a fixed thread count (no adaptation).
    pub fn fixed(threads: usize) -> Self {
        let config = AdaptiveConfig {
            min_threads: threads,
            max_threads: threads,
            ..Default::default()
        };
        let mgr = Self::new(config);
        mgr.enabled.store(false, Ordering::Relaxed);
        mgr.thread_count.store(threads, Ordering::Relaxed);
        mgr
    }

    /// Get the current recommended thread count.
    pub fn current_threads(&self) -> usize {
        self.thread_count.load(Ordering::Relaxed)
    }

    /// Check if adaptation is enabled.
    pub fn is_adaptive(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get initial thread count for display.
    pub fn initial_threads(&self) -> usize {
        self.state.lock().adjustment_log.first()
            .map(|e| e.to_threads)
            .unwrap_or_else(|| self.current_threads())
    }

    /// Record a completed job and potentially adjust thread count.
    /// Returns Some(new_count) if thread count changed.
    pub fn record_job(&self, duration: Duration) -> Option<AdjustmentEvent> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        let mut state = self.state.lock();
        let current_threads = state.current_threads;

        // Record timing
        state.recent_timings.push(JobTiming {
            duration,
            thread_count: current_threads,
        });

        // Keep recent timings bounded
        if state.recent_timings.len() > 64 {
            state.recent_timings.drain(0..32);
        }

        state.jobs_completed += 1;
        state.samples_at_current += 1;

        // Calculate throughput (jobs/sec, accounting for parallelism)
        // We estimate effective throughput as thread_count / avg_duration
        let throughput = current_threads as f64 / duration.as_secs_f64();

        // Update EWMA
        state.current_throughput_ewma = Some(match state.current_throughput_ewma {
            Some(ewma) => {
                self.config.ewma_alpha * throughput + (1.0 - self.config.ewma_alpha) * ewma
            }
            None => throughput,
        });

        // Check if we should consider adjustment
        if state.jobs_completed < self.config.warmup_jobs {
            return None;
        }

        if state.samples_at_current < self.config.min_samples_before_adjust {
            return None;
        }

        if let Some(last_adj) = state.last_adjustment {
            if last_adj.elapsed() < self.config.adjustment_cooldown {
                return None;
            }
        }

        // Decide on adjustment
        let current_ewma = state.current_throughput_ewma.unwrap();
        let adjustment = self.decide_adjustment(&state, current_ewma);

        if let Some((new_threads, reason)) = adjustment {
            if new_threads != current_threads {
                let event = AdjustmentEvent {
                    timestamp: Instant::now(),
                    from_threads: current_threads,
                    to_threads: new_threads,
                    reason,
                    throughput_before: Some(current_ewma),
                    throughput_after: None,
                };

                // Update direction tracking
                let direction = if new_threads > current_threads { 1 } else { -1 };
                if direction == state.last_direction {
                    state.consecutive_same_direction += 1;
                } else {
                    state.consecutive_same_direction = 1;
                    state.last_direction = direction;
                }

                // Update best if current is best
                if let Some((best_tp, _)) = state.best_throughput {
                    if current_ewma > best_tp {
                        state.best_throughput = Some((current_ewma, current_threads));
                    }
                } else {
                    state.best_throughput = Some((current_ewma, current_threads));
                }

                // Apply adjustment
                state.current_threads = new_threads;
                state.current_throughput_ewma = None; // Reset for new level
                state.samples_at_current = 0;
                state.last_adjustment = Some(Instant::now());
                state.adjustment_log.push(event.clone());

                self.thread_count.store(new_threads, Ordering::Relaxed);

                return Some(event);
            }
        }

        None
    }

    /// Decide whether and how to adjust thread count.
    fn decide_adjustment(
        &self,
        state: &AdaptiveState,
        current_ewma: f64,
    ) -> Option<(usize, AdjustmentReason)> {
        let current = state.current_threads;

        // Compare to best known throughput
        if let Some((best_tp, best_threads)) = state.best_throughput {
            let ratio = current_ewma / best_tp;

            // If we're significantly worse than best, move towards best
            if ratio < (1.0 - self.config.degradation_threshold) {
                let target = if current > best_threads {
                    (current - 1).max(self.config.min_threads)
                } else {
                    (current + 1).min(self.config.max_threads)
                };
                if target != current {
                    return Some((target, AdjustmentReason::Degradation));
                }
            }

            // If we're at or better than best, explore further in same direction
            if ratio >= (1.0 + self.config.improvement_threshold) {
                // We improved! Continue in same direction
                let target = match state.last_direction {
                    1 => (current + 1).min(self.config.max_threads),
                    -1 => (current - 1).max(self.config.min_threads),
                    _ => {
                        // Try increasing first
                        if current < self.config.max_threads {
                            current + 1
                        } else if current > self.config.min_threads {
                            current - 1
                        } else {
                            current
                        }
                    }
                };
                if target != current {
                    return Some((target, AdjustmentReason::Improvement));
                }
            }
        }

        // Exploration: occasionally try a different thread count
        // This helps escape local optima and handles changing system load
        if state.samples_at_current >= self.config.min_samples_before_adjust * 2 {
            // Haven't changed in a while, try exploring
            let explore_up = current < self.config.max_threads
                && (state.consecutive_same_direction < 2 || state.last_direction >= 0);
            let explore_down = current > self.config.min_threads
                && (state.consecutive_same_direction < 2 || state.last_direction <= 0);

            if explore_up {
                return Some((current + 1, AdjustmentReason::Exploration));
            } else if explore_down {
                return Some((current - 1, AdjustmentReason::Exploration));
            }
        }

        None
    }

    /// Get a summary of adjustments made.
    pub fn get_summary(&self) -> AdaptiveSummary {
        let state = self.state.lock();
        AdaptiveSummary {
            final_threads: state.current_threads,
            initial_threads: state.adjustment_log.first()
                .map(|e| e.to_threads)
                .unwrap_or(state.current_threads),
            total_adjustments: state.adjustment_log.len().saturating_sub(1),
            best_throughput: state.best_throughput,
            jobs_completed: state.jobs_completed,
            adjustment_log: state.adjustment_log.clone(),
        }
    }
}

/// Summary of adaptive thread management.
#[derive(Debug)]
pub struct AdaptiveSummary {
    pub initial_threads: usize,
    pub final_threads: usize,
    pub total_adjustments: usize,
    pub best_throughput: Option<(f64, usize)>,
    pub jobs_completed: usize,
    pub adjustment_log: Vec<AdjustmentEvent>,
}

impl AdaptiveSummary {
    /// Format a brief log of adjustments.
    pub fn format_log(&self) -> String {
        if self.adjustment_log.len() <= 1 {
            return String::from("(no adjustments)");
        }

        let mut lines = Vec::new();
        let start = self.adjustment_log.first().unwrap().timestamp;

        for event in self.adjustment_log.iter().skip(1) {
            let elapsed = event.timestamp.duration_since(start);
            let tp_str = event.throughput_before
                .map(|tp| format!("{:.1} jobs/s", tp))
                .unwrap_or_else(|| String::from("?"));
            lines.push(format!(
                "  [{:>5.1}s] {} → {} ({}; {})",
                elapsed.as_secs_f64(),
                event.from_threads,
                event.to_threads,
                event.reason,
                tp_str,
            ));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_mode() {
        let mgr = AdaptiveThreadManager::fixed(4);
        assert!(!mgr.is_adaptive());
        assert_eq!(mgr.current_threads(), 4);

        // Recording jobs should not change thread count
        for _ in 0..20 {
            let result = mgr.record_job(Duration::from_millis(100));
            assert!(result.is_none());
        }
        assert_eq!(mgr.current_threads(), 4);
    }

    #[test]
    fn test_adaptive_warmup() {
        let config = AdaptiveConfig {
            warmup_jobs: 5,
            ..Default::default()
        };
        let mgr = AdaptiveThreadManager::new(config);

        // During warmup, no adjustments
        for _ in 0..4 {
            let result = mgr.record_job(Duration::from_millis(100));
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_ewma_calculation() {
        let config = AdaptiveConfig {
            ewma_alpha: 0.5,
            warmup_jobs: 0,
            min_samples_before_adjust: 100, // High to prevent adjustment
            ..Default::default()
        };
        let mgr = AdaptiveThreadManager::new(config);

        // Record several jobs
        for _ in 0..10 {
            mgr.record_job(Duration::from_millis(100));
        }

        let state = mgr.state.lock();
        assert!(state.current_throughput_ewma.is_some());
    }
}
