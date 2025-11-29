//! Lightweight metrics recording for performance analysis.
//!
//! This module provides optional JSONL (JSON Lines) output for tracking
//! the performance of band-structure calculations. Each significant event
//! (k-point solve, etc.) is recorded as a separate JSON object.
//!
//! # Usage
//!
//! Metrics are optional and must be explicitly enabled in the configuration:
//!
//! ```toml
//! [metrics]
//! enabled = true
//! output = "metrics.jsonl"
//! ```

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::polarization::Polarization;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for metrics recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Whether to enable metrics recording.
    pub enabled: bool,
    /// Output file path (JSONL format).
    pub output: Option<PathBuf>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output: None,
        }
    }
}

impl MetricsConfig {
    /// Build a metrics recorder from this configuration.
    ///
    /// Returns `None` if metrics are disabled.
    pub fn build_recorder(&self) -> io::Result<Option<MetricsRecorder>> {
        if !self.enabled {
            return Ok(None);
        }
        let path = self.output.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "metrics.output must be set when metrics are enabled",
            )
        })?;
        MetricsRecorder::new(path).map(Some)
    }
}

// ============================================================================
// Recorder
// ============================================================================

/// Records metrics events to a JSONL file.
pub struct MetricsRecorder {
    writer: Mutex<File>,
}

impl MetricsRecorder {
    /// Create a new metrics recorder writing to the given path.
    pub fn new(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: Mutex::new(file),
        })
    }

    /// Emit a metrics event.
    pub fn emit(&self, event: MetricsEvent) {
        if let Err(err) = self.write_event(event) {
            log::warn!("[metrics] failed to write event: {err}");
        }
    }

    fn write_event(&self, event: MetricsEvent) -> io::Result<()> {
        let envelope = EventEnvelope {
            timestamp_ms: now_millis(),
            event,
        };
        let mut guard = self.writer.lock().expect("metrics writer poisoned");
        serde_json::to_writer(&mut *guard, &envelope)?;
        guard.write_all(b"\n")?;
        guard.flush()
    }
}

// ============================================================================
// Events
// ============================================================================

#[derive(Serialize)]
struct EventEnvelope {
    timestamp_ms: f64,
    #[serde(flatten)]
    event: MetricsEvent,
}

/// A metrics event recorded during band-structure calculation.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum MetricsEvent {
    /// Start of a band-structure calculation.
    PipelineStart {
        backend: String,
        grid_nx: usize,
        grid_ny: usize,
        polarization: Polarization,
        n_bands: usize,
        k_points: usize,
    },
    /// Completion of a single k-point solve.
    KPointSolve {
        k_index: usize,
        kx: f64,
        ky: f64,
        polarization: Polarization,
        iterations: usize,
        bands: usize,
        duration_ms: f64,
        converged: bool,
        max_relative_residual: f64,
    },
    /// End of a band-structure calculation.
    PipelineDone {
        total_k: usize,
        total_iterations: usize,
        duration_ms: f64,
    },
}

fn now_millis() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}
