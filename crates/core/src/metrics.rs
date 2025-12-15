//! Lightweight metrics recorder (JSONL) for heavy pipeline stages.

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::polarization::Polarization;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub output: Option<PathBuf>,
    #[serde(default)]
    pub format: MetricsFormat,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output: None,
            format: MetricsFormat::JsonLines,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricsFormat {
    JsonLines,
}

impl Default for MetricsFormat {
    fn default() -> Self {
        Self::JsonLines
    }
}

pub struct MetricsRecorder {
    writer: Mutex<File>,
    format: MetricsFormat,
}

impl MetricsRecorder {
    pub fn new(path: &Path, format: MetricsFormat) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        Ok(Self {
            writer: Mutex::new(file),
            format,
        })
    }

    pub fn emit(&self, event: MetricsEvent<'_>) {
        if let Err(err) = self.write_event(event) {
            eprintln!("[metrics] failed to write event: {err}");
        }
    }

    fn write_event(&self, event: MetricsEvent<'_>) -> io::Result<()> {
        match self.format {
            MetricsFormat::JsonLines => {
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
    }
}

#[derive(Serialize)]
struct EventEnvelope<'a> {
    timestamp_ms: f64,
    #[serde(flatten)]
    event: MetricsEvent<'a>,
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum MetricsEvent<'a> {
    PipelineStart {
        backend: &'a str,
        grid_nx: usize,
        grid_ny: usize,
        polarization: Polarization,
        n_bands: usize,
        max_iter: usize,
        tol: f64,
        k_points: usize,
        atoms: usize,
    },
    DielectricSample {
        duration_ms: f64,
        grid_points: usize,
    },
    FftWorkspace {
        duration_ms: f64,
        grid_nx: usize,
        grid_ny: usize,
    },
    KPointSolve {
        k_index: usize,
        kx: f64,
        ky: f64,
        distance: f64,
        polarization: Polarization,
        iterations: usize,
        bands: usize,
        duration_ms: f64,
        max_residual: f64,
        avg_residual: f64,
        max_relative_residual: f64,
        avg_relative_residual: f64,
        max_mass_error: f64,
        duplicate_modes_skipped: usize,
        negative_modes_skipped: usize,
        freq_tolerance: f64,
        gamma_deflated: bool,
        seed_count: usize,
        warm_start_hits: usize,
        deflation_workspace: usize,
        symmetry_reflections: usize,
        symmetry_reflections_skipped: usize,
        preconditioner_new_directions: usize,
        preconditioner_trials: usize,
        preconditioner_avg_before: f64,
        preconditioner_avg_after: f64,
    },
    EigenIteration {
        k_index: usize,
        iteration: usize,
        max_residual: f64,
        avg_residual: f64,
        max_relative_residual: f64,
        avg_relative_residual: f64,
        block_size: usize,
        new_directions: usize,
        preconditioner_trials: usize,
        preconditioner_avg_before: f64,
        preconditioner_avg_after: f64,
        preconditioner_accepted: usize,
    },
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

impl MetricsConfig {
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
        MetricsRecorder::new(path, self.format).map(Some)
    }
}
