//! Config serialization helpers.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{
    bandstructure::{BandStructureJob, InspectionOptions},
    dielectric::DielectricOptions,
    eigensolver::EigenOptions,
    geometry::Geometry2D,
    grid::Grid2D,
    metrics::MetricsConfig,
    polarization::Polarization,
    symmetry::{self, PathType},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathPreset {
    Square,
    Hexagonal,
}

impl From<PathPreset> for PathType {
    fn from(value: PathPreset) -> Self {
        match value {
            PathPreset::Square => PathType::Square,
            PathPreset::Hexagonal => PathType::Hexagonal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSpec {
    pub preset: PathPreset,
    #[serde(default = "default_segments_per_leg")]
    pub segments_per_leg: usize,
}

fn default_segments_per_leg() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub geometry: Geometry2D,
    pub grid: Grid2D,
    pub polarization: Polarization,
    #[serde(default)]
    pub k_path: Vec<[f64; 2]>,
    #[serde(default)]
    pub path: Option<PathSpec>,
    #[serde(default)]
    pub eigensolver: EigenOptions,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub inspection: InspectionConfig,
    #[serde(default)]
    pub dielectric: DielectricOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InspectionConfig {
    pub output_dir: Option<PathBuf>,
    pub dump_eps_real: bool,
    pub dump_eps_fourier: bool,
    pub dump_fft_workspace_raw: bool,
    pub dump_fft_workspace_report: bool,
    pub operator: OperatorInspectionConfig,
}

impl Default for InspectionConfig {
    fn default() -> Self {
        Self {
            output_dir: None,
            dump_eps_real: false,
            dump_eps_fourier: false,
            dump_fft_workspace_raw: false,
            dump_fft_workspace_report: false,
            operator: OperatorInspectionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OperatorInspectionConfig {
    pub dump_snapshots: bool,
    pub dump_iteration_traces: bool,
    pub snapshot_k_limit: usize,
    pub snapshot_mode_limit: usize,
}

impl Default for OperatorInspectionConfig {
    fn default() -> Self {
        Self {
            dump_snapshots: false,
            dump_iteration_traces: false,
            snapshot_k_limit: 1,
            snapshot_mode_limit: 2,
        }
    }
}

impl InspectionConfig {
    pub fn enable_with_dir(&mut self, dir: PathBuf) {
        self.output_dir = Some(dir);
        if !self.dump_eps_real {
            self.dump_eps_real = true;
        }
        if !self.dump_eps_fourier {
            self.dump_eps_fourier = true;
        }
        if !self.dump_fft_workspace_raw {
            self.dump_fft_workspace_raw = true;
        }
        if !self.dump_fft_workspace_report {
            self.dump_fft_workspace_report = true;
        }
        if !self.operator.dump_snapshots {
            self.operator.dump_snapshots = true;
        }
        if !self.operator.dump_iteration_traces {
            self.operator.dump_iteration_traces = true;
        }
    }
}

impl From<JobConfig> for BandStructureJob {
    fn from(value: JobConfig) -> Self {
        let mut k_path = value.k_path;
        if k_path.is_empty() {
            if let Some(spec) = &value.path {
                k_path = symmetry::standard_path(
                    &value.geometry.lattice,
                    spec.preset.clone().into(),
                    spec.segments_per_leg,
                );
            }
        }
        assert!(
            !k_path.is_empty(),
            "JobConfig requires either an explicit k_path or a path preset"
        );
        let mut eigensolver = value.eigensolver;
        eigensolver.enforce_recommended_defaults();
        eigensolver
            .symmetry
            .resolve_with_lattice(&value.geometry.lattice);
        BandStructureJob {
            geom: value.geometry,
            grid: value.grid,
            pol: value.polarization,
            k_path,
            eigensolver,
            dielectric: value.dielectric,
            inspection: value.inspection.into(),
        }
    }
}

impl From<InspectionConfig> for InspectionOptions {
    fn from(value: InspectionConfig) -> Self {
        InspectionOptions {
            output_dir: value.output_dir,
            dump_eps_real: value.dump_eps_real,
            dump_eps_fourier: value.dump_eps_fourier,
            dump_fft_workspace_raw: value.dump_fft_workspace_raw,
            dump_fft_workspace_report: value.dump_fft_workspace_report,
            operator: value.operator.into(),
        }
    }
}

impl From<OperatorInspectionConfig> for crate::bandstructure::OperatorInspectionOptions {
    fn from(value: OperatorInspectionConfig) -> Self {
        crate::bandstructure::OperatorInspectionOptions {
            dump_snapshots: value.dump_snapshots,
            dump_iteration_traces: value.dump_iteration_traces,
            snapshot_k_limit: value.snapshot_k_limit,
            snapshot_mode_limit: value.snapshot_mode_limit,
        }
    }
}
