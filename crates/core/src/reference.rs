//! Reference band-structure data loaders (e.g., MPB snapshots).

use std::{fs, path::Path};

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Deserialize, Clone)]
pub struct ReferenceDataset {
    pub metadata: Value,
    pub k_path: Vec<KPointSample>,
    #[serde(default)]
    pub k_nodes: Vec<KNode>,
    pub bands: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct KPointSample {
    pub kx: f64,
    pub ky: f64,
    pub distance: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct KNode {
    pub label: String,
    pub index: usize,
}

#[derive(Debug, Error)]
pub enum ReferenceLoadError {
    #[error("failed to read reference file {path}: {source}")]
    Io {
        #[source]
        source: std::io::Error,
        path: String,
    },
    #[error("failed to parse reference JSON {path}: {source}")]
    Parse {
        #[source]
        source: serde_json::Error,
        path: String,
    },
}

pub fn load_reference_dataset<P: AsRef<Path>>(
    path: P,
) -> Result<ReferenceDataset, ReferenceLoadError> {
    let path_ref = path.as_ref();
    let path_display = path_ref.display().to_string();
    let contents = fs::read_to_string(path_ref).map_err(|source| ReferenceLoadError::Io {
        source,
        path: path_display.clone(),
    })?;
    serde_json::from_str(&contents).map_err(|source| ReferenceLoadError::Parse {
        source,
        path: path_display,
    })
}
