#![cfg(test)]

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use serde_json::json;

use crate::reference::{ReferenceLoadError, load_reference_dataset};

static NEXT_FILE_ID: AtomicUsize = AtomicUsize::new(0);

fn write_temp_reference_file(test_name: &str, contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
    path.push(format!(
        "mpb2d_core_reference_{test_name}_{}_{}.json",
        std::process::id(),
        unique
    ));
    fs::write(&path, contents).expect("failed to write reference fixture");
    path
}

fn remove_if_exists(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

#[test]
fn load_reference_dataset_reads_complete_payload() {
    let path = write_temp_reference_file(
        "complete",
        &json!({
            "metadata": {"source": "mpb", "notes": "test"},
            "k_path": [
                {"kx": 0.0, "ky": 0.0, "distance": 0.0},
                {"kx": 0.5, "ky": 0.25, "distance": 1.0}
            ],
            "k_nodes": [
                {"label": "Gamma", "index": 0},
                {"label": "X", "index": 1}
            ],
            "bands": [
                [0.10, 0.20],
                [0.30, 0.40]
            ]
        })
        .to_string(),
    );

    let dataset = load_reference_dataset(&path).expect("dataset should load");
    remove_if_exists(&path);

    assert_eq!(
        dataset.metadata.get("source").and_then(|v| v.as_str()),
        Some("mpb")
    );
    assert_eq!(dataset.k_path.len(), 2);
    assert_eq!(dataset.k_nodes.len(), 2);
    assert_eq!(dataset.bands.len(), 2);
    assert_eq!(dataset.bands[1][1], 0.40);
}

#[test]
fn load_reference_dataset_defaults_missing_k_nodes() {
    let path = write_temp_reference_file(
        "missing_nodes",
        &json!({
            "metadata": {},
            "k_path": [
                {"kx": 0.0, "ky": 0.0, "distance": 0.0}
            ],
            "bands": [
                [1.0]
            ]
        })
        .to_string(),
    );

    let dataset = load_reference_dataset(&path).expect("dataset should load without nodes");
    remove_if_exists(&path);

    assert!(
        dataset.k_nodes.is_empty(),
        "k_nodes should default to empty vec"
    );
    assert_eq!(dataset.k_path[0].distance, 0.0);
    assert_eq!(dataset.bands[0][0], 1.0);
}

#[test]
fn load_reference_dataset_returns_io_error_for_missing_file() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "mpb2d_core_reference_missing_{}_{}.json",
        std::process::id(),
        NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    remove_if_exists(&path);

    let err = load_reference_dataset(&path).expect_err("expected IO error");
    assert!(matches!(err, ReferenceLoadError::Io { .. }));
}

#[test]
fn load_reference_dataset_returns_parse_error_for_invalid_json() {
    let path = write_temp_reference_file("invalid", "not valid json");

    let err = load_reference_dataset(&path).expect_err("expected parse error");
    remove_if_exists(&path);

    assert!(matches!(err, ReferenceLoadError::Parse { .. }));
}
