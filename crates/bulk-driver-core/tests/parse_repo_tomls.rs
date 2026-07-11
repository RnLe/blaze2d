//! Golden test: every example/benchmark TOML in the repository must parse
//! and validate against the current schema.

use std::path::{Path, PathBuf};

use blaze2d_bulk_driver_core::{Config, expand_jobs, parse_and_validate};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn config_tomls() -> Vec<PathBuf> {
    let root = repo_root();
    let dirs = [
        root.join("examples"),
        root.join("benchmarks"),
        root.join("web/examples"),
        root.join("crates/bulk-driver/examples"),
        root.join("crates/backend-wasm/examples"),
    ];

    let mut files = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "toml") {
                files.push(path);
            }
        }
    }
    assert!(
        files.len() >= 15,
        "expected to find the repo's config TOMLs, got {} files",
        files.len()
    );
    files.sort();
    files
}

#[test]
fn every_repo_toml_parses_and_expands() {
    let mut failures = Vec::new();

    for path in config_tomls() {
        let source = std::fs::read_to_string(&path).expect("read toml");
        match parse_and_validate(&source) {
            Ok(config) => {
                // Expansion must also succeed (it asserts on validated invariants).
                let jobs = expand_jobs(&config);
                assert!(
                    !jobs.is_empty(),
                    "{} expanded to zero jobs",
                    path.display()
                );
                assert_eq!(
                    jobs.len(),
                    config.total_jobs(),
                    "{}: expanded job count disagrees with total_jobs()",
                    path.display()
                );
            }
            Err(diags) => {
                let msgs: Vec<String> = diags.iter().map(|d| d.to_string()).collect();
                failures.push(format!("{}:\n  {}", path.display(), msgs.join("\n  ")));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "config TOMLs failed to parse:\n{}",
        failures.join("\n")
    );
}

#[test]
fn expected_job_counts() {
    let root = repo_root();
    let cases = [
        ("examples/bulk_parameter_sweep.toml", 50),
        ("examples/bulk_simple_sweep.toml", 10),
        ("examples/bulk_benchmark_100.toml", 100),
        ("crates/bulk-driver/examples/two_atom_sweep.toml", 243),
        ("crates/bulk-driver/examples/stream_config.toml", 5),
        ("benchmarks/config_a_square_tm.toml", 1),
        ("web/examples/square_lattice_tutorial.toml", 1),
    ];
    for (rel, expected) in cases {
        let source = std::fs::read_to_string(root.join(rel)).expect(rel);
        let config = Config::from_str(&source).unwrap_or_else(|e| panic!("{}: {}", rel, e));
        assert_eq!(config.total_jobs(), expected, "{}", rel);
    }
}
