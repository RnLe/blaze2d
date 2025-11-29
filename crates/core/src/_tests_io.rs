//! Tests for the io module.

#![cfg(test)]

use super::bandstructure::BandStructureJob;
use super::geometry::{BasisAtom, Geometry2D};
use super::grid::Grid2D;
use super::io::{JobConfig, PathPreset, PathSpec};
use super::lattice::Lattice2D;
use super::polarization::Polarization;

fn sample_geometry() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 12.0,
        atoms: vec![BasisAtom {
            pos: [0.0, 0.0],
            radius: 0.2,
            eps_inside: 1.0,
        }],
    }
}

#[test]
fn path_preset_serde_accepts_lowercase_strings() {
    let json = "\"square\"";
    let preset: PathPreset = serde_json::from_str(json).expect("square should deserialize");
    assert!(matches!(preset, PathPreset::Square));
    let upper = serde_json::to_string(&PathPreset::Hexagonal).expect("serialize");
    assert_eq!(upper, "\"hexagonal\"");
}

#[test]
fn path_spec_defaults_segments_per_leg() {
    let json = r#"{"preset": "hexagonal"}"#;
    let spec: PathSpec = serde_json::from_str(json).expect("spec");
    assert_eq!(spec.segments_per_leg, 8);
}

#[test]
fn job_config_with_explicit_k_path_converts_directly() {
    let config = JobConfig {
        geometry: sample_geometry(),
        grid: Grid2D::new(2, 2, 1.0, 1.0),
        polarization: Polarization::TM,
        k_path: vec![[0.0, 0.0], [0.5, 0.0]],
        path: None,
        eigensolver: Default::default(),
        dielectric: Default::default(),
    };
    let job: BandStructureJob = config.into();
    assert_eq!(job.k_path.len(), 2);
    assert!(matches!(job.pol, Polarization::TM));
}

#[test]
fn job_config_uses_path_preset_when_k_path_missing() {
    let config = JobConfig {
        geometry: sample_geometry(),
        grid: Grid2D::new(2, 2, 1.0, 1.0),
        polarization: Polarization::TE,
        k_path: Vec::new(),
        path: Some(PathSpec {
            preset: PathPreset::Square,
            segments_per_leg: 2,
        }),
        eigensolver: Default::default(),
        dielectric: Default::default(),
    };
    let job: BandStructureJob = config.into();
    assert_eq!(job.k_path.first().copied(), Some([0.0, 0.0]));
    assert_eq!(job.k_path.last().copied(), Some([0.0, 0.0]));
    assert!(job.k_path.len() > 2);
}

#[test]
#[should_panic(expected = "JobConfig requires either an explicit k_path or a path preset")]
fn job_config_panics_without_k_path_or_preset() {
    let config = JobConfig {
        geometry: sample_geometry(),
        grid: Grid2D::new(2, 2, 1.0, 1.0),
        polarization: Polarization::TE,
        k_path: Vec::new(),
        path: None,
        eigensolver: Default::default(),
        dielectric: Default::default(),
    };
    let _: BandStructureJob = config.into();
}
