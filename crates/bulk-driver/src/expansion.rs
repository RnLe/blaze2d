//! Job expansion from parameter ranges.
//!
//! This module re-exports the core expansion logic from `mpb2d_bulk_driver_core`
//! and provides any native-specific extensions.
//!
//! # Expansion Modes
//!
//! ## Ordered Sweeps (New Format)
//!
//! When using `[[sweeps]]` arrays, jobs are expanded in **TOML order**.
//! The first sweep is the outermost loop, the last is the innermost.
//!
//! ## Legacy Ranges
//!
//! The old `[ranges]` format uses a hardcoded loop order:
//! eps_bg → resolution → polarization → lattice_type → atoms
//!
//! # Solver Types
//!
//! - **Maxwell**: Expands over eps_bg, resolution, polarization, lattice_type, atoms.
//! - **EA**: Currently produces a single job (parameter sweeps TBD).

// Re-export everything from core
pub use mpb2d_bulk_driver_core::expansion::{
    expand_jobs, AtomParams, EAJobSpec, ExpandedJob, ExpandedJobType, JobParams,
};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        BaseAtom, BaseGeometry, BulkSection, DefaultsConfig, EAConfig, LatticeTypeSpec,
        OutputConfig, ParameterRange, RangeSpec, SolverSection, SolverType, SweepSpec,
    };
    use mpb2d_core::grid::Grid2D;
    use mpb2d_core::polarization::Polarization;
    use mpb2d_core::{dielectric::DielectricOptions, eigensolver::EigensolverConfig, io::PathSpec};

    fn make_test_config() -> crate::config::BulkConfig {
        crate::config::BulkConfig {
            bulk: BulkSection::default(),
            solver: SolverSection::default(),
            ea: EAConfig::default(),
            sweeps: vec![],
            defaults: DefaultsConfig::default(),
            geometry: Some(BaseGeometry {
                eps_bg: 12.0,
                lattice: crate::config::BaseLattice {
                    a1: None,
                    a2: None,
                    lattice_type: Some(LatticeTypeSpec::Square),
                    a: 1.0,
                    b: None,
                    alpha: None,
                },
                atoms: vec![BaseAtom {
                    pos: [0.5, 0.5],
                    radius: 0.3,
                    eps_inside: 1.0,
                }],
            }),
            grid: Grid2D::new(32, 32, 1.0, 1.0),
            polarization: Polarization::TM,
            path: Some(PathSpec {
                preset: mpb2d_core::io::PathPreset::Square,
                segments_per_leg: 12,
            }),
            k_path: vec![],
            eigensolver: EigensolverConfig::default(),
            dielectric: DielectricOptions::default(),
            ranges: ParameterRange::default(),
            output: OutputConfig::default(),
        }
    }

    #[test]
    fn expand_single_job() {
        let config = make_test_config();
        let jobs = expand_jobs(&config);
        assert_eq!(jobs.len(), 1);
        assert!(matches!(jobs[0].job_type, ExpandedJobType::Maxwell(_)));
    }

    #[test]
    fn expand_eps_range_legacy() {
        let mut config = make_test_config();
        config.ranges.eps_bg = Some(RangeSpec {
            min: 10.0,
            max: 12.0,
            step: 1.0,
        });
        let jobs = expand_jobs(&config);
        assert_eq!(jobs.len(), 3);
    }

    #[test]
    fn expand_multiple_ranges_legacy() {
        let mut config = make_test_config();
        config.ranges.eps_bg = Some(RangeSpec {
            min: 10.0,
            max: 12.0,
            step: 1.0,
        });
        config.ranges.polarization = Some(vec![Polarization::TM, Polarization::TE]);
        let jobs = expand_jobs(&config);
        assert_eq!(jobs.len(), 6); // 3 eps * 2 pol
    }

    #[test]
    fn expand_ordered_sweeps() {
        let mut config = make_test_config();

        // Define sweeps: radius (outer) -> eps_bg (inner)
        config.sweeps = vec![
            SweepSpec {
                parameter: "atom0.radius".to_string(),
                min: Some(0.2),
                max: Some(0.3),
                step: Some(0.1),
                values: None,
            },
            SweepSpec {
                parameter: "eps_bg".to_string(),
                min: Some(10.0),
                max: Some(12.0),
                step: Some(1.0),
                values: None,
            },
        ];

        let jobs = expand_jobs(&config);

        // Should have 2 radius * 3 eps = 6 jobs
        assert_eq!(jobs.len(), 6);

        // Verify order: radius is outer loop, eps is inner
        let eps = 1e-9;

        assert!((jobs[0].params.atoms[0].radius - 0.2).abs() < eps);
        assert!((jobs[0].params.eps_bg - 10.0).abs() < eps);

        assert!((jobs[3].params.atoms[0].radius - 0.3).abs() < eps);
        assert!((jobs[3].params.eps_bg - 10.0).abs() < eps);
    }

    #[test]
    fn expand_ea_single_job() {
        use std::path::PathBuf;

        let config = crate::config::BulkConfig {
            bulk: BulkSection::default(),
            solver: SolverSection {
                solver_type: SolverType::EA,
            },
            ea: EAConfig {
                potential: Some(PathBuf::from("test_V.bin")),
                mass_inv: Some(PathBuf::from("test_M.bin")),
                vg: None,
                eta: 0.5,
                domain_size: [10.0, 10.0],
                periodic: true,
            },
            sweeps: vec![],
            defaults: DefaultsConfig::default(),
            geometry: None,
            grid: Grid2D::new(64, 64, 10.0, 10.0),
            polarization: Polarization::TM,
            path: None,
            k_path: vec![],
            eigensolver: EigensolverConfig {
                n_bands: 8,
                tol: 1e-8,
                max_iter: 200,
                ..Default::default()
            },
            dielectric: DielectricOptions::default(),
            ranges: ParameterRange::default(),
            output: OutputConfig::default(),
        };

        let jobs = expand_jobs(&config);
        assert_eq!(jobs.len(), 1);

        match &jobs[0].job_type {
            ExpandedJobType::EA(spec) => {
                assert_eq!(spec.eta, 0.5);
                assert_eq!(spec.domain_size, [10.0, 10.0]);
                assert_eq!(spec.solve_config.n_bands, 8);
            }
            _ => panic!("expected EA job type"),
        }
    }
}
