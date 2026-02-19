//! Job expansion from parameter ranges.
//!
//! This module takes a `BulkConfig` and expands all parameter ranges into
//! a list of individual `ExpandedJob` configurations ready for execution.

use mpb2d_core::{
    bandstructure::BandStructureJob,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{self, PathType},
};

use crate::config::{AtomRanges, BaseAtom, BulkConfig, LatticeTypeSpec};

// ============================================================================
// Expanded Job
// ============================================================================

/// A single expanded job with all parameters resolved to concrete values.
///
/// This includes metadata about which parameter values were used, enabling
/// proper labeling in output files.
#[derive(Debug, Clone)]
pub struct ExpandedJob {
    /// Unique job index (0-based)
    pub index: usize,

    /// The band structure job to execute
    pub job: BandStructureJob,

    /// Parameter values used for this job (for output columns)
    pub params: JobParams,
}

/// Concrete parameter values for a job.
///
/// These are stored for inclusion in output files, allowing easy identification
/// of which configuration produced which results.
#[derive(Debug, Clone)]
pub struct JobParams {
    /// Background epsilon
    pub eps_bg: f64,

    /// Resolution (nx = ny)
    pub resolution: usize,

    /// Polarization
    pub polarization: Polarization,

    /// Lattice type (if from range)
    pub lattice_type: Option<String>,

    /// Per-atom parameters
    pub atoms: Vec<AtomParams>,
}

/// Parameters for a single atom.
#[derive(Debug, Clone)]
pub struct AtomParams {
    /// Atom index (0-based)
    pub index: usize,

    /// Position (fractional coordinates)
    pub pos: [f64; 2],

    /// Radius
    pub radius: f64,

    /// Epsilon inside
    pub eps_inside: f64,
}

impl JobParams {
    /// Get a flat list of (name, value) pairs for CSV headers.
    pub fn to_columns(&self) -> Vec<(&'static str, String)> {
        let mut cols = vec![
            ("eps_bg", format!("{:.6}", self.eps_bg)),
            ("resolution", self.resolution.to_string()),
            ("polarization", format!("{:?}", self.polarization)),
        ];

        if let Some(ref lt) = self.lattice_type {
            cols.push(("lattice_type", lt.clone()));
        }

        for atom in &self.atoms {
            let prefix = format!("atom{}", atom.index);
            cols.push((
                Box::leak(format!("{}_pos_x", prefix).into_boxed_str()),
                format!("{:.6}", atom.pos[0]),
            ));
            cols.push((
                Box::leak(format!("{}_pos_y", prefix).into_boxed_str()),
                format!("{:.6}", atom.pos[1]),
            ));
            cols.push((
                Box::leak(format!("{}_radius", prefix).into_boxed_str()),
                format!("{:.6}", atom.radius),
            ));
            cols.push((
                Box::leak(format!("{}_eps_inside", prefix).into_boxed_str()),
                format!("{:.6}", atom.eps_inside),
            ));
        }

        cols
    }
}

// ============================================================================
// Expansion Logic
// ============================================================================

/// Expand a bulk configuration into individual jobs.
///
/// This function iterates over all combinations of parameter ranges and
/// creates an `ExpandedJob` for each unique configuration.
pub fn expand_jobs(config: &BulkConfig) -> Vec<ExpandedJob> {
    let ranges = &config.ranges;

    // Collect all parameter axes
    let eps_bg_values = ranges
        .eps_bg
        .as_ref()
        .map(|r| r.values())
        .unwrap_or_else(|| vec![config.geometry.eps_bg]);

    let resolution_values: Vec<usize> = ranges
        .resolution
        .as_ref()
        .map(|r| r.values().into_iter().map(|v| v.round() as usize).collect())
        .unwrap_or_else(|| vec![config.grid.nx]);

    let polarization_values = ranges
        .polarization
        .clone()
        .unwrap_or_else(|| vec![config.polarization]);

    let lattice_type_values = ranges.lattice_type.clone();

    // Collect atom parameter axes
    let atom_configs = collect_atom_configs(config);

    // Calculate total combinations
    let total = eps_bg_values.len()
        * resolution_values.len()
        * polarization_values.len()
        * lattice_type_values.as_ref().map(|v| v.len()).unwrap_or(1)
        * atom_configs.iter().map(|a| a.len()).product::<usize>().max(1);

    let mut jobs = Vec::with_capacity(total);
    let mut index = 0;

    // Iterate over all combinations
    for &eps_bg in &eps_bg_values {
        for &resolution in &resolution_values {
            for &pol in &polarization_values {
                let lattice_iter: Box<dyn Iterator<Item = Option<&LatticeTypeSpec>>> =
                    if let Some(ref types) = lattice_type_values {
                        Box::new(types.iter().map(Some))
                    } else {
                        Box::new(std::iter::once(None))
                    };

                for lattice_type in lattice_iter {
                    // Iterate over atom parameter combinations
                    for atom_combo in atom_combinations(&atom_configs) {
                        let job = create_job(
                            config,
                            eps_bg,
                            resolution,
                            pol,
                            lattice_type,
                            &atom_combo,
                        );

                        let params = JobParams {
                            eps_bg,
                            resolution,
                            polarization: pol,
                            lattice_type: lattice_type.map(|lt| lt.to_string()),
                            atoms: atom_combo
                                .iter()
                                .enumerate()
                                .map(|(i, a)| AtomParams {
                                    index: i,
                                    pos: a.pos,
                                    radius: a.radius,
                                    eps_inside: a.eps_inside,
                                })
                                .collect(),
                        };

                        jobs.push(ExpandedJob {
                            index,
                            job,
                            params,
                        });
                        index += 1;
                    }
                }
            }
        }
    }

    jobs
}

/// Collect possible configurations for each atom.
fn collect_atom_configs(config: &BulkConfig) -> Vec<Vec<BaseAtom>> {
    let base_atoms = &config.geometry.atoms;
    let atom_ranges = &config.ranges.atoms;

    if base_atoms.is_empty() {
        return vec![vec![]];
    }

    base_atoms
        .iter()
        .enumerate()
        .map(|(i, base)| {
            let ranges = atom_ranges.get(i);
            expand_atom_params(base, ranges)
        })
        .collect()
}

/// Expand a single atom's parameters based on ranges.
fn expand_atom_params(base: &BaseAtom, ranges: Option<&AtomRanges>) -> Vec<BaseAtom> {
    let ranges = match ranges {
        Some(r) => r,
        None => return vec![base.clone()],
    };

    let radius_values = ranges
        .radius
        .as_ref()
        .map(|r| r.values())
        .unwrap_or_else(|| vec![base.radius]);

    let pos_x_values = ranges
        .pos_x
        .as_ref()
        .map(|r| r.values())
        .unwrap_or_else(|| vec![base.pos[0]]);

    let pos_y_values = ranges
        .pos_y
        .as_ref()
        .map(|r| r.values())
        .unwrap_or_else(|| vec![base.pos[1]]);

    let eps_values = ranges
        .eps_inside
        .as_ref()
        .map(|r| r.values())
        .unwrap_or_else(|| vec![base.eps_inside]);

    let mut atoms = Vec::new();

    for &radius in &radius_values {
        for &pos_x in &pos_x_values {
            for &pos_y in &pos_y_values {
                for &eps in &eps_values {
                    atoms.push(BaseAtom {
                        pos: [pos_x, pos_y],
                        radius,
                        eps_inside: eps,
                    });
                }
            }
        }
    }

    atoms
}

/// Generate all combinations of atom configurations.
fn atom_combinations(atom_configs: &[Vec<BaseAtom>]) -> Vec<Vec<BaseAtom>> {
    if atom_configs.is_empty() {
        return vec![vec![]];
    }

    let mut result = vec![vec![]];

    for configs in atom_configs {
        let mut new_result = Vec::new();
        for existing in &result {
            for config in configs {
                let mut combo = existing.clone();
                combo.push(config.clone());
                new_result.push(combo);
            }
        }
        result = new_result;
    }

    result
}

/// Create a BandStructureJob from resolved parameters.
fn create_job(
    config: &BulkConfig,
    eps_bg: f64,
    resolution: usize,
    polarization: Polarization,
    lattice_type: Option<&LatticeTypeSpec>,
    atoms: &[BaseAtom],
) -> BandStructureJob {
    // Build lattice
    let lattice = build_lattice(&config.geometry.lattice, lattice_type);

    // Build atoms
    let basis_atoms: Vec<BasisAtom> = atoms
        .iter()
        .map(|a| BasisAtom {
            pos: a.pos,
            radius: a.radius,
            eps_inside: a.eps_inside,
        })
        .collect();

    // Build geometry
    let geometry = Geometry2D {
        lattice,
        eps_bg,
        atoms: basis_atoms,
    };

    // Build grid
    let grid = Grid2D::new(resolution, resolution, config.grid.lx, config.grid.ly);

    // Build k-path
    let k_path = if !config.k_path.is_empty() {
        config.k_path.clone()
    } else if let Some(ref spec) = config.path {
        // Use path preset with appropriate mapping for lattice type
        let path_type = match spec.preset {
            mpb2d_core::io::PathPreset::Square => PathType::Square,
            mpb2d_core::io::PathPreset::Hexagonal | mpb2d_core::io::PathPreset::Triangular => {
                PathType::Hexagonal
            }
            mpb2d_core::io::PathPreset::Rectangular => {
                PathType::Custom(mpb2d_core::brillouin::BrillouinPath::Rectangular.raw_k_points())
            }
        };
        symmetry::standard_path(&geometry.lattice, path_type, spec.segments_per_leg)
    } else {
        // Default: use square path
        symmetry::standard_path(&geometry.lattice, PathType::Square, 12)
    };

    BandStructureJob {
        geom: geometry,
        grid,
        pol: polarization,
        k_path,
        eigensolver: config.eigensolver.clone(),
        dielectric: config.dielectric.clone(),
    }
}

/// Build a Lattice2D from base configuration and optional type override.
fn build_lattice(
    base: &crate::config::BaseLattice,
    type_override: Option<&LatticeTypeSpec>,
) -> Lattice2D {
    // If explicit vectors are given, use them
    if let (Some(a1), Some(a2)) = (base.a1, base.a2) {
        return Lattice2D::oblique(a1, a2);
    }

    // Determine lattice type
    let lattice_type = type_override
        .or(base.lattice_type.as_ref())
        .unwrap_or(&LatticeTypeSpec::Square);

    match lattice_type {
        LatticeTypeSpec::Square => Lattice2D::square(base.a),
        LatticeTypeSpec::Rectangular => {
            Lattice2D::rectangular(base.a, base.b.unwrap_or(base.a * 1.5))
        }
        LatticeTypeSpec::Triangular | LatticeTypeSpec::Hexagonal => Lattice2D::hexagonal(base.a),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BaseGeometry, BulkSection, OutputConfig, ParameterRange, RangeSpec};
    use mpb2d_core::{dielectric::DielectricOptions, eigensolver::EigensolverConfig, io::PathSpec};

    fn make_test_config() -> BulkConfig {
        BulkConfig {
            bulk: BulkSection::default(),
            geometry: BaseGeometry {
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
            },
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
    }

    #[test]
    fn expand_eps_range() {
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
    fn expand_multiple_ranges() {
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
}
