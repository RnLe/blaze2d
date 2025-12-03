//! Job expansion from parameter ranges.
//!
//! This module takes a `BulkConfig` and expands all parameter ranges into
//! a list of individual `ExpandedJob` configurations ready for execution.
//!
//! # Expansion Modes
//!
//! ## Ordered Sweeps (New Format)
//!
//! When using `[[sweeps]]` arrays, jobs are expanded in **TOML order**.
//! The first sweep is the outermost loop, the last is the innermost.
//!
//! ```text
//! [[sweeps]] parameter = "radius"    # outer loop (changes slowest)
//! [[sweeps]] parameter = "eps_bg"    # inner loop (changes fastest)
//!
//! Result order:
//!   job 0: radius=0.2, eps=10
//!   job 1: radius=0.2, eps=11
//!   job 2: radius=0.2, eps=12
//!   job 3: radius=0.3, eps=10
//!   job 4: radius=0.3, eps=11
//!   ...
//! ```
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

use std::collections::HashMap;

use mpb2d_core::{
    bandstructure::BandStructureJob,
    drivers::single_solve::SingleSolveJob,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{self, PathType},
};

use crate::config::{
    AtomRanges, BaseAtom, BulkConfig, LatticeTypeSpec, SolverType,
    SweepDimension, SweepValue, parse_atom_path,
};

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

    /// The job to execute (either Maxwell or EA)
    pub job_type: ExpandedJobType,

    /// Parameter values used for this job (for output columns)
    pub params: JobParams,
}

impl ExpandedJob {
    /// Get the Maxwell job, if this is a Maxwell job type.
    pub fn maxwell_job(&self) -> Option<&BandStructureJob> {
        match &self.job_type {
            ExpandedJobType::Maxwell(job) => Some(job),
            _ => None,
        }
    }

    /// Get the EA job spec, if this is an EA job type.
    pub fn ea_job(&self) -> Option<&EAJobSpec> {
        match &self.job_type {
            ExpandedJobType::EA(spec) => Some(spec),
            _ => None,
        }
    }
}

/// Type of job to execute.
#[derive(Debug, Clone)]
pub enum ExpandedJobType {
    /// Maxwell band structure job (photonic crystals)
    Maxwell(BandStructureJob),
    /// EA single-solve job (moiré lattices)
    EA(EAJobSpec),
}

/// EA job specification (input data configuration).
///
/// Unlike Maxwell jobs which have all parameters resolved, EA jobs
/// refer to input files that are read at execution time.
#[derive(Debug, Clone)]
pub struct EAJobSpec {
    /// Grid dimensions
    pub grid: Grid2D,
    /// SingleSolveJob configuration (n_bands, tolerance, etc.)
    pub solve_config: SingleSolveJob,
    /// EA-specific parameters from config
    pub eta: f64,
    pub domain_size: [f64; 2],
    pub potential_path: std::path::PathBuf,
    pub mass_inv_path: std::path::PathBuf,
    pub vg_path: Option<std::path::PathBuf>,
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

    /// Sweep values in order (for new format).
    /// Each entry is (parameter_name, value_string).
    /// Order matches the sweep order from config.
    #[allow(dead_code)]
    pub sweep_values: Vec<(String, SweepValue)>,
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

    /// Get sweep order string for output (e.g., "atom0.radius=0.2|eps_bg=10.0")
    pub fn sweep_order_string(&self) -> String {
        self.sweep_values
            .iter()
            .map(|(name, val)| format!("{}={}", name, val))
            .collect::<Vec<_>>()
            .join("|")
    }
}

// ============================================================================
// Expansion Logic
// ============================================================================

/// Expand a bulk configuration into individual jobs.
///
/// This function dispatches based on configuration format and solver type:
///
/// - **New format** (`[[sweeps]]`): Uses `expand_ordered_jobs` for user-defined loop order
/// - **Legacy format** (`[ranges]`): Uses `expand_maxwell_jobs` with hardcoded order
/// - **EA solver**: Uses `expand_ea_jobs` (single job for now)
pub fn expand_jobs(config: &BulkConfig) -> Vec<ExpandedJob> {
    match config.solver_type() {
        SolverType::Maxwell => {
            if config.uses_ordered_sweeps() {
                expand_ordered_jobs(config)
            } else {
                expand_maxwell_jobs_legacy(config)
            }
        }
        SolverType::EA => expand_ea_jobs(config),
    }
}

// ============================================================================
// Ordered Expansion (New Format)
// ============================================================================

/// Expand jobs using ordered sweeps from `[[sweeps]]` array.
///
/// Jobs are generated with the first sweep as outermost loop and last as innermost.
/// This gives users full control over the parameter variation order.
fn expand_ordered_jobs(config: &BulkConfig) -> Vec<ExpandedJob> {
    // Build sweep dimensions in order
    let dimensions: Vec<SweepDimension> = match config.build_sweep_dimensions() {
        Ok(dims) => dims,
        Err(e) => {
            eprintln!("Warning: failed to build sweep dimensions: {}", e);
            return vec![];
        }
    };

    // If no sweeps, return single job with defaults
    if dimensions.is_empty() {
        return vec![create_single_default_job(config)];
    }

    // Calculate total number of combinations
    let total: usize = dimensions.iter().map(|d| d.len().max(1)).product();
    
    // Generate all index combinations
    let combinations = generate_ordered_indices(&dimensions);
    
    let mut jobs = Vec::with_capacity(total);
    
    for (job_index, indices) in combinations.into_iter().enumerate() {
        // Build sweep values for this combination
        let sweep_values: Vec<(String, SweepValue)> = dimensions
            .iter()
            .zip(indices.iter())
            .map(|(dim, &idx)| (dim.name.clone(), dim.values[idx].clone()))
            .collect();
        
        // Create the job
        let job = create_job_from_sweep_values(config, &sweep_values, job_index);
        jobs.push(job);
    }
    
    jobs
}

/// Generate all index combinations for the given dimensions.
///
/// The first dimension is the outermost loop (changes slowest),
/// the last dimension is the innermost loop (changes fastest).
fn generate_ordered_indices(dimensions: &[SweepDimension]) -> Vec<Vec<usize>> {
    if dimensions.is_empty() {
        return vec![vec![]];
    }

    let total: usize = dimensions.iter().map(|d| d.len().max(1)).product();
    let mut result = Vec::with_capacity(total);
    
    // Start with all zeros
    let mut indices: Vec<usize> = vec![0; dimensions.len()];
    
    loop {
        result.push(indices.clone());
        
        // Increment indices from right to left (innermost to outermost)
        let mut carry = true;
        for i in (0..dimensions.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] >= dimensions[i].len() {
                    indices[i] = 0;
                    carry = true;
                } else {
                    carry = false;
                }
            }
        }
        
        // If we carried all the way through, we're done
        if carry {
            break;
        }
    }
    
    result
}

/// Create a single job from sweep values.
fn create_job_from_sweep_values(
    config: &BulkConfig,
    sweep_values: &[(String, SweepValue)],
    job_index: usize,
) -> ExpandedJob {
    // Get base values from defaults or geometry
    let geometry = config.effective_geometry();
    
    let mut eps_bg = geometry.map(|g| g.eps_bg).unwrap_or(12.0);
    let mut resolution = config.effective_resolution();
    let mut polarization = config.effective_polarization();
    let mut lattice_type: Option<LatticeTypeSpec> = geometry
        .and_then(|g| g.lattice.lattice_type.clone());
    
    // Build atom parameters from base
    let base_atoms = geometry.map(|g| g.atoms.clone()).unwrap_or_default();
    let mut atom_params: HashMap<usize, (f64, f64, f64, f64)> = base_atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (i, (a.pos[0], a.pos[1], a.radius, a.eps_inside)))
        .collect();
    
    // Apply sweep values
    for (param, value) in sweep_values {
        match param.as_str() {
            "eps_bg" => {
                if let Some(v) = value.as_f64() {
                    eps_bg = v;
                }
            }
            "resolution" => {
                if let Some(v) = value.as_i64() {
                    resolution = v as usize;
                }
            }
            "polarization" => {
                if let Some(s) = value.as_str() {
                    polarization = match s.to_uppercase().as_str() {
                        "TM" => Polarization::TM,
                        "TE" => Polarization::TE,
                        _ => polarization,
                    };
                }
            }
            "lattice_type" => {
                if let Some(s) = value.as_str() {
                    lattice_type = Some(match s.to_lowercase().as_str() {
                        "square" => LatticeTypeSpec::Square,
                        "rectangular" => LatticeTypeSpec::Rectangular,
                        "triangular" => LatticeTypeSpec::Triangular,
                        "hexagonal" => LatticeTypeSpec::Hexagonal,
                        _ => LatticeTypeSpec::Square,
                    });
                }
            }
            _ if param.starts_with("atom") => {
                if let Some((atom_idx, prop)) = parse_atom_path(param) {
                    let entry = atom_params.entry(atom_idx).or_insert((0.5, 0.5, 0.3, 1.0));
                    if let Some(v) = value.as_f64() {
                        match prop {
                            "pos_x" => entry.0 = v,
                            "pos_y" => entry.1 = v,
                            "radius" => entry.2 = v,
                            "eps_inside" => entry.3 = v,
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    // Build atoms list
    let max_atom_idx = atom_params.keys().max().copied().unwrap_or(0);
    let atoms: Vec<AtomParams> = (0..=max_atom_idx)
        .map(|i| {
            let (px, py, r, eps) = atom_params.get(&i).copied().unwrap_or((0.5, 0.5, 0.3, 1.0));
            AtomParams {
                index: i,
                pos: [px, py],
                radius: r,
                eps_inside: eps,
            }
        })
        .collect();
    
    // Build BaseAtom list for job creation
    let base_atom_list: Vec<BaseAtom> = atoms
        .iter()
        .map(|a| BaseAtom {
            pos: a.pos,
            radius: a.radius,
            eps_inside: a.eps_inside,
        })
        .collect();
    
    // Create the Maxwell job
    let job = create_maxwell_job(
        config,
        eps_bg,
        resolution,
        polarization,
        lattice_type.as_ref(),
        &base_atom_list,
    );
    
    let params = JobParams {
        eps_bg,
        resolution,
        polarization,
        lattice_type: lattice_type.map(|lt| lt.to_string()),
        atoms,
        sweep_values: sweep_values.to_vec(),
    };
    
    ExpandedJob {
        index: job_index,
        job_type: ExpandedJobType::Maxwell(job),
        params,
    }
}

/// Create a single job with default parameters (no sweeps).
fn create_single_default_job(config: &BulkConfig) -> ExpandedJob {
    let geometry = config.effective_geometry().expect("geometry required");
    let atoms: Vec<BaseAtom> = geometry.atoms.clone();
    
    let job = create_maxwell_job(
        config,
        geometry.eps_bg,
        config.effective_resolution(),
        config.effective_polarization(),
        geometry.lattice.lattice_type.as_ref(),
        &atoms,
    );
    
    let atom_params: Vec<AtomParams> = atoms
        .iter()
        .enumerate()
        .map(|(i, a)| AtomParams {
            index: i,
            pos: a.pos,
            radius: a.radius,
            eps_inside: a.eps_inside,
        })
        .collect();
    
    let params = JobParams {
        eps_bg: geometry.eps_bg,
        resolution: config.effective_resolution(),
        polarization: config.effective_polarization(),
        lattice_type: geometry.lattice.lattice_type.as_ref().map(|lt| lt.to_string()),
        atoms: atom_params,
        sweep_values: vec![],
    };
    
    ExpandedJob {
        index: 0,
        job_type: ExpandedJobType::Maxwell(job),
        params,
    }
}

// ============================================================================
// Legacy Expansion
// ============================================================================

/// Expand Maxwell solver jobs over all parameter combinations (legacy format).
///
/// Uses hardcoded loop order: eps_bg → resolution → polarization → lattice_type → atoms
fn expand_maxwell_jobs_legacy(config: &BulkConfig) -> Vec<ExpandedJob> {
    let ranges = &config.ranges;

    // Geometry is required for Maxwell
    let geometry = config
        .geometry
        .as_ref()
        .expect("Maxwell solver requires geometry");

    // Collect all parameter axes
    let eps_bg_values = ranges
        .eps_bg
        .as_ref()
        .map(|r| r.values())
        .unwrap_or_else(|| vec![geometry.eps_bg]);

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
    let atom_configs = collect_atom_configs_legacy(config);

    // Calculate total combinations
    let total = eps_bg_values.len()
        * resolution_values.len()
        * polarization_values.len()
        * lattice_type_values.as_ref().map(|v| v.len()).unwrap_or(1)
        * atom_configs.iter().map(|a| a.len()).product::<usize>().max(1);

    let mut jobs = Vec::with_capacity(total);
    let mut index = 0;

    // Iterate over all combinations (hardcoded order)
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
                        let job = create_maxwell_job(
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
                            sweep_values: vec![], // Legacy format doesn't track sweep order
                        };

                        jobs.push(ExpandedJob {
                            index,
                            job_type: ExpandedJobType::Maxwell(job),
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

/// Expand EA solver jobs.
///
/// Currently produces a single job since EA doesn't have parameter sweeps yet.
fn expand_ea_jobs(config: &BulkConfig) -> Vec<ExpandedJob> {
    let ea = &config.ea;

    // Build SingleSolveJob from eigensolver config
    let solve_config = SingleSolveJob::new(config.eigensolver.n_bands)
        .with_tolerance(config.eigensolver.tol)
        .with_max_iterations(config.eigensolver.max_iter);

    let job_spec = EAJobSpec {
        grid: config.grid.clone(),
        solve_config,
        eta: ea.eta,
        domain_size: ea.domain_size,
        potential_path: ea.potential.clone().expect("potential path required"),
        mass_inv_path: ea.mass_inv.clone().expect("mass_inv path required"),
        vg_path: ea.vg.clone(),
    };

    // EA parameters for output labeling
    let params = JobParams {
        eps_bg: 0.0, // Not used for EA
        resolution: config.grid.nx,
        polarization: Polarization::TM, // Not used for EA
        lattice_type: None,
        atoms: vec![],
        sweep_values: vec![],
    };

    vec![ExpandedJob {
        index: 0,
        job_type: ExpandedJobType::EA(job_spec),
        params,
    }]
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Collect possible configurations for each atom (legacy format).
fn collect_atom_configs_legacy(config: &BulkConfig) -> Vec<Vec<BaseAtom>> {
    let geometry = match &config.geometry {
        Some(g) => g,
        None => return vec![vec![]],
    };

    let base_atoms = &geometry.atoms;
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
fn create_maxwell_job(
    config: &BulkConfig,
    eps_bg: f64,
    resolution: usize,
    polarization: Polarization,
    lattice_type: Option<&LatticeTypeSpec>,
    atoms: &[BaseAtom],
) -> BandStructureJob {
    let geometry_config = config
        .geometry
        .as_ref()
        .expect("geometry required for Maxwell solver");

    // Build lattice
    let lattice = build_lattice(&geometry_config.lattice, lattice_type);

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
    use crate::config::{
        BaseGeometry, BulkSection, DefaultsConfig, EAConfig, OutputConfig, ParameterRange,
        RangeSpec, SolverSection, SweepSpec,
    };
    use mpb2d_core::{dielectric::DielectricOptions, eigensolver::EigensolverConfig, io::PathSpec};

    fn make_test_config() -> BulkConfig {
        BulkConfig {
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
        // Job 0: radius=0.2, eps=10
        // Job 1: radius=0.2, eps=11
        // Job 2: radius=0.2, eps=12
        // Job 3: radius=0.3, eps=10
        // ...
        
        // Use approximate comparison for floating point
        let eps = 1e-9;
        
        assert!((jobs[0].params.atoms[0].radius - 0.2).abs() < eps);
        assert!((jobs[0].params.eps_bg - 10.0).abs() < eps);
        
        assert!((jobs[1].params.atoms[0].radius - 0.2).abs() < eps);
        assert!((jobs[1].params.eps_bg - 11.0).abs() < eps);
        
        assert!((jobs[2].params.atoms[0].radius - 0.2).abs() < eps);
        assert!((jobs[2].params.eps_bg - 12.0).abs() < eps);
        
        assert!((jobs[3].params.atoms[0].radius - 0.3).abs() < eps);
        assert!((jobs[3].params.eps_bg - 10.0).abs() < eps);
    }

    #[test]
    fn expand_ordered_discrete_values() {
        let mut config = make_test_config();
        
        // Define sweeps with discrete values
        config.sweeps = vec![
            SweepSpec {
                parameter: "polarization".to_string(),
                min: None,
                max: None,
                step: None,
                values: Some(vec![
                    toml::Value::String("TM".to_string()),
                    toml::Value::String("TE".to_string()),
                ]),
            },
            SweepSpec {
                parameter: "eps_bg".to_string(),
                min: Some(10.0),
                max: Some(11.0),
                step: Some(1.0),
                values: None,
            },
        ];
        
        let jobs = expand_jobs(&config);
        
        // Should have 2 pol * 2 eps = 4 jobs
        assert_eq!(jobs.len(), 4);
        
        // Verify order: pol is outer, eps is inner
        assert_eq!(jobs[0].params.polarization, Polarization::TM);
        assert_eq!(jobs[0].params.eps_bg, 10.0);
        
        assert_eq!(jobs[1].params.polarization, Polarization::TM);
        assert_eq!(jobs[1].params.eps_bg, 11.0);
        
        assert_eq!(jobs[2].params.polarization, Polarization::TE);
        assert_eq!(jobs[2].params.eps_bg, 10.0);
    }

    #[test]
    fn sweep_order_string() {
        let params = JobParams {
            eps_bg: 12.0,
            resolution: 32,
            polarization: Polarization::TM,
            lattice_type: None,
            atoms: vec![],
            sweep_values: vec![
                ("atom0.radius".to_string(), SweepValue::Float(0.3)),
                ("eps_bg".to_string(), SweepValue::Float(12.0)),
            ],
        };
        
        assert_eq!(params.sweep_order_string(), "atom0.radius=0.3|eps_bg=12");
    }

    #[test]
    fn expand_ea_single_job() {
        use std::path::PathBuf;

        let config = BulkConfig {
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
