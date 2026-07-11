//! Job expansion from schema v2 configurations.
//!
//! Takes a validated [`Config`] and expands its `[[sweeps]]` axes into a list
//! of concrete [`ExpandedJob`]s ready for execution.
//!
//! Sweeps form nested loops in TOML order; the first entry is the outermost
//! loop (changes slowest):
//!
//! ```text
//! [[sweeps]] parameter = "atom0.radius"   # outer loop
//! [[sweeps]] parameter = "eps_bg"         # inner loop
//!
//! job 0: radius=0.2, eps=10
//! job 1: radius=0.2, eps=11
//! job 2: radius=0.2, eps=12
//! job 3: radius=0.3, eps=10
//! ...
//! ```
//!
//! Base values come from the main sections: `eps_bg` from `[geometry]`,
//! `resolution` from `[grid].nx`, `polarization` from `[solver]`, the lattice
//! type from `[geometry.lattice]`, and atom parameters from
//! `[[geometry.atoms]]`. All values (base and swept) were already validated at
//! parse time, so expansion cannot fail on a validated config.

use blaze2d_core::{
    bandstructure::BandStructureJob,
    brillouin::generate_path,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    polarization::Polarization,
};

use crate::config::{
    BaseAtom, Config, LatticeKind, SolverType, SweepDimension, SweepValue, parse_atom_path,
    parse_swept_lattice_kind,
};

// ============================================================================
// Expanded Job
// ============================================================================

/// A single expanded job with all parameters resolved to concrete values.
#[derive(Debug, Clone)]
pub struct ExpandedJob {
    /// Unique job index (0-based).
    pub index: usize,

    /// The job to execute.
    pub job_type: ExpandedJobType,

    /// Parameter values used for this job (for output columns).
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
}

/// Type of job to execute.
#[derive(Debug, Clone)]
pub enum ExpandedJobType {
    /// Maxwell band structure job (photonic crystals).
    Maxwell(BandStructureJob),
    /// Operator-data extraction job.
    OperatorData(OperatorDataJobSpec),
}

/// Operator-data extraction job specification.
///
/// Everything needed to run the Maxwell eigenproblem at one (R, k₀) point and
/// extract the multi-band operator-data ingredients.
#[derive(Debug, Clone)]
pub struct OperatorDataJobSpec {
    /// The 2D photonic crystal geometry.
    pub geom: Geometry2D,
    /// Computational grid.
    pub grid: Grid2D,
    /// Polarization mode.
    pub pol: Polarization,
    /// Carrier momentum k₀ in Cartesian reciprocal-space units.
    pub k0: [f64; 2],
    /// Registry point metadata (fractional coordinates).
    pub registry: [f64; 2],
    /// Number of retained bands.
    pub n_retained: usize,
    /// Number of remote bands.
    pub n_remote: usize,
    /// Whether to compute the mass tensor.
    pub compute_mass_tensor: bool,
    /// Whether to compute the Born-Huang potential.
    pub compute_born_huang: bool,
    /// Whether to compute the slow-coefficient potential.
    pub compute_slow_coefficient: bool,
    /// Whether to compute dielectric derivatives.
    pub compute_r_derivatives: bool,
    /// Atom index for R-derivatives.
    pub atom_index: usize,
    /// Finite-difference step.
    pub fd_step: f64,
    /// Eigensolver tolerance.
    pub tolerance: f64,
    /// Maximum eigensolver iterations.
    pub max_iterations: usize,
}

/// Concrete parameter values for a job (for output labeling).
#[derive(Debug, Clone)]
pub struct JobParams {
    /// Background epsilon.
    pub eps_bg: f64,

    /// Resolution (nx = ny for swept resolutions; nx otherwise).
    pub resolution: usize,

    /// Polarization.
    pub polarization: Polarization,

    /// Lattice type name.
    pub lattice_type: Option<String>,

    /// Per-atom parameters.
    pub atoms: Vec<AtomParams>,

    /// Sweep values in loop order: (parameter_name, value).
    pub sweep_values: Vec<(String, SweepValue)>,
}

/// Parameters for a single atom.
#[derive(Debug, Clone)]
pub struct AtomParams {
    /// Atom index (0-based).
    pub index: usize,

    /// Position (fractional coordinates).
    pub pos: [f64; 2],

    /// Radius (units of the lattice constant).
    pub radius: f64,

    /// Epsilon inside.
    pub eps_inside: f64,
}

impl JobParams {
    /// Flat list of (name, value) pairs for CSV headers.
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

    /// Sweep order string for output (e.g. `"atom0.radius=0.2|eps_bg=10"`).
    pub fn sweep_order_string(&self) -> String {
        self.sweep_values
            .iter()
            .map(|(name, val)| format!("{}={}", name, val))
            .collect::<Vec<_>>()
            .join("|")
    }
}

// ============================================================================
// Expansion
// ============================================================================

/// Expand a validated configuration into individual jobs.
pub fn expand_jobs(config: &Config) -> Vec<ExpandedJob> {
    match config.solver_type() {
        SolverType::Maxwell => expand_maxwell_jobs(config),
        SolverType::OperatorData => expand_operator_data_jobs(config),
    }
}

fn expand_maxwell_jobs(config: &Config) -> Vec<ExpandedJob> {
    let dimensions: Vec<SweepDimension> = config
        .build_sweep_dimensions()
        .expect("sweeps were validated at parse time");

    if dimensions.is_empty() {
        return vec![create_job_from_sweep_values(config, &[], 0)];
    }

    let total: usize = dimensions.iter().map(|d| d.len().max(1)).product();
    let combinations = generate_ordered_indices(&dimensions);

    let mut jobs = Vec::with_capacity(total);
    for (job_index, indices) in combinations.into_iter().enumerate() {
        let sweep_values: Vec<(String, SweepValue)> = dimensions
            .iter()
            .zip(indices.iter())
            .map(|(dim, &idx)| (dim.name.clone(), dim.values[idx].clone()))
            .collect();
        jobs.push(create_job_from_sweep_values(config, &sweep_values, job_index));
    }
    jobs
}

/// Generate all index combinations, first dimension outermost.
fn generate_ordered_indices(dimensions: &[SweepDimension]) -> Vec<Vec<usize>> {
    if dimensions.is_empty() {
        return vec![vec![]];
    }

    let total: usize = dimensions.iter().map(|d| d.len().max(1)).product();
    let mut result = Vec::with_capacity(total);
    let mut indices: Vec<usize> = vec![0; dimensions.len()];

    loop {
        result.push(indices.clone());

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
        if carry {
            break;
        }
    }

    result
}

/// Create one job from base values plus the given sweep overrides.
fn create_job_from_sweep_values(
    config: &Config,
    sweep_values: &[(String, SweepValue)],
    job_index: usize,
) -> ExpandedJob {
    // Base values from the main sections.
    let mut eps_bg = config.geometry.eps_bg;
    let mut resolution_override: Option<usize> = None;
    let mut polarization = config.solver.polarization;
    let mut lattice_kind = config.geometry.lattice.kind;
    let mut atoms: Vec<BaseAtom> = config.geometry.atoms.clone();

    // Apply sweep overrides (all values pre-validated at parse time).
    for (param, value) in sweep_values {
        match param.as_str() {
            "eps_bg" => {
                if let Some(v) = value.as_f64() {
                    eps_bg = v;
                }
            }
            "resolution" => {
                if let Some(v) = value.as_i64() {
                    resolution_override = Some(v as usize);
                }
            }
            "polarization" => {
                if let Some(s) = value.as_str() {
                    polarization = match s.to_uppercase().as_str() {
                        "TE" => Polarization::TE,
                        _ => Polarization::TM,
                    };
                }
            }
            "lattice_type" => {
                if let Some(kind) = value.as_str().and_then(parse_swept_lattice_kind) {
                    lattice_kind = kind;
                }
            }
            _ => {
                if let Some((atom_idx, prop)) = parse_atom_path(param) {
                    let atom = atoms
                        .get_mut(atom_idx)
                        .expect("atom references were validated at parse time");
                    if let Some(v) = value.as_f64() {
                        match prop {
                            "pos_x" => atom.pos[0] = v,
                            "pos_y" => atom.pos[1] = v,
                            "radius" => atom.radius = v,
                            "eps_inside" => atom.eps_inside = v,
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // Build the runtime pieces.
    let kind_override = (lattice_kind != config.geometry.lattice.kind).then_some(lattice_kind);
    let lattice = config.geometry.lattice.build(kind_override);
    let basis_atoms: Vec<BasisAtom> = atoms
        .iter()
        .map(|a| BasisAtom {
            pos: a.pos,
            radius: a.radius,
            eps_inside: a.eps_inside,
        })
        .collect();
    let geometry = Geometry2D {
        lattice,
        eps_bg,
        atoms: basis_atoms,
    };
    let grid = config.grid.to_grid(resolution_override);
    let k_path = resolve_k_path(config, lattice_kind);

    let job = BandStructureJob {
        geom: geometry,
        grid,
        pol: polarization,
        k_path,
        eigensolver: config.eigensolver.clone(),
        dielectric: config.dielectric.to_options(),
    };

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
        eps_bg,
        resolution: grid_resolution(&job.grid),
        polarization,
        lattice_type: Some(lattice_kind.to_string()),
        atoms: atom_params,
        sweep_values: sweep_values.to_vec(),
    };

    ExpandedJob {
        index: job_index,
        job_type: ExpandedJobType::Maxwell(job),
        params,
    }
}

fn grid_resolution(grid: &Grid2D) -> usize {
    grid.nx
}

/// Resolve the k-path for a job's effective lattice kind.
fn resolve_k_path(config: &Config, lattice_kind: LatticeKind) -> Vec<[f64; 2]> {
    let path = config
        .path
        .as_ref()
        .expect("validated: Maxwell solver requires [path]");

    if !path.points.is_empty() {
        return path.points.clone();
    }

    let preset = path
        .preset
        .expect("validated: [path] has preset or points");
    let brillouin = preset
        .resolve(lattice_kind)
        .expect("validated: presets require a named lattice type");
    generate_path(&brillouin, path.points_per_segment())
}

/// Expand operator-data extraction jobs (single job at the origin registry).
fn expand_operator_data_jobs(config: &Config) -> Vec<ExpandedJob> {
    let od = &config.operator_data;

    let lattice = config.geometry.lattice.build(None);
    let basis_atoms: Vec<BasisAtom> = config
        .geometry
        .atoms
        .iter()
        .map(|a| BasisAtom {
            pos: a.pos,
            radius: a.radius,
            eps_inside: a.eps_inside,
        })
        .collect();

    let geom = Geometry2D {
        lattice,
        eps_bg: config.geometry.eps_bg,
        atoms: basis_atoms,
    };
    let grid = config.grid.to_grid(None);
    let pol = config.solver.polarization;

    let job_spec = OperatorDataJobSpec {
        geom,
        grid,
        pol,
        k0: od.k0,
        registry: [0.0, 0.0],
        n_retained: od.n_retained,
        n_remote: od.n_remote,
        compute_mass_tensor: od.compute_mass_tensor,
        compute_born_huang: od.compute_born_huang,
        compute_slow_coefficient: od.compute_slow_coefficient,
        compute_r_derivatives: od.compute_r_derivatives,
        atom_index: od.atom_index,
        fd_step: od.fd_step,
        tolerance: config.eigensolver.tol,
        max_iterations: config.eigensolver.max_iter,
    };

    let atom_params: Vec<AtomParams> = config
        .geometry
        .atoms
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
        eps_bg: config.geometry.eps_bg,
        resolution: config.grid.nx,
        polarization: pol,
        lattice_type: Some(config.geometry.lattice.kind.to_string()),
        atoms: atom_params,
        sweep_values: vec![],
    };

    vec![ExpandedJob {
        index: 0,
        job_type: ExpandedJobType::OperatorData(job_spec),
        params,
    }]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    const BASE: &str = r#"
schema = 2

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.3

[grid]
nx = 32

[path]
preset = "auto"
"#;

    fn config(extra: &str) -> Config {
        Config::from_str(&format!("{}\n{}", BASE, extra)).expect("test config must parse")
    }

    #[test]
    fn expand_single_job() {
        let jobs = expand_jobs(&config(""));
        assert_eq!(jobs.len(), 1);
        assert!(matches!(jobs[0].job_type, ExpandedJobType::Maxwell(_)));
        assert_eq!(jobs[0].params.lattice_type.as_deref(), Some("square"));
    }

    #[test]
    fn expand_ordered_sweeps() {
        let cfg = config(
            "[[sweeps]]\nparameter = \"atom0.radius\"\nmin = 0.2\nmax = 0.3\nstep = 0.1\n\
             [[sweeps]]\nparameter = \"eps_bg\"\nmin = 10.0\nmax = 12.0\nstep = 1.0\n",
        );
        let jobs = expand_jobs(&cfg);

        // 2 radius values (outer) x 3 eps values (inner).
        assert_eq!(jobs.len(), 6);
        let eps = 1e-9;
        assert!((jobs[0].params.atoms[0].radius - 0.2).abs() < eps);
        assert!((jobs[0].params.eps_bg - 10.0).abs() < eps);
        assert!((jobs[1].params.atoms[0].radius - 0.2).abs() < eps);
        assert!((jobs[1].params.eps_bg - 11.0).abs() < eps);
        assert!((jobs[3].params.atoms[0].radius - 0.3).abs() < eps);
        assert!((jobs[3].params.eps_bg - 10.0).abs() < eps);
    }

    #[test]
    fn expand_discrete_values() {
        let cfg = config(
            "[[sweeps]]\nparameter = \"polarization\"\nvalues = [\"TM\", \"TE\"]\n\
             [[sweeps]]\nparameter = \"eps_bg\"\nmin = 10.0\nmax = 11.0\nstep = 1.0\n",
        );
        let jobs = expand_jobs(&cfg);
        assert_eq!(jobs.len(), 4);
        assert_eq!(jobs[0].params.polarization, Polarization::TM);
        assert_eq!(jobs[0].params.eps_bg, 10.0);
        assert_eq!(jobs[2].params.polarization, Polarization::TE);
        assert_eq!(jobs[2].params.eps_bg, 10.0);
    }

    #[test]
    fn resolution_sweep_scales_grid() {
        let cfg = config("[[sweeps]]\nparameter = \"resolution\"\nvalues = [16, 64]\n");
        let jobs = expand_jobs(&cfg);
        assert_eq!(jobs.len(), 2);
        let g0 = jobs[0].maxwell_job().unwrap().grid;
        let g1 = jobs[1].maxwell_job().unwrap().grid;
        assert_eq!((g0.nx, g0.ny), (16, 16));
        assert_eq!((g1.nx, g1.ny), (64, 64));
        assert_eq!(jobs[1].params.resolution, 64);
    }

    #[test]
    fn auto_preset_follows_lattice_type_sweep() {
        let cfg = config(
            "[[sweeps]]\nparameter = \"lattice_type\"\nvalues = [\"square\", \"rectangular\"]\n",
        );
        let jobs = expand_jobs(&cfg);
        assert_eq!(jobs.len(), 2);
        let square_len = jobs[0].maxwell_job().unwrap().k_path.len();
        let rect_len = jobs[1].maxwell_job().unwrap().k_path.len();
        // Square path has 3 legs, rectangular has 4: the rectangular path
        // must be strictly longer at equal points_per_segment.
        assert!(
            rect_len > square_len,
            "expected rectangular path ({}) longer than square ({})",
            rect_len,
            square_len
        );
    }

    #[test]
    fn explicit_points_used_verbatim() {
        let toml = BASE.replace(
            "preset = \"auto\"",
            "points = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]",
        );
        let cfg = Config::from_str(&toml).unwrap();
        let jobs = expand_jobs(&cfg);
        assert_eq!(jobs[0].maxwell_job().unwrap().k_path.len(), 3);
    }

    #[test]
    fn dielectric_none_reaches_job() {
        let cfg = config("[dielectric]\nsmoothing = \"none\"\n");
        let jobs = expand_jobs(&cfg);
        let job = jobs[0].maxwell_job().unwrap();
        assert!(!job.dielectric.smoothing_enabled());
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
}
