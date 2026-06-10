//! Python bindings for EA Hamiltonian ingredient extraction.
//!
//! This module provides a Python-callable interface for extracting all physical
//! quantities needed to construct the envelope-approximation Hamiltonian for
//! moiré photonic crystals.
//!
//! # Usage (Python)
//!
//! ```python
//! from blaze import OperatorDataExtractor
//!
//! result = OperatorDataExtractor.extract(
//!     lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
//!     atoms=[{"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 1.0}],
//!     eps_bg=12.0,
//!     k0=[0.0, 0.0],
//!     polarization="TE",
//!     resolution=32,
//!     n_retained=4,
//!     n_remote=8,
//! )
//!
//! # Access the ingredients
//! eigenvalues = result["eigenvalues"]
//! velocity_x = result["velocity_matrices_x"]   # complex list
//! mass_tensor = result["mass_tensor_inv_xx"]     # complex list
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use serde_json;

use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::dielectric::{Dielectric2D, DielectricOptions};
use blaze2d_core::drivers::bandstructure::{self, BandStructureJob, RunOptions};
use blaze2d_core::drivers::operator_data::{self, OperatorDataJob};
use blaze2d_core::drivers::single_solve::{self, SingleSolveJob};
use blaze2d_core::operator_data::OperatorDataConfig;
use blaze2d_core::eigensolver::EigensolverConfig;
use blaze2d_core::field::Field2D;
use blaze2d_core::geometry::{BasisAtom, Geometry2D};
use blaze2d_core::grid::Grid2D;
use blaze2d_core::lattice::Lattice2D;
use blaze2d_core::operators::ThetaOperator;
use blaze2d_core::polarization::Polarization;

/// Parse a polarization string ("TE" or "TM") into the enum.
fn parse_polarization(s: &str) -> PyResult<Polarization> {
    match s.to_uppercase().as_str() {
        "TE" => Ok(Polarization::TE),
        "TM" => Ok(Polarization::TM),
        _ => Err(PyValueError::new_err("polarization must be 'TE' or 'TM'")),
    }
}

/// Parse a list of Python atom dicts into BasisAtom structs.
fn parse_atoms(atoms: &[Bound<'_, PyDict>]) -> PyResult<Vec<BasisAtom>> {
    atoms
        .iter()
        .map(|atom_dict| {
            let pos: [f64; 2] = atom_dict
                .get_item("pos")?
                .ok_or_else(|| PyValueError::new_err("atom dict missing 'pos'"))?
                .extract()?;
            let radius: f64 = atom_dict
                .get_item("radius")?
                .ok_or_else(|| PyValueError::new_err("atom dict missing 'radius'"))?
                .extract()?;
            let eps_inside: f64 = atom_dict
                .get_item("eps_inside")?
                .ok_or_else(|| PyValueError::new_err("atom dict missing 'eps_inside'"))?
                .extract()?;
            Ok(BasisAtom {
                pos,
                radius,
                eps_inside,
            })
        })
        .collect()
}

/// Build dielectric options from a smoothing flag and optional method.
///
/// `smoothing_method`: `None` → default (analytic), `"analytic"`, or `"subgrid"`.
fn build_dielectric_opts(smoothing: bool, smoothing_method: Option<&str>) -> DielectricOptions {
    if !smoothing {
        return DielectricOptions {
            smoothing: blaze2d_core::dielectric::SmoothingOptions {
                mesh_size: 1,
                ..Default::default()
            },
        };
    }
    let method = match smoothing_method {
        Some("subgrid") => blaze2d_core::dielectric::SmoothingMethod::Subgrid,
        Some("analytic") | None => blaze2d_core::dielectric::SmoothingMethod::Analytic,
        Some(other) => {
            log::warn!("Unknown smoothing method '{}', using analytic", other);
            blaze2d_core::dielectric::SmoothingMethod::Analytic
        }
    };
    DielectricOptions {
        smoothing: blaze2d_core::dielectric::SmoothingOptions {
            method,
            ..Default::default()
        },
    }
}

/// Convert a panic payload into a PyRuntimeError.
fn panic_to_pyerr(e: Box<dyn std::any::Any + Send>) -> PyErr {
    let msg = if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "internal solver error".to_string()
    };
    pyo3::exceptions::PyRuntimeError::new_err(format!("Solver panicked: {}", msg))
}

fn wrap_unit_interval(value: f64) -> f64 {
    let wrapped = value % 1.0;
    if wrapped < 0.0 {
        wrapped + 1.0
    } else {
        wrapped
    }
}

fn shifted_basis_atoms(
    base_atoms: &[BasisAtom],
    atom_index: usize,
    registry: [f64; 2],
) -> Vec<BasisAtom> {
    let mut atoms = base_atoms.to_vec();
    atoms[atom_index].pos = [
        wrap_unit_interval(base_atoms[atom_index].pos[0] + registry[0]),
        wrap_unit_interval(base_atoms[atom_index].pos[1] + registry[1]),
    ];
    atoms
}

#[allow(clippy::too_many_arguments)]
fn build_ea_job(
    lattice_vectors: [[f64; 2]; 2],
    atoms: Vec<BasisAtom>,
    eps_bg: f64,
    k0: [f64; 2],
    pol: Polarization,
    resolution: usize,
    band_lo: usize,
    n_retained: usize,
    n_remote: usize,
    compute_born_huang: bool,
    compute_slow_coefficient: bool,
    compute_overlap: bool,
    atom_index: usize,
    fd_step: f64,
    registry: [f64; 2],
    tolerance: f64,
    max_iterations: usize,
    compute_r_derivatives: bool,
    dielectric_opts: DielectricOptions,
) -> OperatorDataJob {
    let lattice = Lattice2D::oblique(lattice_vectors[0], lattice_vectors[1]);
    let geom = Geometry2D {
        lattice,
        eps_bg,
        atoms,
    };
    let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);

    let operator_data_config = OperatorDataConfig {
        band_lo,
        n_retained,
        n_remote,
        compute_mass_tensor: true,
        compute_born_huang,
        compute_slow_coefficient,
        compute_overlap,
    };

    let mut eigensolver_config = EigensolverConfig::default();
    eigensolver_config.n_bands = band_lo + n_retained + n_remote;
    eigensolver_config.tol = tolerance;
    eigensolver_config.max_iter = max_iterations;

    OperatorDataJob {
        geom,
        grid,
        pol,
        k0,
        registry,
        operator_data_config,
        eigensolver: eigensolver_config,
        dielectric: dielectric_opts,
        fd_step,
        atom_index,
        compute_dielectric_derivatives: compute_r_derivatives,
    }
}

fn parse_reference_fields(
    eigenvectors: &[Vec<(f64, f64)>],
    grid: Grid2D,
) -> PyResult<Vec<Field2D>> {
    eigenvectors
        .iter()
        .map(|band| {
            if band.len() != grid.len() {
                return Err(PyValueError::new_err(format!(
                    "reference eigenvector has length {}, expected {} for grid {:?}",
                    band.len(),
                    grid.len(),
                    (grid.nx, grid.ny)
                )));
            }
            let data = band
                .iter()
                .map(|(re, im)| blaze2d_core::field::AccumScalar::new(*re, *im))
                .collect();
            Ok(Field2D::from_f64_vec(grid, data))
        })
        .collect()
}

fn single_solve_to_py_dict(
    py: Python<'_>,
    result: &single_solve::SingleSolveResult,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("eigenvalues", result.eigenvalues.clone())?;

    let two_pi = 2.0 * std::f64::consts::PI;
    let freqs: Vec<f64> = result
        .eigenvalues
        .iter()
        .map(|&lambda| lambda.max(0.0).sqrt() / two_pi)
        .collect();
    dict.set_item("freqs", freqs)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("elapsed_seconds", result.elapsed_seconds)?;
    dict.set_item("final_residuals", result.final_residuals.clone())?;
    Ok(dict.into())
}

/// Python wrapper for EA Hamiltonian ingredient extraction.
#[pyclass(name = "OperatorDataExtractor")]
pub struct OperatorDataExtractorPy;

#[pymethods]
impl OperatorDataExtractorPy {
    /// Solve the Maxwell problem on an externally supplied sampled epsilon map.
    ///
    /// This bypasses Blaze's geometry sampling and smoothing pipeline and uses
    /// the provided real-space epsilon grid directly.
    ///
    /// Parameters
    /// ----------
    /// lattice_vectors : list[list[float]]
    ///     2×2 lattice vectors [[a1x, a1y], [a2x, a2y]].
    /// epsilon : list[float]
    ///     Flattened row-major epsilon grid of length resolution*resolution.
    /// k0 : list[float]
    ///     Bloch wavevector [kx, ky] in Cartesian reciprocal-space units.
    /// polarization : str
    ///     "TE" or "TM".
    /// resolution : int
    ///     Grid resolution (nx = ny = resolution).
    /// n_bands : int
    ///     Number of eigenpairs to solve for.
    /// tolerance : float, optional
    ///     Eigensolver tolerance (default: 1e-8).
    /// max_iterations : int, optional
    ///     Maximum eigensolver iterations (default: 300).
    /// inv_eps_tensors : list[list[float]] | None, optional
    ///     Optional per-pixel inverse-epsilon tensors in row-major form
    ///     ``[xx, xy, yx, yy]``. If omitted, Blaze uses the scalar ε⁻¹ path.
    ///
    /// Returns
    /// -------
    /// dict
    ///     - ``eigenvalues``: raw eigenvalues ω²
    ///     - ``freqs``: reduced frequencies ω/(2π)
    ///     - ``iterations``: LOBPCG iteration count
    ///     - ``converged``: solver convergence flag
    ///     - ``elapsed_seconds``: solve time
    ///     - ``final_residuals``: final residuals per band
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        epsilon,
        k0,
        polarization,
        resolution,
        n_bands,
        tolerance = 1e-8,
        max_iterations = 300,
        inv_eps_tensors = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve_external_map(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        epsilon: Vec<f64>,
        k0: [f64; 2],
        polarization: &str,
        resolution: usize,
        n_bands: usize,
        tolerance: f64,
        max_iterations: usize,
        inv_eps_tensors: Option<Vec<[f64; 4]>>,
    ) -> PyResult<Py<PyDict>> {
        let pol = parse_polarization(polarization)?;
        let lattice = Lattice2D::oblique(lattice_vectors[0], lattice_vectors[1]);
        let reciprocal = lattice.reciprocal();
        let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);

        if epsilon.len() != grid.len() {
            return Err(PyValueError::new_err(format!(
                "epsilon has length {}, expected {} for resolution {}",
                epsilon.len(),
                grid.len(),
                resolution
            )));
        }
        if let Some(tensors) = &inv_eps_tensors {
            if tensors.len() != grid.len() {
                return Err(PyValueError::new_err(format!(
                    "inv_eps_tensors has length {}, expected {} for resolution {}",
                    tensors.len(),
                    grid.len(),
                    resolution
                )));
            }
        }

        let dielectric = Dielectric2D::from_sampled_epsilon(
            grid,
            reciprocal.b1,
            reciprocal.b2,
            epsilon,
            inv_eps_tensors,
        );

        let mut job = SingleSolveJob::new(n_bands);
        job.tolerance = tolerance;
        job.max_iterations = max_iterations;

        let result = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = CpuBackend::<f64>::new();
                let mut theta = ThetaOperator::new(backend, dielectric, pol, k0);
                let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();
                single_solve::solve(&mut theta, Some(&mut preconditioner), &job)
            }))
        });

        let result = result.map_err(panic_to_pyerr)?;
        single_solve_to_py_dict(py, &result)
    }

    /// Extract EA Hamiltonian ingredients at a single (R, k₀) point.
    ///
    /// Parameters
    /// ----------
    /// lattice_vectors : list[list[float]]
    ///     2×2 lattice vectors [[a1x, a1y], [a2x, a2y]].
    /// atoms : list[dict]
    ///     List of atoms, each a dict with keys:
    ///     - "pos": [x, y] in fractional coordinates
    ///     - "radius": float (fractional)
    ///     - "eps_inside": float (permittivity inside atom)
    /// eps_bg : float
    ///     Background permittivity.
    /// k0 : list[float]
    ///     Carrier momentum [kx, ky] in Cartesian reciprocal-space units (2π/a).
    /// polarization : str
    ///     "TE" or "TM".
    /// resolution : int
    ///     Grid resolution (nx = ny = resolution).
    /// n_retained : int, optional
    ///     Number of retained bands in the active subspace (default: 4).
    /// n_remote : int, optional
    ///     Number of remote bands for Löwdin sums (default: 8).
    /// compute_born_huang : bool, optional
    ///     Whether to compute Born–Huang potential (default: False).
    ///     For TM this is the generalized metric-compatible geometric
    ///     potential. Requires compute_r_derivatives=True.
    /// compute_slow_coefficient : bool, optional
    ///     Whether to compute the TE slow-coefficient potential U_sc
    ///     (default: False). This is TE-only and intentionally unavailable for
    ///     TM in the current generalized-eigenproblem convention. Requires
    ///     compute_r_derivatives=True.
    /// atom_index : int, optional
    ///     Which atom to differentiate w.r.t. for R-derivatives (default: 0).
    /// fd_step : float, optional
    ///     Finite-difference step for dielectric derivatives (default: 0.001).
    /// registry : list[float], optional
    ///     Registry point metadata [Rx, Ry] (default: [0, 0]).
    ///     Note: this is metadata only. The atoms in ``atoms`` should already
    ///     be shifted to the desired registry position.
    /// tolerance : float, optional
    ///     Eigensolver convergence tolerance (default: 1e-8).
    /// max_iterations : int, optional
    ///     Maximum eigensolver iterations (default: 300).
    /// compute_r_derivatives : bool, optional
    ///     Whether to compute R-derivative matrices (default: True).
    /// smoothing : bool, optional
    ///     Whether to enable MPB-style dielectric smoothing (default: True).
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing all EA Hamiltonian ingredients:
    ///
    ///     **Metadata:**
    ///     - ``polarization`` (str): "TE" or "TM"
    ///     - ``inner_product`` (str): inner product type used
    ///     - ``k0`` (tuple): carrier momentum (kx, ky)
    ///     - ``registry`` (tuple): registry point (Rx, Ry)
    ///     - ``n_retained`` (int): number of retained bands
    ///     - ``n_remote`` (int): number of remote bands
    ///     - ``grid_dims`` (tuple): (nx, ny)
    ///     - ``n_iterations`` (int): solver iterations
    ///     - ``converged`` (bool): convergence status
    ///     - ``solve_time_seconds`` (float): eigensolver wall time
    ///     - ``extract_time_seconds`` (float): extraction wall time
    ///
    ///     **Eigensystem:**
    ///     - ``eigenvalues`` (list[float]): all n_total eigenvalues, units (2π/a)²
    ///     - ``eigenvectors`` (list[list[tuple]]): solver-grid field values as (re, im)
    ///
    ///     **Velocity matrices** (shape: n_retained × n_total, row-major):
    ///     - ``velocity_matrices_x`` (list[tuple]): v^x_{mn} as (re, im)
    ///     - ``velocity_matrices_y`` (list[tuple]): v^y_{mn} as (re, im)
    ///     - ``velocity_matrix_rows`` (int): n_retained
    ///     - ``velocity_matrix_cols`` (int): n_total
    ///
    ///     **Second-derivative matrices** (shape: n_retained × n_retained):
    ///     - ``w_matrices_xx``, ``w_matrices_xy``, ``w_matrices_yx``, ``w_matrices_yy``
    ///     - ``w_matrix_size`` (int): n_retained
    ///
    ///     **Löwdin-corrected inverse mass tensor** (shape: n_retained × n_retained):
    ///     - ``mass_tensor_inv_xx``, ``mass_tensor_inv_xy``,
    ///       ``mass_tensor_inv_yx``, ``mass_tensor_inv_yy``
    ///
    ///     **R-derivative matrices** (optional, shape: n_retained × n_total):
    ///     - ``r_derivative_matrices_x``, ``r_derivative_matrices_y``
    ///     - ``has_r_derivatives`` (bool)
    ///
    ///     **Born-Huang potential** (optional, shape: n_retained × n_retained):
    ///     - For TM this is the generalized metric-compatible geometric
    ///       potential.
    ///     - ``born_huang`` (list[tuple])
    ///     - ``has_born_huang`` (bool)
    ///
    ///     **Slow-coefficient potential** (optional, shape: n_retained × n_retained):
    ///     - Present for TE only.
    ///     - ``slow_coefficient_potential`` (list[tuple])
    ///     - ``has_slow_coefficient`` (bool)
    ///
    ///     **Metric derivatives** (optional, shape: n_retained × n_total):
    ///     - ``metric_derivative_matrices_x``, ``metric_derivative_matrices_y``
    ///     - ``has_metric_derivatives`` (bool)
    ///
    ///     **Berry connection** (optional, shape: n_retained × n_retained):
    ///     - ``berry_connection_x``, ``berry_connection_y``
    ///     - ``has_berry_connection`` (bool)
    ///
    ///     **Overlap matrix** (optional, shape: n_retained × n_retained):
    ///     - ``overlap_matrix`` (list[tuple])
    ///     - ``has_overlap`` (bool)
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        atoms,
        eps_bg,
        k0,
        polarization,
        resolution,
        band_lo = 0,
        n_retained = 4,
        n_remote = 8,
        compute_born_huang = false,
        compute_slow_coefficient = false,
        compute_overlap = false,
        atom_index = 0,
        fd_step = 0.001,
        registry = None,
        tolerance = 1e-8,
        max_iterations = 300,
        compute_r_derivatives = true,
        smoothing = true,
        smoothing_method = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        k0: [f64; 2],
        polarization: &str,
        resolution: usize,
        band_lo: usize,
        n_retained: usize,
        n_remote: usize,
        compute_born_huang: bool,
        compute_slow_coefficient: bool,
        compute_overlap: bool,
        atom_index: usize,
        fd_step: f64,
        registry: Option<[f64; 2]>,
        tolerance: f64,
        max_iterations: usize,
        compute_r_derivatives: bool,
        smoothing: bool,
        smoothing_method: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let pol = parse_polarization(polarization)?;
        let basis_atoms = parse_atoms(&atoms)?;

        if atom_index >= basis_atoms.len() {
            return Err(PyValueError::new_err(format!(
                "atom_index {} out of range (have {} atoms)",
                atom_index,
                basis_atoms.len()
            )));
        }

        let dielectric_opts = build_dielectric_opts(smoothing, smoothing_method);
        let registry = registry.unwrap_or([0.0, 0.0]);
        let job = build_ea_job(
            lattice_vectors,
            basis_atoms,
            eps_bg,
            k0,
            pol,
            resolution,
            band_lo,
            n_retained,
            n_remote,
            compute_born_huang,
            compute_slow_coefficient,
            compute_overlap,
            atom_index,
            fd_step,
            registry,
            tolerance,
            max_iterations,
            compute_r_derivatives,
            dielectric_opts,
        );

        // Run the extraction (release the GIL during computation)
        // Wrap in catch_unwind to convert panics to Python exceptions
        let result = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = CpuBackend::<f64>::new();
                operator_data::run(backend, &job)
            }))
        });

        let result = result.map_err(panic_to_pyerr)?;

        // Convert to Python dict
        ingredients_to_py_dict(py, &result)
    }

    /// Extract independent EA ingredients over a registry sweep in native Rust.
    ///
    /// The supplied ``atoms`` are treated as the base unit cell. For each
    /// registry point, Blaze shifts the tracked atom ``atom_index`` by that
    /// fractional offset, wraps it into ``[0, 1)``, and then runs an ordinary
    /// EA extraction. Results preserve the input registry order.
    ///
    /// Each returned entry has the exact same schema and conventions as
    /// ``extract``.
    ///
    /// Progress is reported natively to stderr from Rust, printing a status
    /// line every ``stride`` points (stride = sqrt(total)).  Registry points
    /// that hit ``max_iterations`` without converging emit a warning to stderr.
    ///
    /// The legacy ``progress_callback`` parameter is accepted but ignored.
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        atoms,
        eps_bg,
        registries,
        k0,
        polarization,
        resolution,
        band_lo = 0,
        n_retained = 4,
        n_remote = 8,
        compute_born_huang = false,
        compute_slow_coefficient = false,
        atom_index = 0,
        fd_step = 0.001,
        tolerance = 1e-8,
        max_iterations = 300,
        compute_r_derivatives = true,
        smoothing = true,
        smoothing_method = None,
        threads = None,
        progress_callback = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract_registry_sweep(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        registries: Vec<[f64; 2]>,
        k0: [f64; 2],
        polarization: &str,
        resolution: usize,
        band_lo: usize,
        n_retained: usize,
        n_remote: usize,
        compute_born_huang: bool,
        compute_slow_coefficient: bool,
        atom_index: usize,
        fd_step: f64,
        tolerance: f64,
        max_iterations: usize,
        compute_r_derivatives: bool,
        smoothing: bool,
        smoothing_method: Option<&str>,
        threads: Option<usize>,
        #[allow(unused_variables)]
        progress_callback: Option<PyObject>,
    ) -> PyResult<Py<PyList>> {
        let pol = parse_polarization(polarization)?;
        let base_atoms = parse_atoms(&atoms)?;

        if atom_index >= base_atoms.len() {
            return Err(PyValueError::new_err(format!(
                "atom_index {} out of range (have {} atoms)",
                atom_index,
                base_atoms.len()
            )));
        }

        let dielectric_opts = build_dielectric_opts(smoothing, smoothing_method);
        let thread_count = threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        });

        let total = registries.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let unconverged = Arc::new(AtomicUsize::new(0));

        // Progress stride: report every sqrt(total) points (one "row" for
        // an NxN grid).  Minimum 1, maximum total.
        let stride = {
            let s = (total as f64).sqrt().round() as usize;
            s.max(1).min(total)
        };

        // Native progress thread — prints to stderr, no GIL needed.
        let stop_flag = Arc::new(AtomicUsize::new(0));
        let stop_flag2 = Arc::clone(&stop_flag);
        let completed_for_progress = Arc::clone(&completed);
        let unconverged_for_progress = Arc::clone(&unconverged);
        let sweep_start = std::time::Instant::now();
        let sweep_start2 = sweep_start;

        let progress_handle = std::thread::spawn(move || {
            let mut last_reported = 0usize;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let done = completed_for_progress.load(Ordering::Relaxed);
                if done / stride > last_reported / stride || done == total {
                    let elapsed = sweep_start2.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 { done as f64 / elapsed } else { 0.0 };
                    let eta = if rate > 0.0 { (total - done) as f64 / rate } else { f64::INFINITY };
                    let unc = unconverged_for_progress.load(Ordering::Relaxed);
                    let pct = 100.0 * done as f64 / total as f64;
                    eprintln!(
                        "  [Blaze] {done}/{total} ({pct:.1}%) | {elapsed:.0}s elapsed | \
                         {rate:.1} pts/s | ETA {eta:.0}s | {unc} unconverged",
                    );
                    last_reported = done;
                }
                if stop_flag2.load(Ordering::Relaxed) == 1 {
                    break;
                }
            }
        });

        // Closure to run one registry point, bump counters, and warn.
        let completed_ref = &completed;
        let unconverged_ref = &unconverged;
        let run_one = |registry: [f64; 2]| -> operator_data::OperatorDataDriverResult {
            let shifted = shifted_basis_atoms(&base_atoms, atom_index, registry);
            let job = build_ea_job(
                lattice_vectors,
                shifted,
                eps_bg,
                k0,
                pol,
                resolution,
                band_lo,
                n_retained,
                n_remote,
                compute_born_huang,
                compute_slow_coefficient,
                false,
                atom_index,
                fd_step,
                registry,
                tolerance,
                max_iterations,
                compute_r_derivatives,
                dielectric_opts.clone(),
            );
            let backend = CpuBackend::<f64>::new();
            let result = operator_data::run(backend, &job);
            let done = completed_ref.fetch_add(1, Ordering::Relaxed) + 1;
            if !result.ingredients.converged {
                unconverged_ref.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "  [Blaze] WARNING: registry ({:.4}, {:.4}) did NOT converge \
                     ({} iters hit max_iterations={})",
                    registry[0], registry[1],
                    result.ingredients.n_iterations, max_iterations,
                );
            }
            let _ = done; // suppress unused warning
            result
        };

        let results = py.allow_threads(|| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if thread_count <= 1 || registries.len() <= 1 {
                    Ok::<Vec<operator_data::OperatorDataDriverResult>, String>(
                        registries.iter().map(|&r| run_one(r)).collect()
                    )
                } else {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(thread_count)
                        .build()
                        .map_err(|err| err.to_string())?;
                    pool.install(|| {
                        Ok::<Vec<operator_data::OperatorDataDriverResult>, String>(
                            registries.par_iter().map(|&r| run_one(r)).collect()
                        )
                    })
                }
            }));

            stop_flag.store(1, Ordering::Relaxed);
            outcome
        });

        let _ = progress_handle.join();

        // Final summary
        let elapsed = sweep_start.elapsed().as_secs_f64();
        let unc_total = unconverged.load(Ordering::Relaxed);
        eprintln!(
            "  [Blaze] Sweep done: {total} points in {elapsed:.1}s \
             ({:.1} ms/pt, {unc_total} unconverged)",
            elapsed * 1000.0 / total as f64,
        );

        let results = results.map_err(panic_to_pyerr)?;
        let results = results.map_err(PyValueError::new_err)?;

        let py_results = PyList::empty(py);
        for result in &results {
            py_results.append(ingredients_to_py_dict(py, result)?)?;
        }
        Ok(py_results.into())
    }

    /// Extract EA ingredients while transporting against an external reference basis.
    ///
    /// This exposes the native Blaze reference-overlap path for neighboring
    /// registry points. When ``reference_eigenvectors`` is supplied and
    /// ``compute_overlap=True``, the returned ``overlap_matrix`` is computed by
    /// the core extractor rather than reconstructed externally from sampled fields.
    /// ``warmstart_eigenvectors`` can additionally seed the eigensolver for more
    /// stable tracking in nearly degenerate subspaces.
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        atoms,
        eps_bg,
        k0,
        polarization,
        resolution,
        band_lo = 0,
        n_retained = 4,
        n_remote = 8,
        compute_born_huang = false,
        compute_slow_coefficient = false,
        compute_overlap = false,
        atom_index = 0,
        fd_step = 0.001,
        registry = None,
        tolerance = 1e-8,
        max_iterations = 300,
        compute_r_derivatives = true,
        smoothing = true,
        smoothing_method = None,
        reference_eigenvectors = None,
        warmstart_eigenvectors = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract_with_reference(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        k0: [f64; 2],
        polarization: &str,
        resolution: usize,
        band_lo: usize,
        n_retained: usize,
        n_remote: usize,
        compute_born_huang: bool,
        compute_slow_coefficient: bool,
        compute_overlap: bool,
        atom_index: usize,
        fd_step: f64,
        registry: Option<[f64; 2]>,
        tolerance: f64,
        max_iterations: usize,
        compute_r_derivatives: bool,
        smoothing: bool,
        smoothing_method: Option<&str>,
        reference_eigenvectors: Option<Vec<Vec<(f64, f64)>>>,
        warmstart_eigenvectors: Option<Vec<Vec<(f64, f64)>>>,
    ) -> PyResult<Py<PyDict>> {
        let pol = parse_polarization(polarization)?;
        let basis_atoms = parse_atoms(&atoms)?;

        if atom_index >= basis_atoms.len() {
            return Err(PyValueError::new_err(format!(
                "atom_index {} out of range (have {} atoms)",
                atom_index,
                basis_atoms.len()
            )));
        }

        let lattice = Lattice2D::oblique(lattice_vectors[0], lattice_vectors[1]);
        let geom = Geometry2D {
            lattice,
            eps_bg,
            atoms: basis_atoms,
        };
        let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);

        let reference_fields = reference_eigenvectors
            .as_ref()
            .map(|vecs| parse_reference_fields(vecs, grid))
            .transpose()?;
        let warmstart_fields = warmstart_eigenvectors
            .as_ref()
            .map(|vecs| parse_reference_fields(vecs, grid))
            .transpose()?;

        let operator_data_config = OperatorDataConfig {
            band_lo,
            n_retained,
            n_remote,
            compute_mass_tensor: true,
            compute_born_huang,
            compute_slow_coefficient,
            compute_overlap,
        };

        let mut eigensolver_config = EigensolverConfig::default();
        eigensolver_config.n_bands = band_lo + n_retained + n_remote;
        eigensolver_config.tol = tolerance;
        eigensolver_config.max_iter = max_iterations;

        let registry = registry.unwrap_or([0.0, 0.0]);
        let job = OperatorDataJob {
            geom,
            grid,
            pol,
            k0,
            registry,
            operator_data_config,
            eigensolver: eigensolver_config,
            dielectric: build_dielectric_opts(smoothing, smoothing_method),
            fd_step,
            atom_index,
            compute_dielectric_derivatives: compute_r_derivatives,
        };

        let result = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = CpuBackend::<f64>::new();
                match warmstart_fields.as_ref() {
                    Some(warmstart) => operator_data::run_with_warmstart(
                        backend,
                        &job,
                        warmstart,
                        reference_fields.as_deref(),
                    ),
                    None => operator_data::run_with_reference(
                        backend,
                        &job,
                        reference_fields.as_deref(),
                    ),
                }
            }))
        });

        let result = result.map_err(panic_to_pyerr)?;
        ingredients_to_py_dict(py, &result)
    }

    /// Extract EA Hamiltonian ingredients on a k-stencil around a center k₀.
    ///
    /// This method solves at the center k₀ first (cold start), then solves at
    /// neighboring k-points warm-started from the center eigenvectors. The
    /// dielectric is built once and reused for all k-points.
    ///
    /// Parameters
    /// ----------
    /// lattice_vectors : list[list[float]]
    ///     2×2 lattice vectors [[a1x, a1y], [a2x, a2y]].
    /// atoms : list[dict]
    ///     List of atoms, each with keys "pos", "radius", "eps_inside".
    /// eps_bg : float
    ///     Background permittivity.
    /// k0 : list[float]
    ///     Center momentum [kx, ky] in Cartesian reciprocal-space units (2π/a).
    /// polarization : str
    ///     "TE" or "TM".
    /// resolution : int
    ///     Grid resolution.
    /// n_stencil : int
    ///     Number of stencil points per axis (must be odd, ≥ 1).
    ///     1 = center only, 3 = 3×3 grid (9 points), 5 = 5×5 grid (25 points).
    /// delta_k : float
    ///     Maximum displacement from center in each direction (same units as k₀).
    /// n_retained : int, optional
    ///     Number of retained bands (default: 4).
    /// n_remote : int, optional
    ///     Number of remote bands (default: 8).
    /// compute_born_huang : bool, optional
    ///     Compute Born–Huang potential (default: False).
    /// compute_overlap : bool, optional
    ///     Compute overlap matrix (default: False).
    /// atom_index : int, optional
    ///     Atom for R-derivatives (default: 0).
    /// fd_step : float, optional
    ///     Finite-difference step (default: 0.001).
    /// registry : list[float], optional
    ///     Registry metadata (default: [0, 0]).
    /// tolerance : float, optional
    ///     Eigensolver tolerance (default: 1e-8).
    /// max_iterations : int, optional
    ///     Max eigensolver iterations (default: 300).
    /// compute_r_derivatives : bool, optional
    ///     Compute R-derivatives (default: True).
    /// smoothing : bool, optional
    ///     Enable dielectric smoothing (default: True).
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary with keys:
    ///     - ``center`` (dict): EA result at the center k₀
    ///     - ``neighbors`` (list[dict]): EA results at each neighbor k-point
    ///     - ``neighbor_k_points`` (list[tuple]): absolute k-points of neighbors
    ///     - ``n_stencil`` (int): stencil size per axis
    ///     - ``delta_k`` (float): max displacement
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        atoms,
        eps_bg,
        k0,
        polarization,
        resolution,
        n_stencil,
        delta_k,
        band_lo = 0,
        n_retained = 4,
        n_remote = 8,
        compute_born_huang = false,
        compute_slow_coefficient = false,
        compute_overlap = false,
        atom_index = 0,
        fd_step = 0.001,
        registry = None,
        tolerance = 1e-8,
        max_iterations = 300,
        compute_r_derivatives = true,
        smoothing = true,
        smoothing_method = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract_k_stencil(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        k0: [f64; 2],
        polarization: &str,
        resolution: usize,
        n_stencil: usize,
        delta_k: f64,
        band_lo: usize,
        n_retained: usize,
        n_remote: usize,
        compute_born_huang: bool,
        compute_slow_coefficient: bool,
        compute_overlap: bool,
        atom_index: usize,
        fd_step: f64,
        registry: Option<[f64; 2]>,
        tolerance: f64,
        max_iterations: usize,
        compute_r_derivatives: bool,
        smoothing: bool,
        smoothing_method: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        if n_stencil == 0 || n_stencil % 2 == 0 {
            return Err(PyValueError::new_err(
                "n_stencil must be odd and >= 1",
            ));
        }

        let pol = parse_polarization(polarization)?;
        let basis_atoms = parse_atoms(&atoms)?;

        if atom_index >= basis_atoms.len() {
            return Err(PyValueError::new_err(format!(
                "atom_index {} out of range (have {} atoms)",
                atom_index,
                basis_atoms.len()
            )));
        }

        let lattice = Lattice2D::oblique(lattice_vectors[0], lattice_vectors[1]);
        let geom = Geometry2D {
            lattice,
            eps_bg,
            atoms: basis_atoms,
        };
        let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);
        let operator_data_config = OperatorDataConfig {
            band_lo,
            n_retained,
            n_remote,
            compute_mass_tensor: true,
            compute_born_huang,
            compute_slow_coefficient,
            compute_overlap,
        };
        let mut eigensolver_config = EigensolverConfig::default();
        eigensolver_config.n_bands = band_lo + n_retained + n_remote;
        eigensolver_config.tol = tolerance;
        eigensolver_config.max_iter = max_iterations;

        let registry = registry.unwrap_or([0.0, 0.0]);
        let job = OperatorDataJob {
            geom,
            grid,
            pol,
            k0,
            registry,
            operator_data_config,
            eigensolver: eigensolver_config,
            dielectric: build_dielectric_opts(smoothing, smoothing_method),
            fd_step,
            atom_index,
            compute_dielectric_derivatives: compute_r_derivatives,
        };

        let stencil_result = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = CpuBackend::<f64>::new();
                operator_data::run_k_stencil(backend, &job, n_stencil, delta_k)
            }))
        });

        let stencil_result = stencil_result.map_err(panic_to_pyerr)?;

        // Build output dict
        let dict = PyDict::new(py);
        dict.set_item("center", ingredients_to_py_dict(py, &stencil_result.center)?)?;

        let neighbors_list = PyList::empty(py);
        for neighbor in &stencil_result.neighbors {
            neighbors_list.append(ingredients_to_py_dict(py, neighbor)?)?;
        }
        dict.set_item("neighbors", neighbors_list)?;

        let k_points_list = PyList::empty(py);
        for kp in &stencil_result.neighbor_k_points {
            k_points_list.append((kp[0], kp[1]))?;
        }
        dict.set_item("neighbor_k_points", k_points_list)?;

        dict.set_item("n_stencil", n_stencil)?;
        dict.set_item("delta_k", delta_k)?;

        Ok(dict.into())
    }

    /// Compute band diagram along a k-path using the full bandstructure driver.
    ///
    /// This uses the same optimised engine as the Blaze2D CLI:
    /// - dielectric built **once** and reused at every k-point,
    /// - **subspace prediction** warm-start (rotation + extrapolation),
    /// - **band tracking** (polar decomposition + Hungarian reordering),
    /// - **Γ-point reuse** when the path loops back,
    /// - mixed-precision defaults (f32 fields, f64 accumulation).
    ///
    /// Parameters
    /// ----------
    /// lattice_vectors : list[list[float]]
    ///     2×2 lattice vectors [[a1x, a1y], [a2x, a2y]].
    /// atoms : list[dict]
    ///     List of atoms with keys "pos", "radius", "eps_inside".
    /// eps_bg : float
    ///     Background permittivity.
    /// k_points : list[list[float]]
    ///     Ordered list of Bloch wave-vectors [[kx,ky], …] in **fractional
    ///     reciprocal-lattice coordinates** (same convention as MPB).
    /// polarization : str
    ///     "TE" or "TM".
    /// resolution : int
    ///     Grid resolution (nx = ny = resolution).
    /// n_bands : int
    ///     Number of bands to compute at each k-point.
    /// tolerance : float, optional
    ///     Eigensolver convergence tolerance.  Pass 0.0 (default) to use the
    ///     built-in default (1e-4 for mixed-precision, 1e-6 for f64).
    /// max_iterations : int, optional
    ///     Maximum eigensolver iterations (default: 200).
    /// smoothing : bool, optional
    ///     Enable MPB-style dielectric smoothing (default: True).
    ///
    /// Returns
    /// -------
    /// dict
    ///     - ``freqs`` (list[list[float]]): frequencies ω in c/a units
    ///       (same units as MPB) per k-point.
    ///     - ``eigenvalues`` (list[list[float]]): raw eigenvalues ω² per
    ///       k-point, units (2π/a)².
    ///     - ``k_points`` (list[list[float]]): the k-points (echoed back,
    ///       fractional reciprocal coords).
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        atoms,
        eps_bg,
        k_points,
        polarization,
        resolution,
        n_bands,
        tolerance = 0.0,
        max_iterations = 200,
        smoothing = true,
        smoothing_method = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve_k_path(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        k_points: Vec<[f64; 2]>,
        polarization: &str,
        resolution: usize,
        n_bands: usize,
        tolerance: f64,
        max_iterations: usize,
        smoothing: bool,
        smoothing_method: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let pol = parse_polarization(polarization)?;
        let basis_atoms = parse_atoms(&atoms)?;

        let lattice = Lattice2D::oblique(lattice_vectors[0], lattice_vectors[1]);
        let geom = Geometry2D {
            lattice,
            eps_bg,
            atoms: basis_atoms,
        };
        let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);

        let mut eigensolver_config = EigensolverConfig::default();
        eigensolver_config.n_bands = n_bands;
        if tolerance > 0.0 {
            eigensolver_config.tol = tolerance;
        }
        eigensolver_config.max_iter = max_iterations;

        let dielectric_opts = build_dielectric_opts(smoothing, smoothing_method);

        let job = BandStructureJob {
            geom,
            grid,
            pol,
            k_path: k_points.clone(),
            eigensolver: eigensolver_config,
            dielectric: dielectric_opts,
        };

        let bs_result = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = CpuBackend::<f64>::new();
                bandstructure::run_with_options(backend, &job, RunOptions::default())
            }))
        });

        let bs_result = bs_result.map_err(panic_to_pyerr)?;

        // Build output dict
        let dict = PyDict::new(py);

        // Frequencies ω/(2π) in c/a units: same convention as MPB output.
        // The bandstructure driver returns angular frequency ω; divide by 2π
        // to get the reduced frequency f = ωa/(2πc) that MPB reports.
        let two_pi = 2.0 * std::f64::consts::PI;
        let freqs_list = PyList::empty(py);
        for bands in &bs_result.bands {
            let freqs: Vec<f64> = bands.iter().map(|&omega| omega / two_pi).collect();
            let inner = PyList::new(py, &freqs)?;
            freqs_list.append(inner)?;
        }
        dict.set_item("freqs", freqs_list)?;

        // Eigenvalues (ω²): for users who need them
        let evals_list = PyList::empty(py);
        for bands in &bs_result.bands {
            let evals: Vec<f64> = bands.iter().map(|&omega| omega * omega).collect();
            let inner = PyList::new(py, &evals)?;
            evals_list.append(inner)?;
        }
        dict.set_item("eigenvalues", evals_list)?;

        // k_points: echo back
        let kpts_list = PyList::empty(py);
        for kp in &bs_result.k_path {
            kpts_list.append(vec![kp[0], kp[1]])?;
        }
        dict.set_item("k_points", kpts_list)?;

        // Path distances (for plotting)
        dict.set_item("distances", bs_result.distances)?;

        Ok(dict.into())
    }

    /// Return the real-space dielectric grid for a given geometry.
    ///
    /// This is useful for validating that Blaze2D sees the same material
    /// distribution as another solver (e.g. MPB).
    ///
    /// Parameters
    /// ----------
    /// lattice_vectors : list[list[float]]
    ///     2×2 lattice vectors [[a1x, a1y], [a2x, a2y]].
    /// atoms : list[dict]
    ///     List of atoms with keys "pos", "radius", "eps_inside".
    /// eps_bg : float
    ///     Background permittivity.
    /// resolution : int
    ///     Grid resolution (nx = ny = resolution).
    /// smoothing : bool, optional
    ///     Enable MPB-style dielectric smoothing (default: True).
    ///
    /// Returns
    /// -------
    /// dict
    ///     - ``epsilon`` (list[float]): flattened ε(r) grid, row-major
    ///       (index ``iy * resolution + ix``). Smoothed if smoothing is on.
    ///       After ``reshape(resolution, resolution)``, axis 0 is y and axis 1
    ///       is x, so plotting helpers like ``matplotlib.imshow(...,
    ///       origin="lower")`` should use the reshaped array directly without
    ///       an extra transpose.
    ///     - ``inv_epsilon`` (list[float]): flattened ε⁻¹(r) grid, row-major.
    ///     - ``epsilon_raw`` (list[float] or None): unsmoothed ε(r), present
    ///       only when smoothing is enabled.
    ///     - ``inv_eps_tensors`` (list[list[float]] or None): flattened 2×2
    ///       inverse-permittivity tensors in row-major order ``[xx, xy, yx, yy]``
    ///       for each grid point when smoothing is enabled.
    ///     - ``resolution`` (int): grid size.
    ///     - ``lattice_vectors`` (list[list[float]]): echoed back.
    ///     - ``reciprocal_vectors`` (list[list[float]]): reciprocal lattice
    ///       vectors b1, b2 (with 2π).
    #[staticmethod]
    #[pyo3(signature = (lattice_vectors, atoms, eps_bg, resolution, smoothing = true, smoothing_method = None))]
    fn get_epsilon(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        resolution: usize,
        smoothing: bool,
        smoothing_method: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let basis_atoms = parse_atoms(&atoms)?;
        let lattice = Lattice2D::oblique(lattice_vectors[0], lattice_vectors[1]);
        let geom = Geometry2D {
            lattice,
            eps_bg,
            atoms: basis_atoms,
        };
        let grid = Grid2D::new(resolution, resolution, 1.0, 1.0);
        let dielectric_opts = build_dielectric_opts(smoothing, smoothing_method);

        let diel = Dielectric2D::from_geometry(&geom, grid, &dielectric_opts);

        let dict = PyDict::new(py);
        dict.set_item("epsilon", diel.eps().to_vec())?;
        dict.set_item("inv_epsilon", diel.inv_eps().to_vec())?;
        if let Some(raw) = diel.unsmoothed_eps() {
            dict.set_item("epsilon_raw", raw.to_vec())?;
        } else {
            dict.set_item("epsilon_raw", py.None())?;
        }
        if let Some(tensors) = diel.inv_eps_tensors() {
            let tensor_rows: Vec<Vec<f64>> = tensors.iter().map(|tensor| tensor.to_vec()).collect();
            dict.set_item("inv_eps_tensors", tensor_rows)?;
        } else {
            dict.set_item("inv_eps_tensors", py.None())?;
        }
        dict.set_item("resolution", resolution)?;
        dict.set_item("lattice_vectors", lattice_vectors.to_vec())?;
        let b1 = diel.reciprocal_b1();
        let b2 = diel.reciprocal_b2();
        dict.set_item("reciprocal_vectors", vec![vec![b1[0], b1[1]], vec![b2[0], b2[1]]])?;

        Ok(dict.into())
    }

    /// Extract EA ingredients over a registry sweep with **checkpoint-based
    /// streaming** to disk.
    ///
    /// This method is designed for arbitrarily large registry sweeps without
    /// running out of memory.  It processes one *row* of registry points at a
    /// time (``n_per_row`` points), writes the row's results to a JSON
    /// checkpoint file, and drops the data.  If a checkpoint file already
    /// exists, that row is skipped (resume semantics).
    ///
    /// Parameters
    /// ----------
    /// lattice_vectors, atoms, eps_bg, registries, k0, polarization,
    /// resolution, n_retained, n_remote, compute_born_huang,
    /// compute_slow_coefficient, atom_index, fd_step, tolerance,
    /// max_iterations, compute_r_derivatives, smoothing, smoothing_method,
    /// threads
    ///     Same as ``extract_registry_sweep``.
    /// checkpoint_dir : str
    ///     Directory to store per-row checkpoint files.  Created if absent.
    /// n_per_row : int
    ///     Number of registry points per row.  For an N×N grid this is N.
    ///     If not provided, defaults to ``√len(registries)`` rounded to the
    ///     nearest integer.
    /// skip_eigenvectors : bool
    ///     If True (default), eigenvectors are **not** written to checkpoints.
    ///     This dramatically reduces disk usage and memory footprint.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"n_rows": int, "n_points": int, "checkpoint_dir": str,
    ///       "n_skipped": int, "n_computed": int, "elapsed_seconds": float,
    ///       "n_unconverged": int}``
    #[staticmethod]
    #[pyo3(signature = (
        lattice_vectors,
        atoms,
        eps_bg,
        registries,
        k0,
        polarization,
        resolution,
        checkpoint_dir,
        band_lo = 0,
        n_retained = 4,
        n_remote = 8,
        compute_born_huang = false,
        compute_slow_coefficient = false,
        atom_index = 0,
        fd_step = 0.001,
        tolerance = 1e-8,
        max_iterations = 300,
        compute_r_derivatives = true,
        smoothing = true,
        smoothing_method = None,
        threads = None,
        n_per_row = None,
        skip_eigenvectors = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn extract_registry_sweep_checkpointed(
        py: Python<'_>,
        lattice_vectors: [[f64; 2]; 2],
        atoms: Vec<Bound<'_, PyDict>>,
        eps_bg: f64,
        registries: Vec<[f64; 2]>,
        k0: [f64; 2],
        polarization: &str,
        resolution: usize,
        checkpoint_dir: &str,
        band_lo: usize,
        n_retained: usize,
        n_remote: usize,
        compute_born_huang: bool,
        compute_slow_coefficient: bool,
        atom_index: usize,
        fd_step: f64,
        tolerance: f64,
        max_iterations: usize,
        compute_r_derivatives: bool,
        smoothing: bool,
        smoothing_method: Option<&str>,
        threads: Option<usize>,
        n_per_row: Option<usize>,
        skip_eigenvectors: bool,
    ) -> PyResult<Py<PyDict>> {
        let pol = parse_polarization(polarization)?;
        let base_atoms = parse_atoms(&atoms)?;

        if atom_index >= base_atoms.len() {
            return Err(PyValueError::new_err(format!(
                "atom_index {} out of range (have {} atoms)",
                atom_index,
                base_atoms.len()
            )));
        }

        let dielectric_opts = build_dielectric_opts(smoothing, smoothing_method);
        let thread_count = threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        });

        let total = registries.len();
        let row_size = n_per_row.unwrap_or_else(|| {
            let s = (total as f64).sqrt().round() as usize;
            s.max(1)
        });

        // Create checkpoint directory
        let ckpt_dir = std::path::PathBuf::from(checkpoint_dir);
        std::fs::create_dir_all(&ckpt_dir).map_err(|e| {
            PyValueError::new_err(format!("cannot create checkpoint dir: {}", e))
        })?;

        // Write sweep metadata (for Python to read later)
        let meta = serde_json::json!({
            "total_points": total,
            "n_per_row": row_size,
            "n_retained": n_retained,
            "n_remote": n_remote,
            "resolution": resolution,
            "polarization": polarization,
            "k0": k0,
            "skip_eigenvectors": skip_eigenvectors,
        });
        let meta_path = ckpt_dir.join("sweep_meta.json");
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap())
            .map_err(|e| PyValueError::new_err(format!("write meta: {}", e)))?;

        let rows: Vec<&[[f64; 2]]> = registries.chunks(row_size).collect();
        let n_rows = rows.len();

        let completed = Arc::new(AtomicUsize::new(0));
        let unconverged = Arc::new(AtomicUsize::new(0));
        let sweep_start = std::time::Instant::now();

        // Count already-completed rows for resume
        let mut n_skipped = 0usize;
        for (row_idx, row_chunk) in rows.iter().enumerate() {
            let row_file = ckpt_dir.join(format!("row_{:06}.json", row_idx));
            if row_file.exists() {
                n_skipped += 1;
                completed.fetch_add(row_chunk.len(), Ordering::Relaxed);
            }
        }
        if n_skipped > 0 {
            eprintln!(
                "  [Blaze] Resuming: {n_skipped}/{n_rows} rows already checkpointed, \
                 skipping {} points",
                completed.load(Ordering::Relaxed),
            );
        }

        // Native progress thread
        let stop_flag = Arc::new(AtomicUsize::new(0));
        let stop_flag2 = Arc::clone(&stop_flag);
        let completed_for_progress = Arc::clone(&completed);
        let unconverged_for_progress = Arc::clone(&unconverged);
        let sweep_start2 = sweep_start;

        let progress_handle = std::thread::spawn(move || {
            let stride = row_size.max(1);
            let mut last_reported = 0usize;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let done = completed_for_progress.load(Ordering::Relaxed);
                if done / stride > last_reported / stride || done == total {
                    let elapsed = sweep_start2.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 {
                        done as f64 / elapsed
                    } else {
                        0.0
                    };
                    let eta = if rate > 0.0 {
                        (total - done) as f64 / rate
                    } else {
                        f64::INFINITY
                    };
                    let unc = unconverged_for_progress.load(Ordering::Relaxed);
                    let pct = 100.0 * done as f64 / total as f64;
                    eprintln!(
                        "  [Blaze] {done}/{total} ({pct:.1}%) | {elapsed:.0}s elapsed | \
                         {rate:.1} pts/s | ETA {eta:.0}s | {unc} unconverged",
                    );
                    last_reported = done;
                }
                if stop_flag2.load(Ordering::Relaxed) == 1 {
                    break;
                }
            }
        });

        // Process rows — release GIL for the entire computation
        let completed_ref = &completed;
        let unconverged_ref = &unconverged;

        let result = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .map_err(|err| err.to_string())?;

                for (row_idx, row_chunk) in rows.iter().enumerate() {
                    let row_file = ckpt_dir.join(format!("row_{:06}.json", row_idx));

                    // Skip already-checkpointed rows
                    if row_file.exists() {
                        continue;
                    }

                    // Compute this row in parallel
                    let row_results: Vec<operator_data::OperatorDataDriverResult> =
                        pool.install(|| {
                            row_chunk
                                .par_iter()
                                .map(|&registry| {
                                    let shifted = shifted_basis_atoms(
                                        &base_atoms,
                                        atom_index,
                                        registry,
                                    );
                                    let job = build_ea_job(
                                        lattice_vectors,
                                        shifted,
                                        eps_bg,
                                        k0,
                                        pol,
                                        resolution,
                                        band_lo,
                                        n_retained,
                                        n_remote,
                                        compute_born_huang,
                                        compute_slow_coefficient,
                                        false,
                                        atom_index,
                                        fd_step,
                                        registry,
                                        tolerance,
                                        max_iterations,
                                        compute_r_derivatives,
                                        dielectric_opts.clone(),
                                    );
                                    let backend = CpuBackend::<f64>::new();
                                    let result = operator_data::run(backend, &job);
                                    let _ =
                                        completed_ref.fetch_add(1, Ordering::Relaxed);
                                    if !result.ingredients.converged {
                                        unconverged_ref
                                            .fetch_add(1, Ordering::Relaxed);
                                        eprintln!(
                                            "  [Blaze] WARNING: registry ({:.4}, {:.4}) \
                                             did NOT converge ({} iters, max={})",
                                            registry[0],
                                            registry[1],
                                            result.ingredients.n_iterations,
                                            max_iterations,
                                        );
                                    }
                                    result
                                })
                                .collect()
                        });

                    // Serialize the row to JSON (without eigenvectors)
                    let row_json = serialize_row_results(
                        row_idx,
                        &row_results,
                        skip_eigenvectors,
                    );

                    // Write to a temp file first, then rename (atomic)
                    let tmp_file =
                        ckpt_dir.join(format!("row_{:06}.json.tmp", row_idx));
                    std::fs::write(&tmp_file, &row_json).map_err(|e| {
                        format!("write checkpoint row {}: {}", row_idx, e)
                    })?;
                    std::fs::rename(&tmp_file, &row_file).map_err(|e| {
                        format!("rename checkpoint row {}: {}", row_idx, e)
                    })?;

                    // row_results is dropped here — memory freed!
                }

                Ok::<(), String>(())
            }))
        });

        stop_flag.store(1, Ordering::Relaxed);
        let _ = progress_handle.join();

        // Handle errors
        let result = result.map_err(panic_to_pyerr)?;
        result.map_err(PyValueError::new_err)?;

        let elapsed = sweep_start.elapsed().as_secs_f64();
        let unc_total = unconverged.load(Ordering::Relaxed);
        let n_computed = n_rows - n_skipped;
        eprintln!(
            "  [Blaze] Checkpointed sweep done: {total} points in {n_rows} rows, \
             {elapsed:.1}s ({n_skipped} resumed, {n_computed} computed, \
             {unc_total} unconverged)",
        );

        // Return summary dict
        let dict = PyDict::new(py);
        dict.set_item("n_rows", n_rows)?;
        dict.set_item("n_points", total)?;
        dict.set_item("checkpoint_dir", checkpoint_dir)?;
        dict.set_item("n_skipped", n_skipped)?;
        dict.set_item("n_computed", n_computed)?;
        dict.set_item("elapsed_seconds", elapsed)?;
        dict.set_item("n_unconverged", unc_total)?;
        Ok(dict.into())
    }

    /// Load one checkpoint row file and return it as a Python list of dicts.
    ///
    /// This is the companion to ``extract_registry_sweep_checkpointed``:
    /// Python calls this method one row at a time to build the stacked arrays
    /// incrementally, never holding more than one row in memory.
    ///
    /// Parameters
    /// ----------
    /// checkpoint_path : str
    ///     Path to a single ``row_NNNNNN.json`` checkpoint file.
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     One dict per registry point in the row, with the same schema as
    ///     ``extract`` (minus eigenvectors if they were skipped).
    #[staticmethod]
    #[pyo3(signature = (checkpoint_path,))]
    fn load_checkpoint_row(
        py: Python<'_>,
        checkpoint_path: &str,
    ) -> PyResult<Py<PyList>> {
        let data = std::fs::read_to_string(checkpoint_path).map_err(|e| {
            PyValueError::new_err(format!("read checkpoint: {}", e))
        })?;
        let row_json: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| {
                PyValueError::new_err(format!("parse checkpoint: {}", e))
            })?;

        let points = row_json["points"]
            .as_array()
            .ok_or_else(|| PyValueError::new_err("checkpoint missing 'points'"))?;

        let py_list = PyList::empty(py);
        for pt in points {
            let dict = checkpoint_point_to_py_dict(py, pt)?;
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }
}

// ============================================================================
// Checkpoint serialization helpers
// ============================================================================

/// Serialize a complex number as a JSON [re, im] array.
fn complex_to_json(c: &blaze2d_core::field::AccumScalar) -> serde_json::Value {
    serde_json::json!([c.re, c.im])
}

/// Serialize a Vec<Complex64> as a JSON array of [re, im] pairs.
fn complex_vec_to_json(v: &[blaze2d_core::field::AccumScalar]) -> serde_json::Value {
    serde_json::Value::Array(v.iter().map(complex_to_json).collect())
}

fn float_vec_to_json(v: &[f64]) -> serde_json::Value {
    serde_json::Value::Array(v.iter().map(|&x| serde_json::json!(x)).collect())
}

fn exact_tm_to_json(
    tm: &blaze2d_core::operator_data::OperatorDataExactTM,
    n_ret: usize,
    n_total: usize,
) -> serde_json::Value {
    let hermitized_eigvecs: Vec<serde_json::Value> = tm
        .hermitized_eigenvectors
        .iter()
        .map(|band| serde_json::Value::Array(band.iter().map(|c| serde_json::json!([c.re, c.im])).collect()))
        .collect();

    serde_json::json!({
        "hermitized_eigenvectors": hermitized_eigvecs,
        "velocity_matrices_x": complex_vec_to_json(&tm.velocity_matrices[0]),
        "velocity_matrices_y": complex_vec_to_json(&tm.velocity_matrices[1]),
        "velocity_matrix_size": n_total,
        "local_r_derivative_matrices_x": complex_vec_to_json(&tm.local_r_derivative_matrices[0]),
        "local_r_derivative_matrices_y": complex_vec_to_json(&tm.local_r_derivative_matrices[1]),
        "local_r_derivative_matrix_size": n_total,
        "local_r_second_derivative_matrices_x": complex_vec_to_json(&tm.local_r_second_derivative_matrices[0]),
        "local_r_second_derivative_matrices_y": complex_vec_to_json(&tm.local_r_second_derivative_matrices[1]),
        "local_r_second_derivative_matrix_size": n_total,
        "first_order_remainder": complex_vec_to_json(&tm.first_order_remainder),
        "first_order_remainder_size": n_total,
        "direct_metric": complex_vec_to_json(&tm.direct_metric),
        "direct_b_matrix_x": complex_vec_to_json(&tm.direct_b_matrices[0]),
        "direct_b_matrix_y": complex_vec_to_json(&tm.direct_b_matrices[1]),
        "direct_gamma2": complex_vec_to_json(&tm.direct_gamma2),
        "direct_matrix_size": n_ret,
        "mass_tensor_inv_xx": complex_vec_to_json(&tm.mass_tensor_inv[0][0]),
        "mass_tensor_inv_xy": complex_vec_to_json(&tm.mass_tensor_inv[0][1]),
        "mass_tensor_inv_yx": complex_vec_to_json(&tm.mass_tensor_inv[1][0]),
        "mass_tensor_inv_yy": complex_vec_to_json(&tm.mass_tensor_inv[1][1]),
        "epsilon_r_derivatives_x": float_vec_to_json(&tm.epsilon_r_derivatives[0]),
        "epsilon_r_derivatives_y": float_vec_to_json(&tm.epsilon_r_derivatives[1]),
        "epsilon_r_second_derivatives_x": float_vec_to_json(&tm.epsilon_r_second_derivatives[0]),
        "epsilon_r_second_derivatives_y": float_vec_to_json(&tm.epsilon_r_second_derivatives[1]),
        "rho_r_derivatives_x": float_vec_to_json(&tm.rho_r_derivatives[0]),
        "rho_r_derivatives_y": float_vec_to_json(&tm.rho_r_derivatives[1]),
        "rho_r_second_derivatives_x": float_vec_to_json(&tm.rho_r_second_derivatives[0]),
        "rho_r_second_derivatives_y": float_vec_to_json(&tm.rho_r_second_derivatives[1])
    })
}

/// Serialize one row of EAResults to a JSON string (no eigenvectors).
fn serialize_row_results(
    row_idx: usize,
    results: &[operator_data::OperatorDataDriverResult],
    skip_eigenvectors: bool,
) -> String {
    let points: Vec<serde_json::Value> = results
        .iter()
        .map(|r| serialize_one_result(r, skip_eigenvectors))
        .collect();

    let row_obj = serde_json::json!({
        "row_index": row_idx,
        "n_points": results.len(),
        "points": points,
    });
    // Use to_string (compact) — checkpoints don't need to be human-readable
    serde_json::to_string(&row_obj).expect("JSON serialization should not fail")
}

/// Serialize a single OperatorDataDriverResult to a JSON Value, optionally omitting eigenvectors.
fn serialize_one_result(
    result: &operator_data::OperatorDataDriverResult,
    skip_eigenvectors: bool,
) -> serde_json::Value {
    let ing = &result.ingredients;
    let n_ret = ing.n_retained;
    let n_total = ing.band_lo + n_ret + ing.n_remote;

    let mut obj = serde_json::json!({
        "polarization": format!("{:?}", ing.polarization),
        "inner_product": ing.inner_product.label(),
        "k0": ing.k0,
        "registry": ing.registry,
        "band_lo": ing.band_lo,
        "n_retained": ing.n_retained,
        "n_remote": ing.n_remote,
        "grid_dims": ing.grid_dims,
        "n_iterations": ing.n_iterations,
        "converged": ing.converged,
        "solve_time_seconds": result.solve_time_seconds,
        "extract_time_seconds": result.extract_time_seconds,
        "eigenvalues": &ing.eigenvalues,
        "velocity_matrices_x": complex_vec_to_json(&ing.velocity_matrices[0]),
        "velocity_matrices_y": complex_vec_to_json(&ing.velocity_matrices[1]),
        "velocity_matrix_rows": n_ret,
        "velocity_matrix_cols": n_total,
        "w_matrices_xx": complex_vec_to_json(&ing.w_matrices[0][0]),
        "w_matrices_xy": complex_vec_to_json(&ing.w_matrices[0][1]),
        "w_matrices_yx": complex_vec_to_json(&ing.w_matrices[1][0]),
        "w_matrices_yy": complex_vec_to_json(&ing.w_matrices[1][1]),
        "w_matrix_size": n_ret,
        "mass_tensor_inv_xx": complex_vec_to_json(&ing.mass_tensor_inv[0][0]),
        "mass_tensor_inv_xy": complex_vec_to_json(&ing.mass_tensor_inv[0][1]),
        "mass_tensor_inv_yx": complex_vec_to_json(&ing.mass_tensor_inv[1][0]),
        "mass_tensor_inv_yy": complex_vec_to_json(&ing.mass_tensor_inv[1][1]),
    });

    let map = obj.as_object_mut().unwrap();

    // Optional fields
    if let Some(ref r_mats) = ing.r_derivative_matrices {
        map.insert("r_derivative_matrices_x".into(), complex_vec_to_json(&r_mats[0]));
        map.insert("r_derivative_matrices_y".into(), complex_vec_to_json(&r_mats[1]));
        map.insert("r_derivative_matrix_rows".into(), n_ret.into());
        map.insert("r_derivative_matrix_cols".into(), n_total.into());
        map.insert("has_r_derivatives".into(), true.into());
    } else {
        map.insert("has_r_derivatives".into(), false.into());
    }

    if let Some(ref metric_mats) = ing.metric_derivative_matrices {
        map.insert("metric_derivative_matrices_x".into(), complex_vec_to_json(&metric_mats[0]));
        map.insert("metric_derivative_matrices_y".into(), complex_vec_to_json(&metric_mats[1]));
        map.insert("has_metric_derivatives".into(), true.into());
    } else {
        map.insert("has_metric_derivatives".into(), false.into());
    }

    if let Some(ref berry) = ing.berry_connection_matrices {
        map.insert("berry_connection_x".into(), complex_vec_to_json(&berry[0]));
        map.insert("berry_connection_y".into(), complex_vec_to_json(&berry[1]));
        map.insert("berry_connection_size".into(), n_ret.into());
        map.insert("has_berry_connection".into(), true.into());
    } else {
        map.insert("has_berry_connection".into(), false.into());
    }

    if let Some(ref phi) = ing.born_huang {
        map.insert("born_huang".into(), complex_vec_to_json(phi));
        map.insert("has_born_huang".into(), true.into());
    } else {
        map.insert("has_born_huang".into(), false.into());
    }

    if let Some(ref phi_tensor) = ing.born_huang_tensor {
        for (i, i_name) in [(0, "x"), (1, "y")].iter() {
            for (j, j_name) in [(0, "x"), (1, "y")].iter() {
                map.insert(
                    format!("born_huang_tensor_{}{}", i_name, j_name),
                    complex_vec_to_json(&phi_tensor[*i][*j]),
                );
            }
        }
    }

    if let Some(ref usc) = ing.slow_coefficient_potential {
        map.insert("slow_coefficient_potential".into(), complex_vec_to_json(usc));
        map.insert("has_slow_coefficient".into(), true.into());
    } else {
        map.insert("has_slow_coefficient".into(), false.into());
    }

    if let Some(ref usc_tensor) = ing.slow_coefficient_tensor {
        for (i, i_name) in [(0, "x"), (1, "y")].iter() {
            for (j, j_name) in [(0, "x"), (1, "y")].iter() {
                map.insert(
                    format!("slow_coefficient_tensor_{}{}", i_name, j_name),
                    complex_vec_to_json(&usc_tensor[*i][*j]),
                );
            }
        }
    }

    if let Some(ref overlap) = ing.overlap_matrix {
        map.insert("overlap_matrix".into(), complex_vec_to_json(overlap));
        map.insert("has_overlap".into(), true.into());
    } else {
        map.insert("has_overlap".into(), false.into());
    }

        if let Some(ref xi_scalar) = ing.xi_scalar_first_order {
            map.insert("xi_scalar_first_order".into(), complex_vec_to_json(xi_scalar));
            map.insert("has_exact_te_downfolding".into(), true.into());
        } else {
            map.insert("has_exact_te_downfolding".into(), false.into());
        }

        if let Some(ref kappa) = ing.kappa_matrices {
            map.insert("kappa_matrix_x".into(), complex_vec_to_json(&kappa[0]));
            map.insert("kappa_matrix_y".into(), complex_vec_to_json(&kappa[1]));
        }

        if let Some(ref weighted) = ing.weighted_leakage_scalar {
            map.insert("weighted_leakage_scalar".into(), complex_vec_to_json(weighted));
        }

        if let Some(ref lowdin_t) = ing.lowdin_t_matrices {
            map.insert("lowdin_t_matrix_x".into(), complex_vec_to_json(&lowdin_t[0]));
            map.insert("lowdin_t_matrix_y".into(), complex_vec_to_json(&lowdin_t[1]));
            map.insert("lowdin_t_matrix_rows".into(), ing.n_remote.into());
            map.insert("lowdin_t_matrix_cols".into(), n_ret.into());
        }

        if let Some(ref lowdin_r) = ing.lowdin_r_matrix {
            map.insert("lowdin_r_matrix".into(), complex_vec_to_json(lowdin_r));
            map.insert("lowdin_r_matrix_rows".into(), ing.n_remote.into());
            map.insert("lowdin_r_matrix_cols".into(), n_ret.into());
        }

        if let Some(ref tm_exact) = ing.exact_tm {
            map.insert("tm_exact".into(), exact_tm_to_json(tm_exact, n_ret, n_total));
        }

    // Eigenvectors — only if requested
    if !skip_eigenvectors {
        let eigvecs: Vec<serde_json::Value> = ing
            .eigenvectors
            .iter()
            .map(|band| {
                serde_json::Value::Array(
                    band.iter().map(|c| serde_json::json!([c.re, c.im])).collect(),
                )
            })
            .collect();
        map.insert("eigenvectors".into(), serde_json::Value::Array(eigvecs));
    }

    obj
}

/// Convert a JSON checkpoint point dict back into a Python dict.
///
/// This mirrors the schema of `ingredients_to_py_dict` so that Python
/// code sees an identical interface whether it came from the live sweep
/// or from a checkpoint file.
fn checkpoint_point_to_py_dict(
    py: Python<'_>,
    pt: &serde_json::Value,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    // Scalar / string fields
    dict.set_item("polarization", pt["polarization"].as_str().unwrap_or(""))?;
    dict.set_item("inner_product", pt["inner_product"].as_str().unwrap_or(""))?;

    let k0 = &pt["k0"];
    dict.set_item("k0", (
        k0[0].as_f64().unwrap_or(0.0),
        k0[1].as_f64().unwrap_or(0.0),
    ))?;
    let reg = &pt["registry"];
    dict.set_item("registry", (
        reg[0].as_f64().unwrap_or(0.0),
        reg[1].as_f64().unwrap_or(0.0),
    ))?;

    dict.set_item("n_retained", pt["n_retained"].as_u64().unwrap_or(0))?;
    dict.set_item("n_remote", pt["n_remote"].as_u64().unwrap_or(0))?;

    let gd = &pt["grid_dims"];
    dict.set_item("grid_dims", (
        gd[0].as_u64().unwrap_or(0),
        gd[1].as_u64().unwrap_or(0),
    ))?;

    dict.set_item("n_iterations", pt["n_iterations"].as_u64().unwrap_or(0))?;
    dict.set_item("converged", pt["converged"].as_bool().unwrap_or(false))?;
    dict.set_item("solve_time_seconds", pt["solve_time_seconds"].as_f64().unwrap_or(0.0))?;
    dict.set_item("extract_time_seconds", pt["extract_time_seconds"].as_f64().unwrap_or(0.0))?;

    // Eigenvalues — array of floats
    if let Some(evals) = pt["eigenvalues"].as_array() {
        let vals: Vec<f64> = evals.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
        dict.set_item("eigenvalues", vals)?;
    }

    // Velocity matrix rows/cols
    dict.set_item("velocity_matrix_rows", pt["velocity_matrix_rows"].as_u64().unwrap_or(0))?;
    dict.set_item("velocity_matrix_cols", pt["velocity_matrix_cols"].as_u64().unwrap_or(0))?;

    // Complex matrix fields: [[re, im], ...]  → list of (re, im) tuples
    for key in &[
        "velocity_matrices_x", "velocity_matrices_y",
        "w_matrices_xx", "w_matrices_xy", "w_matrices_yx", "w_matrices_yy",
        "mass_tensor_inv_xx", "mass_tensor_inv_xy",
        "mass_tensor_inv_yx", "mass_tensor_inv_yy",
    ] {
        if let Some(arr) = pt[*key].as_array() {
            let py_list = json_complex_array_to_py(py, arr)?;
            dict.set_item(*key, py_list)?;
        }
    }

    dict.set_item("w_matrix_size", pt["w_matrix_size"].as_u64().unwrap_or(0))?;

    // Optional fields
    for key in &[
        "r_derivative_matrices_x", "r_derivative_matrices_y",
        "metric_derivative_matrices_x", "metric_derivative_matrices_y",
        "berry_connection_x", "berry_connection_y",
        "born_huang", "born_huang_tensor_xx", "born_huang_tensor_xy",
        "born_huang_tensor_yx", "born_huang_tensor_yy",
        "slow_coefficient_potential", "slow_coefficient_tensor_xx",
        "slow_coefficient_tensor_xy", "slow_coefficient_tensor_yx",
        "slow_coefficient_tensor_yy", "overlap_matrix",
        "xi_scalar_first_order", "kappa_matrix_x", "kappa_matrix_y",
        "weighted_leakage_scalar", "lowdin_t_matrix_x", "lowdin_t_matrix_y",
        "lowdin_r_matrix",
    ] {
        if let Some(arr) = pt.get(*key).and_then(|v| v.as_array()) {
            let py_list = json_complex_array_to_py(py, arr)?;
            dict.set_item(*key, py_list)?;
        }
    }

    // Boolean flags
    for flag in &[
        "has_r_derivatives", "has_metric_derivatives", "has_berry_connection",
        "has_born_huang", "has_slow_coefficient", "has_overlap",
        "has_exact_te_downfolding",
    ] {
        if let Some(v) = pt.get(*flag) {
            dict.set_item(*flag, v.as_bool().unwrap_or(false))?;
        }
    }

    // Optional row/col metadata
    for key in &[
        "r_derivative_matrix_rows", "r_derivative_matrix_cols",
        "berry_connection_size", "lowdin_t_matrix_rows", "lowdin_t_matrix_cols",
        "lowdin_r_matrix_rows", "lowdin_r_matrix_cols",
    ] {
        if let Some(v) = pt.get(*key).and_then(|v| v.as_u64()) {
            dict.set_item(*key, v)?;
        }
    }

    // Eigenvectors (if present)
    if let Some(eigvecs) = pt.get("eigenvectors").and_then(|v| v.as_array()) {
        let eigvecs_py = PyList::empty(py);
        for band in eigvecs {
            if let Some(band_arr) = band.as_array() {
                let band_list = json_complex_array_to_py(py, band_arr)?;
                eigvecs_py.append(band_list)?;
            }
        }
        dict.set_item("eigenvectors", eigvecs_py)?;
    }

    if let Some(tm_exact) = pt.get("tm_exact") {
        let tm_dict = PyDict::new(py);

        for key in &[
            "velocity_matrices_x", "velocity_matrices_y",
            "local_r_derivative_matrices_x", "local_r_derivative_matrices_y",
            "local_r_second_derivative_matrices_x", "local_r_second_derivative_matrices_y",
            "first_order_remainder", "direct_metric", "direct_b_matrix_x",
            "direct_b_matrix_y", "direct_gamma2", "mass_tensor_inv_xx",
            "mass_tensor_inv_xy", "mass_tensor_inv_yx", "mass_tensor_inv_yy",
        ] {
            if let Some(arr) = tm_exact.get(*key).and_then(|v| v.as_array()) {
                tm_dict.set_item(*key, json_complex_array_to_py(py, arr)?)?;
            }
        }

        for key in &[
            "epsilon_r_derivatives_x", "epsilon_r_derivatives_y",
            "epsilon_r_second_derivatives_x", "epsilon_r_second_derivatives_y",
            "rho_r_derivatives_x", "rho_r_derivatives_y",
            "rho_r_second_derivatives_x", "rho_r_second_derivatives_y",
        ] {
            if let Some(arr) = tm_exact.get(*key).and_then(|v| v.as_array()) {
                let vals: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                tm_dict.set_item(*key, vals)?;
            }
        }

        for key in &[
            "velocity_matrix_size", "local_r_derivative_matrix_size",
            "local_r_second_derivative_matrix_size", "first_order_remainder_size",
            "direct_matrix_size",
        ] {
            if let Some(v) = tm_exact.get(*key).and_then(|v| v.as_u64()) {
                tm_dict.set_item(*key, v)?;
            }
        }

        if let Some(eigvecs) = tm_exact.get("hermitized_eigenvectors").and_then(|v| v.as_array()) {
            let eigvecs_py = PyList::empty(py);
            for band in eigvecs {
                if let Some(arr) = band.as_array() {
                    eigvecs_py.append(json_complex_array_to_py(py, arr)?)?;
                }
            }
            tm_dict.set_item("hermitized_eigenvectors", eigvecs_py)?;
        }

        dict.set_item("tm_exact", tm_dict)?;
    }

    Ok(dict.into())
}

/// Convert a JSON array of [re, im] pairs into a Python list of (re, im) tuples.
fn json_complex_array_to_py<'py>(
    py: Python<'py>,
    arr: &[serde_json::Value],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for v in arr {
        let re = v[0].as_f64().unwrap_or(0.0);
        let im = v[1].as_f64().unwrap_or(0.0);
        list.append((re, im))?;
    }
    Ok(list)
}

/// Convert `OperatorDataDriverResult` to a Python dictionary.
fn ingredients_to_py_dict(
    py: Python<'_>,
    result: &operator_data::OperatorDataDriverResult,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let ing = &result.ingredients;

    // -- Metadata --
    dict.set_item("polarization", format!("{:?}", ing.polarization))?;
    dict.set_item("inner_product", ing.inner_product.label())?;
    dict.set_item("k0", (ing.k0[0], ing.k0[1]))?;
    dict.set_item("registry", (ing.registry[0], ing.registry[1]))?;
    dict.set_item("n_retained", ing.n_retained)?;
    dict.set_item("n_remote", ing.n_remote)?;
    dict.set_item("band_lo", ing.band_lo)?;
    dict.set_item("grid_dims", (ing.grid_dims[0], ing.grid_dims[1]))?;
    dict.set_item("n_iterations", ing.n_iterations)?;
    dict.set_item("converged", ing.converged)?;
    dict.set_item("solve_time_seconds", result.solve_time_seconds)?;
    dict.set_item("extract_time_seconds", result.extract_time_seconds)?;

    // -- Eigenvalues --
    dict.set_item("eigenvalues", ing.eigenvalues.clone())?;

    // -- Eigenvectors as list of list of (re, im) --
    let eigvecs_py = PyList::empty(py);
    for ev in &ing.eigenvectors {
        let band_vec = PyList::empty(py);
        for c in ev {
            band_vec.append((c.re, c.im))?;
        }
        eigvecs_py.append(band_vec)?;
    }
    dict.set_item("eigenvectors", eigvecs_py)?;

    // -- Velocity matrices as list of (re, im) --
    // Shape: [n_retained * n_total], row-major
    let n_ret = ing.n_retained;
    let n_total = n_ret + ing.n_remote;

    for (dir_idx, dir_name) in [(0, "x"), (1, "y")].iter() {
        let v_list = complex_vec_to_py(py, &ing.velocity_matrices[*dir_idx])?;
        dict.set_item(format!("velocity_matrices_{}", dir_name), v_list)?;
    }
    dict.set_item("velocity_matrix_rows", n_ret)?;
    dict.set_item("velocity_matrix_cols", n_total)?;

    // -- w matrices (second derivative) --
    for (i, i_name) in [(0, "x"), (1, "y")].iter() {
        for (j, j_name) in [(0, "x"), (1, "y")].iter() {
            let w_list = complex_vec_to_py(py, &ing.w_matrices[*i][*j])?;
            dict.set_item(format!("w_matrices_{}{}", i_name, j_name), w_list)?;
        }
    }
    dict.set_item("w_matrix_size", n_ret)?;

    // -- Mass tensor inverse (Löwdin-corrected) --
    for (i, i_name) in [(0, "x"), (1, "y")].iter() {
        for (j, j_name) in [(0, "x"), (1, "y")].iter() {
            let m_list = complex_vec_to_py(py, &ing.mass_tensor_inv[*i][*j])?;
            dict.set_item(format!("mass_tensor_inv_{}{}", i_name, j_name), m_list)?;
        }
    }

    // -- R-derivative matrices (optional) --
    if let Some(ref r_mats) = ing.r_derivative_matrices {
        for (dir_idx, dir_name) in [(0, "x"), (1, "y")].iter() {
            let r_list = complex_vec_to_py(py, &r_mats[*dir_idx])?;
            dict.set_item(format!("r_derivative_matrices_{}", dir_name), r_list)?;
        }
        dict.set_item("r_derivative_matrix_rows", n_ret)?;
        dict.set_item("r_derivative_matrix_cols", n_total)?;
        dict.set_item("has_r_derivatives", true)?;

    } else {
        dict.set_item("has_r_derivatives", false)?;
    }

    if let Some(ref metric_mats) = ing.metric_derivative_matrices {
        for (dir_idx, dir_name) in [(0, "x"), (1, "y")].iter() {
            let metric_list = complex_vec_to_py(py, &metric_mats[*dir_idx])?;
            dict.set_item(format!("metric_derivative_matrices_{}", dir_name), metric_list)?;
        }
        dict.set_item("has_metric_derivatives", true)?;
    } else {
        dict.set_item("has_metric_derivatives", false)?;
    }

    if let Some(ref berry) = ing.berry_connection_matrices {
        for (dir_idx, dir_name) in [(0, "x"), (1, "y")].iter() {
            let berry_list = complex_vec_to_py(py, &berry[*dir_idx])?;
            dict.set_item(format!("berry_connection_{}", dir_name), berry_list)?;
        }
        dict.set_item("berry_connection_size", n_ret)?;
        dict.set_item("has_berry_connection", true)?;
    } else {
        dict.set_item("has_berry_connection", false)?;
    }

    // -- Born–Huang potential (optional) --
    if let Some(ref phi) = ing.born_huang {
        let phi_list = complex_vec_to_py(py, phi)?;
        dict.set_item("born_huang", phi_list)?;
        dict.set_item("has_born_huang", true)?;
    } else {
        dict.set_item("has_born_huang", false)?;
    }

    if let Some(ref phi_tensor) = ing.born_huang_tensor {
        for (i, i_name) in [(0, "x"), (1, "y")].iter() {
            for (j, j_name) in [(0, "x"), (1, "y")].iter() {
                let phi_list = complex_vec_to_py(py, &phi_tensor[*i][*j])?;
                dict.set_item(format!("born_huang_tensor_{}{}", i_name, j_name), phi_list)?;
            }
        }
    }

    if let Some(ref usc) = ing.slow_coefficient_potential {
        let usc_list = complex_vec_to_py(py, usc)?;
        dict.set_item("slow_coefficient_potential", usc_list)?;
        dict.set_item("has_slow_coefficient", true)?;
    } else {
        dict.set_item("has_slow_coefficient", false)?;
    }

    if let Some(ref usc_tensor) = ing.slow_coefficient_tensor {
        for (i, i_name) in [(0, "x"), (1, "y")].iter() {
            for (j, j_name) in [(0, "x"), (1, "y")].iter() {
                let usc_list = complex_vec_to_py(py, &usc_tensor[*i][*j])?;
                dict.set_item(format!("slow_coefficient_tensor_{}{}", i_name, j_name), usc_list)?;
            }
        }
    }

    // -- Overlap matrix (optional) --
    if let Some(ref overlap) = ing.overlap_matrix {
        let overlap_list = complex_vec_to_py(py, overlap)?;
        dict.set_item("overlap_matrix", overlap_list)?;
        dict.set_item("has_overlap", true)?;
    } else {
        dict.set_item("has_overlap", false)?;
    }

    if let Some(ref xi_scalar) = ing.xi_scalar_first_order {
        let xi_list = complex_vec_to_py(py, xi_scalar)?;
        dict.set_item("xi_scalar_first_order", xi_list)?;
        dict.set_item("has_exact_te_downfolding", true)?;
    } else {
        dict.set_item("has_exact_te_downfolding", false)?;
    }

    if let Some(ref kappa) = ing.kappa_matrices {
        let kappa_x = complex_vec_to_py(py, &kappa[0])?;
        let kappa_y = complex_vec_to_py(py, &kappa[1])?;
        dict.set_item("kappa_matrix_x", kappa_x)?;
        dict.set_item("kappa_matrix_y", kappa_y)?;
    }

    if let Some(ref weighted) = ing.weighted_leakage_scalar {
        let weighted_list = complex_vec_to_py(py, weighted)?;
        dict.set_item("weighted_leakage_scalar", weighted_list)?;
    }

    if let Some(ref lowdin_t) = ing.lowdin_t_matrices {
        let lowdin_t_x = complex_vec_to_py(py, &lowdin_t[0])?;
        let lowdin_t_y = complex_vec_to_py(py, &lowdin_t[1])?;
        dict.set_item("lowdin_t_matrix_x", lowdin_t_x)?;
        dict.set_item("lowdin_t_matrix_y", lowdin_t_y)?;
        dict.set_item("lowdin_t_matrix_rows", ing.n_remote)?;
        dict.set_item("lowdin_t_matrix_cols", n_ret)?;
    }

    if let Some(ref lowdin_r) = ing.lowdin_r_matrix {
        let lowdin_r_list = complex_vec_to_py(py, lowdin_r)?;
        dict.set_item("lowdin_r_matrix", lowdin_r_list)?;
        dict.set_item("lowdin_r_matrix_rows", ing.n_remote)?;
        dict.set_item("lowdin_r_matrix_cols", n_ret)?;
    }

    if let Some(ref tm_exact) = ing.exact_tm {
        let tm_dict = PyDict::new(py);

        let hermitized_eigvecs = PyList::empty(py);
        for band in &tm_exact.hermitized_eigenvectors {
            let band_list = PyList::empty(py);
            for c in band {
                band_list.append((c.re, c.im))?;
            }
            hermitized_eigvecs.append(band_list)?;
        }
        tm_dict.set_item("hermitized_eigenvectors", hermitized_eigvecs)?;

        tm_dict.set_item("velocity_matrices_x", complex_vec_to_py(py, &tm_exact.velocity_matrices[0])?)?;
        tm_dict.set_item("velocity_matrices_y", complex_vec_to_py(py, &tm_exact.velocity_matrices[1])?)?;
        tm_dict.set_item("velocity_matrix_size", n_total)?;

        tm_dict.set_item(
            "local_r_derivative_matrices_x",
            complex_vec_to_py(py, &tm_exact.local_r_derivative_matrices[0])?,
        )?;
        tm_dict.set_item(
            "local_r_derivative_matrices_y",
            complex_vec_to_py(py, &tm_exact.local_r_derivative_matrices[1])?,
        )?;
        tm_dict.set_item("local_r_derivative_matrix_size", n_total)?;

        tm_dict.set_item(
            "local_r_second_derivative_matrices_x",
            complex_vec_to_py(py, &tm_exact.local_r_second_derivative_matrices[0])?,
        )?;
        tm_dict.set_item(
            "local_r_second_derivative_matrices_y",
            complex_vec_to_py(py, &tm_exact.local_r_second_derivative_matrices[1])?,
        )?;
        tm_dict.set_item("local_r_second_derivative_matrix_size", n_total)?;

        tm_dict.set_item("first_order_remainder", complex_vec_to_py(py, &tm_exact.first_order_remainder)?)?;
        tm_dict.set_item("first_order_remainder_size", n_total)?;
        tm_dict.set_item("direct_metric", complex_vec_to_py(py, &tm_exact.direct_metric)?)?;
        tm_dict.set_item("direct_b_matrix_x", complex_vec_to_py(py, &tm_exact.direct_b_matrices[0])?)?;
        tm_dict.set_item("direct_b_matrix_y", complex_vec_to_py(py, &tm_exact.direct_b_matrices[1])?)?;
        tm_dict.set_item("direct_gamma2", complex_vec_to_py(py, &tm_exact.direct_gamma2)?)?;
        tm_dict.set_item("direct_matrix_size", n_ret)?;

        for (i, i_name) in [(0, "x"), (1, "y")].iter() {
            for (j, j_name) in [(0, "x"), (1, "y")].iter() {
                tm_dict.set_item(
                    format!("mass_tensor_inv_{}{}", i_name, j_name),
                    complex_vec_to_py(py, &tm_exact.mass_tensor_inv[*i][*j])?,
                )?;
            }
        }

        tm_dict.set_item("epsilon_r_derivatives_x", tm_exact.epsilon_r_derivatives[0].clone())?;
        tm_dict.set_item("epsilon_r_derivatives_y", tm_exact.epsilon_r_derivatives[1].clone())?;
        tm_dict.set_item(
            "epsilon_r_second_derivatives_x",
            tm_exact.epsilon_r_second_derivatives[0].clone(),
        )?;
        tm_dict.set_item(
            "epsilon_r_second_derivatives_y",
            tm_exact.epsilon_r_second_derivatives[1].clone(),
        )?;
        tm_dict.set_item("rho_r_derivatives_x", tm_exact.rho_r_derivatives[0].clone())?;
        tm_dict.set_item("rho_r_derivatives_y", tm_exact.rho_r_derivatives[1].clone())?;
        tm_dict.set_item(
            "rho_r_second_derivatives_x",
            tm_exact.rho_r_second_derivatives[0].clone(),
        )?;
        tm_dict.set_item(
            "rho_r_second_derivatives_y",
            tm_exact.rho_r_second_derivatives[1].clone(),
        )?;

        dict.set_item("tm_exact", tm_dict)?;
    }

    Ok(dict.into())
}

/// Convert a Vec<Complex64> to a Python list of (real, imag) tuples.
fn complex_vec_to_py<'py>(py: Python<'py>, vec: &[blaze2d_core::field::AccumScalar]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for c in vec {
        list.append((c.re, c.im))?;
    }
    Ok(list)
}

/// Register the operator-data extraction classes and functions in the Python module.
pub fn register_operator_data(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OperatorDataExtractorPy>()?;
    Ok(())
}
