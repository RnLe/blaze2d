//! Simple Lanczos-style eigensolver utilities for Θ operators.

use std::cmp::Ordering;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    operator::LinearOperator,
    preconditioner::OperatorPreconditioner,
};

const MIN_KRYLOV_EXCESS: usize = 4;
const JACOBI_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EigenOptions {
    pub n_bands: usize,
    pub max_iter: usize,
    pub tol: f64,
    #[serde(default)]
    pub preconditioner: PreconditionerKind,
    #[serde(default)]
    pub gamma: GammaHandling,
}

impl Default for EigenOptions {
    fn default() -> Self {
        Self {
            n_bands: 8,
            max_iter: 200,
            tol: 1e-8,
            preconditioner: PreconditionerKind::None,
            gamma: GammaHandling::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct GammaHandling {
    pub enabled: bool,
    pub tolerance: f64,
}

impl GammaHandling {
    pub fn should_deflate(self, bloch_norm: f64) -> bool {
        self.enabled && bloch_norm <= self.tolerance
    }
}

impl Default for GammaHandling {
    fn default() -> Self {
        Self {
            enabled: true,
            tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GammaContext {
    pub is_gamma: bool,
}

impl GammaContext {
    pub const fn new(is_gamma: bool) -> Self {
        Self { is_gamma }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerKind {
    None,
    RealSpaceJacobi,
}

impl Default for PreconditionerKind {
    fn default() -> Self {
        PreconditionerKind::None
    }
}

#[derive(Debug, Clone)]
pub struct EigenResult {
    pub omegas: Vec<f64>,
    pub iterations: usize,
    pub gamma_deflated: bool,
}

pub fn solve_lowest_eigenpairs<O, B>(
    operator: &mut O,
    opts: &EigenOptions,
    mut preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    gamma: GammaContext,
) -> EigenResult
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let target_bands = opts.n_bands.max(1);
    let krylov_dim = opts.max_iter.max(target_bands + MIN_KRYLOV_EXCESS);
    let mut q_prev = operator.alloc_field();
    zero_buffer(q_prev.as_mut_slice());
    let mut mass_q_prev = operator.alloc_field();
    zero_buffer(mass_q_prev.as_mut_slice());
    let mut q = operator.alloc_field();
    seed_vector(q.as_mut_slice());
    let mut mass_q = operator.alloc_field();
    let gamma_mode = if gamma.is_gamma {
        build_gamma_mode(operator)
    } else {
        None
    };
    let gamma_deflated = gamma_mode.is_some();
    operator.apply_mass(&q, &mut mass_q);
    if let Some(ref mode) = gamma_mode {
        reorthogonalize_with_mass(operator.backend(), &mut q, &mode.vector, &mode.mass_vector);
        operator.apply_mass(&q, &mut mass_q);
    }
    normalize_with_mass_precomputed(operator.backend(), &mut q, &mut mass_q);
    let mut w = operator.alloc_field();
    let mut mass_w = operator.alloc_field();
    let mut alphas = Vec::new();
    let mut betas = Vec::new();
    let mut last_beta = 0.0;

    for step in 0..krylov_dim {
        operator.apply(&q, &mut w);
        if step > 0 {
            operator
                .backend()
                .axpy(Complex64::new(-last_beta, 0.0), &mass_q_prev, &mut w);
        }
        let alpha = operator.backend().dot(&mass_q, &w).re;
        alphas.push(alpha);
        operator
            .backend()
            .axpy(Complex64::new(-alpha, 0.0), &mass_q, &mut w);
        if step > 0 {
            reorthogonalize_with_mass(operator.backend(), &mut w, &q_prev, &mass_q_prev);
        }
        reorthogonalize_with_mass(operator.backend(), &mut w, &q, &mass_q);
        if let Some(ref mode) = gamma_mode {
            reorthogonalize_with_mass(operator.backend(), &mut w, &mode.vector, &mode.mass_vector);
        }
        if let Some(precond) = preconditioner.as_deref_mut() {
            precond.apply(operator.backend(), &mut w);
        }
        operator.apply_mass(&w, &mut mass_w);
        let beta = mass_norm(operator.backend(), &w, &mass_w);
        let converged = beta < opts.tol || step + 1 == krylov_dim;
        if converged {
            break;
        }
        betas.push(beta);
        last_beta = beta;
        operator
            .backend()
            .scale(Complex64::new(1.0 / beta, 0.0), &mut w);
        operator
            .backend()
            .scale(Complex64::new(1.0 / beta, 0.0), &mut mass_w);
        std::mem::swap(&mut q_prev, &mut q);
        std::mem::swap(&mut mass_q_prev, &mut mass_q);
        std::mem::swap(&mut q, &mut w);
        std::mem::swap(&mut mass_q, &mut mass_w);
    }

    if alphas.is_empty() {
        return EigenResult {
            omegas: Vec::new(),
            iterations: 0,
            gamma_deflated,
        };
    }

    let dense_tridiag = build_tridiagonal_dense(&alphas, &betas);
    let evals = jacobi_eigenvalues(dense_tridiag, alphas.len(), opts.tol.max(JACOBI_EPS));
    let mut omegas: Vec<f64> = evals
        .into_iter()
        .filter(|&val| val >= 0.0)
        .map(|val| val.sqrt())
        .collect();
    omegas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    omegas.truncate(target_bands);
    EigenResult {
        omegas,
        iterations: alphas.len(),
        gamma_deflated,
    }
}

pub struct PowerIterationOptions {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for PowerIterationOptions {
    fn default() -> Self {
        Self {
            max_iter: 128,
            tol: 1e-9,
        }
    }
}

pub fn power_iteration<O, B>(
    operator: &mut O,
    vector: &mut B::Buffer,
    opts: &PowerIterationOptions,
) -> f64
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut eig = 0.0;
    let mut applied = operator.alloc_field();
    let mut mass_vec = operator.alloc_field();
    normalize_with_mass(operator, vector, &mut mass_vec);
    let mut mass_applied = operator.alloc_field();
    for _ in 0..opts.max_iter {
        operator.apply(vector, &mut applied);
        operator.apply_mass(&applied, &mut mass_applied);
        let numerator = operator.backend().dot(vector, &mass_applied).re;
        let denom = operator
            .backend()
            .dot(vector, &mass_vec)
            .re
            .max(f64::EPSILON);
        let new_eig = numerator / denom;
        normalize_with_mass_precomputed(operator.backend(), &mut applied, &mut mass_applied);
        vector.as_mut_slice().copy_from_slice(applied.as_slice());
        mass_vec
            .as_mut_slice()
            .copy_from_slice(mass_applied.as_slice());
        if (new_eig - eig).abs() < opts.tol {
            eig = new_eig;
            break;
        }
        eig = new_eig;
    }
    eig
}

fn normalize_with_mass<O, B>(
    operator: &mut O,
    vector: &mut B::Buffer,
    mass_vec: &mut B::Buffer,
) -> f64
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    operator.apply_mass(vector, mass_vec);
    normalize_with_mass_precomputed(operator.backend(), vector, mass_vec)
}

fn normalize_with_mass_precomputed<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass_vec: &mut B::Buffer,
) -> f64 {
    let norm_sq = backend.dot(vector, mass_vec).re.max(0.0);
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        let scale = Complex64::new(1.0 / norm, 0.0);
        backend.scale(scale, vector);
        backend.scale(scale, mass_vec);
    }
    norm
}

fn mass_norm<B: SpectralBackend>(backend: &B, vector: &B::Buffer, mass_vec: &B::Buffer) -> f64 {
    backend.dot(vector, mass_vec).re.max(0.0).sqrt()
}

fn reorthogonalize_with_mass<B: SpectralBackend>(
    backend: &B,
    target: &mut B::Buffer,
    basis: &B::Buffer,
    mass_basis: &B::Buffer,
) {
    let coeff = backend.dot(mass_basis, target);
    backend.axpy(-coeff, basis, target);
}

struct GammaMode<B: SpectralBackend> {
    vector: B::Buffer,
    mass_vector: B::Buffer,
}

fn build_gamma_mode<O, B>(operator: &mut O) -> Option<GammaMode<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut vector = operator.alloc_field();
    for value in vector.as_mut_slice().iter_mut() {
        *value = Complex64::new(1.0, 0.0);
    }
    let mut mass_vector = operator.alloc_field();
    operator.apply_mass(&vector, &mut mass_vector);
    let norm = normalize_with_mass_precomputed(operator.backend(), &mut vector, &mut mass_vector);
    if norm == 0.0 {
        return None;
    }
    Some(GammaMode {
        vector,
        mass_vector,
    })
}

fn seed_vector(data: &mut [Complex64]) {
    for (idx, value) in data.iter_mut().enumerate() {
        let angle = (idx as f64 + 1.0).sin();
        *value = Complex64::new(angle, 0.0);
    }
}

fn zero_buffer(data: &mut [Complex64]) {
    for value in data.iter_mut() {
        *value = Complex64::default();
    }
}

fn build_tridiagonal_dense(alpha: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = alpha.len();
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = alpha[i];
        if i + 1 < n {
            let b = beta.get(i).copied().unwrap_or(0.0);
            mat[i * n + i + 1] = b;
            mat[(i + 1) * n + i] = b;
        }
    }
    mat
}

fn jacobi_eigenvalues(mut matrix: Vec<f64>, n: usize, tol: f64) -> Vec<f64> {
    if n == 1 {
        return vec![matrix[0]];
    }
    let max_sweeps = n * n * 8;
    for _ in 0..max_sweeps {
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = matrix[i * n + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break;
        }
        let idx = |r: usize, c: usize| r * n + c;
        let app = matrix[idx(p, p)];
        let aqq = matrix[idx(q, q)];
        let apq = matrix[idx(p, q)];
        if apq.abs() < tol {
            continue;
        }
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let aip = matrix[idx(i, p)];
            let aiq = matrix[idx(i, q)];
            matrix[idx(i, p)] = c * aip - s * aiq;
            matrix[idx(p, i)] = matrix[idx(i, p)];
            matrix[idx(i, q)] = c * aiq + s * aip;
            matrix[idx(q, i)] = matrix[idx(i, q)];
        }

        let new_app = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        let new_aqq = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        matrix[idx(p, p)] = new_app;
        matrix[idx(q, q)] = new_aqq;
        matrix[idx(p, q)] = 0.0;
        matrix[idx(q, p)] = 0.0;
    }
    (0..n).map(|i| matrix[i * n + i]).collect()
}
