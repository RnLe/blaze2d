//! Simple Lanczos-style eigensolver utilities for Θ operators.

use std::cmp::Ordering;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
#[cfg(debug_assertions)]
use std::cell::RefCell;
#[cfg(debug_assertions)]
use std::collections::VecDeque;

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    field::Field2D,
    operator::LinearOperator,
    preconditioner::OperatorPreconditioner,
    symmetry::{SymmetryOptions, SymmetryProjector},
};

const MIN_RR_TOL: f64 = 1e-12;
const ABSOLUTE_RESIDUAL_GUARD: f64 = 1e-8;
const BLOCK_SIZE_SLACK: usize = 2;
const W_HISTORY_FACTOR: usize = 1;
const PROJECTION_CONDITION_LIMIT: f64 = 1e9;
const RAYLEIGH_RITZ_VALUE_LIMIT: f64 = 1e8;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EigenOptions {
    pub n_bands: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub block_size: usize,
    #[serde(default)]
    pub preconditioner: PreconditionerKind,
    #[serde(default)]
    pub gamma: GammaHandling,
    #[serde(default)]
    pub deflation: DeflationOptions,
    #[serde(default)]
    pub symmetry: SymmetryOptions,
    #[serde(default)]
    pub warm_start: WarmStartOptions,
    #[serde(default)]
    pub debug: SolverDebugOptions,
}

impl Default for EigenOptions {
    fn default() -> Self {
        Self {
            n_bands: 8,
            max_iter: 200,
            tol: 1e-6,
            block_size: 0,
            preconditioner: PreconditionerKind::default(),
            gamma: GammaHandling::default(),
            deflation: DeflationOptions::default(),
            symmetry: SymmetryOptions::default(),
            warm_start: WarmStartOptions::default(),
            debug: SolverDebugOptions::default(),
        }
    }
}

impl EigenOptions {
    fn effective_block_size(&self) -> usize {
        let required = self.n_bands.max(1);
        let target = if self.block_size == 0 {
            required.saturating_add(BLOCK_SIZE_SLACK)
        } else {
            self.block_size
        };
        target.max(required)
    }

    pub(crate) fn enforce_recommended_defaults(&mut self) {
        let recommended_block = self.n_bands.max(1).saturating_add(BLOCK_SIZE_SLACK);
        if self.block_size == 0 || self.block_size < recommended_block {
            self.block_size = recommended_block;
        }
        if !matches!(
            self.preconditioner,
            PreconditionerKind::StructuredDiagonal | PreconditionerKind::HomogeneousJacobi
        ) {
            self.preconditioner = PreconditionerKind::StructuredDiagonal;
        }
        if self.warm_start.enabled && self.warm_start.max_vectors == 0 {
            self.warm_start.max_vectors = self.n_bands.max(1);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeflationOptions {
    pub enabled: bool,
    pub max_vectors: usize,
}

impl Default for DeflationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_vectors: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WarmStartOptions {
    pub enabled: bool,
    pub max_vectors: usize,
}

impl Default for WarmStartOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_vectors: 0,
        }
    }
}

impl WarmStartOptions {
    pub fn effective_limit(&self, fallback: usize) -> usize {
        if !self.enabled {
            return 0;
        }
        if self.max_vectors == 0 {
            fallback.max(1)
        } else {
            self.max_vectors
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverDebugOptions {
    pub disable_deflation: bool,
    pub history_multiplier: Option<usize>,
}

impl Default for SolverDebugOptions {
    fn default() -> Self {
        Self {
            disable_deflation: false,
            history_multiplier: None,
        }
    }
}

impl SolverDebugOptions {
    pub fn history_factor(&self) -> usize {
        self.history_multiplier.unwrap_or(W_HISTORY_FACTOR)
    }
}

impl DeflationOptions {
    pub fn effective_limit(&self, fallback: usize) -> usize {
        if !self.enabled {
            return 0;
        }
        if self.max_vectors == 0 {
            fallback.max(1)
        } else {
            self.max_vectors
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

    pub fn context_for_bloch(self, bloch_norm: f64) -> GammaContext {
        let is_gamma = bloch_norm <= self.tolerance;
        if !is_gamma {
            return GammaContext::default();
        }
        if self.enabled {
            GammaContext::new(true)
        } else {
            GammaContext::with_deflation(true, false)
        }
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
    pub deflate_zero_mode: bool,
}

impl GammaContext {
    pub const fn new(is_gamma: bool) -> Self {
        Self {
            is_gamma,
            deflate_zero_mode: is_gamma,
        }
    }

    pub const fn with_deflation(is_gamma: bool, deflate_zero_mode: bool) -> Self {
        Self {
            is_gamma,
            deflate_zero_mode: is_gamma && deflate_zero_mode,
        }
    }

    pub const fn gamma_without_deflation() -> Self {
        Self {
            is_gamma: true,
            deflate_zero_mode: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerKind {
    None,
    #[serde(
        alias = "fourier_diagonal",
        alias = "real_space_jacobi",
        alias = "homogeneous"
    )]
    HomogeneousJacobi,
    #[serde(alias = "structured", alias = "epsilon_aware")]
    StructuredDiagonal,
}

impl Default for PreconditionerKind {
    fn default() -> Self {
        PreconditionerKind::StructuredDiagonal
    }
}

#[derive(Debug, Clone)]
pub struct EigenResult {
    pub omegas: Vec<f64>,
    pub iterations: usize,
    pub gamma_deflated: bool,
    pub modes: Vec<Field2D>,
    pub diagnostics: EigenDiagnostics,
    pub warm_start_hits: usize,
}

#[derive(Debug, Clone)]
pub struct EigenDiagnostics {
    pub freq_tolerance: f64,
    pub duplicate_modes_skipped: usize,
    pub negative_modes_skipped: usize,
    pub max_residual: f64,
    pub max_relative_residual: f64,
    pub max_relative_scale: f64,
    pub modes: Vec<ModeDiagnostics>,
    pub iterations: Vec<IterationDiagnostics>,
    pub residual_snapshots: Vec<ResidualSnapshot>,
    pub deflation_vectors: usize,
    pub deflation_disabled: bool,
}

impl EigenDiagnostics {
    pub fn new(freq_tolerance: f64) -> Self {
        Self {
            freq_tolerance,
            duplicate_modes_skipped: 0,
            negative_modes_skipped: 0,
            max_residual: 0.0,
            max_relative_residual: 0.0,
            max_relative_scale: 0.0,
            modes: Vec::new(),
            iterations: Vec::new(),
            residual_snapshots: Vec::new(),
            deflation_vectors: 0,
            deflation_disabled: false,
        }
    }

    pub fn avg_residual(&self) -> f64 {
        if self.modes.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.modes.iter().map(|m| m.residual_norm).sum();
        sum / self.modes.len() as f64
    }

    pub fn avg_relative_residual(&self) -> f64 {
        if self.modes.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.modes.iter().map(|m| m.relative_residual).sum();
        sum / self.modes.len() as f64
    }

    pub fn max_mass_error(&self) -> f64 {
        self.modes
            .iter()
            .map(|m| (m.mass_norm - 1.0).abs())
            .fold(0.0, f64::max)
    }
}

impl Default for EigenDiagnostics {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct ModeDiagnostics {
    pub omega: f64,
    pub lambda: f64,
    pub residual_norm: f64,
    pub mass_norm: f64,
    pub relative_residual: f64,
}

#[derive(Debug, Clone)]
pub struct IterationDiagnostics {
    pub iteration: usize,
    pub max_residual: f64,
    pub avg_residual: f64,
    pub max_relative_residual: f64,
    pub avg_relative_residual: f64,
    pub max_relative_scale: f64,
    pub avg_relative_scale: f64,
    pub block_size: usize,
    pub new_directions: usize,
    pub preconditioner_trials: usize,
    pub preconditioner_avg_before: f64,
    pub preconditioner_avg_after: f64,
    pub projection: ProjectionDiagnostics,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectionDiagnostics {
    pub original_dim: usize,
    pub reduced_dim: usize,
    pub requested_dim: usize,
    pub history_dim: usize,
    pub min_mass_eigenvalue: f64,
    pub max_mass_eigenvalue: f64,
    pub condition_estimate: f64,
    pub fallback_used: bool,
}

#[derive(Debug, Clone)]
pub struct ResidualSnapshot {
    pub iteration: usize,
    pub band_index: usize,
    pub stage: ResidualSnapshotStage,
    pub field: Field2D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualSnapshotStage {
    Raw,
    Projected,
    Preconditioned,
}

impl ResidualSnapshotStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            ResidualSnapshotStage::Raw => "raw",
            ResidualSnapshotStage::Projected => "projected",
            ResidualSnapshotStage::Preconditioned => "preconditioned",
        }
    }
}

impl std::fmt::Display for ResidualSnapshotStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct ResidualSnapshotRequest {
    max_snapshots: usize,
    capture_raw: bool,
    capture_projected: bool,
    capture_preconditioned: bool,
}

impl ResidualSnapshotRequest {
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            max_snapshots,
            capture_raw: true,
            capture_projected: true,
            capture_preconditioned: true,
        }
    }

    pub fn with_stages(
        max_snapshots: usize,
        capture_raw: bool,
        capture_projected: bool,
        capture_preconditioned: bool,
    ) -> Self {
        Self {
            max_snapshots,
            capture_raw,
            capture_projected,
            capture_preconditioned,
        }
    }

    pub fn allows_stage(&self, stage: ResidualSnapshotStage) -> bool {
        match stage {
            ResidualSnapshotStage::Raw => self.capture_raw,
            ResidualSnapshotStage::Projected => self.capture_projected,
            ResidualSnapshotStage::Preconditioned => self.capture_preconditioned,
        }
    }

    pub fn max_snapshots(&self) -> usize {
        self.max_snapshots
    }
}

struct ResidualSnapshotManager {
    request: ResidualSnapshotRequest,
    captured: usize,
}

impl ResidualSnapshotManager {
    fn new(request: ResidualSnapshotRequest) -> Self {
        Self {
            request,
            captured: 0,
        }
    }

    fn can_capture(&self) -> bool {
        self.request.max_snapshots > 0 && self.captured < self.request.max_snapshots
    }

    fn capture<O, B>(
        &mut self,
        operator: &O,
        iteration: usize,
        band_index: usize,
        stage: ResidualSnapshotStage,
        buffer: &B::Buffer,
        store: &mut Vec<ResidualSnapshot>,
    ) where
        O: LinearOperator<B>,
        B: SpectralBackend,
    {
        if !self.can_capture() || !self.request.allows_stage(stage) {
            return;
        }
        let grid = operator.grid();
        let data = buffer.as_slice().to_vec();
        store.push(ResidualSnapshot {
            iteration,
            band_index,
            stage,
            field: Field2D::from_vec(grid, data),
        });
        self.captured += 1;
    }
}

pub struct DeflationWorkspace<B: SpectralBackend> {
    entries: Vec<DeflationVector<B>>,
}

impl<B: SpectralBackend> DeflationWorkspace<B> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn project(&self, backend: &B, vector: &mut B::Buffer, mass_vector: &mut B::Buffer) {
        for entry in &self.entries {
            let coeff = backend.dot(vector, &entry.mass_vector);
            backend.axpy(-coeff, &entry.vector, vector);
            backend.axpy(-coeff, &entry.mass_vector, mass_vector);
        }
    }

    pub fn push(&mut self, vector: B::Buffer, mass_vector: B::Buffer) {
        self.entries.push(DeflationVector {
            vector,
            mass_vector,
        });
    }
}

struct DeflationVector<B: SpectralBackend> {
    vector: B::Buffer,
    mass_vector: B::Buffer,
}

pub fn build_deflation_workspace<'a, O, B>(
    operator: &mut O,
    fields: impl IntoIterator<Item = &'a Field2D>,
) -> DeflationWorkspace<B>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut workspace = DeflationWorkspace::new();
    for field in fields {
        let mut vector = operator.alloc_field();
        vector.as_mut_slice().copy_from_slice(field.as_slice());
        let mut mass_vector = operator.alloc_field();
        operator.apply_mass(&vector, &mut mass_vector);
        let norm =
            normalize_with_mass_precomputed(operator.backend(), &mut vector, &mut mass_vector);
        if norm == 0.0 {
            continue;
        }
        workspace.push(vector, mass_vector);
    }
    workspace
}

pub fn solve_lowest_eigenpairs<O, B>(
    operator: &mut O,
    opts: &EigenOptions,
    mut preconditioner: Option<&mut dyn OperatorPreconditioner<B>>,
    gamma: GammaContext,
    warm_start: Option<&[Field2D]>,
    deflation: Option<&DeflationWorkspace<B>>,
    symmetry_override: Option<&SymmetryProjector>,
    residual_request: Option<ResidualSnapshotRequest>,
) -> EigenResult
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    #[cfg(debug_assertions)]
    let _rayleigh_ritz_log_scope = RayleighRitzLogScope::new();

    let target_bands = opts.n_bands.max(1);
    let block_size = if opts.max_iter == 0 {
        warm_start
            .map(|seeds| seeds.len().max(target_bands))
            .unwrap_or_else(|| opts.effective_block_size())
    } else {
        opts.effective_block_size()
    };
    let gamma_mode = if gamma.is_gamma {
        let mode = build_gamma_mode(operator);
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "[gamma-debug] gamma_mode_present={} deflate={}",
                mode.is_some(),
                gamma.deflate_zero_mode
            );
        }
        mode
    } else {
        None
    };
    let gamma_deflated = gamma_mode.is_some() && gamma.deflate_zero_mode;
    let gamma_constraint = if gamma_deflated {
        gamma_mode.as_ref()
    } else {
        None
    };
    let deflation_vectors = deflation.map_or(0, |space| space.len());
    let deflation_active = if opts.debug.disable_deflation {
        None
    } else {
        deflation
    };
    let fallback_symmetry = if symmetry_override.is_none() {
        SymmetryProjector::from_options(&opts.symmetry)
    } else {
        None
    };
    let symmetry_projector = symmetry_override.or(fallback_symmetry.as_ref());
    let mut residual_snapshots = Vec::new();
    let mut snapshot_manager = residual_request.map(ResidualSnapshotManager::new);

    let (mut x_entries, warm_start_hits) = initialize_block(
        operator,
        block_size,
        gamma_constraint,
        deflation_active,
        symmetry_projector,
        warm_start,
    );
    #[cfg(test)]
    eprintln!(
        "[eigensolver-debug] initial block size={} requested={}",
        x_entries.len(),
        block_size
    );
    #[cfg(test)]
    if operator.grid().len() <= 4 {
        for (idx, entry) in x_entries.iter().enumerate() {
            eprintln!(
                "[eigensolver-debug] init vec {} = {:?}",
                idx,
                entry.vector.as_slice()
            );
        }
    }
    if x_entries.is_empty() {
        return EigenResult {
            omegas: Vec::new(),
            iterations: 0,
            gamma_deflated,
            modes: Vec::new(),
            diagnostics: EigenDiagnostics::default(),
            warm_start_hits,
        };
    }

    if let (Some(mode), false) = (gamma_mode.as_ref(), gamma_deflated) {
        inject_gamma_seed(operator, &mut x_entries, None, mode, block_size);
    }

    if opts.max_iter == 0 && warm_start.is_some() {
        let subspace = build_subspace_entries(&x_entries, &[], &[]);
        let (eigenvalues, new_entries) =
            solve_raw_projected_system(operator, &subspace, block_size, opts.tol.max(MIN_RR_TOL));
        let (omegas, modes, mut diagnostics) = finalize_modes(
            operator,
            &new_entries,
            &eigenvalues,
            target_bands,
            opts.tol,
            gamma_deflated,
        );
        diagnostics.deflation_vectors = deflation_vectors;
        diagnostics.deflation_disabled = deflation.is_some() && deflation_active.is_none();
        return EigenResult {
            omegas,
            iterations: 0,
            gamma_deflated,
            modes,
            diagnostics,
            warm_start_hits,
        };
    }

    let mut eigenvalues;
    {
        let subspace = build_subspace_entries(&x_entries, &[], &[]);
        let (vals, new_entries, _) = rayleigh_ritz(
            operator,
            &subspace,
            x_entries.len(),
            0,
            opts.tol.max(MIN_RR_TOL),
        );
        eigenvalues = vals;
        x_entries = new_entries;
    }

    let mut w_entries: Vec<BlockEntry<B>> = Vec::new();
    let mut iterations = 0usize;
    let mut iteration_stats: Vec<IterationDiagnostics> = Vec::new();

    loop {
        if let (Some(mode), false) = (gamma_mode.as_ref(), gamma_deflated) {
            if !x_entries.is_empty() {
                inject_gamma_seed(
                    operator,
                    &mut x_entries,
                    Some(&mut eigenvalues),
                    mode,
                    block_size,
                );
            }
        }
        let history_factor = opts.debug.history_factor();
        let history_enabled = history_factor > 0 && !x_entries.is_empty();
        if history_enabled {
            let history_limit = history_factor
                .saturating_mul(x_entries.len())
                .max(history_factor);
            while w_entries.len() > history_limit {
                w_entries.pop();
            }
        } else if !w_entries.is_empty() {
            w_entries.clear();
        }

        let (residual_stats, mut p_entries) = compute_preconditioned_residuals(
            operator,
            &eigenvalues,
            &x_entries,
            if history_enabled {
                w_entries.as_slice()
            } else {
                &[] as &[BlockEntry<B>]
            },
            gamma_constraint,
            deflation_active,
            symmetry_projector,
            &mut preconditioner,
            opts.tol,
            iterations,
            snapshot_manager.as_mut(),
            &mut residual_snapshots,
        );
        iteration_stats.push(IterationDiagnostics {
            iteration: iterations,
            max_residual: residual_stats.max_residual,
            avg_residual: residual_stats.avg_residual,
            max_relative_residual: residual_stats.max_relative_residual,
            avg_relative_residual: residual_stats.avg_relative_residual,
            max_relative_scale: residual_stats.max_relative_scale,
            avg_relative_scale: residual_stats.avg_relative_scale,
            block_size: x_entries.len(),
            new_directions: residual_stats.accepted,
            preconditioner_trials: residual_stats.preconditioner_trials,
            preconditioner_avg_before: residual_stats.preconditioner_avg_before,
            preconditioner_avg_after: residual_stats.preconditioner_avg_after,
            projection: ProjectionDiagnostics::default(),
        });
        if residual_stats.max_relative_residual <= opts.tol
            || p_entries.is_empty()
            || iterations >= opts.max_iter
        {
            break;
        }

        reorthogonalize_block(operator, &mut p_entries, &x_entries);
        if history_enabled && !w_entries.is_empty() {
            let history_slice = w_entries.as_slice();
            reorthogonalize_block(operator, &mut p_entries, history_slice);
        }
        if history_enabled {
            reorthogonalize_block(operator, &mut w_entries, &x_entries);
        }

        let w_slice: &[BlockEntry<B>] = if history_enabled {
            w_entries.as_slice()
        } else {
            &[]
        };
        let subspace = build_subspace_entries(&x_entries, &p_entries, w_slice);
        let history_dim = w_slice.len();
        let (vals, new_entries, projection_diag) = rayleigh_ritz(
            operator,
            &subspace,
            x_entries.len(),
            history_dim,
            opts.tol.max(MIN_RR_TOL),
        );
        if let Some(latest) = iteration_stats.last_mut() {
            latest.projection = projection_diag;
        }
        eigenvalues = vals;
        x_entries = new_entries;
        if history_enabled {
            w_entries = p_entries;
        }
        iterations += 1;
    }

    let (omegas, modes, mut diagnostics) = finalize_modes(
        operator,
        &x_entries,
        &eigenvalues,
        target_bands,
        opts.tol,
        gamma_deflated,
    );
    diagnostics.iterations = iteration_stats;
    diagnostics.residual_snapshots = residual_snapshots;
    diagnostics.deflation_vectors = deflation_vectors;
    diagnostics.deflation_disabled = opts.debug.disable_deflation;
    if let Some(iter_max) = diagnostics
        .iterations
        .iter()
        .map(|info| info.max_residual)
        .reduce(f64::max)
    {
        diagnostics.max_residual = diagnostics.max_residual.max(iter_max);
    }
    if let Some(scale_max) = diagnostics
        .iterations
        .iter()
        .map(|info| info.max_relative_scale)
        .reduce(f64::max)
    {
        diagnostics.max_relative_scale = diagnostics.max_relative_scale.max(scale_max);
    }
    EigenResult {
        omegas,
        iterations,
        gamma_deflated,
        modes,
        diagnostics,
        warm_start_hits,
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
    let coeff = backend.dot(target, mass_basis);
    backend.axpy(-coeff, basis, target);
}

struct GammaMode<B: SpectralBackend> {
    vector: B::Buffer,
    mass_vector: B::Buffer,
}

struct BlockEntry<B: SpectralBackend> {
    vector: B::Buffer,
    mass: B::Buffer,
    applied: B::Buffer,
}

struct SubspaceEntry<'a, B: SpectralBackend> {
    vector: &'a B::Buffer,
    mass: &'a B::Buffer,
    applied: &'a B::Buffer,
}

struct ResidualComputation {
    max_residual: f64,
    avg_residual: f64,
    max_relative_residual: f64,
    avg_relative_residual: f64,
    max_relative_scale: f64,
    avg_relative_scale: f64,
    accepted: usize,
    preconditioner_trials: usize,
    preconditioner_avg_before: f64,
    preconditioner_avg_after: f64,
}

fn residual_relative_scale<B: SpectralBackend>(
    backend: &B,
    entry: &BlockEntry<B>,
    lambda: f64,
) -> f64 {
    let vector_norm = mass_norm(backend, &entry.vector, &entry.mass);
    let rayleigh_scale = lambda.abs() * vector_norm.max(1e-12);
    let op_norm = backend
        .dot(&entry.applied, &entry.applied)
        .re
        .max(0.0)
        .sqrt();
    let combined = rayleigh_scale.max(op_norm);
    if combined > 0.0 {
        combined
    } else {
        vector_norm
    }
}

fn compute_relative_residual_with_scale<B: SpectralBackend>(
    backend: &B,
    entry: &BlockEntry<B>,
    lambda: f64,
    residual_norm: f64,
) -> (f64, f64) {
    let scale = residual_relative_scale(backend, entry, lambda);
    let relative = if scale > 0.0 {
        residual_norm / scale
    } else {
        residual_norm
    };
    (relative, scale)
}

fn compute_relative_residual<B: SpectralBackend>(
    backend: &B,
    entry: &BlockEntry<B>,
    lambda: f64,
    residual_norm: f64,
) -> f64 {
    compute_relative_residual_with_scale(backend, entry, lambda, residual_norm).0
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

fn seed_block_vector(data: &mut [Complex64], phase: f64) {
    let mut state = phase.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if state == 0 {
        state = 0xDEAD_BEEF_CAFE_BABE;
    }
    for value in data.iter_mut() {
        state = state ^ (state << 13);
        state = state ^ (state >> 7);
        state = state ^ (state << 17);
        let real = ((state >> 12) as f64) / ((1u64 << 52) as f64) * 2.0 - 1.0;
        *value = Complex64::new(real, 0.0);
    }
}

fn zero_buffer(data: &mut [Complex64]) {
    for value in data.iter_mut() {
        *value = Complex64::default();
    }
}

fn jacobi_eigendecomposition(mut matrix: Vec<f64>, n: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    if n == 1 {
        return (vec![matrix[0]], vec![1.0]);
    }
    let max_sweeps = n * n * 8;
    let mut eigenvectors = vec![0.0; n * n];
    for i in 0..n {
        eigenvectors[i * n + i] = 1.0;
    }
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

        for i in 0..n {
            let vip = eigenvectors[i * n + p];
            let viq = eigenvectors[i * n + q];
            eigenvectors[i * n + p] = c * vip - s * viq;
            eigenvectors[i * n + q] = s * vip + c * viq;
        }
    }
    let eigenvalues = (0..n).map(|i| matrix[i * n + i]).collect();
    (eigenvalues, eigenvectors)
}

fn enforce_constraints<O, B>(
    operator: &mut O,
    vector: &mut B::Buffer,
    mass_vector: &mut B::Buffer,
    gamma_mode: Option<&GammaMode<B>>,
    deflation: Option<&DeflationWorkspace<B>>,
    symmetry: Option<&SymmetryProjector>,
) where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    if let Some(sym) = symmetry {
        sym.apply(vector);
        operator.apply_mass(vector, mass_vector);
    }
    if let Some(mode) = gamma_mode {
        let mut passes = 0;
        loop {
            reorthogonalize_with_mass(operator.backend(), vector, &mode.vector, &mode.mass_vector);
            operator.apply_mass(vector, mass_vector);
            let overlap = operator.backend().dot(vector, &mode.mass_vector).norm();
            passes += 1;
            if overlap <= 1e-12 || passes >= 3 {
                break;
            }
        }
    }
    if let Some(space) = deflation {
        space.project(operator.backend(), vector, mass_vector);
    }
}

fn initialize_block<O, B>(
    operator: &mut O,
    count: usize,
    gamma_mode: Option<&GammaMode<B>>,
    deflation: Option<&DeflationWorkspace<B>>,
    symmetry: Option<&SymmetryProjector>,
    warm_start: Option<&[Field2D]>,
) -> (Vec<BlockEntry<B>>, usize)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut entries = Vec::with_capacity(count);
    let mut warm_hits = 0usize;
    if let Some(seeds) = warm_start {
        for field in seeds.iter().take(count) {
            if let Some(entry) =
                build_entry_from_field(operator, field, gamma_mode, deflation, symmetry, &entries)
            {
                entries.push(entry);
                warm_hits += 1;
            }
            if entries.len() == count {
                break;
            }
        }
    }
    let mut attempts = 0usize;
    let max_attempts = count.max(1) * 8;
    let mut phase = 1.0;
    while entries.len() < count && attempts < max_attempts {
        let mut vector = operator.alloc_field();
        seed_block_vector(vector.as_mut_slice(), phase);
        phase += 1.0;
        match build_entry_from_buffer(operator, vector, gamma_mode, deflation, symmetry, &entries) {
            Some(entry) => entries.push(entry),
            None => {
                attempts += 1;
            }
        }
    }
    (entries, warm_hits)
}

fn inject_gamma_seed<O, B>(
    operator: &mut O,
    entries: &mut Vec<BlockEntry<B>>,
    eigenvalues: Option<&mut Vec<f64>>,
    gamma_mode: &GammaMode<B>,
    block_limit: usize,
) where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    if block_limit == 0 {
        return;
    }
    let mut applied = operator.alloc_field();
    operator.apply(&gamma_mode.vector, &mut applied);
    let gamma_entry = BlockEntry {
        vector: gamma_mode.vector.clone(),
        mass: gamma_mode.mass_vector.clone(),
        applied,
    };
    let mut rest = std::mem::take(entries);
    reorthogonalize_block(operator, &mut rest, std::slice::from_ref(&gamma_entry));
    let mut combined = vec![gamma_entry];
    combined.append(&mut rest);
    if combined.len() > block_limit {
        combined.truncate(block_limit);
    }
    *entries = combined;
    if let Some(values) = eigenvalues {
        if entries.is_empty() {
            values.clear();
            return;
        }
        values.insert(0, 0.0);
        if values.len() > entries.len() {
            values.truncate(entries.len());
        } else if values.len() < entries.len() {
            values.resize(entries.len(), 0.0);
        }
    }
}

fn build_entry_from_field<O, B>(
    operator: &mut O,
    field: &Field2D,
    gamma_mode: Option<&GammaMode<B>>,
    deflation: Option<&DeflationWorkspace<B>>,
    symmetry: Option<&SymmetryProjector>,
    basis: &[BlockEntry<B>],
) -> Option<BlockEntry<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut vector = operator.alloc_field();
    vector.as_mut_slice().copy_from_slice(field.as_slice());
    build_entry_from_buffer(operator, vector, gamma_mode, deflation, symmetry, basis)
}

fn build_entry_from_buffer<O, B>(
    operator: &mut O,
    mut vector: B::Buffer,
    gamma_mode: Option<&GammaMode<B>>,
    deflation: Option<&DeflationWorkspace<B>>,
    symmetry: Option<&SymmetryProjector>,
    basis: &[BlockEntry<B>],
) -> Option<BlockEntry<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut mass = operator.alloc_field();
    operator.apply_mass(&vector, &mut mass);
    enforce_constraints(
        operator,
        &mut vector,
        &mut mass,
        gamma_mode,
        deflation,
        symmetry,
    );
    project_against_entries(operator.backend(), &mut vector, &mut mass, basis);
    let norm = mass_norm(operator.backend(), &vector, &mass);
    if norm <= 1e-12 {
        return None;
    }
    let scale = Complex64::new(1.0 / norm, 0.0);
    operator.backend().scale(scale, &mut vector);
    operator.backend().scale(scale, &mut mass);
    let mut applied = operator.alloc_field();
    operator.apply(&vector, &mut applied);
    Some(BlockEntry {
        vector,
        mass,
        applied,
    })
}

fn project_against_entries<B: SpectralBackend>(
    backend: &B,
    vector: &mut B::Buffer,
    mass: &mut B::Buffer,
    basis: &[BlockEntry<B>],
) {
    for entry in basis {
        let coeff = backend.dot(vector, &entry.mass);
        backend.axpy(-coeff, &entry.vector, vector);
        backend.axpy(-coeff, &entry.mass, mass);
    }
}

fn build_subspace_entries<'a, B: SpectralBackend>(
    primary: &'a [BlockEntry<B>],
    p: &'a [BlockEntry<B>],
    w: &'a [BlockEntry<B>],
) -> Vec<SubspaceEntry<'a, B>> {
    let mut entries = Vec::with_capacity(primary.len() + p.len() + w.len());
    for entry in primary {
        entries.push(SubspaceEntry {
            vector: &entry.vector,
            mass: &entry.mass,
            applied: &entry.applied,
        });
    }
    for entry in p {
        entries.push(SubspaceEntry {
            vector: &entry.vector,
            mass: &entry.mass,
            applied: &entry.applied,
        });
    }
    for entry in w {
        entries.push(SubspaceEntry {
            vector: &entry.vector,
            mass: &entry.mass,
            applied: &entry.applied,
        });
    }
    entries
}

fn rayleigh_ritz<O, B>(
    operator: &mut O,
    subspace: &[SubspaceEntry<'_, B>],
    want: usize,
    history_dim: usize,
    tol: f64,
) -> (Vec<f64>, Vec<BlockEntry<B>>, ProjectionDiagnostics)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let dim = subspace.len();
    if dim == 0 {
        return (
            Vec::new(),
            Vec::new(),
            ProjectionDiagnostics {
                history_dim,
                ..ProjectionDiagnostics::default()
            },
        );
    }
    let history_dim = history_dim.min(dim);
    let mut fallback_used = false;
    let mut final_values = Vec::new();
    let mut final_entries = Vec::new();
    let mut final_stats = ProjectionStats {
        original_dim: dim,
        requested_dim: want,
        ..ProjectionStats::default()
    };
    let mut require_fallback;
    if let Some((values, entries, stats)) = attempt_projection(operator, subspace, want, tol) {
        require_fallback = projection_guard_triggered(&stats);
        #[cfg(debug_assertions)]
        {
            rayleigh_ritz_debug_log(format!(
                "[rayleigh-ritz-debug] attempt stats: reduced={} cond={} min_mass={} max_mass={} fallback_guard={}",
                stats.reduced_dim,
                stats.ratio(),
                stats.min_mass_eigenvalue,
                stats.max_mass_eigenvalue,
                require_fallback
            ));
        }
        final_values = values;
        final_entries = entries;
        final_stats = stats;
    } else {
        require_fallback = true;
    }

    if require_fallback {
        if history_dim > 0 {
            let trimmed_len = dim.saturating_sub(history_dim);
            if trimmed_len > 0 {
                if let Some((values, entries, stats)) =
                    attempt_projection(operator, &subspace[..trimmed_len], want, tol)
                {
                    fallback_used = true;
                    final_values = values;
                    final_entries = entries;
                    final_stats = stats;
                    require_fallback = projection_guard_triggered(&final_stats);
                }
            }
        }

        if require_fallback {
            let (raw_values, raw_entries) =
                solve_raw_projected_system(operator, subspace, want, tol);
            if !raw_values.is_empty() && raw_entries.len() == raw_values.len() {
                #[cfg(debug_assertions)]
                {
                    rayleigh_ritz_debug_log(format!(
                        "[rayleigh-ritz-debug] raw solve values={:?}",
                        raw_values
                    ));
                }
                fallback_used = true;
                final_values = raw_values;
                final_entries = raw_entries;
                final_stats.reduced_dim = final_entries.len();
                final_stats.min_mass_eigenvalue = 0.0;
                final_stats.max_mass_eigenvalue = 0.0;
            }
        }
    }
    if final_stats.reduced_dim == 0 && final_values.is_empty() && final_entries.is_empty() {
        log_projected_dimension(dim, 0, want);
        log_rayleigh_ritz_counts(0, 0, want, dim);
        return (
            Vec::new(),
            Vec::new(),
            ProjectionDiagnostics {
                original_dim: dim,
                requested_dim: want,
                history_dim,
                fallback_used,
                reduced_dim: 0,
                min_mass_eigenvalue: 0.0,
                max_mass_eigenvalue: 0.0,
                condition_estimate: f64::INFINITY,
                ..ProjectionDiagnostics::default()
            },
        );
    }
    if final_entries.len() < want && dim >= want {
        let (raw_values, raw_entries) = solve_raw_projected_system(operator, subspace, want, tol);
        if !raw_values.is_empty() && raw_entries.len() == raw_values.len() {
            let lambda_tol = tol.max(MIN_RR_TOL);
            let mut added = 0usize;
            for (raw_value, raw_entry) in raw_values.into_iter().zip(raw_entries.into_iter()) {
                if want > 0 && final_entries.len() >= want {
                    break;
                }
                if final_values
                    .iter()
                    .any(|&existing| (existing - raw_value).abs() <= lambda_tol)
                {
                    continue;
                }
                let insert_pos = final_values
                    .iter()
                    .position(|&existing| raw_value < existing)
                    .unwrap_or(final_values.len());
                final_values.insert(insert_pos, raw_value);
                final_entries.insert(insert_pos, raw_entry);
                added += 1;
                fallback_used = true;
            }
            if added > 0 {
                final_stats.reduced_dim = final_entries.len();
                final_stats.min_mass_eigenvalue = 0.0;
                final_stats.max_mass_eigenvalue = 0.0;
            }
        }
    }
    let diagnostics = final_stats.to_public(history_dim, fallback_used);
    #[cfg(test)]
    {
        if dim <= 4 {
            eprintln!(
                "[rayleigh-ritz-debug] dim={} values={:?} reduced={} requested={} fallback={}",
                dim,
                final_values,
                final_stats.reduced_dim,
                final_stats.requested_dim,
                fallback_used
            );
        }
    }
    log_projected_dimension(final_stats.original_dim, final_stats.reduced_dim, want);
    log_rayleigh_ritz_counts(
        final_values.len(),
        final_entries.len(),
        want,
        final_stats.original_dim,
    );
    #[cfg(debug_assertions)]
    {
        if operator.grid().len() <= 64 {
            rayleigh_ritz_debug_log(format!(
                "[rayleigh-ritz-debug] grid={} eigenvalues={:?}",
                operator.grid().len(),
                final_values
            ));
        }
    }
    (final_values, final_entries, diagnostics)
}

fn attempt_projection<O, B>(
    operator: &mut O,
    subspace: &[SubspaceEntry<'_, B>],
    want: usize,
    tol: f64,
) -> Option<(Vec<f64>, Vec<BlockEntry<B>>, ProjectionStats)>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    if subspace.is_empty() {
        return None;
    }
    let dim = subspace.len();
    let (op_proj, mass_proj) = build_projected_matrices(operator.backend(), subspace);
    let (prepared, mut stats) = stabilize_projected_system(&op_proj, &mass_proj, dim, want, tol)?;
    let (values, entries) =
        solve_projected_eigensystem(operator, subspace, want, tol, dim, prepared);
    stats.original_dim = dim;
    Some((values, entries, stats))
}

fn projection_guard_triggered(stats: &ProjectionStats) -> bool {
    if stats.reduced_dim == 0 {
        return true;
    }
    let cond = stats.ratio();
    !cond.is_finite() || cond > PROJECTION_CONDITION_LIMIT || !stats.satisfies_rank()
}

fn solve_projected_eigensystem<O, B>(
    operator: &mut O,
    subspace: &[SubspaceEntry<'_, B>],
    want: usize,
    tol: f64,
    dim: usize,
    prepared: StabilizedProjection,
) -> (Vec<f64>, Vec<BlockEntry<B>>)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    if prepared.reduced_dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let (values, eigenvectors) = generalized_eigen(
        prepared.op_matrix,
        prepared.mass_matrix,
        prepared.reduced_dim,
        tol,
    )
    .expect("generalized eigen solve failed in block solver");
    #[cfg(test)]
    if prepared.reduced_dim <= 4 {
        eprintln!(
            "[rayleigh-ritz-debug] raw generalized eigen values={:?}",
            values
        );
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
    let mut new_entries = Vec::with_capacity(want.min(dim));
    let mut selected_values: Vec<f64> = Vec::with_capacity(want.min(dim));
    let lambda_tol = tol.max(MIN_RR_TOL);
    let eigen_cols = if prepared.reduced_dim == 0 {
        0
    } else {
        eigenvectors.len() / prepared.reduced_dim
    };
    let lifted_vectors = lift_projected_eigenvectors(
        &prepared.transform,
        dim,
        prepared.reduced_dim,
        &eigenvectors,
        eigen_cols,
    );
    for idx in order {
        if idx >= eigen_cols {
            continue;
        }
        if want > 0 && new_entries.len() >= want {
            break;
        }
        let value = values[idx];
        if !value.is_finite() {
            continue;
        }
        if value < -tol.abs().max(MIN_RR_TOL) {
            continue;
        }
        if value > RAYLEIGH_RITZ_VALUE_LIMIT {
            continue;
        }
        if selected_values
            .iter()
            .any(|existing| (*existing - value).abs() <= lambda_tol)
        {
            continue;
        }
        selected_values.push(value);
        let coeffs = extract_complex_column(&lifted_vectors, dim, eigen_cols, idx);
        let entry = combine_entries(operator, subspace, &coeffs);
        new_entries.push(entry);
    }
    #[cfg(test)]
    if prepared.reduced_dim <= 4 {
        eprintln!(
            "[rayleigh-ritz-debug] selected eigenvalues={:?}",
            selected_values
        );
    }
    (selected_values, new_entries)
}

fn solve_raw_projected_system<O, B>(
    operator: &mut O,
    subspace: &[SubspaceEntry<'_, B>],
    want: usize,
    tol: f64,
) -> (Vec<f64>, Vec<BlockEntry<B>>)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let dim = subspace.len();
    if dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let (op_proj, mass_proj) = build_projected_matrices(operator.backend(), subspace);
    let (values, eigenvectors) = match generalized_eigen(op_proj, mass_proj, dim, tol) {
        Some(res) => res,
        None => return (Vec::new(), Vec::new()),
    };
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
    let eigen_cols = if dim == 0 {
        0
    } else {
        eigenvectors.len() / dim
    };
    if eigen_cols == 0 {
        return (Vec::new(), Vec::new());
    }
    let limit = if want == 0 {
        eigen_cols
    } else {
        want.min(eigen_cols)
    };
    let mut new_entries = Vec::with_capacity(limit);
    let mut selected_values: Vec<f64> = Vec::with_capacity(limit);
    let lambda_tol = tol.max(MIN_RR_TOL);
    for idx in order {
        if idx >= eigen_cols {
            continue;
        }
        if selected_values.len() >= limit {
            break;
        }
        let value = values[idx];
        if !value.is_finite() {
            continue;
        }
        if value < -tol.abs().max(MIN_RR_TOL) {
            continue;
        }
        if value > RAYLEIGH_RITZ_VALUE_LIMIT {
            continue;
        }
        if selected_values
            .iter()
            .any(|existing| (*existing - value).abs() <= lambda_tol)
        {
            continue;
        }
        selected_values.push(value);
        let coeffs = extract_complex_column(&eigenvectors, dim, eigen_cols, idx);
        let entry = combine_entries(operator, subspace, &coeffs);
        new_entries.push(entry);
    }
    (selected_values, new_entries)
}

#[cfg(debug_assertions)]
fn log_rayleigh_ritz_counts(total_values: usize, produced: usize, want: usize, dim: usize) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] values={} produced={} want={} dim={}",
        total_values, produced, want, dim
    ));
}

#[cfg(not(debug_assertions))]
fn log_rayleigh_ritz_counts(_total_values: usize, _produced: usize, _want: usize, _dim: usize) {}

#[cfg(debug_assertions)]
struct AttemptLogBuffer {
    front_limit: usize,
    tail_limit: usize,
    total: usize,
    tail: VecDeque<String>,
    flushed: bool,
}

#[cfg(debug_assertions)]
impl AttemptLogBuffer {
    fn new(front_limit: usize, tail_limit: usize) -> Self {
        Self {
            front_limit,
            tail_limit,
            total: 0,
            tail: VecDeque::with_capacity(tail_limit.max(1)),
            flushed: false,
        }
    }

    fn log(&mut self, message: String) {
        self.total += 1;
        if self.total <= self.front_limit {
            eprintln!("{}", message);
            return;
        }
        self.tail.push_back(message);
        if self.tail.len() > self.tail_limit {
            self.tail.pop_front();
        }
    }

    fn flush(&mut self) {
        if self.flushed {
            return;
        }
        if self.total > self.front_limit {
            let suppressed = self
                .total
                .saturating_sub(self.front_limit + self.tail.len());
            if suppressed > 0 {
                eprintln!(
                    "[rayleigh-ritz-debug] ... suppressed {} intermediate attempt logs ...",
                    suppressed
                );
            }
            for entry in self.tail.iter() {
                eprintln!("{}", entry);
            }
        }
        self.flushed = true;
    }
}

#[cfg(debug_assertions)]
impl Drop for AttemptLogBuffer {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(debug_assertions)]
const RAYLEIGH_RITZ_LOG_FRONT: usize = 5;
#[cfg(debug_assertions)]
const RAYLEIGH_RITZ_LOG_TAIL: usize = 5;

#[cfg(debug_assertions)]
thread_local! {
    static RAYLEIGH_RITZ_LOG_STATE: RefCell<RayleighRitzLogState> =
        RefCell::new(RayleighRitzLogState::new());
}

#[cfg(debug_assertions)]
struct RayleighRitzLogState {
    depth: usize,
    buffer: Option<AttemptLogBuffer>,
}

#[cfg(debug_assertions)]
impl RayleighRitzLogState {
    fn new() -> Self {
        Self {
            depth: 0,
            buffer: None,
        }
    }

    fn enter(&mut self) {
        if self.depth == 0 {
            self.buffer = Some(AttemptLogBuffer::new(
                RAYLEIGH_RITZ_LOG_FRONT,
                RAYLEIGH_RITZ_LOG_TAIL,
            ));
        }
        self.depth = self.depth.saturating_add(1);
    }

    fn exit(&mut self) {
        if self.depth == 0 {
            return;
        }
        self.depth -= 1;
        if self.depth == 0 {
            if let Some(mut buffer) = self.buffer.take() {
                buffer.flush();
            }
        }
    }

    fn log(&mut self, message: String) {
        if let Some(buffer) = self.buffer.as_mut() {
            buffer.log(message);
        } else {
            eprintln!("{}", message);
        }
    }
}

#[cfg(debug_assertions)]
struct RayleighRitzLogScope;

#[cfg(debug_assertions)]
impl RayleighRitzLogScope {
    fn new() -> Self {
        RAYLEIGH_RITZ_LOG_STATE.with(|state| state.borrow_mut().enter());
        Self
    }
}

#[cfg(debug_assertions)]
impl Drop for RayleighRitzLogScope {
    fn drop(&mut self) {
        RAYLEIGH_RITZ_LOG_STATE.with(|state| state.borrow_mut().exit());
    }
}

#[cfg(debug_assertions)]
fn rayleigh_ritz_debug_log(message: String) {
    RAYLEIGH_RITZ_LOG_STATE.with(|state| state.borrow_mut().log(message));
}

#[cfg(not(debug_assertions))]
fn rayleigh_ritz_debug_log(_message: String) {}

fn extract_complex_column(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    col: usize,
) -> Vec<Complex64> {
    (0..rows).map(|row| matrix[row * cols + col]).collect()
}

fn build_projected_matrices<B: SpectralBackend>(
    backend: &B,
    subspace: &[SubspaceEntry<'_, B>],
) -> (Vec<Complex64>, Vec<Complex64>) {
    let dim = subspace.len();
    let mut op_proj = vec![Complex64::default(); dim * dim];
    let mut mass_proj = vec![Complex64::default(); dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mass_val = backend.dot(subspace[i].vector, subspace[j].mass);
            let op_val = backend.dot(subspace[i].vector, subspace[j].applied);
            let idx = i * dim + j;
            mass_proj[idx] = mass_val;
            op_proj[idx] = op_val;
            if i != j {
                mass_proj[j * dim + i] = mass_val.conj();
                op_proj[j * dim + i] = op_val.conj();
            }
        }
    }
    enforce_hermitian(&mut op_proj, dim);
    enforce_hermitian(&mut mass_proj, dim);
    #[cfg(test)]
    if dim <= 4 {
        eprintln!("[rayleigh-ritz-debug] op_proj={:?}", op_proj);
        eprintln!("[rayleigh-ritz-debug] mass_proj={:?}", mass_proj);
    }
    (op_proj, mass_proj)
}

struct StabilizedProjection {
    op_matrix: Vec<Complex64>,
    mass_matrix: Vec<Complex64>,
    transform: Vec<Complex64>,
    reduced_dim: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProjectionStats {
    original_dim: usize,
    reduced_dim: usize,
    requested_dim: usize,
    min_mass_eigenvalue: f64,
    max_mass_eigenvalue: f64,
}

impl ProjectionStats {
    fn ratio(&self) -> f64 {
        if self.min_mass_eigenvalue <= 0.0 {
            f64::INFINITY
        } else {
            self.max_mass_eigenvalue / self.min_mass_eigenvalue
        }
    }

    fn satisfies_rank(&self) -> bool {
        if self.original_dim == 0 {
            return false;
        }
        // Even if stabilization trims the subspace below the requested
        // dimension, proceed as long as we have at least one stable
        // direction; otherwise, fall back to a safer solve path.
        self.reduced_dim > 0
    }

    fn to_public(self, history_dim: usize, fallback_used: bool) -> ProjectionDiagnostics {
        ProjectionDiagnostics {
            original_dim: self.original_dim,
            reduced_dim: self.reduced_dim,
            requested_dim: self.requested_dim,
            history_dim,
            min_mass_eigenvalue: self.min_mass_eigenvalue,
            max_mass_eigenvalue: self.max_mass_eigenvalue,
            condition_estimate: self.ratio(),
            fallback_used,
        }
    }
}

fn stabilize_projected_system(
    op_proj: &[Complex64],
    mass_proj: &[Complex64],
    dim: usize,
    want: usize,
    tol: f64,
) -> Option<(StabilizedProjection, ProjectionStats)> {
    if dim == 0 {
        return None;
    }
    let mut stats = ProjectionStats {
        original_dim: dim,
        requested_dim: want,
        reduced_dim: 0,
        min_mass_eigenvalue: f64::INFINITY,
        max_mass_eigenvalue: 0.0,
    };
    let block_dim = dim * 2;
    let min_required = want.min(dim).max(1);
    let mass_block = expand_complex_hermitian(mass_proj, dim);
    let (mass_vals, mass_vecs) = jacobi_eigendecomposition(mass_block, block_dim, tol);
    let mut min_mass_eval = f64::INFINITY;
    let mut max_mass_eval = 0.0_f64;
    let mut has_small_or_negative = false;
    for &val in &mass_vals {
        if val > 0.0 {
            if val < min_mass_eval {
                min_mass_eval = val;
            }
            if val > max_mass_eval {
                max_mass_eval = val;
            }
        } else if val.is_finite() {
            has_small_or_negative = true;
        }
    }
    let max_val = mass_vals.iter().cloned().fold(0.0_f64, f64::max);
    if max_val <= 0.0 {
        return None;
    }
    // Filter out directions whose mass eigenvalues are too small to yield a
    // well-conditioned projected problem. A slightly more aggressive cutoff
    // helps prevent the Rayleigh–Ritz stage from injecting wildly scaled
    // vectors when the block has accumulated near-null directions.
    const REL_TOL: f64 = 1e-6;
    const ABS_TOL: f64 = 1e-10;
    let cutoff = (max_val * REL_TOL).max(ABS_TOL);
    let cond = if has_small_or_negative || min_mass_eval <= cutoff {
        f64::INFINITY
    } else if min_mass_eval > 0.0 {
        max_mass_eval / min_mass_eval
    } else {
        f64::INFINITY
    };
    let can_skip_stabilization =
        dim <= want.max(1) && min_mass_eval > cutoff && cond <= PROJECTION_CONDITION_LIMIT;
    if can_skip_stabilization {
        stats.reduced_dim = dim;
        stats.min_mass_eigenvalue = min_mass_eval;
        stats.max_mass_eigenvalue = max_mass_eval;
        let mut transform = vec![Complex64::default(); dim * dim];
        for i in 0..dim {
            transform[i * dim + i] = Complex64::new(1.0, 0.0);
        }
        return Some((
            StabilizedProjection {
                op_matrix: op_proj.to_vec(),
                mass_matrix: mass_proj.to_vec(),
                transform,
                reduced_dim: dim,
            },
            stats,
        ));
    }
    let mut selected: Vec<(usize, f64)> = mass_vals
        .iter()
        .enumerate()
        .filter_map(|(idx, val)| {
            if *val > cutoff {
                Some((idx, *val))
            } else {
                None
            }
        })
        .collect();
    if selected.is_empty() {
        return None;
    }
    selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let target = want.max(1);
    const MULTIPLIER: usize = 3;
    const ABS_MAX_DIM: usize = 64;
    let mut limit = target * MULTIPLIER;
    if limit < target + 2 {
        limit = target + 2;
    }
    limit = limit.min(dim).min(ABS_MAX_DIM);
    if selected.len() > limit {
        selected.truncate(limit);
    }
    selected.sort_by_key(|&(idx, _)| idx);
    let complex_vecs = convert_block_vectors(&mass_vecs, dim, block_dim);
    let mut columns: Vec<Vec<Complex64>> = Vec::new();
    let mut raw_columns: Vec<Vec<Complex64>> = Vec::new();
    let mut raw_norms: Vec<f64> = Vec::new();
    const DUP_THRESHOLD: f64 = 1.0 - 1e-6;
    const VEC_NORM_TOL: f64 = 1e-14;
    for &(eig_idx, value) in selected.iter() {
        if columns.len() >= dim {
            break;
        }
        let mut raw = vec![Complex64::default(); dim];
        for row in 0..dim {
            raw[row] = complex_vecs[row * block_dim + eig_idx];
        }
        let norm_sq = raw.iter().map(|c| c.norm_sqr()).sum::<f64>();
        if norm_sq < VEC_NORM_TOL {
            continue;
        }
        let mut duplicate = false;
        for (existing, &existing_norm_sq) in raw_columns.iter().zip(raw_norms.iter()) {
            let overlap = complex_dot(existing, &raw);
            let ratio = overlap.norm_sqr() / (norm_sq * existing_norm_sq);
            if ratio >= DUP_THRESHOLD {
                duplicate = true;
                break;
            }
        }
        if duplicate && columns.len() >= min_required {
            continue;
        }
        let scale = 1.0 / value.sqrt();
        let mut candidate = raw.clone();
        for val in candidate.iter_mut() {
            *val *= scale;
        }
        raw_columns.push(raw);
        raw_norms.push(norm_sq);
        columns.push(candidate);
        stats.reduced_dim += 1;
        stats.min_mass_eigenvalue = stats.min_mass_eigenvalue.min(value);
        stats.max_mass_eigenvalue = stats.max_mass_eigenvalue.max(value);
    }
    if columns.is_empty() {
        return None;
    }
    if !stats.min_mass_eigenvalue.is_finite() {
        stats.min_mass_eigenvalue = 0.0;
    }
    let mut reduced_dim = columns.len();
    let mut transform = vec![Complex64::default(); dim * reduced_dim];
    for (col_idx, data) in columns.iter().enumerate() {
        for row in 0..dim {
            transform[row * reduced_dim + col_idx] = data[row];
        }
    }

    let mut temp_op = complex_matmul(dim, dim, reduced_dim, op_proj, &transform);
    let mut op_reduced =
        complex_matmul_conj_transpose_left(dim, reduced_dim, reduced_dim, &transform, &temp_op);
    let mut temp_mass = complex_matmul(dim, dim, reduced_dim, mass_proj, &transform);
    let mut mass_reduced =
        complex_matmul_conj_transpose_left(dim, reduced_dim, reduced_dim, &transform, &temp_mass);

    // Drop directions whose reduced mass eigenvalues fall below the cutoff to
    // avoid feeding nearly singular systems into the generalized eigen solve.
    let (mass_eval_block, _) = jacobi_eigendecomposition(
        expand_complex_hermitian(&mass_reduced, reduced_dim),
        reduced_dim * 2,
        tol,
    );
    let filtered_indices: Vec<usize> = mass_eval_block
        .iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val > cutoff { Some(idx) } else { None })
        .collect();
    if filtered_indices.len() < reduced_dim && !filtered_indices.is_empty() {
        let new_dim = filtered_indices.len();
        let mut filtered_transform = vec![Complex64::default(); dim * new_dim];
        for (out_col, &keep_idx) in filtered_indices.iter().enumerate() {
            for row in 0..dim {
                filtered_transform[row * new_dim + out_col] =
                    transform[row * reduced_dim + keep_idx];
            }
        }
        transform = filtered_transform;
        reduced_dim = new_dim;
        temp_op = complex_matmul(dim, dim, reduced_dim, op_proj, &transform);
        op_reduced =
            complex_matmul_conj_transpose_left(dim, reduced_dim, reduced_dim, &transform, &temp_op);
        temp_mass = complex_matmul(dim, dim, reduced_dim, mass_proj, &transform);
        mass_reduced = complex_matmul_conj_transpose_left(
            dim,
            reduced_dim,
            reduced_dim,
            &transform,
            &temp_mass,
        );
    }
    let final_mass_evals = jacobi_eigendecomposition(
        expand_complex_hermitian(&mass_reduced, reduced_dim),
        reduced_dim * 2,
        tol,
    )
    .0;
    stats.reduced_dim = reduced_dim;
    stats.min_mass_eigenvalue = final_mass_evals
        .iter()
        .cloned()
        .filter(|v| *v > 0.0)
        .fold(f64::INFINITY, f64::min);
    if !stats.min_mass_eigenvalue.is_finite() {
        stats.min_mass_eigenvalue = 0.0;
    }
    stats.max_mass_eigenvalue = final_mass_evals.iter().cloned().fold(0.0_f64, f64::max);
    if has_small_or_negative {
        stats.min_mass_eigenvalue = 0.0;
    }
    if stats.reduced_dim < min_required {
        return None;
    }
    let final_cond = if stats.min_mass_eigenvalue > 0.0 {
        stats.max_mass_eigenvalue / stats.min_mass_eigenvalue
    } else {
        f64::INFINITY
    };
    if !final_cond.is_finite() || final_cond > PROJECTION_CONDITION_LIMIT {
        return None;
    }
    Some((
        StabilizedProjection {
            op_matrix: op_reduced,
            mass_matrix: mass_reduced,
            transform,
            reduced_dim,
        },
        stats,
    ))
}

fn lift_projected_eigenvectors(
    transform: &[Complex64],
    rows: usize,
    reduced_dim: usize,
    eigenvectors: &[Complex64],
    cols: usize,
) -> Vec<Complex64> {
    if rows == 0 || reduced_dim == 0 || cols == 0 {
        return Vec::new();
    }
    complex_matmul(rows, reduced_dim, cols, transform, eigenvectors)
}

#[cfg(debug_assertions)]
fn log_projected_dimension(original: usize, reduced: usize, want: usize) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] projected dimension {} -> {} (want={})",
        original, reduced, want
    ));
}

#[cfg(not(debug_assertions))]
fn log_projected_dimension(_original: usize, _reduced: usize, _want: usize) {}

fn expand_complex_hermitian(matrix: &[Complex64], dim: usize) -> Vec<f64> {
    let block_dim = dim * 2;
    let mut block = vec![0.0; block_dim * block_dim];
    for row in 0..dim {
        for col in 0..dim {
            let val = matrix[row * dim + col];
            let re = val.re;
            let im = val.im;
            block[row * block_dim + col] = re;
            block[row * block_dim + (col + dim)] = -im;
            block[(row + dim) * block_dim + col] = im;
            block[(row + dim) * block_dim + (col + dim)] = re;
        }
    }
    block
}

fn enforce_hermitian(matrix: &mut [Complex64], dim: usize) {
    for i in 0..dim {
        let diag = matrix[i * dim + i];
        matrix[i * dim + i] = Complex64::new(diag.re, 0.0);
        for j in (i + 1)..dim {
            let upper = matrix[i * dim + j];
            let lower = matrix[j * dim + i].conj();
            let average = (upper + lower) * 0.5;
            matrix[i * dim + j] = average;
            matrix[j * dim + i] = average.conj();
        }
    }
}

fn complex_matmul(
    rows: usize,
    mid: usize,
    cols: usize,
    a: &[Complex64],
    b: &[Complex64],
) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), rows * mid);
    debug_assert_eq!(b.len(), mid * cols);
    let mut out = vec![Complex64::default(); rows * cols];
    for i in 0..rows {
        for k in 0..mid {
            let aik = a[i * mid + k];
            if aik == Complex64::default() {
                continue;
            }
            for j in 0..cols {
                out[i * cols + j] += aik * b[k * cols + j];
            }
        }
    }
    out
}

fn complex_matmul_conj_transpose_left(
    rows: usize,
    cols_left: usize,
    cols_right: usize,
    a: &[Complex64],
    b: &[Complex64],
) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), rows * cols_left);
    debug_assert_eq!(b.len(), rows * cols_right);
    let mut out = vec![Complex64::default(); cols_left * cols_right];
    for i in 0..cols_left {
        for j in 0..cols_right {
            let mut sum = Complex64::default();
            for row in 0..rows {
                let aval = a[row * cols_left + i].conj();
                let bval = b[row * cols_right + j];
                sum += aval * bval;
            }
            out[i * cols_right + j] = sum;
        }
    }
    out
}

fn complex_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = Complex64::default();
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum += ai.conj() * bi;
    }
    sum
}

fn combine_entries<O, B>(
    operator: &mut O,
    subspace: &[SubspaceEntry<'_, B>],
    coeffs: &[Complex64],
) -> BlockEntry<B>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut vector = operator.alloc_field();
    zero_buffer(vector.as_mut_slice());
    let mut mass = operator.alloc_field();
    zero_buffer(mass.as_mut_slice());
    let mut applied = operator.alloc_field();
    zero_buffer(applied.as_mut_slice());
    for (entry, &coeff) in subspace.iter().zip(coeffs.iter()) {
        if coeff.norm_sqr() < 1e-24 {
            continue;
        }
        operator.backend().axpy(coeff, entry.vector, &mut vector);
        operator.backend().axpy(coeff, entry.mass, &mut mass);
        operator.backend().axpy(coeff, entry.applied, &mut applied);
    }
    let norm = mass_norm(operator.backend(), &vector, &mass);
    if norm > 0.0 {
        let scale = Complex64::new(1.0 / norm, 0.0);
        operator.backend().scale(scale, &mut vector);
        operator.backend().scale(scale, &mut mass);
        operator.backend().scale(scale, &mut applied);
    }
    BlockEntry {
        vector,
        mass,
        applied,
    }
}

fn compute_preconditioned_residuals<O, B>(
    operator: &mut O,
    eigenvalues: &[f64],
    x_entries: &[BlockEntry<B>],
    w_entries: &[BlockEntry<B>],
    gamma_mode: Option<&GammaMode<B>>,
    deflation: Option<&DeflationWorkspace<B>>,
    symmetry: Option<&SymmetryProjector>,
    preconditioner: &mut Option<&mut dyn OperatorPreconditioner<B>>,
    tol: f64,
    iteration_index: usize,
    snapshot_manager: Option<&mut ResidualSnapshotManager>,
    snapshot_store: &mut Vec<ResidualSnapshot>,
) -> (ResidualComputation, Vec<BlockEntry<B>>)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut snapshot_manager = snapshot_manager;
    let mut max_residual: f64 = 0.0;
    let mut sum_residual: f64 = 0.0;
    let mut max_relative: f64 = 0.0;
    let mut sum_relative: f64 = 0.0;
    let mut max_scale: f64 = 0.0;
    let mut sum_scale: f64 = 0.0;
    let mut evaluated: usize = 0;
    let mut accepted: usize = 0;
    let mut preconditioner_trials = 0usize;
    let mut preconditioner_sum_before = 0.0;
    let mut preconditioner_sum_after = 0.0;
    let mut p_entries = Vec::new();
    let direction_limit = x_entries.len().max(1);
    for (idx, entry) in x_entries.iter().enumerate() {
        let lambda = *eigenvalues.get(idx).unwrap_or(&0.0);
        let mut vector = operator.alloc_field();
        vector
            .as_mut_slice()
            .copy_from_slice(entry.applied.as_slice());
        operator
            .backend()
            .axpy(Complex64::new(-lambda, 0.0), &entry.mass, &mut vector);
        let mut mass = operator.alloc_field();
        operator.apply_mass(&vector, &mut mass);
        if let Some(manager) = snapshot_manager.as_deref_mut() {
            manager.capture(
                operator,
                iteration_index,
                idx,
                ResidualSnapshotStage::Raw,
                &vector,
                snapshot_store,
            );
        }
        enforce_constraints(
            operator,
            &mut vector,
            &mut mass,
            gamma_mode,
            deflation,
            symmetry,
        );
        project_against_entries(operator.backend(), &mut vector, &mut mass, x_entries);
        project_against_entries(operator.backend(), &mut vector, &mut mass, &p_entries);
        project_against_entries(operator.backend(), &mut vector, &mut mass, w_entries);
        if let Some(manager) = snapshot_manager.as_deref_mut() {
            manager.capture(
                operator,
                iteration_index,
                idx,
                ResidualSnapshotStage::Projected,
                &vector,
                snapshot_store,
            );
        }
        let mut norm = mass_norm(operator.backend(), &vector, &mass);
        max_residual = max_residual.max(norm);
        let (mut relative, scale) =
            compute_relative_residual_with_scale(operator.backend(), entry, lambda, norm);
        max_relative = max_relative.max(relative);
        max_scale = max_scale.max(scale);
        sum_residual += norm;
        sum_relative += relative;
        sum_scale += scale;
        evaluated += 1;
        if relative <= tol {
            continue;
        }
        if norm <= ABSOLUTE_RESIDUAL_GUARD && relative <= tol * 10.0 {
            continue;
        }
        let mut preconditioned = false;
        if let Some(precond) = preconditioner.as_mut() {
            preconditioned = true;
            preconditioner_trials += 1;
            preconditioner_sum_before += norm;
            let backend = operator.backend();
            (**precond).apply(backend, &mut vector);
        }
        operator.apply_mass(&vector, &mut mass);
        enforce_constraints(
            operator,
            &mut vector,
            &mut mass,
            gamma_mode,
            deflation,
            symmetry,
        );
        project_against_entries(operator.backend(), &mut vector, &mut mass, x_entries);
        project_against_entries(operator.backend(), &mut vector, &mut mass, &p_entries);
        project_against_entries(operator.backend(), &mut vector, &mut mass, w_entries);
        if preconditioned {
            if let Some(manager) = snapshot_manager.as_deref_mut() {
                manager.capture(
                    operator,
                    iteration_index,
                    idx,
                    ResidualSnapshotStage::Preconditioned,
                    &vector,
                    snapshot_store,
                );
            }
        }
        norm = mass_norm(operator.backend(), &vector, &mass);
        if preconditioned {
            preconditioner_sum_after += norm;
        }
        if scale > 0.0 {
            relative = norm / scale;
        } else {
            relative = norm;
        }
        if norm <= 1e-12 {
            continue;
        }
        if relative <= tol {
            continue;
        }
        accepted += 1;
        let scale = Complex64::new(1.0 / norm, 0.0);
        operator.backend().scale(scale, &mut vector);
        operator.backend().scale(scale, &mut mass);
        let mut applied = operator.alloc_field();
        operator.apply(&vector, &mut applied);
        p_entries.push(BlockEntry {
            vector,
            mass,
            applied,
        });
        if p_entries.len() >= direction_limit {
            break;
        }
    }
    let avg_residual = if evaluated > 0 {
        sum_residual / evaluated as f64
    } else {
        0.0
    };
    let avg_relative = if evaluated > 0 {
        sum_relative / evaluated as f64
    } else {
        0.0
    };
    let avg_scale = if evaluated > 0 {
        sum_scale / evaluated as f64
    } else {
        0.0
    };
    (
        ResidualComputation {
            max_residual,
            avg_residual,
            max_relative_residual: max_relative,
            avg_relative_residual: avg_relative,
            max_relative_scale: max_scale,
            avg_relative_scale: avg_scale,
            accepted,
            preconditioner_trials,
            preconditioner_avg_before: if preconditioner_trials > 0 {
                preconditioner_sum_before / preconditioner_trials as f64
            } else {
                0.0
            },
            preconditioner_avg_after: if preconditioner_trials > 0 {
                preconditioner_sum_after / preconditioner_trials as f64
            } else {
                0.0
            },
        },
        p_entries,
    )
}

fn reorthogonalize_block<O, B>(
    operator: &mut O,
    block: &mut Vec<BlockEntry<B>>,
    reference: &[BlockEntry<B>],
) where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut idx = 0;
    while idx < block.len() {
        let mut remove = false;
        {
            let backend = operator.backend();
            let (earlier, rest) = block.split_at_mut(idx);
            let entry = &mut rest[0];

            const MAX_REORTHOG_PASSES: usize = 3;
            const OVERLAP_TOL: f64 = 1e-12;

            for _ in 0..MAX_REORTHOG_PASSES {
                project_against_entries(backend, &mut entry.vector, &mut entry.mass, reference);
                project_against_entries(backend, &mut entry.vector, &mut entry.mass, earlier);
                let norm =
                    normalize_with_mass_precomputed(backend, &mut entry.vector, &mut entry.mass);
                if norm <= 1e-12 {
                    remove = true;
                    break;
                }
                let overlap = max_mass_overlap(backend, entry, reference, earlier);
                if overlap <= OVERLAP_TOL {
                    break;
                }
            }
            if !remove {
                zero_buffer(entry.applied.as_mut_slice());
                operator.apply(&entry.vector, &mut entry.applied);
            }
        }
        if remove {
            block.remove(idx);
        } else {
            idx += 1;
        }
    }
}

fn max_mass_overlap<B: SpectralBackend>(
    backend: &B,
    entry: &BlockEntry<B>,
    reference: &[BlockEntry<B>],
    earlier: &[BlockEntry<B>],
) -> f64 {
    reference
        .iter()
        .chain(earlier.iter())
        .map(|basis| backend.dot(&entry.vector, &basis.mass).norm())
        .fold(0.0, f64::max)
}

fn finalize_modes<O, B>(
    operator: &mut O,
    entries: &[BlockEntry<B>],
    eigenvalues: &[f64],
    target: usize,
    tol: f64,
    gamma_deflated: bool,
) -> (Vec<f64>, Vec<Field2D>, EigenDiagnostics)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut omegas = Vec::new();
    let mut modes = Vec::new();
    let mut diagnostics = EigenDiagnostics::new(tol.max(1e-9));
    let grid = operator.grid();
    let freq_tol = diagnostics.freq_tolerance;
    let mut last_kept: Option<f64> = None;
    log_finalize_context(gamma_deflated, target, eigenvalues.len());
    let total_candidates = entries.len().min(eigenvalues.len());
    for (idx, (entry, &lambda)) in entries
        .iter()
        .zip(eigenvalues.iter())
        .take(total_candidates)
        .enumerate()
    {
        log_finalize_candidate(lambda);
        if lambda < 0.0 {
            log_finalize_skip("negative", lambda);
            diagnostics.negative_modes_skipped += 1;
            continue;
        }
        let omega = lambda.sqrt();
        if gamma_deflated {
            let gamma_floor = freq_tol.max(1.0);
            if omega <= gamma_floor {
                log_finalize_skip("gamma_zero", lambda);
                diagnostics.duplicate_modes_skipped += 1;
                continue;
            }
        }
        if let Some(prev) = last_kept {
            let remaining_candidates = total_candidates.saturating_sub(idx + 1);
            let can_skip_duplicate = omegas.len() + remaining_candidates >= target;
            if (omega - prev).abs() <= freq_tol && can_skip_duplicate {
                log_finalize_skip("duplicate", lambda);
                diagnostics.duplicate_modes_skipped += 1;
                continue;
            }
        }
        let mass_norm = mass_norm(operator.backend(), &entry.vector, &entry.mass);
        let residual_norm = compute_residual_norm(operator, entry, lambda);
        let relative_residual =
            compute_relative_residual(operator.backend(), entry, lambda, residual_norm);
        diagnostics.max_residual = diagnostics.max_residual.max(residual_norm);
        diagnostics.max_relative_residual =
            diagnostics.max_relative_residual.max(relative_residual);
        diagnostics.modes.push(ModeDiagnostics {
            omega,
            lambda,
            residual_norm,
            mass_norm,
            relative_residual,
        });

        let data = entry.vector.as_slice().to_vec();
        modes.push(Field2D::from_vec(grid, data));
        omegas.push(omega);
        last_kept = Some(omega);
        if omegas.len() == target {
            break;
        }
    }
    (omegas, modes, diagnostics)
}

#[cfg(debug_assertions)]
fn log_finalize_context(gamma_deflated: bool, target: usize, eig_count: usize) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] finalize context gamma_deflated={} target={} eigenvalues={}",
        gamma_deflated, target, eig_count
    ));
}

#[cfg(not(debug_assertions))]
fn log_finalize_context(_gamma_deflated: bool, _target: usize, _eig_count: usize) {}

#[cfg(debug_assertions)]
fn log_finalize_candidate(lambda: f64) {
    rayleigh_ritz_debug_log(format!("[rayleigh-ritz] finalize lambda={}", lambda));
}

#[cfg(debug_assertions)]
fn log_finalize_skip(reason: &str, lambda: f64) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] finalize skipped {} candidate lambda={}",
        reason, lambda
    ));
}

#[cfg(not(debug_assertions))]
fn log_finalize_candidate(_lambda: f64) {}

#[cfg(not(debug_assertions))]
fn log_finalize_skip(_reason: &str, _lambda: f64) {}

fn compute_residual_norm<O, B>(operator: &mut O, entry: &BlockEntry<B>, lambda: f64) -> f64
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut residual = operator.alloc_field();
    residual
        .as_mut_slice()
        .copy_from_slice(entry.applied.as_slice());
    operator
        .backend()
        .axpy(Complex64::new(-lambda, 0.0), &entry.mass, &mut residual);
    let mut residual_mass = operator.alloc_field();
    operator.apply_mass(&residual, &mut residual_mass);
    mass_norm(operator.backend(), &residual, &residual_mass)
}

fn generalized_eigen(
    op_matrix: Vec<Complex64>,
    mass_matrix: Vec<Complex64>,
    dim: usize,
    tol: f64,
) -> Option<(Vec<f64>, Vec<Complex64>)> {
    let l = match cholesky_decompose_hermitian(&mass_matrix, dim) {
        Some(factor) => factor,
        None => {
            log_eigen_failure("cholesky", dim);
            return generalized_eigen_with_whitening(op_matrix, mass_matrix, dim, tol);
        }
    };
    let temp = match solve_lower_triangular_complex(&l, &op_matrix, dim, dim) {
        Some(val) => val,
        None => {
            log_eigen_failure("solve_lower", dim);
            return generalized_eigen_with_whitening(op_matrix, mass_matrix, dim, tol);
        }
    };
    let c = match solve_upper_triangular_right_conj(&l, &temp, dim) {
        Some(val) => val,
        None => {
            log_eigen_failure("solve_right", dim);
            return generalized_eigen_with_whitening(op_matrix, mass_matrix, dim, tol);
        }
    };
    let block = expand_complex_hermitian(&c, dim);
    let block_dim = dim * 2;
    let (values, eigenvectors) = jacobi_eigendecomposition(block, block_dim, tol);
    let complex_eigenvectors = convert_block_vectors(&eigenvectors, dim, block_dim);
    let coeffs = match solve_upper_triangular_conj(&l, &complex_eigenvectors, dim, block_dim) {
        Some(val) => val,
        None => {
            log_eigen_failure("solve_upper", dim);
            return generalized_eigen_with_whitening(op_matrix, mass_matrix, dim, tol);
        }
    };
    Some((values, coeffs))
}

#[cfg(debug_assertions)]
fn log_eigen_failure(stage: &str, dim: usize) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] generalized eigen failed at {} (dim={})",
        stage, dim
    ));
}

#[cfg(not(debug_assertions))]
fn log_eigen_failure(_stage: &str, _dim: usize) {}

fn cholesky_decompose_hermitian(matrix: &[Complex64], dim: usize) -> Option<Vec<Complex64>> {
    let mut l = vec![Complex64::default(); dim * dim];
    const NEG_TOL: f64 = 1e-10;
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = matrix[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k].conj();
            }
            if i == j {
                let diag = sum.re;
                if diag <= 0.0 {
                    if diag > -NEG_TOL {
                        l[i * dim + j] = Complex64::new(NEG_TOL.sqrt(), 0.0);
                        continue;
                    }
                    log_cholesky_diag(diag, i);
                    return None;
                }
                l[i * dim + j] = Complex64::new(diag.sqrt(), 0.0);
            } else {
                let denom = l[j * dim + j];
                if denom.norm_sqr() < 1e-24 {
                    log_cholesky_diag(0.0, j);
                    return None;
                }
                l[i * dim + j] = sum / denom;
            }
        }
    }
    Some(l)
}

#[cfg(debug_assertions)]
fn log_cholesky_diag(value: f64, index: usize) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] cholesky diag failure at row {} (value={})",
        index, value
    ));
}

#[cfg(not(debug_assertions))]
fn log_cholesky_diag(_value: f64, _index: usize) {}

fn solve_lower_triangular_complex(
    l: &[Complex64],
    b: &[Complex64],
    dim: usize,
    cols: usize,
) -> Option<Vec<Complex64>> {
    let mut x = vec![Complex64::default(); dim * cols];
    for col in 0..cols {
        for row in 0..dim {
            let mut sum = b[row * cols + col];
            for k in 0..row {
                sum -= l[row * dim + k] * x[k * cols + col];
            }
            let diag = l[row * dim + row];
            if diag.norm_sqr() < 1e-24 {
                return None;
            }
            x[row * cols + col] = sum / diag;
        }
    }
    Some(x)
}

fn solve_upper_triangular_right_conj(
    l: &[Complex64],
    b: &[Complex64],
    dim: usize,
) -> Option<Vec<Complex64>> {
    let mut x = vec![Complex64::default(); dim * dim];
    for row in 0..dim {
        for rev_col in 0..dim {
            let col = dim - 1 - rev_col;
            let mut sum = b[row * dim + col];
            for k in (col + 1)..dim {
                sum -= x[row * dim + k] * l[k * dim + col].conj();
            }
            let diag = l[col * dim + col].conj();
            if diag.norm_sqr() < 1e-24 {
                return None;
            }
            x[row * dim + col] = sum / diag;
        }
    }
    Some(x)
}

fn solve_upper_triangular_conj(
    l: &[Complex64],
    b: &[Complex64],
    dim: usize,
    cols: usize,
) -> Option<Vec<Complex64>> {
    let mut x = vec![Complex64::default(); dim * cols];
    for col in 0..cols {
        for rev_row in 0..dim {
            let row = dim - 1 - rev_row;
            let mut sum = b[row * cols + col];
            for k in (row + 1)..dim {
                sum -= l[k * dim + row].conj() * x[k * cols + col];
            }
            let diag = l[row * dim + row].conj();
            if diag.norm_sqr() < 1e-24 {
                return None;
            }
            x[row * cols + col] = sum / diag;
        }
    }
    Some(x)
}

fn convert_block_vectors(data: &[f64], dim: usize, cols: usize) -> Vec<Complex64> {
    let mut complex = vec![Complex64::default(); dim * cols];
    for col in 0..cols {
        for row in 0..dim {
            let real = data[row * cols + col];
            let imag = data[(row + dim) * cols + col];
            complex[row * cols + col] = Complex64::new(real, imag);
        }
    }
    complex
}

fn generalized_eigen_with_whitening(
    op_matrix: Vec<Complex64>,
    mass_matrix: Vec<Complex64>,
    dim: usize,
    tol: f64,
) -> Option<(Vec<f64>, Vec<Complex64>)> {
    let block_dim = dim * 2;
    let op_block = expand_complex_hermitian(&op_matrix, dim);
    let mass_block = expand_complex_hermitian(&mass_matrix, dim);
    let (mass_vals, mass_vecs) = jacobi_eigendecomposition(mass_block, block_dim, tol);
    const MASS_TOL: f64 = 1e-10;
    let mut keep = Vec::new();
    for (idx, &value) in mass_vals.iter().enumerate() {
        if value > MASS_TOL {
            keep.push(idx);
        }
    }
    if keep.is_empty() {
        log_whitening_rank(0, block_dim);
        return generalized_eigen_identity(op_matrix, dim, tol);
    }
    let k = keep.len();
    log_whitening_rank(k, block_dim);
    let mut whitening = vec![0.0; block_dim * k];
    for (col_idx, &eig_idx) in keep.iter().enumerate() {
        let scale = 1.0 / mass_vals[eig_idx].sqrt();
        for row in 0..block_dim {
            whitening[row * k + col_idx] = mass_vecs[row * block_dim + eig_idx] * scale;
        }
    }
    let temp = matmul(block_dim, block_dim, k, &op_block, &whitening);
    let reduced = matmul_transpose_left(block_dim, k, k, &whitening, &temp);
    let (values, reduced_vecs) = jacobi_eigendecomposition(reduced, k, tol);
    let block_coeffs = matmul(block_dim, k, k, &whitening, &reduced_vecs);
    let complex_eigenvectors = convert_block_vectors(&block_coeffs, dim, k);
    Some((values, complex_eigenvectors))
}

#[cfg(debug_assertions)]
fn log_whitening_rank(rank: usize, total: usize) {
    rayleigh_ritz_debug_log(format!(
        "[rayleigh-ritz] whitening fallback rank {} of {}",
        rank, total
    ));
}

#[cfg(not(debug_assertions))]
fn log_whitening_rank(_rank: usize, _total: usize) {}

fn matmul(rows: usize, mid: usize, cols: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for k in 0..mid {
            let aik = a[i * mid + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..cols {
                out[i * cols + j] += aik * b[k * cols + j];
            }
        }
    }
    out
}

fn matmul_transpose_left(
    rows: usize,
    cols_left: usize,
    cols_right: usize,
    a: &[f64],
    b: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; cols_left * cols_right];
    for i in 0..cols_left {
        for row in 0..rows {
            let ari = a[row * cols_left + i];
            if ari == 0.0 {
                continue;
            }
            for col in 0..cols_right {
                out[i * cols_right + col] += ari * b[row * cols_right + col];
            }
        }
    }
    out
}

fn generalized_eigen_identity(
    op_matrix: Vec<Complex64>,
    dim: usize,
    tol: f64,
) -> Option<(Vec<f64>, Vec<Complex64>)> {
    if dim == 0 {
        return Some((Vec::new(), Vec::new()));
    }
    let block = expand_complex_hermitian(&op_matrix, dim);
    let block_dim = dim * 2;
    let (values, eigenvectors) = jacobi_eigendecomposition(block, block_dim, tol);
    let complex = convert_block_vectors(&eigenvectors, dim, block_dim);
    Some((values, complex))
}

#[cfg(test)]
mod eigensolver_internal_tests {
    use super::{EigenOptions, generalized_eigen};
    use num_complex::Complex64;

    #[test]
    fn generalized_eigen_matches_diagonal_inputs() {
        let diag = [0.25, 1.0, 4.0];
        let dim = diag.len();
        let mut op = vec![Complex64::default(); dim * dim];
        let mut mass = vec![Complex64::default(); dim * dim];
        for i in 0..dim {
            op[i * dim + i] = Complex64::new(diag[i], 0.0);
            mass[i * dim + i] = Complex64::new(1.0, 0.0);
        }
        let (vals, _) = generalized_eigen(op, mass, dim, 1e-12).expect("generalized eigen solve");
        for (i, want) in diag.iter().enumerate() {
            assert!(
                (vals[i] - want).abs() < 1e-9,
                "eigenvalue mismatch at {i}: got {}, want {}",
                vals[i],
                want
            );
        }
    }

    #[test]
    fn default_block_size_adds_slack() {
        let mut opts = EigenOptions::default();
        opts.n_bands = 5;
        assert_eq!(opts.effective_block_size(), 7);
        opts.block_size = 10;
        assert_eq!(opts.effective_block_size(), 10);
    }
}

#[cfg(test)]
mod projection_stability_tests {
    use super::Complex64;
    use super::rayleigh_ritz;
    use super::{BlockEntry, LinearOperator, SpectralBackend};
    use crate::field::Field2D;
    use crate::grid::Grid2D;

    #[derive(Clone, Copy, Default)]
    struct IdentityBackend;

    impl SpectralBackend for IdentityBackend {
        type Buffer = Field2D;

        fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
            Field2D::zeros(grid)
        }

        fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}

        fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

        fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
            for value in buffer.as_mut_slice() {
                *value *= alpha;
            }
        }

        fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
            for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
                *dst += alpha * src;
            }
        }

        fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
            x.as_slice()
                .iter()
                .zip(y.as_slice())
                .map(|(a, b)| a.conj() * b)
                .sum()
        }
    }

    struct IdentityOp {
        backend: IdentityBackend,
        grid: Grid2D,
    }

    impl IdentityOp {
        fn new(size: usize) -> Self {
            Self {
                backend: IdentityBackend,
                grid: Grid2D::new(size, 1, 1.0, 1.0),
            }
        }
    }

    impl LinearOperator<IdentityBackend> for IdentityOp {
        fn apply(&mut self, input: &Field2D, output: &mut Field2D) {
            output.as_mut_slice().copy_from_slice(input.as_slice());
        }

        fn apply_mass(&mut self, input: &Field2D, output: &mut Field2D) {
            output.as_mut_slice().copy_from_slice(input.as_slice());
        }

        fn alloc_field(&self) -> Field2D {
            self.backend.alloc_field(self.grid)
        }

        fn backend(&self) -> &IdentityBackend {
            &self.backend
        }

        fn backend_mut(&mut self) -> &mut IdentityBackend {
            &mut self.backend
        }

        fn grid(&self) -> Grid2D {
            self.grid
        }
    }

    #[test]
    fn trims_singular_mass_when_dim_matches_request() {
        let mut op = IdentityOp::new(2);

        let mut v1 = op.alloc_field();
        v1.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
        let mut m1 = op.alloc_field();
        op.apply_mass(&v1, &mut m1);
        let mut a1 = op.alloc_field();
        op.apply(&v1, &mut a1);

        let mut v2 = op.alloc_field();
        // Deliberately inject a nearly null direction to emulate a singular
        // mass matrix.
        v2.as_mut_slice()[0] = Complex64::new(0.0, 0.0);
        v2.as_mut_slice()[1] = Complex64::new(0.0, 0.0);
        let mut m2 = op.alloc_field();
        op.apply_mass(&v2, &mut m2);
        let mut a2 = op.alloc_field();
        op.apply(&v2, &mut a2);

        let entries = vec![
            BlockEntry {
                vector: v1,
                mass: m1,
                applied: a1,
            },
            BlockEntry {
                vector: v2,
                mass: m2,
                applied: a2,
            },
        ];
        let subspace = super::build_subspace_entries(&entries, &[], &[]);

        let (values, _, diag) = rayleigh_ritz(&mut op, &subspace, 2, 0, 1e-12);

        assert!(diag.min_mass_eigenvalue == 0.0);
        assert!(diag.condition_estimate.is_infinite());
        assert!(
            diag.fallback_used,
            "unstable mass matrix should trigger fallback"
        );
        assert!(values.len() >= 1, "at least one mode should survive");
    }
}
