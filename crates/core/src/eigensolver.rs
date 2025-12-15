//! Simple Lanczos-style eigensolver utilities for Θ operators.

use std::cmp::Ordering;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

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
const W_HISTORY_FACTOR: usize = 2;

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
        self.history_multiplier.unwrap_or(W_HISTORY_FACTOR).max(1)
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
            let coeff = backend.dot(&entry.mass_vector, vector);
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
    let target_bands = opts.n_bands.max(1);
    let block_size = opts.effective_block_size();
    let gamma_mode = if gamma.is_gamma {
        build_gamma_mode(operator)
    } else {
        None
    };
    let gamma_deflated = gamma_mode.is_some();
    let deflation_vectors = deflation.map_or(0, |space| space.len());
    let deflation_active = if opts.debug.disable_deflation {
        None
    } else {
        deflation
    };
    let mut fallback_symmetry = None;
    let symmetry_projector = if let Some(custom) = symmetry_override {
        Some(custom)
    } else {
        fallback_symmetry = SymmetryProjector::from_options(&opts.symmetry);
        fallback_symmetry.as_ref()
    };
    let mut residual_snapshots = Vec::new();
    let mut snapshot_manager = residual_request.map(ResidualSnapshotManager::new);

    let (mut x_entries, warm_start_hits) = initialize_block(
        operator,
        block_size,
        gamma_mode.as_ref(),
        deflation_active,
        symmetry_projector,
        warm_start,
    );
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

    let mut eigenvalues;
    {
        let subspace = build_subspace_entries(&x_entries, &[], &[]);
        let (vals, new_entries) = rayleigh_ritz(
            operator,
            &subspace,
            x_entries.len(),
            opts.tol.max(MIN_RR_TOL),
        );
        eigenvalues = vals;
        x_entries = new_entries;
    }

    let mut w_entries: Vec<BlockEntry<B>> = Vec::new();
    let mut iterations = 0usize;
    let mut iteration_stats: Vec<IterationDiagnostics> = Vec::new();

    loop {
        let (residual_stats, mut p_entries) = compute_preconditioned_residuals(
            operator,
            &eigenvalues,
            &x_entries,
            &w_entries,
            gamma_mode.as_ref(),
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
        });
        if residual_stats.max_relative_residual <= opts.tol
            || p_entries.is_empty()
            || iterations >= opts.max_iter
        {
            break;
        }

        reorthogonalize_block(operator, &mut p_entries, &x_entries);
        reorthogonalize_block(operator, &mut p_entries, &w_entries);
        reorthogonalize_block(operator, &mut w_entries, &x_entries);
        let history_limit = opts
            .debug
            .history_factor()
            .saturating_mul(x_entries.len())
            .max(1);
        while w_entries.len() > history_limit {
            w_entries.pop();
        }

        let subspace = build_subspace_entries(&x_entries, &p_entries, &w_entries);
        let (vals, new_entries) = rayleigh_ritz(
            operator,
            &subspace,
            x_entries.len(),
            opts.tol.max(MIN_RR_TOL),
        );
        eigenvalues = vals;
        x_entries = new_entries;
        w_entries = p_entries;
        iterations += 1;
    }

    let (omegas, modes, mut diagnostics) =
        finalize_modes(operator, &x_entries, &eigenvalues, target_bands, opts.tol);
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
    let coeff = backend.dot(mass_basis, target);
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
        reorthogonalize_with_mass(operator.backend(), vector, &mode.vector, &mode.mass_vector);
        operator.apply_mass(vector, mass_vector);
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
        let coeff = backend.dot(&entry.mass, vector);
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
    tol: f64,
) -> (Vec<f64>, Vec<BlockEntry<B>>)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let dim = subspace.len();
    let (op_proj, mass_proj) = build_projected_matrices(operator.backend(), subspace);
    let (values, eigenvectors) = generalized_eigen(op_proj, mass_proj, dim, tol)
        .expect("generalized eigen solve failed in block solver");
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
    let mut new_entries = Vec::with_capacity(want.min(dim));
    let mut selected_values = Vec::with_capacity(want.min(dim));
    for idx in order {
        if new_entries.len() == want {
            break;
        }
        selected_values.push(values[idx]);
        let coeffs = extract_column(&eigenvectors, dim, idx);
        let entry = combine_entries(operator, subspace, &coeffs);
        new_entries.push(entry);
    }
    (selected_values, new_entries)
}

fn extract_column(matrix: &[f64], dim: usize, col: usize) -> Vec<f64> {
    (0..dim).map(|row| matrix[row * dim + col]).collect()
}

fn build_projected_matrices<B: SpectralBackend>(
    backend: &B,
    subspace: &[SubspaceEntry<'_, B>],
) -> (Vec<f64>, Vec<f64>) {
    let dim = subspace.len();
    let mut op_proj = vec![0.0; dim * dim];
    let mut mass_proj = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mass_val = backend.dot(subspace[i].vector, subspace[j].mass).re;
            let op_val = backend.dot(subspace[i].vector, subspace[j].applied).re;
            let idx = i * dim + j;
            mass_proj[idx] = mass_val;
            op_proj[idx] = op_val;
            if i != j {
                mass_proj[j * dim + i] = mass_val;
                op_proj[j * dim + i] = op_val;
            }
        }
    }
    (op_proj, mass_proj)
}

fn combine_entries<O, B>(
    operator: &mut O,
    subspace: &[SubspaceEntry<'_, B>],
    coeffs: &[f64],
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
        if coeff.abs() < 1e-12 {
            continue;
        }
        let c = Complex64::new(coeff, 0.0);
        operator.backend().axpy(c, entry.vector, &mut vector);
        operator.backend().axpy(c, entry.mass, &mut mass);
        operator.backend().axpy(c, entry.applied, &mut applied);
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
        if relative <= tol || norm <= ABSOLUTE_RESIDUAL_GUARD {
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
            let entry = &mut block[idx];
            project_against_entries(backend, &mut entry.vector, &mut entry.mass, reference);
            let norm = mass_norm(backend, &entry.vector, &entry.mass);
            if norm <= 1e-12 {
                remove = true;
            } else {
                let scale = Complex64::new(1.0 / norm, 0.0);
                backend.scale(scale, &mut entry.vector);
                backend.scale(scale, &mut entry.mass);
            }
        }
        if remove {
            block.remove(idx);
            continue;
        }
        {
            let entry = &mut block[idx];
            operator.apply(&entry.vector, &mut entry.applied);
        }
        idx += 1;
    }
}

fn finalize_modes<O, B>(
    operator: &mut O,
    entries: &[BlockEntry<B>],
    eigenvalues: &[f64],
    target: usize,
    tol: f64,
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
    for (entry, &lambda) in entries.iter().zip(eigenvalues.iter()) {
        if lambda < 0.0 {
            diagnostics.negative_modes_skipped += 1;
            continue;
        }
        let omega = lambda.sqrt();
        if let Some(prev) = last_kept {
            if (omega - prev).abs() <= freq_tol {
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
    op_matrix: Vec<f64>,
    mass_matrix: Vec<f64>,
    dim: usize,
    tol: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let l = cholesky_decompose(&mass_matrix, dim)?;
    let temp = solve_lower_triangular(&l, &op_matrix, dim)?;
    let c = solve_upper_triangular_right(&l, &temp, dim)?;
    let (values, eigenvectors) = jacobi_eigendecomposition(c, dim, tol);
    let coeffs = solve_upper_triangular(&l, &eigenvectors, dim)?;
    Some((values, coeffs))
}

fn cholesky_decompose(matrix: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = matrix[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                let diag = l[j * dim + j];
                if diag.abs() < 1e-12 {
                    return None;
                }
                l[i * dim + j] = sum / diag;
            }
        }
    }
    Some(l)
}

fn solve_lower_triangular(l: &[f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut x = vec![0.0; dim * dim];
    for col in 0..dim {
        for row in 0..dim {
            let mut sum = b[row * dim + col];
            for k in 0..row {
                sum -= l[row * dim + k] * x[k * dim + col];
            }
            let diag = l[row * dim + row];
            if diag.abs() < 1e-12 {
                return None;
            }
            x[row * dim + col] = sum / diag;
        }
    }
    Some(x)
}

fn solve_upper_triangular(l: &[f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut x = vec![0.0; dim * dim];
    for col in 0..dim {
        for rev_row in 0..dim {
            let row = dim - 1 - rev_row;
            let mut sum = b[row * dim + col];
            for k in (row + 1)..dim {
                sum -= l[k * dim + row] * x[k * dim + col];
            }
            let diag = l[row * dim + row];
            if diag.abs() < 1e-12 {
                return None;
            }
            x[row * dim + col] = sum / diag;
        }
    }
    Some(x)
}

fn solve_upper_triangular_right(l: &[f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut x = vec![0.0; dim * dim];
    for row in 0..dim {
        for rev_col in 0..dim {
            let col = dim - 1 - rev_col;
            let mut sum = b[row * dim + col];
            for k in (col + 1)..dim {
                sum -= x[row * dim + k] * l[k * dim + col];
            }
            let diag = l[col * dim + col];
            if diag.abs() < 1e-12 {
                return None;
            }
            x[row * dim + col] = sum / diag;
        }
    }
    Some(x)
}

#[cfg(test)]
mod eigensolver_internal_tests {
    use super::{EigenOptions, generalized_eigen};

    #[test]
    fn generalized_eigen_matches_diagonal_inputs() {
        let diag = [0.25, 1.0, 4.0];
        let dim = diag.len();
        let mut op = vec![0.0; dim * dim];
        let mut mass = vec![0.0; dim * dim];
        for i in 0..dim {
            op[i * dim + i] = diag[i];
            mass[i * dim + i] = 1.0;
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
