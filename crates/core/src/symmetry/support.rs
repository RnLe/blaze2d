//! A separately declared finite plane-wave space for a Galerkin pencil.
//!
//! Selecting this support changes the discretization. It never authorizes an
//! action on the original full FFT rectangle. Expansion still uses the full
//! embedding FFT; a smaller coefficient count is not a smaller-transform claim.
use std::collections::{HashMap, HashSet};

use crate::field::{Real, cscalar, from_accum, to_accum};
use num_complex::Complex;
use num_complex::Complex64 as C;
use serde::Serialize;

use super::affine::AffineOp;
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SupportBudget {
    pub max_coefficients: usize,
    pub max_embedding_points: usize,
    pub max_enumerated_indices: usize,
}

impl Default for SupportBudget {
    fn default() -> Self {
        Self {
            max_coefficients: 4096,
            max_embedding_points: 1_048_576,
            max_enumerated_indices: 4_194_304,
        }
    }
}

/// Lossless identity of the actual coefficient space, independent of a cutoff
/// label that might select the same modes. Material identity is additional.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SupportIdentity {
    pub version: &'static str,
    pub grid: [usize; 2],
    pub grid_length_bits: [u64; 2],
    pub lattice_bits: [[u64; 2]; 2],
    pub k_bits: [u64; 2],
    pub modes: Vec<[i64; 2]>,
}

#[derive(Debug, Clone)]
pub struct FourierSupport {
    grid: Grid2D,
    lattice: Lattice2D,
    k: [f64; 2],
    angular_cutoff: f64,
    modes: Vec<[i64; 2]>,
    embedding_indices: Vec<usize>,
}

fn same_grid(a: Grid2D, b: Grid2D) -> bool {
    a.nx == b.nx
        && a.ny == b.ny
        && a.centered == b.centered
        && a.lx.to_bits() == b.lx.to_bits()
        && a.ly.to_bits() == b.ly.to_bits()
}

impl FourierSupport {
    /// Keep every integer mode with |B(k+g)| <= cutoff, including boundary ties
    /// within a relative floating-point guard. Enumerate the complete metric
    /// ball before checking the embedding: never silently clip it to the FFT.
    /// The budgets bound indices/storage, not solver RSS or dense workspaces.
    pub fn metric_ball(
        grid: Grid2D,
        lattice: Lattice2D,
        k: [f64; 2],
        angular_cutoff: f64,
        budget: SupportBudget,
    ) -> Result<Self, String> {
        let points = grid
            .nx
            .checked_mul(grid.ny)
            .ok_or("embedding size overflow")?;
        if points == 0
            || grid.nx > i32::MAX as usize
            || grid.ny > i32::MAX as usize
            || grid.centered
            || !grid.lx.is_finite()
            || !grid.ly.is_finite()
            || grid.lx <= 0.0
            || grid.ly <= 0.0
            || !angular_cutoff.is_finite()
            || angular_cutoff <= 0.0
            || k.iter().any(|x| !x.is_finite() || x.abs() >= 1e6)
            || lattice.a1.iter().chain(&lattice.a2).any(|x| !x.is_finite())
            || !lattice.cell_area().is_finite()
            || lattice.cell_area() <= 0.0
        {
            return Err("invalid metric support input or unsupported sampling origin".into());
        }
        if points > budget.max_embedding_points || budget.max_coefficients == 0 {
            return Err("metric support exceeds its embedding/coefficient budget".into());
        }
        let reciprocal = lattice.reciprocal();
        if reciprocal
            .b1
            .iter()
            .chain(&reciprocal.b2)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite reciprocal metric".into());
        }
        // g_i+k_i = a_i dot q / (2*pi). These continuous bounds contain every
        // integer point of the ball, also for oblique primitive cells.
        let mut bounds = [[0i64; 2]; 2];
        for (i, a) in [lattice.a1, lattice.a2].iter().enumerate() {
            let radius = angular_cutoff * a[0].hypot(a[1]) / std::f64::consts::TAU;
            if !radius.is_finite() || radius >= i32::MAX as f64 {
                return Err("metric support enumeration exceeds coordinate bounds".into());
            }
            bounds[i] = [
                (-k[i] - radius).floor() as i64 - 1,
                (-k[i] + radius).ceil() as i64 + 1,
            ];
        }
        let enumerated = ((bounds[0][1] - bounds[0][0] + 1) as u64)
            .checked_mul((bounds[1][1] - bounds[1][0] + 1) as u64)
            .ok_or("metric support enumeration overflow")?;
        if enumerated > budget.max_enumerated_indices as u64 {
            return Err("metric support exceeds its enumeration budget".into());
        }
        let mut retained = Vec::new();
        for gy in bounds[1][0]..=bounds[1][1] {
            for gx in bounds[0][0]..=bounds[0][1] {
                let q = reciprocal.fractional_to_cartesian([gx as f64 + k[0], gy as f64 + k[1]]);
                let norm = q[0].hypot(q[1]);
                if !norm.is_finite() {
                    return Err("nonfinite mode norm".into());
                }
                if norm / angular_cutoff > 1.0 + 64.0 * f64::EPSILON {
                    continue;
                }
                for (g, n) in [(gx, grid.nx), (gy, grid.ny)] {
                    // Match action::fft_indices: the even Nyquist is POSITIVE.
                    // Modulo reduction of a missing negative Nyquist is invalid.
                    if g < -((n as i64 - 1) / 2) || g > n as i64 / 2 {
                        return Err(format!("metric ball leaves FFT embedding at [{gx}, {gy}]"));
                    }
                }
                if retained.len() == budget.max_coefficients {
                    return Err("metric support exceeds its coefficient budget".into());
                }
                let index = grid.idx(
                    gx.rem_euclid(grid.nx as i64) as usize,
                    gy.rem_euclid(grid.ny as i64) as usize,
                );
                retained.push((index, [gx, gy]));
            }
        }
        if retained.is_empty() {
            return Err("empty metric support".into());
        }
        retained.sort_unstable_by_key(|&(index, _)| index);
        let (embedding_indices, modes) = retained.into_iter().unzip();
        Ok(Self {
            grid,
            lattice,
            k,
            angular_cutoff,
            modes,
            embedding_indices,
        })
    }

    pub fn grid(&self) -> Grid2D {
        self.grid
    }
    pub fn lattice(&self) -> Lattice2D {
        self.lattice
    }
    pub fn k(&self) -> [f64; 2] {
        self.k
    }
    pub fn angular_cutoff(&self) -> f64 {
        self.angular_cutoff
    }
    pub fn len(&self) -> usize {
        self.modes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.modes.is_empty()
    }
    pub fn modes(&self) -> &[[i64; 2]] {
        &self.modes
    }
    pub fn embedding_indices(&self) -> &[usize] {
        &self.embedding_indices
    }
    pub fn coordinate_grid(&self) -> Grid2D {
        Grid2D::new(self.len(), 1, 1.0, 1.0)
    }
    pub fn index_payload_bytes(&self) -> usize {
        self.len() * (2 * std::mem::size_of::<i64>() + std::mem::size_of::<usize>())
    }
    pub fn identity(&self) -> SupportIdentity {
        SupportIdentity {
            version: "metric-plane-waves/1",
            grid: [self.grid.nx, self.grid.ny],
            grid_length_bits: [self.grid.lx.to_bits(), self.grid.ly.to_bits()],
            lattice_bits: [
                self.lattice.a1.map(f64::to_bits),
                self.lattice.a2.map(f64::to_bits),
            ],
            k_bits: self.k.map(f64::to_bits),
            modes: self.modes.clone(),
        }
    }

    /// S has orthonormal sampled plane-wave columns, normalized by sqrt(N).
    /// Input/output buffers are explicit. FFT plans and internal workspace are
    /// owned by the backend and may be allocated lazily on first use.
    pub fn expand<B: SpectralBackend, R: Real>(
        &self,
        backend: &B,
        coefficients: &[Complex<R>],
        output: &mut B::Buffer,
    ) {
        assert_eq!(coefficients.len(), self.len());
        assert!(same_grid(output.grid(), self.grid));
        crate::profiler::count("symmetry.support.expand_coefficients", self.len());
        output.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
        let scale = (self.grid.len() as f64).sqrt();
        for (&index, &value) in self.embedding_indices.iter().zip(coefficients) {
            output.as_mut_slice()[index] = from_accum(to_accum(value) * scale);
        }
        backend.inverse_fft_2d(output);
    }

    /// S† is the adjoint, not an interpolation onto a differently sampled grid.
    pub fn restrict<B: SpectralBackend, R: Real>(
        &self,
        backend: &B,
        input: &B::Buffer,
        scratch: &mut B::Buffer,
        coefficients: &mut [Complex<R>],
    ) {
        assert_eq!(coefficients.len(), self.len());
        assert!(same_grid(input.grid(), self.grid) && same_grid(scratch.grid(), self.grid));
        crate::profiler::count("symmetry.support.restrict_coefficients", self.len());
        scratch.as_mut_slice().copy_from_slice(input.as_slice());
        backend.forward_fft_2d(scratch);
        let scale = 1.0 / (self.grid.len() as f64).sqrt();
        for (&index, value) in self.embedding_indices.iter().zip(coefficients) {
            *value = from_accum(to_accum(scratch.as_slice()[index]) * scale);
        }
    }
}

/// Exact retained-index bijection and Bloch phases. This alone is not material,
/// metric or projected-operator qualification, and carries no such permission.
#[derive(Debug, Clone)]
pub struct CoefficientAction {
    operation: AffineOp,
    folding: [i64; 2],
    destination: Vec<usize>,
    phases: Vec<C>,
}

impl CoefficientAction {
    pub fn new(
        operation: AffineOp,
        source: &FourierSupport,
        target: &FourierSupport,
        polarization: Polarization,
        tolerance: f64,
    ) -> Result<Self, String> {
        if !tolerance.is_finite()
            || tolerance <= 0.0
            || tolerance >= 1e-3
            || !same_grid(source.grid, target.grid)
            || source.len() != target.len()
            || [
                source.lattice.a1.map(f64::to_bits),
                source.lattice.a2.map(f64::to_bits),
            ] != [
                target.lattice.a1.map(f64::to_bits),
                target.lattice.a2.map(f64::to_bits),
            ]
        {
            return Err("incompatible coefficient action inputs".into());
        }
        let raw = operation.map_wavevector(source.k);
        let mut folding = [0; 2];
        for i in 0..2 {
            let delta = raw[i] - target.k[i];
            if !delta.is_finite()
                || delta.abs() >= i64::MAX as f64
                || (delta - delta.round()).abs() > tolerance
            {
                return Err("coefficient action connects different Bloch fibers".into());
            }
            folding[i] = delta.round() as i64;
        }
        let lookup: HashMap<_, _> = target
            .modes
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i))
            .collect();
        let w = operation.reciprocal_matrix();
        let sign = if operation.conjugates() { -1i64 } else { 1 };
        let character = match polarization {
            Polarization::TM => 1.0,
            Polarization::TE => operation.determinant() as f64 * sign as f64,
        };
        let tau = operation.translation().map(|x| x.value());
        let mut destination = Vec::with_capacity(source.len());
        let mut phases = Vec::with_capacity(source.len());
        for &g in &source.modes {
            let mut mapped = [0; 2];
            for i in 0..2 {
                mapped[i] = w[i][0]
                    .checked_mul(g[0])
                    .and_then(|a| w[i][1].checked_mul(g[1]).and_then(|b| a.checked_add(b)))
                    .and_then(|v| v.checked_mul(sign))
                    .and_then(|v| v.checked_add(folding[i]))
                    .ok_or("coefficient action integer overflow")?;
            }
            destination.push(
                *lookup.get(&mapped).ok_or_else(|| {
                    format!("coefficient support is not closed: {g:?} → {mapped:?}")
                })?,
            );
            let q =
                operation.map_wavevector([g[0] as f64 + source.k[0], g[1] as f64 + source.k[1]]);
            let phase = C::from_polar(
                character,
                -std::f64::consts::TAU * (q[0] * tau[0] + q[1] * tau[1]),
            );
            if !phase.re.is_finite() || !phase.im.is_finite() {
                return Err("nonfinite Bloch phase".into());
            }
            phases.push(phase);
        }
        if destination.iter().copied().collect::<HashSet<_>>().len() != source.len() {
            return Err("coefficient action is not a bijection".into());
        }
        Ok(Self {
            operation,
            folding,
            destination,
            phases,
        })
    }

    pub fn operation(&self) -> AffineOp {
        self.operation
    }
    pub fn folding(&self) -> [i64; 2] {
        self.folding
    }
    pub fn permutation(&self) -> &[usize] {
        &self.destination
    }
    pub fn phases(&self) -> &[C] {
        &self.phases
    }
    pub fn trace(&self) -> Result<C, String> {
        if self.operation.conjugates() {
            return Err("antiunitary action has no character trace".into());
        }
        Ok(self
            .destination
            .iter()
            .enumerate()
            .filter_map(|(i, &j)| (i == j).then_some(self.phases[i]))
            .sum())
    }
    pub fn apply<R: Real>(&self, input: &[Complex<R>], output: &mut [Complex<R>]) {
        assert_eq!(input.len(), self.destination.len());
        assert_eq!(output.len(), input.len());
        for (i, &j) in self.destination.iter().enumerate() {
            let z = if self.operation.conjugates() {
                input[i].conj()
            } else {
                input[i]
            };
            output[j] = from_accum::<R>(self.phases[i]) * z;
        }
    }
}
