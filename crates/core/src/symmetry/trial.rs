//! Shared trial-space algebra for qualified sectors.
//!
//! This is a bounded building block for a shared iterative engine. It does not
//! claim that the independent sector solver already shares an iteration loop.
//! The full and block modes consume exactly the same prepared fields. The
//! setup projects and rank-compresses each block in the target mass metric;
//! timed Gram and Ritz operations never change that trial space.

use num_complex::Complex64 as C;
use serde::Serialize;

use super::{affine::AffineGroup, reuse::SymmetryOptions, sectors::SectorPlan};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::dense::{DenseEigenResult, solve_hermitian_eigen},
    field::{Field2D, Real, cscalar},
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GramMode {
    Full,
    Blocks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RitzMode {
    Full,
    Blocks,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrialReport {
    pub rows: usize,
    pub block_columns: Vec<usize>,
    pub projection_leakage: f64,
    pub mass_orthogonality_defect: f64,
    pub persistent_field_payload_bytes: usize,
    pub full_gram_multiply_terms: usize,
    pub block_gram_multiply_terms: usize,
    pub full_dense_cubic_dimension: usize,
    pub block_dense_cubic_dimension: usize,
}

pub struct SharedSectorTrial<B: SpectralBackend> {
    vectors: Vec<B::Buffer>,
    applied: Vec<B::Buffer>,
    mass: Vec<B::Buffer>,
    ranges: Vec<std::ops::Range<usize>>,
    pub report: TrialReport,
}

/// A Hermitian matrix formed from one qualified, fixed trial space. Construction
/// is private so callers cannot label arbitrary nonzero couplings structural
/// zeros. The full matrix is retained for the dense reference and auditing.
pub struct ProjectedTrial {
    matrix: Vec<C>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl<B: SpectralBackend + Clone> SharedSectorTrial<B> {
    /// Project the same candidate block into every qualified isotypic sector.
    /// Source rank is capped per sector before the target-metric compression.
    /// This prepares a trial space only, never a lowest-spectrum certificate.
    pub fn prepare(
        operator: &mut ThetaOperator<B>,
        lattice: &Lattice2D,
        k: [f64; 2],
        group: AffineGroup,
        candidates: &[Field2D],
        per_sector_limit: usize,
        options: &SymmetryOptions,
    ) -> Result<Self, String> {
        if candidates.is_empty() || per_sector_limit == 0 {
            return Err("a sector trial requires candidates and a positive column limit".into());
        }
        if candidates.iter().any(|x| {
            (
                x.grid().nx,
                x.grid().ny,
                x.grid().lx,
                x.grid().ly,
                x.grid().centered,
            ) != (
                operator.grid().nx,
                operator.grid().ny,
                operator.grid().lx,
                operator.grid().ly,
                operator.grid().centered,
            )
        }) {
            return Err("trial fields differ from the target grid".into());
        }
        let mut qualification_options = options.clone();
        qualification_options.reconstruct_partners = false;
        let plan = SectorPlan::qualify(operator, lattice, k, group, &qualification_options)?;
        let mut scratch = operator.alloc_field();
        let mut projected = operator.alloc_field();
        let mut input = operator.alloc_field();
        let mut vectors = Vec::new();
        let mut ranges = Vec::new();
        let mut leakage: f64 = 0.0;
        for (sector, rank) in plan.ranks().into_iter().enumerate() {
            let start = vectors.len();
            if rank == 0 {
                ranges.push(start..start);
                continue;
            }
            let mut fields = Vec::new();
            for field in candidates.iter().take(per_sector_limit.min(rank)) {
                for (out, z) in input.as_mut_slice().iter_mut().zip(field.as_slice()) {
                    *out = cscalar::<B::Real>(z.re, z.im);
                }
                plan.project(sector, &input, &mut projected, &mut scratch);
                fields.push(promote::<B>(&projected));
            }
            let seed =
                super::seed::rank_aware_seed(operator, &fields, options.relative_rank_tolerance)
                    .map_err(|e| e.to_string())?;
            for field in seed.fields {
                let mut v = operator.alloc_field();
                for (out, z) in v.as_mut_slice().iter_mut().zip(field.as_slice()) {
                    *out = cscalar::<B::Real>(z.re, z.im);
                }
                plan.project(sector, &v, &mut projected, &mut scratch);
                let error = v
                    .as_slice()
                    .iter()
                    .zip(projected.as_slice())
                    .map(|(a, b)| {
                        let delta = C::new((a.re - b.re).to_accum(), (a.im - b.im).to_accum());
                        delta.norm_sqr()
                    })
                    .sum::<f64>();
                let norm = operator.backend().dot(&v, &v).re;
                leakage = leakage.max((error / norm).sqrt());
                vectors.push(v);
            }
            ranges.push(start..vectors.len());
        }
        if vectors.is_empty() || !leakage.is_finite() || leakage > 1e-9 {
            return Err("projected trial is empty or leaks between qualified sectors".into());
        }
        let mut applied: Vec<_> = vectors.iter().map(|_| operator.alloc_field()).collect();
        let mut mass: Vec<_> = vectors.iter().map(|_| operator.alloc_field()).collect();
        operator.batch_apply(&vectors, &mut applied);
        operator.batch_apply_mass(&vectors, &mut mass);
        // This full check is setup work in both modes. It guards roundoff and
        // rank decisions before any structural-zero shortcut is used.
        let gram = operator.backend().gram_matrix(&vectors, &mass);
        let n = vectors.len();
        let mass_defect = gram
            .iter()
            .enumerate()
            .map(|(i, z)| (*z - if i / n == i % n { C::ONE } else { C::ZERO }).norm_sqr())
            .sum::<f64>()
            .sqrt();
        if !mass_defect.is_finite() || mass_defect > 1e-8 {
            return Err(format!(
                "shared trial failed B-orthogonality: {mass_defect:e}"
            ));
        }
        let rows = operator.grid().len();
        let block_columns: Vec<_> = ranges.iter().map(|r| r.len()).collect();
        let report = TrialReport {
            rows,
            projection_leakage: leakage,
            mass_orthogonality_defect: mass_defect,
            persistent_field_payload_bytes: 3 * rows * n * std::mem::size_of::<C>(),
            full_gram_multiply_terms: rows * n * n,
            block_gram_multiply_terms: rows * block_columns.iter().map(|p| p * p).sum::<usize>(),
            full_dense_cubic_dimension: n * n * n,
            block_dense_cubic_dimension: block_columns.iter().map(|p| p * p * p).sum(),
            block_columns,
        };
        Ok(Self {
            vectors,
            applied,
            mass,
            ranges,
            report,
        })
    }

    pub fn columns(&self) -> usize {
        self.vectors.len()
    }

    /// Column-major Hermitian trial matrix. Cross-block work is omitted only
    /// for the qualified and leakage-checked partition created by `prepare`.
    pub fn project(&self, backend: &B, mode: GramMode, mass: bool) -> ProjectedTrial {
        crate::profile_scope!("symmetry.shared_gram");
        let n = self.columns();
        let right = if mass { &self.mass } else { &self.applied };
        let mut matrix = vec![C::ZERO; n * n];
        let ranges = if mode == GramMode::Full {
            vec![0..n]
        } else {
            self.ranges.clone()
        };
        for range in ranges {
            let width = range.len();
            if width == 0 {
                continue;
            }
            let gram = backend.gram_matrix(&self.vectors[range.clone()], &right[range.clone()]);
            crate::profiler::count(
                "shared_gram.multiply_terms",
                self.report.rows * width * width,
            );
            for (i, row) in range.clone().enumerate() {
                for (j, col) in range.clone().enumerate() {
                    matrix[row + n * col] =
                        0.5 * (gram[i * width + j] + gram[j * width + i].conj());
                }
            }
        }
        ProjectedTrial {
            matrix,
            ranges: self.ranges.clone(),
        }
    }

    pub fn reconstruct(&self, backend: &B, ritz: &DenseEigenResult) -> Vec<Field2D> {
        assert_eq!(ritz.eigenvectors.len(), self.columns() * self.columns());
        backend
            .linear_combinations(&self.vectors, &ritz.eigenvectors, self.columns())
            .iter()
            .map(promote::<B>)
            .collect()
    }
}

impl ProjectedTrial {
    pub fn matrix(&self) -> &[C] {
        &self.matrix
    }
    pub fn dimension(&self) -> usize {
        self.ranges.last().map_or(0, |r| r.end)
    }

    /// Relative Frobenius norm of the explicitly computed forbidden blocks.
    /// In block-Gram mode they are zero by construction; audit with Full mode.
    pub fn cross_block_defect(&self) -> f64 {
        let n = self.dimension();
        let mut labels = vec![0; n];
        for (label, range) in self.ranges.iter().enumerate() {
            labels[range.clone()].fill(label);
        }
        let mut error = 0.0;
        let mut norm = 0.0;
        for j in 0..n {
            for i in 0..n {
                let value = self.matrix[i + n * j].norm_sqr();
                norm += value;
                if labels[i] != labels[j] {
                    error += value;
                }
            }
        }
        (error / norm.max(f64::MIN_POSITIVE)).sqrt()
    }

    pub fn solve(&self, mode: RitzMode) -> Result<DenseEigenResult, String> {
        crate::profile_scope!("symmetry.shared_ritz");
        let n = self.dimension();
        if self
            .matrix
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err("nonfinite trial matrix".into());
        }
        if mode == RitzMode::Full {
            crate::profiler::count("shared_ritz.cubic_dimension", n * n * n);
            return Ok(solve_hermitian_eigen(&self.matrix, n));
        }
        let defect = self.cross_block_defect();
        if !defect.is_finite() || defect > 1e-10 {
            return Err(format!(
                "forbidden Ritz couplings exceed tolerance: {defect:e}"
            ));
        }
        let mut pairs = Vec::new();
        for range in &self.ranges {
            let width = range.len();
            if width == 0 {
                continue;
            }
            let mut block = vec![C::ZERO; width * width];
            for (i, row) in range.clone().enumerate() {
                for (j, col) in range.clone().enumerate() {
                    block[i + width * j] = self.matrix[row + n * col];
                }
            }
            crate::profiler::count("shared_ritz.cubic_dimension", width * width * width);
            let eig = solve_hermitian_eigen(&block, width);
            for j in 0..width {
                let mut column = vec![C::ZERO; n];
                column[range.clone()]
                    .copy_from_slice(&eig.eigenvectors[width * j..width * (j + 1)]);
                pairs.push((eig.eigenvalues[j], column));
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(DenseEigenResult {
            dim: n,
            eigenvalues: pairs.iter().map(|(v, _)| *v).collect(),
            eigenvectors: pairs.into_iter().flat_map(|(_, v)| v).collect(),
        })
    }
}

fn promote<B: SpectralBackend>(field: &B::Buffer) -> Field2D {
    Field2D::from_vec(
        field.grid(),
        field
            .as_slice()
            .iter()
            .map(|z| C::new(z.re.to_accum(), z.im.to_accum()))
            .collect(),
    )
}
