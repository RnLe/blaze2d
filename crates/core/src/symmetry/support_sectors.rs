//! Sparse symmetry sectors of an independently qualified Galerkin pencil.
//! Coefficient coordinates remain complex, including nonreal cyclic and glide
//! labels. Physical FFTs still use the complete embedding grid.
use std::collections::BTreeMap;

use crate::field::{Real, cscalar, from_accum};
use num_complex::{Complex, Complex64 as C};
use serde::Serialize;

use super::{support::FourierSupport, support_qualification::QualifiedSupportGroup};
use crate::{backend::SpectralBackend, grid::Grid2D, operators::ThetaOperator};

mod solve;
pub use solve::{
    SupportModeLabel, SupportSectorAttempt, SupportSectorOptions, SupportSectorReport,
    SupportSectorResult, solve, solve_with_plan,
};

/// Euclidean-orthonormal coefficient columns, each confined to a group orbit.
/// Constructed only from a qualified group; no dense M-by-M basis is retained.
#[derive(Debug, Clone)]
pub struct CoefficientBasis {
    full_dimension: usize,
    columns: Vec<Vec<(usize, C)>>,
}
impl CoefficientBasis {
    fn from_projector(
        group: &QualifiedSupportGroup,
        weights: &[C],
        expected_rank: usize,
    ) -> Result<Self, String> {
        let actions = group.actions();
        let m = group.support_identity().modes.len();
        let mut parent: Vec<_> = (0..m).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for action in actions {
            for (i, &j) in action.permutation().iter().enumerate() {
                let x = root(&mut parent, i);
                let y = root(&mut parent, j);
                parent[y] = x;
            }
        }
        let mut orbits: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..m {
            orbits.entry(root(&mut parent, i)).or_default().push(i);
        }
        let mut columns = Vec::with_capacity(expected_rank);
        for orbit in orbits.into_values() {
            let d = orbit.len();
            if d > actions.len() {
                return Err("coefficient actions do not close a finite orbit".into());
            }
            let mut projector = vec![C::ZERO; d * d];
            for (action, &weight) in actions.iter().zip(weights) {
                // CoefficientAction maps SOURCE to DESTINATION. Spatial
                // pullback permutations elsewhere use the inverse convention.
                for (j, &source) in orbit.iter().enumerate() {
                    let i = orbit
                        .binary_search(&action.permutation()[source])
                        .map_err(|_| "coefficient orbit is incomplete")?;
                    projector[i + d * j] += weight * action.phases()[source];
                }
            }
            for i in 0..d {
                for j in 0..d {
                    let square: C = (0..d)
                        .map(|r| projector[i + d * r] * projector[r + d * j])
                        .sum();
                    if (projector[i + d * j] - projector[j + d * i].conj()).norm() > 1e-9
                        || (square - projector[i + d * j]).norm() > 1e-9
                    {
                        return Err(
                            "coefficient weights do not form an orthogonal projector".into()
                        );
                    }
                }
            }
            let mut local: Vec<Vec<C>> = Vec::new();
            for j in 0..d {
                let mut v = projector[j * d..(j + 1) * d].to_vec();
                for _ in 0..2 {
                    for q in &local {
                        let dot: C = q.iter().zip(&v).map(|(x, y)| x.conj() * y).sum();
                        for (x, &y) in v.iter_mut().zip(q) {
                            *x -= dot * y;
                        }
                    }
                }
                let norm = v.iter().map(C::norm_sqr).sum::<f64>().sqrt();
                if norm <= 1e-10 {
                    continue;
                }
                for z in &mut v {
                    *z /= norm;
                }
                columns.push(
                    orbit
                        .iter()
                        .copied()
                        .zip(v.iter().copied())
                        .filter(|(_, z)| z.norm_sqr() > 0.0)
                        .collect(),
                );
                local.push(v);
            }
        }
        if columns.len() != expected_rank {
            return Err(format!(
                "coefficient basis rank {} differs from character rank {expected_rank}",
                columns.len()
            ));
        }
        Ok(Self {
            full_dimension: m,
            columns,
        })
    }
    pub fn rank(&self) -> usize {
        self.columns.len()
    }
    pub fn full_dimension(&self) -> usize {
        self.full_dimension
    }
    pub fn coordinate_grid(&self) -> Grid2D {
        Grid2D::new(self.rank(), 1, 1.0, 1.0)
    }
    /// Numerical index/coefficient payload, excluding Vec/allocator metadata.
    pub fn payload_bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|c| c.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<C>()))
            .sum()
    }
    pub fn expand<R: Real>(&self, small: &[Complex<R>], full: &mut [Complex<R>]) {
        assert_eq!(small.len(), self.rank());
        assert_eq!(full.len(), self.full_dimension);
        full.fill(cscalar::<R>(0.0, 0.0));
        for (column, &z) in self.columns.iter().zip(small) {
            for &(i, weight) in column {
                full[i] += from_accum::<R>(weight) * z;
            }
        }
        crate::profiler::count(
            "symmetry.support.orbit.expand_terms",
            self.columns.iter().map(Vec::len).sum(),
        );
    }
    pub fn restrict<R: Real>(&self, full: &[Complex<R>], small: &mut [Complex<R>]) {
        assert_eq!(full.len(), self.full_dimension);
        assert_eq!(small.len(), self.rank());
        for (column, z) in self.columns.iter().zip(small) {
            *z = column
                .iter()
                .map(|&(i, weight)| from_accum::<R>(weight.conj()) * full[i])
                .sum();
        }
        crate::profiler::count(
            "symmetry.support.orbit.restrict_terms",
            self.columns.iter().map(Vec::len).sum(),
        );
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SupportSectorDescriptor {
    pub label: String,
    pub irrep: usize,
    pub rank: usize,
    pub partner_rows: usize,
    pub owns_analytic_kernel: bool,
    pub basis_payload_bytes: usize,
    pub arithmetic: &'static str,
    pub representation_has_nonreal_entries: bool,
}

#[derive(Debug, Clone)]
struct Sector {
    descriptor: SupportSectorDescriptor,
    basis: CoefficientBasis,
    weights: Vec<C>,
    kernel: Option<Vec<C>>,
}

/// Contains either one row per irrep plus exact partner intertwiners, or the
/// complete isotypic sector. Both cover the same finite coefficient pencil.
#[derive(Debug, Clone)]
pub struct SupportSectorPlan {
    group: QualifiedSupportGroup,
    sectors: Vec<Sector>,
    estimated_plan_payload_bytes: usize,
    reconstruct_partners: bool,
}
impl SupportSectorPlan {
    pub fn new<B: SpectralBackend>(
        group: QualifiedSupportGroup,
        full: &ThetaOperator<B>,
        support: &FourierSupport,
        reconstruct_partners: bool,
        max_estimated_payload_bytes: usize,
    ) -> Result<Self, String> {
        crate::profile_scope!("symmetry.support.sector_plan");
        let m = support.len() as u128;
        let h = group.actions().len() as u128;
        // All sparse bases together have at most M*|H| nonzero entries.
        // Includes orbit construction, projected kernel and weight workspaces,
        // plus the qualified group's own conservative payload allowance.
        let bytes =
            group.report().estimated_payload_bytes as u128 + 24 * m * h + 128 * m + 128 * h * h;
        let estimated_plan_payload_bytes =
            usize::try_from(bytes).map_err(|_| "sector plan size overflow")?;
        if estimated_plan_payload_bytes > max_estimated_payload_bytes {
            return Err("coefficient sector plan exceeds its payload allowance".into());
        }
        group.validate_for(full, support)?;
        let kernel_index = support
            .embedding_indices()
            .iter()
            .position(|&i| full.k_plus_g_squares()[i] == 0.0);
        let mut kernel = vec![C::ZERO; support.len()];
        if let Some(i) = kernel_index {
            kernel[i] = C::ONE;
        }
        let representation = group.representation();
        let mut sectors = Vec::new();
        for (index, irrep) in representation.irreps.iter().enumerate() {
            let weights = if reconstruct_partners {
                representation.matrix_unit_weights(index, 0, 0)
            } else {
                representation.isotypic_weights(index)
            };
            let trace: C = group
                .actions()
                .iter()
                .zip(&weights)
                .map(|(a, &w)| a.trace().expect("qualified unitary group") * w)
                .sum();
            if trace.im.abs() > 1e-7
                || (trace.re - trace.re.round()).abs() > 1e-7
                || trace.re < -1e-7
            {
                return Err("nonintegral coefficient sector rank".into());
            }
            let rank = trace.re.round() as usize;
            let basis = CoefficientBasis::from_projector(&group, &weights, rank)?;
            let mut reduced_kernel = vec![C::ZERO; rank];
            basis.restrict(&kernel, &mut reduced_kernel);
            let norm = reduced_kernel.iter().map(C::norm_sqr).sum::<f64>();
            if kernel_index.is_some() && norm > 1e-9 && (norm - 1.0).abs() > 1e-9 {
                return Err("ambiguous coefficient kernel ownership".into());
            }
            let owns = norm > 0.5;
            sectors.push(Sector {
                descriptor: SupportSectorDescriptor {
                    label: irrep.label.clone(),
                    irrep: index,
                    rank,
                    partner_rows: if reconstruct_partners {
                        irrep.dimension
                    } else {
                        1
                    },
                    owns_analytic_kernel: owns,
                    basis_payload_bytes: basis.payload_bytes(),
                    arithmetic: "complex_f64",
                    representation_has_nonreal_entries: irrep
                        .matrices
                        .iter()
                        .flatten()
                        .any(|z| z.im.abs() > 1e-12),
                },
                basis,
                weights,
                kernel: owns.then_some(reduced_kernel),
            });
        }
        if sectors
            .iter()
            .map(|s| s.descriptor.rank * s.descriptor.partner_rows)
            .sum::<usize>()
            != support.len()
            || sectors.iter().filter(|s| s.kernel.is_some()).count()
                != usize::from(kernel_index.is_some())
        {
            return Err(
                "coefficient sectors do not cover the finite space and its kernel exactly once"
                    .into(),
            );
        }
        Ok(Self {
            group,
            sectors,
            estimated_plan_payload_bytes,
            reconstruct_partners,
        })
    }
    pub fn group(&self) -> &QualifiedSupportGroup {
        &self.group
    }
    pub fn descriptors(&self) -> impl Iterator<Item = &SupportSectorDescriptor> {
        self.sectors.iter().map(|s| &s.descriptor)
    }
    pub fn basis(&self, sector: usize) -> &CoefficientBasis {
        &self.sectors[sector].basis
    }
    pub fn estimated_plan_payload_bytes(&self) -> usize {
        self.estimated_plan_payload_bytes
    }
    pub fn validate_for<B: SpectralBackend>(
        &self,
        full: &ThetaOperator<B>,
        support: &FourierSupport,
    ) -> Result<(), String> {
        self.group.validate_for(full, support)
    }
    /// Apply the exact projector for the stored row or complete isotypic space.
    pub fn project(&self, sector: usize, input: &[C], output: &mut [C], scratch: &mut [C]) {
        self.apply_weights(&self.sectors[sector].weights, input, output, scratch);
    }
    /// Row zero maps to its requested partner. A one-row/isotypic sector only
    /// accepts row zero; no complex partner is silently merged or deduplicated.
    pub fn reconstruct_partner(
        &self,
        sector: usize,
        row: usize,
        input: &[C],
        output: &mut [C],
        scratch: &mut [C],
    ) -> Result<(), String> {
        let s = &self.sectors[sector];
        if row >= s.descriptor.partner_rows {
            return Err("partner row outside this sector plan".into());
        }
        if row == 0 {
            self.project(sector, input, output, scratch);
        } else {
            let weights =
                self.group
                    .representation()
                    .matrix_unit_weights(s.descriptor.irrep, row, 0);
            self.apply_weights(&weights, input, output, scratch);
        }
        Ok(())
    }
    fn apply_weights(&self, weights: &[C], input: &[C], output: &mut [C], scratch: &mut [C]) {
        let m = self.group.support_identity().modes.len();
        assert_eq!((input.len(), output.len(), scratch.len()), (m, m, m));
        output.fill(C::ZERO);
        for (action, &weight) in self.group.actions().iter().zip(weights) {
            action.apply(input, scratch);
            for (dst, &src) in output.iter_mut().zip(scratch.iter()) {
                *dst += weight * src;
            }
        }
        crate::profiler::count(
            "symmetry.support.project_coefficients",
            m * self.group.actions().len(),
        );
    }
}
