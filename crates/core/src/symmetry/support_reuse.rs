//! Invocation-local coefficient reuse with explicit support and target checks.
use super::{
    affine::AffineOp,
    galerkin::{self, GalerkinBudget, GalerkinResult},
    reuse::{ReuseReport, SymmetryOptions},
    support::FourierSupport,
};
use crate::{
    backend::SpectralBackend, eigensolver::EigensolverConfig, field::Field2D,
    operators::ThetaOperator,
};
use std::collections::VecDeque;

struct Entry {
    source: usize,
    support: FourierSupport,
    values: Vec<f64>,
    columns: Vec<Field2D>,
    bytes: usize,
}
pub(crate) struct SupportSpectralCache {
    entries: VecDeque<Entry>,
    bytes: usize,
    limit: usize,
}
impl SupportSpectralCache {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn insert(&mut self, source: usize, support: &FourierSupport, spectrum: &GalerkinResult) {
        if !spectrum.report.accepted
            || self
                .entries
                .iter()
                .any(|e| e.support.identity() == support.identity())
        {
            return;
        }
        let bytes = spectrum
            .coefficients
            .iter()
            .map(|f| f.len() * 16)
            .sum::<usize>()
            + spectrum.eigenvalues.len() * 8
            + support.index_payload_bytes();
        if bytes > self.limit {
            return;
        }
        while bytes > self.limit - self.bytes {
            self.bytes -= self.entries.pop_front().unwrap().bytes;
        }
        self.bytes += bytes;
        self.entries.push_back(Entry {
            source,
            support: support.clone(),
            values: spectrum.eigenvalues.clone(),
            columns: spectrum.coefficients.clone(),
            bytes,
        });
    }
    #[allow(clippy::too_many_arguments)]
    pub fn reuse<B: SpectralBackend + Clone>(
        &self,
        target: &mut ThetaOperator<B>,
        support: &FourierSupport,
        config: &EigensolverConfig,
        operations: &[AffineOp],
        options: &SymmetryOptions,
        budget: GalerkinBudget,
        report: &mut ReuseReport,
    ) -> Option<GalerkinResult> {
        crate::profile_scope!("symmetry.support.reuse");
        report.candidate_operation_count = operations.len();
        for entry in &self.entries {
            for &operation in operations {
                let mapped = operation.map_wavevector(entry.support.k());
                if (0..2).any(|i| {
                    let d = mapped[i] - support.k()[i];
                    (d - d.round()).abs()
                        > 8.0 * f64::EPSILON * mapped[i].abs().max(support.k()[i].abs()).max(1.0)
                }) {
                    continue;
                }
                let exact = operation == AffineOp::IDENTITY
                    && entry.support.identity() == support.identity();
                let (columns, qualification) = if exact {
                    (entry.columns.clone(), None)
                } else {
                    report.qualification_attempts += 1;
                    let (action, qualification) =
                        match super::support_qualification::qualify_transport(
                            target,
                            &entry.support,
                            support,
                            operation,
                            options.qualification_tolerance,
                            budget.max_estimated_payload_bytes,
                        ) {
                            Ok(v) => v,
                            Err(reason) => {
                                report.rejected.push(reason);
                                continue;
                            }
                        };
                    (
                        entry
                            .columns
                            .iter()
                            .map(|field| {
                                let mut out = Field2D::zeros(support.coordinate_grid());
                                action.apply(field.as_slice(), out.as_mut_slice());
                                out
                            })
                            .collect(),
                        Some(qualification),
                    )
                };
                match galerkin::certify_coefficients(
                    target,
                    support,
                    config,
                    entry.values.clone(),
                    columns,
                    budget,
                    "qualified_support_reuse",
                ) {
                    Ok(result) => {
                        report.source_index = Some(entry.source);
                        report.operation = Some(operation);
                        report.support_transport = qualification;
                        return Some(result);
                    }
                    Err(reason) => report.rejected.push(reason),
                }
            }
        }
        None
    }
}
