//! Selection rules in a common, freshly qualified source/target representation.
use super::{
    action::BlochActionPlan, affine::AffineGroup, qualification::qualify, reduced::OrbitBasis,
    representation::RepresentationPlan, response::ResponseWork, reuse::SymmetryOptions,
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    field::Real,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
};
use num_complex::Complex64 as C;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SelectionReport {
    pub mode_irreps: Vec<usize>,
    pub irrep_labels: Vec<String>,
    pub sector_ranks: Vec<usize>,
    pub retained_columns_per_sector: Vec<usize>,
    pub removed_deflation_pairs: usize,
    pub qualification_table_payload_bytes: usize,
    pub retained_basis_payload_bytes: usize,
    pub reduced_deflation_payload_bytes: usize,
    pub maximum_source_sector_defect: f64,
    pub maximum_rhs_sector_defect: f64,
}

pub(super) struct ResponseSelection<B: SpectralBackend> {
    pub bases: Vec<OrbitBasis>,
    pub deflation: Vec<Vec<B::Buffer>>,
    pub labels: Vec<usize>,
    pub report: SelectionReport,
}

impl<B: SpectralBackend + Clone> ResponseSelection<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: &mut ThetaOperator<B>,
        target: &mut ThetaOperator<B>,
        whitened: &[B::Buffer],
        lattice: &Lattice2D,
        k: [f64; 2],
        group: AffineGroup,
        options: &SymmetryOptions,
        work: &mut ResponseWork,
    ) -> Result<Self, String> {
        crate::profile_scope!("response.qualify_selection");
        let backend = source.backend().clone();
        let representation = RepresentationPlan::ordinary(group.clone(), k)
            .or_else(|_| RepresentationPlan::projective(group, k))
            .map_err(|e| e.to_string())?;
        let mut actions = Vec::new();
        let mut table_bytes = 0;
        // Qualify invariance separately, not a source-to-target intertwiner:
        // the material changes, but both pencils must preserve the same group.
        for (index, operator) in [source, target].into_iter().enumerate() {
            let mut duplicate = ThetaOperator::new(
                operator.backend().clone(),
                operator.dielectric().clone(),
                operator.polarization(),
                operator.bloch(),
            );
            for &operation in &representation.group.operations {
                let plan = BlochActionPlan::new(
                    operation,
                    k,
                    k,
                    operator.grid(),
                    operator.polarization(),
                    1e-12,
                )
                .map_err(|e| e.to_string())?;
                table_bytes += plan.storage_bytes();
                let result = qualify(
                    plan,
                    lattice,
                    operator,
                    &mut duplicate,
                    options.qualification_tolerance,
                );
                let probes = match &result {
                    Ok(a) => a.report().probe_count,
                    Err(r) => r.probe_count,
                };
                work.qualification_a_columns += 2 * probes;
                work.qualification_b_columns += 2 * probes;
                let action = result.map_err(|r| {
                    format!(
                        "response selection qualification: {:?}",
                        r.rejection_reasons
                    )
                })?;
                if index == 0 {
                    actions.push(action);
                }
            }
        }
        let mut bases = Vec::new();
        for i in 0..representation.irreps.len() {
            let weights = representation.isotypic_weights(i);
            let trace: C = actions
                .iter()
                .zip(&weights)
                .map(|(a, w)| a.plan().spatial_trace() * w)
                .sum();
            if trace.im.abs() > 1e-8
                || (trace.re - trace.re.round()).abs() > 1e-8
                || trace.re < -1e-8
            {
                return Err("nonintegral response sector dimension".into());
            }
            bases.push(OrbitBasis::from_projector(
                &actions,
                &weights,
                trace.re.round() as usize,
            )?);
        }
        if bases.iter().map(OrbitBasis::rank).sum::<usize>() != whitened[0].len() {
            return Err("response sectors do not cover the finite field space".into());
        }
        let mut deflation: Vec<Vec<B::Buffer>> = (0..bases.len()).map(|_| Vec::new()).collect();
        let mut labels = Vec::new();
        let mut maximum_source_defect: f64 = 0.0;
        for column in whitened {
            let norm = column
                .as_slice()
                .iter()
                .map(|z| z.re.to_accum().powi(2) + z.im.to_accum().powi(2))
                .sum::<f64>()
                .sqrt();
            let mut selected = None;
            for (i, basis) in bases.iter().enumerate().filter(|(_, b)| b.rank() > 0) {
                let mut small = backend.alloc_field(basis.coordinate_grid());
                basis.restrict(column, &mut small);
                let mut projected = backend.alloc_field(column.grid());
                basis.expand(&small, &mut projected);
                let defect = difference_norm(column, &projected) / norm.max(f64::MIN_POSITIVE);
                if defect <= options.max_orthogonality_defect {
                    if selected.is_some() {
                        return Err("ambiguous response sector label".into());
                    }
                    selected = Some((i, small, defect));
                }
            }
            let (i,small,defect) = selected.ok_or("source modes mix inequivalent irreps; separate the source cluster before applying response selection rules")?;
            maximum_source_defect = maximum_source_defect.max(defect);
            labels.push(i);
            deflation[i].push(small);
        }
        let counts: Vec<_> = deflation.iter().map(Vec::len).collect();
        let report = SelectionReport {
            mode_irreps: labels.clone(),
            irrep_labels: representation
                .irreps
                .iter()
                .map(|r| r.label.clone())
                .collect(),
            sector_ranks: bases.iter().map(OrbitBasis::rank).collect(),
            removed_deflation_pairs: whitened.len().pow(2)
                - counts.iter().map(|q| q * q).sum::<usize>(),
            retained_columns_per_sector: counts,
            qualification_table_payload_bytes: table_bytes,
            retained_basis_payload_bytes: bases.iter().map(OrbitBasis::payload_bytes).sum(),
            reduced_deflation_payload_bytes: deflation.iter().flatten().map(|v| v.len() * 16).sum(),
            maximum_source_sector_defect: maximum_source_defect,
            maximum_rhs_sector_defect: 0.0,
        };
        Ok(Self {
            bases,
            deflation,
            labels,
            report,
        })
    }
}

pub(super) fn difference_norm<F: SpectralBuffer>(a: &F, b: &F) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| {
            (x.re.to_accum() - y.re.to_accum()).powi(2)
                + (x.im.to_accum() - y.im.to_accum()).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}
