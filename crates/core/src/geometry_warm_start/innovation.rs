//! Compressed previous-span directions in the newest snapshot's mass metric.
//! Innovations are trial directions, never certified eigenpairs or guards.
use super::snapshot::{BasisGeneration, FiberKey, NativeSnapshot};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{initialization, normalization},
    field::{Field2D, Real},
    operators::{LinearOperator, ThetaOperator},
};
use num_complex::Complex64 as C;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CompressionReport {
    pub parents: Vec<BasisGeneration>,
    pub input_columns: usize,
    pub retained_columns: usize,
    pub singular_values: Vec<f64>,
    pub rank_threshold: f64,
    pub maximum_newest_overlap: f64,
    pub maximum_kernel_component: f64,
    pub payload_bytes: usize,
    pub workspace_payload_bytes: usize,
}
pub struct Innovations<R: Real> {
    pub(crate) fiber: FiberKey,
    pub(crate) generation: BasisGeneration,
    fields: Vec<Field2D<R>>,
    pub report: CompressionReport,
}
impl<R: Real> Innovations<R> {
    pub fn fields(&self) -> &[Field2D<R>] {
        &self.fields
    }
    pub fn payload_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.fields.capacity() * std::mem::size_of::<Field2D<R>>()
            + self
                .fields
                .iter()
                .map(|f| f.len() * std::mem::size_of::<num_complex::Complex<R>>())
                .sum::<usize>()
            + self.report.parents.capacity() * std::mem::size_of::<BasisGeneration>()
            + self.report.singular_values.capacity() * 8
    }
    pub fn matches(&self, snapshot: &NativeSnapshot<R>) -> bool {
        self.fiber == *snapshot.key() && self.generation == snapshot.basis_generation()
    }
    pub(crate) fn fit_payload(&mut self, limit: usize) {
        while !self.fields.is_empty() && self.payload_bytes() > limit {
            self.fields.pop();
            self.report.singular_values.pop();
        }
        self.report.retained_columns = self.fields.len();
        self.report.payload_bytes = self.payload_bytes();
    }
}

/// Compress (I - X_new X_new^H B_new) T_old,new X_old by its mass Gram
/// eigendecomposition. T is the TM metric isometry and TE identity. The newest
/// positive block is projected twice before numerical ranking. At Gamma the
/// stored directions may contain the analytic kernel: TM metric isometry moves
/// that component at the next geometry. Dropping it here would change the span
/// of the two-block history. The eventual target initializer applies its own
/// physical Gamma constraint before accepting any active column.
pub fn compress<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    newest: &NativeSnapshot<B::Real>,
    previous: &NativeSnapshot<B::Real>,
    max_columns: usize,
    relative_tolerance: f64,
    max_bytes: usize,
) -> Result<Innovations<B::Real>, String> {
    crate::profile_scope!("geometry.innovation_compression");
    let key = FiberKey::from_operator(theta);
    if newest.key() != &key
        || previous.key() != &key
        || !relative_tolerance.is_finite()
        || relative_tolerance <= 0.
        || relative_tolerance >= 1.
    {
        return Err(
            "innovation compression requires a compatible fiber and finite rank tolerance".into(),
        );
    }
    if newest.material.eps() != theta.dielectric().eps()
        || newest.material.inv_eps() != theta.dielectric().inv_eps()
        || newest.material.inv_eps_tensors() != theta.dielectric().inv_eps_tensors()
    {
        return Err("innovation frame must use the newest material metric".into());
    }
    let p = previous.fields().len();
    let n = theta.grid().len();
    let q = newest.fields().len() + usize::from(newest.kernel().is_some());
    let bytes = (4 * p as u128 + 2 * q as u128 + 8)
        * n as u128
        * std::mem::size_of::<num_complex::Complex<B::Real>>() as u128
        + (p as u128).pow(2) * 16 * 5;
    if bytes > max_bytes as u128 {
        return Err("innovation compression exceeds its workspace payload budget".into());
    }
    let mut basis = Vec::with_capacity(q);
    let kernel = theta
        .analytic_kernel_available()
        .then(|| initialization::create_gamma_mode(theta));
    for field in newest.fields() {
        let mut x = theta.alloc_field();
        x.as_mut_slice().copy_from_slice(field.as_slice());
        let mut bx = theta.alloc_field();
        theta.apply_mass(&x, &mut bx);
        basis.push((x, bx));
    }
    let mut vectors = super::predict::imported(theta, previous, true);
    let mut masses = Vec::with_capacity(p);
    for x in &mut vectors {
        let mut bx = theta.alloc_field();
        theta.apply_mass(x, &mut bx);
        for _ in 0..2 {
            for (q, bq) in &basis {
                normalization::project_out(theta.backend(), x, &mut bx, q, bq);
            }
            theta.apply_mass(x, &mut bx);
        }
        masses.push(bx);
    }
    let mut gram = vec![C::ZERO; p * p];
    for j in 0..p {
        for i in 0..=j {
            let value = 0.5
                * (theta.backend().dot(&vectors[i], &masses[j])
                    + theta.backend().dot(&vectors[j], &masses[i]).conj());
            gram[i + j * p] = value;
            gram[j + i * p] = value.conj();
        }
    }
    if gram.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err("nonfinite innovation Gram matrix".into());
    }
    let spectrum = theta.backend().dense_eigen(&gram, p);
    let maximum = spectrum
        .eigenvalues
        .last()
        .copied()
        .unwrap_or(0.)
        .max(0.)
        .sqrt();
    let noise = 32. * <B::Real as num_traits::Float>::epsilon().to_accum();
    let threshold = (relative_tolerance * maximum).max(noise);
    let selected: Vec<_> = (0..p)
        .rev()
        .filter(|&i| spectrum.eigenvalues[i] > threshold.powi(2))
        .take(max_columns)
        .collect();
    let coefficients: Vec<_> = selected
        .iter()
        .flat_map(|&j| {
            let norm = spectrum.eigenvalues[j].sqrt();
            (0..p).map(move |i| (i, j, norm))
        })
        .map(|(i, j, norm)| spectrum.eigenvectors[i + j * p] / norm)
        .collect();
    let candidates = if selected.is_empty() {
        Vec::new()
    } else {
        theta
            .backend()
            .linear_combinations(&vectors, &coefficients, selected.len())
    };
    drop(vectors);
    drop(masses);
    let mut kept = Vec::new();
    let mut kept_mass = Vec::new();
    let mut singular_values = Vec::new();
    let mut overlap = 0.0_f64;
    let mut kernel_component = 0.0_f64;
    for (mut x, &index) in candidates.into_iter().zip(&selected) {
        let mut bx = theta.alloc_field();
        theta.apply_mass(&x, &mut bx);
        // Reproject after dividing small innovations by their singular values.
        // Otherwise f32 projection noise would be amplified into the frame.
        for _ in 0..2 {
            for (q, bq) in &basis {
                normalization::project_out(theta.backend(), &mut x, &mut bx, q, bq);
            }
            for (q, bq) in kept.iter().zip(&kept_mass) {
                normalization::project_out(theta.backend(), &mut x, &mut bx, q, bq);
            }
            theta.apply_mass(&x, &mut bx);
        }
        if normalization::normalize_to_unit_b_norm_with_tol(theta.backend(), &mut x, &mut bx, noise)
            == 0.
        {
            continue;
        }
        theta.apply_mass(&x, &mut bx);
        for (q, _) in &basis {
            overlap = overlap.max(theta.backend().dot(q, &bx).norm());
        }
        if let Some((q, _, _)) = &kernel {
            kernel_component = kernel_component.max(theta.backend().dot(q, &bx).norm());
        }
        kept.push(x);
        kept_mass.push(bx);
        singular_values.push(spectrum.eigenvalues[index].sqrt());
    }
    let fields: Vec<_> = kept.into_iter().map(SpectralBuffer::into_field).collect();
    let mut result = Innovations {
        fiber: key,
        generation: newest.basis_generation(),
        report: CompressionReport {
            parents: vec![newest.basis_generation(), previous.basis_generation()],
            input_columns: p,
            retained_columns: fields.len(),
            singular_values,
            rank_threshold: threshold,
            maximum_newest_overlap: overlap,
            maximum_kernel_component: kernel_component,
            payload_bytes: 0,
            workspace_payload_bytes: bytes as usize,
        },
        fields,
    };
    result.report.payload_bytes = result.payload_bytes();
    Ok(result)
}
