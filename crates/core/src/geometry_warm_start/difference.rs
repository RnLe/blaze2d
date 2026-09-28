//! Exact increments of the sampled finite pencil, including smoothing tensors.
use super::snapshot::FiberKey;
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    field::{Real, cscalar},
    operators::ThetaOperator,
    polarization::Polarization,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DifferenceReport {
    pub changed_epsilon_cells: usize,
    pub changed_tensor_cells: usize,
    pub epsilon_relative_frobenius: f64,
    pub inverse_relative_frobenius: f64,
    pub payload_bytes: usize,
}
pub struct DiscreteDifference {
    key: FiberKey,
    mass: Vec<f64>,
    inverse: Vec<f64>,
    tensors: Vec<[f64; 4]>,
    pub report: DifferenceReport,
}
pub(crate) fn inverse_tensor(d: &Dielectric2D, i: usize) -> [f64; 4] {
    d.inv_eps_tensors()
        .map_or([d.inv_eps()[i], 0., 0., d.inv_eps()[i]], |t| t[i])
}
impl DiscreteDifference {
    pub fn delta_epsilon(&self) -> &[f64] {
        &self.mass
    }
    pub fn delta_inverse_tensor(&self) -> &[[f64; 4]] {
        &self.tensors
    }

    pub fn new<B: SpectralBackend>(
        source: &ThetaOperator<B>,
        target: &ThetaOperator<B>,
    ) -> Result<Self, String> {
        Self::from_material(
            source.dielectric(),
            &FiberKey::from_operator(source),
            target,
        )
    }
    pub(crate) fn from_material<B: SpectralBackend>(
        source: &Dielectric2D,
        key: &FiberKey,
        target: &ThetaOperator<B>,
    ) -> Result<Self, String> {
        if key != &FiberKey::from_operator(target) {
            return Err("coefficient increments require the same physical Bloch fiber".into());
        }
        let next = target.dielectric();
        let n = target.grid().len();
        let mass: Vec<_> = next
            .eps()
            .iter()
            .zip(source.eps())
            .map(|(a, b)| a - b)
            .collect();
        let inverse: Vec<_> = next
            .inv_eps()
            .iter()
            .zip(source.inv_eps())
            .map(|(a, b)| a - b)
            .collect();
        let tensors: Vec<_> = (0..n)
            .map(|i| {
                let a = inverse_tensor(next, i);
                let b = inverse_tensor(source, i);
                std::array::from_fn(|j| a[j] - b[j])
            })
            .collect();
        if mass
            .iter()
            .chain(&inverse)
            .chain(tensors.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite sampled coefficient increment".into());
        }
        let epsilon_relative_frobenius = (mass.iter().map(|x| x * x).sum::<f64>()
            / source.eps().iter().map(|x| x * x).sum::<f64>())
        .sqrt();
        let inverse_relative_frobenius = (tensors.iter().flatten().map(|x| x * x).sum::<f64>()
            / (0..n)
                .flat_map(|i| inverse_tensor(source, i))
                .map(|x| x * x)
                .sum::<f64>())
        .sqrt();
        let report = DifferenceReport {
            changed_epsilon_cells: mass.iter().filter(|&&x| x != 0.).count(),
            changed_tensor_cells: tensors
                .iter()
                .filter(|t| t.iter().any(|&x| x != 0.))
                .count(),
            epsilon_relative_frobenius,
            inverse_relative_frobenius,
            payload_bytes: n * 6 * 8,
        };
        Ok(Self {
            key: key.clone(),
            mass,
            inverse,
            tensors,
            report,
        })
    }
    /// Return delta A x and delta B x. TM stiffness and TE mass are exactly
    /// invariant; their zero outputs require no FFT or identity mass copy.
    pub fn apply<B: SpectralBackend>(
        &self,
        target: &mut ThetaOperator<B>,
        x: &B::Buffer,
        delta_a: &mut B::Buffer,
        delta_b: &mut B::Buffer,
    ) -> Result<(), String> {
        if self.key != FiberKey::from_operator(target)
            || x.len() != target.grid().len()
            || delta_a.len() != x.len()
            || delta_b.len() != x.len()
        {
            return Err("coefficient increment action has incompatible coordinates".into());
        }
        match target.polarization() {
            Polarization::TM => {
                delta_a.as_mut_slice().fill(cscalar(0., 0.));
                for ((out, &v), &d) in delta_b
                    .as_mut_slice()
                    .iter_mut()
                    .zip(x.as_slice())
                    .zip(&self.mass)
                {
                    *out = v * B::Real::from_accum(d);
                }
                crate::profiler::count("geometry.delta_B.columns", 1);
                crate::profiler::count("geometry.delta_A.exact_zero_columns", 1);
            }
            Polarization::TE => {
                delta_b.as_mut_slice().fill(cscalar(0., 0.));
                target.apply_inverse_coefficient_increment(
                    x,
                    delta_a,
                    &self.inverse,
                    Some(&self.tensors),
                );
                crate::profiler::count("geometry.delta_B.exact_zero_columns", 1);
            }
        }
        Ok(())
    }
}
