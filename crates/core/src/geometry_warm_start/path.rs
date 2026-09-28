//! Two immutable native path frames, with canonical tracking represented by indices.
use super::snapshot::{BasisGeneration, FiberKey, NativeSnapshot};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::subspace_prediction::{OverlapMatrix, polar_decomposition},
    field::Real,
    operators::ThetaOperator,
    polarization::Polarization,
};
use num_complex::Complex64 as C;
use serde::Serialize;
use std::{collections::VecDeque, sync::Arc};

struct Frame<R: Real> {
    snapshot: Arc<NativeSnapshot<R>>,
    k: [f64; 2],
    order: Vec<usize>,
}

pub struct PathHistory<R: Real> {
    frames: VecDeque<Frame<R>>,
    rotation: bool,
    extrapolation: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PathReport {
    pub method: &'static str,
    pub parents: Vec<BasisGeneration>,
    pub columns: usize,
    pub storage_bits: usize,
    pub minimum_overlap: Option<f64>,
    pub extrapolation_ratio: Option<f64>,
    /// Conservative resident payload; snapshots may also be shared with the bank.
    pub resident_bytes: usize,
}

impl<R: Real> PathHistory<R> {
    pub fn new(rotation: bool, extrapolation: bool) -> Self {
        Self {
            frames: VecDeque::new(),
            rotation,
            extrapolation,
        }
    }
    pub fn clear(&mut self) {
        self.frames.clear();
    }
    pub fn payload_bytes(&self) -> usize {
        self.frames
            .iter()
            .map(|f| f.snapshot.payload_bytes() + f.order.capacity() * std::mem::size_of::<usize>())
            .sum()
    }
    /// The caller supplies a freshly admitted frame and a public-band permutation.
    pub fn push(&mut self, snapshot: Arc<NativeSnapshot<R>>, k: [f64; 2], order: Vec<usize>) {
        if order.len() != snapshot.requested_bands
            || order.iter().any(|&i| snapshot.public_field(i).is_none())
            || (0..order.len()).any(|i| !order.contains(&i))
        {
            self.clear();
            return;
        }
        if self.frames.back().is_some_and(|f| {
            !f.snapshot.key().same_discretization(snapshot.key())
                || f.snapshot.material.eps() != snapshot.material.eps()
                || f.snapshot.material.inv_eps() != snapshot.material.inv_eps()
                || f.snapshot.material.inv_eps_tensors() != snapshot.material.inv_eps_tensors()
        }) {
            self.clear();
        }
        while self.frames.len() >= if self.rotation { 2 } else { 1 } {
            self.frames.pop_front();
        }
        self.frames.push_back(Frame { snapshot, k, order });
    }
    pub fn available<B: SpectralBackend<Real = R>>(&self, theta: &ThetaOperator<B>) -> bool {
        self.frames.back().is_some_and(|f| {
            f.snapshot
                .key()
                .same_discretization(&FiberKey::from_operator(theta))
                && f.snapshot.material.eps() == theta.dielectric().eps()
                && f.snapshot.material.inv_eps() == theta.dielectric().inv_eps()
                && f.snapshot.material.inv_eps_tensors() == theta.dielectric().inv_eps_tensors()
        })
    }
    /// No geometry transport or acceptance claim. Target initialization performs
    /// constraint projection, rank repair and Ritz extraction on this native block.
    pub fn predict<B: SpectralBackend<Real = R>>(
        &self,
        theta: &ThetaOperator<B>,
        k: [f64; 2],
    ) -> Option<(Vec<B::Buffer>, PathReport)> {
        if !self.available(theta) {
            return None;
        }
        crate::profile_scope!("geometry.path_prediction");
        let current = self.frames.back()?;
        let mut report = PathReport {
            method: "copy",
            parents: vec![current.snapshot.basis_generation()],
            columns: current.order.len(),
            storage_bits: 8 * std::mem::size_of::<R>(),
            minimum_overlap: None,
            extrapolation_ratio: None,
            resident_bytes: self.payload_bytes(),
        };
        let import = |frame: &Frame<R>| {
            frame
                .order
                .iter()
                .map(|&i| {
                    let field = frame.snapshot.public_field(i).unwrap();
                    let mut out = theta.alloc_field();
                    out.as_mut_slice().copy_from_slice(field.as_slice());
                    crate::profiler::count(
                        "geometry.path_import_bytes",
                        field.len() * std::mem::size_of::<num_complex::Complex<R>>(),
                    );
                    out
                })
                .collect::<Vec<_>>()
        };
        let current_fields = import(current);
        if self.frames.len() < 2 || !self.rotation {
            return Some((current_fields, report));
        }
        let previous = self.frames.front()?;
        let m = current.order.len();
        if previous.order.len() != m {
            return Some((current_fields, report));
        }
        let mut overlap = OverlapMatrix::new(m);
        let tm = theta.polarization() == Polarization::TM;
        // Only the small overlap matrix is promoted. Full history remains native.
        for i in 0..m {
            let left = previous.snapshot.public_field(previous.order[i]).unwrap();
            for j in 0..m {
                let right = current.snapshot.public_field(current.order[j]).unwrap();
                let value = left
                    .as_slice()
                    .iter()
                    .zip(right.as_slice())
                    .enumerate()
                    .map(|(row, (a, b))| {
                        C::new(a.re.to_accum(), a.im.to_accum()).conj()
                            * C::new(b.re.to_accum(), b.im.to_accum())
                            * if tm {
                                theta.dielectric().eps()[row]
                            } else {
                                1.
                            }
                    })
                    .sum();
                overlap.set(i, j, value);
            }
        }
        crate::profiler::count(
            "geometry.path_overlap_dot_terms",
            m * m * theta.grid().len(),
        );
        let (rotation, sigma) = polar_decomposition(&overlap);
        report.minimum_overlap = sigma.is_finite().then_some(sigma);
        if !sigma.is_finite() {
            return Some((current_fields, report));
        }
        let dk = [current.k[0] - previous.k[0], current.k[1] - previous.k[1]];
        let next = [k[0] - current.k[0], k[1] - current.k[1]];
        let norm = dk[0].hypot(dk[1]);
        let alpha = if norm > 1e-12 {
            next[0].hypot(next[1]) / norm
        } else {
            1.
        };
        // Bound extrapolation at unevenly sampled vertices; rotation preserves span.
        let extrapolate = self.extrapolation
            && sigma >= 0.1
            && dk[0] * next[0] + dk[1] * next[1] >= 0.
            && alpha.is_finite()
            && alpha <= 2.;
        let coefficients: Vec<_> = (0..m)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .map(|(i, j)| rotation.get(j, i).conj() * if extrapolate { 1. + alpha } else { 1. })
            .collect();
        let mut fields = theta
            .backend()
            .linear_combinations(&current_fields, &coefficients, m);
        report.method = if extrapolate {
            "extrapolation"
        } else {
            "rotation"
        };
        report
            .parents
            .insert(0, previous.snapshot.basis_generation());
        if extrapolate {
            report.extrapolation_ratio = Some(alpha);
            for (x, old) in fields.iter_mut().zip(import(previous)) {
                theta.backend().axpy(C::new(-alpha, 0.), &old, x);
            }
        }
        Some((fields, report))
    }
}
