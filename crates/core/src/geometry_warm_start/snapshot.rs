//! Native immutable spectral donors. Public f64 export is a separate boundary.
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    field::{Field2D, Real},
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
};
use num_complex::Complex64 as C;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FiberKey {
    pub shape: [usize; 2],
    pub centered: bool,
    pub storage_bits: usize,
    /// Grid lengths, reciprocal lattice, Bloch k, and polarization tag.
    pub coordinate_bits: [u64; 9],
}
impl FiberKey {
    pub fn from_operator<B: SpectralBackend>(operator: &ThetaOperator<B>) -> Self {
        let g = operator.grid();
        let d = operator.dielectric();
        let k = operator.bloch();
        let b1 = d.reciprocal_b1();
        let b2 = d.reciprocal_b2();
        let bits = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
        Self {
            shape: [g.nx, g.ny],
            centered: g.centered,
            storage_bits: 8 * std::mem::size_of::<B::Real>(),
            coordinate_bits: [
                bits(g.lx),
                bits(g.ly),
                bits(b1[0]),
                bits(b1[1]),
                bits(b2[0]),
                bits(b2[1]),
                bits(k[0]),
                bits(k[1]),
                if operator.polarization() == Polarization::TM {
                    0
                } else {
                    1
                },
            ],
        }
    }
    pub fn is_gamma(&self) -> bool {
        self.coordinate_bits[6..8] == [0, 0]
    }
    pub fn same_discretization(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.centered == other.centered
            && self.storage_bits == other.storage_bits
            && self.coordinate_bits[..6] == other.coordinate_bits[..6]
            && self.coordinate_bits[8] == other.coordinate_bits[8]
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Quality {
    pub relative_residual: f64,
    pub absolute_residual: f64,
    pub orthogonality: f64,
}
impl Default for Quality {
    fn default() -> Self {
        Self {
            relative_residual: 1e-5,
            absolute_residual: 1e-7,
            orthogonality: 1e-8,
        }
    }
}
impl Quality {
    pub fn valid(&self) -> bool {
        [
            self.relative_residual,
            self.absolute_residual,
            self.orthogonality,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.)
    }
    pub fn accepts(&self, c: &NativeCertificate) -> bool {
        self.valid()
            && c.orthogonality.is_finite()
            && c.orthogonality <= self.orthogonality
            && c.kernel_leakage.is_finite()
            && c.kernel_leakage <= self.orthogonality
            && c.relative.len() == c.absolute.len()
            && c.relative.iter().zip(&c.absolute).all(|(&r, &a)| {
                r.is_finite()
                    && a.is_finite()
                    && (r <= self.relative_residual || a <= self.absolute_residual)
            })
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct NativeCertificate {
    pub relative: Vec<f64>,
    pub absolute: Vec<f64>,
    /// Absolute residual in the dual mass norm, with unit B-normalized vectors.
    pub dual: Vec<f64>,
    pub orthogonality: f64,
    pub kernel_leakage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BasisGeneration {
    pub job: usize,
    pub k_index: usize,
    pub accepted_generation: u64,
}

struct TmActions<R: Real> {
    fiber: FiberKey,
    basis_generation: BasisGeneration,
    fields: Arc<[Field2D<R>]>,
}

/// A block and its direct certificate, before cache admission. Fields include
/// explicitly solved guards; iterative slack is never imported here.
pub struct NativeSnapshot<R: Real> {
    pub(crate) key: FiberKey,
    pub(crate) basis_generation: BasisGeneration,
    pub(crate) parent_generations: Vec<BasisGeneration>,
    pub(crate) parameters: Vec<f64>,
    pub(crate) requested_bands: usize,
    pub(crate) guard_count: usize,
    pub(crate) eigenvalues: Vec<f64>,
    pub(crate) certificate: NativeCertificate,
    pub(crate) converged: bool,
    pub(crate) material: Dielectric2D,
    fields: Arc<[Field2D<R>]>,
    /// Full public Gamma kernel, kept out of the positive continuation block.
    kernel: Option<Field2D<R>>,
    /// Fresh actions on this exact immutable basis. Same-fiber TM permits reuse.
    applied: Option<TmActions<R>>,
}

impl<R: Real> NativeSnapshot<R> {
    pub(crate) fn shared_fields(&self)->Arc<[Field2D<R>]> {self.fields.clone()}
    pub fn fields(&self) -> &[Field2D<R>] {
        &self.fields
    }
    pub fn key(&self) -> &FiberKey {
        &self.key
    }
    pub fn basis_generation(&self) -> BasisGeneration {
        self.basis_generation
    }
    pub fn certificate(&self) -> &NativeCertificate {
        &self.certificate
    }
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
    pub fn guard_count(&self) -> usize {
        self.guard_count
    }
    pub fn kernel(&self) -> Option<&Field2D<R>> {
        self.kernel.as_ref()
    }
    pub fn applied(&self) -> Option<&[Field2D<R>]> {
        self.applied_for(&self.key)
    }
    /// A reusable action belongs to both this physical fiber and this basis.
    pub fn applied_for(&self, key: &FiberKey) -> Option<&[Field2D<R>]> {
        self.applied
            .as_ref()
            .filter(|a| a.fiber == *key && a.basis_generation == self.basis_generation)
            .map(|a| a.fields.as_ref())
    }
    pub fn positive_requested(&self) -> usize {
        self.requested_bands
            .saturating_sub(usize::from(self.kernel.is_some()))
    }
    pub fn admitted(&self, quality: Quality) -> bool {
        self.converged
            && self.fields.len() == self.positive_requested() + self.guard_count
            && self.eigenvalues.len() == self.fields.len()
            && quality.accepts(&self.certificate)
    }
    /// Conservative payload including material arrays. Shared material is
    /// charged per snapshot; allocator metadata and solver scratch are separate.
    pub fn payload_bytes(&self) -> usize {
        let n = self.key.shape[0] * self.key.shape[1];
        let columns = self.fields.len()
            + self.applied.as_ref().map_or(0, |x| x.fields.len())
            + usize::from(self.kernel.is_some());
        std::mem::size_of::<Self>()
            + self.parent_generations.capacity() * std::mem::size_of::<BasisGeneration>()
            + columns
                * (n * std::mem::size_of::<num_complex::Complex<R>>()
                    + std::mem::size_of::<Field2D<R>>())
            + (self.eigenvalues.capacity()
                + self.parameters.capacity()
                + self.certificate.relative.capacity()
                + self.certificate.absolute.capacity()
                + self.certificate.dual.capacity())
                * 8
            + n * (2
                + 4 * usize::from(self.material.inv_eps_tensors().is_some())
                + usize::from(self.material.unsmoothed_eps().is_some())
                + usize::from(self.material.unsmoothed_inv_eps().is_some()))
                * 8
    }

    #[allow(clippy::too_many_arguments)]
    pub fn capture<B: SpectralBackend<Real = R>>(
        operator: &mut ThetaOperator<B>,
        mut pairs: Vec<(f64, B::Buffer)>,
        requested_bands: usize,
        generation: BasisGeneration,
        parents: Vec<BasisGeneration>,
        parameters: Vec<f64>,
        converged: bool,
        retain_actions: bool,
    ) -> Result<Self, String> {
        crate::profile_scope!("geometry.capture");
        if operator.backend().real_coefficients() {
            return Err("native snapshots require physical spatial fields".into());
        }
        if requested_bands == 0
            || pairs.len() < requested_bands
            || parameters.iter().any(|x| !x.is_finite())
        {
            return Err("invalid native block dimensions or parameters".into());
        }
        let key = FiberKey::from_operator(operator);
        let grid = operator.grid();
        if pairs.iter().any(|(l, x)| {
            !l.is_finite()
                || x.len() != grid.len()
                || (
                    x.grid().nx,
                    x.grid().ny,
                    x.grid().lx,
                    x.grid().ly,
                    x.grid().centered,
                ) != (grid.nx, grid.ny, grid.lx, grid.ly, grid.centered)
                || x.as_slice()
                    .iter()
                    .any(|z| !z.re.to_accum().is_finite() || !z.im.to_accum().is_finite())
        }) {
            return Err("incompatible or nonfinite native solution".into());
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let guard_count = pairs.len() - requested_bands;
        let gamma = operator.analytic_kernel_available();
        if gamma && pairs[0].0 != 0.0 {
            return Err("analytic Gamma kernel is missing from the native solution".into());
        }
        let mut certificate = NativeCertificate::default();
        let mut mass =
            (operator.polarization() == Polarization::TM).then(|| operator.alloc_field());
        let mut applied = Vec::new();
        // Certify the stored fields with fresh target actions, never cached AX.
        for (j, (lambda, x)) in pairs.iter().enumerate() {
            let mut ax = operator.alloc_field();
            operator.apply(x, &mut ax);
            let bx = if let Some(ref mut mass) = mass {
                operator.apply_mass(x, mass);
                &*mass
            } else {
                crate::profiler::count("operator_B.identity_aliases", 1);
                x
            };
            let mut rr = 0.;
            let mut aa = 0.;
            let mut bb = 0.;
            let mut xx = 0.;
            let mut dual = 0.;
            for (i, ((a, b), x)) in ax
                .as_slice()
                .iter()
                .zip(bx.as_slice())
                .zip(x.as_slice())
                .enumerate()
            {
                let a = C::new(a.re.to_accum(), a.im.to_accum());
                let b = C::new(b.re.to_accum(), b.im.to_accum());
                let r = (a - *lambda * b).norm_sqr();
                rr += r;
                aa += a.norm_sqr();
                bb += b.norm_sqr();
                xx += x.re.to_accum().powi(2) + x.im.to_accum().powi(2);
                dual += r / if operator.polarization() == Polarization::TM {
                    operator.dielectric().eps()[i]
                } else {
                    1.
                };
            }
            let denominator = aa.sqrt() + lambda.abs() * bb.sqrt();
            certificate.relative.push(if denominator > 0. {
                rr.sqrt() / denominator
            } else {
                rr.sqrt()
            });
            certificate.absolute.push(if xx > 0. {
                (rr / xx).sqrt()
            } else {
                f64::INFINITY
            });
            certificate.dual.push(dual.sqrt());
            for (i, (_, y)) in pairs.iter().enumerate() {
                let gram = operator.backend().dot(y, bx);
                certificate.orthogonality = certificate
                    .orthogonality
                    .max((gram - C::new(if i == j { 1. } else { 0. }, 0.)).norm());
                if gamma && i == 0 && j > 0 {
                    certificate.kernel_leakage = certificate.kernel_leakage.max(gram.norm());
                }
            }
            if retain_actions && operator.polarization() == Polarization::TM && !(gamma && j == 0) {
                applied.push(ax.into_field());
            }
        }
        let kernel = if gamma {
            Some(pairs.remove(0).1.into_field())
        } else {
            None
        };
        let (eigenvalues, fields): (Vec<_>, Vec<_>) =
            pairs.into_iter().map(|(l, x)| (l, x.into_field())).unzip();
        let applied = if applied.is_empty() {
            None
        } else {
            Some(TmActions {
                fiber: key.clone(),
                basis_generation: generation,
                fields: applied.into(),
            })
        };
        Ok(Self {
            key,
            basis_generation: generation,
            parent_generations: parents,
            parameters,
            requested_bands,
            guard_count,
            eigenvalues,
            certificate,
            converged,
            material: operator.dielectric().clone(),
            fields: fields.into(),
            kernel,
            applied,
        })
    }
}
