//! Exact small pencils on an immutable native basis, not spectral certificates.
use super::{
    difference::inverse_tensor,
    snapshot::{BasisGeneration, FiberKey, NativeSnapshot},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    field::{Field2D, Real},
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
};
use num_complex::Complex64 as C;
use std::sync::Arc;

pub struct SmallPencil {
    pub dimension: usize,
    pub stiffness: Vec<C>,
    pub mass: Vec<C>,
}
enum Products<R: Real> {
    Tm {
        stiffness: Vec<C>,
        derivatives: [Vec<C>; 2],
        gram: Vec<C>,
        mass: Vec<C>,
    },
    Te {
        gradients: Vec<[Field2D<R>; 2]>,
        gram: Vec<C>,
    },
}
pub struct ProjectedPencil<R: Real> {
    key: FiberKey,
    generation: BasisGeneration,
    fields: Arc<[Field2D<R>]>,
    source_material: Dielectric2D,
    products: Products<R>,
    pub payload_bytes: usize,
}
fn accum<R: Real>(z: num_complex::Complex<R>) -> C {
    C::new(z.re.to_accum(), z.im.to_accum())
}
fn dot<R: Real>(x: &Field2D<R>, y: &[num_complex::Complex<R>]) -> C {
    x.as_slice()
        .iter()
        .zip(y)
        .map(|(&a, &b)| accum(a).conj() * accum(b))
        .sum()
}
impl<R: Real> ProjectedPencil<R> {
    pub fn prepare<B: SpectralBackend<Real = R>>(
        operator: &mut ThetaOperator<B>,
        source: &NativeSnapshot<R>,
        max_bytes: usize,
    ) -> Result<Self, String> {
        crate::profile_scope!("geometry.projected_prepare");
        let key = FiberKey::from_operator(operator);
        if source.key() != &key
            || source.material.eps() != operator.dielectric().eps()
            || source.material.inv_eps() != operator.dielectric().inv_eps()
            || source.material.inv_eps_tensors() != operator.dielectric().inv_eps_tensors()
        {
            return Err(
                "projected products require the exact source pencil and native basis".into(),
            );
        }
        let m = source.fields().len();
        let n = operator.grid().len();
        let payload =
            (m as u128) * (n as u128) * std::mem::size_of::<num_complex::Complex<R>>() as u128 * 3
                + (m as u128).pow(2) * 16 * 5
                + n as u128 * 8 * 8;
        if payload > max_bytes as u128 {
            return Err("fixed-basis products exceed their payload budget".into());
        }
        let fields = source.shared_fields();
        let mut gram = vec![C::ZERO; m * m];
        for j in 0..m {
            for i in 0..m {
                gram[i + j * m] = dot(&fields[i], fields[j].as_slice());
            }
        }
        let products = match operator.polarization() {
            Polarization::TM => {
                let mut stiffness = vec![C::ZERO; m * m];
                let mut mass = stiffness.clone();
                let mut derivatives = [stiffness.clone(), stiffness.clone()];
                let mut vector = operator.alloc_field();
                let mut output = operator.alloc_field();
                for j in 0..m {
                    vector.as_mut_slice().copy_from_slice(fields[j].as_slice());
                    if let Some(ax) = source.applied_for(&key) {
                        output.as_mut_slice().copy_from_slice(ax[j].as_slice());
                    } else {
                        operator.apply(&vector, &mut output);
                    }
                    for i in 0..m {
                        stiffness[i + j * m] = dot(&fields[i], output.as_slice());
                    }
                    for (axis, matrix) in derivatives.iter_mut().enumerate() {
                        operator.apply_dL_dk(&vector, &mut output, axis);
                        for i in 0..m {
                            matrix[i + j * m] = dot(&fields[i], output.as_slice());
                        }
                    }
                    for i in 0..m {
                        mass[i + j * m] = fields[i]
                            .as_slice()
                            .iter()
                            .zip(fields[j].as_slice())
                            .zip(source.material.eps())
                            .map(|((&x, &y), &b)| accum(x).conj() * accum(y) * b)
                            .sum();
                    }
                }
                Products::Tm {
                    stiffness,
                    derivatives,
                    gram,
                    mass,
                }
            }
            Polarization::TE => {
                let mut vector = operator.alloc_field();
                let gradients = fields
                    .iter()
                    .map(|x| {
                        vector.as_mut_slice().copy_from_slice(x.as_slice());
                        operator
                            .spatial_gradients(&vector)
                            .map(SpectralBuffer::into_field)
                    })
                    .collect();
                Products::Te { gradients, gram }
            }
        };
        Ok(Self {
            key,
            generation: source.basis_generation(),
            fields,
            source_material: source.material.clone(),
            products,
            payload_bytes: payload as usize,
        })
    }
    /// The basis generation is explicit at use sites. No vector transport,
    /// normalization or Ritz rotation may reuse products from another basis.
    pub fn evaluate<B: SpectralBackend<Real = R>>(
        &self,
        target: &ThetaOperator<B>,
        generation: BasisGeneration,
    ) -> Result<SmallPencil, String> {
        crate::profile_scope!("geometry.projected_update");
        let key = FiberKey::from_operator(target);
        if !self.key.same_discretization(&key) || generation != self.generation {
            return Err(
                "fixed-basis update has incompatible coordinates or basis generation".into(),
            );
        }
        let dk = [
            f64::from_bits(key.coordinate_bits[6]) - f64::from_bits(self.key.coordinate_bits[6]),
            f64::from_bits(key.coordinate_bits[7]) - f64::from_bits(self.key.coordinate_bits[7]),
        ];
        let m = self.fields.len();
        let n = target.grid().len();
        let mut stiffness = vec![C::ZERO; m * m];
        let mut mass = stiffness.clone();
        match &self.products {
            Products::Tm {
                stiffness: h,
                derivatives,
                gram,
                mass: b,
            } => {
                for j in 0..m {
                    for i in 0..m {
                        let index = i + j * m;
                        stiffness[index] = h[index]
                            + dk[0] * derivatives[0][index]
                            + dk[1] * derivatives[1][index]
                            + (dk[0] * dk[0] + dk[1] * dk[1]) * gram[index];
                        let delta: C = self.fields[i]
                            .as_slice()
                            .iter()
                            .zip(self.fields[j].as_slice())
                            .zip(
                                target
                                    .dielectric()
                                    .eps()
                                    .iter()
                                    .zip(self.source_material.eps()),
                            )
                            .map(|((&x, &y), (&next, &old))| {
                                accum(x).conj() * accum(y) * (next - old)
                            })
                            .sum();
                        mass[index] = b[index] + delta;
                    }
                }
            }
            Products::Te { gradients, gram } => {
                mass.copy_from_slice(gram);
                for j in 0..m {
                    for i in 0..m {
                        let mut sum = C::ZERO;
                        for cell in 0..n {
                            let eta = inverse_tensor(target.dielectric(), cell);
                            let xi = accum(self.fields[i].as_slice()[cell]);
                            let xj = accum(self.fields[j].as_slice()[cell]);
                            let gi = [
                                accum(gradients[i][0].as_slice()[cell]) + C::new(0., dk[0]) * xi,
                                accum(gradients[i][1].as_slice()[cell]) + C::new(0., dk[1]) * xi,
                            ];
                            let gj = [
                                accum(gradients[j][0].as_slice()[cell]) + C::new(0., dk[0]) * xj,
                                accum(gradients[j][1].as_slice()[cell]) + C::new(0., dk[1]) * xj,
                            ];
                            sum += gi[0].conj() * (eta[0] * gj[0] + eta[1] * gj[1])
                                + gi[1].conj() * (eta[2] * gj[0] + eta[3] * gj[1]);
                        }
                        stiffness[i + j * m] = sum;
                    }
                }
            }
        }
        crate::profiler::count("geometry.projected_dot_elements", n * m * m);
        Ok(SmallPencil {
            dimension: m,
            stiffness,
            mass,
        })
    }
}
