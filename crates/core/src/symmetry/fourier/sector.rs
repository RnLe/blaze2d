//! Linear sector constraints in the qualified Fourier/PT coordinate gauge.
use super::*;
use crate::symmetry::qualification::QualifiedAction;

pub(crate) struct SectorProjection<'a> {
    pub actions: &'a [QualifiedAction],
    pub weights: &'a [C],
    pub dimension: usize,
    pub owns_gamma_kernel: bool,
    pub correction_projection_interval: usize,
}
impl SectorProjection<'_> {
    /// Inspect every nonzero projector entry after the diagonal PT gauge.
    /// Terms mapping the same source to the same destination must be summed
    /// before testing reality: individual group terms need not be real.
    pub(crate) fn reality_defect(&self, setup: &CoordinateSetup) -> Result<f64, String> {
        let mut entries = Vec::<(usize, C)>::with_capacity(self.actions.len());
        let mut imaginary = 0.0;
        let mut norm = 0.0;
        let mut maximum: f64 = 0.0;
        for source in 0..setup.gauge.len() {
            entries.clear();
            for (action, weight) in self.actions.iter().zip(self.weights) {
                let p = action.plan();
                if p.operation().conjugates() {
                    return Err("sector projector must be complex-linear".into());
                }
                let destination = p.fourier_permutation()[source];
                let value = weight
                    * p.fourier_phases()[source]
                    * setup.gauge[source]
                    * setup.gauge[destination].conj();
                if let Some((_, z)) = entries.iter_mut().find(|(i, _)| *i == destination) {
                    *z += value;
                } else {
                    entries.push((destination, value));
                }
            }
            for (_, z) in &entries {
                imaginary += z.im * z.im;
                norm += z.norm_sqr();
                maximum = maximum.max(z.im.abs());
            }
        }
        let defect = (imaginary / norm.max(f64::MIN_POSITIVE))
            .sqrt()
            .max(maximum);
        if !defect.is_finite() {
            return Err("nonfinite Fourier-sector reality check".into());
        }
        Ok(defect)
    }

    pub(crate) fn project<F: SpectralBuffer>(&self, input: &F, output: &mut F) {
        crate::profile_scope!("symmetry.project_fourier");
        output.as_mut_slice().fill(cscalar::<F::Real>(0.0, 0.0));
        for (action, weight) in self.actions.iter().zip(self.weights) {
            let p = action.plan();
            debug_assert!(!p.operation().conjugates());
            for (source, value) in input.as_slice().iter().enumerate() {
                let z = C::new(value.re.to_accum(), value.im.to_accum())
                    * p.fourier_phases()[source]
                    * weight;
                output.as_mut_slice()[p.fourier_permutation()[source]] +=
                    cscalar::<F::Real>(z.re, z.im);
            }
            crate::profiler::count("symmetry.fourier_sector_projection.elements", input.len());
        }
    }
}

/// Complete finite spectrum using literal coordinate axes. Packed real axes
/// include both scalar slots of each complex storage element.
pub(super) fn dense_complete<B: SpectralBackend + Clone>(
    op: &mut FourierOperator<'_, B>,
) -> Result<(Vec<f64>, Vec<Field2D>), String> {
    crate::profile_scope!("symmetry.dense_complete");
    let n = op.backend.nreal;
    let rank = op.subspace_dimension();
    if n.checked_mul(rank)
        .and_then(|x| x.checked_mul(48))
        .is_none_or(|x| x > 64 * 1024 * 1024)
    {
        return Err("complete sector basis exceeds 64 MiB workspace budget".into());
    }
    let mut q = Vec::<CoordinateBuffer>::new();
    let mut bq = Vec::<CoordinateBuffer>::new();
    for j in 0..n {
        let mut v = op.alloc_field();
        if op.backend.real_space {
            set_scalar(v.as_mut_slice(), j, 1.0, op.backend.packed);
        } else {
            v.as_mut_slice()[j] = C::ONE;
        }
        op.constrain(&mut v);
        let mut b = op.alloc_field();
        op.apply_mass(&v, &mut b);
        for _ in 0..2 {
            for (x, bx) in q.iter().zip(&bq) {
                let dot = op.backend.dot(x, &b);
                op.backend.axpy(-dot, x, &mut v);
                op.backend.axpy(-dot, bx, &mut b);
            }
        }
        let norm = op.backend.dot(&v, &b).re.max(0.0).sqrt();
        if norm <= 1e-10 {
            continue;
        }
        op.backend.scale(C::new(norm.recip(), 0.0), &mut v);
        op.backend.scale(C::new(norm.recip(), 0.0), &mut b);
        q.push(v);
        bq.push(b);
        if q.len() == rank {
            break;
        }
    }
    if q.len() != rank {
        return Err(format!(
            "Fourier-sector basis rank {} differs from trace {rank}",
            q.len()
        ));
    }
    let mut h = vec![C::ZERO; rank * rank];
    let mut a = op.alloc_field();
    for j in 0..rank {
        op.apply(&q[j], &mut a);
        for i in 0..rank {
            h[i + rank * j] = op.backend.dot(&q[i], &a);
        }
    }
    let eig = op.backend.dense_eigen(&h, rank);
    let mut fields = Vec::new();
    for j in 0..rank {
        let mut v = op.alloc_field();
        for i in 0..rank {
            op.backend
                .axpy(eig.eigenvectors[i + rank * j], &q[i], &mut v);
        }
        fields.push(v.to_field());
    }
    let mut values = eig.eigenvalues;
    if op.owns_gamma_kernel()
        && op.bloch() == [0.0; 2]
        && values.first().is_some_and(|v| v.abs() < 1e-8)
    {
        values[0] = 0.0;
    }
    Ok((values, fields))
}
