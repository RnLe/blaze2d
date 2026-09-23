//! Fourier prolongation for a coarse-grid initializer of the unchanged fine
//! problem. This changes only the seed; the fine solver still needs complete
//! physical residual and spectrum validation.
use super::action::fft_indices;
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    field::{Field2D, Real, cscalar},
    grid::Grid2D,
};

/// Interpolate periodic complex fields by zero-padding the stored signed
/// Fourier support. Blaze stores an even axis's Nyquist index as positive;
/// that single mode is preserved, without splitting it into ± partners.
/// A real coarse Nyquist sample is therefore not promised a real interpolant.
/// Target B-normalization and Rayleigh-Ritz preparation are separate steps.
pub fn prolongate_fourier<B: SpectralBackend>(
    backend: &B,
    fields: &[Field2D],
    fine: Grid2D,
) -> Result<Vec<Field2D>, String> {
    crate::profile_scope!("initialization.coarse_prolongation");
    if std::mem::size_of::<B::Real>() != 8 {
        return Err("coarse Fourier prolongation currently requires f64".into());
    }
    if fine.nx == 0
        || fine.ny == 0
        || fine.centered
        || !fine.lx.is_finite()
        || !fine.ly.is_finite()
        || fine.lx <= 0.0
        || fine.ly <= 0.0
    {
        return Err("fine grid must be finite, positive and node-centered".into());
    }
    let Some(first) = fields.first() else {
        return Ok(Vec::new());
    };
    let coarse = first.grid();
    if coarse.nx == 0
        || coarse.ny == 0
        || coarse.centered
        || coarse.nx > fine.nx
        || coarse.ny > fine.ny
        || coarse.lx.to_bits() != fine.lx.to_bits()
        || coarse.ly.to_bits() != fine.ly.to_bits()
    {
        return Err(
            "coarse and fine grids must share the same node-centered cell with nested support"
                .into(),
        );
    }
    if fields.iter().any(|f| {
        let g = f.grid();
        (g.nx, g.ny, g.centered, g.lx.to_bits(), g.ly.to_bits())
            != (
                coarse.nx,
                coarse.ny,
                coarse.centered,
                coarse.lx.to_bits(),
                coarse.ly.to_bits(),
            )
            || f.as_slice()
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
    }) {
        return Err("coarse fields have inconsistent grids or nonfinite values".into());
    }
    let axis_map = |source: usize, target: usize| -> Result<Vec<usize>, String> {
        let indices = fft_indices(target);
        fft_indices(source)
            .into_iter()
            .map(|mode| {
                indices
                    .iter()
                    .position(|&m| m == mode)
                    .ok_or_else(|| "coarse Fourier support is not nested".into())
            })
            .collect()
    };
    let x_map = axis_map(coarse.nx, fine.nx)?;
    let y_map = axis_map(coarse.ny, fine.ny)?;
    let mut input = backend.alloc_field(coarse);
    let mut output = backend.alloc_field(fine);
    let scale = fine.len() as f64 / coarse.len() as f64;
    let mut result = Vec::with_capacity(fields.len());
    for field in fields {
        for (z, a) in input.as_mut_slice().iter_mut().zip(field.as_slice()) {
            *z = cscalar::<B::Real>(a.re, a.im);
        }
        backend.forward_fft_2d(&mut input);
        output.as_mut_slice().fill(cscalar::<B::Real>(0.0, 0.0));
        for (j, &y) in y_map.iter().enumerate() {
            for (i, &x) in x_map.iter().enumerate() {
                output.as_mut_slice()[fine.idx(x, y)] =
                    input.as_slice()[coarse.idx(i, j)] * B::Real::from_accum(scale);
            }
        }
        backend.inverse_fft_2d(&mut output);
        result.push(Field2D::from_vec(
            fine,
            output
                .as_slice()
                .iter()
                .map(|z| num_complex::Complex64::new(z.re.to_accum(), z.im.to_accum()))
                .collect(),
        ));
        crate::profiler::count("coarse.prolongated_columns", 1);
    }
    Ok(result)
}
