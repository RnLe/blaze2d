//! Periodic inverse maps for initializer transport, not transformed Maxwell pencils.
use super::{
    predict::{self, Prediction},
    snapshot::{FiberKey, NativeSnapshot},
};
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    field::{Field2D, Real},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
};
use num_complex::{Complex, Complex64 as C};
use serde::Serialize;
use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    Cubic,
    Fourier,
}
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub interpolation: Interpolation,
    pub jacobian: bool,
    pub max_stretch: f64,
    pub max_interpolation_terms: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            interpolation: Interpolation::Cubic,
            jacobian: false,
            max_stretch: 2.,
            max_interpolation_terms: 32_000_000,
        }
    }
}
impl Options {
    pub fn valid(self) -> bool {
        self.max_stretch.is_finite()
            && self.max_stretch > 1.
            && self.max_stretch <= 4.
            && self.max_interpolation_terms > 0
    }
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub accepted: bool,
    pub map: Option<&'static str>,
    pub interpolation: Option<Interpolation>,
    pub jacobian: bool,
    pub derivative_bound: f64,
    pub min_determinant: Option<f64>,
    pub max_determinant: Option<f64>,
    pub transported_columns: usize,
    pub trial_columns: usize,
    pub interpolation_terms: usize,
    pub forward_ffts: usize,
    pub inverse_ffts: usize,
    pub estimated_peak_payload_bytes: usize,
    pub target_preparation_columns: usize,
    pub reason: Option<String>,
}
#[derive(Debug, Clone)]
struct Patch {
    center: [f64; 2],
    delta: [f64; 2],
    radius: f64,
    support: f64,
    scale: f64,
}
#[derive(Debug, Clone)]
pub struct Map {
    lattice: Lattice2D,
    k: [f64; 2],
    rigid: Option<[f64; 2]>,
    patches: Vec<Patch>,
    pub derivative_bound: f64,
}
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub source: [f64; 2],
    pub displacement: [f64; 2],
    pub determinant: f64,
}
fn wrap(x: f64) -> f64 {
    (x + 0.5).rem_euclid(1.) - 0.5
}
fn norm(x: [f64; 2]) -> f64 {
    x[0].hypot(x[1])
}
fn add(x: [f64; 2], y: [f64; 2]) -> [f64; 2] {
    [x[0] + y[0], x[1] + y[1]]
}
fn close(x: f64, y: f64) -> bool {
    (x - y).abs() <= 64. * f64::EPSILON * x.abs().max(y.abs()).max(1.)
}
impl Map {
    /// The parameter layout is background epsilon followed by (fractional x,
    /// fractional y, radius in lattice units, epsilon) for each ordered circle.
    pub fn new(
        key: &FiberKey,
        source: &[f64],
        target: &[f64],
        options: Options,
    ) -> Result<Self, String> {
        if !options.valid()
            || source.len() != target.len()
            || source.is_empty()
            || (source.len() - 1) % 4 != 0
            || source.iter().chain(target).any(|x| !x.is_finite())
        {
            return Err(
                "transport needs matching finite ordered circular geometries and valid limits"
                    .into(),
            );
        }
        let c = key.coordinate_bits.map(f64::from_bits);
        let det = c[2] * c[5] - c[3] * c[4];
        if !det.is_finite() || det == 0. {
            return Err("singular transport lattice".into());
        }
        let lattice = Lattice2D::oblique(
            [TAU * c[5] / det, -TAU * c[4] / det],
            [-TAU * c[3] / det, TAU * c[2] / det],
        );
        let unit = lattice.characteristic_length();
        let height = (TAU / c[2].hypot(c[3])).min(TAU / c[4].hypot(c[5]));
        let mut result = Self {
            lattice,
            k: [c[6], c[7]],
            rigid: None,
            patches: Vec::new(),
            derivative_bound: 0.,
        };
        let pairs: Vec<_> = source[1..]
            .chunks_exact(4)
            .zip(target[1..].chunks_exact(4))
            .collect();
        if pairs.iter().any(|(s, t)| s[2] <= 0. || t[2] <= 0.) {
            return Err("transport needs positive circle radii".into());
        }
        let shift = pairs
            .first()
            .map_or([0., 0.], |(s, t)| [wrap(s[0] - t[0]), wrap(s[1] - t[1])]);
        if pairs.iter().all(|(s, t)| {
            close(s[2], t[2])
                && close(wrap(s[0] - t[0] - shift[0]), 0.)
                && close(wrap(s[1] - t[1] - shift[1]), 0.)
        }) {
            result.rigid = Some(shift);
            return Ok(result);
        }
        for (i, (s, t)) in pairs.iter().enumerate() {
            let delta = lattice.fractional_to_cartesian([wrap(s[0] - t[0]), wrap(s[1] - t[1])]);
            let scale = s[2] / t[2] - 1.;
            if close(scale, 0.) && norm(delta) <= 64. * f64::EPSILON * unit {
                continue;
            }
            let mut support = 0.45 * height;
            for (j, (_, other)) in pairs.iter().enumerate() {
                if i == j {
                    continue;
                }
                let base = [wrap(t[0] - other[0]), wrap(t[1] - other[1])];
                // Any image capable of overlapping these small patches lies in
                // this neighborhood, even on an oblique cell.
                for iy in -1..=1 {
                    for ix in -1..=1 {
                        let distance =
                            norm(lattice.fractional_to_cartesian([
                                base[0] + ix as f64,
                                base[1] + iy as f64,
                            ]));
                        support = support.min(0.45 * distance);
                        // A stationary large neighbor must also stay outside.
                        support = support.min(distance - other[2] * unit);
                    }
                }
            }
            let radius = t[2] * unit;
            if support <= radius.max(s[2] * unit) || !support.is_finite() {
                return Err(
                    "circle supports overlap or leave no periodic transition annulus".into(),
                );
            }
            let bound =
                scale.abs() + 1.875 / (support - radius) * (norm(delta) + scale.abs() * support);
            if !bound.is_finite() || bound >= 1. - 1. / options.max_stretch {
                return Err("map exceeds its global injectivity/stretch bound".into());
            }
            result.derivative_bound = result.derivative_bound.max(bound);
            result.patches.push(Patch {
                center: [wrap(t[0]), wrap(t[1])],
                delta,
                radius,
                support,
                scale,
            });
        }
        Ok(result)
    }
    pub fn kind(&self) -> &'static str {
        match self.rigid {
            Some([0., 0.]) => "identity",
            Some(_) => "rigid",
            None => "local",
        }
    }
    pub fn sample(&self, fractional: [f64; 2]) -> Sample {
        if let Some(shift) = self.rigid {
            return Sample {
                source: add(fractional, shift),
                displacement: self.lattice.fractional_to_cartesian(shift),
                determinant: 1.,
            };
        }
        for p in &self.patches {
            let r = self.lattice.fractional_to_cartesian([
                wrap(fractional[0] - p.center[0]),
                wrap(fractional[1] - p.center[1]),
            ]);
            let radius = norm(r);
            if radius >= p.support {
                continue;
            }
            let (w, dw) = if radius <= p.radius {
                (1., 0.)
            } else {
                let t = (radius - p.radius) / (p.support - p.radius);
                (
                    1. - t * t * t * (10. - 15. * t + 6. * t * t),
                    -30. * t * t * (1. - t).powi(2) / (p.support - p.radius),
                )
            };
            let v = add(p.delta, [p.scale * r[0], p.scale * r[1]]);
            let displacement = [w * v[0], w * v[1]];
            let gradient = if radius > 0. {
                [dw * r[0] / radius, dw * r[1] / radius]
            } else {
                [0., 0.]
            };
            let diagonal = 1. + p.scale * w;
            // Matrix determinant lemma for diagonal*I + v outer grad(w).
            let determinant = diagonal * (diagonal + v[0] * gradient[0] + v[1] * gradient[1]);
            return Sample {
                source: add(
                    fractional,
                    self.lattice.cartesian_to_fractional(displacement),
                ),
                displacement,
                determinant,
            };
        }
        Sample {
            source: fractional,
            displacement: [0., 0.],
            determinant: 1.,
        }
    }
    fn phase(&self, s: Sample, jacobian: bool) -> C {
        C::from_polar(
            if jacobian { s.determinant.sqrt() } else { 1. },
            self.k[0] * s.displacement[0] + self.k[1] * s.displacement[1],
        )
    }
}
fn signed(i: usize, n: usize) -> f64 {
    if i <= n / 2 {
        i as f64
    } else {
        i as f64 - n as f64
    }
}
fn offset(n: usize, centered: bool) -> f64 {
    if centered { 0.5 / n as f64 - 0.5 } else { 0. }
}
fn cubic(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1. - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2. * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}
fn cast<R: Real>(v: C) -> Complex<R> {
    Complex::new(R::from_accum(v.re), R::from_accum(v.im))
}
fn accum<R: Real>(v: Complex<R>) -> C {
    C::new(v.re.to_accum(), v.im.to_accum())
}

/// Evaluates the periodic field pullback including its Bloch phase. The rigid
/// branch is exact in represented Fourier space, independent of interpolation.
/// Callers still need target constraints, rank repair and fresh target actions.
pub fn transport<B: SpectralBackend>(
    backend: &B,
    grid: Grid2D,
    fields: &[Field2D<B::Real>],
    map: &Map,
    options: Options,
    max_bytes: usize,
    report: &mut Report,
) -> Result<Vec<B::Buffer>, String> {
    crate::profile_scope!("geometry.shape_transport");
    let n = grid.len();
    let p = fields.len();
    report.map = Some(map.kind());
    report.interpolation = Some(options.interpolation);
    report.jacobian = options.jacobian;
    report.derivative_bound = map.derivative_bound;
    if !options.valid()
        || n == 0
        || p == 0
        || fields.iter().any(|f| {
            f.grid().nx != grid.nx || f.grid().ny != grid.ny || f.grid().centered != grid.centered
        })
    {
        return Err("invalid transport fields or options".into());
    }
    let terms = (n as u128)
        * (p as u128)
        * if map.rigid.is_some() {
            1
        } else if options.interpolation == Interpolation::Cubic {
            16
        } else {
            n as u128
        };
    if terms > options.max_interpolation_terms as u128 {
        return Err("transport exceeds its interpolation work budget".into());
    }
    // Output, one FFT buffer, query coordinates and separable cubic stencils.
    let bytes = (p as u128 + 2) * n as u128 * std::mem::size_of::<Complex<B::Real>>() as u128
        + n as u128 * 256;
    report.estimated_peak_payload_bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
    if bytes > max_bytes as u128 {
        return Err("transport exceeds its payload budget".into());
    }
    let samples: Vec<_> = (0..n)
        .map(|i| {
            map.sample([
                offset(grid.nx, grid.centered) + (i % grid.nx) as f64 / grid.nx as f64,
                offset(grid.ny, grid.centered) + (i / grid.nx) as f64 / grid.ny as f64,
            ])
        })
        .collect();
    let min = samples
        .iter()
        .map(|s| s.determinant)
        .fold(f64::INFINITY, f64::min);
    let max = samples.iter().map(|s| s.determinant).fold(0., f64::max);
    report.min_determinant = Some(min);
    report.max_determinant = Some(max);
    if !min.is_finite() || min <= 0. || !max.is_finite() {
        return Err("invalid inverse-map Jacobian".into());
    }
    let phases: Vec<_> = samples
        .iter()
        .map(|s| map.phase(*s, options.jacobian))
        .collect();
    let mut outputs = Vec::with_capacity(p);
    if let Some(shift) = map.rigid {
        let grid_shift = [shift[0] * grid.nx as f64, shift[1] * grid.ny as f64];
        let integer = grid_shift.iter().all(|x| close(*x, x.round()));
        let multipliers: Vec<_> = (0..n)
            .map(|i| {
                phases[0]
                    * C::from_polar(
                        1.,
                        TAU * (signed(i % grid.nx, grid.nx) * shift[0]
                            + signed(i / grid.nx, grid.ny) * shift[1]),
                    )
            })
            .collect();
        for field in fields {
            let mut out = backend.alloc_field(grid);
            if integer {
                for (i, v) in out.as_mut_slice().iter_mut().enumerate() {
                    let x = (i % grid.nx) as isize + grid_shift[0].round() as isize;
                    let y = (i / grid.nx) as isize + grid_shift[1].round() as isize;
                    *v = cast(
                        accum(
                            field.as_slice()[y.rem_euclid(grid.ny as isize) as usize * grid.nx
                                + x.rem_euclid(grid.nx as isize) as usize],
                        ) * phases[0],
                    );
                }
            } else {
                out.as_mut_slice().copy_from_slice(field.as_slice());
                backend.forward_fft_2d(&mut out);
                report.forward_ffts += 1;
                for (v, m) in out.as_mut_slice().iter_mut().zip(&multipliers) {
                    *v = cast(accum(*v) * m);
                }
                backend.inverse_fft_2d(&mut out);
                report.inverse_ffts += 1;
            }
            outputs.push(out);
        }
    } else if options.interpolation == Interpolation::Cubic {
        let queries: Vec<_> = samples
            .iter()
            .map(|s| {
                let x = (s.source[0] - offset(grid.nx, grid.centered)) * grid.nx as f64;
                let y = (s.source[1] - offset(grid.ny, grid.centered)) * grid.ny as f64;
                (
                    x.floor() as isize,
                    y.floor() as isize,
                    cubic(x - x.floor()),
                    cubic(y - y.floor()),
                )
            })
            .collect();
        for field in fields {
            let mut out = backend.alloc_field(grid);
            for (i, (x, y, wx, wy)) in queries.iter().enumerate() {
                let mut sum = C::ZERO;
                for dy in 0..4 {
                    for dx in 0..4 {
                        let ix = (x + dx as isize - 1).rem_euclid(grid.nx as isize) as usize;
                        let iy = (y + dy as isize - 1).rem_euclid(grid.ny as isize) as usize;
                        sum += accum(field.as_slice()[iy * grid.nx + ix]) * wx[dx] * wy[dy];
                    }
                }
                out.as_mut_slice()[i] = cast(sum * phases[i]);
            }
            outputs.push(out);
        }
    } else {
        for field in fields {
            let mut frequency = backend.alloc_field(grid);
            frequency.as_mut_slice().copy_from_slice(field.as_slice());
            backend.forward_fft_2d(&mut frequency);
            report.forward_ffts += 1;
            let mut out = backend.alloc_field(grid);
            for (i, s) in samples.iter().enumerate() {
                let q = [
                    s.source[0] - offset(grid.nx, grid.centered),
                    s.source[1] - offset(grid.ny, grid.centered),
                ];
                let sum = frequency
                    .as_slice()
                    .iter()
                    .enumerate()
                    .map(|(j, v)| {
                        accum(*v)
                            * C::from_polar(
                                1.,
                                TAU * (signed(j % grid.nx, grid.nx) * q[0]
                                    + signed(j / grid.nx, grid.ny) * q[1]),
                            )
                    })
                    .sum::<C>();
                out.as_mut_slice()[i] = cast(sum * phases[i] / n as f64);
            }
            outputs.push(out);
        }
    }
    report.interpolation_terms = terms as usize;
    report.transported_columns = p;
    crate::profiler::count("geometry.shape_interpolation_terms", terms as usize);
    Ok(outputs)
}

/// Enrich an independent predictor with a transported source. Applying the map
/// to an already predicted response would count the first-order motion twice.
#[allow(clippy::too_many_arguments)]
pub fn prepare<B: SpectralBackend>(
    theta: &mut ThetaOperator<B>,
    source: &NativeSnapshot<B::Real>,
    parameters: &[f64],
    options: Options,
    predictor: Option<&[B::Buffer]>,
    retained: usize,
    max_columns: usize,
    max_bytes: usize,
) -> (Option<Prediction<B>>, Report) {
    crate::profile_scope!("geometry.shape_prepare");
    let mut report = Report::default();
    let attempt = (|| {
        if source.key() != &FiberKey::from_operator(theta) {
            return Err("transport requires the same physical Bloch fiber".into());
        }
        let map = Map::new(source.key(), &source.parameters, parameters, options)?;
        report.map = Some(map.kind());
        if map.kind() == "identity" {
            return Err("geometry map is identity; retain the ordinary initializer".into());
        }
        let base = predictor.map_or(0, |v| v.len());
        let width = retained
            .min(source.fields().len())
            .min(max_columns.saturating_sub(base));
        if width == 0 || base + width < retained {
            return Err("transport trial exceeds its column budget".into());
        }
        let count = base + width;
        let n = theta.grid().len() as u128;
        let element = std::mem::size_of::<Complex<B::Real>>() as u128;
        // Includes still-live predictor vectors/actions, transported outputs,
        // query storage and the constrained target Ritz workspace.
        let bytes = (10 * count as u128 + 8) * n * element + n * 256 + (count as u128).pow(2) * 128;
        report.estimated_peak_payload_bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
        if bytes > max_bytes as u128 {
            return Err("combined transport trial exceeds its payload budget".into());
        }
        let mut vectors = transport(
            theta.backend(),
            theta.grid(),
            &source.fields()[..width],
            &map,
            options,
            max_bytes,
            &mut report,
        )?;
        report.estimated_peak_payload_bytes = bytes as usize;
        if let Some(predicted) = predictor {
            vectors.splice(0..0, predicted.iter().cloned());
        }
        report.trial_columns = vectors.len();
        report.target_preparation_columns = vectors.len();
        predict::rayleigh_ritz_trial(theta, vectors, retained, max_bytes)
    })();
    match attempt {
        Ok(prediction) => {
            report.accepted = true;
            (Some(prediction), report)
        }
        Err(reason) => {
            report.reason = Some(reason);
            (None, report)
        }
    }
}
