//! Dense products of separate columns without copying them: register-tiled SIMD kernels
//! over row blocks, dispatched at runtime through pulp (safe code; used where
//! [`vectorized`] holds, i.e. AVX2/FMA on x86-64). Every column is read straight from its own storage; within a
//! row block the segments of all operands stay in cache across the tiles, so each column is
//! read from memory once per product and each output written once.

use pulp::{Arch, Simd, WithSimd};

/// Rows per block: the block's segments of all columns stay in L2 across the tiles.
const ROW_BLOCK: usize = 256;
/// Gram tile: `MR` columns of x against `NR` columns of y (12 accumulators).
const MR: usize = 4;
const NR: usize = 3;
/// Combination tile: `VR` row vectors against `CR` outputs (12 accumulators).
const VR: usize = 2;
const CR: usize = 6;

/// Whether the kernels run vectorized here (AVX2 and FMA on x86-64). Elsewhere callers keep
/// their packed faer path: pulp's scalar fallback is several times slower than faer's SSE2.
pub fn vectorized() -> bool {
    static VECTORIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VECTORIZED.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            matches!(Arch::new(), Arch::V3(_))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Row-major p × q matrix G[i][j] = Σ_k x_i[k]·y_j[k]. With `lower` only i ≥ j is formed
/// and mirrored, for products known to be symmetric (p = q).
pub fn gram(x: &[&[f64]], y: &[&[f64]], lower: bool) -> Vec<f64> {
    let (p, q) = (x.len(), y.len());
    let mut g = vec![0.0; p * q];
    if p == 0 || q == 0 || x[0].is_empty() {
        return g;
    }
    let lower = lower && p == q;
    Arch::new().dispatch(Gram { x, y, g: &mut g, lower });
    if lower {
        for i in 0..p {
            for j in i + 1..q {
                g[i * q + j] = g[j * q + i];
            }
        }
    }
    g
}

/// out_j = Σ_i q_i·c[i + r·j] (or out_j += … with `accumulate`) for column-major r × m
/// coefficients `c`, r = q.len(), m = out.len().
pub fn combine(q: &[&[f64]], c: &[f64], out: &mut [&mut [f64]], accumulate: bool) {
    let (r, m) = (q.len(), out.len());
    assert_eq!(c.len(), r * m, "coefficients must be r × m");
    if m == 0 || out[0].is_empty() {
        return;
    }
    if r == 0 {
        if !accumulate {
            out.iter_mut().for_each(|o| o.fill(0.0));
        }
        return;
    }
    Arch::new().dispatch(Combine { q, c, out, accumulate });
}

struct Gram<'a> {
    x: &'a [&'a [f64]],
    y: &'a [&'a [f64]],
    g: &'a mut [f64],
    lower: bool,
}

impl WithSimd for Gram<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Gram { x, y, g, lower } = self;
        let (p, q, n) = (x.len(), y.len(), x[0].len());
        let mut start = 0;
        while start < n {
            let rows = ROW_BLOCK.min(n - start);
            for i0 in (0..p).step_by(MR) {
                let mr = MR.min(p - i0);
                for j0 in (0..q).step_by(NR) {
                    // Tiles strictly above the diagonal are mirrored afterwards.
                    if lower && j0 >= i0 + mr {
                        break;
                    }
                    gram_tile(simd, x, y, g, q, (i0, mr), (j0, NR.min(q - j0)), start, rows);
                }
            }
            start += rows;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gram_tile<S: Simd>(simd: S, x: &[&[f64]], y: &[&[f64]], g: &mut [f64], q: usize, (i0, mr): (usize, usize), (j0, nr): (usize, usize), start: usize, rows: usize) {
    let zero = simd.splat_f64s(0.0);
    let mut acc = [[zero; NR]; MR];
    // Short edge tiles repeat their last column; the duplicates are not stored.
    let xs: [&[S::f64s]; MR] = core::array::from_fn(|a| S::as_simd_f64s(&x[i0 + a.min(mr - 1)][start..start + rows]).0);
    let ys: [&[S::f64s]; NR] = core::array::from_fn(|b| S::as_simd_f64s(&y[j0 + b.min(nr - 1)][start..start + rows]).0);
    let vectors = xs[0].len();
    for k in 0..vectors {
        let yv: [S::f64s; NR] = core::array::from_fn(|b| ys[b][k]);
        for a in 0..MR {
            let xv = xs[a][k];
            for b in 0..NR {
                acc[a][b] = simd.mul_add_e_f64s(xv, yv[b], acc[a][b]);
            }
        }
    }
    let done = start + rows - S::as_simd_f64s(&x[i0][start..start + rows]).1.len();
    for a in 0..mr {
        for b in 0..nr {
            let mut sum = simd.reduce_sum_f64s(acc[a][b]);
            for k in done..start + rows {
                sum += x[i0 + a][k] * y[j0 + b][k];
            }
            g[(i0 + a) * q + j0 + b] += sum;
        }
    }
}

struct Combine<'a, 'b> {
    q: &'a [&'a [f64]],
    c: &'a [f64],
    out: &'a mut [&'b mut [f64]],
    accumulate: bool,
}

impl WithSimd for Combine<'_, '_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Combine { q, c, out, accumulate } = self;
        let (r, m, n) = (q.len(), out.len(), q[0].len());
        let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let step = VR * lanes;
        let full = n / step * step;
        let zero = simd.splat_f64s(0.0);
        // Coefficients per output tile, i-major: tiles[t][i][b] = c[i + r·(CR·t + b)].
        let tiles: Vec<Vec<[f64; CR]>> = (0..m.div_ceil(CR))
            .map(|t| (0..r).map(|i| core::array::from_fn(|b| if CR * t + b < m { c[i + r * (CR * t + b)] } else { 0.0 })).collect())
            .collect();
        let block = (ROW_BLOCK / step).max(1) * step;
        let mut start = 0;
        while start < full {
            let end = (start + block).min(full);
            for (t, coefficients) in tiles.iter().enumerate() {
                let (j0, nr) = (CR * t, CR.min(m - CR * t));
                let mut row = start;
                while row < end {
                    let mut acc = [[zero; CR]; VR];
                    if accumulate {
                        for b in 0..nr {
                            let (current, _) = S::as_simd_f64s(&out[j0 + b][row..row + step]);
                            for v in 0..VR {
                                acc[v][b] = current[v];
                            }
                        }
                    }
                    for (i, cb) in coefficients.iter().enumerate() {
                        let (qv, _) = S::as_simd_f64s(&q[i][row..row + step]);
                        for b in 0..CR {
                            let s = simd.splat_f64s(cb[b]);
                            for v in 0..VR {
                                acc[v][b] = simd.mul_add_e_f64s(qv[v], s, acc[v][b]);
                            }
                        }
                    }
                    for b in 0..nr {
                        let (dst, _) = S::as_mut_simd_f64s(&mut out[j0 + b][row..row + step]);
                        for v in 0..VR {
                            dst[v] = acc[v][b];
                        }
                    }
                    row += step;
                }
            }
            start = end;
        }
        for k in full..n {
            for (j, o) in out.iter_mut().enumerate() {
                let value: f64 = (0..r).map(|i| q[i][k] * c[i + r * j]).sum();
                o[k] = if accumulate { o[k] + value } else { value };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn columns(count: usize, n: usize, seed: usize) -> Vec<Vec<f64>> {
        (0..count).map(|j| (0..n).map(|k| (((k * 7919 + j * 104729 + seed) % 2003) as f64 / 1001.0) - 1.0).collect()).collect()
    }

    #[test]
    fn gram_matches_dots_including_edges_and_tails() {
        for &(p, q, n) in &[(1, 1, 1), (5, 7, 13), (42, 42, 1031), (9, 4, 300)] {
            let (xs, ys) = (columns(p, n, 1), columns(q, n, 2));
            let (x, y): (Vec<&[f64]>, Vec<&[f64]>) = (xs.iter().map(|v| v.as_slice()).collect(), ys.iter().map(|v| v.as_slice()).collect());
            let g = gram(&x, &y, false);
            for i in 0..p {
                for j in 0..q {
                    let exact: f64 = (0..n).map(|k| x[i][k] * y[j][k]).sum();
                    assert!((g[i * q + j] - exact).abs() <= 1e-12 * (1.0 + exact.abs()), "{p}x{q}x{n} ({i},{j})");
                }
            }
            if p == q {
                let s = gram(&x, &x, true);
                for i in 0..p {
                    for j in 0..p {
                        let exact: f64 = (0..n).map(|k| x[i][k] * x[j][k]).sum();
                        assert!((s[i * p + j] - exact).abs() <= 1e-12 * (1.0 + exact.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn combine_matches_sums_and_accumulates() {
        for &(r, m, n) in &[(1, 1, 1), (3, 7, 17), (42, 28, 1033), (5, 13, 64)] {
            let qs = columns(r, n, 3);
            let q: Vec<&[f64]> = qs.iter().map(|v| v.as_slice()).collect();
            let c: Vec<f64> = (0..r * m).map(|k| ((k * 31) % 17) as f64 / 8.0 - 1.0).collect();
            let mut outs: Vec<Vec<f64>> = (0..m).map(|j| vec![j as f64; n]).collect();
            {
                let mut views: Vec<&mut [f64]> = outs.iter_mut().map(|v| v.as_mut_slice()).collect();
                combine(&q, &c, &mut views, false);
            }
            for j in 0..m {
                for k in 0..n {
                    let exact: f64 = (0..r).map(|i| q[i][k] * c[i + r * j]).sum();
                    assert!((outs[j][k] - exact).abs() <= 1e-12 * (1.0 + exact.abs()));
                }
            }
            let before = outs.clone();
            {
                let mut views: Vec<&mut [f64]> = outs.iter_mut().map(|v| v.as_mut_slice()).collect();
                combine(&q, &c, &mut views, true);
            }
            for j in 0..m {
                for k in 0..n {
                    let exact: f64 = before[j][k] + (0..r).map(|i| q[i][k] * c[i + r * j]).sum::<f64>();
                    assert!((outs[j][k] - exact).abs() <= 1e-12 * (1.0 + exact.abs()));
                }
            }
        }
    }
}
