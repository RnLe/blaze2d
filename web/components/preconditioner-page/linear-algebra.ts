/**
 * Small, dependency-free linear algebra for the preconditioner page: 2×2
 * symmetric matrices for the vector playground and a one-dimensional Maxwell
 * model with a single-vector LOBPCG for the convergence comparison.
 */

export type Vec2 = [number, number];
/** Row-major 2×2 matrix [[a, b], [c, d]]. */
export type Mat2 = [[number, number], [number, number]];

export const dot = (a: Vec2, b: Vec2) => a[0] * b[0] + a[1] * b[1];
export const norm = (a: Vec2) => Math.hypot(a[0], a[1]);
export const add = (a: Vec2, b: Vec2): Vec2 => [a[0] + b[0], a[1] + b[1]];
export const sub = (a: Vec2, b: Vec2): Vec2 => [a[0] - b[0], a[1] - b[1]];
export const scale = (a: Vec2, s: number): Vec2 => [a[0] * s, a[1] * s];
export const apply = (m: Mat2, x: Vec2): Vec2 => [m[0][0] * x[0] + m[0][1] * x[1], m[1][0] * x[0] + m[1][1] * x[1]];
export const multiply = (p: Mat2, q: Mat2): Mat2 => [
  [p[0][0] * q[0][0] + p[0][1] * q[1][0], p[0][0] * q[0][1] + p[0][1] * q[1][1]],
  [p[1][0] * q[0][0] + p[1][1] * q[1][0], p[1][0] * q[0][1] + p[1][1] * q[1][1]],
];

/** Inverse of a 2×2 matrix, or null when it is (numerically) singular. */
export function inverse(m: Mat2): Mat2 | null {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  const size = Math.max(...m.flat().map(Math.abs));
  if (!(size > 0) || !(Math.abs(det) > 1e-12 * size * size)) return null;
  return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
}

/** Angle between two directions in degrees, ignoring orientation (0–90°). */
export function lineAngle(a: Vec2, b: Vec2) {
  const c = Math.abs(dot(a, b)) / Math.max(norm(a) * norm(b), 1e-300);
  return Math.acos(Math.min(1, c)) * 180 / Math.PI;
}

/** Signed angle from a to b in degrees (−180–180°). */
export function signedAngle(a: Vec2, b: Vec2) {
  return Math.atan2(a[0] * b[1] - a[1] * b[0], dot(a, b)) * 180 / Math.PI;
}

/**
 * Symmetric matrix with eigenvalues λ₁ ≤ λ₂ whose first eigenvector points at
 * `theta` radians: A = R diag(λ₁, λ₂) Rᵀ.
 */
export function symmetric(lambda1: number, lambda2: number, theta: number) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const v1: Vec2 = [c, s], v2: Vec2 = [-s, c];
  const matrix: Mat2 = [
    [lambda1 * c * c + lambda2 * s * s, (lambda1 - lambda2) * c * s],
    [(lambda1 - lambda2) * c * s, lambda1 * s * s + lambda2 * c * c],
  ];
  return { matrix, v1, v2, lambda1, lambda2 };
}

/** Eigenvalues (ascending) of a 2×2 matrix with real spectrum, e.g. T·A for symmetric positive T. */
export function eigenvalues2(m: Mat2): [number, number] {
  const tr = m[0][0] + m[1][1], det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  return [tr / 2 - disc, tr / 2 + disc];
}

/** Rayleigh quotient ρ(x) = xᵀAx / xᵀx. */
export const rayleigh = (a: Mat2, x: Vec2) => dot(x, apply(a, x)) / Math.max(dot(x, x), 1e-300);

// ---------------------------------------------------------------------------
// One-dimensional Maxwell model for the convergence comparison
// ---------------------------------------------------------------------------

/**
 * TE-like operator A u = −d/dx (ε⁻¹ du/dx) on a unit cell with N grid points,
 * at the edge of the Brillouin zone (Bloch phase −1, so the matrix is real,
 * symmetric and positive definite). A layer of permittivity `contrast` fills
 * the first 5/16 of the cell (exactly, for every N divisible by 16); the rest is air.
 */
export const LAYER_FRACTION = 5 / 16;

export function maxwellModel(n: number, contrast: number) {
  const h = 1 / n;
  // Inverse permittivity on the half points between grid nodes.
  const inv = Float64Array.from({ length: n }, (_, i) => ((i + 0.5) * h < LAYER_FRACTION ? 1 / contrast : 1));
  const meanInv = inv.reduce((s, v) => s + v, 0) / n;
  const c = 1 / (h * h);
  /** y = A x; the wrap-around couplings carry the Bloch sign −1. */
  function applyA(x: Float64Array, y: Float64Array) {
    for (let i = 0; i < n; i++) {
      const left = i === 0 ? -x[n - 1] : x[i - 1];
      const right = i === n - 1 ? -x[0] : x[i + 1];
      const wl = inv[(i - 1 + n) % n], wr = inv[i];
      y[i] = c * (wl * (x[i] - left) + wr * (x[i] - right));
    }
  }
  const diagonal = Float64Array.from({ length: n }, (_, i) => c * (inv[(i - 1 + n) % n] + inv[i]));
  return { n, inv, meanInv, applyA, diagonal };
}

export type Model = ReturnType<typeof maxwellModel>;

/** In-place radix-2 FFT of (re, im); `sign` −1 forward, +1 inverse (unnormalized). */
function fft(re: Float64Array, im: Float64Array, sign: number) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = sign * 2 * Math.PI / len, wr = Math.cos(angle), wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let ur = 1, ui = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * ur - im[b] * ui, ti = re[b] * ui + im[b] * ur;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        [ur, ui] = [ur * wr - ui * wi, ur * wi + ui * wr];
      }
    }
  }
}

export type PreconditionerKind = 'none' | 'jacobi' | 'fourier' | 'exact';

/**
 * A preconditioner y = T r for the model. 'fourier' is the Blaze-style
 * Fourier-diagonal approximation 1 / (⟨ε⁻¹⟩ |q|²) of the inverse; 'exact'
 * factorizes A (Cholesky) once.
 */
export function preconditioner(model: Model, kind: PreconditionerKind): (r: Float64Array, y: Float64Array) => void {
  const { n } = model;
  if (kind === 'none') return (r, y) => y.set(r);
  if (kind === 'jacobi') return (r, y) => { for (let i = 0; i < n; i++) y[i] = r[i] / model.diagonal[i]; };
  if (kind === 'fourier') {
    // Anti-periodic modes e^{iπ(2m+1)x}: twist by e^{−iπ j/n}, FFT, divide by the
    // finite-difference symbol of the homogeneous operator, transform back.
    const twistRe = Float64Array.from({ length: n }, (_, j) => Math.cos(Math.PI * j / n));
    const twistIm = Float64Array.from({ length: n }, (_, j) => -Math.sin(Math.PI * j / n));
    const symbol = Float64Array.from({ length: n }, (_, m) => model.meanInv * 4 * n * n * Math.sin(Math.PI * (2 * m + 1) / (2 * n)) ** 2);
    const re = new Float64Array(n), im = new Float64Array(n);
    return (r, y) => {
      for (let j = 0; j < n; j++) { re[j] = r[j] * twistRe[j]; im[j] = r[j] * twistIm[j]; }
      fft(re, im, -1);
      for (let m = 0; m < n; m++) { re[m] /= symbol[m]; im[m] /= symbol[m]; }
      fft(re, im, 1);
      // Undo the twist (multiply by e^{+iπ j/n}) and the FFT normalization.
      for (let j = 0; j < n; j++) y[j] = (re[j] * twistRe[j] + im[j] * twistIm[j]) / n;
    };
  }
  // Dense Cholesky factor L of A (lower triangle, row-major).
  const a = new Float64Array(n * n), column = new Float64Array(n), out = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    column.fill(0); column[j] = 1;
    model.applyA(column, out);
    for (let i = 0; i < n; i++) a[i * n + j] = out[i];
  }
  for (let j = 0; j < n; j++) {
    let d = a[j * n + j];
    for (let k = 0; k < j; k++) d -= a[j * n + k] ** 2;
    const ljj = Math.sqrt(d);
    a[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = a[i * n + j];
      for (let k = 0; k < j; k++) s -= a[i * n + k] * a[j * n + k];
      a[i * n + j] = s / ljj;
    }
  }
  return (r, y) => {
    for (let i = 0; i < n; i++) {
      let s = r[i];
      for (let k = 0; k < i; k++) s -= a[i * n + k] * y[k];
      y[i] = s / a[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= a[k * n + i] * y[k];
      y[i] = s / a[i * n + i];
    }
  };
}

/** Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations. */
function symmetricEigen(h: number[][]) {
  const k = h.length, a = h.map(row => row.slice());
  const v: number[][] = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) off += a[p][q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let r = 0; r < k; r++) {
        const arp = a[r][p], arq = a[r][q];
        a[r][p] = c * arp - s * arq; a[r][q] = s * arp + c * arq;
      }
      for (let r = 0; r < k; r++) {
        const apr = a[p][r], aqr = a[q][r];
        a[p][r] = c * apr - s * aqr; a[q][r] = s * apr + c * aqr;
      }
      for (let r = 0; r < k; r++) {
        const vrp = v[r][p], vrq = v[r][q];
        v[r][p] = c * vrp - s * vrq; v[r][q] = s * vrp + c * vrq;
      }
    }
  }
  const values = a.map((row, i) => row[i]);
  const order = values.map((_, i) => i).sort((i, j) => values[i] - values[j]);
  return { values: order.map(i => values[i]), vectors: order.map(i => v.map(row => row[i])) };
}

const vdot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

export type Run = {
  kind: PreconditionerKind;
  /** Relative residual ‖Ax − ρx‖ / (‖Ax‖ + |ρ|‖x‖) before each iteration. */
  history: number[];
  converged: boolean;
  eigenvalue: number;
};

/**
 * Single-vector LOBPCG for the lowest eigenpair: Rayleigh–Ritz on
 * span{x, T r, p} each iteration, stopping at a relative residual of `tol`.
 */
export function lobpcg(model: Model, kind: PreconditionerKind, tol = 1e-7, maxIterations = 600): Run {
  const { n } = model;
  const precondition = preconditioner(model, kind);
  // The same start field for every preconditioner and grid: smooth modes plus
  // fine-scale content, sampled at the grid points.
  let x: Float64Array = Float64Array.from({ length: n }, (_, i) => {
    const t = (i + 0.5) / n;
    return Math.cos(Math.PI * t + 0.4) + 0.5 * Math.sin(5 * Math.PI * t) + 0.3 * Math.sin(97.3 * t) + 0.2 * Math.cos(611.7 * t * t);
  });
  let p: Float64Array | null = null;
  const ax = new Float64Array(n), r = new Float64Array(n), w = new Float64Array(n);
  const history: number[] = [];
  let rho = 0;
  const normalize = (v: Float64Array): Float64Array => { const s = Math.sqrt(vdot(v, v)); for (let i = 0; i < n; i++) v[i] /= s; return v; };
  normalize(x);
  for (let iteration = 0; iteration <= maxIterations; iteration++) {
    model.applyA(x, ax);
    rho = vdot(x, ax);
    for (let i = 0; i < n; i++) r[i] = ax[i] - rho * x[i];
    const eta = Math.sqrt(vdot(r, r)) / (Math.sqrt(vdot(ax, ax)) + Math.abs(rho));
    history.push(eta);
    if (eta <= tol) return { kind, history, converged: true, eigenvalue: rho };
    if (iteration === maxIterations) break;
    precondition(r, w);
    // Orthonormal basis of span{x, w, p} (Gram–Schmidt twice, dropping dependent vectors).
    const basis: Float64Array[] = [Float64Array.from(x)];
    for (const candidate of [w, p]) {
      if (!candidate) continue;
      const v = Float64Array.from(candidate);
      const before = Math.sqrt(vdot(v, v));
      for (let pass = 0; pass < 2; pass++) for (const b of basis) { const c = vdot(v, b); for (let i = 0; i < n; i++) v[i] -= c * b[i]; }
      const after = Math.sqrt(vdot(v, v));
      if (after > 1e-10 * before) basis.push(normalize(v));
    }
    const images = basis.map(b => { const y = new Float64Array(n); model.applyA(b, y); return y; });
    const h = basis.map((_, i) => basis.map((_, j) => 0.5 * (vdot(basis[i], images[j]) + vdot(basis[j], images[i]))));
    const { vectors } = symmetricEigen(h);
    const y = vectors[0];
    const next = new Float64Array(n), direction = new Float64Array(n);
    basis.forEach((b, k) => { for (let i = 0; i < n; i++) { next[i] += y[k] * b[i]; if (k > 0) direction[i] += y[k] * b[i]; } });
    x = normalize(next);
    p = basis.length > 1 ? direction : null;
  }
  return { kind, history, converged: false, eigenvalue: rho };
}

/** Rough floating-point operations for one application, used for cost estimates. */
export function costs(n: number) {
  const log = Math.log2(n);
  return {
    operator: 6 * n,
    rayleighRitz: 40 * n,
    apply: { none: 0, jacobi: n, fourier: 10 * n * log + 6 * n, exact: 2 * n * n } as Record<PreconditionerKind, number>,
    setup: { none: 0, jacobi: n, fourier: 4 * n, exact: n * n * n / 3 } as Record<PreconditionerKind, number>,
  };
}
