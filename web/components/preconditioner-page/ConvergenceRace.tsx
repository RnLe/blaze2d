'use client';

import { useId, useState, type PointerEvent as ReactPointerEvent } from 'react';
import data from './convergence-data.json';
import { costs, type PreconditionerKind } from './linear-algebra';

type Point = [number, number];
type RunData = { iterations: number | null; eigenvalue: number; points: Point[] };
const runs = data.runs as unknown as Record<string, Record<string, Record<PreconditionerKind, RunData>>>;

const METHODS: { kind: PreconditionerKind; label: string; short: string; color: string; dashed?: boolean }[] = [
  { kind: 'none', label: 'No preconditioner (raw residual)', short: 'None', color: 'var(--pc-residual)' },
  { kind: 'jacobi', label: 'Diagonal of A', short: 'Diagonal', color: 'var(--text-subtle)', dashed: true },
  { kind: 'exact', label: 'Exact inverse A⁻¹ (dense factorization)', short: 'Exact', color: 'var(--text-heading)' },
  { kind: 'fourier', label: 'Fourier diagonal (uniform-medium inverse)', short: 'Fourier', color: 'var(--pc-preconditioned)' },
];

const W = 560, H = 340, M = { left: 54, right: 92, top: 18, bottom: 42 };
const log10 = Math.log10;
/** Chart coordinates rounded to 0.01 px, so server and browser render identical markup. */
const round = (value: number) => Math.round(value * 100) / 100;
const superscript = (exponent: number) => String(exponent).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)]);
const formatCount = (value: number) => value.toLocaleString('en-US');

function workMflop(kind: PreconditionerKind, n: number, iterations: number | null) {
  if (iterations === null) return null;
  const c = costs(n);
  return (iterations * (c.operator + c.rayleighRitz + c.apply[kind]) + c.setup[kind]) / 1e6;
}

function ResidualChart({ n, contrast }: { n: number; contrast: number }) {
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const [hover, setHover] = useState<number | null>(null);
  const series = METHODS.map(method => ({ ...method, run: runs[contrast][n][method.kind] }));
  const xMax = 1e4, yMin = 1e-9, yMax = 1;
  const px = (iteration: number) => round(M.left + (log10(iteration + 1) / log10(xMax)) * (W - M.left - M.right));
  const py = (eta: number) => round(M.top + (log10(yMax / Math.max(eta, yMin)) / log10(yMax / yMin)) * (H - M.top - M.bottom));
  const path = (points: Point[]) => points.map(([i, eta], k) => `${k ? 'L' : 'M'}${px(i).toFixed(1)},${py(eta).toFixed(1)}`).join('');
  /** Residual at `iteration`, interpolated in (log(i + 1), log η) like the drawn path; null once converged. */
  const valueAt = (points: Point[], iteration: number) => {
    if (iteration > points[points.length - 1][0]) return null;
    let k = 1;
    while (k < points.length - 1 && points[k][0] < iteration) k++;
    const [i0, e0] = points[k - 1], [i1, e1] = points[Math.min(k, points.length - 1)];
    if (i1 === i0) return e0;
    const f = (log10(iteration + 1) - log10(i0 + 1)) / (log10(i1 + 1) - log10(i0 + 1));
    return 10 ** (log10(e0) + Math.min(1, Math.max(0, f)) * (log10(e1) - log10(e0)));
  };

  function onMove(event: ReactPointerEvent<SVGSVGElement>) {
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return;
    const { x } = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    const fraction = (x - M.left) / (W - M.left - M.right);
    setHover(fraction < 0 || fraction > 1 ? null : Math.max(0, Math.round(10 ** (fraction * log10(xMax)) - 1)));
  }

  // Direct labels at each converged end point, nudged apart vertically.
  const ends = series.map(s => {
    const last = s.run.points[s.run.points.length - 1];
    return { ...s, x: px(last[0]), y: py(last[1]) };
  }).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    for (let j = 0; j < i; j++) if (Math.abs(ends[i].x - ends[j].x) < 110) ends[i].y = Math.max(ends[i].y, ends[j].y + 15);
  }

  return <figure className="pc-chart">
    <figcaption><strong>Residual per iteration</strong><span>N = {n} grid points, ε = {contrast} · log scales</span></figcaption>
    <div className="pc-chart-frame">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        aria-label={`Relative residual versus iteration for N = ${n} and ε = ${contrast}. ${series.map(s => `${s.label}: ${s.run.iterations === null ? `not converged within ${formatCount(data.maxIterations)} iterations` : `${s.run.iterations} iterations`}`).join('; ')}.`}>
        <defs><clipPath id={`${id}-clip`}><rect x={M.left} y={M.top} width={W - M.left - M.right} height={H - M.top - M.bottom} /></clipPath></defs>
        {[0, -1, -2, -3, -4, -5, -6, -7, -8, -9].map(e => <g key={e} className="pc-chart-grid">
          <line x1={M.left} x2={W - M.right} y1={py(10 ** e)} y2={py(10 ** e)} />
          {e % 2 === 0 && <text x={M.left - 8} y={py(10 ** e) + 4} textAnchor="end">10{superscript(e)}</text>}
        </g>)}
        {[1, 10, 100, 1000, 10000].map(v => <g key={v} className="pc-chart-grid">
          <line x1={px(v - 1)} x2={px(v - 1)} y1={M.top} y2={H - M.bottom} />
          <text x={px(v - 1)} y={H - M.bottom + 18} textAnchor="middle">{formatCount(v)}</text>
        </g>)}
        <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="pc-chart-axis-title">iteration + 1</text>
        <line x1={M.left} x2={W - M.right} y1={py(data.tolerance)} y2={py(data.tolerance)} className="pc-chart-tolerance" />
        <text x={M.left + 6} y={py(data.tolerance) - 6} className="pc-chart-note">tolerance 10⁻⁷</text>
        <g clipPath={`url(#${id}-clip)`}>
          {series.map(s => <path key={s.kind} d={path(s.run.points)} fill="none" stroke={s.color} strokeWidth={s.kind === 'fourier' ? 2.6 : 2}
            strokeDasharray={s.dashed ? '5 5' : undefined} strokeLinejoin="round" />)}
        </g>
        {ends.map(e => <circle key={e.kind} cx={e.x} cy={py(e.run.points[e.run.points.length - 1][1])} r="4" fill={e.color} stroke="var(--surface-sunken)" strokeWidth="2" />)}
        {ends.map(e => <text key={e.kind} x={Math.min(e.x + 9, W - M.right + 6)} y={e.y + 4} className="pc-chart-label">{e.short} · {e.run.iterations === null ? `>${formatCount(data.maxIterations)}` : formatCount(e.run.iterations)}</text>)}
        {hover !== null && <g className="pc-chart-crosshair">
          <line x1={px(hover)} x2={px(hover)} y1={M.top} y2={H - M.bottom} />
          {series.map(s => {
            const value = valueAt(s.run.points, hover);
            return value === null ? null : <circle key={s.kind} cx={px(hover)} cy={py(value)} r="4" fill={s.color} stroke="var(--surface-sunken)" strokeWidth="2" />;
          })}
        </g>}
      </svg>
      {hover !== null && <div className="pc-tooltip" style={{ left: `${(px(hover) / W) * 100}%`, transform: `translateX(${px(hover) < W * 0.3 ? '0' : px(hover) > W * 0.7 ? '-100%' : '-50%'})` }}>
        <strong>Iteration {formatCount(hover)}</strong>
        {series.map(s => {
          const value = valueAt(s.run.points, hover);
          return <span key={s.kind}><i style={{ background: s.color }} />{s.short}<b>{value === null ? 'converged' : value.toExponential(1)}</b></span>;
        })}
      </div>}
    </div>
  </figure>;
}

function ScalingChart({ n, contrast }: { n: number; contrast: number }) {
  const grids = data.grids;
  const yMin = 1, yMax = 1e4;
  const px = (grid: number) => round(M.left + (Math.log2(grid / grids[0]) / Math.log2(grids[grids.length - 1] / grids[0])) * (W - M.left - M.right));
  const py = (count: number) => round(M.top + (log10(yMax / count) / log10(yMax / yMin)) * (H - M.top - M.bottom));
  const lines = METHODS.map(method => ({ ...method, points: grids.map(grid => ({ grid, count: runs[contrast][grid][method.kind].iterations })) }));
  const ends = lines.map(line => {
    const last = line.points[line.points.length - 1];
    return { ...line, y: py(last.count ?? data.maxIterations) };
  }).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);

  return <figure className="pc-chart">
    <figcaption><strong>Iterations to reach the tolerance</strong><span>ε = {contrast} · doubling the grid each step</span></figcaption>
    <div className="pc-chart-frame">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Iterations versus grid size for ε = ${contrast}. ${lines.map(l => `${l.label}: ${l.points.map(p => `${p.grid}: ${p.count ?? 'not converged'}`).join(', ')}`).join('; ')}.`}>
        {[1, 10, 100, 1000, 10000].map(v => <g key={v} className="pc-chart-grid">
          <line x1={M.left} x2={W - M.right} y1={py(v)} y2={py(v)} />
          <text x={M.left - 8} y={py(v) + 4} textAnchor="end">{v >= 1000 ? `${v / 1000}k` : v}</text>
        </g>)}
        {grids.map(grid => <g key={grid} className="pc-chart-grid">
          <line x1={px(grid)} x2={px(grid)} y1={M.top} y2={H - M.bottom} className={grid === n ? 'pc-chart-selected' : undefined} />
          <text x={px(grid)} y={H - M.bottom + 18} textAnchor="middle">{grid}</text>
        </g>)}
        <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="pc-chart-axis-title">grid points N</text>
        {lines.map(line => <g key={line.kind}>
          <path d={line.points.filter(p => p.count !== null).map((p, k) => `${k ? 'L' : 'M'}${px(p.grid).toFixed(1)},${py(p.count!).toFixed(1)}`).join('')}
            fill="none" stroke={line.color} strokeWidth={line.kind === 'fourier' ? 2.6 : 2} strokeDasharray={line.dashed ? '5 5' : undefined} />
          {line.points.map(p => p.count === null ? null : <circle key={p.grid} cx={px(p.grid)} cy={py(p.count)} r={p.grid === n ? 5.5 : 4} fill={line.color} stroke="var(--surface-sunken)" strokeWidth="2">
            <title>{`${line.label}, N = ${p.grid}: ${formatCount(p.count)} iterations`}</title>
          </circle>)}
        </g>)}
        {ends.map(e => <text key={e.kind} x={W - M.right + 10} y={e.y + 4} className="pc-chart-label">{e.short}</text>)}
      </svg>
    </div>
  </figure>;
}

export default function ConvergenceRace() {
  const [n, setN] = useState(128);
  const [contrast, setContrast] = useState(13);
  const id = useId();
  const fourierWork = workMflop('fourier', n, runs[contrast][n].fourier.iterations) ?? 1;
  return <section className="pc-race" aria-label="Convergence comparison">
    <div className="pc-race-controls">
      <fieldset className="pc-segmented"><legend>Grid points N</legend>
        {data.grids.map(grid => <label key={grid}><input type="radio" name={`${id}-n`} checked={n === grid} onChange={() => setN(grid)} /><span>{grid}</span></label>)}
      </fieldset>
      <fieldset className="pc-segmented"><legend>Layer permittivity ε</legend>
        {data.contrasts.map(value => <label key={value}><input type="radio" name={`${id}-eps`} checked={contrast === value} onChange={() => setContrast(value)} /><span>{value === 1 ? '1 · uniform' : value}</span></label>)}
      </fieldset>
      <ul className="pc-legend" aria-label="Legend">
        {METHODS.map(method => <li key={method.kind}><i data-dashed={method.dashed || undefined} style={{ ['--swatch' as string]: method.color }} />{method.label}</li>)}
      </ul>
    </div>
    <div className="pc-race-charts">
      <ResidualChart n={n} contrast={contrast} />
      <ScalingChart n={n} contrast={contrast} />
    </div>
    <p className="pc-race-caption">N = {n}, ε = {contrast}: iterations and estimated arithmetic work. The exact inverse is priced as a dense factorization, as for Blaze’s plane-wave TE operator; this 1D finite-difference matrix alone could be solved in O(N).</p>
    <div className="pc-table-scroll">
    <table className="pc-race-table">
      <thead><tr><th scope="col">Preconditioner</th><th scope="col">Iterations</th><th scope="col" data-wide>Work per T·r</th><th scope="col" data-wide>Total work</th><th scope="col">vs. Fourier</th></tr></thead>
      <tbody>{METHODS.map(method => {
        const run = runs[contrast][n][method.kind];
        const work = workMflop(method.kind, n, run.iterations);
        const perApply = costs(n).apply[method.kind];
        return <tr key={method.kind} data-highlight={method.kind === 'fourier' || undefined}>
          <th scope="row"><i style={{ background: method.color }} />{method.label}</th>
          <td>{run.iterations === null ? `>${formatCount(data.maxIterations)}` : formatCount(run.iterations)}</td>
          <td data-wide>{perApply === 0 ? '—' : perApply < 1e4 ? `${formatCount(Math.round(perApply))} flop` : `${(perApply / 1e6).toPrecision(2)} Mflop`}</td>
          <td data-wide>{work === null ? '—' : `${work < 10 ? work.toFixed(2) : Math.round(work).toLocaleString('en-US')} Mflop`}</td>
          <td>{work === null ? '—' : `${(work / fourierWork).toFixed(work / fourierWork < 10 ? 1 : 0)}×`}</td>
        </tr>;
      })}</tbody>
    </table>
    </div>
  </section>;
}
