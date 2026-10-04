/**
 * The factors that multiply each eigen-component of the error: λᵢ − ρ in the
 * raw residual, 1 − ρ/λᵢ after applying T = A⁻¹. The spectrum is the
 * distinct zone-edge plane-wave eigenvalues λᵢ ∝ (2i − 1)² of a uniform
 * one-dimensional crystal, with ρ 8% above λ₁.
 */
const COUNT = 10;
const lambdas = Array.from({ length: COUNT }, (_, i) => (2 * i + 1) ** 2);
const rho = 1.08 * lambdas[0];
const raw = lambdas.map(l => l - rho);
const preconditioned = lambdas.map(l => 1 - rho / l);

const W = 520, H = 250, M = { left: 46, right: 14, top: 30, bottom: 40 };

function Bars({ values, min, max, color, title, axis, ticks }: { values: number[]; min: number; max: number; color: string; title: string; axis: string; ticks: number[] }) {
  const band = (W - M.left - M.right) / values.length;
  const py = (v: number) => M.top + ((max - v) / (max - min)) * (H - M.top - M.bottom);
  const zero = py(0);
  return <figure className="pc-weights-panel">
    <figcaption>{title}</figcaption>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${values.map((v, i) => `component ${i + 1}: ${v.toFixed(2)}`).join(', ')}`}>
      {ticks.map(t => <g key={t} className="pc-chart-grid">
        <line x1={M.left} x2={W - M.right} y1={py(t)} y2={py(t)} />
        <text x={M.left - 8} y={py(t) + 4} textAnchor="end">{t}</text>
      </g>)}
      {values.map((v, i) => {
        const x = M.left + i * band + band * 0.18, width = band * 0.64;
        const top = py(Math.max(v, 0)), bottom = py(Math.min(v, 0));
        return <g key={i}>
          <rect x={x} y={top} width={width} height={Math.max(bottom - top, 1)} rx="3" fill={color}><title>{`v${i + 1}: ${v.toFixed(3)}`}</title></rect>
          <text x={x + width / 2} y={H - M.bottom + 18} textAnchor="middle" className="pc-chart-tick">{i + 1}</text>
        </g>;
      })}
      <line x1={M.left} x2={W - M.right} y1={zero} y2={zero} className="pc-chart-baseline" />
      <text x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" className="pc-chart-axis-title">{axis}</text>
    </svg>
  </figure>;
}

export default function ResidualWeights() {
  return <div className="pc-weights">
    <Bars values={raw} min={-40} max={400} color="var(--pc-residual)" title="Raw residual: weights λᵢ − ρ" axis="eigen-direction i, λᵢ ∝ (2i − 1)²" ticks={[0, 100, 200, 300, 400]} />
    <Bars values={preconditioned} min={-0.1} max={1} color="var(--pc-preconditioned)" title="Preconditioned, T = A⁻¹: weights 1 − ρ/λᵢ" axis="eigen-direction i" ticks={[0, 0.25, 0.5, 0.75, 1]} />
  </div>;
}
