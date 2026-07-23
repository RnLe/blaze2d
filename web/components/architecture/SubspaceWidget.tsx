'use client';

import { useState } from 'react';
import { cssColor, KIND_COLORS, PACKET_PRECISION_COLORS } from '../../lib/archboard/palette';

/**
 * Why LOBPCG is affordable, interactively: the trial basis [X, P, W] is
 * N x 3m, and the only problem ever solved exactly is 3m x 3m. Drawn to true
 * scale against the N x N operator that never materializes, with the real
 * byte counts underneath.
 */

const F32 = cssColor(PACKET_PRECISION_COLORS.f32);
const F64 = cssColor(PACKET_PRECISION_COLORS.f64);
const BLUE = cssColor(KIND_COLORS.movement);
const PANEL = '#0d0d0d';
const TEXT = 'rgba(255,255,255,0.85)';
const MUTED = 'rgba(255,255,255,0.45)';
const BORDER = 'rgba(255,255,255,0.16)';

const N_PRESETS = [
  { label: '16 × 16', n: 256 },
  { label: '32 × 32', n: 1024 },
  { label: '64 × 64', n: 4096 },
  { label: '128 × 128', n: 16384 },
];

function fmtBytes(b: number): string {
  if (b >= 1 << 30) return `${(b / (1 << 30)).toFixed(1)} GB`;
  if (b >= 1 << 20) return `${(b / (1 << 20)).toFixed(1)} MB`;
  if (b >= 1 << 10) return `${(b / (1 << 10)).toFixed(1)} KB`;
  return `${b} B`;
}

const chip = (label: string, value: string, accent?: string) => (
  <div
    key={label}
    style={{
      background: '#141414',
      border: `1px solid ${BORDER}`,
      borderRadius: 8,
      padding: '5px 12px',
      fontSize: 11.5,
      color: MUTED,
    }}
  >
    {label}: <span style={{ color: accent ?? TEXT, fontWeight: 600 }}>{value}</span>
  </div>
);

export default function SubspaceWidget() {
  const [m, setM] = useState(8);
  const [n, setN] = useState(1024);

  const r = 3 * m;
  // Geometry: the ghost N x N square is SIDE px tall; everything else true scale.
  const SIDE = 230;
  const gx = 24;
  const gy = 30;
  const stripW = Math.max(5, m * 2.2);
  const gap = 4;
  const projSide = Math.max(1, (r / n) * SIDE);
  const magSide = 84;
  const magX = 495;
  const magY = gy + 30;
  const projX = 420;
  const projY = gy + SIDE - projSide;

  const strips: Array<[string, number]> = [
    ['X', 0.95],
    ['P', 0.6],
    ['W', 0.35],
  ];

  return (
    <div
      style={{
        background: PANEL,
        border: '1px solid rgba(128,128,128,0.25)',
        borderRadius: 12,
        padding: '18px 20px',
        color: TEXT,
        fontSize: 13,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12.5, color: MUTED }}>
          bands m = <span style={{ color: TEXT, fontWeight: 600, minWidth: 18 }}>{m}</span>
          <input
            type="range"
            min={2}
            max={16}
            value={m}
            onChange={(e) => setM(Number(e.target.value))}
            style={{ width: 140, accentColor: F32 }}
          />
        </label>
        <div style={{ display: 'flex', gap: 6 }}>
          {N_PRESETS.map((p) => (
            <button
              key={p.n}
              onClick={() => setN(p.n)}
              style={{
                background: n === p.n ? 'rgba(255,255,255,0.12)' : 'transparent',
                color: n === p.n ? TEXT : MUTED,
                border: `1px solid ${n === p.n ? 'rgba(255,255,255,0.3)' : BORDER}`,
                borderRadius: 6,
                padding: '3px 10px',
                fontSize: 11.5,
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <svg viewBox="0 0 620 300" style={{ width: '100%', maxWidth: 720, display: 'block', margin: '10px auto 0' }}>
        {/* The N x N operator that never exists. */}
        <rect
          x={gx}
          y={gy}
          width={SIDE}
          height={SIDE}
          fill="none"
          stroke={MUTED}
          strokeWidth={1}
          strokeDasharray="5 5"
          rx={4}
        />
        <text x={gx + SIDE / 2} y={gy + SIDE / 2 - 8} textAnchor="middle" fill={MUTED} fontSize={12}>
          N × N operator
        </text>
        <text x={gx + SIDE / 2} y={gy + SIDE / 2 + 10} textAnchor="middle" fill={MUTED} fontSize={11}>
          never formed ({fmtBytes(n * n * 16)})
        </text>

        {/* The trial basis strips, N tall and m columns wide each, true scale. */}
        {strips.map(([label, alpha], i) => {
          const x = gx + 10 + i * (stripW + gap);
          return (
            <g key={label}>
              <rect x={x} y={gy} width={stripW} height={SIDE} fill={F32} opacity={alpha} rx={2} />
              <text x={x + stripW / 2} y={gy + SIDE + 16} textAnchor="middle" fill={TEXT} fontSize={12}>
                {label}
              </text>
            </g>
          );
        })}
        <text x={gx + 10 + (3 * stripW + 2 * gap) / 2} y={gy + SIDE + 32} textAnchor="middle" fill={MUTED} fontSize={11}>
          [X, P, W] : N × {r}
        </text>

        {/* Projection arrow. */}
        <line x1={gx + SIDE + 14} y1={gy + SIDE / 2} x2={projX - 18} y2={gy + SIDE / 2} stroke={MUTED} strokeWidth={1.5} />
        <polygon
          points={`${projX - 18},${gy + SIDE / 2 - 4} ${projX - 18},${gy + SIDE / 2 + 4} ${projX - 10},${gy + SIDE / 2}`}
          fill={MUTED}
        />
        <text x={(gx + SIDE + projX) / 2 - 2} y={gy + SIDE / 2 - 22} textAnchor="middle" fill={TEXT} fontSize={12}>
          Aₛ = Qᴴ(A Q)
        </text>
        <text x={(gx + SIDE + projX) / 2 - 2} y={gy + SIDE / 2 - 7} textAnchor="middle" fill={MUTED} fontSize={10.5}>
          GEMM · f64
        </text>

        {/* The projected problem at true scale... */}
        <rect x={projX} y={projY} width={Math.max(projSide, 2)} height={Math.max(projSide, 2)} fill={F64} rx={1} />
        <text x={projX + 4} y={gy + SIDE + 16} textAnchor="start" fill={MUTED} fontSize={11}>
          true scale
        </text>

        {/* ...and magnified so it can be read. */}
        <line x1={projX + projSide} y1={projY} x2={magX} y2={magY} stroke={MUTED} strokeWidth={0.75} strokeDasharray="3 3" />
        <line
          x1={projX + projSide}
          y1={projY + projSide}
          x2={magX}
          y2={magY + magSide}
          stroke={MUTED}
          strokeWidth={0.75}
          strokeDasharray="3 3"
        />
        <rect
          x={magX}
          y={magY}
          width={magSide}
          height={magSide}
          fill={F64}
          opacity={0.25}
          stroke={F64}
          strokeWidth={1.5}
          rx={4}
        />
        <text x={magX + magSide / 2} y={magY + magSide / 2 - 4} textAnchor="middle" fill={TEXT} fontSize={12} fontWeight={600}>
          {r} × {r}
        </text>
        <text x={magX + magSide / 2} y={magY + magSide / 2 + 12} textAnchor="middle" fill={MUTED} fontSize={10}>
          dense · f64
        </text>
        <text x={magX + magSide / 2} y={magY + magSide + 16} textAnchor="middle" fill={MUTED} fontSize={10.5}>
          solved exactly
        </text>
      </svg>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
        {chip('N = nx·ny', String(n))}
        {chip('r = 3m', String(r), F64)}
        {chip('trial basis, storage precision', fmtBytes(n * r * 8), F32)}
        {chip('projected matrix, f64', fmtBytes(r * r * 16), F64)}
        {chip('fraction of the full problem', `${((r / n) * (r / n) * 100).toPrecision(2)}%`)}
      </div>
    </div>
  );
}
