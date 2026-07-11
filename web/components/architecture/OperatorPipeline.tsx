'use client';

import { useEffect, useState } from 'react';
import { cssColor, KIND_COLORS } from '../../lib/archboard/palette';

/**
 * Step through one application of the matrix-free Maxwell operator, stage by
 * stage: what runs, in which domain (real space or G-space), and which of the
 * preallocated scratch buffers it reads and writes. The TE pipeline is the
 * six-FFT story from the prose; the buffer strip is the reuse payoff.
 */

const ORANGE = cssColor(KIND_COLORS.compute);
const BLUE = cssColor(KIND_COLORS.movement);
const PANEL = '#0d0d0d';
const BOX = '#141414';
const TEXT = 'rgba(255,255,255,0.85)';
const MUTED = 'rgba(255,255,255,0.45)';
const BORDER = 'rgba(255,255,255,0.16)';

interface Stage {
  name: string;
  domain: string;
  desc: string;
  reads: string[];
  writes: string[];
  ffts: number;
}

const TM_STAGES: Stage[] = [
  {
    name: 'forward FFT',
    domain: 'ℝ → G',
    desc: 'The E_z field enters G-space: one complex FFT over the grid.',
    reads: ['field'],
    writes: ['scratch'],
    ffts: 1,
  },
  {
    name: '× |k+G|²',
    domain: 'G',
    desc: 'The Laplacian is diagonal in G-space: one pointwise multiply with the precomputed |k+G|² table, in place.',
    reads: ['scratch'],
    writes: ['scratch'],
    ffts: 0,
  },
  {
    name: 'inverse FFT',
    domain: 'G → ℝ',
    desc: 'Back to real space. That is the entire TM operator: two FFTs and one multiply.',
    reads: ['scratch'],
    writes: ['field'],
    ffts: 1,
  },
];

const TE_STAGES: Stage[] = [
  {
    name: 'forward FFT',
    domain: 'ℝ → G',
    desc: 'The potential enters G-space: the first of six FFTs.',
    reads: ['field'],
    writes: ['scratch'],
    ffts: 1,
  },
  {
    name: '× i(k+G)',
    domain: 'G',
    desc: 'Both gradient components at once: pointwise multiplies with the k+G tables, one per direction.',
    reads: ['scratch'],
    writes: ['grad_x', 'grad_y'],
    ffts: 0,
  },
  {
    name: '2 inverse FFTs',
    domain: 'G → ℝ',
    desc: 'The gradient lands in real space, one inverse FFT per component.',
    reads: ['grad_x', 'grad_y'],
    writes: ['grad_x', 'grad_y'],
    ffts: 2,
  },
  {
    name: 'ε⁻¹ contraction',
    domain: 'ℝ',
    desc: 'The 2×2 inverse-permittivity tensor acts pointwise in real space. This is where the subpixel-smoothed boundary tensors do their work.',
    reads: ['grad_x', 'grad_y'],
    writes: ['grad_x', 'grad_y'],
    ffts: 0,
  },
  {
    name: '2 forward FFTs',
    domain: 'ℝ → G',
    desc: 'The two flux components go back to G-space.',
    reads: ['grad_x', 'grad_y'],
    writes: ['grad_x', 'grad_y'],
    ffts: 2,
  },
  {
    name: 'divergence',
    domain: 'G',
    desc: 'i(k+G) · flux, summed into a single G-space field.',
    reads: ['grad_x', 'grad_y'],
    writes: ['scratch'],
    ffts: 0,
  },
  {
    name: 'inverse FFT',
    domain: 'G → ℝ',
    desc: 'One last inverse FFT and the application is complete: six FFTs, zero allocations.',
    reads: ['scratch'],
    writes: ['field'],
    ffts: 1,
  },
];

const BUFFERS: Record<'TM' | 'TE', string[]> = {
  TM: ['field', 'scratch'],
  TE: ['field', 'scratch', 'grad_x', 'grad_y'],
};

const btnStyle = (active: boolean): React.CSSProperties => ({
  background: active ? 'rgba(255,255,255,0.12)' : 'transparent',
  color: active ? TEXT : MUTED,
  border: `1px solid ${active ? 'rgba(255,255,255,0.3)' : BORDER}`,
  borderRadius: 6,
  padding: '4px 12px',
  fontSize: 12.5,
  cursor: 'pointer',
  fontFamily: 'inherit',
});

export default function OperatorPipeline() {
  const [pol, setPol] = useState<'TM' | 'TE'>('TE');
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  const stages = pol === 'TM' ? TM_STAGES : TE_STAGES;
  const stage = stages[Math.min(step, stages.length - 1)];
  const totalFfts = stages.reduce((s, st) => s + st.ffts, 0);
  const fftsSoFar = stages.slice(0, step + 1).reduce((s, st) => s + st.ffts, 0);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(false);
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => (s + 1) % stages.length), 2400);
    return () => clearInterval(id);
  }, [playing, stages.length]);

  const switchPol = (p: 'TM' | 'TE') => {
    setPol(p);
    setStep(0);
  };

  return (
    <div
      style={{
        background: PANEL,
        border: `1px solid rgba(128,128,128,0.25)`,
        borderRadius: 12,
        padding: '18px 20px',
        color: TEXT,
        fontSize: 13,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <button style={btnStyle(pol === 'TM')} onClick={() => switchPol('TM')}>
            TM: 2 FFTs
          </button>
          <button style={btnStyle(pol === 'TE')} onClick={() => switchPol('TE')}>
            TE: 6 FFTs
          </button>
        </div>
        <div style={{ color: MUTED, fontSize: 12 }}>
          FFTs so far:{' '}
          <span style={{ color: ORANGE, fontWeight: 600 }}>
            {fftsSoFar} of {totalFfts}
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6, margin: '16px 0 14px' }}>
        {stages.map((s, i) => (
          <div key={s.name + i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              onClick={() => {
                setStep(i);
                setPlaying(false);
              }}
              style={{
                background: i === step ? 'rgba(224,110,28,0.15)' : BOX,
                border: `1.5px solid ${i === step ? ORANGE : i < step ? 'rgba(255,255,255,0.3)' : BORDER}`,
                borderRadius: 8,
                padding: '6px 10px',
                cursor: 'pointer',
                color: i === step ? TEXT : MUTED,
                fontFamily: 'inherit',
                fontSize: 12,
                lineHeight: 1.35,
                textAlign: 'center',
              }}
            >
              <div style={{ fontWeight: i === step ? 600 : 400 }}>{s.name}</div>
              <div style={{ fontSize: 10.5, color: i === step ? ORANGE : 'rgba(255,255,255,0.3)' }}>{s.domain}</div>
            </button>
            {i < stages.length - 1 && <span style={{ color: 'rgba(255,255,255,0.25)' }}>→</span>}
          </div>
        ))}
      </div>

      <div style={{ minHeight: 44, color: TEXT, fontSize: 13, lineHeight: 1.55 }}>{stage.desc}</div>

      <div style={{ marginTop: 14, paddingTop: 12, borderTop: `1px solid ${BORDER}` }}>
        <div style={{ color: MUTED, fontSize: 11, marginBottom: 8 }}>
          Workspace, allocated once when the operator is built and reused on every apply:
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          {BUFFERS[pol].map((buf) => {
            const reads = stage.reads.includes(buf);
            const writes = stage.writes.includes(buf);
            return (
              <div
                key={buf}
                style={{
                  background: BOX,
                  border: `1.5px solid ${writes ? ORANGE : reads ? BLUE : BORDER}`,
                  borderRadius: 8,
                  padding: '5px 12px',
                  fontFamily: 'var(--font-mono, monospace)',
                  fontSize: 12,
                  color: reads || writes ? TEXT : MUTED,
                }}
              >
                {buf}
                {(reads || writes) && (
                  <span style={{ fontSize: 10, marginLeft: 8, color: writes ? ORANGE : BLUE }}>
                    {reads && writes ? 'read · write' : writes ? 'write' : 'read'}
                  </span>
                )}
              </div>
            );
          })}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            <button style={btnStyle(false)} onClick={() => { setStep((s) => (s - 1 + stages.length) % stages.length); setPlaying(false); }} aria-label="Previous stage">
              ‹
            </button>
            <button style={btnStyle(playing)} onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause' : 'Play'}>
              {playing ? '⏸' : '▶'}
            </button>
            <button style={btnStyle(false)} onClick={() => { setStep((s) => (s + 1) % stages.length); setPlaying(false); }} aria-label="Next stage">
              ›
            </button>
            <span style={{ color: MUTED, fontSize: 11.5, minWidth: 34, textAlign: 'right' }}>
              {step + 1} / {stages.length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
