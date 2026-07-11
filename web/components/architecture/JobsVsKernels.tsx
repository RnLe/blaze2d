'use client';

import { useEffect, useRef } from 'react';
import { cssColor, KIND_COLORS } from '../../lib/archboard/palette';

/**
 * The Par::Seq punchline as a race: the same four cores, scheduled two ways.
 * Top machine runs whole solve jobs data-parallel, back to back, no shared
 * state. Bottom machine parallelizes inside one job's GEMMs: every micro-phase
 * ends at a sync barrier where the fast threads idle on the slowest, and the
 * barriers eat the speedup (measured 5 to 10x slower at production grids).
 * A time cursor sweeps both schedules; the schedules are deterministic.
 */

const ORANGE = cssColor(KIND_COLORS.compute);
const BLUE = cssColor(KIND_COLORS.movement);
const TEXT = 'rgba(255,255,255,0.85)';
const MUTED = 'rgba(255,255,255,0.45)';
const PANEL = '#0d0d0d';
const BORDER = 'rgba(255,255,255,0.16)';

const SWEEP = 11000; // ms of schedule shown
const HOLD = 2200;
const CYCLE = SWEEP + HOLD;
const SYNC = 0.25; // s, barrier cost per micro-phase

/** Job wall-times per worker lane (s). Whole solves, run back to back. */
const JOB_LANES: number[][] = [
  [1.15, 0.85, 1.3, 0.95, 1.2, 1.05, 0.9, 1.25, 1.1, 0.8, 1.05],
  [0.9, 1.2, 1.0, 1.35, 0.8, 1.15, 1.25, 0.95, 1.05, 1.1, 0.85],
  [1.3, 1.0, 0.85, 1.1, 1.25, 0.9, 1.2, 1.0, 0.95, 1.15, 1.05],
  [1.0, 1.1, 1.2, 0.9, 1.05, 1.3, 0.85, 1.15, 1.2, 0.95, 1.1],
];

/** Per-thread work chunks (s) of each GEMM micro-phase; the phase ends when the slowest does. */
const PHASE_CHUNKS: number[][] = [
  [0.42, 0.63, 0.35, 0.5],
  [0.55, 0.4, 0.62, 0.44],
  [0.38, 0.52, 0.45, 0.65],
  [0.6, 0.42, 0.5, 0.36],
  [0.45, 0.58, 0.4, 0.52],
  [0.5, 0.36, 0.63, 0.47],
  [0.4, 0.55, 0.48, 0.62],
  [0.58, 0.44, 0.38, 0.52],
  [0.47, 0.6, 0.52, 0.4],
  [0.36, 0.5, 0.44, 0.58],
  [0.52, 0.45, 0.6, 0.42],
  [0.44, 0.62, 0.38, 0.55],
];
const PHASES_PER_JOB = 6;

interface JobSeg {
  lane: number;
  start: number; // s
  end: number;
  alt: boolean;
}
interface PhaseSeg {
  start: number;
  workEnd: number[]; // per thread
  phaseEnd: number; // slowest thread done
  syncEnd: number;
}

const jobSegs: JobSeg[] = [];
for (let lane = 0; lane < JOB_LANES.length; lane++) {
  let t = 0;
  JOB_LANES[lane].forEach((dur, i) => {
    jobSegs.push({ lane, start: t, end: t + dur, alt: i % 2 === 1 });
    t += dur;
  });
}

const phaseSegs: PhaseSeg[] = [];
{
  let t = 0;
  for (const chunks of PHASE_CHUNKS) {
    const phaseEnd = t + Math.max(...chunks);
    phaseSegs.push({ start: t, workEnd: chunks.map((c) => t + c), phaseEnd, syncEnd: phaseEnd + SYNC });
    t = phaseEnd + SYNC;
  }
}

const FONT = 'Inter, system-ui, sans-serif';

function drawPanel(
  ctx: CanvasRenderingContext2D,
  w: number,
  y: number,
  panelH: number,
  tSec: number,
  kind: 'jobs' | 'kernel'
) {
  const m = 18;
  const labelW = 64;
  const trackX = m + labelW;
  const trackW = w - trackX - m;
  const laneH = 16;
  const laneGap = 8;
  const lanesY = y + 30;
  const pxPerSec = trackW / (SWEEP / 1000);
  const cursorX = trackX + Math.min(tSec, SWEEP / 1000) * pxPerSec;

  ctx.font = `600 12.5px ${FONT}`;
  ctx.fillStyle = TEXT;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  const compact = w < 780;
  ctx.fillText(
    kind === 'jobs'
      ? compact
        ? 'Jobs: whole solves per worker'
        : 'Jobs, data-parallel: each worker owns a whole solve'
      : compact
        ? 'Kernel-parallel GEMM: sync inside one solve'
        : 'Kernel-parallel GEMM (what Blaze avoids): threads sync inside one solve',
    m,
    y + 14
  );

  // Lane tracks.
  for (let i = 0; i < 4; i++) {
    const ly = lanesY + i * (laneH + laneGap);
    ctx.fillStyle = MUTED;
    ctx.font = `10.5px ${FONT}`;
    ctx.textBaseline = 'middle';
    ctx.fillText(kind === 'jobs' ? `worker ${i + 1}` : `thread ${i + 1}`, m, ly + laneH / 2);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(trackX, ly, trackW, laneH);
  }

  const clipW = cursorX - trackX;
  if (clipW > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(trackX, lanesY - 4, clipW, 4 * (laneH + laneGap) + 8);
    ctx.clip();

    if (kind === 'jobs') {
      for (const seg of jobSegs) {
        const ly = lanesY + seg.lane * (laneH + laneGap);
        ctx.fillStyle = ORANGE;
        ctx.globalAlpha = seg.alt ? 0.55 : 0.9;
        ctx.fillRect(trackX + seg.start * pxPerSec + 1, ly, (seg.end - seg.start) * pxPerSec - 2, laneH);
      }
      ctx.globalAlpha = 1;
    } else {
      let idleLabeled = false;
      for (const [pi, ph] of phaseSegs.entries()) {
        for (let i = 0; i < 4; i++) {
          const ly = lanesY + i * (laneH + laneGap);
          // Work chunk.
          ctx.fillStyle = ORANGE;
          ctx.globalAlpha = pi % 2 === 1 ? 0.55 : 0.9;
          ctx.fillRect(trackX + ph.start * pxPerSec + 1, ly, (ph.workEnd[i] - ph.start) * pxPerSec - 1, laneH);
          ctx.globalAlpha = 1;
          // Idle until the slowest thread finishes.
          const idleW = (ph.phaseEnd - ph.workEnd[i]) * pxPerSec;
          if (idleW > 1) {
            ctx.fillStyle = 'rgba(255,255,255,0.06)';
            ctx.fillRect(trackX + ph.workEnd[i] * pxPerSec, ly, idleW, laneH);
            if (!idleLabeled && idleW > 34) {
              ctx.fillStyle = MUTED;
              ctx.font = `9px ${FONT}`;
              ctx.textAlign = 'center';
              ctx.fillText('idle', trackX + (ph.workEnd[i] + ph.phaseEnd) / 2 * pxPerSec, ly + laneH / 2);
              ctx.textAlign = 'left';
              idleLabeled = true;
            }
          }
        }
        // Sync barrier across all lanes.
        const bx = trackX + ph.phaseEnd * pxPerSec;
        const bw = SYNC * pxPerSec;
        ctx.fillStyle = 'rgba(255,80,80,0.22)';
        ctx.fillRect(bx, lanesY - 2, bw, 4 * (laneH + laneGap) - laneGap + 4);
        if (pi === 0) {
          ctx.fillStyle = 'rgba(255,120,120,0.85)';
          ctx.font = `9px ${FONT}`;
          ctx.textAlign = 'center';
          ctx.fillText('sync', bx + bw / 2, lanesY - 10);
          ctx.textAlign = 'left';
        }
        // Job boundary marker.
        if ((pi + 1) % PHASES_PER_JOB === 0) {
          ctx.strokeStyle = BLUE;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(bx + bw, lanesY - 4);
          ctx.lineTo(bx + bw, lanesY + 4 * (laneH + laneGap) - laneGap + 4);
          ctx.stroke();
        }
      }
    }
    ctx.restore();
  }

  // Time cursor.
  if (tSec < SWEEP / 1000) {
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cursorX, lanesY - 6);
    ctx.lineTo(cursorX, lanesY + 4 * (laneH + laneGap) - laneGap + 6);
    ctx.stroke();
  }

  // Jobs-finished counter.
  const finished =
    kind === 'jobs'
      ? jobSegs.filter((s) => s.end <= tSec).length
      : Math.floor(phaseSegs.filter((p) => p.syncEnd <= tSec).length / PHASES_PER_JOB);
  ctx.font = `11.5px ${FONT}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = MUTED;
  ctx.fillText('jobs finished: ', w - m - 22, y + 14);
  ctx.fillStyle = kind === 'jobs' ? ORANGE : 'rgba(255,120,120,0.9)';
  ctx.font = `600 12.5px ${FONT}`;
  ctx.fillText(String(finished), w - m, y + 14);
  ctx.textAlign = 'left';
}

function draw(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, w, h);

  const tSec = Math.min(t, SWEEP) / 1000;
  const panelH = (h - 34) / 2;
  drawPanel(ctx, w, 10, panelH, tSec, 'jobs');
  drawPanel(ctx, w, 10 + panelH, panelH, tSec, 'kernel');

  ctx.fillStyle = MUTED;
  ctx.font = `11px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(
    w < 780
      ? 'Same four cores; the barriers eat the speedup.'
      : 'Same four cores. At production grid sizes there is not enough work per GEMM to amortize a fork, so the barriers win.',
    w / 2,
    h - 14
  );
}

export default function JobsVsKernels({ height = 340 }: { height?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!wrap || !canvas || !ctx) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    let elapsed = 0;
    let last: number | null = null;
    let visible = true;
    let w = wrap.clientWidth;

    const resize = () => {
      w = wrap.clientWidth;
      if (w === 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(ctx, w, height, reduced ? SWEEP : elapsed % CYCLE);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    resize();

    if (reduced) {
      return () => ro.disconnect();
    }

    const io = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? true;
    });
    io.observe(wrap);

    const tick = (ts: number) => {
      raf = requestAnimationFrame(tick);
      if (!visible || document.hidden) {
        last = null;
        return;
      }
      if (last !== null) elapsed += Math.min(ts - last, 100);
      last = ts;
      draw(ctx, w, height, elapsed % CYCLE);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, [height]);

  return (
    <div
      ref={wrapRef}
      style={{
        width: '100%',
        background: PANEL,
        border: '1px solid rgba(128,128,128,0.25)',
        borderRadius: 12,
        overflow: 'hidden',
      }}
    >
      <canvas ref={canvasRef} style={{ display: 'block' }} />
    </div>
  );
}
