'use client';

import { useEffect, useRef, useState } from 'react';
import { cssColor, KIND_COLORS, PACKET_PRECISION_COLORS } from '../../lib/archboard/palette';

/**
 * The path a field element takes on real hardware: DRAM over a dual-channel
 * bus into L3, up through L2 and L1, into the FPU. Box proportions roughly
 * follow capacity; dot speed per lane follows measured latency, anchored to
 * the L1 lane and square-root damped so the slow lanes stay watchable (the
 * true multiplier is printed on every lane). Dots live only on lanes: they
 * spawn at the source border and despawn at the target border.
 *
 * Latencies are approximate load-to-use figures for an i9-13900K P-core.
 */

const F32 = cssColor(PACKET_PRECISION_COLORS.f32);
const F64 = cssColor(PACKET_PRECISION_COLORS.f64);
const MAGENTA = cssColor(KIND_COLORS.memory);
const ORANGE = cssColor(KIND_COLORS.compute);
const TEXT = 'rgba(255,255,255,0.85)';
const MUTED = 'rgba(255,255,255,0.45)';
const FAINT = 'rgba(255,255,255,0.28)';
const PANEL = '#0d0d0d';
const BOX_FILL = '#141414';
const BOX_STROKE = 'rgba(255,255,255,0.18)';
const FONT = 'Inter, system-ui, sans-serif';

/** Approximate load-to-use latencies (ns), i9-13900K P-core + DDR5-5600. */
const LAT = { l1: 0.9, l2: 2.7, l3: 10, dram: 85 };
type Level = keyof typeof LAT;

const trueMult = (lv: Level) => LAT[lv] / LAT.l1;
/** Square-root damping keeps the DRAM lane visible instead of geological. */
const shownMult = (lv: Level) => Math.sqrt(trueMult(lv));
const fmtMult = (x: number) => (x >= 10 ? `${Math.round(x)}` : `${Math.round(x * 10) / 10}`);

/** L1-lane dot velocity at 1x (px/s); every other lane derives from it. */
const L1_SPEED = 300;
const SPEEDS = ['0.25', '0.5', '1', '2', '4'] as const;

interface LaneDef {
  id: string;
  level: Level;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Geom {
  dram: Box;
  l3: Box;
  l2: Box;
  l1: Box;
  core: Box;
  fpu: Box;
  lanes: LaneDef[];
  diamondX: number;
}

function layout(w: number): Geom {
  const m = 18;
  const W = w - 2 * m;
  const top = 46;
  const H = 150;

  const dram: Box = { x: m, y: top, w: W * 0.16, h: H };
  const l3: Box = { x: m + W * 0.3, y: top, w: W * 0.068, h: H };
  const l2: Box = { x: m + W * 0.475, y: top, w: W * 0.105, h: 62 };
  const cx = l2.x + l2.w / 2;
  const l1: Box = { x: cx - W * 0.0425, y: top + 62 + 36, w: W * 0.085, h: 26 };
  const core: Box = { x: m + W * 0.7, y: top, w: W * 0.3, h: H };
  const fpu: Box = { x: core.x + core.w * 0.32, y: l1.y - 8, w: core.w * 0.5, h: 42 };

  const busY = top + H / 2;
  const laneY = l1.y + l1.h / 2;
  const lanes: LaneDef[] = [
    { id: 'dram-a', level: 'dram', x0: dram.x + dram.w, y0: busY - 16, x1: l3.x, y1: busY - 16 },
    { id: 'dram-b', level: 'dram', x0: dram.x + dram.w, y0: busY + 16, x1: l3.x, y1: busY + 16 },
    { id: 'l3-l2', level: 'l3', x0: l3.x + l3.w, y0: l2.y + l2.h / 2, x1: l2.x, y1: l2.y + l2.h / 2 },
    { id: 'l2-l1', level: 'l2', x0: cx, y0: l2.y + l2.h, x1: cx, y1: l1.y },
    { id: 'l1-core', level: 'l1', x0: l1.x + l1.w, y0: laneY, x1: fpu.x, y1: laneY },
  ];

  return { dram, l3, l2, l1, core, fpu, lanes, diamondX: core.x };
}

const laneLen = (l: LaneDef) => Math.hypot(l.x1 - l.x0, l.y1 - l.y0);

interface Dot {
  p: number; // 0..1 along the lane
}

type DotMap = Map<string, Dot[]>;

function drawBox(ctx: CanvasRenderingContext2D, b: Box, stroke: string) {
  ctx.fillStyle = BOX_FILL;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(b.x, b.y, b.w, b.h, 6);
  ctx.fill();
  ctx.stroke();
}

function centeredLines(ctx: CanvasRenderingContext2D, b: Box, lines: Array<[string, string, number]>) {
  const total = lines.length;
  const cx = b.x + b.w / 2;
  const cy = b.y + b.h / 2;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  lines.forEach(([text, color, size], i) => {
    ctx.fillStyle = color;
    ctx.font = `${size}px ${FONT}`;
    ctx.fillText(text, cx, cy + (i - (total - 1) / 2) * (size + 4));
  });
}

function draw(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  geom: Geom,
  dots: DotMap,
  storage: 'f32' | 'f64',
  pulseT: number
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, w, h);

  const storageColor = storage === 'f32' ? F32 : F64;

  // Legend, top right.
  ctx.font = `11px ${FONT}`;
  ctx.textBaseline = 'middle';
  const legend: Array<[string, string]> =
    storage === 'f32'
      ? [
          ['f32 in storage', F32],
          ['f64 in the core', F64],
        ]
      : [['f64 everywhere', F64]];
  let lx = w - 18;
  for (const [label, color] of [...legend].reverse()) {
    ctx.textAlign = 'right';
    ctx.fillStyle = MUTED;
    ctx.fillText(label, lx, 18);
    lx -= ctx.measureText(label).width + 12;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx, 18, 4, 0, Math.PI * 2);
    ctx.fill();
    lx -= 16;
  }

  // Lanes first (under the boxes' borders they touch).
  for (const lane of geom.lanes) {
    ctx.strokeStyle = 'rgba(255,255,255,0.13)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(lane.x0, lane.y0);
    ctx.lineTo(lane.x1, lane.y1);
    ctx.stroke();
    // Direction arrowhead at the target border.
    const ang = Math.atan2(lane.y1 - lane.y0, lane.x1 - lane.x0);
    ctx.fillStyle = FAINT;
    ctx.save();
    ctx.translate(lane.x1, lane.y1);
    ctx.rotate(ang);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-7, -4);
    ctx.lineTo(-7, 4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Per-lane multiplier labels.
  ctx.textAlign = 'center';
  const busMidX = (geom.lanes[0].x0 + geom.lanes[0].x1) / 2;
  ctx.fillStyle = MUTED;
  ctx.font = `10px ${FONT}`;
  ctx.fillText('DDR5 bus · 2 channels', busMidX, geom.lanes[0].y0 - 14);
  ctx.fillStyle = FAINT;
  ctx.font = `9.5px ${FONT}`;
  ctx.fillText(
    `true ×${fmtMult(trueMult('dram'))} vs L1 · shown ×${fmtMult(shownMult('dram'))}`,
    busMidX,
    geom.lanes[1].y0 + 16
  );

  const l3l2 = geom.lanes[2];
  ctx.fillText(
    `true ×${fmtMult(trueMult('l3'))} · shown ×${fmtMult(shownMult('l3'))}`,
    (l3l2.x0 + l3l2.x1) / 2,
    l3l2.y0 - 10
  );

  const l2l1 = geom.lanes[3];
  ctx.textAlign = 'right';
  ctx.fillText(`true ×${fmtMult(trueMult('l2'))} · shown ×${fmtMult(shownMult('l2'))}`, l2l1.x0 - 10, (l2l1.y0 + l2l1.y1) / 2);

  const l1core = geom.lanes[4];
  ctx.textAlign = 'center';
  ctx.fillText('×1 · anchor', (l1core.x0 + geom.diamondX) / 2, l1core.y0 + 16);

  // Boxes with size and latency labels.
  drawBox(ctx, geom.dram, MAGENTA);
  centeredLines(ctx, geom.dram, [
    ['DRAM', TEXT, 12.5],
    ['DDR5 · 32 GB', MUTED, 10.5],
    ['≈85 ns', FAINT, 10],
  ]);

  drawBox(ctx, geom.l3, MAGENTA);
  centeredLines(ctx, geom.l3, [
    ['L3', TEXT, 12.5],
    ['36 MB', MUTED, 10.5],
    ['≈10 ns', FAINT, 10],
  ]);

  drawBox(ctx, geom.l2, MAGENTA);
  centeredLines(ctx, geom.l2, [
    ['L2 · 2 MB', TEXT, 11.5],
    ['≈2.7 ns', FAINT, 10],
  ]);

  drawBox(ctx, geom.l1, MAGENTA);
  centeredLines(ctx, geom.l1, [['L1d · 48 KB', TEXT, 10.5]]);
  ctx.fillStyle = FAINT;
  ctx.font = `10px ${FONT}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('≈0.9 ns', geom.l1.x + geom.l1.w / 2, geom.l1.y + geom.l1.h + 14);

  drawBox(ctx, geom.core, BOX_STROKE);
  ctx.fillStyle = MUTED;
  ctx.font = `10px ${FONT}`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('P-core', geom.core.x + 8, geom.core.y + 16);

  const pulse = 0.5 + 0.5 * Math.sin((pulseT / 1200) * Math.PI * 2);
  ctx.save();
  ctx.shadowColor = ORANGE;
  ctx.shadowBlur = 6 + 8 * pulse;
  drawBox(ctx, geom.fpu, ORANGE);
  ctx.restore();
  centeredLines(ctx, geom.fpu, [['FPU · f64', TEXT, 11.5]]);

  // Cast boundary diamond (only when storage is f32), board-style marker.
  if (storage === 'f32') {
    ctx.save();
    ctx.translate(geom.diamondX, geom.lanes[4].y0);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(-4.5, -4.5, 9, 9);
    ctx.restore();
    ctx.fillStyle = MUTED;
    ctx.font = `9.5px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('upcast', geom.diamondX, geom.lanes[4].y0 + 20);
  }

  // Dots, spawned and despawned at the box borders, moving only on lanes.
  const dotR = storage === 'f32' ? 3.2 : 4.4;
  for (const lane of geom.lanes) {
    const list = dots.get(lane.id);
    if (!list) continue;
    for (const dot of list) {
      const x = lane.x0 + (lane.x1 - lane.x0) * dot.p;
      const y = lane.y0 + (lane.y1 - lane.y0) * dot.p;
      const color = storage === 'f64' ? F64 : lane.id === 'l1-core' && x >= geom.diamondX ? F64 : storageColor;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

const toggleStyle = (active: boolean): React.CSSProperties => ({
  background: active ? 'rgba(255,255,255,0.12)' : 'transparent',
  color: active ? TEXT : MUTED,
  border: `1px solid ${active ? 'rgba(255,255,255,0.3)' : BOX_STROKE}`,
  borderRadius: 6,
  padding: '3px 10px',
  fontSize: 12,
  cursor: 'pointer',
  fontFamily: 'inherit',
});

export default function MemoryHierarchy({ height = 250 }: { height?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [storage, setStorage] = useState<'f32' | 'f64'>('f32');
  const [speed, setSpeed] = useState(1);
  const opts = useRef({ storage, speed });
  opts.current = { storage, speed };

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!wrap || !canvas || !ctx) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    let last: number | null = null;
    let pulseT = 0;
    let visible = true;
    let w = wrap.clientWidth;
    let geom = layout(w);

    const dots: DotMap = new Map();
    const acc = new Map<string, number>();
    // Element spacing on every lane (px); f64 elements are twice the bytes,
    // so at fixed bus bandwidth half as many arrive.
    const spacing = () => (opts.current.storage === 'f32' ? 32 : 64);
    const velocity = (lane: LaneDef) => (L1_SPEED / shownMult(lane.level)) * opts.current.speed;

    const seed = () => {
      dots.clear();
      acc.clear();
      for (const lane of geom.lanes) {
        const list: Dot[] = [];
        const step = spacing() / laneLen(lane);
        for (let p = step / 2; p < 1; p += step) list.push({ p });
        dots.set(lane.id, list);
        acc.set(lane.id, 0);
      }
    };

    const resize = () => {
      w = wrap.clientWidth;
      if (w === 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      geom = layout(w);
      draw(ctx, w, height, geom, dots, opts.current.storage, pulseT);
    };

    seed();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    resize();

    if (reduced) {
      // One static frame; re-seed and redraw on prop toggles via rAF-less loop.
      const id = setInterval(() => {
        seed();
        draw(ctx, w, height, geom, dots, opts.current.storage, 0);
      }, 400);
      return () => {
        clearInterval(id);
        ro.disconnect();
      };
    }

    const io = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? true;
    });
    io.observe(wrap);

    let lastStorage = opts.current.storage;
    const tick = (ts: number) => {
      raf = requestAnimationFrame(tick);
      if (!visible || document.hidden) {
        last = null;
        return;
      }
      const dt = last === null ? 0 : Math.min(ts - last, 100);
      last = ts;
      pulseT += dt;

      if (opts.current.storage !== lastStorage) {
        lastStorage = opts.current.storage;
        seed();
      }

      for (const lane of geom.lanes) {
        const len = laneLen(lane);
        const v = velocity(lane);
        const list = dots.get(lane.id)!;
        for (const dot of list) dot.p += (v * dt) / 1000 / len;
        // Despawn at the target border.
        dots.set(
          lane.id,
          list.filter((d) => d.p <= 1)
        );
        // Spawn at the source border, one element per `spacing` of travel.
        const interval = (spacing() / v) * 1000;
        let a = (acc.get(lane.id) ?? 0) + dt;
        while (a >= interval) {
          a -= interval;
          dots.get(lane.id)!.push({ p: ((a / 1000) * v) / len });
        }
        acc.set(lane.id, a);
      }

      draw(ctx, w, height, geom, dots, opts.current.storage, pulseT);
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
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 10,
          padding: '12px 18px 2px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button style={toggleStyle(storage === 'f32')} onClick={() => setStorage('f32')}>
            f32 storage
          </button>
          <button style={toggleStyle(storage === 'f64')} onClick={() => setStorage('f64')}>
            f64 storage
          </button>
          <span style={{ color: MUTED, fontSize: 11.5 }}>
            {storage === 'f32' ? '8 B per element: twice the elements on every lane' : '16 B per element: half the arrival rate'}
          </span>
        </div>
        <label style={{ color: MUTED, fontSize: 11.5, display: 'flex', alignItems: 'center', gap: 6 }}>
          speed
          <select
            value={String(speed)}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{
              background: BOX_FILL,
              color: TEXT,
              border: `1px solid ${BOX_STROKE}`,
              borderRadius: 6,
              padding: '3px 6px',
              fontSize: 12,
              fontFamily: 'inherit',
            }}
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
      </div>
      <canvas ref={canvasRef} style={{ display: 'block' }} />
      <div style={{ padding: '2px 18px 14px', color: MUTED, fontSize: 11, lineHeight: 1.55 }}>
        Lane speeds are anchored to the L1 lane and square-root damped so the slow lanes stay watchable: at
        true scale the DRAM bus would run about ×{fmtMult(trueMult('dram'))} slower than L1, not ×
        {fmtMult(shownMult('dram'))}. Latencies are approximate load-to-use figures for an i9-13900K
        (L1d 0.9 ns, L2 2.7 ns, L3 10 ns, dual-channel DDR5 ≈85 ns).
      </div>
    </div>
  );
}
