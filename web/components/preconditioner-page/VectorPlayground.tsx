'use client';

import { useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { apply, eigenvalues2, inverse, lineAngle, multiply, norm, rayleigh, scale, sub, symmetric, type Mat2, type Vec2 } from './linear-algebra';

type Stage = 'operator' | 'residual' | 'preconditioned';
type Kind = 'identity' | 'jacobi' | 'exact' | 'shifted';

const PRECONDITIONERS: { kind: Kind; label: string; formula: string }[] = [
  { kind: 'identity', label: 'None', formula: 'T = I' },
  { kind: 'jacobi', label: 'Diagonal', formula: 'T = diag(A)⁻¹' },
  { kind: 'exact', label: 'Exact inverse', formula: 'T = A⁻¹' },
  { kind: 'shifted', label: 'Shifted inverse', formula: 'T = (A − ρI)⁻¹' },
];

/** Smallest eigenvalue; the stretch slider sets λ₂ = ratio · λ₁. */
const LAMBDA1 = 1.25;
const X_MAX = 1.7;
const X_MIN = 0.25;
const SIZE = 600;
const SNAP_DEGREES = 1.5;
const IDENTITY: Mat2 = [[1, 0], [0, 1]];
/** Start vectors at |x| = 1.6: far from v₁ where A visibly turns x; for the
 * preconditioned stage 25° from v₁, where the plain step without T makes no progress. */
const polarVector = (degrees: number): Vec2 => [1.6 * Math.cos(degrees * Math.PI / 180), 1.6 * Math.sin(degrees * Math.PI / 180)];
const START: Record<Stage, Vec2> = { operator: polarVector(80), residual: polarVector(80), preconditioned: polarVector(55) };
/** Screen coordinates rounded to 0.01 px, so server and browser render identical markup. */
const px = (value: number) => Math.round(value * 100) / 100;

const format = (value: number, digits = 2) => (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits);
const formatVector = ([a, b]: Vec2) => `(${format(a)}, ${format(b)})`;

/** Keep |x| inside the drawable range, and lock onto an eigen-direction when close. */
function constrain(x: Vec2, directions: Vec2[]): Vec2 {
  const length = Math.min(X_MAX, Math.max(X_MIN, norm(x)));
  let result = scale(x, length / norm(x));
  for (const v of directions) {
    if (lineAngle(result, v) < SNAP_DEGREES) {
      const sign = Math.sign(result[0] * v[0] + result[1] * v[1]) || 1;
      result = scale(v, sign * length);
    }
  }
  return result;
}

function preconditionerMatrix(kind: Kind, a: Mat2, rho: number): Mat2 | null {
  if (kind === 'identity') return IDENTITY;
  if (kind === 'jacobi') return [[1 / a[0][0], 0], [0, 1 / a[1][1]]];
  if (kind === 'exact') return inverse(a);
  return inverse([[a[0][0] - rho, a[0][1]], [a[1][0], a[1][1] - rho]]);
}

type Label = { text: string; color: string; at: Vec2 };

/** A label beside a segment at fraction `along` of its length, on the side given by `side` (±1). */
function midLabel(from: Vec2, to: Vec2, text: string, color: string, side = 1, distance = 16, along = 0.5): Label {
  const dx = to[0] - from[0], dy = to[1] - from[1], length = Math.hypot(dx, dy) || 1;
  return { text, color, at: [px(from[0] + along * dx - side * dy / length * distance), px(from[1] + along * dy + side * dx / length * distance)] };
}

function Arrow({ from, to, color, marker, width = 2.4, dashed = false }: { from: Vec2; to: Vec2; color: string; marker: string; width?: number; dashed?: boolean }) {
  if (Math.hypot(to[0] - from[0], to[1] - from[1]) < 3) return null;
  return <line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} stroke={color} strokeWidth={width} strokeLinecap="round"
    strokeDasharray={dashed ? '6 6' : undefined} markerEnd={`url(#${marker})`} />;
}

export default function VectorPlayground({ stage }: { stage: Stage }) {
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const svg = useRef<SVGSVGElement>(null);
  const handle = useRef<SVGCircleElement>(null);
  const [ratio, setRatio] = useState(4);
  const [angle, setAngle] = useState(30);
  const [x, setX] = useState<Vec2>(START[stage]);
  const [kind, setKind] = useState<Kind>('identity');
  const [trail, setTrail] = useState<Vec2[]>([]);
  const [steps, setSteps] = useState(0);
  const [dragging, setDragging] = useState(false);

  // Touch drags must not scroll the page. React registers touch listeners as passive,
  // so the handle gets a native one that may cancel the default.
  useEffect(() => {
    const element = handle.current;
    if (!element) return;
    const prevent = (event: TouchEvent) => { if (event.cancelable) event.preventDefault(); };
    element.addEventListener('touchstart', prevent, { passive: false });
    return () => element.removeEventListener('touchstart', prevent);
  }, []);

  const { matrix: a, v1, v2, lambda1, lambda2 } = symmetric(LAMBDA1, LAMBDA1 * ratio, angle * Math.PI / 180);
  const ax = apply(a, x);
  const rho = rayleigh(a, x);
  const r = sub(ax, scale(x, rho));
  const residual = norm(r);
  const eigen = residual < 1e-9 * norm(ax);
  const t = preconditionerMatrix(kind, a, rho);
  // (A − ρI)⁻¹(A − ρI)x = x exactly; computing it numerically would only add rounding.
  const tr: Vec2 | null = eigen ? null : kind === 'shifted' ? x : t ? apply(t, r) : null;
  const collapsed = kind === 'shifted' && !eigen;
  const next = tr && !collapsed ? constrain(sub(x, tr), [v1, v2]) : null;
  const ta = t ? eigenvalues2(multiply(t, a)) : null;
  const indefinite = ta !== null && ta[0] * ta[1] <= 0;
  const conditionTA = ta && !indefinite ? Math.max(Math.abs(ta[0]), Math.abs(ta[1])) / Math.min(Math.abs(ta[0]), Math.abs(ta[1])) : null;

  // World coordinates → SVG pixels. The view fits the longest image A·x of the drawable range.
  const radius = 1.14 * Math.max(lambda2 * X_MAX, X_MAX * 1.2);
  const pixels = SIZE / 2 / radius;
  const toScreen = ([u, v]: Vec2): Vec2 => [px(SIZE / 2 + u * pixels), px(SIZE / 2 - v * pixels)];
  const o = toScreen([0, 0]);
  const gridStep = radius > 5 ? 1 : 0.5;
  const gridLines = Array.from({ length: Math.floor(radius / gridStep) * 2 + 1 }, (_, i) => (i - Math.floor(radius / gridStep)) * gridStep);

  const ellipse = (m: Mat2) => Array.from({ length: 97 }, (_, i) => {
    const phi = (i / 96) * 2 * Math.PI;
    return toScreen(apply(m, [norm(x) * Math.cos(phi), norm(x) * Math.sin(phi)])).join(',');
  }).join(' ');

  function reset() {
    setTrail([]);
    setSteps(0);
  }

  function moveTo(target: Vec2) {
    // A point on the origin has no direction; keep the current one.
    setX(constrain(norm(target) < 1e-6 ? x : target, [v1, v2]));
    reset();
  }

  function pointerToWorld(event: ReactPointerEvent): Vec2 | null {
    const matrix = svg.current?.getScreenCTM();
    if (!matrix) return null;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return [(point.x - SIZE / 2) / pixels, (SIZE / 2 - point.y) / pixels];
  }

  function onPointerDown(event: ReactPointerEvent<SVGElement>) {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    const onHandle = event.target === handle.current;
    if (!onHandle && event.pointerType === 'touch') return; // touches on the plot scroll the page
    const world = pointerToWorld(event);
    if (!world) return;
    event.preventDefault();
    svg.current?.setPointerCapture(event.pointerId);
    setDragging(true);
    moveTo(world);
  }

  function onPointerMove(event: ReactPointerEvent<SVGElement>) {
    if (!dragging) return;
    const world = pointerToWorld(event);
    if (world) moveTo(world);
  }

  function onKeyDown(event: KeyboardEvent<SVGElement>) {
    const delta = (event.shiftKey ? 0.2 : 0.05) * X_MAX;
    const moves: Record<string, Vec2> = { ArrowRight: [delta, 0], ArrowLeft: [-delta, 0], ArrowUp: [0, delta], ArrowDown: [0, -delta] };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    moveTo([x[0] + move[0], x[1] + move[1]]);
  }

  function step() {
    if (!next) return;
    setTrail(previous => [...previous, x].slice(-10));
    setSteps(count => count + 1);
    setX(next);
  }

  const colors = { x: 'var(--text-heading)', ax: 'var(--pc-image)', r: 'var(--pc-residual)', tr: 'var(--pc-preconditioned)', guide: 'var(--text-subtle)' };
  const markers = Object.entries(colors).map(([key, color]) => <marker key={key} id={`${id}-${key}`} viewBox="0 0 10 10" refX="8.5" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 Z" fill={color} />
  </marker>);

  const sx = toScreen(x), sax = toScreen(ax), srx = toScreen(scale(x, rho));
  const showResidual = stage !== 'operator';
  const showPreconditioned = stage === 'preconditioned';
  const directionUnit = scale(x, 1 / norm(x));
  const residualUnit = residual > 0 ? scale(r, 1 / residual) : null;
  // Screen-space side of r relative to x, so the ρx label sits opposite the residual.
  const rSide = residualUnit ? Math.sign(directionUnit[0] * residualUnit[1] - directionUnit[1] * residualUnit[0]) || 1 : 1;
  const corner = (along: number, across: number): Vec2 => {
    if (!residualUnit) return srx;
    return [px(srx[0] + (directionUnit[0] * along + residualUnit[0] * across) * 11), px(srx[1] - (directionUnit[1] * along + residualUnit[1] * across) * 11)];
  };
  const eigenLabel = (v: Vec2, name: string, value: number) => {
    let end = scale(v, radius * 0.9);
    let screen = toScreen(end);
    if (screen[1] > SIZE - 40 || screen[0] > SIZE - 120 || screen[1] < 30) { end = scale(v, -radius * 0.9); screen = toScreen(end); }
    // Offset perpendicular to the line so the dashes do not strike through the text.
    const normal: Vec2 = [-v[1], v[0]];
    const offset = toScreen(sub(end, scale(normal, 14 / pixels)));
    return <text x={offset[0]} y={offset[1]} className="pc-plot-label pc-plot-guide-label" textAnchor={screen[0] > SIZE / 2 ? 'end' : 'start'}>{name} · λ = {format(value)}</text>;
  };

  const labels: Label[] = [midLabel(o, sx, 'x', colors.x, -rSide), midLabel(o, sax, 'Ax', colors.ax, eigen ? 1 : rSide, eigen ? 20 : 16, eigen ? 0.5 : 0.82)];
  if (showResidual && !eigen) {
    labels.push(midLabel(srx, sax, 'r', colors.r, 1, 14));
    if (!showPreconditioned) labels.push(midLabel(o, srx, 'ρx', colors.guide, -rSide, 18, 0.95));
  }
  if (showPreconditioned && tr && next) labels.push(midLabel(sx, toScreen(sub(x, tr)), '−Tr', colors.tr, -1, 18));

  const taDescription = indefinite ? 'TA is indefinite' : conditionTA ? `κ(TA) = ${format(conditionTA, 1)}` : '';
  const description = stage === 'operator'
    ? `x = ${formatVector(x)} is mapped to Ax = ${formatVector(ax)}. ${eigen ? 'x lies on an eigen-direction, so Ax is parallel to x.' : `The angle between x and Ax is ${format(lineAngle(x, ax), 1)} degrees.`}`
    : stage === 'residual'
      ? `Rayleigh quotient ρ = ${format(rho)}, residual r = ${formatVector(r)} with length ${format(residual, 3)}.`
      : `Preconditioner ${PRECONDITIONERS.find(p => p.kind === kind)?.formula}. ${collapsed ? 'Tr equals x, so the step removes x and adds no new direction.' : tr ? `Preconditioned residual Tr = ${formatVector(tr)}.` : 'x is an eigenvector; the residual is zero.'} ${taDescription}. The angle between x and the lowest eigenvector is ${format(lineAngle(x, v1), 1)} degrees.`;

  const polar = Math.round(Math.atan2(x[1], x[0]) * 180 / Math.PI);

  return <figure className={`pc-playground pc-stage-${stage}`}>
    <div className="pc-playground-body">
      <div className="pc-playground-panel">
        <div className="pc-matrix-block">
          <span className="pc-kicker">Matrix A</span>
          <div className="pc-matrix" role="img" aria-label={`A equals ${format(a[0][0])}, ${format(a[0][1])}; ${format(a[1][0])}, ${format(a[1][1])}`}>
            <span>{format(a[0][0])}</span><span>{format(a[0][1])}</span><span>{format(a[1][0])}</span><span>{format(a[1][1])}</span>
          </div>
          <div className="pc-range">
            <span><label htmlFor={`${id}-ratio`}>Stretch λ₂/λ₁</label><output htmlFor={`${id}-ratio`}>{format(ratio, 1)}</output></span>
            <input id={`${id}-ratio`} type="range" min="1" max="5" step="0.1" value={ratio} onChange={event => { setRatio(Number(event.target.value)); reset(); }} />
          </div>
          <div className="pc-range">
            <span><label htmlFor={`${id}-angle`}>Eigenvector angle</label><output htmlFor={`${id}-angle`}>{Math.round(angle)}°</output></span>
            <input id={`${id}-angle`} type="range" min="0" max="90" step="1" value={angle} onChange={event => { setAngle(Number(event.target.value)); reset(); }} />
          </div>
        </div>

        {showPreconditioned && <fieldset className="pc-choices">
          <legend>Preconditioner</legend>
          {PRECONDITIONERS.map(option => <label key={option.kind}>
            <input type="radio" name={`${id}-kind`} value={option.kind} checked={kind === option.kind} onChange={() => { setKind(option.kind); reset(); }} />
            <span><strong>{option.label}</strong><small>{option.formula}</small></span>
          </label>)}
        </fieldset>}

        <dl className="pc-readout">
          {stage === 'operator' && <>
            <div><dt>x</dt><dd>{formatVector(x)}</dd></div>
            <div><dt>Ax</dt><dd>{formatVector(ax)}</dd></div>
            <div><dt>∠(x, Ax)</dt><dd>{format(lineAngle(x, ax), 1)}°</dd></div>
            <div><dt>|Ax| / |x|</dt><dd>{format(norm(ax) / norm(x))}</dd></div>
          </>}
          {showResidual && <>
            <div><dt>ρ(x)</dt><dd>{format(rho, 3)}</dd></div>
            <div><dt>‖r‖</dt><dd>{format(residual, 3)}</dd></div>
            {stage === 'residual' && <div><dt>x · r</dt><dd>{format(x[0] * r[0] + x[1] * r[1], 3)}</dd></div>}
          </>}
          {showPreconditioned && <>
            <div><dt>κ(A) → κ(TA)</dt><dd>{format(lambda2 / lambda1, 1)} → {indefinite ? 'indefinite' : conditionTA ? format(conditionTA, 1) : '—'}</dd></div>
            <div><dt>∠(x, v₁)</dt><dd>{format(lineAngle(x, v1), 1)}°</dd></div>
            <div><dt>after a plain step</dt><dd>{next ? `${format(lineAngle(next, v1), 1)}°` : '—'}</dd></div>
          </>}
        </dl>

        {showPreconditioned && <div className="pc-step">
          <button type="button" onClick={step} disabled={!next}>Step x ← x − Tr</button>
          <button type="button" className="pc-secondary" onClick={() => { reset(); setX(START[stage]); }}>Reset</button>
          <span aria-live="polite">{steps > 0 ? `${steps} step${steps > 1 ? 's' : ''}` : ''}</span>
        </div>}

        <div className="pc-snaps">
          <span>Jump to</span>
          <button type="button" onClick={() => moveTo(scale(v1, norm(x)))}>v₁</button>
          <button type="button" onClick={() => moveTo(scale(v2, norm(x)))}>v₂</button>
          <button type="button" onClick={() => moveTo(START[stage])}>start</button>
        </div>
      </div>

      <div className="pc-plot-frame">
        <svg ref={svg} viewBox={`0 0 ${SIZE} ${SIZE}`} role="group" aria-label={description}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={() => setDragging(false)} onPointerCancel={() => setDragging(false)}
          data-dragging={dragging || undefined}>
          <defs>
            {markers}
            <clipPath id={`${id}-clip`}><rect x="0" y="0" width={SIZE} height={SIZE} /></clipPath>
          </defs>
          <g clipPath={`url(#${id}-clip)`} aria-hidden="true">
            {gridLines.map(value => {
              const [gx] = toScreen([value, 0]);
              const [, gy] = toScreen([0, value]);
              return <g key={value} className={value === 0 ? 'pc-axis' : 'pc-grid'}>
                <line x1={gx} y1="0" x2={gx} y2={SIZE} /><line x1="0" y1={gy} x2={SIZE} y2={gy} />
              </g>;
            })}
            {[v1, v2].map((v, index) => {
              const [p, q] = [toScreen(scale(v, -radius * 1.5)), toScreen(scale(v, radius * 1.5))];
              return <line key={index} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} className="pc-eigenline" />;
            })}
            {eigenLabel(v1, 'v₁', lambda1)}
            {eigenLabel(v2, 'v₂', lambda2)}

            <polygon points={ellipse(IDENTITY)} className="pc-circle" />
            {!showPreconditioned && <polygon points={ellipse(a)} className="pc-ellipse" />}
            {showPreconditioned && t && !indefinite && <polygon points={ellipse(multiply(t, a))} className="pc-ellipse pc-ellipse-preconditioned" />}

            {trail.map((point, index) => {
              const [tx, ty] = toScreen(point);
              return <circle key={index} cx={tx} cy={ty} r="4" className="pc-trail" style={{ opacity: 0.25 + 0.6 * (index + 1) / trail.length }} />;
            })}

            {showResidual && !eigen && <>
              <Arrow from={o} to={srx} color={colors.guide} marker={`${id}-guide`} width={1.6} dashed />
              <Arrow from={srx} to={sax} color={colors.r} marker={`${id}-r`} width={showPreconditioned ? 1.8 : 2.4} />
              {residualUnit && <polyline points={[corner(-1, 0), corner(-1, 1), corner(0, 1)].map(point => point.join(',')).join(' ')} className="pc-right-angle" />}
            </>}
            {!eigen && <Arrow from={o} to={sax} color={colors.ax} marker={`${id}-ax`} width={showPreconditioned ? 1.6 : 2.4} />}
            {showPreconditioned && tr && next && <>
              <Arrow from={sx} to={toScreen(sub(x, tr))} color={colors.tr} marker={`${id}-tr`} width={2.6} />
              <circle cx={toScreen(next)[0]} cy={toScreen(next)[1]} r="6" className="pc-next" />
            </>}
            <Arrow from={o} to={sx} color={colors.x} marker={`${id}-x`} width={2.8} />
            {/* On an eigen-direction Ax is parallel to x: draw it on top so it stays visible. */}
            {eigen && <Arrow from={o} to={sax} color={colors.ax} marker={`${id}-ax`} width={2} />}
            {labels.map(label => <text key={label.text} x={label.at[0]} y={label.at[1] + 5} textAnchor="middle" className="pc-plot-label" fill={label.color}>{label.text}</text>)}
          </g>
          <circle ref={handle} cx={sx[0]} cy={sx[1]} r="28" className="pc-handle" tabIndex={0} role="slider"
            aria-label="Tip of the vector x. Drag it, or use the arrow keys; hold Shift for larger steps."
            aria-roledescription="vector handle" aria-valuemin={-180} aria-valuemax={180} aria-valuenow={polar}
            aria-valuetext={`x = ${formatVector(x)}, at ${polar} degrees`} onKeyDown={onKeyDown} />
          <circle cx={sx[0]} cy={sx[1]} r="6.5" className="pc-handle-dot" aria-hidden="true" />
          {eigen && <text x={SIZE - 18} y={SIZE - 18} textAnchor="end" className="pc-plot-badge" aria-hidden="true">Ax = λx · eigenvector</text>}
          {showPreconditioned && collapsed && <text x={SIZE - 18} y={SIZE - 18} textAnchor="end" className="pc-plot-badge pc-plot-badge-warning" aria-hidden="true">Tr = x · no new direction</text>}
        </svg>
        <span className="pc-plot-hint" aria-hidden="true">Drag the tip of x</span>
      </div>
    </div>
  </figure>;
}
