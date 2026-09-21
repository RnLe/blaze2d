'use client';

import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Expand, Shrink } from 'lucide-react';
import { latticeBasis, latticePoints, moireGeometry, rotate, type Lattice, type Point } from '@/lib/moire';
import { theme } from '@/lib/theme';

function UnitCell({ lattice, angle, layer }: { lattice: Lattice; angle: number; layer: 'fixed' | 'rotated' }) {
  const id = useId();
  const radians = angle * Math.PI / 180;
  const basis = latticeBasis(lattice).map(point => rotate(point, radians));
  const project = ([x, y]: Point): Point => [100 + 52 * x, 94 - 52 * y];
  const corners: Point[] = [[0, 0], basis[0], [basis[0][0] + basis[1][0], basis[0][1] + basis[1][1]], basis[1]];
  return <div className={`bilayer-unit bilayer-unit-${layer}`}>
    <svg viewBox="0 0 220 168" role="img" aria-label={`${layer === 'fixed' ? 'Blue fixed' : 'Orange rotated'} layer primitive cell with two lattice vectors`}>
      <defs><marker id={id} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6Z" fill="currentColor" /></marker></defs>
      {latticePoints(lattice, 2.7).map((point, index) => {
        const [x, y] = project(rotate(point, radians));
        return <circle key={index} cx={x} cy={y} r="3" fill="currentColor" opacity="0.45" />;
      })}
      <polygon points={corners.map(point => project(point).join(',')).join(' ')} fill="currentColor" fillOpacity="0.13" stroke="currentColor" strokeWidth="1.5" />
      {basis.map((point, index) => {
        const [x, y] = project(point);
        return <g key={index}><line x1="100" y1="94" x2={x} y2={y} stroke="currentColor" strokeWidth="2" markerEnd={`url(#${id})`} /><text x={x + 10} y={y - 8}>a{index === 0 ? '₁' : '₂'}{layer === 'rotated' ? '′' : ''}</text></g>;
      })}
    </svg>
    <span>{layer === 'fixed' ? 'Layer 1 · fixed' : `Layer 2 · ${angle.toFixed(1)}°`}</span>
  </div>;
}

export default function BilayerExplorer() {
  const [lattice, setLattice] = useState<Lattice>('triangular');
  const [angle, setAngle] = useState(8);
  const [viewWidth, setViewWidth] = useState(40);
  const [expanded, setExpanded] = useState(false);
  const slot = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const scene = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 1000, height: 600 });
  const { width, height } = size;
  const id = useId();
  const geometry = moireGeometry(lattice, angle);
  const points = useMemo(() => latticePoints(lattice, Math.hypot(viewWidth / 2, viewWidth * height / width / 2) + 1), [lattice, viewWidth, width, height]);
  const scale = width / viewWidth;
  const project = ([x, y]: Point): Point => [width / 2 + x * scale, height / 2 - y * scale];
  const scaleBarY = width < 440 ? 72 : height - 34;
  const scaleBarX = width < 440 ? 18 : 35;
  const periodLabel = Number.isFinite(geometry.period) ? geometry.period.toFixed(2) : '∞';

  useEffect(() => {
    const element = slot.current;
    const shell = element?.closest('.docs-shell');
    if (!element || !shell) return;
    const navigation = Array.from(shell.querySelectorAll<HTMLElement>('.site-sidebar, .article-toc'));
    const previousInert = navigation.map(nav => nav.inert);
    shell.classList.toggle('bilayer-focused', expanded);
    navigation.forEach((nav, index) => { nav.inert = expanded || previousInert[index]; });
    let frame = 0;
    function update() {
      if (!element || !shell) return;
      const bounds = element.getBoundingClientRect();
      const viewport = document.documentElement.clientWidth;
      const gutter = viewport <= 640 ? 8 : 16;
      const header = shell.querySelector('.site-header')?.getBoundingClientRect().height ?? 72;
      // Widening the child leaves the surrounding article layout unchanged.
      element.style.setProperty('--bilayer-full-width', `${viewport - 2 * gutter}px`);
      element.style.setProperty('--bilayer-offset', `${gutter - bounds.left}px`);
      // Scrolling can only collapse the view. Expansion always needs the button.
      if (expanded && (bounds.bottom <= header || bounds.top >= window.innerHeight)) setExpanded(false);
    }
    function schedule() { cancelAnimationFrame(frame); frame = requestAnimationFrame(update); }
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    schedule();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
      shell.classList.remove('bilayer-focused');
      navigation.forEach((nav, index) => { nav.inert = previousInert[index]; });
    };
  }, [expanded]);

  useEffect(() => {
    const element = scene.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.round(entry.contentRect.width);
      const height = Math.round(entry.contentRect.height);
      if (width && height) setSize(previous => previous.width === width && previous.height === height ? previous : { width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    const frame = requestAnimationFrame(() => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      element.width = Math.round(width * ratio);
      element.height = Math.round(height * ratio);
      context.setTransform(element.width / width, 0, 0, element.height / height, 0, 0);
      context.clearRect(0, 0, width, height);
      for (const layer of [0, 1]) {
        const radians = layer * angle * Math.PI / 180;
        const radius = Math.max(1.25, 0.145 * scale);
        context.beginPath();
        for (const point of points) {
          const [x, y] = rotate(point, radians);
          const px = width / 2 + x * scale;
          const py = height / 2 - y * scale;
          if (px < -radius || px > width + radius || py < -radius || py > height + radius) continue;
          context.moveTo(px + radius, py);
          context.arc(px, py, radius, 0, Math.PI * 2);
        }
        context.globalCompositeOperation = layer === 0 ? 'source-over' : 'screen';
        context.fillStyle = layer === 0 ? theme.brandBlue : theme.brandOrange;
        context.fill();
      }
      context.globalCompositeOperation = 'source-over';
    });
    return () => cancelAnimationFrame(frame);
  }, [angle, points, scale, width, height]);

  function toggleExpansion() {
    setExpanded(!expanded);
    const element = slot.current;
    if (expanded || !element) return;
    const bounds = element.getBoundingClientRect();
    const header = element.closest('.docs-shell')?.querySelector('.site-header')?.getBoundingClientRect().height ?? 0;
    window.scrollTo({
      top: window.scrollY + bounds.top + bounds.height / 2 - (window.innerHeight + header) / 2,
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
    });
  }

  const vectors = geometry.vectors;
  const corners: Point[] | null = vectors ? [[0, 0], vectors[0], [vectors[0][0] + vectors[1][0], vectors[0][1] + vectors[1][1]], vectors[1]] : null;

  return <div className="bilayer-slot" ref={slot} data-expanded={expanded}>
    <section id={`${id}-explorer`} className="bilayer-explorer" aria-label="Interactive bilayer explorer">
    <div className="bilayer-workspace">
      <div className="bilayer-controls">
        <fieldset className="bilayer-lattice"><legend>Lattice</legend>
          {(['triangular', 'square'] as const).map(value => <label key={value}>
            <input type="radio" name={`${id}-lattice`} value={value} checked={lattice === value} onChange={() => { setLattice(value); setAngle(Math.min(angle, value === 'triangular' ? 60 : 90)); }} />
            <span>{value === 'triangular' ? 'Triangular' : 'Square'}</span>
          </label>)}
        </fieldset>
        <fieldset className="bilayer-view"><legend>View width</legend>
          {[{ width: 24, label: 'Closer' }, { width: 40, label: 'Overview' }, { width: 64, label: 'Wider' }].map(view =>
            <button type="button" key={view.width} aria-pressed={viewWidth === view.width} onClick={() => setViewWidth(view.width)}>
              <strong>{view.width}a</strong><span>{view.label}</span>
            </button>)}
        </fieldset>
        <div className="bilayer-angle">
          <label htmlFor={`${id}-angle`}>Twist angle <output>{angle.toFixed(1)}°</output></label>
          <div className="bilayer-angle-track">
            <input id={`${id}-angle`} type="range" min="0" max={geometry.symmetry} step="0.1" value={angle} aria-orientation="vertical"
              aria-valuetext={`${angle.toFixed(1)} degrees. ${vectors ? `Moiré period ${periodLabel} lattice constants.` : 'Layers aligned; no finite moiré period.'}`}
              onChange={event => setAngle(Number(event.target.value))} />
            <div className="bilayer-range-labels" aria-hidden="true">{[1, 0.75, 0.5, 0.25, 0].map(fraction => <span key={fraction}>{geometry.symmetry * fraction}°</span>)}</div>
          </div>
        </div>
        <div className="bilayer-readout">
          <span className="thesis-kicker">Two scales</span>
          <dl><div><dt>Spacing a</dt><dd>1</dd></div><div><dt>Period L/a</dt><dd data-moire-period={geometry.period}>{periodLabel}</dd></div><div><dt>Ratio η</dt><dd>{geometry.eta.toFixed(3)}</dd></div></dl>
        </div>
      </div>
      <div className="bilayer-scene">
        <button type="button" className="bilayer-expand" aria-label={expanded ? 'Collapse explorer' : 'Expand explorer'} aria-expanded={expanded} aria-controls={`${id}-explorer`} onClick={toggleExpansion}>
          {expanded ? <Shrink size={15} aria-hidden="true" /> : <Expand size={15} aria-hidden="true" />}
          <span>{expanded ? 'Collapse' : 'Expand'}</span>
        </button>
    <div className="bilayer-canvas-frame" ref={scene}>
      <canvas ref={canvas} aria-hidden="true" />
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${lattice} bilayer at ${angle.toFixed(1)} degrees. Blue sites are fixed; orange sites are rotated. ${vectors ? `White arrows are the moiré vectors, each ${periodLabel} times the monolayer spacing.` : 'Both layers coincide.'}`}>
        <defs><marker id={`${id}-arrow`} markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8Z" fill="var(--text-primary)" /></marker></defs>
        {corners && <polygon points={corners.map(point => project(point).join(',')).join(' ')} fill="var(--text-primary)" fillOpacity="0.035" stroke="var(--text-primary)" strokeOpacity="0.5" strokeWidth="1.5" strokeDasharray="6 7" />}
        {[0, 1].map(layer => {
          const [a, b] = geometry.basis.map(point => rotate(point, layer * angle * Math.PI / 180));
          return <polygon key={layer} points={([[0, 0], a, [a[0] + b[0], a[1] + b[1]], b] as Point[]).map(point => project(point).join(',')).join(' ')} fill="none" stroke={layer === 0 ? 'var(--accent)' : 'var(--brand-orange)'} strokeWidth="2" />;
        })}
        {vectors?.map((vector, index) => {
          const dx = vector[0] * scale, dy = -vector[1] * scale;
          const fraction = Math.min(1, (width / 2 - 45) / Math.abs(dx || 1e-12), (height / 2 - 45) / Math.abs(dy || 1e-12));
          const x = width / 2 + fraction * dx, y = height / 2 + fraction * dy;
          return <g key={index} data-moire-vector={index + 1}>
            <line x1={width / 2} y1={height / 2} x2={x} y2={y} stroke="var(--text-primary)" strokeWidth="2.2" strokeDasharray={fraction < 1 ? '7 5' : undefined} markerEnd={`url(#${id}-arrow)`} />
            <text x={x + (dx < 0 ? 12 : -12)} y={y + (dy > 0 ? -14 : 24)} textAnchor={dx < 0 ? 'start' : 'end'} className="bilayer-vector-label">L{index === 0 ? '₁' : '₂'}{fraction < 1 ? ' ↗' : ''}</text>
          </g>;
        })}
        <line x1={scaleBarX} y1={scaleBarY} x2={scaleBarX + scale * 5} y2={scaleBarY} stroke="var(--text-primary)" strokeWidth="2" />
        <text x={scaleBarX} y={scaleBarY - 14} className="bilayer-scale-label">5a</text>
      </svg>
      <span className="bilayer-canvas-label">Two layers<span> · real space</span></span>
    </div>
        <div className="bilayer-cell-pair" aria-label="Enlarged monolayer cells">
          <span className="bilayer-cell-note">Monolayer cells · enlarged</span>
          <UnitCell lattice={lattice} angle={0} layer="fixed" /><UnitCell lattice={lattice} angle={angle} layer="rotated" />
        </div>
      </div>
    </div>
  </section>
  </div>;
}
