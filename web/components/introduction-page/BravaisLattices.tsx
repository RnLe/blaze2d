type Vector = readonly [number, number];
type Lattice = { name: string; subtitle?: string; a1: Vector; a2: Vector; centered?: boolean };

const lattices: Lattice[] = [
  { name: 'Square', a1: [1, 0], a2: [0, 1] },
  { name: 'Rectangular', a1: [1.4, 0], a2: [0, 0.85] },
  { name: 'Centered rectangular', a1: [0.8, 0.5], a2: [-0.8, 0.5], centered: true },
  { name: 'Hexagonal', subtitle: '(triangular)', a1: [1, 0], a2: [0.5, Math.sqrt(3) / 2] },
  { name: 'Oblique', a1: [1.2, 0], a2: [0.35, 0.9] },
];

function add(a: Vector, b: Vector): Vector { return [a[0] + b[0], a[1] + b[1]]; }

function projection(points: Vector[], cx: number, cy: number, width: number, height: number) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const scale = Math.min(width / (xmax - xmin), height / (ymax - ymin));
  return ([x, y]: Vector): Vector => [cx + (x - (xmin + xmax) / 2) * scale, cy - (y - (ymin + ymax) / 2) * scale];
}

function Cell({ lattice, cx }: { lattice: Lattice; cx: number }) {
  const { a1, a2 } = lattice;
  const points: Vector[] = [[0, 0], a1, add(a1, a2), a2];
  const project = projection(points, cx, 145, 120, 110);
  const origin = project([0, 0]);
  return <g className="intro-bravais-cell">
    <polygon points={points.map(p => project(p).join(',')).join(' ')} />
    {points.map((p, i) => { const [x, y] = project(p); return <circle key={i} cx={x} cy={y} r={3.6} />; })}
    {[a1, a2].map((a, i) => {
      const end = project(a);
      const dx = end[0] - origin[0], dy = end[1] - origin[1], length = Math.hypot(dx, dy);
      const side = i === 0 ? 1 : -1;
      return <g key={i} className={`intro-vector intro-vector-${i + 1}`}>
        <line x1={origin[0]} y1={origin[1]} x2={end[0]} y2={end[1]} markerEnd={`url(#intro-a${i + 1})`} />
        <text x={(origin[0] + end[0]) / 2 - side * dy * 16 / length} y={(origin[1] + end[1]) / 2 + side * dx * 16 / length + 5}>
          a<tspan baselineShift="sub" fontSize="11">{i + 1}</tspan>
        </text>
      </g>;
    })}
  </g>;
}

function Patch({ lattice, cx }: { lattice: Lattice; cx: number }) {
  const { a1, a2 } = lattice;
  const at = (i: number, j: number): Vector => [i * a1[0] + j * a2[0], i * a1[1] + j * a2[1]];
  const points = Array.from({ length: 25 }, (_, n) => at(n % 5 - 2, Math.floor(n / 5) - 2));
  const project = projection(points, cx, 348, 164, 148);
  const cell: Vector[] = [[0, 0], a1, add(a1, a2), a2];
  const conventional: Vector[] = [[0, 0], [1.6, 0], [1.6, 1], [0, 1]];
  return <g className="intro-bravais-patch">
    {[-2, -1, 0, 1, 2].flatMap(i => [[at(i, -2), at(i, 2)], [at(-2, i), at(2, i)]].map(([a, b], j) => {
      const start = project(a), end = project(b);
      return <line key={`${i}-${j}`} x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} />;
    }))}
    <polygon className="intro-patch-cell" points={cell.map(p => project(p).join(',')).join(' ')} />
    {lattice.centered && <polygon className="intro-conventional-cell" points={conventional.map(p => project(p).join(',')).join(' ')} />}
    {points.map((p, i) => { const [x, y] = project(p); return <circle key={i} cx={x} cy={y} r={3.4} />; })}
  </g>;
}

export default function BravaisLattices() {
  return <figure className="intro-bravais">
    <div className="intro-bravais-scroll" tabIndex={0} role="region" aria-label="Five Bravais lattices. Scroll horizontally on a small screen to see every column.">
      <svg viewBox="0 0 1080 462" role="img" aria-labelledby="intro-bravais-title intro-bravais-description">
        <title id="intro-bravais-title">The five two-dimensional Bravais lattices</title>
        <desc id="intro-bravais-description">Five columns show square, rectangular, centered rectangular, hexagonal or triangular, and oblique lattices. The first row shows a primitive cell with vectors a1 in blue and a2 in orange. The second row repeats each lattice as a five by five patch of points. A dashed rectangle identifies a conventional centered rectangular cell.</desc>
        <defs>
          {[1, 2].map(i => <marker key={i} id={`intro-a${i}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse" markerUnits="userSpaceOnUse">
            <path d="M0,0 L8,4 L0,8 Z" className={`intro-arrow-${i}`} />
          </marker>)}
        </defs>
        <line className="intro-bravais-rule" x1="22" y1="236" x2="1058" y2="236" />
        {lattices.map((lattice, index) => {
          const cx = 108 + index * 216;
          return <g key={lattice.name}>
            <text className="intro-lattice-name" x={cx} y="32">{lattice.name}</text>
            {lattice.subtitle && <text className="intro-lattice-subtitle" x={cx} y="51">{lattice.subtitle}</text>}
            <Cell lattice={lattice} cx={cx} />
            <Patch lattice={lattice} cx={cx} />
          </g>;
        })}
      </svg>
    </div>
    <figcaption>Primitive cells and their translation vectors above; 5 × 5 lattice patches below. The shaded cell repeats to fill the plane. The dashed rectangle marks a conventional centered cell.<span className="intro-scroll-hint"> Scroll sideways to compare all five lattices.</span></figcaption>
  </figure>;
}
