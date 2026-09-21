import type { ReactNode } from 'react';
import Image from 'next/image';
import { ArrowRight, BookOpen, FileCode2, FileText, Globe2, Microchip, Settings2 } from 'lucide-react';
import benchmark from '@/public/data/benchmarks/multi-core.json';
import bandData from '@/public/data/benchmarks/series6-accuracy.json';
import { getAssetPath } from '@/lib/paths';
import type { ModuleId } from './roadmap-data';
import SymmetryPreview from './SymmetryPreview';

function Art({ label, children, viewBox = '0 0 180 110', className = '' }: { label: string; children: ReactNode; viewBox?: string; className?: string }) {
  return <svg className={`roadmap-art ${className}`} viewBox={viewBox} role="img" aria-label={label}>{children}</svg>;
}
function Panel({ label, children, className = '' }: { label: string; children: ReactNode; className?: string }) {
  return <div className={`roadmap-art-panel ${className}`}>{children}<span className="roadmap-art-label">{label}</span></div>;
}
function Arrow({ long = false }: { long?: boolean }) {
  return long
    ? <svg className="roadmap-banner-arrow is-long" viewBox="0 0 140 24" aria-hidden="true"><path d="M4 12H134 M125 4l10 8-10 8" /></svg>
    : <ArrowRight className="roadmap-banner-arrow" aria-hidden="true" />;
}

function Crystal({ radius = 8, stronger = false, variable = false }: { radius?: number; stronger?: boolean; variable?: boolean }) {
  return <Art label={`Square crystal with nine ${variable ? 'individually varied' : stronger ? 'larger' : 'small'} air holes`} viewBox="0 0 120 110">
    <rect x="9" y="4" width="102" height="102" rx="4" className={`roadmap-material${stronger ? ' is-stronger' : ''}`} />
    <path d="M43 4V106 M77 4V106 M9 38H111 M9 72H111" className="roadmap-art-grid" />
    {Array.from({ length: 9 }, (_, i) => <circle key={i} cx={26 + (i % 3) * 34} cy={21 + Math.floor(i / 3) * 34}
      r={variable ? radius + [0, 2, 1, 3, 0, 2, 1, 2, 0][i] : radius} className="roadmap-hole" />)}
  </Art>;
}
function Bands({ gap = false }: { gap?: boolean }) {
  const paths = [0, 1, 2, 3].map(band => Array.from({ length: 33 }, (_, i) => {
    const t = i / 32, y = [81 - 10 * Math.sin(Math.PI * t), 66 - 9 * Math.sin(Math.PI * t), 39 - 10 * Math.sin(Math.PI * t), 17 + 10 * Math.sin(Math.PI * t)][band];
    return `${i ? 'L' : 'M'}${15 + t * 150},${y}`;
  }).join(' '));
  return <Art label={gap ? 'Schematic band diagram with a gap between the second and third bands' : 'Schematic four-band dispersion diagram'}>
    <path d="M13 6V93H170 M89 6V93" className="roadmap-art-grid" />
    {gap && <rect x="14" y="39" width="151" height="18" className="roadmap-gap" />}
    {paths.map((d, i) => <path key={d} d={d} className={i < 2 ? 'roadmap-line-blue' : 'roadmap-line-orange'} />)}
    <text x="6" y="10" className="roadmap-art-text">ω</text>
    <g className="roadmap-art-ticks"><text x="14" y="107">Γ</text><text x="88" y="107">X</text><text x="162" y="107">M</text></g>
  </Art>;
}
function OperatorMatrix() {
  return <Art label="Schematic projected operator: diagonal entries and weaker off-diagonal couplings">
    <path d="M38 7H31V103H38 M142 7H149V103H142" className="roadmap-art-outline" />
    {Array.from({ length: 36 }, (_, i) => {
      const row = Math.floor(i / 6), col = i % 6, distance = Math.abs(row - col);
      return <rect key={i} x={42 + col * 16} y={9 + row * 16} width="13" height="13" rx="2"
        className={distance === 0 ? 'roadmap-fill-orange' : 'roadmap-fill-blue'} opacity={distance === 0 ? .95 : distance === 1 ? .65 : .14} />;
    })}
  </Art>;
}
function CoreArt() {
  return <div className="roadmap-art-flow">
    <Panel label="Dielectric geometry"><Crystal /></Panel><Arrow />
    <Panel label="Photonic bands"><Bands /></Panel><Arrow />
    <Panel label="Projected operators"><OperatorMatrix /></Panel>
  </div>;
}
function WarmArt() {
  return <div className="roadmap-warm-art">
    <Panel label="Configuration A"><Crystal radius={6} /></Panel>
    <div className="roadmap-warm-paths">
      <div className="roadmap-cold-path"><strong>Cold start</strong><span>New initial guess</span>
        <svg viewBox="0 0 200 16" aria-hidden="true"><path d="M2 8H193" /><path d="m187 2 7 6-7 6" /></svg>
      </div>
      <div className="roadmap-warm-path">
        <svg viewBox="0 0 200 16" aria-hidden="true"><path d="M2 8H193" /><path d="m187 2 7 6-7 6" /></svg>
        <span>Transfer the subspace</span><strong>Warm start</strong>
      </div>
    </div>
    <Panel label="Configuration B"><Crystal radius={11} stronger /></Panel>
  </div>;
}

const cases = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te'] as const;
const caseLabels = ['Sq. TM', 'Sq. TE', 'Tri. TM', 'Tri. TE'];
const benchmarkSeries = [
  { name: 'MPB', key: 'mpb', css: 'var(--series-reference)' },
  { name: 'Blaze f64', key: 'blazeFull', css: 'var(--series-primary)' },
  { name: 'Blaze mixed', key: 'blaze', css: 'var(--series-highlight)' },
] as const;
function BenchmarkArt() {
  const y = (ms: number) => 91 - ms / 3000 * 75;
  return <div className="roadmap-documentation-art">
    <Panel label="Website"><Globe2 className="roadmap-document-icon" aria-hidden="true" /></Panel>
    <Panel label="Theory"><div className="roadmap-paper-icons"><BookOpen aria-hidden="true" /><FileText aria-hidden="true" /></div></Panel>
    <div className="roadmap-mini-benchmark">
      <Art viewBox="0 0 300 116" label="Archived 16-thread runtime comparison in milliseconds for square and triangular TM and TE crystals. MPB, Blaze full precision, and Blaze mixed precision.">
        {[0, 1500, 3000].map(ms => <g key={ms}><path d={`M28 ${y(ms)}H294`} className="roadmap-art-grid" /><text x="23" y={y(ms) + 4} textAnchor="end" className="roadmap-art-ticks">{ms}</text></g>)}
        {cases.map((key, i) => <g key={key}>{benchmarkSeries.map((series, j) => {
          const value = benchmark[series.key][key], x = 42 + i * 64 + j * 13;
          return <g key={series.key} data-benchmark-case={key} data-benchmark-series={series.key}>
            <title>{`${caseLabels[i]}, ${series.name}: ${value.mean_ms.toFixed(1)} ± ${value.std_ms.toFixed(1)} ms`}</title>
            <rect x={x} y={y(value.mean_ms)} width="10" height={91 - y(value.mean_ms)} rx="1" fill={series.css} />
            <path d={`M${x + 5} ${y(value.mean_ms + value.std_ms)}V${y(value.mean_ms - value.std_ms)} M${x + 2} ${y(value.mean_ms + value.std_ms)}h6`} stroke={series.css} />
          </g>;
        })}<text x={58 + i * 64} y="109" textAnchor="middle" className="roadmap-art-ticks">{caseLabels[i]}</text></g>)}
      </Art>
      <div className="roadmap-mini-legend">{benchmarkSeries.map(series => <span key={series.key}><i style={{ background: series.css }} />{series.name}</span>)}</div>
    </div>
  </div>;
}
function GeometryArt() {
  return <div className="roadmap-geometry-art">
    <Panel label="Circular inclusion"><Art label="A single circular dielectric inclusion"><circle cx="90" cy="55" r="38" className="roadmap-shape-blue" /></Art></Panel>
    <Arrow long />
    <Panel label="Polygons, cutouts, and imported grids">
      <Art viewBox="0 0 330 110" label="A notched polygon, a hexagonal inclusion with a hexagonal hole, and an imported dielectric grid">
        <path d="M10 15H91V40H63V67H91V96H10V70H34V43H10Z" className="roadmap-shape-orange" />
        <path d="M160 7L204 32V82L160 107L116 82V32Z M160 31L183 44V70L160 83L137 70V44Z" fillRule="evenodd" className="roadmap-shape-blue" />
        {Array.from({ length: 64 }, (_, i) => {
          const x = i % 8, y = Math.floor(i / 8), solid = (x - 3.5) ** 2 + (y - 3.5) ** 2 > 7;
          return <rect key={i} x={230 + x * 11} y={12 + y * 11} width="9" height="9" rx="1" className={solid ? 'roadmap-fill-orange' : 'roadmap-fill-blue'} opacity={solid ? .7 : .2} />;
        })}
      </Art>
    </Panel>
  </div>;
}
function TargetBandsArt() {
  const points = bandData.TM.blaze_f64;
  const left = 34, right = 232, top = 22, bottom = 126;
  const x = (distance: number) => left + distance / points.at(-1)!.k_distance * (right - left);
  const y = (frequency: number) => bottom - frequency / 1.5 * (bottom - top);
  const bands = points[0].frequencies.map((_, band) => points.map(point => `${x(point.k_distance)},${y(point.frequencies[band])}`).join(' '));
  const windowTop = y(0.9), windowHeight = y(0.7) - windowTop;

  return <Art viewBox="0 0 250 153" className="roadmap-target-bands" label="Computed Blaze TM bands for a square lattice of dielectric rods, with an illustrative target at normalized frequency 0.8">
    <desc>Saved full-precision Blaze results for rods of radius 0.2a and dielectric constant 8.9. Ten bands follow the Gamma, X, M, Gamma path. The orange window highlights frequencies near the illustrative solver target.</desc>
    <defs><clipPath id="roadmap-frequency-window"><rect x={left} y={windowTop} width={right - left} height={windowHeight} /></clipPath></defs>
    <rect x={left} y={windowTop} width={right - left} height={windowHeight} className="roadmap-gap" />
    {[0, 0.5, 1, 1.5].map(value => <g key={value}>
      <path d={`M${left} ${y(value)}H${right}`} className="roadmap-art-grid" />
      <text x={left - 7} y={y(value) + 4} textAnchor="end" className="roadmap-art-ticks">{value}</text>
    </g>)}
    {[0, 15, 30, 45].map((index, i) => <g key={index}>
      <path d={`M${x(points[index].k_distance)} ${top}V${bottom}`} className="roadmap-art-grid" />
      <text x={x(points[index].k_distance)} y={bottom + 17} textAnchor="middle" className="roadmap-art-ticks">{['Γ', 'X', 'M', 'Γ'][i]}</text>
    </g>)}
    <path d={`M${left} ${top}V${bottom}H${right}`} className="roadmap-art-outline" />
    {bands.map((line, index) => <polyline key={index} points={line} className="roadmap-line-blue" />)}
    <g clipPath="url(#roadmap-frequency-window)">
      {bands.map((line, index) => <polyline key={index} points={line} className="roadmap-line-orange" />)}
    </g>
    <path d={`M${left} ${y(0.8)}H${right}`} className="roadmap-target-line" />
    <text x="10" y="12" className="roadmap-art-ticks">ωa / (2πc)</text>
    <text x={right} y="12" textAnchor="end" className="roadmap-target-label">ω₀ = 0.8</text>
  </Art>;
}

function SolverArt() {
  return <div className="roadmap-art-flow roadmap-solver-art">
    <Panel label="Profile the work"><Art label="Schematic work blocks for FFTs, material operations, and block algebra">
      {[['FFT', 85], ['ε', 54], ['Q†Q', 115]].map(([label, width], i) => <g key={label}>
        <text x="18" y={24 + i * 32} className="roadmap-art-text">{label}</text>
        <rect x="54" y={10 + i * 32} width={width} height="20" rx="3" className={i === 0 ? 'roadmap-fill-blue' : 'roadmap-fill-orange'} opacity={.85 - i * .16} />
      </g>)}
    </Art></Panel><Arrow />
    <Panel label="Tune iterations"><Art label="Schematic iterative refinement with residual checks">
      <path d="M28 22H143V81H28Z" className="roadmap-art-outline" strokeDasharray="3 4" />
      <path d="m135 15 9 7-9 7 M36 74l-9 7 9 7" className="roadmap-line-orange" />
      <Settings2 x="62" y="29" width="47" height="47" className="roadmap-stroke-blue" />
      <text x="87" y="105" textAnchor="middle" className="roadmap-art-text">Check residual</text>
    </Art></Panel><Arrow />
    <Panel label="Choose a frequency"><TargetBandsArt /></Panel>
  </div>;
}
function GradientArt() {
  return <div className="roadmap-art-flow">
    <Panel label="Vary the geometry"><Crystal radius={7} variable /></Panel><Arrow />
    <Panel label="Compute sensitivities"><Art label="A derivative measures how a frequency changes with a geometric parameter">
      <path d="M24 9V89H160" className="roadmap-art-grid" />
      <path d="M30 75C66 73 80 49 103 35S140 18 159 16" className="roadmap-line-blue" />
      <path d="M62 67L126 20" className="roadmap-line-orange" />
      <circle cx="96" cy="42" r="4" className="roadmap-fill-orange" />
      <text x="77" y="16" className="roadmap-art-text">∂ω/∂r</text><text x="159" y="105" className="roadmap-art-text">r</text>
    </Art></Panel><Arrow />
    <Panel label="Optimize a band gap"><Bands gap /></Panel>
  </div>;
}
function CompatibilityArt() {
  return <div className="roadmap-art-flow">
    <Panel label="MPB calculation"><div className="roadmap-code-art"><FileCode2 aria-hidden="true" /><span>geometry<br />k_points<br />num_bands</span></div></Panel><Arrow />
    <Panel label="Preserve meaning"><Art label="Match geometry, units, and field normalization during conversion">
      <rect x="26" y="12" width="126" height="86" rx="5" className="roadmap-art-outline" />
      {['Geometry', 'Units', 'Fields'].map((label, i) => <g key={label}>
        <path d={`m38 ${30 + i * 26} 4 4 7-9`} className="roadmap-line-orange" />
        <text x="58" y={34 + i * 26} className="roadmap-art-text">{label}</text>
      </g>)}
    </Art></Panel><Arrow />
    <Panel label="Blaze outputs"><div className="roadmap-code-art"><FileText aria-hidden="true" /><span>Bands<br />Fields · HDF5<br />Metadata</span></div></Panel>
  </div>;
}
function GpuArt() {
  return <div className="roadmap-gpu-art">
    <Panel label="Set up once"><Microchip className="roadmap-document-icon" aria-hidden="true" /></Panel><Arrow />
    <div className="roadmap-device">
      <strong>GPU memory</strong>
      <div className="roadmap-device-steps"><span>FFT</span><span>Material</span><span>Block algebra</span></div>
      <svg viewBox="0 0 300 20" aria-hidden="true"><path d="M285 2V13H15V2 M9 8l6-6 6 6" /></svg>
      <span>Fields stay on the device</span>
    </div><Arrow />
    <Panel label="Export datasets"><Art label="A stack of band diagrams from a parameter sweep">
      {[0, 1, 2].map(i => <g key={i} transform={`translate(${i * 11},${-i * 9})`}>
        <rect x="23" y="32" width="112" height="67" rx="4" className="roadmap-dataset-sheet" />
        <path d="M34 84Q64 49 121 68 M34 65Q78 40 121 50" className={i === 2 ? 'roadmap-line-orange' : 'roadmap-line-blue'} />
      </g>)}
    </Art></Panel>
  </div>;
}
function ThreeDArt() {
  return <div className="roadmap-3d-art">
    <Panel label="Periodic 3D geometry">
      <Image src={getAssetPath('/images/roadmap/woodpile-crystal.webp')} width={960} height={720}
        className="roadmap-art roadmap-woodpile-image" loading="lazy"
        alt="3D woodpile photonic crystal with stacked square-section dielectric rods. Blue and orange distinguish perpendicular rod orientations." />
    </Panel><Arrow />
    <Panel label="Coupled vector fields"><Art label="Three vector components and a divergence-free field condition">
      <path d="M76 80L25 57 M76 80L139 67 M76 80V12" className="roadmap-art-outline" />
      <path d="m29 65-4-8 10-1 M130 63l9 4-7 6 M70 20l6-8 6 8" className="roadmap-art-outline" />
      <path d="M77 79Q63 53 50 66T26 57 M77 78Q99 47 109 66T138 67" className="roadmap-line-blue" />
      <path d="M76 78Q55 64 76 49T76 15" className="roadmap-line-orange" />
      <text x="14" y="47" className="roadmap-art-text">H<tspan baselineShift="sub" fontSize="10">x</tspan></text>
      <text x="142" y="64" className="roadmap-art-text">H<tspan baselineShift="sub" fontSize="10">y</tspan></text>
      <text x="83" y="17" className="roadmap-art-text">H<tspan baselineShift="sub" fontSize="10">z</tspan></text>
      <text x="87" y="105" textAnchor="middle" className="roadmap-art-text">∇ · H = 0</text>
    </Art></Panel>
  </div>;
}
function SlabArt() {
  // Project every hole and radiation anchor through the same slab-plane basis.
  const onSlab = (u: number, v: number) => [20 + 115 * u + 55 * v, 75 + 25 * u - 40 * v];
  return <div className="roadmap-slab-art">
    <Panel label="A patterned film"><Art viewBox="0 0 210 140" label="A finite-thickness slab with a centered three by three hole array and two arrows pointing away from its surface">
      <path d="M20 75L135 100L190 60V74L135 114L20 89Z" className="roadmap-shape-blue" />
      <path d="M20 75L75 35L190 60L135 100Z M135 100V114" className="roadmap-shape-blue" />
      <g transform="matrix(115 25 55 -40 20 75)">
        {Array.from({ length: 9 }, (_, i) => <circle key={i} cx={0.24 + (i % 3) * 0.26} cy={0.24 + Math.floor(i / 3) * 0.26}
          r="0.065" className="roadmap-hole" vectorEffect="non-scaling-stroke" />)}
      </g>
      {[0.35, 0.66].map(u => {
        const [x, y] = onSlab(u, 0.66);
        return <g key={u} className="roadmap-slab-radiation">
          <circle cx={x} cy={y} r="1.7" className="roadmap-fill-orange" />
          <path d={`M${x} ${y - 3}V${y - 37}`} className="roadmap-line-orange" strokeDasharray="3 3" />
          <path d={`M${x - 4} ${y - 31}l4 -6 4 6`} className="roadmap-line-orange" />
        </g>;
      })}
    </Art></Panel><Arrow />
    <Panel label="Vertical confinement"><Art viewBox="0 0 190 140" label="A schematic field profile localized in the slab, decaying into the cladding">
      <rect x="43" y="50" width="118" height="34" className="roadmap-gap" />
      <path d="M40 18V112H166 M40 50H166 M40 84H166" className="roadmap-art-grid" />
      <path d="M45 21C45 36 50 41 67 50C99 58 140 60 140 67C140 74 99 76 67 84C50 93 45 96 45 108" className="roadmap-line-blue" />
      <text x="24" y="25" className="roadmap-art-text">z</text>
      <text x="103" y="134" textAnchor="middle" className="roadmap-art-text">|H(z)|</text>
    </Art></Panel>
  </div>;
}

const illustrations: Record<Exclude<ModuleId, 'symmetries'>, () => ReactNode> = {
  core: CoreArt, warm: WarmArt, evidence: BenchmarkArt, geometry: GeometryArt,
  solver: SolverArt, gradients: GradientArt, compatibility: CompatibilityArt,
  gpu: GpuArt, maxwell3d: ThreeDArt, slabs: SlabArt,
};

export default function ModuleBanner({ module }: { module: ModuleId }) {
  if (module === 'symmetries') return <SymmetryPreview />;
  return <figure className="roadmap-banner" data-visual-for={module}>
    {illustrations[module]()}
  </figure>;
}
