import { ArrowRight } from 'lucide-react';
import { apply, eigenvalues2, inverse, multiply, symmetric, type Mat2, type Vec2 } from './linear-algebra';

const A = symmetric(0.42, 2.6, 0.5);
/** A cheap approximation of A whose inverse serves as T. */
const APPROXIMATION = symmetric(0.5, 2.3, 0.42);
const T = inverse(APPROXIMATION.matrix)!;
const TA = multiply(T, A.matrix);

function condition(m: Mat2) {
  const [low, high] = eigenvalues2(m);
  return high / low;
}

/** Spokes from the unit circle to its image under `m`. */
function Panel({ m, color, label }: { m: Mat2; color: string; label: string }) {
  const [cx, cy, radius] = [260, 190, 64];
  const count = 48;
  const angles = Array.from({ length: count }, (_, i) => (i / count) * 2 * Math.PI);
  const point = ([u, v]: Vec2): Vec2 => [Math.round((cx + u * radius) * 100) / 100, Math.round((cy - v * radius) * 100) / 100];
  const image = angles.map(phi => point(apply(m, [Math.cos(phi), Math.sin(phi)])));
  const circle = angles.map(phi => point([Math.cos(phi), Math.sin(phi)]));
  return <svg viewBox="0 0 520 380" role="img" aria-label={label}>
    <circle cx={cx} cy={cy} r={radius} className="pc-hero-circle" />
    {circle.map((start, i) => <line key={i} x1={start[0]} y1={start[1]} x2={image[i][0]} y2={image[i][1]} stroke={color} className="pc-hero-spoke" />)}
    <polygon points={image.map(p => p.join(',')).join(' ')} fill={color} className="pc-hero-image" stroke={color} />
    {image.filter((_, i) => i % 4 === 0).map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3.2" fill={color} />)}
  </svg>;
}

export default function PreconditionerHero() {
  const kA = condition(A.matrix), kTA = condition(TA);
  return <figure className="pc-hero">
    <div className="pc-hero-panel">
      <Panel m={A.matrix} color="var(--pc-image)" label={`A matrix A maps the unit circle to an elongated ellipse; condition number ${kA.toFixed(1)}.`} />
      <p><strong>A stretches the unit circle</strong><span>condition number κ(A) = {kA.toFixed(1)}</span></p>
    </div>
    <div className="pc-hero-operator" aria-hidden="true">
      <code>multiply by T</code>
      <ArrowRight size={28} />
      <span>a cheap approximate inverse</span>
    </div>
    <div className="pc-hero-panel">
      <Panel m={TA} color="var(--pc-preconditioned)" label={`After a cheap approximate inverse T, the product TA maps the circle to a nearly round shape; condition number ${kTA.toFixed(2)}.`} />
      <p><strong>TA is nearly round again</strong><span>κ(TA) = {kTA.toFixed(2)} with T ≈ A⁻¹</span></p>
    </div>
  </figure>;
}
