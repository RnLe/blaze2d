import type { Applied } from '@/lib/compute/controller';
import { ArrowRight, Layers3, MapPin, Waves } from 'lucide-react';

export function StudyOverview({ applied }: { applied: Applied }) {
  const config = applied.report.config!, resolved = applied.report.resolved!, summary = applied.report.summary!;
  const points = resolved.k_points_cartesian;
  const extent = points.reduce((maximum, point) => Math.max(maximum, Math.abs(point[0]), Math.abs(point[1])), 1e-10);
  const stride = Math.max(1, Math.ceil(points.length / 1000));
  const shown = points.filter((_, index) => index % stride === 0 || index === points.length - 1 || resolved.k_label_indices.includes(index));
  const x = (value: number) => 210 + value / extent * 120, y = (value: number) => 155 - value / extent * 115;
  const labels = new Map<number, string>();
  resolved.k_label_indices.forEach((index, i) => { if (!labels.has(index)) labels.set(index, resolved.k_labels[i]); });
  return <div className="wb-study-overview">
    <span className="wb-eyebrow">Calculation plan</span><h1>{config.task === 'bands' ? 'Follow the bands' : 'Inspect the operators'}</h1>
    <p>{config.task === 'bands' ? 'Solve the periodic Maxwell equation at each point along this reciprocal-space path.' : 'Extract projected operators at the carrier wavevector and any requested stencil points.'}</p>
    <div className="wb-plan-stats"><div><Waves size={18} /><strong>{resolved.solved_bands}</strong><span>Solved bands</span></div><div><Layers3 size={18} /><strong>{summary.jobs}</strong><span>Configurations</span></div><div><MapPin size={18} /><strong>{summary.solves}</strong><span>Eigensolver calls</span></div></div>
    <figure className="wb-path-figure"><svg viewBox="0 0 440 310" role="img" aria-label={config.task === 'bands' ? 'Planned path in reciprocal Cartesian coordinates' : 'Carrier in reciprocal Cartesian coordinates'}>
      <path d="M40 155H398M210 278V20" className="wb-path-axes" /><text x="400" y="174">kₓ</text><text x="220" y="26">kᵧ</text>
      <polyline points={shown.map(point => `${x(point[0])},${y(point[1])}`).join(' ')} className="wb-path-line" />
      {(labels.size ? [...labels] : [[0, 'carrier']] as [number, string][]).map(([index, label]) => points[index] && <g key={index}><circle cx={x(points[index][0])} cy={y(points[index][1])} r="4" /><text x={x(points[index][0]) + 9} y={y(points[index][1]) - 10}>{label}</text></g>)}
    </svg><figcaption>{config.task === 'bands' ? (resolved.k_labels.length ? resolved.k_labels.join(' → ') : 'Custom path') : `Carrier: ${config.operators!.k_point.value.join(', ')} (${config.operators!.k_point.basis.replaceAll('_', ' ')})`} · Cartesian angular coordinates</figcaption></figure>
    {!!config.sweeps?.length && <section><h2>Parameter sweeps</h2><p className="wb-muted">The last axis varies fastest.</p>{config.sweeps.map(sweep => <div className="wb-sweep-summary" key={sweep.name}><strong>{sweep.name}</strong><ArrowRight size={14} /><code>{sweep.target}</code><span>{sweep.linspace ? `${sweep.linspace.start} to ${sweep.linspace.stop}, ${sweep.linspace.count} values` : JSON.stringify(sweep.values)}</span></div>)}</section>}
    {config.operators && <section><h2>Requested data</h2><div className="wb-example-tags">{config.operators.quantities?.map(quantity => <span key={quantity}>{quantity.replaceAll('_', ' ')}</span>)}</div>
      <p className="wb-muted">{config.operators.retained_bands} retained bands from index {config.operators.band_lo}, with {config.operators.remote_bands} upper remote bands.</p></section>}
    <p className="wb-plan-budget">{resolved.resolution.join(' × ')} grid · f64 · estimated peak memory {Math.ceil(summary.estimated_peak_bytes / 1048576)} MiB</p>
  </div>;
}
