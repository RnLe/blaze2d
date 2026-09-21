'use client';
import Link from 'next/link';
import { RotateCcw } from 'lucide-react';
import { useMultiCoreBenchmarks } from '@/lib/use-benchmarks';
import { useSequence, phase, smooth } from './useSequence';
import { useSize } from '@/components/workbench/useSize';
import { series, theme } from '@/lib/theme';
import { useUiScale } from '@/lib/use-ui-scale';

const configurations = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te'];
const labels = ['Square TM','Square TE','Triangular TM','Triangular TE'];
export default function SpeedComparison() {
  const { data, loading, source } = useMultiCoreBenchmarks();
  const { ref, elapsed, replay } = useSequence(5400, !loading);
  const { ref: sizeRef, width } = useSize<HTMLDivElement>();
  const ui = useUiScale();
  const w = Math.max(280, width / ui), mobile = w < 520, left = mobile ? 46 : 66, right = w - 14, top = 44, bottom = mobile ? 285 : 335;
  const gap = (right - left) / configurations.length, barWidth = Math.min(37, gap / 5);
  const bars = [{ name: 'MPB', color: series.reference, values: data.mpb, start: 150, duration: 2000 },
    ...(data.blazeFull ? [{ name: 'Blaze f64', color: series.primary, values: data.blazeFull, start: 2500, duration: 900 }] : []),
    { name: 'Blaze mixed', color: series.highlight, values: data.blaze, start: 3600, duration: 900 }];
  const max = Math.ceil(Math.max(...bars.flatMap(item => configurations.map(key => (item.values[key]?.mean_ms ?? 0) + (item.values[key]?.std_ms ?? 0)))) / 500) * 500;
  const scale = (bottom - top) / max, maxSpeedup = Math.max(...configurations.map(key => data.mpb[key].mean_ms / data.blaze[key].mean_ms));
  return <div className="pitch-speed" ref={ref}>
    <div className="pitch-speed-headline"><strong>{maxSpeedup.toFixed(1)}<span>×</span></strong><div>faster in the best recorded workload</div></div>
    <div className="pitch-plot-controls"><span>Time per configuration <span className="pitch-unit">(ms, lower is faster)</span></span><button className="pitch-replay" onClick={replay}><RotateCcw size={14} />Replay</button></div>
    <div className="pitch-legend">{bars.map(item => <span key={item.name} style={{ color: item.color }}>■ {item.name}</span>)}</div>
    <div className="pitch-speed-stage" ref={sizeRef}><svg width={w * ui} height={(bottom + 74) * ui} viewBox={`0 0 ${w} ${bottom + 74}`} role="img" aria-label="Recorded MPB and Blaze computation times. MPB bars rise first, followed by Blaze f64 and mixed precision.">
      {Array.from({ length: 6 }, (_, index) => max * index / 5).map(value => <g key={value}><line x1={left} x2={right} y1={bottom - value * scale} y2={bottom - value * scale} stroke="var(--series-grid)" /><text x={left - 11} y={bottom - value * scale + 6} textAnchor="end" fontSize={16} fill={theme.textPrimary}>{value}</text></g>)}
      {configurations.map((key, i) => <g key={key}>
        {bars.map((item, j) => {
          const value = item.values[key]?.mean_ms ?? 0, std = item.values[key]?.std_ms ?? 0, progress = smooth(phase(elapsed, item.start + i * 65, item.duration));
          const x = left + gap * (i + .5) + (j - (bars.length - 1) / 2) * (barWidth + 3) - barWidth / 2, y = bottom - value * scale * progress;
          return <g key={item.name} data-series={item.name}><title>{`${labels[i]} · ${item.name}: ${value.toFixed(1)} ± ${std.toFixed(1)} ms`}</title>
            <rect x={x} y={y} width={barWidth} height={bottom - y} rx={3} fill={item.color} fillOpacity={j === 0 ? .73 : .95} />
            {progress > .97 && <g stroke={item.color} strokeWidth={1}><line x1={x + barWidth/2} x2={x + barWidth/2} y1={y - std * scale} y2={y + std * scale} /><line x1={x + barWidth/2 - 4} x2={x + barWidth/2 + 4} y1={y - std * scale} y2={y - std * scale} /></g>}
            {progress === 1 && !mobile && <text x={x + barWidth/2} y={y - Math.max(11, std * scale + 8)} textAnchor="middle" fontSize={14} fill={item.color}>{Math.round(value)}</text>}
          </g>;
        })}
        <text x={left + gap * (i + .5)} y={bottom + 27} textAnchor="middle" fontSize={mobile ? 13 : 16} fill={theme.textPrimary}><tspan x={left + gap * (i + .5)}>{labels[i].split(' ')[0]}</tspan><tspan x={left + gap * (i + .5)} dy={21}>{labels[i].split(' ')[1]}</tspan></text>
      </g>)}
    </svg></div>
    <p className="pitch-speed-conditions">Mixed precision · 16-thread parameter sweeps</p>
    <details className="pitch-benchmark-details"><summary>Recorded times and benchmark conditions</summary>
      <div className="scroll-x"><table><thead><tr><th>Configuration</th>{bars.map(item => <th key={item.name}>{item.name} (ms)</th>)}</tr></thead><tbody>{configurations.map((key, index) => <tr key={key}><th>{labels[index]}</th>{bars.map(item => <td key={item.name}>{item.values[key]?.mean_ms.toFixed(1)} ± {item.values[key]?.std_ms.toFixed(1)}</td>)}</tr>)}</tbody></table></div>
      <p>Historical 16-thread measurements, 64 × 64 grids, eight bands. MPB uses threading within each solve; Blaze schedules independent configurations. Error bars show the recorded standard deviation. {source === 'fallback' && 'The archived backup dataset is shown.'} <Link href="/blaze#multi-core-performance">Full conditions and limitations ↗</Link></p>
    </details>
  </div>;
}
