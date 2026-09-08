'use client';
import Link from 'next/link';
import { MultiCorePerformanceChart } from '../../components/charts';

export default function SpeedComparison() {
  return <div style={{ width: '100%', maxWidth: 850 }}>
    <MultiCorePerformanceChart width={850} height={430} />
    <p style={{ color: 'var(--site-muted)', maxWidth: '70ch', margin: '1rem auto', lineHeight: 1.7 }}>
      Historical 16-thread measurements for square and triangular crystal configurations.
      Timings depend on numerical settings and the execution environment.
      See the <Link href="/blaze" style={{ color: 'var(--site-accent)' }}>technical report</Link> for
      the recorded conditions and limits of this comparison.
    </p>
  </div>;
}
