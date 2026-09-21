'use client';

import { useMemo } from 'react';
import BarChart, { BarDataPoint } from './BarChart';
import { useSeries5Benchmarks } from '@/lib/use-benchmarks';
import ChartPlaceholder from './ChartPlaceholder';
import { series, theme } from '@/lib/theme';

const COLORS = {
  MPB: series.reference,     // Blue-gray
  Blaze: series.highlight,   // Light blue-white
};

interface MemoryUsageChartProps {
  width?: number;
  height?: number;
  /** Which sweep to display: 'resolution' or 'num_bands' */
  sweep: 'resolution' | 'num_bands';
}

export default function MemoryUsageChart({
  width = 650,
  height = 420,
  sweep,
}: MemoryUsageChartProps) {
  const { data: benchmarkData, loading } = useSeries5Benchmarks();

  const sweepData = benchmarkData[sweep];
  const sweepLabel = sweep === 'resolution' ? 'Resolution' : 'Number of Bands';

  // Transform data for grouped bar chart
  // Each value is a position on the x-axis, with TM and TE as pairs of MPB/Blaze
  const { tmData, teData } = useMemo(() => {
    const tm: BarDataPoint[] = [];
    const te: BarDataPoint[] = [];

    for (const val of sweepData.values) {
      const mpbTM = sweepData.mpb.TM.find(d => d.value === val);
      const blazeTM = sweepData.blaze.TM.find(d => d.value === val);
      const mpbTE = sweepData.mpb.TE.find(d => d.value === val);
      const blazeTE = sweepData.blaze.TE.find(d => d.value === val);

      if (mpbTM) {
        tm.push({
          id: `${val}`,
          label: `${val}`,
          value: mpbTM.memory_mb,
          std: mpbTM.memory_mb_std,
          group: 'MPB',
        });
      }
      if (blazeTM) {
        tm.push({
          id: `${val}`,
          label: `${val}`,
          value: blazeTM.memory_mb,
          std: blazeTM.memory_mb_std,
          group: 'Blaze',
        });
      }

      if (mpbTE) {
        te.push({
          id: `${val}`,
          label: `${val}`,
          value: mpbTE.memory_mb,
          std: mpbTE.memory_mb_std,
          group: 'MPB',
        });
      }
      if (blazeTE) {
        te.push({
          id: `${val}`,
          label: `${val}`,
          value: blazeTE.memory_mb,
          std: blazeTE.memory_mb_std,
          group: 'Blaze',
        });
      }
    }

    return { tmData: tm, teData: te };
  }, [sweepData]);


  if (loading) {
    return (
      <ChartPlaceholder width={width} height={height} />
    );
  }

  const chartWidth = (width - 20) / 2;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%', maxWidth: width, minWidth: 0 }}>
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap', width: '100%', minWidth: 0 }}>
        <BarChart
          data={tmData}
          width={chartWidth}
          height={height}
          title={`Peak Memory vs ${sweepLabel} (TM)`}
          yLabel="Peak Memory (MB)"
          yTickFormat={(v) => `${v.toFixed(0)}`}
          valueFormat={(v) => `${v.toFixed(0)}`}
          labelAngle={0}
          showValues={false}
          showStd={true}
          groupColors={COLORS}
          caption=""
          margin={{ top: 60, right: 30, bottom: 50, left: 75 }}
        />
        <BarChart
          data={teData}
          width={chartWidth}
          height={height}
          title={`Peak Memory vs ${sweepLabel} (TE)`}
          yLabel="Peak Memory (MB)"
          yTickFormat={(v) => `${v.toFixed(0)}`}
          valueFormat={(v) => `${v.toFixed(0)}`}
          labelAngle={0}
          showValues={false}
          showStd={true}
          groupColors={COLORS}
          caption=""
          margin={{ top: 60, right: 30, bottom: 50, left: 75 }}
        />
      </div>
      <p style={{
        marginTop: '1rem',
        fontSize: '0.875rem',
        color: theme.textSubtle,
        lineHeight: 1.5,
        fontFamily: 'var(--font-sans), system-ui, sans-serif',
        fontStyle: 'italic',
        textAlign: 'left',
      }}>
      </p>
    </div>
  );
}
