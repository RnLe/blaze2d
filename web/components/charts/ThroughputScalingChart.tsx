'use client';

import { useMemo } from 'react';
import LineChart, { LineSeries } from './LineChart';
import { useSeries7Benchmarks } from '@/lib/use-benchmarks';
import ChartPlaceholder from './ChartPlaceholder';
import { series, theme } from '@/lib/theme';

// Colors matching the bar chart style
const COLORS = {
  Blaze: series.highlight,       // Light blue-white (same as bar charts)
  MPB: series.reference,         // Blue-gray (MPB OMP - same as bar charts)
  MPB_Multiproc: series.primary, // Light blue for MPB Multiprocessor
};

interface ThroughputScalingChartProps {
  width?: number;
  height?: number;
}

export default function ThroughputScalingChart({
  width = 650,
  height = 380,
}: ThroughputScalingChartProps) {
  const { data: benchmarkData, loading } = useSeries7Benchmarks();

  // Transform data for line chart - two plots side by side
  const { lowResSeries, highResSeries } = useMemo(() => {
    const lowRes: LineSeries[] = [
      {
        id: 'blaze-low',
        label: 'Blaze',
        color: COLORS.Blaze,
        marker: 'circle',
        data: benchmarkData.low.blaze.map(d => ({
          x: d.threads,
          y: d.mean_throughput,
          std: d.std_throughput,
        })),
      },
      {
        id: 'mpb-omp-low',
        label: 'MPB (OpenMP)',
        color: COLORS.MPB,
        marker: 'square',
        data: benchmarkData.low.mpb_omp.map(d => ({
          x: d.threads,
          y: d.mean_throughput,
          std: d.std_throughput,
        })),
      },
      {
        id: 'mpb-multiproc-low',
        label: 'MPB (Multiprocessor)',
        color: COLORS.MPB_Multiproc,
        marker: 'triangle',
        data: benchmarkData.low.mpb_multiproc.map(d => ({
          x: d.threads,
          y: d.mean_throughput,
          std: d.std_throughput,
        })),
      },
    ];

    const highRes: LineSeries[] = [
      {
        id: 'blaze-high',
        label: 'Blaze',
        color: COLORS.Blaze,
        marker: 'circle',
        data: benchmarkData.high.blaze.map(d => ({
          x: d.threads,
          y: d.mean_throughput,
          std: d.std_throughput,
        })),
      },
      {
        id: 'mpb-omp-high',
        label: 'MPB (OpenMP)',
        color: COLORS.MPB,
        marker: 'square',
        data: benchmarkData.high.mpb_omp.map(d => ({
          x: d.threads,
          y: d.mean_throughput,
          std: d.std_throughput,
        })),
      },
      {
        id: 'mpb-multiproc-high',
        label: 'MPB (Multiprocessor)',
        color: COLORS.MPB_Multiproc,
        marker: 'triangle',
        data: benchmarkData.high.mpb_multiproc.map(d => ({
          x: d.threads,
          y: d.mean_throughput,
          std: d.std_throughput,
        })),
      },
    ];

    return { lowResSeries: lowRes, highResSeries: highRes };
  }, [benchmarkData]);

  if (loading) {
    return (
      <ChartPlaceholder width={width} height={height} />
    );
  }

  const chartWidth = (width - 20) / 2;
  const lowResolution = benchmarkData.low.resolution;
  const highResolution = benchmarkData.high.resolution;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%', maxWidth: width, minWidth: 0 }}>
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap', width: '100%', minWidth: 0 }}>
        <LineChart
          series={lowResSeries}
          width={chartWidth}
          height={height}
          title={`Throughput Scaling (${lowResolution}×${lowResolution})`}
          xLabel="Thread Count"
          yLabel="Throughput (jobs/s)"
          xTickFormat={(v) => `${v}`}
          yTickFormat={(v) => `${v.toFixed(0)}`}
          showErrorBars={true}
          xDomain={[0, 17]}
          margin={{ top: 50, right: 20, bottom: 60, left: 70 }}
        />
        <LineChart
          series={highResSeries}
          width={chartWidth}
          height={height}
          title={`Throughput Scaling (${highResolution}×${highResolution})`}
          xLabel="Thread Count"
          yLabel="Throughput (jobs/s)"
          xTickFormat={(v) => `${v}`}
          yTickFormat={(v) => `${v.toFixed(2)}`}
          showErrorBars={true}
          xDomain={[0, 17]}
          margin={{ top: 50, right: 20, bottom: 60, left: 70 }}
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
        Throughput scaling with thread count at low and high resolutions. Blaze maintains strong scaling across thread counts, while MPB (OpenMP) shows minimal parallel efficiency. MPB multiprocessor mode scales but with higher overhead.
      </p>
    </div>
  );
}
