'use client';

import { useMemo } from 'react';
import BarChart, { BarDataPoint } from './BarChart';
import { useSingleCoreBenchmarks } from '../../lib/use-benchmarks';

const CONFIG_LABELS: Record<string, string> = {
  config_a_tm: 'TM',
  config_a_te: 'TE',
  config_b_tm: 'TM',
  config_b_te: 'TE',
};

// Full labels for caption
const CONFIG_FULL_LABELS: Record<string, string> = {
  config_a_tm: 'Square TM',
  config_a_te: 'Square TE',
  config_b_tm: 'Hex TM',
  config_b_te: 'Hex TE',
};

const COLORS = {
  MPB: '#5477c4',     // Blue-gray
  Blaze: '#eaf1fe',   // Light blue-white
};

interface SingleCoreChartProps {
  width?: number;
  height?: number;
}

export default function SingleCorePerformanceChart({
  width = 650,
  height = 420,
}: SingleCoreChartProps) {
  const { data: benchmarkData, loading, source } = useSingleCoreBenchmarks();

  const configs = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te'];

  // Transform data for grouped bar chart
  // Each config is a position on the x-axis, with MPB and Blaze as grouped bars at each position
  // The 'id' field identifies the x-position, 'label' is what's displayed
  const data: BarDataPoint[] = useMemo(() => {
    return configs.flatMap(config => [
      {
        id: config,  // Position identifier (same for both MPB and Blaze at this position)
        label: CONFIG_LABELS[config],  // Display label (TM or TE)
        value: benchmarkData.mpb[config]?.mean_ms || 0,
        std: benchmarkData.mpb[config]?.std_ms || 0,
        group: 'MPB',
      },
      {
        id: config,  // Same position identifier
        label: CONFIG_LABELS[config],  // Display label (TM or TE)
        value: benchmarkData.blaze[config]?.mean_ms || 0,
        std: benchmarkData.blaze[config]?.std_ms || 0,
        group: 'Blaze',
      },
    ]);
  }, [benchmarkData]);

  // Calculate speedups for caption
  const { speedups, avgSpeedup } = useMemo(() => {
    const speeds = configs.map(config => {
      const mpb = benchmarkData.mpb[config]?.mean_ms || 1;
      const blaze = benchmarkData.blaze[config]?.mean_ms || 1;
      return (mpb / blaze).toFixed(1);
    });

    const avg = (
      configs.reduce((sum, config) => {
        const mpb = benchmarkData.mpb[config]?.mean_ms || 1;
        const blaze = benchmarkData.blaze[config]?.mean_ms || 1;
        return sum + mpb / blaze;
      }, 0) / configs.length
    ).toFixed(1);

    return { speedups: speeds, avgSpeedup: avg };
  }, [benchmarkData]);

  // Category brackets for Square and Hex groupings (using label indices)
  // Labels are: TM(0), TE(1), TM(2), TE(3) -> Square covers 0-1, Hex covers 2-3
  const categoryBrackets = [
    { labelIndices: [0, 1] as [number, number], category: 'Square' },
    { labelIndices: [2, 3] as [number, number], category: 'Hexagonal' },
  ];

  // Show loading state
  if (loading) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>
        Loading benchmark data...
      </div>
    );
  }

  // Data source indicator for caption
  const sourceNote = source === 'static' 
    ? '' 
    : ' (fallback data)';

  return (
    <BarChart
      data={data}
      width={width}
      height={height}
      title="Single-Core Performance"
      yLabel="Time per job (ms)"
      yTickFormat={(v) => `${v.toFixed(0)}`}
      valueFormat={(v) => `${v.toFixed(0)}`}
      labelAngle={0}
      showValues={true}
      showStd={true}
      groupColors={COLORS}
      categoryBrackets={categoryBrackets}
      showCategoryBrackets={true}
      bracketOffset={28}
      caption={`Blaze achieves ${avgSpeedup}× average speedup over MPB on single-core workloads.`}
      margin={{ top: 60, right: 30, bottom: 70, left: 75 }}
    />
  );
}
