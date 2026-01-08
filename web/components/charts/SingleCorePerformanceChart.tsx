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
  MPB: '#5477c4',                       // Blue-gray
  'Blaze (Full Precision)': '#a3befa',  // Slightly darker blue (full precision)
  'Blaze (Mixed Precision)': '#eaf1fe', // Light blue-white (mixed precision)
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
  // Each config is a position on the x-axis, with MPB, Blaze f64, and Blaze as grouped bars
  // The 'id' field identifies the x-position, 'label' is what's displayed
  const data: BarDataPoint[] = useMemo(() => {
    const hasFullPrecision = benchmarkData.metadata?.hasFullPrecision && 
      Object.keys(benchmarkData.blazeFull || {}).length > 0;
    
    return configs.flatMap(config => {
      const bars: BarDataPoint[] = [
        {
          id: config,
          label: CONFIG_LABELS[config],
          value: benchmarkData.mpb[config]?.mean_ms || 0,
          std: benchmarkData.mpb[config]?.std_ms || 0,
          group: 'MPB',
        },
      ];
      
      // Add full precision bar first if data is available
      if (hasFullPrecision && benchmarkData.blazeFull?.[config]) {
        bars.push({
          id: config,
          label: CONFIG_LABELS[config],
          value: benchmarkData.blazeFull[config]?.mean_ms || 0,
          std: benchmarkData.blazeFull[config]?.std_ms || 0,
          group: 'Blaze (Full Precision)',
        });
      }
      
      // Then add mixed precision
      bars.push({
        id: config,
        label: CONFIG_LABELS[config],
        value: benchmarkData.blaze[config]?.mean_ms || 0,
        std: benchmarkData.blaze[config]?.std_ms || 0,
        group: 'Blaze (Mixed Precision)',
      });
      
      return bars;
    });
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

  const hasFullPrecision = benchmarkData.metadata?.hasFullPrecision && 
    Object.keys(benchmarkData.blazeFull || {}).length > 0;
  
  const captionText = hasFullPrecision
    ? `Blaze achieves ${avgSpeedup}× average speedup over MPB on single-core workloads. Mixed precision (f32/f64) shown alongside full precision (f64).`
    : `Blaze achieves ${avgSpeedup}× average speedup over MPB on single-core workloads in its native mixed-precision mode.`;

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
      caption={captionText}
      margin={{ top: 60, right: 30, bottom: 70, left: 75 }}
    />
  );
}
