'use client';

import { useMemo } from 'react';
import BarChart, { type BarDataPoint } from './BarChart';
import ChartPlaceholder from './ChartPlaceholder';
import { series } from '@/lib/theme';
import type { MultiCoreBenchmarkData, SingleCoreBenchmarkData } from '@/lib/benchmark-data';
import type { Benchmark } from '@/lib/use-benchmarks';

/**
 * Grouped MPB-versus-Blaze timing comparison.
 *
 * Shared by the single-core and multi-core sections of the technical report;
 * the two differ only in which dataset they read and how they describe the
 * workload.
 */

const CONFIGS = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te'] as const;
const POLARIZATION: Record<string, string> = {
  config_a_tm: 'TM',
  config_a_te: 'TE',
  config_b_tm: 'TM',
  config_b_te: 'TE',
};

const GROUPS = {
  MPB: 'MPB',
  FULL: 'Blaze (Full Precision)',
  MIXED: 'Blaze (Mixed Precision)',
} as const;

const COLORS = {
  [GROUPS.MPB]: series.reference,
  [GROUPS.FULL]: series.primary,
  [GROUPS.MIXED]: series.highlight,
};

// Labels run TM, TE, TM, TE; the first pair is the square lattice, the second
// the hexagonal one.
const CATEGORY_BRACKETS = [
  { labelIndices: [0, 1] as [number, number], category: 'Square' },
  { labelIndices: [2, 3] as [number, number], category: 'Hexagonal' },
];

export interface CorePerformanceChartProps {
  benchmark: Benchmark<SingleCoreBenchmarkData | MultiCoreBenchmarkData>;
  title: string;
  /** How the workload is described in the caption, e.g. "single-core". */
  workload: string;
  width?: number;
  height?: number;
}

export default function CorePerformanceChart({
  benchmark,
  title,
  workload,
  width = 650,
  height = 420,
}: CorePerformanceChartProps) {
  const { data: benchmarkData, loading } = benchmark;

  const hasFullPrecision = Boolean(
    benchmarkData.metadata?.hasFullPrecision && Object.keys(benchmarkData.blazeFull ?? {}).length,
  );

  const data: BarDataPoint[] = useMemo(
    () =>
      CONFIGS.flatMap(config => {
        const bar = (group: string, entry?: { mean_ms: number; std_ms: number }): BarDataPoint => ({
          id: config,
          label: POLARIZATION[config],
          value: entry?.mean_ms ?? 0,
          std: entry?.std_ms ?? 0,
          group,
        });
        return [
          bar(GROUPS.MPB, benchmarkData.mpb[config]),
          ...(hasFullPrecision && benchmarkData.blazeFull?.[config]
            ? [bar(GROUPS.FULL, benchmarkData.blazeFull[config])]
            : []),
          bar(GROUPS.MIXED, benchmarkData.blaze[config]),
        ];
      }),
    [benchmarkData, hasFullPrecision],
  );

  const averageSpeedup = useMemo(() => {
    const total = CONFIGS.reduce(
      (sum, config) => sum + (benchmarkData.mpb[config]?.mean_ms ?? 1) / (benchmarkData.blaze[config]?.mean_ms ?? 1),
      0,
    );
    return (total / CONFIGS.length).toFixed(1);
  }, [benchmarkData]);

  if (loading) return <ChartPlaceholder width={width} height={height} />;

  const caption = hasFullPrecision
    ? `Blaze achieves ${averageSpeedup}× average speedup over MPB on ${workload} workloads. Mixed precision (f32/f64) shown alongside full precision (f64).`
    : `Blaze achieves ${averageSpeedup}× average speedup over MPB on ${workload} workloads in its native mixed-precision mode.`;

  return (
    <BarChart
      data={data}
      width={width}
      height={height}
      title={title}
      yLabel="Time per job (ms)"
      yTickFormat={value => value.toFixed(0)}
      valueFormat={value => value.toFixed(0)}
      labelAngle={0}
      showValues
      showStd
      groupColors={COLORS}
      categoryBrackets={CATEGORY_BRACKETS}
      showCategoryBrackets
      bracketOffset={28}
      caption={caption}
      margin={{ top: 60, right: 30, bottom: 70, left: 75 }}
    />
  );
}
