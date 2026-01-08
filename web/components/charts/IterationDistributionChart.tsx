'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { Bar } from '@visx/shape';
import { scaleBand, scaleLinear } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';
import { useSeries4Benchmarks } from '../../lib/use-benchmarks';
import { CHART_STYLES } from './BarChart';

// Colors matching other charts
const COLORS = {
  mpb: '#5477c4',     // Blue-gray
  blaze: '#4caf50',   // Green for better contrast when overlapping
};

interface IterationDistributionChartProps {
  width?: number;
  height?: number;
}

export default function IterationDistributionChart({
  width = 520,  // Reduced from 650 (~20% narrower)
  height = 380,
}: IterationDistributionChartProps) {
  const { data: benchmarkData, loading } = useSeries4Benchmarks();

  // Compute histogram data with bin width of 2 iterations
  const { tmHistogram, teHistogram, binCenters } = useMemo(() => {
    const binWidth = 2; // Fixed bin width of 2 iterations
    
    // Get all iteration values to determine range
    const allIters = [
      ...(benchmarkData.TM.mpb?.k_points.map(kp => kp.iterations) || []),
      ...(benchmarkData.TM.blaze?.k_points.map(kp => kp.iterations) || []),
      ...(benchmarkData.TE.mpb?.k_points.map(kp => kp.iterations) || []),
      ...(benchmarkData.TE.blaze?.k_points.map(kp => kp.iterations) || []),
    ];
    
    const minIter = Math.min(...allIters);
    const maxIter = Math.max(...allIters);
    
    // Align to even numbers for cleaner bins
    const alignedMin = Math.floor(minIter / binWidth) * binWidth;
    const alignedMax = Math.ceil((maxIter + 1) / binWidth) * binWidth;
    const numBins = Math.ceil((alignedMax - alignedMin) / binWidth);
    
    // Create bin centers for labeling
    const centers: number[] = [];
    for (let i = 0; i < numBins; i++) {
      centers.push(alignedMin + i * binWidth + binWidth / 2);
    }
    
    // Helper to compute histogram
    const computeHist = (iterations: number[]) => {
      const counts = new Array(numBins).fill(0);
      for (const iter of iterations) {
        const binIdx = Math.min(Math.floor((iter - alignedMin) / binWidth), numBins - 1);
        if (binIdx >= 0) counts[binIdx]++;
      }
      return counts;
    };
    
    const tmHist = {
      mpb: computeHist(benchmarkData.TM.mpb?.k_points.map(kp => kp.iterations) || []),
      blaze: computeHist(benchmarkData.TM.blaze?.k_points.map(kp => kp.iterations) || []),
    };
    
    const teHist = {
      mpb: computeHist(benchmarkData.TE.mpb?.k_points.map(kp => kp.iterations) || []),
      blaze: computeHist(benchmarkData.TE.blaze?.k_points.map(kp => kp.iterations) || []),
    };
    
    return { tmHistogram: tmHist, teHistogram: teHist, binCenters: centers };
  }, [benchmarkData]);

  if (loading) {
    return (
      <div style={{ width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>
        Loading benchmark data...
      </div>
    );
  }

  const chartWidth = (width - 20) / 2;
  const margin = { top: 65, right: 20, bottom: 60, left: 70 };
  const innerWidth = chartWidth - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const renderChart = (
    histogram: { mpb: number[]; blaze: number[] },
    polarization: string
  ) => {
    const maxCount = Math.max(
      ...histogram.mpb,
      ...histogram.blaze
    );

    const binIndices = histogram.mpb.map((_, i) => i);
    
    const xScale = scaleBand<number>({
      domain: binIndices,
      range: [0, innerWidth],
      padding: 0.1,
    });

    const yScale = scaleLinear<number>({
      domain: [0, maxCount * 1.1],
      range: [innerHeight, 0],
    });

    // Full bandwidth for overlapping bars
    const barWidth = xScale.bandwidth();

    // Create bin labels (show center value)
    const getBinLabel = (idx: number) => {
      const center = binCenters[idx];
      return `${Math.floor(center)}`;
    };

    // Show every Nth tick for readability
    const tickStep = binIndices.length > 20 ? 4 : binIndices.length > 10 ? 2 : 1;
    const xTickValues = binIndices.filter((_, i) => i % tickStep === 0);

    return (
      <svg width={chartWidth} height={height} style={{ overflow: 'visible' }}>
        {/* Title */}
        <Text
          x={0}
          y={16}
          fontSize={14}
          fontFamily={CHART_STYLES.fontFamily}
          fill={CHART_STYLES.labelColor}
          fontWeight={700}
        >
          {`Iteration Distribution (${polarization})`}
        </Text>

        <Group left={margin.left} top={margin.top}>
          {/* Grid */}
          <GridRows
            scale={yScale}
            width={innerWidth}
            stroke={CHART_STYLES.gridColor}
            strokeOpacity={0.4}
            strokeDasharray="3,3"
          />

          {/* Bars - overlapping with transparency */}
          {binIndices.map((binIdx) => {
            const xPos = xScale(binIdx) || 0;
            const mpbCount = histogram.mpb[binIdx];
            const blazeCount = histogram.blaze[binIdx];

            return (
              <g key={binIdx}>
                {/* MPB bar (render first, behind) */}
                {mpbCount > 0 && (
                  <Bar
                    x={xPos}
                    y={yScale(mpbCount)}
                    width={barWidth}
                    height={innerHeight - yScale(mpbCount)}
                    fill={COLORS.mpb}
                    opacity={0.6}
                  />
                )}
                {/* Blaze bar (render second, in front with transparency) */}
                {blazeCount > 0 && (
                  <Bar
                    x={xPos}
                    y={yScale(blazeCount)}
                    width={barWidth}
                    height={innerHeight - yScale(blazeCount)}
                    fill={COLORS.blaze}
                    opacity={0.5}
                  />
                )}
              </g>
            );
          })}

          {/* Y Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => `${v}`}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'end' as const,
              dx: -4,
              dy: 4,
            })}
            numTicks={5}
          />

          {/* X Axis */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickValues={xTickValues}
            tickFormat={(v) => getBinLabel(v as number)}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 9,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'middle' as const,
              dy: 4,
            })}
          />

          {/* Y Label */}
          <Text
            x={-innerHeight / 2}
            y={-55}
            transform="rotate(-90)"
            fontSize={11}
            fontFamily={CHART_STYLES.fontFamily}
            fill={CHART_STYLES.labelColor}
            textAnchor="middle"
          >
            Count
          </Text>

          {/* X Label */}
          <Text
            x={innerWidth / 2}
            y={innerHeight + 45}
            fontSize={11}
            fontFamily={CHART_STYLES.fontFamily}
            fill={CHART_STYLES.labelColor}
            textAnchor="middle"
          >
            Iterations
          </Text>
        </Group>

        {/* Legend */}
        <Group top={margin.top - 25} left={margin.left}>
          <rect x={0} y={-6} width={12} height={12} fill={COLORS.mpb} opacity={0.6} />
          <Text x={18} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            MPB
          </Text>
          <rect x={60} y={-6} width={12} height={12} fill={COLORS.blaze} opacity={0.5} />
          <Text x={78} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            Blaze2D
          </Text>
        </Group>
      </svg>
    );
  };

  return (
    <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
      {renderChart(tmHistogram, 'TM')}
      {renderChart(teHistogram, 'TE')}
    </div>
  );
}
