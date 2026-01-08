'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { Bar } from '@visx/shape';
import { scaleBand, scaleLinear } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';
import { useSeries3Benchmarks } from '../../lib/use-benchmarks';
import { CHART_STYLES } from './BarChart';

// Colors for speedup
const COLORS = {
  positive: '#eaf1fe',  // Frost for speedup > 1
  negative: '#ef4444',  // Red for speedup < 1
};

interface ResolutionSpeedupChartProps {
  width?: number;
  height?: number;
}

export default function ResolutionSpeedupChart({
  width = 650,
  height = 380,
}: ResolutionSpeedupChartProps) {
  const { data: benchmarkData, loading } = useSeries3Benchmarks();

  // Calculate speedups
  const { tmSpeedups, teSpeedups, resolutions } = useMemo(() => {
    const resolutions = benchmarkData.TM.resolution;
    
    const tmSpeedups = benchmarkData.TM.mpb.map((mpb, i) => {
      const blaze = benchmarkData.TM.blaze[i];
      if (!mpb || !blaze || blaze.mean === 0) return 0;
      return mpb.mean / blaze.mean;
    });

    const teSpeedups = benchmarkData.TE.mpb.map((mpb, i) => {
      const blaze = benchmarkData.TE.blaze[i];
      if (!mpb || !blaze || blaze.mean === 0) return 0;
      return mpb.mean / blaze.mean;
    });

    return { tmSpeedups, teSpeedups, resolutions };
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

  const renderChart = (speedups: number[], polarization: string) => {
    const maxValue = Math.max(...speedups, 1) * 1.15;

    const xScale = scaleBand<number>({
      domain: resolutions,
      range: [0, innerWidth],
      padding: 0.3,
    });

    const yScale = scaleLinear<number>({
      domain: [0, maxValue],
      range: [innerHeight, 0],
    });

    const barWidth = xScale.bandwidth();

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
          {`Speedup (${polarization})`}
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

          {/* Reference line at 1× */}
          <line
            x1={0}
            y1={yScale(1)}
            x2={innerWidth}
            y2={yScale(1)}
            stroke="#fff"
            strokeWidth={1}
            strokeDasharray="5,5"
            strokeOpacity={0.7}
          />

          {/* Bars */}
          {resolutions.map((res, i) => {
            const speedup = speedups[i];
            const xPos = xScale(res) || 0;
            const color = speedup >= 1 ? COLORS.positive : COLORS.negative;

            return (
              <g key={res}>
                <Bar
                  x={xPos}
                  y={yScale(speedup)}
                  width={barWidth}
                  height={innerHeight - yScale(speedup)}
                  fill={color}
                />
                {/* Value label */}
                <Text
                  x={xPos + barWidth / 2}
                  y={yScale(speedup) - 5}
                  fontSize={9}
                  fontFamily={CHART_STYLES.fontFamily}
                  fill={CHART_STYLES.labelColor}
                  textAnchor="middle"
                >
                  {`${speedup.toFixed(1)}×`}
                </Text>
              </g>
            );
          })}

          {/* Y Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => `${v}×`}
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
            tickFormat={(v) => `${v}`}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
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
            Speedup (MPB / Blaze2D)
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
            Resolution (N×N)
          </Text>
        </Group>
      </svg>
    );
  };

  return (
    <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
      {renderChart(tmSpeedups, 'TM')}
      {renderChart(teSpeedups, 'TE')}
    </div>
  );
}
