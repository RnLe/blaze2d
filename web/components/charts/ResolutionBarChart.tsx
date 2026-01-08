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

// Colors matching other charts
const COLORS = {
  mpb: '#5477c4',     // Blue-gray
  blaze: '#eaf1fe',   // Light blue-white
};

interface ResolutionBarChartProps {
  width?: number;
  height?: number;
}

export default function ResolutionBarChart({
  width = 650,
  height = 380,
}: ResolutionBarChartProps) {
  const { data: benchmarkData, loading } = useSeries3Benchmarks();

  // Transform data for the bar charts
  const { tmData, teData } = useMemo(() => {
    const tm = {
      resolutions: benchmarkData.TM.resolution,
      mpb: benchmarkData.TM.mpb.map((d, i) => ({
        resolution: benchmarkData.TM.resolution[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
      blaze: benchmarkData.TM.blaze.map((d, i) => ({
        resolution: benchmarkData.TM.resolution[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
    };

    const te = {
      resolutions: benchmarkData.TE.resolution,
      mpb: benchmarkData.TE.mpb.map((d, i) => ({
        resolution: benchmarkData.TE.resolution[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
      blaze: benchmarkData.TE.blaze.map((d, i) => ({
        resolution: benchmarkData.TE.resolution[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
    };

    return { tmData: tm, teData: te };
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
    data: {
      resolutions: number[];
      mpb: { resolution: number; mean: number; std: number }[];
      blaze: { resolution: number; mean: number; std: number }[];
    },
    polarization: string
  ) => {
    const maxValue = Math.max(
      ...data.mpb.map(d => d.mean + d.std),
      ...data.blaze.map(d => d.mean + d.std)
    );

    const xScale = scaleBand<number>({
      domain: data.resolutions,
      range: [0, innerWidth],
      padding: 0.3,
    });

    const yScale = scaleLinear<number>({
      domain: [0, maxValue * 1.1],
      range: [innerHeight, 0],
    });

    const barWidth = xScale.bandwidth() / 2 - 2;

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
          {`Time Comparison (${polarization})`}
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

          {/* Bars */}
          {data.resolutions.map((res, i) => {
            const xPos = xScale(res) || 0;
            const mpbData = data.mpb[i];
            const blazeData = data.blaze[i];

            return (
              <g key={res}>
                {/* MPB bar */}
                <Bar
                  x={xPos}
                  y={yScale(mpbData.mean)}
                  width={barWidth}
                  height={innerHeight - yScale(mpbData.mean)}
                  fill={COLORS.mpb}
                />
                {/* MPB error bar */}
                {mpbData.std > 0 && (
                  <g>
                    <line
                      x1={xPos + barWidth / 2}
                      y1={yScale(mpbData.mean - mpbData.std)}
                      x2={xPos + barWidth / 2}
                      y2={yScale(mpbData.mean + mpbData.std)}
                      stroke={COLORS.mpb}
                      strokeWidth={1.5}
                    />
                    <line
                      x1={xPos + barWidth / 2 - 3}
                      y1={yScale(mpbData.mean + mpbData.std)}
                      x2={xPos + barWidth / 2 + 3}
                      y2={yScale(mpbData.mean + mpbData.std)}
                      stroke={COLORS.mpb}
                      strokeWidth={1.5}
                    />
                  </g>
                )}

                {/* Blaze bar */}
                <Bar
                  x={xPos + barWidth + 4}
                  y={yScale(blazeData.mean)}
                  width={barWidth}
                  height={innerHeight - yScale(blazeData.mean)}
                  fill={COLORS.blaze}
                />
                {/* Blaze error bar */}
                {blazeData.std > 0 && (
                  <g>
                    <line
                      x1={xPos + barWidth + 4 + barWidth / 2}
                      y1={yScale(blazeData.mean - blazeData.std)}
                      x2={xPos + barWidth + 4 + barWidth / 2}
                      y2={yScale(blazeData.mean + blazeData.std)}
                      stroke={COLORS.blaze}
                      strokeWidth={1.5}
                    />
                    <line
                      x1={xPos + barWidth + 4 + barWidth / 2 - 3}
                      y1={yScale(blazeData.mean + blazeData.std)}
                      x2={xPos + barWidth + 4 + barWidth / 2 + 3}
                      y2={yScale(blazeData.mean + blazeData.std)}
                      stroke={COLORS.blaze}
                      strokeWidth={1.5}
                    />
                  </g>
                )}
              </g>
            );
          })}

          {/* Y Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => {
              const val = v as number;
              if (val >= 1000) return `${(val / 1000).toFixed(0)}s`;
              return `${val.toFixed(0)}ms`;
            }}
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
            Time (ms)
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

        {/* Legend */}
        <Group top={margin.top - 25} left={margin.left}>
          <rect x={0} y={-6} width={12} height={12} fill={COLORS.mpb} />
          <Text x={18} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            MPB
          </Text>
          <rect x={60} y={-6} width={12} height={12} fill={COLORS.blaze} />
          <Text x={78} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            Blaze2D
          </Text>
        </Group>
      </svg>
    );
  };

  return (
    <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }}>
      {renderChart(tmData, 'TM')}
      {renderChart(teData, 'TE')}
    </div>
  );
}
