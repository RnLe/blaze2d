'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { Bar } from '@visx/shape';
import { scaleBand, scaleLinear } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';
import { useSeries2Benchmarks } from '../../lib/use-benchmarks';
import { CHART_STYLES } from './BarChart';

// Colors matching other charts
const COLORS = {
  mpb: '#5477c4',     // Blue-gray
  blaze: '#eaf1fe',   // Light blue-white
};

interface BandsBarChartProps {
  width?: number;
  height?: number;
}

export default function BandsBarChart({
  width = 650,
  height = 380,
}: BandsBarChartProps) {
  const { data: benchmarkData, loading } = useSeries2Benchmarks();

  // Transform data for the bar charts
  const { tmData, teData } = useMemo(() => {
    const bands = benchmarkData.band_values;
    
    const tm = {
      bands,
      mpb: benchmarkData.TM.mpb.map((d, i) => ({
        band: bands[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
      blaze: benchmarkData.TM.blaze.map((d, i) => ({
        band: bands[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
    };

    const te = {
      bands,
      mpb: benchmarkData.TE.mpb.map((d, i) => ({
        band: bands[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
      blaze: benchmarkData.TE.blaze.map((d, i) => ({
        band: bands[i],
        mean: d?.mean || 0,
        std: d?.std || 0,
      })),
    };

    return { tmData: tm, teData: te };
  }, [benchmarkData]);

  if (loading || tmData.mpb.length === 0) {
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
      bands: number[];
      mpb: { band: number; mean: number; std: number }[];
      blaze: { band: number; mean: number; std: number }[];
    },
    polarization: string
  ) => {
    const maxValue = Math.max(
      ...data.mpb.map(d => d.mean + d.std),
      ...data.blaze.map(d => d.mean + d.std)
    );

    const xScale = scaleBand<number>({
      domain: data.bands,
      range: [0, innerWidth],
      padding: 0.2,
    });

    const yScale = scaleLinear<number>({
      domain: [0, maxValue * 1.1],
      range: [innerHeight, 0],
    });

    const barWidth = xScale.bandwidth() / 2 - 1;

    // Show every other tick for readability if many bands
    const tickStep = data.bands.length > 12 ? 2 : 1;
    const xTickValues = data.bands.filter((_, i) => i % tickStep === 0);

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
          {`Time vs Number of Bands (${polarization})`}
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
          {data.bands.map((band, i) => {
            const xPos = xScale(band) || 0;
            const mpbData = data.mpb[i];
            const blazeData = data.blaze[i];

            return (
              <g key={band}>
                {/* MPB bar */}
                {mpbData.mean > 0 && (
                  <>
                    <Bar
                      x={xPos}
                      y={yScale(mpbData.mean)}
                      width={barWidth}
                      height={innerHeight - yScale(mpbData.mean)}
                      fill={COLORS.mpb}
                      opacity={0.8}
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
                          x1={xPos + barWidth / 2 - 2}
                          y1={yScale(mpbData.mean + mpbData.std)}
                          x2={xPos + barWidth / 2 + 2}
                          y2={yScale(mpbData.mean + mpbData.std)}
                          stroke={COLORS.mpb}
                          strokeWidth={1.5}
                        />
                      </g>
                    )}
                  </>
                )}

                {/* Blaze bar */}
                {blazeData.mean > 0 && (
                  <>
                    <Bar
                      x={xPos + barWidth}
                      y={yScale(blazeData.mean)}
                      width={barWidth}
                      height={innerHeight - yScale(blazeData.mean)}
                      fill={COLORS.blaze}
                      opacity={0.8}
                    />
                    {/* Blaze error bar */}
                    {blazeData.std > 0 && (
                      <g>
                        <line
                          x1={xPos + barWidth + barWidth / 2}
                          y1={yScale(blazeData.mean - blazeData.std)}
                          x2={xPos + barWidth + barWidth / 2}
                          y2={yScale(blazeData.mean + blazeData.std)}
                          stroke={COLORS.blaze}
                          strokeWidth={1.5}
                        />
                        <line
                          x1={xPos + barWidth + barWidth / 2 - 2}
                          y1={yScale(blazeData.mean + blazeData.std)}
                          x2={xPos + barWidth + barWidth / 2 + 2}
                          y2={yScale(blazeData.mean + blazeData.std)}
                          stroke={COLORS.blaze}
                          strokeWidth={1.5}
                        />
                      </g>
                    )}
                  </>
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
            tickFormat={(v) => `${v}`}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 10,
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
            Runtime (ms)
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
            Number of Bands
          </Text>
        </Group>

        {/* Legend */}
        <Group top={margin.top - 25} left={margin.left}>
          <rect x={0} y={-6} width={12} height={12} fill={COLORS.mpb} opacity={0.8} />
          <Text x={18} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            MPB
          </Text>
          <rect x={60} y={-6} width={12} height={12} fill={COLORS.blaze} opacity={0.8} />
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
