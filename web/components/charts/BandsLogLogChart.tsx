'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { LinePath } from '@visx/shape';
import { scaleLog, scaleLinear } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows, GridColumns } from '@visx/grid';
import { Text } from '@visx/text';
import { useSeries2Benchmarks } from '../../lib/use-benchmarks';
import { CHART_STYLES } from './BarChart';

// Colors matching other charts
const COLORS = {
  mpb: '#5477c4',     // Blue-gray
  blaze: '#f97316',   // Orange for better contrast
};

interface BandsLogLogChartProps {
  width?: number;
  height?: number;
}

export default function BandsLogLogChart({
  width = 650,
  height = 380,
}: BandsLogLogChartProps) {
  const { data: benchmarkData, loading } = useSeries2Benchmarks();

  // Transform data for the charts
  const { tmData, teData } = useMemo(() => {
    const bands = benchmarkData.band_values;
    
    const tm = {
      bands,
      mpb: benchmarkData.TM.mpb
        .map((d, i) => ({
          band: bands[i],
          mean: d?.mean || 0,
          std: d?.std || 0,
        }))
        .filter(d => d.mean > 0),
      blaze: benchmarkData.TM.blaze
        .map((d, i) => ({
          band: bands[i],
          mean: d?.mean || 0,
          std: d?.std || 0,
        }))
        .filter(d => d.mean > 0),
    };

    const te = {
      bands,
      mpb: benchmarkData.TE.mpb
        .map((d, i) => ({
          band: bands[i],
          mean: d?.mean || 0,
          std: d?.std || 0,
        }))
        .filter(d => d.mean > 0),
      blaze: benchmarkData.TE.blaze
        .map((d, i) => ({
          band: bands[i],
          mean: d?.mean || 0,
          std: d?.std || 0,
        }))
        .filter(d => d.mean > 0),
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
    const allMeans = [...data.mpb.map(d => d.mean), ...data.blaze.map(d => d.mean)];
    const minBand = Math.min(...data.bands);
    const maxBand = Math.max(...data.bands);
    const minValue = Math.min(...allMeans);
    const maxValue = Math.max(...allMeans);

    const xScale = scaleLog<number>({
      domain: [minBand * 0.9, maxBand * 1.1],
      range: [0, innerWidth],
      base: 10,
    });

    const yScale = scaleLog<number>({
      domain: [minValue * 0.8, maxValue * 1.2],
      range: [innerHeight, 0],
      base: 10,
    });

    // Format tick labels for log scale
    const formatLogTick = (value: number) => {
      if (value >= 1000) return `${(value / 1000).toFixed(0)}k`;
      if (value >= 100) return `${value.toFixed(0)}`;
      if (value >= 10) return `${value.toFixed(0)}`;
      return `${value.toFixed(1)}`;
    };

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
          {`Log-Log: Time vs Bands (${polarization})`}
        </Text>

        <Group left={margin.left} top={margin.top}>
          {/* Grid */}
          <GridRows
            scale={yScale}
            width={innerWidth}
            stroke={CHART_STYLES.gridColor}
            strokeOpacity={0.3}
            strokeDasharray="3,3"
            numTicks={5}
          />
          <GridColumns
            scale={xScale}
            height={innerHeight}
            stroke={CHART_STYLES.gridColor}
            strokeOpacity={0.3}
            strokeDasharray="3,3"
            numTicks={5}
          />

          {/* MPB line with markers */}
          <LinePath
            data={data.mpb}
            x={d => xScale(d.band)}
            y={d => yScale(d.mean)}
            stroke={COLORS.mpb}
            strokeWidth={2}
            strokeOpacity={0.9}
          />
          {data.mpb.map((d, i) => (
            <g key={`mpb-${i}`}>
              <circle
                cx={xScale(d.band)}
                cy={yScale(d.mean)}
                r={4}
                fill={COLORS.mpb}
                opacity={0.9}
              />
              {/* Error bars */}
              {d.std > 0 && d.mean - d.std > 0 && (
                <>
                  <line
                    x1={xScale(d.band)}
                    y1={yScale(d.mean - d.std)}
                    x2={xScale(d.band)}
                    y2={yScale(d.mean + d.std)}
                    stroke={COLORS.mpb}
                    strokeWidth={1}
                    opacity={0.7}
                  />
                </>
              )}
            </g>
          ))}

          {/* Blaze line with markers */}
          <LinePath
            data={data.blaze}
            x={d => xScale(d.band)}
            y={d => yScale(d.mean)}
            stroke={COLORS.blaze}
            strokeWidth={2}
            strokeOpacity={0.9}
          />
          {data.blaze.map((d, i) => (
            <g key={`blaze-${i}`}>
              <circle
                cx={xScale(d.band)}
                cy={yScale(d.mean)}
                r={4}
                fill={COLORS.blaze}
                opacity={0.9}
              />
              {/* Error bars */}
              {d.std > 0 && d.mean - d.std > 0 && (
                <>
                  <line
                    x1={xScale(d.band)}
                    y1={yScale(d.mean - d.std)}
                    x2={xScale(d.band)}
                    y2={yScale(d.mean + d.std)}
                    stroke={COLORS.blaze}
                    strokeWidth={1}
                    opacity={0.7}
                  />
                </>
              )}
            </g>
          ))}

          {/* Y Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => formatLogTick(v as number)}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 10,
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
              fontSize: 10,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'middle' as const,
              dy: 4,
            })}
            numTicks={5}
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
          <line x1={0} y1={0} x2={16} y2={0} stroke={COLORS.mpb} strokeWidth={2} />
          <circle cx={8} cy={0} r={3} fill={COLORS.mpb} />
          <Text x={22} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            MPB
          </Text>
          <line x1={70} y1={0} x2={86} y2={0} stroke={COLORS.blaze} strokeWidth={2} />
          <circle cx={78} cy={0} r={3} fill={COLORS.blaze} />
          <Text x={92} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
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
