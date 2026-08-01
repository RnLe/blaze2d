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
  blaze: '#eaf1fe',   // Light blue-white
};

interface IterationTimeChartProps {
  width?: number;
  height?: number;
}

export default function IterationTimeChart({
  width = 650,
  height = 380,
}: IterationTimeChartProps) {
  const { data: benchmarkData, loading } = useSeries4Benchmarks();

  // Transform data for the bar charts
  const { tmData, teData, kIndices, highSymmetryPoints } = useMemo(() => {
    const kPerSeg = benchmarkData.parameters.k_points_per_segment;
    
    // High symmetry points for square lattice: Γ→X→M→Γ
    const hsPoints = [
      { index: 0, label: 'Γ' },
      { index: kPerSeg, label: 'X' },
      { index: 2 * kPerSeg, label: 'M' },
      { index: 3 * kPerSeg, label: 'Γ' },
    ];

    // Calculate time per iteration in ms
    const calcTimePerIter = (kp: { iterations: number; elapsed_seconds: number }) => {
      if (kp.iterations === 0) return 0;
      return (kp.elapsed_seconds / kp.iterations) * 1000; // Convert to ms
    };

    const tm = {
      mpb: benchmarkData.TM.mpb?.k_points.map(kp => ({
        k_index: kp.k_index,
        time_per_iter: calcTimePerIter(kp),
      })) || [],
      blaze: benchmarkData.TM.blaze?.k_points.map(kp => ({
        k_index: kp.k_index,
        time_per_iter: calcTimePerIter(kp),
      })) || [],
    };

    const te = {
      mpb: benchmarkData.TE.mpb?.k_points.map(kp => ({
        k_index: kp.k_index,
        time_per_iter: calcTimePerIter(kp),
      })) || [],
      blaze: benchmarkData.TE.blaze?.k_points.map(kp => ({
        k_index: kp.k_index,
        time_per_iter: calcTimePerIter(kp),
      })) || [],
    };

    // Get all k-indices
    const allIndices = tm.mpb.length >= tm.blaze.length 
      ? tm.mpb.map(d => d.k_index)
      : tm.blaze.map(d => d.k_index);

    return { tmData: tm, teData: te, kIndices: allIndices, highSymmetryPoints: hsPoints };
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
      mpb: { k_index: number; time_per_iter: number }[];
      blaze: { k_index: number; time_per_iter: number }[];
    },
    polarization: string
  ) => {
    const maxTime = Math.max(
      ...data.mpb.map(d => d.time_per_iter),
      ...data.blaze.map(d => d.time_per_iter)
    );

    const xScale = scaleBand<number>({
      domain: kIndices,
      range: [0, innerWidth],
      padding: 0.05,
    });

    const yScale = scaleLinear<number>({
      domain: [0, maxTime * 1.1],
      range: [innerHeight, 0],
    });

    const barWidth = Math.max(xScale.bandwidth() / 2 - 1, 1);

    // Create lookup maps
    const mpbMap = new Map(data.mpb.map(d => [d.k_index, d.time_per_iter]));
    const blazeMap = new Map(data.blaze.map(d => [d.k_index, d.time_per_iter]));

    // Show every Nth tick on x-axis
    const tickStep = kIndices.length > 40 ? 10 : 5;
    const xTickValues = kIndices.filter((_, i) => i % tickStep === 0);

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
          {`Time per Iteration (${polarization})`}
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

          {/* High symmetry point markers */}
          {highSymmetryPoints.map((hs) => {
            const xPos = xScale(hs.index);
            if (xPos === undefined) return null;
            return (
              <g key={hs.label + hs.index}>
                <line
                  x1={xPos + xScale.bandwidth() / 2}
                  y1={0}
                  x2={xPos + xScale.bandwidth() / 2}
                  y2={innerHeight}
                  stroke="#666"
                  strokeWidth={1}
                  strokeDasharray="2,2"
                  strokeOpacity={0.5}
                />
                <Text
                  x={xPos + xScale.bandwidth() / 2}
                  y={-8}
                  fontSize={11}
                  fontFamily={CHART_STYLES.fontFamily}
                  fill={CHART_STYLES.labelColor}
                  textAnchor="middle"
                >
                  {hs.label}
                </Text>
              </g>
            );
          })}

          {/* Bars */}
          {kIndices.map((kIdx) => {
            const xPos = xScale(kIdx) || 0;
            const mpbTime = mpbMap.get(kIdx) || 0;
            const blazeTime = blazeMap.get(kIdx) || 0;

            return (
              <g key={kIdx}>
                {/* MPB bar */}
                {mpbTime > 0 && (
                  <Bar
                    x={xPos}
                    y={yScale(mpbTime)}
                    width={barWidth}
                    height={innerHeight - yScale(mpbTime)}
                    fill={COLORS.mpb}
                  />
                )}
                {/* Blaze bar */}
                {blazeTime > 0 && (
                  <Bar
                    x={xPos + barWidth}
                    y={yScale(blazeTime)}
                    width={barWidth}
                    height={innerHeight - yScale(blazeTime)}
                    fill={COLORS.blaze}
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
            tickFormat={(v) => `${(v as number).toFixed(1)}`}
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
            Time per Iteration (ms)
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
            K-Point Index
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
