'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { LinePath } from '@visx/shape';
import { scaleLog } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows, GridColumns } from '@visx/grid';
import { Text } from '@visx/text';
import { useSeries3Benchmarks } from '../../lib/use-benchmarks';
import { CHART_STYLES } from './BarChart';

// Colors matching other charts
const COLORS = {
  mpb: '#5477c4',     // Blue-gray
  blaze: '#eaf1fe',   // Light blue-white
  refLine: '#999',    // Gray for reference lines
};

// Marker size
const MARKER_SIZE = 5;

interface ResolutionScalingChartProps {
  width?: number;
  height?: number;
}

export default function ResolutionScalingChart({
  width = 650,
  height = 380,
}: ResolutionScalingChartProps) {
  const { data: benchmarkData, loading } = useSeries3Benchmarks();

  // Transform data for the log-log plot
  const { tmData, teData, refLines } = useMemo(() => {
    const tm = {
      resolutions: benchmarkData.TM.resolution,
      mpb: benchmarkData.TM.mpb.map((d, i) => ({
        x: benchmarkData.TM.resolution[i],
        y: d?.mean || 0,
        std: d?.std || 0,
      })).filter(d => d.y > 0),
      blaze: benchmarkData.TM.blaze.map((d, i) => ({
        x: benchmarkData.TM.resolution[i],
        y: d?.mean || 0,
        std: d?.std || 0,
      })).filter(d => d.y > 0),
    };

    const te = {
      resolutions: benchmarkData.TE.resolution,
      mpb: benchmarkData.TE.mpb.map((d, i) => ({
        x: benchmarkData.TE.resolution[i],
        y: d?.mean || 0,
        std: d?.std || 0,
      })).filter(d => d.y > 0),
      blaze: benchmarkData.TE.blaze.map((d, i) => ({
        x: benchmarkData.TE.resolution[i],
        y: d?.mean || 0,
        std: d?.std || 0,
      })).filter(d => d.y > 0),
    };

    // Compute reference lines O(N²) and O(N³)
    const midIdx = Math.floor(tm.blaze.length / 2);
    const refX = tm.blaze[midIdx]?.x || 64;
    const refY = tm.blaze[midIdx]?.y || 600;
    
    const refLines = {
      n2: { refX, refY },
      n3: { refX, refY: refY / 5 },
    };

    return { tmData: tm, teData: te, refLines };
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

  // Common domain for both plots
  const xDomain: [number, number] = [10, 250];
  const yDomain: [number, number] = [30, 100000];

  // Generate power-of-10 tick values
  const getPowerOf10Ticks = (domain: [number, number]): number[] => {
    const ticks: number[] = [];
    const minPow = Math.ceil(Math.log10(domain[0]));
    const maxPow = Math.floor(Math.log10(domain[1]));
    for (let pow = minPow; pow <= maxPow; pow++) {
      ticks.push(Math.pow(10, pow));
    }
    return ticks;
  };

  const xTicks = getPowerOf10Ticks(xDomain);
  const yTicks = getPowerOf10Ticks(yDomain);

  // Create scales
  const xScale = scaleLog<number>({
    domain: xDomain,
    range: [0, innerWidth],
    base: 10,
  });

  const yScale = scaleLog<number>({
    domain: yDomain,
    range: [innerHeight, 0],
    base: 10,
  });

  // Generate reference line points
  const generateRefLine = (power: number, refX: number, refY: number) => {
    const points = [];
    for (let x = xDomain[0]; x <= xDomain[1]; x *= 1.1) {
      const y = refY * Math.pow(x / refX, power);
      if (y >= yDomain[0] && y <= yDomain[1]) {
        points.push({ x, y });
      }
    }
    return points;
  };

  const renderChart = (
    data: {
      resolutions: number[];
      mpb: { x: number; y: number; std: number }[];
      blaze: { x: number; y: number; std: number }[];
    },
    polarization: string
  ) => {
    // Reference lines for this chart
    const midIdx = Math.floor(data.blaze.length / 2);
    const refX = data.blaze[midIdx]?.x || 64;
    const refY = (data.mpb[midIdx]?.y + data.blaze[midIdx]?.y) / 2 || 1000;
    
    const n2Line = generateRefLine(2, refX, refY);
    const n3Line = generateRefLine(3, refX, refY / 10);

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
          {`Scaling Behavior (${polarization})`}
        </Text>

        <Group left={margin.left} top={margin.top}>
          {/* Grid */}
          <GridRows
            scale={yScale}
            width={innerWidth}
            stroke={CHART_STYLES.gridColor}
            strokeOpacity={0.2}
            strokeDasharray="3,3"
          />
          <GridColumns
            scale={xScale}
            height={innerHeight}
            stroke={CHART_STYLES.gridColor}
            strokeOpacity={0.2}
            strokeDasharray="3,3"
          />

          {/* O(N²) reference line */}
          <LinePath
            data={n2Line}
            x={(d) => xScale(d.x)}
            y={(d) => yScale(d.y)}
            stroke={COLORS.refLine}
            strokeWidth={1}
            strokeDasharray="4,4"
            strokeOpacity={0.5}
          />
          {n2Line.length > 0 && (
            <Text
              x={xScale(n2Line[n2Line.length - 1].x) - 25}
              y={yScale(n2Line[n2Line.length - 1].y) - 5}
              fontSize={11}
              fontFamily={CHART_STYLES.fontFamily}
              fill={COLORS.refLine}
            >
              O(N²)
            </Text>
          )}

          {/* O(N³) reference line */}
          <LinePath
            data={n3Line}
            x={(d) => xScale(d.x)}
            y={(d) => yScale(d.y)}
            stroke={COLORS.refLine}
            strokeWidth={1}
            strokeDasharray="2,2"
            strokeOpacity={0.5}
          />
          {n3Line.length > 0 && (
            <Text
              x={xScale(n3Line[Math.floor(n3Line.length / 2)].x) + 5}
              y={yScale(n3Line[Math.floor(n3Line.length / 2)].y) + 10}
              fontSize={11}
              fontFamily={CHART_STYLES.fontFamily}
              fill={COLORS.refLine}
            >
              O(N³)
            </Text>
          )}

          {/* MPB data points and line */}
          <LinePath
            data={data.mpb}
            x={(d) => xScale(d.x)}
            y={(d) => yScale(d.y)}
            stroke={COLORS.mpb}
            strokeWidth={2}
          />
          {data.mpb.map((d, i) => (
            <circle
              key={`mpb-${i}`}
              cx={xScale(d.x)}
              cy={yScale(d.y)}
              r={MARKER_SIZE}
              fill={COLORS.mpb}
            />
          ))}

          {/* Blaze data points and line */}
          <LinePath
            data={data.blaze}
            x={(d) => xScale(d.x)}
            y={(d) => yScale(d.y)}
            stroke={COLORS.blaze}
            strokeWidth={2}
          />
          {data.blaze.map((d, i) => (
            <rect
              key={`blaze-${i}`}
              x={xScale(d.x) - MARKER_SIZE}
              y={yScale(d.y) - MARKER_SIZE}
              width={MARKER_SIZE * 2}
              height={MARKER_SIZE * 2}
              fill={COLORS.blaze}
            />
          ))}

          {/* Y Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => {
              const val = v as number;
              if (val >= 1000) return `${(val / 1000).toFixed(0)}s`;
              return `${val}ms`;
            }}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'end' as const,
              dx: -4,
              dy: 4,
            })}
            tickValues={yTicks}
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
            tickValues={xTicks}
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
            Resolution (N)
          </Text>
        </Group>

        {/* Legend */}
        <Group top={margin.top - 25} left={margin.left}>
          <circle cx={6} cy={0} r={5} fill={COLORS.mpb} />
          <Text x={16} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
            MPB
          </Text>
          <rect x={60} y={-5} width={10} height={10} fill={COLORS.blaze} />
          <Text x={76} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
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
