'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { LinePath, Line } from '@visx/shape';
import { scaleLog } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows, GridColumns } from '@visx/grid';
import { Text } from '@visx/text';
import { useSeries5Benchmarks } from '../../lib/use-benchmarks';
import { CHART_STYLES } from './BarChart';

// Colors matching other charts
const COLORS = {
  mpb: '#5477c4',     // Blue-gray
  blaze: '#eaf1fe',   // Light blue-white
  frozen: '#7183ad',  // Slightly lighter blue-gray than mpb
  fit: '#888',        // Gray for fit lines
};

// Marker size
const MARKER_SIZE = 5;

interface MemoryScalingChartProps {
  width?: number;
  height?: number;
}

export default function MemoryScalingChart({
  width = 650,
  height = 380,
}: MemoryScalingChartProps) {
  const { data: benchmarkData, loading } = useSeries5Benchmarks();

  // Transform data for the log-log plot
  const { tmData, teData, fitCoeffs } = useMemo(() => {
    const resData = benchmarkData.resolution;

    const tm = {
      mpb: resData.mpb.TM.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
      blaze: resData.blaze.TM.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
    };

    const te = {
      mpb: resData.mpb.TE.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
      blaze: resData.blaze.TE.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
    };

    // Compute power law fit coefficients: y = a * x^b
    // Using least squares on log-transformed data
    const computeFit = (data: { x: number; y: number }[]): { a: number; b: number } => {
      const n = data.length;
      if (n < 2) return { a: 1, b: 0 };

      const logX = data.map(d => Math.log(d.x));
      const logY = data.map(d => Math.log(d.y));

      const sumLogX = logX.reduce((a, b) => a + b, 0);
      const sumLogY = logY.reduce((a, b) => a + b, 0);
      const sumLogXLogY = logX.reduce((sum, lx, i) => sum + lx * logY[i], 0);
      const sumLogX2 = logX.reduce((sum, lx) => sum + lx * lx, 0);

      const b = (n * sumLogXLogY - sumLogX * sumLogY) / (n * sumLogX2 - sumLogX * sumLogX);
      const logA = (sumLogY - b * sumLogX) / n;
      const a = Math.exp(logA);

      return { a, b };
    };

    const fitCoeffs = {
      tm: {
        mpb: computeFit(tm.mpb),
        blaze: computeFit(tm.blaze),
      },
      te: {
        mpb: computeFit(te.mpb),
        blaze: computeFit(te.blaze),
      },
    };

    return { tmData: tm, teData: te, fitCoeffs };
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

  // Common domain for both plots - extend x domain to cover fit line
  const xDomain: [number, number] = [10, 160];
  const yDomain: [number, number] = [3, 300];

  // Generate power-of-10 tick values for a given domain
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

  // Generate fit line points
  const generateFitLine = (coeffs: { a: number; b: number }) => {
    const points = [];
    // Start from xDomain[0] to ensure line reaches y-axis
    for (let x = xDomain[0]; x <= xDomain[1]; x *= 1.08) {
      const y = coeffs.a * Math.pow(x, coeffs.b);
      // Only include points within the y domain
      if (y >= yDomain[0] && y <= yDomain[1]) {
        points.push({ x, y });
      }
    }
    return points;
  };

  // Find a good position for the label along the fit line
  const getFitLabelPosition = (fitLine: { x: number; y: number }[], offset: number = 0) => {
    if (fitLine.length < 2) return { x: 0, y: 0 };
    // Use a point around 70% along the line
    const idx = Math.min(Math.floor(fitLine.length * 0.2), fitLine.length - 1);
    return { x: fitLine[idx].x, y: fitLine[idx].y * (1 + offset) };
  };

  // Generate O(N²) reference line
  const generateN2RefLine = (refX: number, refY: number) => {
    const points = [];
    for (let x = xDomain[0]; x <= xDomain[1]; x *= 1.1) {
      const y = refY * Math.pow(x / refX, 2);
      if (y >= yDomain[0] && y <= yDomain[1]) {
        points.push({ x, y });
      }
    }
    return points;
  };

  const renderChart = (
    data: { mpb: { x: number; y: number; std: number }[]; blaze: { x: number; y: number; std: number }[] },
    fits: { mpb: { a: number; b: number }; blaze: { a: number; b: number } },
    polarization: string
  ) => {
    const mpbFitLine = generateFitLine(fits.mpb);
    const blazeFitLine = generateFitLine(fits.blaze);
    
    // Generate N² reference line (use Blaze data as reference point, scaled down)
    const midIdx = Math.floor(data.blaze.length / 2);
    const refX = data.blaze[midIdx]?.x || 64;
    const refY = (data.blaze[midIdx]?.y || 20) / 5; // Divide by X to push line down
    const n2RefLine = generateN2RefLine(refX, refY);
    
    // Get label positions for fit lines
    const mpbLabelPos = getFitLabelPosition(mpbFitLine, 0.15);
    const blazeLabelPos = getFitLabelPosition(blazeFitLine, -0.25);

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
        {`Memory Scaling (${polarization})`}
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
        {n2RefLine.length > 0 && (
          <>
            <LinePath
              data={n2RefLine}
              x={(d) => xScale(d.x)}
              y={(d) => yScale(d.y)}
              stroke="#999"
              strokeWidth={1}
              strokeDasharray="4,4"
              strokeOpacity={0.7}
            />
            <text
              x={xScale(n2RefLine[n2RefLine.length - 1].x) - 30}
              y={yScale(n2RefLine[n2RefLine.length - 1].y) + 40}
              fontSize={11}
              fontFamily={CHART_STYLES.fontFamily}
              fill="#888"
            >
              O(N
              <tspan fontSize={8} dy={-3}>
                2
              </tspan>
              <tspan fontSize={11} dy={3}>
                )
              </tspan>
            </text>
          </>
        )}

        {/* MPB fit line (dashed) */}
        <LinePath
          data={mpbFitLine}
          x={(d) => xScale(d.x)}
          y={(d) => yScale(d.y)}
          stroke={COLORS.mpb}
          strokeWidth={1.5}
          strokeDasharray="5,5"
          strokeOpacity={0.6}
        />
        {/* MPB fit label */}
        {mpbLabelPos.x > 0 && (
          <text
            x={xScale(mpbLabelPos.x)}
            y={yScale(mpbLabelPos.y)}
            fontSize={13}
            fontFamily={CHART_STYLES.fontFamily}
            fill={COLORS.frozen}
          >
            ∝ N
            <tspan fontSize={9} dy={-5}>
              {fits.mpb.b.toFixed(2)}
            </tspan>
          </text>
        )}

        {/* Blaze fit line (dashed) */}
        <LinePath
          data={blazeFitLine}
          x={(d) => xScale(d.x)}
          y={(d) => yScale(d.y)}
          stroke={COLORS.blaze}
          strokeWidth={1.5}
          strokeDasharray="5,5"
          strokeOpacity={0.6}
        />
        {/* Blaze fit label */}
        {blazeLabelPos.x > 0 && (
          <text
            x={xScale(blazeLabelPos.x)}
            y={yScale(blazeLabelPos.y)}
            fontSize={13}
            fontFamily={CHART_STYLES.fontFamily}
            fill={COLORS.blaze}
          >
            ∝ N
            <tspan fontSize={9} dy={-5}>
              {fits.blaze.b.toFixed(2)}
            </tspan>
          </text>
        )}

        {/* MPB error bars */}
        {data.mpb.map((d, i) => d.std > 0 && (
          <g key={`mpb-err-${i}`}>
            <line
              x1={xScale(d.x)}
              y1={yScale(d.y - d.std)}
              x2={xScale(d.x)}
              y2={yScale(d.y + d.std)}
              stroke={COLORS.mpb}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
            <line
              x1={xScale(d.x) - 3}
              y1={yScale(d.y - d.std)}
              x2={xScale(d.x) + 3}
              y2={yScale(d.y - d.std)}
              stroke={COLORS.mpb}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
            <line
              x1={xScale(d.x) - 3}
              y1={yScale(d.y + d.std)}
              x2={xScale(d.x) + 3}
              y2={yScale(d.y + d.std)}
              stroke={COLORS.mpb}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
          </g>
        ))}

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

        {/* Blaze error bars */}
        {data.blaze.map((d, i) => d.std > 0 && (
          <g key={`blaze-err-${i}`}>
            <line
              x1={xScale(d.x)}
              y1={yScale(d.y - d.std)}
              x2={xScale(d.x)}
              y2={yScale(d.y + d.std)}
              stroke={COLORS.blaze}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
            <line
              x1={xScale(d.x) - 3}
              y1={yScale(d.y - d.std)}
              x2={xScale(d.x) + 3}
              y2={yScale(d.y - d.std)}
              stroke={COLORS.blaze}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
            <line
              x1={xScale(d.x) - 3}
              y1={yScale(d.y + d.std)}
              x2={xScale(d.x) + 3}
              y2={yScale(d.y + d.std)}
              stroke={COLORS.blaze}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
          </g>
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
          tickFormat={(v) => `${v}`}
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
          Peak Memory (MB)
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
        {/* MPB */}
        <circle cx={6} cy={0} r={5} fill={COLORS.mpb} />
        <Text x={16} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
          MPB
        </Text>

        {/* Blaze */}
        <rect x={60} y={-5} width={10} height={10} fill={COLORS.blaze} />
        <Text x={76} y={4} fontSize={11} fill={CHART_STYLES.labelColor} fontFamily={CHART_STYLES.fontFamily}>
          Blaze2D
        </Text>
      </Group>
    </svg>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
        {renderChart(tmData, fitCoeffs.tm, 'TM')}
        {renderChart(teData, fitCoeffs.te, 'TE')}
      </div>
      <p style={{
        marginTop: '1rem',
        fontSize: '0.875rem',
        color: '#888',
        lineHeight: 1.5,
        fontFamily: 'var(--font-sans), system-ui, sans-serif',
        fontStyle: 'italic',
        textAlign: 'left',
      }}>
        Log-log plot of memory scaling with resolution. Dashed lines show power-law fits to the measured data.
      </p>
    </div>
  );
}
