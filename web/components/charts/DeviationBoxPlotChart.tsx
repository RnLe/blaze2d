'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { scaleLinear, scaleBand } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';
import useSWR from 'swr';
import { CHART_STYLES } from './BarChart';
import { getAssetPath } from '../../lib/paths';

interface BoxPlotStats {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  whiskerLow: number;
  whiskerHigh: number;
  mean: number;
}

interface Series6Data {
  parameters: {
    resolution: number;
    num_bands: number;
  };
  TM: {
    deviations: {
      f32_vs_mpb: BoxPlotStats | null;
      f64_vs_mpb: BoxPlotStats | null;
      f32_vs_f64: BoxPlotStats | null;
    };
  };
  TE: {
    deviations: {
      f32_vs_mpb: BoxPlotStats | null;
      f64_vs_mpb: BoxPlotStats | null;
      f32_vs_f64: BoxPlotStats | null;
    };
  };
  metadata: {
    timestamp: string;
  };
}

export interface DeviationBoxPlotChartProps {
  width?: number;
  height?: number;
  polarization?: 'TM' | 'TE';
}

const defaultMargin = { top: 60, right: 30, bottom: 60, left: 80 };

// Colors matching band diagram
const F64_COLOR = '#a3befa';  // Full precision - light blue
const F32_COLOR = '#bbc1cb';  // Mixed precision - gray
const COMPARE_COLOR = '#435f9d';  // f32 vs f64 - reference blue

const fetcher = (url: string) => fetch(url).then(res => res.json());

export default function DeviationBoxPlotChart({
  width = 400,
  height = 400,
  polarization = 'TM',
}: DeviationBoxPlotChartProps) {
  const { data, error, isLoading } = useSWR<Series6Data>(
    getAssetPath('/data/benchmarks/series6-accuracy.json'),
    fetcher
  );

  const margin = defaultMargin;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Extract deviation data for the selected polarization
  const deviations = data?.[polarization]?.deviations;

  // Prepare boxplot data
  const boxData = useMemo(() => {
    if (!deviations) return [];
    
    const items: { label: string; stats: BoxPlotStats; color: string }[] = [];
    
    if (deviations.f32_vs_mpb) {
      items.push({ label: 'Mixed (f32)', stats: deviations.f32_vs_mpb, color: F32_COLOR });
    }
    if (deviations.f64_vs_mpb) {
      items.push({ label: 'Full (f64)', stats: deviations.f64_vs_mpb, color: F64_COLOR });
    }
    if (deviations.f32_vs_f64) {
      items.push({ label: 'f32 vs f64', stats: deviations.f32_vs_f64, color: COMPARE_COLOR });
    }
    
    return items;
  }, [deviations]);

  // X scale (categorical)
  const xScale = useMemo(() => {
    return scaleBand<string>({
      domain: boxData.map(d => d.label),
      range: [0, innerWidth],
      padding: 0.4,
    });
  }, [boxData, innerWidth]);

  // Y scale (log scale for deviations)
  const { yScale, yMin, yMax } = useMemo(() => {
    // Gather all stats from both polarizations if available to ensure shared axis
    const allStats: BoxPlotStats[] = [];
    if (data?.TM?.deviations) {
      const dev = data.TM.deviations;
      if (dev.f32_vs_mpb) allStats.push(dev.f32_vs_mpb);
      if (dev.f64_vs_mpb) allStats.push(dev.f64_vs_mpb);
      if (dev.f32_vs_f64) allStats.push(dev.f32_vs_f64);
    }
    if (data?.TE?.deviations) {
      const dev = data.TE.deviations;
      if (dev.f32_vs_mpb) allStats.push(dev.f32_vs_mpb);
      if (dev.f64_vs_mpb) allStats.push(dev.f64_vs_mpb);
      if (dev.f32_vs_f64) allStats.push(dev.f32_vs_f64);
    }

    if (!allStats.length) {
      return { 
        yScale: scaleLinear<number>({ domain: [-6, -2], range: [innerHeight, 0] }),
        yMin: -6,
        yMax: -2
      };
    }

    // Get min/max from Q1 and Q3 (more stable than whiskers which can be 0)
    const allQ1 = allStats.map(d => d.q1).filter(v => v > 0);
    const allWhiskerHigh = allStats.map(d => d.whiskerHigh).filter(v => v > 0);
    
    // Use Q1 for min (since whiskerLow can be 0)
    const minVal = allQ1.length > 0 ? Math.min(...allQ1) * 0.1 : 1e-6;
    const maxVal = allWhiskerHigh.length > 0 ? Math.max(...allWhiskerHigh) * 5 : 1e-2;

    const logMin = Math.floor(Math.log10(minVal));
    const logMax = Math.ceil(Math.log10(maxVal));

    return {
      yScale: scaleLinear<number>({
        domain: [logMin, logMax],
        range: [innerHeight, 0],
      }),
      yMin: logMin,
      yMax: logMax
    };
  }, [data, innerHeight]);

  // Helper to convert value to y position (log scale)
  const toY = (value: number) => {
    // Clamp very small values to the bottom of the scale
    if (value <= 0) return innerHeight;
    return yScale(Math.log10(value));
  };

  // Box visual width
  const boxWidth = xScale.bandwidth() * 0.7;

  if (isLoading) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: CHART_STYLES.labelColor }}>
        Loading...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: CHART_STYLES.labelColor }}>
        Error loading data
      </div>
    );
  }

  return (
    <div className="deviation-boxplot-container" style={{ width: '100%', maxWidth: width }}>
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        {/* Title */}
        <Text
          x={0}
          y={16}
          fontSize={14}
          fontFamily={CHART_STYLES.fontFamily}
          fill={CHART_STYLES.labelColor}
          fontWeight={700}
        >
          {`${polarization} Deviation Distribution`}
        </Text>

        <Group left={margin.left} top={margin.top}>
          {/* Grid */}
          <GridRows
            scale={yScale}
            width={innerWidth}
            stroke={CHART_STYLES.gridColor}
            strokeOpacity={0.3}
            strokeDasharray="3,3"
          />

          {/* Box plots */}
          {boxData.map((d, i) => {
            const x = (xScale(d.label) || 0) + (xScale.bandwidth() - boxWidth) / 2;
            const color = d.color;
            const stats = d.stats;
            
            // For log scale, if whiskerLow is 0, use q1 as the lower bound
            const effectiveWhiskerLow = stats.whiskerLow > 0 ? stats.whiskerLow : stats.q1;
            
            const whiskerLowY = toY(effectiveWhiskerLow);
            const q1Y = toY(stats.q1);
            const medianY = toY(stats.median);
            const q3Y = toY(stats.q3);
            const whiskerHighY = toY(stats.whiskerHigh);
            
            const boxHeight = q1Y - q3Y;
            const centerX = x + boxWidth / 2;

            return (
              <Group key={`box-${i}`}>
                {/* Whisker line (vertical) - only draw if there's a valid lower whisker */}
                {stats.whiskerLow > 0 ? (
                  <>
                    <line
                      x1={centerX}
                      y1={whiskerLowY}
                      x2={centerX}
                      y2={q1Y}
                      stroke={color}
                      strokeWidth={1.5}
                    />
                    {/* Lower whisker cap */}
                    <line
                      x1={x + boxWidth * 0.2}
                      y1={whiskerLowY}
                      x2={x + boxWidth * 0.8}
                      y2={whiskerLowY}
                      stroke={color}
                      strokeWidth={2}
                    />
                  </>
                ) : (
                  /* When whiskerLow is 0, draw line from q1 to x-axis */
                  <line
                    x1={centerX}
                    y1={q1Y}
                    x2={centerX}
                    y2={innerHeight}
                    stroke={color}
                    strokeWidth={1.5}
                    strokeDasharray="4,2"
                  />
                )}
                
                {/* Upper whisker line */}
                <line
                  x1={centerX}
                  y1={q3Y}
                  x2={centerX}
                  y2={whiskerHighY}
                  stroke={color}
                  strokeWidth={1.5}
                />
                
                {/* Upper whisker cap */}
                <line
                  x1={x + boxWidth * 0.2}
                  y1={whiskerHighY}
                  x2={x + boxWidth * 0.8}
                  y2={whiskerHighY}
                  stroke={color}
                  strokeWidth={2}
                />
                
                {/* Box */}
                <rect
                  x={x}
                  y={q3Y}
                  width={boxWidth}
                  height={Math.max(0, boxHeight)}
                  fill={color}
                  fillOpacity={0.5}
                  stroke={color}
                  strokeWidth={2}
                  rx={2}
                />
                
                {/* Median line */}
                <line
                  x1={x}
                  y1={medianY}
                  x2={x + boxWidth}
                  y2={medianY}
                  stroke="#ffffff"
                  strokeWidth={1.5}
                />
                
                {/* Mean marker (diamond) */}
                {stats.mean !== undefined && (
                  <polygon
                    points={`
                      ${centerX},${toY(stats.mean) - 5}
                      ${centerX + 5},${toY(stats.mean)}
                      ${centerX},${toY(stats.mean) + 5}
                      ${centerX - 5},${toY(stats.mean)}
                    `}
                    fill="#ffffff"
                    stroke={color}
                    strokeWidth={1}
                  />
                )}
              </Group>
            );
          })}

          {/* Y Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickLabelProps={{
              fill: CHART_STYLES.labelColor,
              fontSize: 10,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'end',
              dy: '0.33em',
              dx: -4,
            }}
            tickValues={Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i)}
            tickFormat={(v) => {
              const val = Math.pow(10, v as number);
              return val.toExponential(0);
            }}
          />

          {/* Y Label */}
          <Text
            x={-innerHeight / 2}
            y={-60}
            transform="rotate(-90)"
            fontSize={12}
            fontFamily={CHART_STYLES.fontFamily}
            fill={CHART_STYLES.labelColor}
            textAnchor="middle"
          >
            Relative Deviation
          </Text>

          {/* X Axis - line only, no tick labels */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            stroke={CHART_STYLES.axisColor}
            tickStroke="transparent"
            hideTicks
            tickLabelProps={{
              fill: 'transparent',
              fontSize: 0,
            }}
          />
        </Group>

        {/* Legend - solver colors */}
        <Group top={margin.top - 22} left={margin.left}>
          {/* Mixed precision (f32) vs MPB */}
          <Group left={0}>
            <circle cx={6} cy={6} r={5} fill={F32_COLOR} />
            <Text
              x={16}
              y={10}
              fontSize={10}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
            >
              Blaze f32
            </Text>
          </Group>
          
          {/* Full precision (f64) vs MPB */}
          <Group left={85}>
            <circle cx={6} cy={6} r={5} fill={F64_COLOR} />
            <Text
              x={16}
              y={10}
              fontSize={10}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
            >
              Blaze f64
            </Text>
          </Group>
          
          {/* f32 vs f64 */}
          <Group left={175}>
            <circle cx={6} cy={6} r={5} fill={COMPARE_COLOR} />
            <Text
              x={16}
              y={10}
              fontSize={10}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
            >
              f32 vs f64
            </Text>
          </Group>
        </Group>
      </svg>

      {/* Caption */}
      <p style={{
        marginTop: '1rem',
        fontSize: '0.875rem',
        color: CHART_STYLES.captionColor,
        lineHeight: 1.5,
        maxWidth: width,
        fontStyle: 'italic',
      }}>
        {`${polarization} relative deviation distribution (log scale). Dashed line indicates minimum = 0 (perfect match).`}
      </p>
    </div>
  );
}
