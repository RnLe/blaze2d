'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { scaleLinear, scaleBand } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';
import { CHART_STYLES } from './BarChart';

export interface BoxPlotDataPoint {
  /** Label shown on the x-axis */
  label: string;
  /** Minimum value (excluding outliers) */
  whiskerLow: number;
  /** First quartile (25th percentile) */
  q1: number;
  /** Median (50th percentile) */
  median: number;
  /** Third quartile (75th percentile) */
  q3: number;
  /** Maximum value (excluding outliers) */
  whiskerHigh: number;
  /** Optional: mean value to show as a marker */
  mean?: number;
  /** Color for the box */
  color?: string;
}

export interface BoxPlotProps {
  data: BoxPlotDataPoint[];
  width?: number;
  height?: number;
  title?: string;
  caption?: string;
  yLabel?: string;
  yTickFormat?: (value: number) => string;
  /** Use logarithmic scale for y-axis */
  logScale?: boolean;
  defaultBoxColor?: string;
  margin?: { top: number; right: number; bottom: number; left: number };
  /** Width of each box as fraction of available space (0-1) */
  boxWidth?: number;
}

const defaultMargin = { top: 60, right: 30, bottom: 60, left: 70 };

export default function BoxPlot({
  data,
  width = 600,
  height = 400,
  title,
  caption,
  yLabel,
  yTickFormat = (v) => v.toExponential(1),
  logScale = false,
  defaultBoxColor = '#3b82f6',
  margin = defaultMargin,
  boxWidth = 0.6,
}: BoxPlotProps) {
  // Calculate inner dimensions
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // X scale (categorical)
  const xScale = useMemo(() => {
    return scaleBand<string>({
      domain: data.map(d => d.label),
      range: [0, innerWidth],
      padding: 0.3,
    });
  }, [data, innerWidth]);

  // Y scale (linear or log)
  const yScale = useMemo(() => {
    const allValues = data.flatMap(d => [d.whiskerLow, d.whiskerHigh]);
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    
    // Add some padding
    const padding = (maxVal - minVal) * 0.1;
    
    if (logScale) {
      // For log scale, ensure minimum is positive
      const logMin = minVal > 0 ? minVal * 0.5 : 1e-10;
      const logMax = maxVal * 2;
      return scaleLinear<number>({
        domain: [Math.log10(logMin), Math.log10(logMax)],
        range: [innerHeight, 0],
        nice: true,
      });
    }
    
    return scaleLinear<number>({
      domain: [Math.max(0, minVal - padding), maxVal + padding],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [data, innerHeight, logScale]);

  // Helper to convert value to y position
  const toY = (value: number) => {
    if (logScale) {
      return yScale(Math.log10(Math.max(value, 1e-10)));
    }
    return yScale(value);
  };

  // Box visual width
  const actualBoxWidth = xScale.bandwidth() * boxWidth;

  return (
    <div className="boxplot-container" style={{ width: '100%', maxWidth: width }}>
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        {/* Title - left aligned to component edge */}
        {title && (
          <Text
            x={0}
            y={16}
            fontSize={14}
            fontFamily={CHART_STYLES.fontFamily}
            fill={CHART_STYLES.labelColor}
            fontWeight={700}
          >
            {title}
          </Text>
        )}

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
          {data.map((d, i) => {
            const x = (xScale(d.label) || 0) + (xScale.bandwidth() - actualBoxWidth) / 2;
            const color = d.color || defaultBoxColor;
            
            const whiskerLowY = toY(d.whiskerLow);
            const q1Y = toY(d.q1);
            const medianY = toY(d.median);
            const q3Y = toY(d.q3);
            const whiskerHighY = toY(d.whiskerHigh);
            
            const boxHeight = q1Y - q3Y;
            const centerX = x + actualBoxWidth / 2;

            return (
              <Group key={`box-${i}`}>
                {/* Whisker line (vertical) */}
                <line
                  x1={centerX}
                  y1={whiskerLowY}
                  x2={centerX}
                  y2={whiskerHighY}
                  stroke={color}
                  strokeWidth={1.5}
                />
                
                {/* Lower whisker cap */}
                <line
                  x1={x + actualBoxWidth * 0.25}
                  y1={whiskerLowY}
                  x2={x + actualBoxWidth * 0.75}
                  y2={whiskerLowY}
                  stroke={color}
                  strokeWidth={2}
                />
                
                {/* Upper whisker cap */}
                <line
                  x1={x + actualBoxWidth * 0.25}
                  y1={whiskerHighY}
                  x2={x + actualBoxWidth * 0.75}
                  y2={whiskerHighY}
                  stroke={color}
                  strokeWidth={2}
                />
                
                {/* Box */}
                <rect
                  x={x}
                  y={q3Y}
                  width={actualBoxWidth}
                  height={boxHeight}
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
                  x2={x + actualBoxWidth}
                  y2={medianY}
                  stroke="#ffffff"
                  strokeWidth={2.5}
                />
                
                {/* Mean marker (diamond) */}
                {d.mean !== undefined && (
                  <polygon
                    points={`
                      ${centerX},${toY(d.mean) - 5}
                      ${centerX + 5},${toY(d.mean)}
                      ${centerX},${toY(d.mean) + 5}
                      ${centerX - 5},${toY(d.mean)}
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
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'end',
              dy: '0.33em',
              dx: -4,
            }}
            tickFormat={(v) => {
              if (logScale) {
                return yTickFormat(Math.pow(10, v as number));
              }
              return yTickFormat(v as number);
            }}
            numTicks={6}
          />

          {/* Y Label */}
          {yLabel && (
            <Text
              x={-innerHeight / 2}
              y={-50}
              transform="rotate(-90)"
              fontSize={12}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
              textAnchor="middle"
            >
              {yLabel}
            </Text>
          )}

          {/* X Axis */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickLabelProps={{
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'middle',
              dy: '0.5em',
            }}
          />
        </Group>
      </svg>

      {/* Caption */}
      {caption && (
        <p style={{
          marginTop: '1rem',
          fontSize: '0.875rem',
          color: CHART_STYLES.captionColor,
          lineHeight: 1.5,
          maxWidth: width,
          fontStyle: 'italic',
        }}>
          {caption}
        </p>
      )}
    </div>
  );
}
