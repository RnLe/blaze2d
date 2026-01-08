'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { LinePath } from '@visx/shape';
import { scaleLinear } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';
import { CHART_STYLES } from './BarChart';

export interface LineDataPoint {
  x: number;
  y: number;
  std?: number;
}

export interface LineSeries {
  id: string;
  label: string;
  data: LineDataPoint[];
  color: string;
  /** Shape of the data point marker: 'circle', 'square', 'triangle' */
  marker?: 'circle' | 'square' | 'triangle';
  /** Whether this is a reference/ideal line (dashed, no markers) */
  isReference?: boolean;
}

export interface LineChartProps {
  series: LineSeries[];
  width?: number;
  height?: number;
  title?: string;
  caption?: string;
  xLabel?: string;
  yLabel?: string;
  xTickFormat?: (value: number) => string;
  yTickFormat?: (value: number) => string;
  showLegend?: boolean;
  showGrid?: boolean;
  showErrorBars?: boolean;
  margin?: { top: number; right: number; bottom: number; left: number };
  /** Force x-axis domain [min, max] */
  xDomain?: [number, number];
  /** Force y-axis domain [min, max] */
  yDomain?: [number, number];
}

const defaultMargin = { top: 60, right: 30, bottom: 60, left: 70 };

// Marker size
const MARKER_SIZE = 6;

/**
 * Render a marker shape at a given position
 */
function Marker({ 
  x, 
  y, 
  shape, 
  color, 
  size = MARKER_SIZE 
}: { 
  x: number; 
  y: number; 
  shape: 'circle' | 'square' | 'triangle'; 
  color: string;
  size?: number;
}) {
  switch (shape) {
    case 'circle':
      return (
        <circle
          cx={x}
          cy={y}
          r={size}
          fill={color}
        />
      );
    case 'square':
      return (
        <rect
          x={x - size}
          y={y - size}
          width={size * 2}
          height={size * 2}
          fill={color}
        />
      );
    case 'triangle':
      const h = size * 1.7;
      const points = [
        [x, y - h * 0.6],
        [x - size, y + h * 0.4],
        [x + size, y + h * 0.4],
      ].map(p => p.join(',')).join(' ');
      return (
        <polygon
          points={points}
          fill={color}
        />
      );
    default:
      return null;
  }
}

export default function LineChart({
  series,
  width = 600,
  height = 400,
  title,
  caption,
  xLabel,
  yLabel,
  xTickFormat = (v) => `${v}`,
  yTickFormat = (v) => `${v}`,
  showLegend = true,
  showGrid = true,
  showErrorBars = false,
  margin = defaultMargin,
  xDomain,
  yDomain,
}: LineChartProps) {
  // Calculate inner dimensions
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Calculate domains from data if not provided
  const { computedXDomain, computedYDomain } = useMemo(() => {
    const allX = series.flatMap(s => s.data.map(d => d.x));
    const allY = series.flatMap(s => s.data.map(d => d.y + (showErrorBars && d.std ? d.std : 0)));
    const allYMin = series.flatMap(s => s.data.map(d => d.y - (showErrorBars && d.std ? d.std : 0)));
    
    const xMin = Math.min(...allX);
    const xMax = Math.max(...allX);
    const yMin = Math.min(0, ...allYMin);
    const yMax = Math.max(...allY) * 1.1;

    return {
      computedXDomain: xDomain || [xMin, xMax],
      computedYDomain: yDomain || [yMin, yMax],
    };
  }, [series, xDomain, yDomain, showErrorBars]);

  // Scales
  const xScale = useMemo(() => {
    return scaleLinear<number>({
      domain: computedXDomain,
      range: [0, innerWidth],
      nice: true,
    });
  }, [computedXDomain, innerWidth]);

  const yScale = useMemo(() => {
    return scaleLinear<number>({
      domain: computedYDomain,
      range: [innerHeight, 0],
      nice: true,
    });
  }, [computedYDomain, innerHeight]);

  // Legend layout - compute wrapping
  const legendItemWidths = useMemo(() => {
    // Estimate widths based on label lengths (rough approximation: 7px per char + marker space)
    return series.map(s => Math.max(70, s.label.length * 7 + 30));
  }, [series]);

  const legendLayout = useMemo(() => {
    const maxWidth = innerWidth;
    const items: { seriesIdx: number; x: number; row: number }[] = [];
    let currentX = 0;
    let currentRow = 0;
    const rowHeight = 20;
    const itemGap = 16;

    series.forEach((s, i) => {
      const itemWidth = legendItemWidths[i];
      if (currentX + itemWidth > maxWidth && currentX > 0) {
        currentRow++;
        currentX = 0;
      }
      items.push({ seriesIdx: i, x: currentX, row: currentRow });
      currentX += itemWidth + itemGap;
    });

    return { items, totalRows: currentRow + 1, rowHeight };
  }, [series, legendItemWidths, innerWidth]);

  return (
    <div className="line-chart-container" style={{ width: '100%', maxWidth: width }}>
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
          {showGrid && (
            <GridRows
              scale={yScale}
              width={innerWidth}
              stroke={CHART_STYLES.gridColor}
              strokeOpacity={0.3}
              strokeDasharray="3,3"
            />
          )}

          {/* Lines and markers */}
          {series.map((s) => (
            <Group key={s.id}>
              {/* Line */}
              <LinePath
                data={s.data}
                x={(d) => xScale(d.x)}
                y={(d) => yScale(d.y)}
                stroke={s.color}
                strokeWidth={s.isReference ? 1.5 : 2}
                strokeDasharray={s.isReference ? '6,4' : undefined}
                strokeOpacity={s.isReference ? 0.7 : 1}
              />

              {/* Error bars */}
              {showErrorBars && !s.isReference && s.data.map((d, i) => {
                if (!d.std) return null;
                const x = xScale(d.x);
                const yTop = yScale(d.y + d.std);
                const yBottom = yScale(d.y - d.std);
                const capWidth = 4;
                return (
                  <Group key={`error-${s.id}-${i}`}>
                    <line
                      x1={x}
                      y1={yTop}
                      x2={x}
                      y2={yBottom}
                      stroke={s.color}
                      strokeWidth={1.5}
                    />
                    <line
                      x1={x - capWidth}
                      y1={yTop}
                      x2={x + capWidth}
                      y2={yTop}
                      stroke={s.color}
                      strokeWidth={1.5}
                    />
                    <line
                      x1={x - capWidth}
                      y1={yBottom}
                      x2={x + capWidth}
                      y2={yBottom}
                      stroke={s.color}
                      strokeWidth={1.5}
                    />
                  </Group>
                );
              })}

              {/* Markers */}
              {!s.isReference && s.data.map((d, i) => (
                <Marker
                  key={`marker-${s.id}-${i}`}
                  x={xScale(d.x)}
                  y={yScale(d.y)}
                  shape={s.marker || 'circle'}
                  color={s.color}
                />
              ))}
            </Group>
          ))}

          {/* Y-Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => yTickFormat(v as number)}
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

          {/* X-Axis */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => xTickFormat(v as number)}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'middle' as const,
              dy: 4,
            })}
            numTicks={6}
          />

          {/* Y-axis label */}
          {yLabel && (
            <Text
              x={-innerHeight / 2}
              y={-50}
              fontSize={12}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
              textAnchor="middle"
              transform="rotate(-90)"
            >
              {yLabel}
            </Text>
          )}

          {/* X-axis label */}
          {xLabel && (
            <Text
              x={innerWidth / 2}
              y={innerHeight + 45}
              fontSize={12}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
              textAnchor="middle"
            >
              {xLabel}
            </Text>
          )}
        </Group>

        {/* Legend - top left inside plot area, aligned with y-axis */}
        {showLegend && (
          <Group top={margin.top - 22} left={margin.left}>
            {legendLayout.items.map(({ seriesIdx, x, row }) => {
              const s = series[seriesIdx];
              const y = row * legendLayout.rowHeight;
              
              if (s.isReference) {
                return (
                  <Group key={`legend-${s.id}`} left={x} top={y}>
                    <line
                      x1={0}
                      y1={6}
                      x2={16}
                      y2={6}
                      stroke={s.color}
                      strokeWidth={1.5}
                      strokeDasharray="4,3"
                      strokeOpacity={0.7}
                    />
                    <Text
                      x={20}
                      y={10}
                      fontSize={11}
                      fontFamily={CHART_STYLES.fontFamily}
                      fill={CHART_STYLES.labelColor}
                    >
                      {s.label}
                    </Text>
                  </Group>
                );
              }
              
              return (
                <Group key={`legend-${s.id}`} left={x} top={y}>
                  <Marker
                    x={6}
                    y={6}
                    shape={s.marker || 'circle'}
                    color={s.color}
                    size={5}
                  />
                  <Text
                    x={18}
                    y={10}
                    fontSize={11}
                    fontFamily={CHART_STYLES.fontFamily}
                    fill={CHART_STYLES.labelColor}
                  >
                    {s.label}
                  </Text>
                </Group>
              );
            })}
          </Group>
        )}
      </svg>

      {/* Caption */}
      {caption && (
        <p style={{
          marginTop: '1rem',
          fontSize: '0.875rem',
          color: CHART_STYLES.captionColor,
          lineHeight: 1.5,
          fontFamily: CHART_STYLES.fontFamily,
          fontStyle: 'italic',
          textAlign: 'left',
        }}>
          {caption}
        </p>
      )}
    </div>
  );
}
