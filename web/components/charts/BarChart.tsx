'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { Bar } from '@visx/shape';
import { scaleLinear, scaleBand, scaleOrdinal } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { Text } from '@visx/text';

export interface BarDataPoint {
  /** Unique identifier for this data point (used for keys). Falls back to label if not provided */
  id?: string;
  /** Display label shown on the x-axis */
  label: string;
  value: number;
  std?: number;
  color?: string;
  group?: string;
}

export interface BarChartProps {
  data: BarDataPoint[];
  width?: number;
  height?: number;
  title?: string;
  caption?: string;
  yLabel?: string;
  yTickFormat?: (value: number) => string;
  labelAngle?: number;
  showValues?: boolean;
  valueFormat?: (value: number) => string;
  showStd?: boolean;
  defaultBarColor?: string;
  margin?: { top: number; right: number; bottom: number; left: number };
  groupColors?: Record<string, string>;
  /** Category brackets group adjacent labels under a shared label. Use labelIndices (0-based) to specify which labels to group */
  categoryBrackets?: { labelIndices: [number, number]; category: string }[];
  showCategoryBrackets?: boolean;
  /** Vertical offset for category brackets from bottom of plot area (default: 50) */
  bracketOffset?: number;
}

const defaultMargin = { top: 60, right: 30, bottom: 100, left: 70 };

// Shared chart styling constants - export for use in other chart components
export const CHART_STYLES = {
  labelColor: '#ffffff',
  gridColor: '#333',
  axisColor: '#555',
  captionColor: '#888',
  barBorderColor: '#eaf1fe',
  fontFamily: 'var(--font-sans), system-ui, sans-serif',
};

export default function BarChart({
  data,
  width = 600,
  height = 400,
  title,
  caption,
  yLabel,
  yTickFormat = (v) => `${v}`,
  labelAngle = -45,
  showValues = true,
  valueFormat = (v) => v.toFixed(0),
  showStd = true,
  defaultBarColor = '#3b82f6',
  margin = defaultMargin,
  groupColors,
  categoryBrackets,
  showCategoryBrackets = false,
  bracketOffset = 50,
}: BarChartProps) {
  // Calculate inner dimensions
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Determine if we have groups
  const hasGroups = data.some(d => d.group);
  const groups = hasGroups 
    ? [...new Set(data.map(d => d.group).filter(Boolean))] as string[]
    : [];
  
  // Get unique data point identifiers (use id if present, otherwise label)
  const getDataId = (d: BarDataPoint) => d.id || d.label;
  
  // Get unique position IDs in order of first appearance
  // For grouped charts, multiple data points share the same position ID (one per group)
  const positionIds = useMemo(() => {
    const seen = new Set<string>();
    const ids: string[] = [];
    data.forEach(d => {
      const id = getDataId(d);
      if (!seen.has(id)) {
        seen.add(id);
        ids.push(id);
      }
    });
    return ids;
  }, [data]);

  // Scales
  const xScale = useMemo(() => {
    if (hasGroups) {
      return scaleBand<string>({
        domain: positionIds,
        range: [0, innerWidth],
        padding: 0.2,
      });
    }
    return scaleBand<string>({
      domain: data.map(d => getDataId(d)),
      range: [0, innerWidth],
      padding: 0.3,
    });
  }, [data, positionIds, innerWidth, hasGroups]);

  const groupScale = useMemo(() => {
    if (!hasGroups) return null;
    return scaleBand<string>({
      domain: groups,
      range: [0, xScale.bandwidth()],
      padding: 0.1,
    });
  }, [groups, xScale, hasGroups]);

  const maxValue = useMemo(() => {
    return Math.max(...data.map(d => d.value + (d.std || 0))) * 1.15;
  }, [data]);

  const yScale = useMemo(() => {
    return scaleLinear<number>({
      domain: [0, maxValue],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [maxValue, innerHeight]);

  // Color scale for groups
  const colorScale = useMemo(() => {
    if (groupColors) {
      return (group: string) => groupColors[group] || defaultBarColor;
    }
    const defaultGroupColors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b'];
    return scaleOrdinal<string, string>({
      domain: groups,
      range: defaultGroupColors,
    });
  }, [groups, groupColors, defaultBarColor]);

  const getBarColor = (d: BarDataPoint) => {
    if (d.color) return d.color;
    if (d.group && hasGroups) return colorScale(d.group);
    return defaultBarColor;
  };

  return (
    <div className="bar-chart-container" style={{ width: '100%', maxWidth: width }}>
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

          {/* Bars */}
          {hasGroups ? (
            // Grouped bars - iterate over unique position IDs
            positionIds.map((posId) => {
              // Get all data points for this position (one per group)
              const positionData = data.filter(d => getDataId(d) === posId);
              const x0 = xScale(posId) || 0;
              
              return (
                <Group key={posId} left={x0}>
                  {positionData.map((d) => {
                    const barWidth = groupScale?.bandwidth() || 0;
                    const barX = groupScale?.(d.group || '') || 0;
                    const barHeight = innerHeight - yScale(d.value);
                    const barY = yScale(d.value);
                    const color = getBarColor(d);

                    return (
                      <Group key={`${getDataId(d)}-${d.group}`}>
                        <Bar
                          x={barX}
                          y={barY}
                          width={barWidth}
                          height={barHeight}
                          fill={color}
                          stroke={CHART_STYLES.barBorderColor}
                          strokeWidth={1}
                          rx={2}
                        />
                        
                        {/* Error bar */}
                        {showStd && d.std && d.std > 0 && (
                          <>
                            <line
                              x1={barX + barWidth / 2}
                              x2={barX + barWidth / 2}
                              y1={yScale(d.value - d.std)}
                              y2={yScale(d.value + d.std)}
                              stroke="#fff"
                              strokeWidth={1.5}
                              opacity={0.8}
                            />
                            <line
                              x1={barX + barWidth / 2 - 4}
                              x2={barX + barWidth / 2 + 4}
                              y1={yScale(d.value + d.std)}
                              y2={yScale(d.value + d.std)}
                              stroke="#fff"
                              strokeWidth={1.5}
                              opacity={0.8}
                            />
                            <line
                              x1={barX + barWidth / 2 - 4}
                              x2={barX + barWidth / 2 + 4}
                              y1={yScale(d.value - d.std)}
                              y2={yScale(d.value - d.std)}
                              stroke="#fff"
                              strokeWidth={1.5}
                              opacity={0.8}
                            />
                          </>
                        )}

                        {/* Value label */}
                        {showValues && (
                          <Text
                            x={barX + barWidth / 2}
                            y={barY - 8}
                            fontSize={11}
                            fontFamily={CHART_STYLES.fontFamily}
                            fill={CHART_STYLES.labelColor}
                            textAnchor="middle"
                          >
                            {valueFormat(d.value)}
                          </Text>
                        )}
                      </Group>
                    );
                  })}
                </Group>
              );
            })
          ) : (
            // Simple bars
            data.map((d) => {
              const barWidth = xScale.bandwidth();
              const barX = xScale(d.label) || 0;
              const barHeight = innerHeight - yScale(d.value);
              const barY = yScale(d.value);
              const color = getBarColor(d);

              return (
                <Group key={d.label}>
                  <Bar
                    x={barX}
                    y={barY}
                    width={barWidth}
                    height={barHeight}
                    fill={color}
                    stroke={CHART_STYLES.barBorderColor}
                    strokeWidth={1}
                    rx={2}
                  />

                  {/* Error bar */}
                  {showStd && d.std && d.std > 0 && (
                    <>
                      <line
                        x1={barX + barWidth / 2}
                        x2={barX + barWidth / 2}
                        y1={yScale(d.value - d.std)}
                        y2={yScale(d.value + d.std)}
                        stroke="#fff"
                        strokeWidth={1.5}
                        opacity={0.8}
                      />
                      <line
                        x1={barX + barWidth / 2 - 4}
                        x2={barX + barWidth / 2 + 4}
                        y1={yScale(d.value + d.std)}
                        y2={yScale(d.value + d.std)}
                        stroke="#fff"
                        strokeWidth={1.5}
                        opacity={0.8}
                      />
                      <line
                        x1={barX + barWidth / 2 - 4}
                        x2={barX + barWidth / 2 + 4}
                        y1={yScale(d.value - d.std)}
                        y2={yScale(d.value - d.std)}
                        stroke="#fff"
                        strokeWidth={1.5}
                        opacity={0.8}
                      />
                    </>
                  )}

                  {/* Value label */}
                  {showValues && (
                    <Text
                      x={barX + barWidth / 2}
                      y={barY - 8}
                      fontSize={11}
                      fontFamily={CHART_STYLES.fontFamily}
                      fill={CHART_STYLES.labelColor}
                      textAnchor="middle"
                    >
                      {valueFormat(d.value)}
                    </Text>
                  )}
                </Group>
              );
            })
          )}

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
            tickFormat={(v) => yTickFormat(v as number)}
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
            tickLabelProps={(value) => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: labelAngle < 0 ? 'end' : labelAngle > 0 ? 'start' : 'middle',
              dy: labelAngle !== 0 ? '0.25em' : '0.5em',
              dx: labelAngle < 0 ? -4 : labelAngle > 0 ? 4 : 0,
              angle: labelAngle,
            })}
            tickFormat={(posId) => {
              // Find the display label for this position ID
              const dataPoint = data.find(d => getDataId(d) === posId);
              return dataPoint?.label || String(posId);
            }}
          />

          {/* Category brackets */}
          {showCategoryBrackets && categoryBrackets && categoryBrackets.map((bracket, i) => {
            const [firstIdx, lastIdx] = bracket.labelIndices;
            const firstPosId = positionIds[firstIdx];
            const lastPosId = positionIds[lastIdx];
            
            if (!firstPosId || !lastPosId) return null;
            
            const x1 = (xScale(firstPosId) || 0) + xScale.bandwidth() * 0.1;
            const x2 = (xScale(lastPosId) || 0) + xScale.bandwidth() * 0.9;
            const bracketY = innerHeight + bracketOffset;
            const bracketHeight = 8;

            return (
              <Group key={`bracket-${i}`}>
                {/* Left vertical line */}
                <line
                  x1={x1}
                  x2={x1}
                  y1={bracketY}
                  y2={bracketY + bracketHeight}
                  stroke={CHART_STYLES.axisColor}
                  strokeWidth={1}
                />
                {/* Horizontal line */}
                <line
                  x1={x1}
                  x2={x2}
                  y1={bracketY + bracketHeight}
                  y2={bracketY + bracketHeight}
                  stroke={CHART_STYLES.axisColor}
                  strokeWidth={1}
                />
                {/* Right vertical line */}
                <line
                  x1={x2}
                  x2={x2}
                  y1={bracketY}
                  y2={bracketY + bracketHeight}
                  stroke={CHART_STYLES.axisColor}
                  strokeWidth={1}
                />
                {/* Category label */}
                <Text
                  x={(x1 + x2) / 2}
                  y={bracketY + bracketHeight + 14}
                  fontSize={11}
                  fontFamily={CHART_STYLES.fontFamily}
                  fill={CHART_STYLES.labelColor}
                  textAnchor="middle"
                >
                  {bracket.category}
                </Text>
              </Group>
            );
          })}

        </Group>

        {/* Legend for grouped bars - aligned to y-axis, above plot area */}
        {hasGroups && groups.length > 0 && (
          <Group top={margin.top - 22} left={margin.left}>
            {groups.map((group, i) => (
              <Group key={group} left={i * 80}>
                <rect
                  width={12}
                  height={12}
                  fill={colorScale(group)}
                  rx={2}
                />
                <Text
                  x={18}
                  y={10}
                  fontSize={11}
                  fontFamily={CHART_STYLES.fontFamily}
                  fill={CHART_STYLES.labelColor}
                >
                  {group}
                </Text>
              </Group>
            ))}
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
          maxWidth: width,
          fontStyle: 'italic',
        }}>
          {caption}
        </p>
      )}
    </div>
  );
}
