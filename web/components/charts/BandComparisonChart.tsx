'use client';

import { useMemo } from 'react';
import { Group } from '@visx/group';
import { LinePath } from '@visx/shape';
import { scaleLinear } from '@visx/scale';
import { AxisLeft, AxisBottom } from '@visx/axis';
import { GridRows, GridColumns } from '@visx/grid';
import { Text } from '@visx/text';
import useSWR from 'swr';
import { CHART_STYLES } from './BarChart';
import { getAssetPath } from '../../lib/paths';

interface KPointData {
  k_distance: number;
  frequencies: number[];
}

interface Series6Data {
  parameters: {
    resolution: number;
    num_bands: number;
    k_points_per_segment: number;
    total_k_points: number;
    epsilon: number;
    radius: number;
  };
  TM: {
    mpb: KPointData[];
    blaze_f32: KPointData[];
    blaze_f64: KPointData[];
  };
  TE: {
    mpb: KPointData[];
    blaze_f32: KPointData[];
    blaze_f64: KPointData[];
  };
  metadata: {
    timestamp: string;
  };
}

export interface BandComparisonChartProps {
  width?: number;
  height?: number;
  polarization?: 'TM' | 'TE';
}

const defaultMargin = { top: 60, right: 30, bottom: 60, left: 70 };

// Colors
const MPB_COLOR = '#435f9d';  // Reference blue
const F64_COLOR = '#4caf50';  // Full precision - green
const F32_COLOR = '#bbc1cb';  // Mixed precision - gray

// Marker sizes
const F64_MARKER_SIZE = 3;
const F32_MARKER_SIZE = 2.5;

const fetcher = (url: string) => fetch(url).then(res => res.json());

/**
 * Render a circle marker
 */
function CircleMarker({ x, y, size, color, opacity = 1 }: { x: number; y: number; size: number; color: string; opacity?: number }) {
  return (
    <circle
      cx={x}
      cy={y}
      r={size}
      fill={color}
      fillOpacity={opacity}
    />
  );
}

/**
 * Render a triangle marker (pointing up)
 */
function TriangleMarker({ x, y, size, color, opacity = 1 }: { x: number; y: number; size: number; color: string; opacity?: number }) {
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
      fillOpacity={opacity}
    />
  );
}

export default function BandComparisonChart({
  width = 600,
  height = 400,
  polarization = 'TM',
}: BandComparisonChartProps) {
  const { data, error, isLoading } = useSWR<Series6Data>(
    getAssetPath('/data/benchmarks/series6-accuracy.json'),
    fetcher
  );

  const margin = defaultMargin;
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Extract data for the selected polarization
  const polData = data?.[polarization];
  const mpbData = polData?.mpb || [];
  const f32Data = polData?.blaze_f32 || [];
  const f64Data = polData?.blaze_f64 || [];
  const numBands = data?.parameters?.num_bands || 8;

  // Compute scales
  const { xScale, yScale } = useMemo(() => {
    if (!mpbData.length) {
      return {
        xScale: scaleLinear<number>({ domain: [0, 1], range: [0, innerWidth] }),
        yScale: scaleLinear<number>({ domain: [0, 1], range: [innerHeight, 0] }),
      };
    }

    const xMax = Math.max(...mpbData.map(d => d.k_distance));
    const allFreqs = mpbData.flatMap(d => d.frequencies);
    const yMax = Math.max(...allFreqs) * 1.05;

    return {
      xScale: scaleLinear<number>({
        domain: [0, xMax],
        range: [0, innerWidth],
      }),
      yScale: scaleLinear<number>({
        domain: [0, yMax],
        range: [innerHeight, 0],
      }),
    };
  }, [mpbData, innerWidth, innerHeight]);

  // Find high-symmetry points (Γ, X, M, Γ for square lattice)
  const highSymmetryPoints = useMemo(() => {
    if (!mpbData.length) return [];
    
    const points: { distance: number; label: string }[] = [];
    
    // First point is Γ
    points.push({ distance: 0, label: 'Γ' });
    
    // Find X (k = 0.5, 0)
    const xPoint = mpbData.find((_, i) => {
      if (i === 0) return false;
      const prev = mpbData[i - 1];
      const curr = mpbData[i];
      // X point is around k_distance = 0.5
      return prev.k_distance < 0.5 && curr.k_distance >= 0.5;
    });
    if (xPoint) {
      const xIdx = mpbData.findIndex(d => d.k_distance >= 0.5);
      if (xIdx > 0) {
        points.push({ distance: mpbData[xIdx].k_distance, label: 'X' });
      }
    }
    
    // Find M (k = 0.5, 0.5) - around k_distance = 1.0
    const mPoint = mpbData.find((_, i) => {
      if (i === 0) return false;
      const prev = mpbData[i - 1];
      const curr = mpbData[i];
      return prev.k_distance < 1.0 && curr.k_distance >= 1.0;
    });
    if (mPoint) {
      const mIdx = mpbData.findIndex(d => d.k_distance >= 1.0);
      if (mIdx > 0) {
        points.push({ distance: mpbData[mIdx].k_distance, label: 'M' });
      }
    }
    
    // Last point is Γ again
    const lastDist = mpbData[mpbData.length - 1]?.k_distance;
    if (lastDist) {
      points.push({ distance: lastDist, label: 'Γ' });
    }
    
    return points;
  }, [mpbData]);

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
    <div className="band-comparison-container" style={{ width: '100%', maxWidth: width }}>
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
          {`${polarization} Band Structure: MPB vs Blaze`}
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
          
          {/* Vertical lines at high-symmetry points */}
          {highSymmetryPoints.map((pt, i) => (
            <line
              key={`vline-${i}`}
              x1={xScale(pt.distance)}
              y1={0}
              x2={xScale(pt.distance)}
              y2={innerHeight}
              stroke={CHART_STYLES.gridColor}
              strokeOpacity={0.5}
              strokeDasharray="2,2"
            />
          ))}

          {/* MPB reference lines (one per band) */}
          {Array.from({ length: numBands }).map((_, bandIdx) => {
            const bandData = mpbData.map(kp => ({
              x: kp.k_distance,
              y: kp.frequencies[bandIdx] || 0,
            }));
            
            return (
              <LinePath
                key={`mpb-band-${bandIdx}`}
                data={bandData}
                x={d => xScale(d.x)}
                y={d => yScale(d.y)}
                stroke={MPB_COLOR}
                strokeWidth={1.5}
              />
            );
          })}

          {/* Blaze f64 markers (big circles) */}
          {f64Data.map((kp, kIdx) => (
            <Group key={`f64-kp-${kIdx}`}>
              {kp.frequencies.map((freq, bandIdx) => (
                <CircleMarker
                  key={`f64-${kIdx}-${bandIdx}`}
                  x={xScale(kp.k_distance)}
                  y={yScale(freq)}
                  size={F64_MARKER_SIZE}
                  color={F64_COLOR}
                  opacity={0.7}
                />
              ))}
            </Group>
          ))}

          {/* Blaze f32 markers (small triangles) */}
          {f32Data.map((kp, kIdx) => (
            <Group key={`f32-kp-${kIdx}`}>
              {kp.frequencies.map((freq, bandIdx) => (
                <TriangleMarker
                  key={`f32-${kIdx}-${bandIdx}`}
                  x={xScale(kp.k_distance)}
                  y={yScale(freq)}
                  size={F32_MARKER_SIZE}
                  color={F32_COLOR}
                  opacity={1}
                />
              ))}
            </Group>
          ))}

          {/* Y-Axis */}
          <AxisLeft
            scale={yScale}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickFormat={(v) => (v as number).toFixed(1)}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'end' as const,
              dx: -4,
              dy: 4,
            })}
            numTicks={6}
          />

          {/* X-Axis with high-symmetry labels */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            stroke={CHART_STYLES.axisColor}
            tickStroke={CHART_STYLES.axisColor}
            tickValues={highSymmetryPoints.map(p => p.distance)}
            tickFormat={(v) => {
              const pt = highSymmetryPoints.find(p => Math.abs(p.distance - (v as number)) < 0.01);
              return pt?.label || '';
            }}
            tickLabelProps={() => ({
              fill: CHART_STYLES.labelColor,
              fontSize: 11,
              fontFamily: CHART_STYLES.fontFamily,
              textAnchor: 'middle' as const,
              dy: 4,
            })}
          />

          {/* Y-axis label */}
          <Text
            x={-innerHeight / 2}
            y={-50}
            fontSize={12}
            fontFamily={CHART_STYLES.fontFamily}
            fill={CHART_STYLES.labelColor}
            textAnchor="middle"
            transform="rotate(-90)"
          >
            ωa / 2πc
          </Text>

          {/* X-axis label */}
          <Text
            x={innerWidth / 2}
            y={innerHeight + 45}
            fontSize={12}
            fontFamily={CHART_STYLES.fontFamily}
            fill={CHART_STYLES.labelColor}
            textAnchor="middle"
          >
            k-path
          </Text>
        </Group>

        {/* Legend */}
        <Group top={margin.top - 22} left={margin.left}>
          {/* MPB */}
          <Group left={0}>
            <line
              x1={0}
              y1={6}
              x2={20}
              y2={6}
              stroke={MPB_COLOR}
              strokeWidth={2}
            />
            <Text
              x={26}
              y={10}
              fontSize={11}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
            >
              MPB (reference)
            </Text>
          </Group>
          
          {/* Blaze f64 */}
          <Group left={130}>
            <CircleMarker x={6} y={6} size={4} color={F64_COLOR} />
            <Text
              x={18}
              y={10}
              fontSize={11}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
            >
              Blaze f64
            </Text>
          </Group>
          
          {/* Blaze f32 */}
          <Group left={220}>
            <TriangleMarker x={6} y={6} size={3} color={F32_COLOR} />
            <Text
              x={18}
              y={10}
              fontSize={11}
              fontFamily={CHART_STYLES.fontFamily}
              fill={CHART_STYLES.labelColor}
            >
              Blaze f32
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
        fontFamily: CHART_STYLES.fontFamily,
        fontStyle: 'italic',
        textAlign: 'left',
      }}>
      </p>
    </div>
  );
}
