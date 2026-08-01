'use client';

import { useMemo } from 'react';
import LineChart, { LineSeries } from './LineChart';
import { useSeries5Benchmarks } from '../../lib/use-benchmarks';

// Colors for TM and TE polarizations
const COLORS = {
  TM: '#eaf1fe',   // Light blue-white
  TE: '#5477c4',   // Blue-gray
};

interface MemoryRatioChartProps {
  width?: number;
  height?: number;
}

export default function MemoryRatioChart({
  width = 650,
  height = 380,
}: MemoryRatioChartProps) {
  const { data: benchmarkData, loading } = useSeries5Benchmarks();

  // Transform data for line charts
  const { resolutionSeries, bandsSeries } = useMemo(() => {
    const resData = benchmarkData.resolution;
    const bandsData = benchmarkData.num_bands;

    // Helper to compute ratio and propagated error
    const computeRatio = (
      mpbVal: number, mpbStd: number, 
      blazeVal: number, blazeStd: number
    ): { ratio: number; std: number } => {
      if (blazeVal === 0) return { ratio: 0, std: 0 };
      
      const ratio = mpbVal / blazeVal;
      // Error propagation for R = A/B: std_R = |R| * sqrt((std_A/A)^2 + (std_B/B)^2)
      // Assuming independent errors
      const relErrA = mpbStd / mpbVal;
      const relErrB = blazeStd / blazeVal;
      const std = ratio * Math.sqrt(relErrA * relErrA + relErrB * relErrB);
      
      return { ratio, std };
    };

    // Resolution sweep - compute ratio (MPB / Blaze)
    const resTM: { x: number; y: number; std?: number }[] = [];
    const resTE: { x: number; y: number; std?: number }[] = [];

    for (const val of resData.values) {
      const mpbTM = resData.mpb.TM.find(d => d.value === val);
      const blazeTM = resData.blaze.TM.find(d => d.value === val);
      const mpbTE = resData.mpb.TE.find(d => d.value === val);
      const blazeTE = resData.blaze.TE.find(d => d.value === val);

      if (mpbTM && blazeTM) {
        const { ratio, std } = computeRatio(
          mpbTM.memory_mb, mpbTM.memory_mb_std || 0,
          blazeTM.memory_mb, blazeTM.memory_mb_std || 0
        );
        resTM.push({ x: val, y: ratio, std: std > 0 ? std : undefined });
      }
      if (mpbTE && blazeTE) {
        const { ratio, std } = computeRatio(
          mpbTE.memory_mb, mpbTE.memory_mb_std || 0,
          blazeTE.memory_mb, blazeTE.memory_mb_std || 0
        );
        resTE.push({ x: val, y: ratio, std: std > 0 ? std : undefined });
      }
    }

    // Bands sweep - compute ratio (MPB / Blaze)
    const bandsTM: { x: number; y: number; std?: number }[] = [];
    const bandsTE: { x: number; y: number; std?: number }[] = [];

    for (const val of bandsData.values) {
      const mpbTM = bandsData.mpb.TM.find(d => d.value === val);
      const blazeTM = bandsData.blaze.TM.find(d => d.value === val);
      const mpbTE = bandsData.mpb.TE.find(d => d.value === val);
      const blazeTE = bandsData.blaze.TE.find(d => d.value === val);

      if (mpbTM && blazeTM) {
        const { ratio, std } = computeRatio(
          mpbTM.memory_mb, mpbTM.memory_mb_std || 0,
          blazeTM.memory_mb, blazeTM.memory_mb_std || 0
        );
        bandsTM.push({ x: val, y: ratio, std: std > 0 ? std : undefined });
      }
      if (mpbTE && blazeTE) {
        const { ratio, std } = computeRatio(
          mpbTE.memory_mb, mpbTE.memory_mb_std || 0,
          blazeTE.memory_mb, blazeTE.memory_mb_std || 0
        );
        bandsTE.push({ x: val, y: ratio, std: std > 0 ? std : undefined });
      }
    }

    const resolutionSeries: LineSeries[] = [
      {
        id: 'res-tm',
        label: 'TM',
        color: COLORS.TM,
        marker: 'circle',
        data: resTM,
      },
      {
        id: 'res-te',
        label: 'TE',
        color: COLORS.TE,
        marker: 'square',
        data: resTE,
      },
    ];

    const bandsSeries: LineSeries[] = [
      {
        id: 'bands-tm',
        label: 'TM',
        color: COLORS.TM,
        marker: 'circle',
        data: bandsTM,
      },
      {
        id: 'bands-te',
        label: 'TE',
        color: COLORS.TE,
        marker: 'square',
        data: bandsTE,
      },
    ];

    return { resolutionSeries, bandsSeries };
  }, [benchmarkData]);

  if (loading) {
    return (
      <div style={{ width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>
        Loading benchmark data...
      </div>
    );
  }

  const chartWidth = (width - 20) / 2;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
        <LineChart
          series={resolutionSeries}
          width={chartWidth}
          height={height}
          title="Memory Efficiency Ratio vs Resolution"
          xLabel="Resolution"
          yLabel="Ratio (MPB / Blaze)"
          xTickFormat={(v) => `${v}`}
          yTickFormat={(v) => `${v.toFixed(0)}×`}
          showErrorBars={true}
          yDomain={[0, 35]}
          margin={{ top: 50, right: 20, bottom: 60, left: 70 }}
        />
        <LineChart
          series={bandsSeries}
          width={chartWidth}
          height={height}
          title="Memory Efficiency Ratio vs Bands"
          xLabel="Number of Bands"
          yLabel="Ratio (MPB / Blaze)"
          xTickFormat={(v) => `${v}`}
          yTickFormat={(v) => `${v.toFixed(0)}×`}
          showErrorBars={true}
          yDomain={[0, 25]}
          margin={{ top: 50, right: 20, bottom: 60, left: 70 }}
        />
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
      </p>
    </div>
  );
}
