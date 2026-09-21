'use client';

import CorePerformanceChart from './CorePerformanceChart';
import { useSingleCoreBenchmarks } from '@/lib/use-benchmarks';

export default function SingleCorePerformanceChart(props: { width?: number; height?: number }) {
  return (
    <CorePerformanceChart
      benchmark={useSingleCoreBenchmarks()}
      title="Single-Core Performance"
      workload="single-core"
      {...props}
    />
  );
}
