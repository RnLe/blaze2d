'use client';

import CorePerformanceChart from './CorePerformanceChart';
import { useMultiCoreBenchmarks } from '@/lib/use-benchmarks';

export default function MultiCorePerformanceChart(props: { width?: number; height?: number }) {
  const benchmark = useMultiCoreBenchmarks();
  const threads = benchmark.data.metadata?.num_threads ?? 16;
  return (
    <CorePerformanceChart
      benchmark={benchmark}
      title="Multi-Core Performance"
      workload={`${threads}-thread`}
      {...props}
    />
  );
}
