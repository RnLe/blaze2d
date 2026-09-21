'use client';

import { useEffect, useState } from 'react';
import {
  type SingleCoreBenchmarkData,
  type MultiCoreBenchmarkData,
  type Series7ScalingData,
  type Series5MemoryData,
  type Series3ResolutionData,
  type Series4IterationsData,
  type Series1EpsilonData,
  type Series2BandsData,
  FALLBACK_SINGLE_CORE_DATA,
  FALLBACK_MULTI_CORE_DATA,
  FALLBACK_SERIES7_DATA,
  FALLBACK_SERIES5_DATA,
  FALLBACK_SERIES3_DATA,
  FALLBACK_SERIES4_DATA,
  FALLBACK_SERIES1_DATA,
  FALLBACK_SERIES2_DATA,
} from './benchmark-data';
import { getAssetPath } from './paths';

/**
 * Recorded benchmark datasets.
 *
 * Every chart on the technical report loads one of these. The numbers are
 * measurements, not defaults, so they live in `public/data/benchmarks/` as
 * exported JSON; `benchmark-data.ts` carries an embedded copy of each so a chart
 * still renders if a fetch fails after deployment.
 */

export type BenchmarkSource = 'static' | 'fallback';

export interface Benchmark<T> {
  data: T;
  loading: boolean;
  error: string | null;
  /** Whether the numbers came from the deployed JSON or the embedded copy. */
  source: BenchmarkSource;
}

/** One in-flight or settled request per dataset, shared by every chart. */
const cache = new Map<string, Promise<unknown>>();

function load(file: string): Promise<unknown> {
  let request = cache.get(file);
  if (!request) {
    request = fetch(getAssetPath(`/data/benchmarks/${file}`)).then(response => {
      if (!response.ok) throw new Error(`${file}: ${response.status}`);
      return response.json();
    });
    // A failed request must not be remembered, or a retry can never succeed.
    request.catch(() => cache.delete(file));
    cache.set(file, request);
  }
  return request;
}

function useBenchmark<T>(file: string, fallback: T): Benchmark<T> {
  const [state, setState] = useState<Benchmark<T>>({
    data: fallback,
    loading: true,
    error: null,
    source: 'fallback',
  });

  useEffect(() => {
    let current = true;
    load(file)
      .then(data => {
        if (current) setState({ data: data as T, loading: false, error: null, source: 'static' });
      })
      .catch(() => {
        if (current) {
          setState({ data: fallback, loading: false, error: 'Using embedded fallback data', source: 'fallback' });
        }
      });
    return () => {
      current = false;
    };
  }, [file, fallback]);

  return state;
}

// The first chart on the technical report is the single-core comparison, so its
// dataset is requested as soon as this module is evaluated.
if (typeof window !== 'undefined') void load('single-core.json').catch(() => {});

export const useSingleCoreBenchmarks = () =>
  useBenchmark<SingleCoreBenchmarkData>('single-core.json', FALLBACK_SINGLE_CORE_DATA);
export const useMultiCoreBenchmarks = () =>
  useBenchmark<MultiCoreBenchmarkData>('multi-core.json', FALLBACK_MULTI_CORE_DATA);
export const useSeries1Benchmarks = () =>
  useBenchmark<Series1EpsilonData>('series1-epsilon.json', FALLBACK_SERIES1_DATA);
export const useSeries2Benchmarks = () =>
  useBenchmark<Series2BandsData>('series2-bands.json', FALLBACK_SERIES2_DATA);
export const useSeries3Benchmarks = () =>
  useBenchmark<Series3ResolutionData>('series3-resolution.json', FALLBACK_SERIES3_DATA);
export const useSeries4Benchmarks = () =>
  useBenchmark<Series4IterationsData>('series4-iterations.json', FALLBACK_SERIES4_DATA);
export const useSeries5Benchmarks = () =>
  useBenchmark<Series5MemoryData>('series5-memory.json', FALLBACK_SERIES5_DATA);
export const useSeries7Benchmarks = () =>
  useBenchmark<Series7ScalingData>('series7-scaling.json', FALLBACK_SERIES7_DATA);
