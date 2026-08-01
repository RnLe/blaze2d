// Synchronous replacement for web/lib/use-benchmarks.ts.
//
// The real hooks start from the embedded FALLBACK_* constants and swap in the
// hoisted JSON from web/public/data/ once a useEffect-driven fetch resolves.
// Under react-dom/server that effect never fires, so rendering the untouched
// hooks would silently export the fallback numbers instead of the published
// ones. Each hook here reads the same file the browser would fetch and reports
// the settled state directly, which is why the exported figures carry the same
// values as the live site.
//
// Signatures mirror the originals exactly ({ data, loading, error, source }) so
// the chart components are used unmodified.

import {
  SingleCoreBenchmarkData,
  MultiCoreBenchmarkData,
  Series7ScalingData,
  Series5MemoryData,
  Series3ResolutionData,
  Series4IterationsData,
  Series1EpsilonData,
  Series2BandsData,
} from './benchmark-data';
import { loadJson } from './data-source';

interface HookResult<T> {
  data: T;
  loading: boolean;
  error: string | null;
  source: 'static' | 'fallback';
}

function settled<T>(assetPath: string): HookResult<T> {
  return {
    data: loadJson<T>(assetPath),
    loading: false,
    error: null,
    source: 'static',
  };
}

export function useSingleCoreBenchmarks() {
  return settled<SingleCoreBenchmarkData>('/data/benchmarks/single-core.json');
}

export function useMultiCoreBenchmarks() {
  return settled<MultiCoreBenchmarkData>('/data/benchmarks/multi-core.json');
}

export function useSeries7Benchmarks() {
  return settled<Series7ScalingData>('/data/benchmarks/series7-scaling.json');
}

export function useSeries5Benchmarks() {
  return settled<Series5MemoryData>('/data/benchmarks/series5-memory.json');
}

export function useSeries3Benchmarks() {
  return settled<Series3ResolutionData>('/data/benchmarks/series3-resolution.json');
}

export function useSeries4Benchmarks() {
  return settled<Series4IterationsData>('/data/benchmarks/series4-iterations.json');
}

export function useSeries1Benchmarks() {
  return settled<Series1EpsilonData>('/data/benchmarks/series1-epsilon.json');
}

export function useSeries2Benchmarks() {
  return settled<Series2BandsData>('/data/benchmarks/series2-bands.json');
}
