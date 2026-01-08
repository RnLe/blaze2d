'use client';

import { useState, useEffect } from 'react';
import { SingleCoreBenchmarkData, MultiCoreBenchmarkData, Series7ScalingData, Series5MemoryData, Series3ResolutionData, Series4IterationsData, Series1EpsilonData, Series2BandsData, FALLBACK_SINGLE_CORE_DATA, FALLBACK_MULTI_CORE_DATA, FALLBACK_SERIES7_DATA, FALLBACK_SERIES5_DATA, FALLBACK_SERIES3_DATA, FALLBACK_SERIES4_DATA, FALLBACK_SERIES1_DATA, FALLBACK_SERIES2_DATA } from './benchmark-data';
import { getAssetPath } from './paths';

// Cache for loaded benchmark data to avoid re-fetching
const dataCache = new Map<string, any>();

// Prefetch priority data immediately on module load
if (typeof window !== 'undefined') {
  // Prefetch single-core data (first chart on the page)
  const prefetchUrl = getAssetPath('/data/benchmarks/single-core.json');
  fetch(prefetchUrl, { priority: 'high' } as RequestInit)
    .then(res => res.ok ? res.json() : null)
    .then(data => {
      if (data) dataCache.set('single-core', data);
    })
    .catch(() => {});
}

/**
 * Hook to load single-core benchmark data
 * 
 * Loading strategy:
 * 1. Check cache first (may be prefetched)
 * 2. Try static data from public/data/ (hoisted by pre-commit hook)
 * 3. If static fails: Use fallback data embedded in code
 * 
 * The pre-commit hook (scripts/hoist-benchmark-data.js) copies benchmark
 * data to public/data/ so it's available for static export (GitHub Pages).
 */
export function useSingleCoreBenchmarks() {
  const [data, setData] = useState<SingleCoreBenchmarkData>(FALLBACK_SINGLE_CORE_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Check cache first (may be prefetched)
      const cached = dataCache.get('single-core');
      if (cached) {
        setData(cached);
        setSource('static');
        setLoading(false);
        return;
      }

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/single-core.json'));
        if (response.ok) {
          const staticData = await response.json();
          dataCache.set('single-core', staticData);
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SINGLE_CORE_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load multi-core benchmark data
 * 
 * Loading strategy:
 * 1. Try static data from public/data/ (hoisted by pre-commit hook)
 * 2. If static fails: Use fallback data embedded in code
 */
export function useMultiCoreBenchmarks() {
  const [data, setData] = useState<MultiCoreBenchmarkData>(FALLBACK_MULTI_CORE_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/multi-core.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_MULTI_CORE_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load Series 7 scaling benchmark data
 */
export function useSeries7Benchmarks() {
  const [data, setData] = useState<Series7ScalingData>(FALLBACK_SERIES7_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/series7-scaling.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SERIES7_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load Series 5 memory benchmark data
 */
export function useSeries5Benchmarks() {
  const [data, setData] = useState<Series5MemoryData>(FALLBACK_SERIES5_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/series5-memory.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SERIES5_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load Series 3 resolution benchmark data
 */
export function useSeries3Benchmarks() {
  const [data, setData] = useState<Series3ResolutionData>(FALLBACK_SERIES3_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/series3-resolution.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SERIES3_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load Series 4 iterations benchmark data
 */
export function useSeries4Benchmarks() {
  const [data, setData] = useState<Series4IterationsData>(FALLBACK_SERIES4_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/series4-iterations.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SERIES4_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load Series 1 epsilon benchmark data
 */
export function useSeries1Benchmarks() {
  const [data, setData] = useState<Series1EpsilonData>(FALLBACK_SERIES1_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/series1-epsilon.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SERIES1_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}

/**
 * Hook to load Series 2 bands benchmark data
 */
export function useSeries2Benchmarks() {
  const [data, setData] = useState<Series2BandsData>(FALLBACK_SERIES2_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'fallback'>('fallback');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch(getAssetPath('/data/benchmarks/series2-bands.json'));
        if (response.ok) {
          const staticData = await response.json();
          setData(staticData);
          setSource('static');
          setLoading(false);
          return;
        }
      } catch (e) {
        // Static data not available
      }

      // Fallback to embedded data
      setData(FALLBACK_SERIES2_DATA);
      setSource('fallback');
      setError('Using embedded fallback data');
      setLoading(false);
    }

    loadData();
  }, []);

  return { data, loading, error, source };
}
