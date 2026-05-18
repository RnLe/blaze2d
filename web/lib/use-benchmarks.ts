'use client';

import { useState, useEffect } from 'react';
import { SingleCoreBenchmarkData, MultiCoreBenchmarkData, Series7ScalingData, FALLBACK_SINGLE_CORE_DATA, FALLBACK_MULTI_CORE_DATA, FALLBACK_SERIES7_DATA } from './benchmark-data';

/**
 * Hook to load single-core benchmark data
 * 
 * Loading strategy:
 * 1. Try static data from public/data/ (hoisted by pre-commit hook)
 * 2. If static fails: Use fallback data embedded in code
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

      // Try static data from public/data/ (hoisted by pre-commit hook)
      try {
        const response = await fetch('/data/benchmarks/single-core.json');
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
        const response = await fetch('/data/benchmarks/multi-core.json');
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
        const response = await fetch('/data/benchmarks/series7-scaling.json');
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
