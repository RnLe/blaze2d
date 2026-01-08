/**
 * Benchmark Data Loading Utilities
 * 
 * This module provides hybrid data loading:
 * - In development: Loads fresh data from the benchmarks folder via API route
 * - In production (GitHub Pages): Loads static data from public/data/
 * 
 * The precommit hook copies benchmark data to public/data/ before each commit,
 * ensuring the deployed site always has the latest committed data.
 */

// Types for benchmark data
export interface SingleCoreBenchmarkData {
  mpb: {
    [config: string]: {
      mean_ms: number;
      std_ms: number;
      description: string;
    };
  };
  blaze: {
    [config: string]: {
      mean_ms: number;
      std_ms: number;
      description: string;
    };
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
  };
}

// Raw benchmark file types
interface BlazeBenchmarkFile {
  benchmark: string;
  timestamp: string;
  solver: string;
  configs: {
    [key: string]: {
      mean_ms: number;
      std_ms: number;
      description: string;
      polarization: string;
    };
  };
}

interface MpbBenchmarkFile {
  benchmark: string;
  timestamp: string;
  solver: string;
  configs: {
    [key: string]: {
      description: string;
      polarizations: {
        [pol: string]: {
          mean_ms: number;
          std_ms: number;
        };
      };
    };
  };
}

/**
 * Check if we're in development mode
 */
export function isDevelopment(): boolean {
  return process.env.NODE_ENV === 'development';
}

/**
 * Transform raw benchmark files into SingleCoreBenchmarkData format
 */
export function transformBenchmarkData(
  blazeData: BlazeBenchmarkFile,
  mpbData: MpbBenchmarkFile
): SingleCoreBenchmarkData {
  const result: SingleCoreBenchmarkData = {
    mpb: {},
    blaze: {},
    metadata: {
      timestamp: blazeData.timestamp,
      source: isDevelopment() ? 'live' : 'static',
    },
  };

  // Transform Blaze data
  for (const [configKey, config] of Object.entries(blazeData.configs)) {
    result.blaze[configKey] = {
      mean_ms: config.mean_ms,
      std_ms: config.std_ms,
      description: config.description,
    };
  }

  // Transform MPB data
  for (const [configKey, config] of Object.entries(mpbData.configs)) {
    for (const [pol, polData] of Object.entries(config.polarizations)) {
      const fullKey = `${configKey}_${pol}`;
      result.mpb[fullKey] = {
        mean_ms: polData.mean_ms,
        std_ms: polData.std_ms,
        description: config.description,
      };
    }
  }

  return result;
}

// Default fallback data (used if both sources fail)
export const FALLBACK_SINGLE_CORE_DATA: SingleCoreBenchmarkData = {
  mpb: {
    config_a_tm: { mean_ms: 1212.41, std_ms: 19.69, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_a_te: { mean_ms: 1978.09, std_ms: 16.63, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_b_tm: { mean_ms: 1052.90, std_ms: 14.91, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
    config_b_te: { mean_ms: 2001.98, std_ms: 15.23, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
  },
  blaze: {
    config_a_tm: { mean_ms: 421.00, std_ms: 1.97, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_a_te: { mean_ms: 579.70, std_ms: 0.24, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_b_tm: { mean_ms: 405.18, std_ms: 0.84, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
    config_b_te: { mean_ms: 558.70, std_ms: 0.24, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
  },
  metadata: {
    timestamp: '2026-01-07T19:59:30.379942',
    source: 'static',
  },
};

// Multi-core benchmark data types
export interface MultiCoreBenchmarkData {
  mpb: {
    [config: string]: {
      mean_ms: number;
      std_ms: number;
      description: string;
    };
  };
  blaze: {
    [config: string]: {
      mean_ms: number;
      std_ms: number;
      description: string;
    };
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    num_threads?: number;
  };
}

// Multi-core fallback data (MPB Native only, no Process)
export const FALLBACK_MULTI_CORE_DATA: MultiCoreBenchmarkData = {
  mpb: {
    config_a_tm: { mean_ms: 1501.48, std_ms: 66.26, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_a_te: { mean_ms: 2421.32, std_ms: 39.75, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_b_tm: { mean_ms: 1304.07, std_ms: 89.30, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
    config_b_te: { mean_ms: 2471.04, std_ms: 53.62, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
  },
  blaze: {
    config_a_tm: { mean_ms: 157.47, std_ms: 0.18, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_a_te: { mean_ms: 169.23, std_ms: 0.27, description: 'Square lattice, air bg, ε=8.9 rods, r=0.2a' },
    config_b_tm: { mean_ms: 149.57, std_ms: 1.84, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
    config_b_te: { mean_ms: 169.60, std_ms: 2.17, description: 'Hex lattice, ε=13 bg, air rods, r=0.48a' },
  },
  metadata: {
    timestamp: '2026-01-07T22:10:21.238905',
    source: 'static',
    num_threads: 16,
  },
};

// Series 7 Scaling benchmark data types
export interface ScalingDataPoint {
  threads: number;
  mean_throughput: number;
  std_throughput: number;
}

export interface Series7ScalingData {
  low: {
    resolution: number;
    blaze: ScalingDataPoint[];
    mpb_omp: ScalingDataPoint[];
    mpb_multiproc: ScalingDataPoint[];
  };
  high: {
    resolution: number;
    blaze: ScalingDataPoint[];
    mpb_omp: ScalingDataPoint[];
    mpb_multiproc: ScalingDataPoint[];
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
  };
}

// Series 7 fallback data (extracted from series7_scaling_results.json)
export const FALLBACK_SERIES7_DATA: Series7ScalingData = {
  low: {
    resolution: 16,
    blaze: [
      { threads: 1, mean_throughput: 28.25, std_throughput: 1.77 },
      { threads: 2, mean_throughput: 58.74, std_throughput: 1.67 },
      { threads: 4, mean_throughput: 111.62, std_throughput: 0.14 },
      { threads: 8, mean_throughput: 172.95, std_throughput: 7.25 },
      { threads: 12, mean_throughput: 201.81, std_throughput: 9.09 },
      { threads: 16, mean_throughput: 220.40, std_throughput: 14.94 },
    ],
    mpb_omp: [
      { threads: 1, mean_throughput: 14.37, std_throughput: 0.51 },
      { threads: 2, mean_throughput: 14.44, std_throughput: 0.18 },
      { threads: 4, mean_throughput: 14.17, std_throughput: 0.12 },
      { threads: 8, mean_throughput: 14.33, std_throughput: 0.31 },
      { threads: 12, mean_throughput: 13.24, std_throughput: 0.28 },
      { threads: 16, mean_throughput: 14.06, std_throughput: 0.06 },
    ],
    mpb_multiproc: [
      { threads: 1, mean_throughput: 13.94, std_throughput: 0.35 },
      { threads: 2, mean_throughput: 27.09, std_throughput: 0.04 },
      { threads: 4, mean_throughput: 49.89, std_throughput: 1.16 },
      { threads: 8, mean_throughput: 76.09, std_throughput: 4.21 },
      { threads: 12, mean_throughput: 96.14, std_throughput: 1.67 },
      { threads: 16, mean_throughput: 106.64, std_throughput: 5.31 },
    ],
  },
  high: {
    resolution: 128,
    blaze: [
      { threads: 1, mean_throughput: 0.448, std_throughput: 0.0004 },
      { threads: 2, mean_throughput: 0.954, std_throughput: 0.006 },
      { threads: 4, mean_throughput: 1.423, std_throughput: 0.009 },
      { threads: 8, mean_throughput: 1.513, std_throughput: 0.019 },
      { threads: 12, mean_throughput: 1.468, std_throughput: 0.006 },
      { threads: 16, mean_throughput: 1.447, std_throughput: 0.006 },
    ],
    mpb_omp: [
      { threads: 1, mean_throughput: 0.192, std_throughput: 0.001 },
      { threads: 2, mean_throughput: 0.204, std_throughput: 0.00003 },
      { threads: 4, mean_throughput: 0.193, std_throughput: 0.002 },
      { threads: 8, mean_throughput: 0.174, std_throughput: 0.001 },
      { threads: 12, mean_throughput: 0.168, std_throughput: 0.004 },
      { threads: 16, mean_throughput: 0.168, std_throughput: 0.001 },
    ],
    mpb_multiproc: [
      { threads: 1, mean_throughput: 0.195, std_throughput: 0.0005 },
      { threads: 2, mean_throughput: 0.337, std_throughput: 0.0009 },
      { threads: 4, mean_throughput: 0.489, std_throughput: 0.004 },
      { threads: 8, mean_throughput: 0.442, std_throughput: 0.009 },
      { threads: 12, mean_throughput: 0.381, std_throughput: 0.002 },
      { threads: 16, mean_throughput: 0.340, std_throughput: 0.0006 },
    ],
  },
  metadata: {
    timestamp: '2026-01-07T21:16:46.134409',
    source: 'static',
  },
};
