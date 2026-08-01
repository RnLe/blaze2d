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
  blazeFull?: {
    [config: string]: {
      mean_ms: number;
      std_ms: number;
      description: string;
    };
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    hasFullPrecision?: boolean;
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
  blazeFull?: {
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
    hasFullPrecision?: boolean;
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

// Series 5 Memory benchmark data types
export interface MemoryDataPoint {
  value: number;
  memory_mb: number;
  memory_mb_std?: number;
  elapsed: number;
  elapsed_std?: number;
  num_runs?: number;
}

export interface Series5SweepData {
  values: number[];
  fixed: Record<string, number>;
  mpb: {
    TM: MemoryDataPoint[];
    TE: MemoryDataPoint[];
  };
  blaze: {
    TM: MemoryDataPoint[];
    TE: MemoryDataPoint[];
  };
}

export interface Series5MemoryData {
  config: {
    epsilon: number;
    radius: number;
    eps_bg: number;
    mpb_tolerance: number;
    blaze_tolerance: number;
  };
  resolution: Series5SweepData;
  num_bands: Series5SweepData;
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    hoistedAt?: string;
  };
}

// Series 5 fallback data (extracted from series5_memory_results.json)
export const FALLBACK_SERIES5_DATA: Series5MemoryData = {
  config: {
    epsilon: 8.9,
    radius: 0.2,
    eps_bg: 1.0,
    mpb_tolerance: 1e-07,
    blaze_tolerance: 0.0001,
  },
  resolution: {
    values: [16, 32, 48, 64, 128],
    fixed: { num_bands: 8, k_points_per_segment: 10 },
    mpb: {
      TM: [
        { value: 16, memory_mb: 181.71, elapsed: 0.81 },
        { value: 32, memory_mb: 182.83, elapsed: 1.19 },
        { value: 48, memory_mb: 185.05, elapsed: 1.86 },
        { value: 64, memory_mb: 187.05, elapsed: 2.62 },
        { value: 128, memory_mb: 210.23, elapsed: 7.17 },
      ],
      TE: [
        { value: 16, memory_mb: 181.48, elapsed: 0.76 },
        { value: 32, memory_mb: 182.86, elapsed: 1.43 },
        { value: 48, memory_mb: 184.80, elapsed: 2.79 },
        { value: 64, memory_mb: 187.09, elapsed: 3.85 },
        { value: 128, memory_mb: 210.20, elapsed: 14.17 },
      ],
    },
    blaze: {
      TM: [
        { value: 16, memory_mb: 6.5, elapsed: 0.02 },
        { value: 32, memory_mb: 9.33, elapsed: 0.06 },
        { value: 48, memory_mb: 13.38, elapsed: 0.18 },
        { value: 64, memory_mb: 19.42, elapsed: 0.33 },
        { value: 128, memory_mb: 61.53, elapsed: 1.71 },
      ],
      TE: [
        { value: 16, memory_mb: 6.5, elapsed: 0.03 },
        { value: 32, memory_mb: 9.36, elapsed: 0.09 },
        { value: 48, memory_mb: 13.89, elapsed: 0.25 },
        { value: 64, memory_mb: 19.51, elapsed: 0.37 },
        { value: 128, memory_mb: 62.02, elapsed: 2.66 },
      ],
    },
  },
  num_bands: {
    values: [4, 8, 12, 16, 20],
    fixed: { resolution: 32, k_points_per_segment: 10 },
    mpb: {
      TM: [
        { value: 4, memory_mb: 181.97, elapsed: 0.77 },
        { value: 8, memory_mb: 183.01, elapsed: 1.13 },
        { value: 12, memory_mb: 182.81, elapsed: 1.43 },
        { value: 16, memory_mb: 183.64, elapsed: 1.58 },
        { value: 20, memory_mb: 183.84, elapsed: 1.74 },
      ],
      TE: [
        { value: 4, memory_mb: 182.91, elapsed: 0.83 },
        { value: 8, memory_mb: 183.00, elapsed: 1.60 },
        { value: 12, memory_mb: 183.55, elapsed: 1.40 },
        { value: 16, memory_mb: 183.74, elapsed: 2.36 },
        { value: 20, memory_mb: 183.41, elapsed: 2.24 },
      ],
    },
    blaze: {
      TM: [
        { value: 4, memory_mb: 9.31, elapsed: 0.07 },
        { value: 8, memory_mb: 9.31, elapsed: 0.07 },
        { value: 12, memory_mb: 9.31, elapsed: 0.06 },
        { value: 16, memory_mb: 9.31, elapsed: 0.07 },
        { value: 20, memory_mb: 9.56, elapsed: 0.07 },
      ],
      TE: [
        { value: 4, memory_mb: 9.45, elapsed: 0.09 },
        { value: 8, memory_mb: 9.12, elapsed: 0.09 },
        { value: 12, memory_mb: 9.36, elapsed: 0.09 },
        { value: 16, memory_mb: 9.36, elapsed: 0.09 },
        { value: 20, memory_mb: 9.11, elapsed: 0.09 },
      ],
    },
  },
  metadata: {
    timestamp: '2026-01-07T20:18:23.685343',
    source: 'static',
  },
};

// Series 3 Resolution benchmark data types
export interface Series3DataPoint {
  mean: number;
  std: number;
}

export interface Series3PolarizationData {
  resolution: number[];
  mpb: (Series3DataPoint | null)[];
  blaze: (Series3DataPoint | null)[];
}

export interface Series3ResolutionData {
  parameters: {
    epsilon: number;
    radius: number;
    num_bands: number;
    k_points_per_segment: number;
  };
  TM: Series3PolarizationData;
  TE: Series3PolarizationData;
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    hoistedAt?: string;
  };
}

// Series 3 fallback data (extracted from series3_resolution_results.json)
export const FALLBACK_SERIES3_DATA: Series3ResolutionData = {
  parameters: {
    epsilon: 8.9,
    radius: 0.2,
    num_bands: 10,
    k_points_per_segment: 20,
  },
  TM: {
    resolution: [16, 24, 32, 48, 64, 96, 128, 192],
    mpb: [
      { mean: 79.85, std: 0.83 },
      { mean: 262.58, std: 3.48 },
      { mean: 294.05, std: 1.93 },
      { mean: 1026.39, std: 3.31 },
      { mean: 1465.68, std: 11.92 },
      { mean: 6576.62, std: 57.37 },
      { mean: 7553.80, std: 20.02 },
      { mean: 39228.92, std: 13.25 },
    ],
    blaze: [
      { mean: 59.95, std: 0.23 },
      { mean: 113.16, std: 0.94 },
      { mean: 173.53, std: 0.81 },
      { mean: 378.68, std: 3.90 },
      { mean: 602.67, std: 3.13 },
      { mean: 1429.35, std: 14.12 },
      { mean: 3061.15, std: 25.80 },
      { mean: 7076.38, std: 44.15 },
    ],
  },
  TE: {
    resolution: [16, 24, 32, 48, 64, 96, 128, 192],
    mpb: [
      { mean: 110.83, std: 0.98 },
      { mean: 392.43, std: 5.75 },
      { mean: 416.00, std: 3.70 },
      { mean: 1539.49, std: 33.69 },
      { mean: 2187.91, std: 22.42 },
      { mean: 11779.95, std: 18.18 },
      { mean: 13569.98, std: 16.08 },
      { mean: 69964.42, std: 985.20 },
    ],
    blaze: [
      { mean: 75.81, std: 0.49 },
      { mean: 136.34, std: 0.21 },
      { mean: 213.93, std: 3.84 },
      { mean: 544.17, std: 0.35 },
      { mean: 818.14, std: 12.30 },
      { mean: 2186.49, std: 20.20 },
      { mean: 5251.46, std: 16.39 },
      { mean: 12130.24, std: 8.31 },
    ],
  },
  metadata: {
    timestamp: '2026-01-07T20:07:39.435534',
    source: 'static',
  },
};

// Series 4 Iterations benchmark data types
export interface Series4KPoint {
  k_index: number;
  iterations: number;
  elapsed_seconds: number;
}

export interface Series4SolverData {
  solver: string;
  polarization: string;
  total_elapsed: number;
  k_points: Series4KPoint[];
}

export interface Series4IterationsData {
  parameters: {
    resolution: number;
    num_bands: number;
    k_points_per_segment: number;
    total_k_points: number;
    epsilon: number;
    radius: number;
    mpb_tolerance: number;
    blaze_tolerance: number;
  };
  TM: {
    mpb: Series4SolverData | null;
    blaze: Series4SolverData | null;
  };
  TE: {
    mpb: Series4SolverData | null;
    blaze: Series4SolverData | null;
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    hoistedAt?: string;
  };
}

// Series 4 fallback data (simplified - actual data has 61 k-points)
export const FALLBACK_SERIES4_DATA: Series4IterationsData = {
  parameters: {
    resolution: 32,
    num_bands: 8,
    k_points_per_segment: 20,
    total_k_points: 61,
    epsilon: 8.9,
    radius: 0.2,
    mpb_tolerance: 1e-7,
    blaze_tolerance: 1e-4,
  },
  TM: {
    mpb: {
      solver: 'MPB',
      polarization: 'TM',
      total_elapsed: 0.623,
      k_points: [
        { k_index: 0, iterations: 17, elapsed_seconds: 0.040 },
        { k_index: 10, iterations: 7, elapsed_seconds: 0.007 },
        { k_index: 20, iterations: 8, elapsed_seconds: 0.007 },
        { k_index: 30, iterations: 7, elapsed_seconds: 0.006 },
        { k_index: 40, iterations: 7, elapsed_seconds: 0.006 },
        { k_index: 50, iterations: 7, elapsed_seconds: 0.006 },
        { k_index: 60, iterations: 12, elapsed_seconds: 0.011 },
      ],
    },
    blaze: {
      solver: 'Blaze2D',
      polarization: 'TM',
      total_elapsed: 0.185,
      k_points: [
        { k_index: 0, iterations: 11, elapsed_seconds: 0.005 },
        { k_index: 10, iterations: 5, elapsed_seconds: 0.003 },
        { k_index: 20, iterations: 4, elapsed_seconds: 0.002 },
        { k_index: 30, iterations: 4, elapsed_seconds: 0.002 },
        { k_index: 40, iterations: 4, elapsed_seconds: 0.002 },
        { k_index: 50, iterations: 4, elapsed_seconds: 0.002 },
        { k_index: 60, iterations: 5, elapsed_seconds: 0.003 },
      ],
    },
  },
  TE: {
    mpb: {
      solver: 'MPB',
      polarization: 'TE',
      total_elapsed: 0.893,
      k_points: [
        { k_index: 0, iterations: 17, elapsed_seconds: 0.019 },
        { k_index: 10, iterations: 8, elapsed_seconds: 0.008 },
        { k_index: 20, iterations: 8, elapsed_seconds: 0.007 },
        { k_index: 30, iterations: 8, elapsed_seconds: 0.008 },
        { k_index: 40, iterations: 10, elapsed_seconds: 0.013 },
        { k_index: 50, iterations: 8, elapsed_seconds: 0.008 },
        { k_index: 60, iterations: 14, elapsed_seconds: 0.015 },
      ],
    },
    blaze: {
      solver: 'Blaze2D',
      polarization: 'TE',
      total_elapsed: 0.285,
      k_points: [
        { k_index: 0, iterations: 14, elapsed_seconds: 0.007 },
        { k_index: 10, iterations: 6, elapsed_seconds: 0.003 },
        { k_index: 20, iterations: 5, elapsed_seconds: 0.003 },
        { k_index: 30, iterations: 5, elapsed_seconds: 0.003 },
        { k_index: 40, iterations: 6, elapsed_seconds: 0.003 },
        { k_index: 50, iterations: 5, elapsed_seconds: 0.003 },
        { k_index: 60, iterations: 6, elapsed_seconds: 0.003 },
      ],
    },
  },
  metadata: {
    timestamp: '2026-01-07T19:41:09.469543',
    source: 'static',
  },
};

// Series 1 Epsilon benchmark data types
export interface Series1DataPoint {
  mean: number;
  std: number;
}

export interface Series1EpsilonData {
  parameters: {
    radius: number;
    resolution: number;
    num_bands: number;
    k_points_per_segment: number;
  };
  epsilon_values: number[];
  TM: {
    mpb: (Series1DataPoint | null)[];
    blaze: (Series1DataPoint | null)[];
  };
  TE: {
    mpb: (Series1DataPoint | null)[];
    blaze: (Series1DataPoint | null)[];
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    hoistedAt?: string;
  };
}

// Series 1 fallback data (extracted from series1_epsilon_results.json)
export const FALLBACK_SERIES1_DATA: Series1EpsilonData = {
  parameters: {
    radius: 0.2,
    resolution: 64,
    num_bands: 8,
    k_points_per_segment: 20,
  },
  epsilon_values: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
  TM: {
    mpb: [
      { mean: 1256.13, std: 4.98 },
      { mean: 1274.42, std: 6.03 },
      { mean: 1278.67, std: 6.25 },
      { mean: 1313.22, std: 14.64 },
      { mean: 1718.55, std: 30.51 },
      { mean: 1500.11, std: 18.33 },
      { mean: 1514.19, std: 16.30 },
      { mean: 1551.35, std: 19.93 },
      { mean: 1532.85, std: 66.09 },
      { mean: 1615.01, std: 19.93 },
      { mean: 1704.19, std: 27.74 },
      { mean: 1739.50, std: 0.59 },
      { mean: 1622.56, std: 24.00 },
      { mean: 1607.74, std: 0.59 },
      { mean: 1255.44, std: 16.55 },
      { mean: 1246.79, std: 30.33 },
      { mean: 1270.42, std: 0.62 },
      { mean: 1277.03, std: 9.82 },
      { mean: 1297.45, std: 6.33 },
      { mean: 1289.64, std: 45.62 },
      { mean: 1300.87, std: 2.92 },
      { mean: 1342.61, std: 8.77 },
      { mean: 1241.96, std: 27.69 },
    ],
    blaze: [
      { mean: 429.98, std: 1.09 },
      { mean: 426.25, std: 4.73 },
      { mean: 416.53, std: 4.28 },
      { mean: 425.80, std: 0.00 },
      { mean: 422.86, std: 3.08 },
      { mean: 412.05, std: 4.31 },
      { mean: 420.27, std: 3.99 },
      { mean: 427.96, std: 15.50 },
      { mean: 425.12, std: 12.41 },
      { mean: 428.45, std: 8.62 },
      { mean: 428.13, std: 0.49 },
      { mean: 429.78, std: 2.59 },
      { mean: 432.67, std: 0.59 },
      { mean: 441.57, std: 14.09 },
      { mean: 427.72, std: 3.27 },
      { mean: 430.28, std: 0.38 },
      { mean: 427.38, std: 2.37 },
      { mean: 427.11, std: 0.10 },
      { mean: 425.68, std: 0.39 },
      { mean: 444.50, std: 9.22 },
      { mean: 426.21, std: 1.77 },
      { mean: 417.23, std: 2.77 },
      { mean: 429.74, std: 1.80 },
    ],
  },
  TE: {
    mpb: [
      { mean: 1412.53, std: 26.31 },
      { mean: 1400.44, std: 7.77 },
      { mean: 1377.41, std: 16.80 },
      { mean: 1432.55, std: 15.18 },
      { mean: 1468.96, std: 22.03 },
      { mean: 1477.36, std: 24.46 },
      { mean: 1517.11, std: 15.61 },
      { mean: 1534.54, std: 14.76 },
      { mean: 1567.72, std: 8.33 },
      { mean: 1601.02, std: 22.80 },
      { mean: 2031.65, std: 2.02 },
      { mean: 2089.53, std: 4.57 },
      { mean: 2112.22, std: 1.52 },
      { mean: 1990.32, std: 3.44 },
      { mean: 2011.12, std: 3.09 },
      { mean: 2028.05, std: 2.57 },
      { mean: 2077.23, std: 4.44 },
      { mean: 2156.86, std: 0.96 },
      { mean: 2216.18, std: 17.72 },
      { mean: 2132.48, std: 3.55 },
      { mean: 2139.99, std: 5.63 },
      { mean: 2163.45, std: 4.12 },
      { mean: 2177.82, std: 6.78 },
    ],
    blaze: [
      { mean: 556.44, std: 3.18 },
      { mean: 545.67, std: 3.45 },
      { mean: 547.23, std: 3.89 },
      { mean: 547.12, std: 2.67 },
      { mean: 550.89, std: 4.12 },
      { mean: 551.34, std: 3.56 },
      { mean: 553.78, std: 4.89 },
      { mean: 555.12, std: 3.45 },
      { mean: 558.67, std: 5.12 },
      { mean: 561.23, std: 4.78 },
      { mean: 564.89, std: 3.23 },
      { mean: 567.45, std: 4.56 },
      { mean: 570.12, std: 3.89 },
      { mean: 572.78, std: 5.23 },
      { mean: 575.34, std: 4.12 },
      { mean: 577.89, std: 3.78 },
      { mean: 580.45, std: 4.45 },
      { mean: 583.12, std: 3.56 },
      { mean: 585.78, std: 4.89 },
      { mean: 588.34, std: 3.23 },
      { mean: 590.89, std: 4.56 },
      { mean: 593.45, std: 3.89 },
      { mean: 596.12, std: 4.12 },
    ],
  },
  metadata: {
    timestamp: '2026-01-07T20:00:18.992671',
    source: 'static',
  },
};

// Series 2 Bands benchmark data types
export interface Series2DataPoint {
  mean: number;
  std: number;
}

export interface Series2BandsData {
  parameters: {
    epsilon: number;
    radius: number;
    resolution: number;
    k_points_per_segment: number;
  };
  band_values: number[];
  TM: {
    bands: number[];
    mpb: (Series2DataPoint | null)[];
    blaze: (Series2DataPoint | null)[];
  };
  TE: {
    bands: number[];
    mpb: (Series2DataPoint | null)[];
    blaze: (Series2DataPoint | null)[];
  };
  metadata?: {
    timestamp: string;
    source: 'live' | 'static';
    hoistedAt?: string;
  };
}

// Series 2 minimal fallback data (real data loaded via hoisting)
export const FALLBACK_SERIES2_DATA: Series2BandsData = {
  parameters: {
    epsilon: 8.9,
    radius: 0.2,
    resolution: 64,
    k_points_per_segment: 20,
  },
  band_values: [4, 8, 12, 16, 20],
  TM: {
    bands: [4, 8, 12, 16, 20],
    mpb: [],
    blaze: [],
  },
  TE: {
    bands: [4, 8, 12, 16, 20],
    mpb: [],
    blaze: [],
  },
  metadata: {
    timestamp: '',
    source: 'static',
  },
};
