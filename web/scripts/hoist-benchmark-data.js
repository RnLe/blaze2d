#!/usr/bin/env node
/**
 * Pre-commit hook: Hoist benchmark data to public/data/
 * 
 * This script copies benchmark results from the benchmarks folder to 
 * the web/public/data/ folder so they're available in production builds
 * (e.g., GitHub Pages which can't access files outside public/).
 * 
 * Run manually: node scripts/hoist-benchmark-data.js
 * Or via pre-commit hook
 */

import { readFile, writeFile, mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = join(__dirname, '..');
const benchmarksPath = join(webRoot, '..', 'benchmarks', 'results');
const outputPath = join(webRoot, 'public', 'data', 'benchmarks');

async function hoistSingleCoreData() {
  console.log('Hoisting single-core benchmark data...');
  
  const blazePath = join(benchmarksPath, 'blaze2d_speed_single_results.json');
  const mpbPath = join(benchmarksPath, 'mpb_speed_single_results.json');

  // Check if source files exist
  if (!existsSync(blazePath) || !existsSync(mpbPath)) {
    console.log('Single-core benchmark files not found, skipping...');
    return false;
  }

  const [blazeRaw, mpbRaw] = await Promise.all([
    readFile(blazePath, 'utf-8'),
    readFile(mpbPath, 'utf-8'),
  ]);

  const blazeData = JSON.parse(blazeRaw);
  const mpbData = JSON.parse(mpbRaw);

  // Transform to unified format
  const result = {
    mpb: {},
    blaze: {},
    metadata: {
      timestamp: blazeData.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      blazeFile: 'blaze2d_speed_single_results.json',
      mpbFile: 'mpb_speed_single_results.json',
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

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'single-core.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function main() {
  console.log('Hoisting benchmark data to public/data/...\n');

  try {
    const singleCoreSuccess = await hoistSingleCoreData();
    const multiCoreSuccess = await hoistMultiCoreData();
    const series7Success = await hoistSeries7Data();
    const series5Success = await hoistSeries5Data();
    const series3Success = await hoistSeries3Data();
    const series4Success = await hoistSeries4Data();
    const series1Success = await hoistSeries1Data();
    const series2Success = await hoistSeries2Data();
    const series6Success = await hoistSeries6Data();
    
    if (singleCoreSuccess || multiCoreSuccess || series7Success || series5Success || series3Success || series4Success || series1Success || series2Success || series6Success) {
      console.log('\nBenchmark data hoisted successfully!');
    } else {
      console.log('\nSome data could not be hoisted (files may not exist yet)');
    }
  } catch (error) {
    console.error('\nError hoisting benchmark data:', error);
    process.exit(1);
  }
}

async function hoistSeries7Data() {
  console.log('Hoisting series7 scaling benchmark data...');
  
  const dataPath = join(benchmarksPath, 'series7_scaling', 'series7_scaling_results.json');

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series7 scaling benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Transform to unified format
  const result = {
    low: {
      resolution: data.results.low.resolution,
      blaze: data.results.low.blaze.map(d => ({
        threads: d.threads,
        mean_throughput: d.mean_throughput,
        std_throughput: d.std_throughput,
      })),
      mpb_omp: data.results.low.mpb_omp.map(d => ({
        threads: d.threads,
        mean_throughput: d.mean_throughput,
        std_throughput: d.std_throughput,
      })),
      mpb_multiproc: data.results.low.mpb_multiproc.map(d => ({
        threads: d.threads,
        mean_throughput: d.mean_throughput,
        std_throughput: d.std_throughput,
      })),
    },
    high: {
      resolution: data.results.high.resolution,
      blaze: data.results.high.blaze.map(d => ({
        threads: d.threads,
        mean_throughput: d.mean_throughput,
        std_throughput: d.std_throughput,
      })),
      mpb_omp: data.results.high.mpb_omp.map(d => ({
        threads: d.threads,
        mean_throughput: d.mean_throughput,
        std_throughput: d.std_throughput,
      })),
      mpb_multiproc: data.results.high.mpb_multiproc.map(d => ({
        threads: d.threads,
        mean_throughput: d.mean_throughput,
        std_throughput: d.std_throughput,
      })),
    },
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series7_scaling/series7_scaling_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series7-scaling.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistMultiCoreData() {
  console.log('Hoisting multi-core benchmark data...');
  
  const blazePath = join(benchmarksPath, 'blaze2d_speed_multi_results.json');
  const mpbPath = join(benchmarksPath, 'mpb_speed_multi-native_results.json');

  // Check if source files exist
  if (!existsSync(blazePath) || !existsSync(mpbPath)) {
    console.log('Multi-core benchmark files not found, skipping...');
    return false;
  }

  const [blazeRaw, mpbRaw] = await Promise.all([
    readFile(blazePath, 'utf-8'),
    readFile(mpbPath, 'utf-8'),
  ]);

  const blazeData = JSON.parse(blazeRaw);
  const mpbData = JSON.parse(mpbRaw);

  // Transform to unified format
  const result = {
    mpb: {},
    blaze: {},
    metadata: {
      timestamp: blazeData.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      num_threads: blazeData.num_threads || 16,
      blazeFile: 'blaze2d_speed_multi_results.json',
      mpbFile: 'mpb_speed_multi-native_results.json',
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

  // Transform MPB Native data (nested structure with polarizations)
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

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'multi-core.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistSeries5Data() {
  console.log('Hoisting series5 memory benchmark data...');
  
  const dataPath = join(benchmarksPath, 'series5_memory', 'series5_memory_results.json');

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series5 memory benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Helper function to extract sweep data
  // Supports both old format (peak_rss_mb) and new format (peak_rss_mb_mean/std)
  function extractSweepData(sweepData) {
    const result = {
      values: sweepData.values,
      fixed: sweepData.fixed,
      mpb: { TM: [], TE: [] },
      blaze: { TM: [], TE: [] },
    };

    for (const r of sweepData.results) {
      const solver = r.solver === 'MPB' ? 'mpb' : 'blaze';
      const pol = r.polarization;
      
      // Check if new format with mean/std or old format
      const hasStd = 'peak_rss_mb_mean' in r;
      
      result[solver][pol].push({
        value: r[sweepData.variable],
        memory_mb: hasStd ? r.peak_rss_mb_mean : r.peak_rss_mb,
        memory_mb_std: hasStd ? r.peak_rss_mb_std : 0,
        elapsed: hasStd ? r.elapsed_seconds_mean : r.elapsed_seconds,
        elapsed_std: hasStd ? r.elapsed_seconds_std : 0,
        num_runs: r.num_runs || 1,
      });
    }

    return result;
  }

  // Transform to unified format (only resolution and num_bands, not k_points_per_segment)
  const result = {
    config: data.config,
    resolution: extractSweepData(data.sweeps.resolution),
    num_bands: extractSweepData(data.sweeps.num_bands),
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series5_memory/series5_memory_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series5-memory.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistSeries3Data() {
  console.log('Hoisting series3 resolution benchmark data...');
  
  const dataPath = join(benchmarksPath, 'series3_resolution', 'series3_resolution_results.json');

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series3 resolution benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Transform to unified format matching the website's expected structure
  const result = {
    parameters: data.parameters,
    TM: {
      resolution: data.resolution_values || data.TM?.resolution || [],
      mpb: data.TM?.mpb || [],
      blaze: data.TM?.blaze || [],
    },
    TE: {
      resolution: data.resolution_values || data.TE?.resolution || [],
      mpb: data.TE?.mpb || [],
      blaze: data.TE?.blaze || [],
    },
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series3_resolution/series3_resolution_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series3-resolution.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistSeries4Data() {
  console.log('Hoisting series4 iterations benchmark data...');
  
  // Try both possible folder names
  let dataPath = join(benchmarksPath, 'series4_iterations', 'series4_iterations_results.json');
  if (!existsSync(dataPath)) {
    dataPath = join(benchmarksPath, 'series4_iterations_fixed', 'series4_iterations_results.json');
  }

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series4 iterations benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Helper to transform k_points array (keep only essential fields)
  function transformKPoints(kPoints) {
    return kPoints.map(kp => ({
      k_index: kp.k_index,
      iterations: kp.iterations,
      elapsed_seconds: kp.elapsed_seconds,
    }));
  }

  // Transform to unified format
  const result = {
    parameters: data.parameters,
    TM: {
      mpb: data.TM?.mpb ? {
        solver: data.TM.mpb.solver,
        polarization: data.TM.mpb.polarization,
        total_elapsed: data.TM.mpb.total_elapsed,
        k_points: transformKPoints(data.TM.mpb.k_points),
      } : null,
      blaze: data.TM?.blaze ? {
        solver: data.TM.blaze.solver,
        polarization: data.TM.blaze.polarization,
        total_elapsed: data.TM.blaze.total_elapsed,
        k_points: transformKPoints(data.TM.blaze.k_points),
      } : null,
    },
    TE: {
      mpb: data.TE?.mpb ? {
        solver: data.TE.mpb.solver,
        polarization: data.TE.mpb.polarization,
        total_elapsed: data.TE.mpb.total_elapsed,
        k_points: transformKPoints(data.TE.mpb.k_points),
      } : null,
      blaze: data.TE?.blaze ? {
        solver: data.TE.blaze.solver,
        polarization: data.TE.blaze.polarization,
        total_elapsed: data.TE.blaze.total_elapsed,
        k_points: transformKPoints(data.TE.blaze.k_points),
      } : null,
    },
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series4_iterations/series4_iterations_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series4-iterations.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistSeries1Data() {
  console.log('Hoisting series1 epsilon benchmark data...');
  
  const dataPath = join(benchmarksPath, 'series1_epsilon', 'series1_epsilon_results.json');

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series1 epsilon benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Transform to unified format matching the website's expected structure
  const result = {
    parameters: data.parameters,
    epsilon_values: data.epsilon_values,
    TM: {
      mpb: data.TM?.mpb || [],
      blaze: data.TM?.blaze || [],
    },
    TE: {
      mpb: data.TE?.mpb || [],
      blaze: data.TE?.blaze || [],
    },
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series1_epsilon/series1_epsilon_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series1-epsilon.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistSeries2Data() {
  console.log('Hoisting series2 bands benchmark data...');
  
  const dataPath = join(benchmarksPath, 'series2_bands', 'series2_bands_results.json');

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series2 bands benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Transform to unified format matching the website's expected structure
  const result = {
    parameters: data.parameters,
    band_values: data.band_values,
    TM: {
      bands: data.band_values,
      mpb: data.TM?.mpb || [],
      blaze: data.TM?.blaze || [],
    },
    TE: {
      bands: data.band_values,
      mpb: data.TE?.mpb || [],
      blaze: data.TE?.blaze || [],
    },
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series2_bands/series2_bands_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series2-bands.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

async function hoistSeries6Data() {
  console.log('Hoisting series6 accuracy benchmark data...');
  
  const dataPath = join(benchmarksPath, 'series6_accuracy', 'series6_accuracy_results.json');

  // Check if source file exists
  if (!existsSync(dataPath)) {
    console.log('Series6 accuracy benchmark file not found, skipping...');
    return false;
  }

  const rawData = await readFile(dataPath, 'utf-8');
  const data = JSON.parse(rawData);

  // Helper to extract band data (k_distance and frequencies for each k-point)
  function extractBandData(kPoints) {
    return kPoints.map(kp => ({
      k_distance: kp.k_distance,
      frequencies: kp.frequencies,
    }));
  }

  // Helper to compute boxplot statistics from deviation data
  function computeBoxplotStats(deviations) {
    if (!deviations || deviations.length === 0) return null;
    
    const values = deviations.map(d => d.rel_deviation).sort((a, b) => a - b);
    const n = values.length;
    
    const q1Idx = Math.floor(n * 0.25);
    const q2Idx = Math.floor(n * 0.5);
    const q3Idx = Math.floor(n * 0.75);
    
    const q1 = values[q1Idx];
    const median = values[q2Idx];
    const q3 = values[q3Idx];
    const iqr = q3 - q1;
    
    // Whiskers: min/max within 1.5*IQR of Q1/Q3
    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;
    
    const whiskerLow = values.find(v => v >= lowerFence) || values[0];
    const whiskerHigh = [...values].reverse().find(v => v <= upperFence) || values[n - 1];
    
    return {
      min: values[0],
      q1,
      median,
      q3,
      max: values[n - 1],
      whiskerLow,
      whiskerHigh,
      mean: values.reduce((a, b) => a + b, 0) / n,
    };
  }

  // Transform to unified format
  const result = {
    parameters: data.parameters,
    TM: {
      mpb: extractBandData(data.results.TM?.mpb?.k_points || []),
      blaze_f32: extractBandData(data.results.TM?.blaze_f32?.k_points || []),
      blaze_f64: extractBandData(data.results.TM?.blaze_f64?.k_points || []),
      deviations: {
        f32_vs_mpb: computeBoxplotStats(data.results.TM?.f32_vs_mpb),
        f64_vs_mpb: computeBoxplotStats(data.results.TM?.f64_vs_mpb),
        f32_vs_f64: computeBoxplotStats(data.results.TM?.f32_vs_f64),
      },
    },
    TE: {
      mpb: extractBandData(data.results.TE?.mpb?.k_points || []),
      blaze_f32: extractBandData(data.results.TE?.blaze_f32?.k_points || []),
      blaze_f64: extractBandData(data.results.TE?.blaze_f64?.k_points || []),
      deviations: {
        f32_vs_mpb: computeBoxplotStats(data.results.TE?.f32_vs_mpb),
        f64_vs_mpb: computeBoxplotStats(data.results.TE?.f64_vs_mpb),
        f32_vs_f64: computeBoxplotStats(data.results.TE?.f32_vs_f64),
      },
    },
    metadata: {
      timestamp: data.timestamp,
      source: 'static',
      hoistedAt: new Date().toISOString(),
      sourceFile: 'series6_accuracy/series6_accuracy_results.json',
    },
  };

  // Ensure output directory exists
  await mkdir(outputPath, { recursive: true });

  // Write transformed data
  const outputFile = join(outputPath, 'series6-accuracy.json');
  await writeFile(outputFile, JSON.stringify(result, null, 2));
  console.log(`✅ Wrote ${outputFile}`);

  return true;
}

main();
