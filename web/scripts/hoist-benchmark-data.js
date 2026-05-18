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
    
    if (singleCoreSuccess || multiCoreSuccess || series7Success) {
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

main();
