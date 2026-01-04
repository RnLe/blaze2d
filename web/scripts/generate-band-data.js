#!/usr/bin/env node
/**
 * Build-time script to convert band diagram JSON data to efficient binary format.
 * 
 * This script:
 * 1. Reads the source band_diagram.json from the research folder
 * 2. Converts the band data to compact Float32 binary format
 * 3. Generates metadata JSON for loading
 * 4. Copies files to public/data/blaze/
 * 
 * Run this script during build or manually with:
 *   node scripts/generate-band-data.js
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Paths
const WEBSITE_ROOT = join(__dirname, '..')
const SOURCE_JSON = join(WEBSITE_ROOT, 'public', 'band_diagram.json')
const OUTPUT_DIR = join(WEBSITE_ROOT, 'public', 'data', 'blaze')
const OUTPUT_BIN = join(OUTPUT_DIR, 'band_data.bin')
const OUTPUT_META = join(OUTPUT_DIR, 'band_data_meta.json')

function main() {
  console.log('📊 Generating BLAZE band data...')
  console.log(`   Source: ${SOURCE_JSON}`)
  
  // Check source exists
  if (!existsSync(SOURCE_JSON)) {
    console.error(`❌ Source file not found: ${SOURCE_JSON}`)
    process.exit(1)
  }
  
  // Create output directory
  mkdirSync(OUTPUT_DIR, { recursive: true })
  
  // Read source JSON
  const jsonData = JSON.parse(readFileSync(SOURCE_JSON, 'utf-8'))
  
  const { bands, n_bands, n_points, k_path, params, symmetry_points } = jsonData
  
  console.log(`   Bands: ${n_bands}, Points: ${n_points}`)
  console.log(`   Lattice: ${params.lattice}, Polarization: ${params.polarization}`)
  
  // Calculate statistics for each band
  const bandStats = bands.map((band, i) => {
    const min = Math.min(...band)
    const max = Math.max(...band)
    return { index: i, min, max }
  })
  
  // Global min/max for y-axis scaling
  const globalMin = Math.min(...bandStats.map(s => s.min))
  const globalMax = Math.max(...bandStats.map(s => s.max))
  
  // Create binary buffer for band data
  // Layout: [band0_point0, band0_point1, ..., band0_pointN, band1_point0, ...]
  const totalFloats = n_bands * n_points
  const buffer = new ArrayBuffer(totalFloats * 4) // 4 bytes per float32
  const floatView = new Float32Array(buffer)
  
  let offset = 0
  for (let b = 0; b < n_bands; b++) {
    for (let p = 0; p < n_points; p++) {
      floatView[offset++] = bands[b][p]
    }
  }
  
  // Write binary file
  writeFileSync(OUTPUT_BIN, Buffer.from(buffer))
  console.log(`   ✅ Binary data: ${OUTPUT_BIN} (${buffer.byteLength} bytes)`)
  
  // Create metadata JSON (without the raw band data)
  const metadata = {
    n_bands,
    n_points,
    params,
    symmetry_points,
    k_path,
    stats: {
      globalMin,
      globalMax,
      bands: bandStats,
    },
    binary: {
      format: 'float32',
      layout: 'band-major', // [band][point] order
      totalBytes: buffer.byteLength,
      floatsPerBand: n_points,
    },
  }
  
  writeFileSync(OUTPUT_META, JSON.stringify(metadata, null, 2))
  console.log(`   ✅ Metadata: ${OUTPUT_META}`)
  
  // Summary
  const compressionRatio = (readFileSync(SOURCE_JSON).byteLength / buffer.byteLength).toFixed(1)
  console.log(`\n📈 Summary:`)
  console.log(`   Original JSON: ${(readFileSync(SOURCE_JSON).byteLength / 1024).toFixed(1)} KB`)
  console.log(`   Binary data: ${(buffer.byteLength / 1024).toFixed(1)} KB`)
  console.log(`   Metadata: ${(readFileSync(OUTPUT_META).byteLength / 1024).toFixed(1)} KB`)
  console.log(`   Compression: ~${compressionRatio}x smaller (data only)`)
  console.log(`\n✨ Band data generation complete!`)
}

main()
