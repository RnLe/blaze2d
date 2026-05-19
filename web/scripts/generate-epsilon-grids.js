#!/usr/bin/env node

/**
 * Generate pre-smoothed epsilon grids for various resolutions.
 * 
 * This script calls blaze2d-cli to export epsilon data for resolutions 16-256 (step 16).
 * The data is saved as JSON files in public/data/epsilon-grids/.
 * 
 * Usage:
 *   node scripts/generate-epsilon-grids.js
 * 
 * Prerequisites:
 *   - blaze2d must be built: cargo build --release -p blaze2d-cli
 */

import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const RESOLUTIONS = [];
for (let r = 16; r <= 256; r += 16) {
  RESOLUTIONS.push(r);
}

const OUTPUT_DIR = path.join(__dirname, '../public/data/epsilon-grids');
const TEMP_DIR = path.join(os.tmpdir(), 'blaze-epsilon-export');
const REPO_ROOT = path.join(__dirname, '../..');

// Ensure output directories exist
fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.mkdirSync(TEMP_DIR, { recursive: true });

/**
 * Generate a TOML config for a specific resolution
 */
function generateTomlConfig(resolution) {
  return `# Auto-generated config for epsilon export (res=${resolution})

polarization = "TE"

[geometry]
eps_bg = 13.0

[geometry.lattice]
type = "square"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.3
eps_inside = 1.0

[grid]
nx = ${resolution}
ny = ${resolution}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 1

[eigensolver]
n_bands = 1
max_iter = 1
tol = 1e-2
`;
}

/**
 * Parse CSV output from blaze into a structured format
 */
function parseEpsilonCsv(csvPath) {
  const content = fs.readFileSync(csvPath, 'utf-8');
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',');
  
  const ixIdx = headers.indexOf('ix');
  const iyIdx = headers.indexOf('iy');
  const epsSmoothedIdx = headers.indexOf('eps_smoothed');
  
  if (ixIdx === -1 || iyIdx === -1 || epsSmoothedIdx === -1) {
    throw new Error(`Missing required columns in CSV. Headers: ${headers.join(', ')}`);
  }
  
  const data = [];
  let maxIx = 0;
  let maxIy = 0;
  
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    const ix = parseInt(cols[ixIdx], 10);
    const iy = parseInt(cols[iyIdx], 10);
    const eps = parseFloat(cols[epsSmoothedIdx]);
    
    maxIx = Math.max(maxIx, ix);
    maxIy = Math.max(maxIy, iy);
    data.push({ ix, iy, eps });
  }
  
  const nx = maxIx + 1;
  const ny = maxIy + 1;
  
  // Create a 2D array (row-major: grid[iy][ix])
  const grid = Array.from({ length: ny }, () => Array(nx).fill(0));
  
  for (const { ix, iy, eps } of data) {
    grid[iy][ix] = eps;
  }
  
  return { nx, ny, grid };
}

/**
 * Export epsilon for a specific resolution
 */
async function exportEpsilonForResolution(resolution) {
  const tomlPath = path.join(TEMP_DIR, `config_res${resolution}.toml`);
  const csvPath = path.join(TEMP_DIR, `epsilon_res${resolution}.csv`);
  const jsonPath = path.join(OUTPUT_DIR, `epsilon-res${resolution}.json`);
  
  console.log(`\n[${resolution}×${resolution}] Generating epsilon grid...`);
  
  // Write TOML config
  fs.writeFileSync(tomlPath, generateTomlConfig(resolution));
  
  // Run blaze CLI
  const command = [
    'cargo', 'run', '--release', '-p', 'blaze2d-cli', '--',
    '--config', tomlPath,
    '--mesh-size', '4',  // Enable smoothing (mesh-size > 1)
    '--export-epsilon', csvPath,
    '--skip-solve'
  ].join(' ');
  
  try {
    execSync(command, {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 60000  // 60 second timeout
    });
  } catch (error) {
    console.error(`  Error running blaze: ${error.message}`);
    if (error.stderr) {
      console.error(`  stderr: ${error.stderr.toString()}`);
    }
    throw error;
  }
  
  // Parse CSV and convert to JSON
  if (!fs.existsSync(csvPath)) {
    throw new Error(`CSV output not found: ${csvPath}`);
  }
  
  const { nx, ny, grid } = parseEpsilonCsv(csvPath);
  
  // Save as compact JSON
  const output = {
    resolution,
    nx,
    ny,
    epsilon_background: 13.0,
    epsilon_hole: 1.0,
    radius: 0.3,
    // Flatten grid for compact storage (row-major)
    grid: grid.flat()
  };
  
  fs.writeFileSync(jsonPath, JSON.stringify(output));
  
  const stats = fs.statSync(jsonPath);
  console.log(`  ✅ Wrote ${jsonPath} (${(stats.size / 1024).toFixed(1)} KB)`);
  
  // Clean up temp files
  fs.unlinkSync(tomlPath);
  fs.unlinkSync(csvPath);
  
  return output;
}

/**
 * Generate an index file listing all available resolutions
 */
function generateIndex(results) {
  const indexPath = path.join(OUTPUT_DIR, 'index.json');
  const index = {
    generated: new Date().toISOString(),
    epsilon_background: 13.0,
    epsilon_hole: 1.0,
    radius: 0.3,
    resolutions: results.map(r => ({
      resolution: r.resolution,
      file: `epsilon-res${r.resolution}.json`
    }))
  };
  
  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
  console.log(`\n✅ Generated index: ${indexPath}`);
}

async function main() {
  console.log('='.repeat(60));
  console.log('Generating smoothed epsilon grids for website visualization');
  console.log('='.repeat(60));
  console.log(`Resolutions: ${RESOLUTIONS.join(', ')}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  
  // Check if blaze is available
  try {
    execSync('cargo build --release -p blaze2d-cli', {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    console.log('\n✅ blaze2d-cli built successfully');
  } catch (error) {
    console.error('Failed to build blaze2d-cli');
    process.exit(1);
  }
  
  const results = [];
  
  for (const resolution of RESOLUTIONS) {
    try {
      const result = await exportEpsilonForResolution(resolution);
      results.push(result);
    } catch (error) {
      console.error(`Failed to generate epsilon for resolution ${resolution}: ${error.message}`);
    }
  }
  
  if (results.length > 0) {
    generateIndex(results);
    console.log(`\n✅ Generated ${results.length} epsilon grids`);
  } else {
    console.error('\n❌ No epsilon grids were generated');
    process.exit(1);
  }
  
  // Clean up temp directory
  try {
    fs.rmdirSync(TEMP_DIR);
  } catch (e) {
    // Ignore cleanup errors
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
