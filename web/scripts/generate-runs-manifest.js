#!/usr/bin/env node
/**
 * Build-time script to scan moire_envelope/runs folders and generate a manifest
 * for the Envelope Approximation Pipeline viewer.
 * 
 * This script:
 * 1. Scans the runs directory for pipeline run folders
 * 2. For each run, finds the TOPMOST (first alphabetically) candidate
 * 3. Checks which phase files exist
 * 4. Copies relevant JSON/CSV data files to public/data/runs/
 * 5. Converts HDF5 field data to compact binary format for web
 * 6. Generates a runs-manifest.json in public/data/
 * 
 * Run this script during build or manually with:
 *   node scripts/generate-runs-manifest.js
 * 
 * NOTE: This cannot be run from the browser due to security restrictions.
 * The browser sandbox prevents filesystem access outside of user-initiated file picks.
 */

import { readdirSync, statSync, existsSync, writeFileSync, mkdirSync, copyFileSync, rmSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execSync } from 'node:child_process'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Paths
const WEBSITE_ROOT = join(__dirname, '..')
const RUNS_DIR = join(WEBSITE_ROOT, '..', '..', '..', 'research', 'moire_envelope', 'runs')
const OUTPUT_DIR = join(WEBSITE_ROOT, 'public', 'data')
const RUNS_OUTPUT_DIR = join(OUTPUT_DIR, 'runs')
const OUTPUT_FILE = join(OUTPUT_DIR, 'runs-manifest.json')
const PYTHON_CONVERTER = join(__dirname, 'convert_phase1_data.py')

// Conda environment name
const CONDA_ENV = 'msl'

// Files to copy to public (web-accessible formats only)
const FILES_TO_COPY = [
  'phase0_meta.json',
  'phase1_field_stats.json',
  'phase2_operator_meta.json',
  'phase2_operator_info.csv',
  'phase3_eigenvalues.csv',
  'phase3_solver_meta.json',
  'phase4_bandstructure.csv',
  'phase4_validation_summary.csv',
  'phase5_q_factor_results.csv',
]

// Phase file requirements
const PHASE_REQUIREMENTS = {
  phase0: ['phase0_meta.json'],
  phase1: ['phase1_band_data.h5', 'phase1_field_stats.json'],
  phase2: ['phase2_operator.npz', 'phase2_R_grid.npy', 'phase2_operator_meta.json', 'phase2_report.md'],
  phase3: ['phase3_eigenstates.h5', 'phase3_eigenvalues.csv', 'phase3_report.md'],
  phase4: ['phase4_bandstructure.csv', 'phase4_gamma_modes.h5', 'phase4_validation_summary.csv'],
  phase5: ['phase5_q_factor_results.csv', 'phase5_report.md'],
}

const PHASE_ORDER = ['phase0', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5']

/**
 * Determine pipeline type from run folder name
 */
function getPipelineType(folderName) {
  if (folderName.includes('blaze')) {
    return 'blaze'
  }
  return 'mpb'
}

/**
 * Parse timestamp from folder name
 * Expected formats:
 *   phase0_blaze_20251201_140949 -> 2025-12-01 14:09:49
 *   phase0_real_run_20251119_172848 -> 2025-11-19 17:28:48
 */
function parseTimestamp(folderName) {
  // Match patterns like _20251201_140949 or _20251119_172848
  const match = folderName.match(/_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/)
  if (match) {
    const [, year, month, day, hour, min, sec] = match
    return `${year}-${month}-${day} ${hour}:${min}:${sec}`
  }
  return folderName
}

/**
 * List files in a directory (non-recursive)
 */
function listFiles(dirPath) {
  if (!existsSync(dirPath)) return []
  try {
    return readdirSync(dirPath).filter(f => {
      const fullPath = join(dirPath, f)
      return statSync(fullPath).isFile()
    })
  } catch {
    return []
  }
}

/**
 * List subdirectories in a directory
 */
function listDirs(dirPath) {
  if (!existsSync(dirPath)) return []
  try {
    return readdirSync(dirPath).filter(f => {
      const fullPath = join(dirPath, f)
      return statSync(fullPath).isDirectory()
    })
  } catch {
    return []
  }
}

/**
 * Compute phase status from file list
 */
function computePhaseStatus(files) {
  const status = {}
  for (const phase of Object.keys(PHASE_REQUIREMENTS)) {
    const required = PHASE_REQUIREMENTS[phase]
    status[phase] = required.every(req => files.includes(req))
  }
  return status
}

/**
 * Copy relevant data files to public/data/runs/
 */
function copyDataFiles(runFolder, candidateId, sourcePath) {
  const destDir = join(RUNS_OUTPUT_DIR, runFolder, candidateId)
  
  // Ensure destination directory exists
  if (!existsSync(destDir)) {
    mkdirSync(destDir, { recursive: true })
  }
  
  let copiedCount = 0
  for (const file of FILES_TO_COPY) {
    const srcFile = join(sourcePath, file)
    const destFile = join(destDir, file)
    
    if (existsSync(srcFile)) {
      copyFileSync(srcFile, destFile)
      copiedCount++
    }
  }
  
  return copiedCount
}

/**
 * Convert phase1_band_data.h5 to binary format using Python
 */
function convertPhase1Data(runFolder, candidateId, sourcePath) {
  const h5File = join(sourcePath, 'phase1_band_data.h5')
  const destDir = join(RUNS_OUTPUT_DIR, runFolder, candidateId)
  
  if (!existsSync(h5File)) {
    return false
  }
  
  // Ensure destination directory exists
  if (!existsSync(destDir)) {
    mkdirSync(destDir, { recursive: true })
  }
  
  try {
    // Run Python converter using mamba
    const cmd = `mamba run -n ${CONDA_ENV} python "${PYTHON_CONVERTER}" "${h5File}" "${destDir}"`
    execSync(cmd, { stdio: 'pipe' })
    return true
  } catch (err) {
    console.error(`     ⚠️  Failed to convert HDF5: ${err.message}`)
    return false
  }
}

/**
 * Main function to scan runs and generate manifest
 */
function generateManifest() {
  console.log('📁 Scanning runs directory:', RUNS_DIR)
  
  if (!existsSync(RUNS_DIR)) {
    console.error('❌ Runs directory not found:', RUNS_DIR)
    process.exit(1)
  }
  
  // Clean and recreate runs output directory
  if (existsSync(RUNS_OUTPUT_DIR)) {
    console.log('🧹 Cleaning existing runs output directory...')
    rmSync(RUNS_OUTPUT_DIR, { recursive: true, force: true })
  }
  mkdirSync(RUNS_OUTPUT_DIR, { recursive: true })
  
  const runFolders = listDirs(RUNS_DIR)
    .filter(f => f.startsWith('phase0_'))
    .sort()
  
  console.log(`   Found ${runFolders.length} run folders`)
  
  const runs = []
  let totalFilesCopied = 0
  
  for (const runFolder of runFolders) {
    const runPath = join(RUNS_DIR, runFolder)
    const pipelineType = getPipelineType(runFolder)
    const timestamp = parseTimestamp(runFolder)
    
    // Find candidate folders (start with "candidate_")
    const candidateFolders = listDirs(runPath)
      .filter(f => f.startsWith('candidate_'))
      .sort()
    
    if (candidateFolders.length === 0) {
      console.log(`   ⚠️  ${runFolder}: No candidate folders found, skipping`)
      continue
    }
    
    // Take only the TOPMOST (first alphabetically) candidate
    const topCandidate = candidateFolders[0]
    const candidatePath = join(runPath, topCandidate)
    const files = listFiles(candidatePath)
    const phaseStatus = computePhaseStatus(files)
    
    // Copy data files to public directory
    const copiedCount = copyDataFiles(runFolder, topCandidate, candidatePath)
    totalFilesCopied += copiedCount
    
    // Convert HDF5 field data to binary format
    const h5Converted = convertPhase1Data(runFolder, topCandidate, candidatePath)
    
    // Count completed phases
    const completedPhases = PHASE_ORDER.filter(p => phaseStatus[p]).length
    
    const h5Status = h5Converted ? '✓ h5' : ''
    console.log(`   ✓ ${runFolder}: ${topCandidate} (${pipelineType}, ${completedPhases}/${PHASE_ORDER.length} phases, ${copiedCount} files ${h5Status})`)
    
    runs.push({
      name: runFolder,
      path: `/data/runs/${runFolder}`,
      timestamp,
      pipelineType,
      candidate: {
        id: topCandidate,
        path: `/data/runs/${runFolder}/${topCandidate}`,
        files,
        phaseStatus,
      },
    })
  }
  
  // Sort by timestamp descending (newest first)
  runs.sort((a, b) => b.timestamp.localeCompare(a.timestamp))
  
  const manifest = {
    generatedAt: new Date().toISOString(),
    runsDir: RUNS_DIR,
    runs,
  }
  
  // Ensure output directory exists
  if (!existsSync(OUTPUT_DIR)) {
    mkdirSync(OUTPUT_DIR, { recursive: true })
  }
  
  // Write manifest
  writeFileSync(OUTPUT_FILE, JSON.stringify(manifest, null, 2))
  console.log(`\n✅ Manifest written to: ${OUTPUT_FILE}`)
  console.log(`   Total runs: ${runs.length}`)
  console.log(`   Total files copied: ${totalFilesCopied}`)
  
  return manifest
}

// Run if called directly
generateManifest()
