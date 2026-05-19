#!/usr/bin/env python3
"""
Benchmark Series 5: Memory Usage Comparison

This benchmark compares memory usage between MPB and Blaze2D across different
configurations. Memory is measured as Peak RSS (Resident Set Size) - the maximum
physical memory the process uses during execution.

Three parameter sweeps:
1. Resolution: 16, 32, 48, 64, 128 (fixed: bands=8, k_per_seg=10)
2. Bands: 4, 8, 12, 16, 20 (fixed: res=32, k_per_seg=10)
3. K-points per segment: 6, 12, 18, 24, 30 (fixed: res=32, bands=8)

Memory is measured using /usr/bin/time -v for subprocess calls, which gives
reliable Peak RSS measurements that are comparable across Python and Rust.

Each configuration is run multiple times (default: 3) to get error bars.

Output: results/series5_memory/
"""

import subprocess
import time
import os
import sys
import json
import argparse
import tempfile
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Fixed parameters (Config A: Square lattice with ε=8.9 rods)
EPSILON = 8.9
RADIUS = 0.2
EPS_BG = 1.0

# Tolerances (MPB uses f64, Blaze uses mixed-precision f32)
MPB_TOLERANCE = 1e-7
BLAZE_TOLERANCE = 1e-4

# Parameter sweeps (5 steps each)
RESOLUTION_SWEEP = [16, 32, 48, 64, 128]
BANDS_SWEEP = [4, 8, 12, 16, 20]
K_POINTS_SWEEP = [6, 12, 18, 24, 30]

# Default fixed values
DEFAULT_RESOLUTION = 32
DEFAULT_BANDS = 8
DEFAULT_K_PER_SEG = 10

# Number of runs per configuration for error bars
DEFAULT_NUM_RUNS = 3

# Single-core environment variables
SINGLE_CORE_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
}


@dataclass
class MemoryResult:
    """Memory measurement result for a single run."""
    solver: str
    polarization: str
    resolution: int
    num_bands: int
    k_points_per_segment: int
    peak_rss_kb: int  # Peak RSS in KB
    peak_rss_mb: float  # Peak RSS in MB
    elapsed_seconds: float
    success: bool
    error_message: str = ""


@dataclass
class AggregatedMemoryResult:
    """Aggregated memory measurement result across multiple runs."""
    solver: str
    polarization: str
    resolution: int
    num_bands: int
    k_points_per_segment: int
    peak_rss_mb_mean: float
    peak_rss_mb_std: float
    elapsed_seconds_mean: float
    elapsed_seconds_std: float
    num_runs: int
    success: bool
    raw_results: List[Dict] = field(default_factory=list)


def parse_time_v_output(output: str) -> Tuple[int, float]:
    """
    Parse /usr/bin/time -v output to extract peak RSS and elapsed time.
    
    Returns:
        Tuple of (peak_rss_kb, elapsed_seconds)
    """
    peak_rss_kb = 0
    elapsed_seconds = 0.0
    
    # Peak RSS pattern: "Maximum resident set size (kbytes): 12345"
    rss_match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', output)
    if rss_match:
        peak_rss_kb = int(rss_match.group(1))
    
    # Elapsed time pattern: "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.23"
    # or "0:00.12" for sub-minute times
    time_match = re.search(r'Elapsed \(wall clock\) time.*?:\s*(?:(\d+):)?(\d+):(\d+\.?\d*)', output)
    if time_match:
        hours = int(time_match.group(1) or 0)
        minutes = int(time_match.group(2))
        seconds = float(time_match.group(3))
        elapsed_seconds = hours * 3600 + minutes * 60 + seconds
    
    return peak_rss_kb, elapsed_seconds


def run_mpb_memory_test(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int,
) -> MemoryResult:
    """
    Run MPB and measure peak memory usage.
    
    Uses /usr/bin/time -v to measure Peak RSS.
    """
    pol_lower = polarization.lower()
    
    # Create MPB script
    script = f'''
import meep as mp
from meep import mpb

geometry = [
    mp.Cylinder(
        radius={RADIUS},
        material=mp.Medium(epsilon={EPSILON}),
        center=mp.Vector3(0, 0, 0),
    )
]

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))

k_points = mp.interpolate({k_points_per_segment - 1}, [
    mp.Vector3(0, 0, 0),      # Gamma
    mp.Vector3(0.5, 0, 0),    # X
    mp.Vector3(0.5, 0.5, 0),  # M
    mp.Vector3(0, 0, 0),      # Gamma
])

solver = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    resolution={resolution},
    num_bands={num_bands},
    default_material=mp.Medium(epsilon={EPS_BG}),
    tolerance={MPB_TOLERANCE},
)

solver.run_{pol_lower}()
'''
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = Path(f.name)
    
    try:
        mpb_python = "/home/renlephy/.local/share/mamba/envs/mpb-reference/bin/python"
        
        # Merge single-core env with current env
        env = os.environ.copy()
        env.update(SINGLE_CORE_ENV)
        
        # Run with /usr/bin/time -v to measure memory
        result = subprocess.run(
            ["/usr/bin/time", "-v", mpb_python, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            env=env,
        )
        
        # /usr/bin/time outputs to stderr
        output = result.stderr
        
        peak_rss_kb, elapsed = parse_time_v_output(output)
        
        return MemoryResult(
            solver="MPB",
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            k_points_per_segment=k_points_per_segment,
            peak_rss_kb=peak_rss_kb,
            peak_rss_mb=peak_rss_kb / 1024,
            elapsed_seconds=elapsed,
            success=result.returncode == 0,
            error_message="" if result.returncode == 0 else result.stderr[:200],
        )
    except subprocess.TimeoutExpired:
        return MemoryResult(
            solver="MPB",
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            k_points_per_segment=k_points_per_segment,
            peak_rss_kb=0,
            peak_rss_mb=0.0,
            elapsed_seconds=0.0,
            success=False,
            error_message="Timeout after 600s",
        )
    finally:
        script_path.unlink(missing_ok=True)


def create_blaze_config(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int,
) -> str:
    """Create a TOML configuration for Blaze2D."""
    return f'''# Blaze2D memory test config
polarization = "{polarization}"

[geometry]
eps_bg = {EPS_BG}

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {RADIUS}
eps_inside = {EPSILON}

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = {k_points_per_segment}

[eigensolver]
num_bands = {num_bands}
tol = {BLAZE_TOLERANCE}
'''


def run_blaze_memory_test(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int,
) -> MemoryResult:
    """
    Run Blaze2D and measure peak memory usage.
    
    Uses /usr/bin/time -v to measure Peak RSS.
    """
    # Create config file
    config = create_blaze_config(resolution, num_bands, polarization, k_points_per_segment)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config)
        config_path = Path(f.name)
    
    # Output file for CSV (we don't need it but Blaze requires it)
    output_path = tempfile.mktemp(suffix='.csv')
    
    try:
        blaze_cli = PROJECT_ROOT / "target" / "release" / "blaze2d-cli"
        
        if not blaze_cli.exists():
            # Try to build it
            subprocess.run(
                ["cargo", "build", "--release", "-p", "blaze2d-cli"],
                cwd=PROJECT_ROOT,
                capture_output=True,
            )
        
        # Merge single-core env with current env
        env = os.environ.copy()
        env.update(SINGLE_CORE_ENV)
        
        # Run with /usr/bin/time -v to measure memory
        result = subprocess.run(
            ["/usr/bin/time", "-v", str(blaze_cli), 
             "--config", str(config_path),
             "--output", output_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=PROJECT_ROOT,
            env=env,
        )
        
        output = result.stderr
        peak_rss_kb, elapsed = parse_time_v_output(output)
        
        return MemoryResult(
            solver="Blaze2D",
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            k_points_per_segment=k_points_per_segment,
            peak_rss_kb=peak_rss_kb,
            peak_rss_mb=peak_rss_kb / 1024,
            elapsed_seconds=elapsed,
            success=result.returncode == 0,
            error_message="" if result.returncode == 0 else result.stderr[:200],
        )
    except subprocess.TimeoutExpired:
        return MemoryResult(
            solver="Blaze2D",
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            k_points_per_segment=k_points_per_segment,
            peak_rss_kb=0,
            peak_rss_mb=0.0,
            elapsed_seconds=0.0,
            success=False,
            error_message="Timeout after 600s",
        )
    finally:
        config_path.unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def run_sweep(
    sweep_variable: str,
    sweep_values: List[int],
    fixed_resolution: int,
    fixed_bands: int,
    fixed_k_per_seg: int,
    polarizations: List[str] = ["TM", "TE"],
    num_runs: int = DEFAULT_NUM_RUNS,
) -> List[AggregatedMemoryResult]:
    """Run a parameter sweep and collect aggregated memory results with error bars."""
    
    aggregated_results = []
    
    for value in sweep_values:
        # Set parameters based on sweep variable
        if sweep_variable == "resolution":
            res, bands, k_seg = value, fixed_bands, fixed_k_per_seg
        elif sweep_variable == "num_bands":
            res, bands, k_seg = fixed_resolution, value, fixed_k_per_seg
        elif sweep_variable == "k_points_per_segment":
            res, bands, k_seg = fixed_resolution, fixed_bands, value
        else:
            raise ValueError(f"Unknown sweep variable: {sweep_variable}")
        
        for pol in polarizations:
            print(f"  {sweep_variable}={value}, {pol}:", end=" ", flush=True)
            
            # Run MPB multiple times
            print(f"MPB×{num_runs}...", end=" ", flush=True)
            mpb_runs = []
            for run_idx in range(num_runs):
                mpb_result = run_mpb_memory_test(res, bands, pol, k_seg)
                if mpb_result.success:
                    mpb_runs.append(mpb_result)
            
            if mpb_runs:
                mpb_mem_values = [r.peak_rss_mb for r in mpb_runs]
                mpb_time_values = [r.elapsed_seconds for r in mpb_runs]
                mpb_agg = AggregatedMemoryResult(
                    solver="MPB",
                    polarization=pol,
                    resolution=res,
                    num_bands=bands,
                    k_points_per_segment=k_seg,
                    peak_rss_mb_mean=statistics.mean(mpb_mem_values),
                    peak_rss_mb_std=statistics.stdev(mpb_mem_values) if len(mpb_mem_values) > 1 else 0.0,
                    elapsed_seconds_mean=statistics.mean(mpb_time_values),
                    elapsed_seconds_std=statistics.stdev(mpb_time_values) if len(mpb_time_values) > 1 else 0.0,
                    num_runs=len(mpb_runs),
                    success=True,
                    raw_results=[asdict(r) for r in mpb_runs],
                )
                aggregated_results.append(mpb_agg)
            
            # Run Blaze2D multiple times
            print(f"Blaze×{num_runs}...", end=" ", flush=True)
            blaze_runs = []
            for run_idx in range(num_runs):
                blaze_result = run_blaze_memory_test(res, bands, pol, k_seg)
                if blaze_result.success:
                    blaze_runs.append(blaze_result)
            
            if blaze_runs:
                blaze_mem_values = [r.peak_rss_mb for r in blaze_runs]
                blaze_time_values = [r.elapsed_seconds for r in blaze_runs]
                blaze_agg = AggregatedMemoryResult(
                    solver="Blaze2D",
                    polarization=pol,
                    resolution=res,
                    num_bands=bands,
                    k_points_per_segment=k_seg,
                    peak_rss_mb_mean=statistics.mean(blaze_mem_values),
                    peak_rss_mb_std=statistics.stdev(blaze_mem_values) if len(blaze_mem_values) > 1 else 0.0,
                    elapsed_seconds_mean=statistics.mean(blaze_time_values),
                    elapsed_seconds_std=statistics.stdev(blaze_time_values) if len(blaze_time_values) > 1 else 0.0,
                    num_runs=len(blaze_runs),
                    success=True,
                    raw_results=[asdict(r) for r in blaze_runs],
                )
                aggregated_results.append(blaze_agg)
            
            # Print summary
            if mpb_runs and blaze_runs:
                mpb_mean = statistics.mean([r.peak_rss_mb for r in mpb_runs])
                blaze_mean = statistics.mean([r.peak_rss_mb for r in blaze_runs])
                ratio = mpb_mean / blaze_mean if blaze_mean > 0 else 0
                mpb_std = statistics.stdev([r.peak_rss_mb for r in mpb_runs]) if len(mpb_runs) > 1 else 0
                blaze_std = statistics.stdev([r.peak_rss_mb for r in blaze_runs]) if len(blaze_runs) > 1 else 0
                print(f"MPB={mpb_mean:.1f}±{mpb_std:.1f}MB, Blaze={blaze_mean:.1f}±{blaze_std:.1f}MB, ratio={ratio:.1f}×")
            else:
                print(f"MPB={'OK' if mpb_runs else 'FAIL'}, Blaze={'OK' if blaze_runs else 'FAIL'}")
    
    return aggregated_results


def run_series(output_dir: Path, num_runs: int = DEFAULT_NUM_RUNS):
    """Run the full memory benchmark series."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Benchmark Series 5: Memory Usage Comparison")
    print("=" * 70)
    print(f"Config: Square lattice, ε={EPSILON} rods, r={RADIUS}a")
    print(f"Runs per config: {num_runs}")
    print(f"Single-core mode: OMP_NUM_THREADS=1")
    print(f"MPB tolerance: {MPB_TOLERANCE} (f64)")
    print(f"Blaze tolerance: {BLAZE_TOLERANCE} (mixed-precision f32)")
    print("=" * 70)
    
    all_results = {
        "series": "series5_memory",
        "description": "Memory usage comparison (Peak RSS)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epsilon": EPSILON,
            "radius": RADIUS,
            "eps_bg": EPS_BG,
            "mpb_tolerance": MPB_TOLERANCE,
            "blaze_tolerance": BLAZE_TOLERANCE,
            "num_runs": num_runs,
            "single_core": True,
        },
        "sweeps": {},
    }
    
    # Sweep 1: Resolution
    print(f"\n[Sweep 1/3] Resolution: {RESOLUTION_SWEEP}")
    print(f"  Fixed: bands={DEFAULT_BANDS}, k_per_seg={DEFAULT_K_PER_SEG}")
    print("-" * 50)
    
    res_results = run_sweep(
        sweep_variable="resolution",
        sweep_values=RESOLUTION_SWEEP,
        fixed_resolution=0,  # Will be overridden
        fixed_bands=DEFAULT_BANDS,
        fixed_k_per_seg=DEFAULT_K_PER_SEG,
        num_runs=num_runs,
    )
    all_results["sweeps"]["resolution"] = {
        "variable": "resolution",
        "values": RESOLUTION_SWEEP,
        "fixed": {"num_bands": DEFAULT_BANDS, "k_points_per_segment": DEFAULT_K_PER_SEG},
        "results": [asdict(r) for r in res_results],
    }
    
    # Sweep 2: Number of bands
    print(f"\n[Sweep 2/3] Bands: {BANDS_SWEEP}")
    print(f"  Fixed: resolution={DEFAULT_RESOLUTION}, k_per_seg={DEFAULT_K_PER_SEG}")
    print("-" * 50)
    
    bands_results = run_sweep(
        sweep_variable="num_bands",
        sweep_values=BANDS_SWEEP,
        fixed_resolution=DEFAULT_RESOLUTION,
        fixed_bands=0,  # Will be overridden
        fixed_k_per_seg=DEFAULT_K_PER_SEG,
        num_runs=num_runs,
    )
    all_results["sweeps"]["num_bands"] = {
        "variable": "num_bands",
        "values": BANDS_SWEEP,
        "fixed": {"resolution": DEFAULT_RESOLUTION, "k_points_per_segment": DEFAULT_K_PER_SEG},
        "results": [asdict(r) for r in bands_results],
    }
    
    # Sweep 3: K-points per segment
    print(f"\n[Sweep 3/3] K-points per segment: {K_POINTS_SWEEP}")
    print(f"  Fixed: resolution={DEFAULT_RESOLUTION}, bands={DEFAULT_BANDS}")
    print("-" * 50)
    
    kpts_results = run_sweep(
        sweep_variable="k_points_per_segment",
        sweep_values=K_POINTS_SWEEP,
        fixed_resolution=DEFAULT_RESOLUTION,
        fixed_bands=DEFAULT_BANDS,
        fixed_k_per_seg=0,  # Will be overridden
        num_runs=num_runs,
    )
    all_results["sweeps"]["k_points_per_segment"] = {
        "variable": "k_points_per_segment",
        "values": K_POINTS_SWEEP,
        "fixed": {"resolution": DEFAULT_RESOLUTION, "num_bands": DEFAULT_BANDS},
        "results": [asdict(r) for r in kpts_results],
    }
    
    # Save results
    results_file = output_dir / "series5_memory_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_file}")
    print("=" * 70)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Series 5: Memory Usage")
    parser.add_argument("--output", type=str, default="results/series5_memory",
                        help="Output directory")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Number of runs per configuration (default: {DEFAULT_NUM_RUNS})")
    args = parser.parse_args()
    
    output_dir = SCRIPT_DIR / args.output
    run_series(output_dir, num_runs=args.runs)


if __name__ == "__main__":
    main()
