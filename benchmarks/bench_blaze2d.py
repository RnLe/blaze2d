#!/usr/bin/env python3
"""
Blaze2D Speed Benchmark - Unified Single/Multi-Core

Uses the bulk driver for both single-core (-j 1) and multi-core (-j N) modes.
This avoids per-job initialization overhead and provides fair comparison.

Configurations (Joannopoulos 1997):
- Config A: Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a
- Config B: Hex lattice, ε=13 bg, air rods (ε=1), r=0.48a

Parameters: 12 bands, 20 k-points per segment, 32×32 resolution

Default runs:
- Single-core: 10 runs × 10 iterations = 100 total per config
- Multi-core: 100 runs × 10 iterations = 1000 total per config
"""

import subprocess
import time
import os
import sys
import json
import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================================
# Benchmark Configuration
# ============================================================================
SINGLE_CORE_RUNS = 10    # Jobs per iteration for single-core
MULTI_CORE_RUNS = 100    # Jobs per iteration for multi-core
NUM_ITERATIONS = 5
NUM_THREADS_MULTI = 16

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Bulk config templates for each configuration
BULK_CONFIG_TEMPLATES = {
    "config_a_tm": {
        "desc": "Square lattice, air bg, ε=8.9 rods, r=0.2a",
        "polarization": "TM",
        "config_type": "config_a",
        "template": """# Config A: Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a - TM
polarization = "TM"

[bulk]
dry_run = false

[geometry]
eps_bg = 1.0

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.2
eps_inside = 8.9

[grid]
nx = 64
ny = 64
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 20

[eigensolver]
n_bands = 8

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 0.999, max = 1.001, step = {step:.10f} }}
"""
    },
    "config_a_te": {
        "desc": "Square lattice, air bg, ε=8.9 rods, r=0.2a",
        "polarization": "TE",
        "config_type": "config_a",
        "template": """# Config A: Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a - TE
polarization = "TE"

[bulk]
dry_run = false

[geometry]
eps_bg = 1.0

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.2
eps_inside = 8.9

[grid]
nx = 64
ny = 64
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 20

[eigensolver]
n_bands = 8

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 0.999, max = 1.001, step = {step:.10f} }}
"""
    },
    "config_b_tm": {
        "desc": "Hex lattice, ε=13 bg, air rods, r=0.48a",
        "polarization": "TM",
        "config_type": "config_b",
        "template": """# Config B: Hex lattice, ε=13 bg, air rods, r=0.48a - TM
polarization = "TM"

[bulk]
dry_run = false

[geometry]
eps_bg = 13.0

[geometry.lattice]
type = "hexagonal"

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.48
eps_inside = 1.0

[grid]
nx = 64
ny = 64
lx = 1.0
ly = 1.0

[path]
preset = "hexagonal"
segments_per_leg = 20

[eigensolver]
n_bands = 8

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 12.999, max = 13.001, step = {step:.10f} }}
"""
    },
    "config_b_te": {
        "desc": "Hex lattice, ε=13 bg, air rods, r=0.48a",
        "polarization": "TE",
        "config_type": "config_b",
        "template": """# Config B: Hex lattice, ε=13 bg, air rods, r=0.48a - TE
polarization = "TE"

[bulk]
dry_run = false

[geometry]
eps_bg = 13.0

[geometry.lattice]
type = "hexagonal"

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.48
eps_inside = 1.0

[grid]
nx = 64
ny = 64
lx = 1.0
ly = 1.0

[path]
preset = "hexagonal"
segments_per_leg = 20

[eigensolver]
n_bands = 8

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 12.999, max = 13.001, step = {step:.10f} }}
"""
    },
}


def build_bulk_driver():
    """Build blaze2d-bulk-driver in release mode."""
    print("Building blaze2d-bulk-driver (release)...")
    # Enable native CPU optimizations for maximum performance
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "blaze2d-bulk-driver"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env
    )
    if result.returncode != 0:
        print("ERROR: Failed to build blaze2d-bulk-driver")
        print(result.stderr)
        sys.exit(1)
    print("Build successful.")


def get_bulk_driver_binary():
    """Get path to blaze2d-bulk-driver binary."""
    binary = PROJECT_ROOT / "target" / "release" / "blaze2d-bulk-driver"
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")
    return binary


def generate_bulk_config(template: str, num_jobs: int) -> str:
    """Generate bulk config with correct step size for num_jobs."""
    # Step size to generate exactly num_jobs
    step = 0.002 / (num_jobs - 1) if num_jobs > 1 else 0.001
    return template.format(step=step)


def run_bulk_benchmark(binary: Path, config_content: str, num_threads: int) -> float:
    """Run bulk driver and return elapsed time in seconds.
    
    Uses --benchmark flag: runs real solves but skips file output.
    """
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        config_path = Path(f.name)
    
    try:
        cmd = [
            str(binary),
            "--config", str(config_path),
            "--benchmark",  # Real solves, NO file output
            "-j", str(num_threads),
        ]
        
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"\nWARNING: bulk driver returned non-zero exit code")
            print(result.stderr[:500] if result.stderr else "No stderr")
        
        return elapsed
    finally:
        config_path.unlink(missing_ok=True)


def run_benchmark_config(config_name: str, config_info: dict, binary: Path,
                         num_jobs: int, num_iterations: int, num_threads: int) -> dict:
    """Run benchmark for a specific configuration."""
    
    results = {
        "config": config_name,
        "description": config_info["desc"],
        "polarization": config_info["polarization"].lower(),
        "config_type": config_info["config_type"],
        "jobs_per_iteration": num_jobs,
        "num_iterations": num_iterations,
        "num_threads": num_threads,
        "iteration_times": [],
    }
    
    print(f"\n[{config_name}] {config_info['desc']} - {config_info['polarization']}")
    
    config_content = generate_bulk_config(config_info["template"], num_jobs)
    iteration_times = []
    
    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}: ", end="", flush=True)
        
        elapsed = run_bulk_benchmark(binary, config_content, num_threads)
        iteration_times.append(elapsed)
        
        throughput = num_jobs / elapsed
        mean_per_job = elapsed / num_jobs * 1000
        
        print(f"{elapsed:.1f}s ({throughput:.1f} jobs/s, {mean_per_job:.2f}ms/job)")
    
    results["iteration_times"] = iteration_times
    results["total_jobs"] = num_jobs * num_iterations
    
    # Calculate per-job statistics (total time / jobs in that iteration)
    per_job_times_ms = [t / num_jobs * 1000 for t in iteration_times]
    results["mean_ms"] = float(np.mean(per_job_times_ms))
    results["std_ms"] = float(np.std(per_job_times_ms))
    results["min_ms"] = float(np.min(per_job_times_ms))
    results["max_ms"] = float(np.max(per_job_times_ms))
    
    # Throughput stats
    throughputs = [num_jobs / t for t in iteration_times]
    results["mean_throughput"] = float(np.mean(throughputs))
    results["std_throughput"] = float(np.std(throughputs))
    
    return results


def convert_for_json(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Blaze2D Speed Benchmark (Bulk Driver)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single-core benchmark (10 runs × 10 iterations):
    python bench_blaze2d.py --mode single
    
  Multi-core benchmark (100 runs × 10 iterations, 16 threads):
    python bench_blaze2d.py --mode multi
    
  Quick test:
    python bench_blaze2d.py --mode single --quick
    python bench_blaze2d.py --mode multi --quick
"""
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["single", "multi"],
                        help="Benchmark mode: 'single' (1 thread) or 'multi' (N threads)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Jobs per iteration (default: 10 for single, 100 for multi)")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS,
                        help=f"Number of iterations (default: {NUM_ITERATIONS})")
    parser.add_argument("--threads", "-j", type=int, default=NUM_THREADS_MULTI,
                        help=f"Number of threads for multi mode (default: {NUM_THREADS_MULTI})")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5 jobs × 2 iterations)")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip building (use existing binary)")
    parser.add_argument("--config", type=str, default=None,
                        help="Run only specific config (e.g., 'config_a_tm')")
    args = parser.parse_args()
    
    # Set defaults based on mode
    if args.mode == "single":
        num_threads = 1
        default_jobs = SINGLE_CORE_RUNS
        core_mode = "single"
    else:
        num_threads = args.threads
        default_jobs = MULTI_CORE_RUNS
        core_mode = "multi"
    
    num_jobs = args.jobs if args.jobs is not None else default_jobs
    
    if args.quick:
        num_jobs = 5
        args.iterations = 2
    
    output_dir = SCRIPT_DIR / args.output
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print(f"Blaze2D Speed Benchmark - {core_mode.upper()}-CORE MODE (Bulk Driver)")
    print("=" * 70)
    print(f"Resolution: 64×64")
    print(f"Bands: 8")
    print(f"K-points: 61 (20 per segment)")
    print(f"Jobs per iteration: {num_jobs}")
    print(f"Iterations: {args.iterations}")
    print(f"Total jobs per config: {num_jobs * args.iterations}")
    print(f"Threads: {num_threads}")
    print(f"Output mode: --benchmark (no file output)")
    print("=" * 70)
    
    if not args.no_build:
        build_bulk_driver()
    
    binary = get_bulk_driver_binary()
    print(f"Using binary: {binary}")
    
    all_results = {
        "benchmark": f"blaze2d_speed_{core_mode}",
        "timestamp": datetime.now().isoformat(),
        "solver": "Blaze2D",
        "core_mode": core_mode,
        "num_threads": num_threads,
        "resolution": 32,
        "num_bands": 12,
        "k_points_per_segment": 20,
        "jobs_per_iteration": num_jobs,
        "num_iterations": args.iterations,
        "configs": {}
    }
    
    for config_name, config_info in BULK_CONFIG_TEMPLATES.items():
        # Skip if --config specified and doesn't match
        if args.config and config_name != args.config:
            continue
        results = run_benchmark_config(
            config_name, config_info, binary,
            num_jobs, args.iterations, num_threads
        )
        all_results["configs"][config_name] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for config_name, config_data in all_results["configs"].items():
        print(f"{config_name}: {config_data['mean_ms']:.2f} ± {config_data['std_ms']:.2f} ms/job "
              f"({config_data['mean_throughput']:.1f} jobs/s)")
    
    # Save results
    output_file = output_dir / f"blaze2d_speed_{core_mode}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # CSV summary
    csv_file = output_dir / f"blaze2d_speed_{core_mode}_summary.csv"
    with open(csv_file, 'w') as f:
        f.write("config,polarization,mean_ms,std_ms,min_ms,max_ms,throughput_jobs_s,total_jobs,core_mode,num_threads\n")
        for config_name, config_data in all_results["configs"].items():
            f.write(f"{config_name},{config_data['polarization']},"
                    f"{config_data['mean_ms']:.4f},{config_data['std_ms']:.4f},"
                    f"{config_data['min_ms']:.4f},{config_data['max_ms']:.4f},"
                    f"{config_data['mean_throughput']:.2f},{config_data['total_jobs']},"
                    f"{core_mode},{num_threads}\n")
    
    print(f"Summary CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
