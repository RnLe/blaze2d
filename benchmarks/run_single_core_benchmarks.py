#!/usr/bin/env python3
"""
Single-Core Benchmark Runner - Generates separate output files per config

This script runs single-core benchmarks for both MPB and Blaze2D,
generating separate JSON files for each configuration to allow
incremental runs without losing data.

Output structure:
  results/single_core/
    mpb_config_a_tm.json
    mpb_config_a_te.json
    mpb_config_b_tm.json
    mpb_config_b_te.json
    blaze2d_config_a_tm.json
    blaze2d_config_a_te.json
    blaze2d_config_b_tm.json
    blaze2d_config_b_te.json

Usage:
  # Run all MPB benchmarks
  python run_single_core_benchmarks.py --solver mpb

  # Run all Blaze2D benchmarks  
  python run_single_core_benchmarks.py --solver blaze2d

  # Run specific config only
  python run_single_core_benchmarks.py --solver mpb --config config_a_tm

  # Run all benchmarks
  python run_single_core_benchmarks.py --solver all

  # Quick test mode
  python run_single_core_benchmarks.py --solver all --quick
"""

import sys
import os
import argparse
import json
import time
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# Parse cores arg BEFORE importing numpy/meep (critical for MPB single-core)
def _should_set_single_core():
    for arg in sys.argv:
        if "mpb" in arg.lower() or "--solver" not in sys.argv:
            return True
    return True

if _should_set_single_core():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["GOTO_NUM_THREADS"] = "1"
    os.environ["OMP_PROC_BIND"] = "false"

import numpy as np

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "single_core"

# Default parameters
DEFAULT_RUNS = 10
DEFAULT_ITERATIONS = 5
QUICK_RUNS = 3
QUICK_ITERATIONS = 2

# Resolution and bands
RESOLUTION = 64
NUM_BANDS = 8
K_POINTS_PER_SEGMENT = 20

# Config definitions
CONFIGS = {
    "config_a_tm": {
        "lattice": "square",
        "desc": "Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a",
        "polarization": "tm",
        "eps_bg": 1.0,
        "eps_rod": 8.9,
        "radius": 0.2,
    },
    "config_a_te": {
        "lattice": "square",
        "desc": "Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a",
        "polarization": "te",
        "eps_bg": 1.0,
        "eps_rod": 8.9,
        "radius": 0.2,
    },
    "config_b_tm": {
        "lattice": "hexagonal",
        "desc": "Hex lattice, ε=13 bg, air rods (ε=1), r=0.48a",
        "polarization": "tm",
        "eps_bg": 13.0,
        "eps_rod": 1.0,
        "radius": 0.48,
    },
    "config_b_te": {
        "lattice": "hexagonal",
        "desc": "Hex lattice, ε=13 bg, air rods (ε=1), r=0.48a",
        "polarization": "te",
        "eps_bg": 13.0,
        "eps_rod": 1.0,
        "radius": 0.48,
    },
}

# ============================================================================
# MPB Benchmark Functions
# ============================================================================

def run_mpb_benchmark(config_name: str, config: dict, num_runs: int, 
                      num_iterations: int, output_dir: Path) -> dict:
    """Run MPB benchmark for a single configuration."""
    
    try:
        import meep as mp
        from meep import mpb
    except ImportError:
        print("ERROR: MPB not available. Activate mpb-reference environment.")
        return None
    
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_output():
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stdout_fd)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)
    
    def build_k_path_square(density: int):
        nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
        return interpolate_path(nodes, density, mp)
    
    def build_k_path_hex(density: int):
        nodes = [(0.0, 0.0), (0.5, 0.0), (1/3, 1/3), (0.0, 0.0)]
        return interpolate_path(nodes, density, mp)
    
    def interpolate_path(nodes, density, mp):
        vectors = []
        for seg in range(len(nodes) - 1):
            start = np.array(nodes[seg])
            end = np.array(nodes[seg + 1])
            if seg == 0:
                vectors.append(mp.Vector3(start[0], start[1], 0.0))
            for step in range(1, density + 1):
                t = step / density
                point = (1.0 - t) * start + t * end
                vectors.append(mp.Vector3(point[0], point[1], 0.0))
        return vectors
    
    def create_solver():
        if config["lattice"] == "square":
            k_pts = build_k_path_square(K_POINTS_PER_SEGMENT)
            lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
        else:
            k_pts = build_k_path_hex(K_POINTS_PER_SEGMENT)
            lattice = mp.Lattice(
                basis1=mp.Vector3(1, 0, 0),
                basis2=mp.Vector3(0.5, np.sqrt(3)/2, 0),
                size=mp.Vector3(1, 1, 0)
            )
        
        geometry = [mp.Cylinder(
            radius=config["radius"],
            material=mp.Medium(epsilon=config["eps_rod"]),
            height=mp.inf
        )]
        
        return mpb.ModeSolver(
            num_bands=NUM_BANDS,
            k_points=k_pts,
            geometry_lattice=lattice,
            geometry=geometry,
            default_material=mp.Medium(epsilon=config["eps_bg"]),
            resolution=RESOLUTION,
            tolerance=1e-7,
            dimensions=2,
        )
    
    print(f"\n  [{config_name.upper()}] {config['desc']} - {config['polarization'].upper()}")
    print(f"  Runs per iteration: {num_runs}, Iterations: {num_iterations}")
    
    all_times = []
    iteration_stats = []
    
    for iteration in range(num_iterations):
        iteration_times = []
        print(f"    Iteration {iteration + 1}/{num_iterations}: ", end="", flush=True)
        iter_start = time.perf_counter()
        
        for run in range(num_runs):
            solver = create_solver()
            start = time.perf_counter()
            with suppress_output():
                if config["polarization"] == "tm":
                    solver.run_tm()
                else:
                    solver.run_te()
            elapsed = time.perf_counter() - start
            iteration_times.append(elapsed)
        
        iter_total = time.perf_counter() - iter_start
        iter_mean = np.mean(iteration_times) * 1000
        all_times.extend(iteration_times)
        iteration_stats.append({
            "iteration": iteration + 1,
            "times": iteration_times,
            "mean_ms": float(iter_mean),
            "total_s": float(iter_total)
        })
        
        print(f"{iter_total:.1f}s (mean: {iter_mean:.2f}ms/job)")
    
    # Compile results
    times_ms = [t * 1000 for t in all_times]
    results = {
        "benchmark": "mpb_single_core",
        "config": config_name,
        "description": config["desc"],
        "polarization": config["polarization"],
        "lattice": config["lattice"],
        "timestamp": datetime.now().isoformat(),
        "solver": "MPB",
        "core_mode": "single",
        "resolution": RESOLUTION,
        "num_bands": NUM_BANDS,
        "k_points_per_segment": K_POINTS_PER_SEGMENT,
        "total_k_points": K_POINTS_PER_SEGMENT * 3 + 1,
        "runs_per_iteration": num_runs,
        "num_iterations": num_iterations,
        "total_runs": len(all_times),
        "times_ms": times_ms,
        "iteration_stats": iteration_stats,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "median_ms": float(np.median(times_ms)),
    }
    
    # Save results
    output_file = output_dir / f"mpb_{config_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"    Saved: {output_file.name}")
    print(f"    Result: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms/job")
    
    return results


# ============================================================================
# Blaze2D Benchmark Functions
# ============================================================================

BLAZE_CONFIG_TEMPLATES = {
    "config_a_tm": """# Config A TM: Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a
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
max_iter = 200
tol = 1e-4

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 0.999, max = 1.001, step = {step:.10f} }}
""",
    "config_a_te": """# Config A TE: Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a
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
max_iter = 200
tol = 1e-4

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 0.999, max = 1.001, step = {step:.10f} }}
""",
    "config_b_tm": """# Config B TM: Hex lattice, ε=13 bg, air rods, r=0.48a
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
max_iter = 200
tol = 1e-4

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 12.999, max = 13.001, step = {step:.10f} }}
""",
    "config_b_te": """# Config B TE: Hex lattice, ε=13 bg, air rods, r=0.48a
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
max_iter = 200
tol = 1e-4

[output]
mode = "full"
directory = "./benchmark_bulk_output"

[ranges]
eps_bg = {{ min = 12.999, max = 13.001, step = {step:.10f} }}
""",
}


def build_bulk_driver():
    """Build blaze2d-bulk-driver in release mode."""
    print("Building blaze2d-bulk-driver (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "blaze2d-bulk-driver"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
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


def run_blaze2d_benchmark(config_name: str, config: dict, num_runs: int,
                          num_iterations: int, output_dir: Path, 
                          no_build: bool = False) -> dict:
    """Run Blaze2D benchmark for a single configuration."""
    
    if not no_build:
        build_bulk_driver()
    
    binary = get_bulk_driver_binary()
    
    # Generate bulk config with correct step for num_runs jobs
    step = 0.002 / (num_runs - 1) if num_runs > 1 else 0.001
    template = BLAZE_CONFIG_TEMPLATES[config_name]
    config_content = template.format(step=step)
    
    print(f"\n  [{config_name.upper()}] {config['desc']} - {config['polarization'].upper()}")
    print(f"  Jobs per iteration: {num_runs}, Iterations: {num_iterations}")
    
    iteration_times = []
    iteration_stats = []
    
    for iteration in range(num_iterations):
        print(f"    Iteration {iteration + 1}/{num_iterations}: ", end="", flush=True)
        
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)
        
        try:
            cmd = [
                str(binary),
                "--config", str(config_path),
                "--benchmark",  # Real solves, NO file output
                "-j", "1",  # Single core
            ]
            
            start = time.perf_counter()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            elapsed = time.perf_counter() - start
            
            if result.returncode != 0:
                print(f"\nWARNING: bulk driver returned non-zero exit code")
                print(result.stderr[:500] if result.stderr else "No stderr")
            
            iteration_times.append(elapsed)
            mean_per_job = elapsed / num_runs * 1000
            throughput = num_runs / elapsed
            
            iteration_stats.append({
                "iteration": iteration + 1,
                "total_time_s": float(elapsed),
                "mean_ms_per_job": float(mean_per_job),
                "throughput_jobs_s": float(throughput),
            })
            
            print(f"{elapsed:.1f}s ({throughput:.1f} jobs/s, {mean_per_job:.2f}ms/job)")
        finally:
            config_path.unlink(missing_ok=True)
    
    # Compile results
    per_job_times_ms = [t / num_runs * 1000 for t in iteration_times]
    throughputs = [num_runs / t for t in iteration_times]
    
    results = {
        "benchmark": "blaze2d_single_core",
        "config": config_name,
        "description": config["desc"],
        "polarization": config["polarization"],
        "lattice": config["lattice"],
        "timestamp": datetime.now().isoformat(),
        "solver": "Blaze2D",
        "core_mode": "single",
        "num_threads": 1,
        "resolution": RESOLUTION,
        "num_bands": NUM_BANDS,
        "k_points_per_segment": K_POINTS_PER_SEGMENT,
        "total_k_points": K_POINTS_PER_SEGMENT * 3 + 1,
        "jobs_per_iteration": num_runs,
        "num_iterations": num_iterations,
        "total_jobs": num_runs * num_iterations,
        "iteration_times_s": iteration_times,
        "iteration_stats": iteration_stats,
        "mean_ms": float(np.mean(per_job_times_ms)),
        "std_ms": float(np.std(per_job_times_ms)),
        "min_ms": float(np.min(per_job_times_ms)),
        "max_ms": float(np.max(per_job_times_ms)),
        "mean_throughput_jobs_s": float(np.mean(throughputs)),
        "std_throughput_jobs_s": float(np.std(throughputs)),
    }
    
    # Save results
    output_file = output_dir / f"blaze2d_{config_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"    Saved: {output_file.name}")
    print(f"    Result: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms/job")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-Core Benchmark Runner with Separate Output Files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all MPB single-core benchmarks
  python run_single_core_benchmarks.py --solver mpb

  # Run specific config
  python run_single_core_benchmarks.py --solver mpb --config config_a_tm

  # Run all Blaze2D benchmarks
  python run_single_core_benchmarks.py --solver blaze2d

  # Run everything
  python run_single_core_benchmarks.py --solver all

  # Quick test
  python run_single_core_benchmarks.py --solver all --quick
"""
    )
    parser.add_argument("--solver", type=str, required=True,
                        choices=["mpb", "blaze2d", "all"],
                        help="Which solver to benchmark")
    parser.add_argument("--config", type=str, default=None,
                        choices=list(CONFIGS.keys()),
                        help="Run specific config only")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Runs per iteration (default: {DEFAULT_RUNS})")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Number of iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick test mode ({QUICK_RUNS} runs × {QUICK_ITERATIONS} iterations)")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip building Blaze2D (use existing binary)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results/single_core)")
    args = parser.parse_args()
    
    # Set parameters
    if args.quick:
        num_runs = QUICK_RUNS
        num_iterations = QUICK_ITERATIONS
    else:
        num_runs = args.runs
        num_iterations = args.iterations
    
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which configs to run
    configs_to_run = [args.config] if args.config else list(CONFIGS.keys())
    
    print("=" * 70)
    print("Single-Core Benchmark Runner")
    print("=" * 70)
    print(f"Solver(s): {args.solver}")
    print(f"Config(s): {', '.join(configs_to_run)}")
    print(f"Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"Bands: {NUM_BANDS}")
    print(f"K-points: {K_POINTS_PER_SEGMENT * 3 + 1} ({K_POINTS_PER_SEGMENT} per segment)")
    print(f"Runs per iteration: {num_runs}")
    print(f"Iterations: {num_iterations}")
    print(f"Total runs per config: {num_runs * num_iterations}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    results_summary = {}
    
    # Run MPB benchmarks
    if args.solver in ["mpb", "all"]:
        print("\n" + "=" * 70)
        print("MPB SINGLE-CORE BENCHMARKS")
        print("=" * 70)
        
        for config_name in configs_to_run:
            config = CONFIGS[config_name]
            result = run_mpb_benchmark(
                config_name, config, num_runs, num_iterations, output_dir
            )
            if result:
                results_summary[f"mpb_{config_name}"] = {
                    "mean_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                }
    
    # Run Blaze2D benchmarks
    if args.solver in ["blaze2d", "all"]:
        print("\n" + "=" * 70)
        print("BLAZE2D SINGLE-CORE BENCHMARKS")
        print("=" * 70)
        
        # Build once before running all configs
        if not args.no_build:
            build_bulk_driver()
        
        for config_name in configs_to_run:
            config = CONFIGS[config_name]
            result = run_blaze2d_benchmark(
                config_name, config, num_runs, num_iterations, output_dir,
                no_build=True  # Already built above
            )
            if result:
                results_summary[f"blaze2d_{config_name}"] = {
                    "mean_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                }
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for name, stats in sorted(results_summary.items()):
        print(f"  {name}: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms/job")
    
    print(f"\nResults saved to: {output_dir}/")
    print("Files created:")
    for f in sorted(output_dir.glob("*.json")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
