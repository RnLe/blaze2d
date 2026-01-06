#!/usr/bin/env python3
"""
MPB Speed Benchmark - Single-Core and Multi-Core modes

Configurations (Joannopoulos 1997):
- Config A: Square lattice, air bg (ε=1), ε=8.9 rods, r=0.2a
- Config B: Hex lattice, ε=13 bg, air rods (ε=1), r=0.48a

Parameters: 12 bands, 20 k-points per segment, 32×32 resolution

Benchmark modes:
- Single-core: All threading env vars set to 1 BEFORE imports
- Multi-core: Use all available cores (default MPB behavior)
"""

import sys
import os
import argparse

# =============================================================================
# CRITICAL: Parse args and set threading env vars BEFORE importing numpy/meep!
# =============================================================================
def _parse_cores_early():
    """Parse --cores argument before importing libraries."""
    for i, arg in enumerate(sys.argv):
        if arg == "--cores" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return None
        if arg.startswith("--cores="):
            try:
                return int(arg.split("=")[1])
            except ValueError:
                return None
    return None

_cores = _parse_cores_early()
if _cores == 1:
    # Set ALL threading environment variables to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["GOTO_NUM_THREADS"] = "1"
    os.environ["OMP_PROC_BIND"] = "false"
    print("Single-core mode: All threading env vars set to 1")

# Now safe to import libraries
import time
import json
from contextlib import contextmanager
from datetime import datetime

@contextmanager
def suppress_output():
    """Suppress MPB's verbose output."""
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

try:
    import meep as mp
    from meep import mpb
    import numpy as np
except ImportError:
    print("ERROR: This script requires the mpb-reference environment.")
    print("Run: mamba activate mpb-reference")
    sys.exit(1)

# ============================================================================
# Benchmark Configuration
# ============================================================================
RESOLUTION = 64
NUM_BANDS = 8
K_POINTS_PER_SEGMENT = 20

# Different run counts for single vs multi-core
SINGLE_CORE_RUNS = 10    # Jobs per iteration for single-core
MULTI_CORE_RUNS = 20    # Jobs per iteration for multi-core
NUM_ITERATIONS = 5

def build_k_path_square(density: int):
    """Γ → X → M → Γ for square lattice"""
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
    return interpolate_path(nodes, density)

def build_k_path_hex(density: int):
    """Γ → M → K → Γ for triangular/hex lattice"""
    nodes = [(0.0, 0.0), (0.5, 0.0), (1/3, 1/3), (0.0, 0.0)]
    return interpolate_path(nodes, density)

def interpolate_path(nodes, density):
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

def create_solver_config_a(polarization: str):
    """
    Config A: Square lattice, air background, ε=8.9 rods, r=0.2a
    """
    k_pts = build_k_path_square(K_POINTS_PER_SEGMENT)
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    
    # Dielectric rod in air
    geometry = [mp.Cylinder(
        radius=0.2,
        material=mp.Medium(epsilon=8.9),
        height=mp.inf
    )]
    
    return mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=1.0),
        resolution=RESOLUTION,
        tolerance=1e-7,
        dimensions=2,
    ), polarization

def create_solver_config_b(polarization: str):
    """
    Config B: Hex lattice, ε=13 background, air rods, r=0.48a
    """
    k_pts = build_k_path_hex(K_POINTS_PER_SEGMENT)
    lattice = mp.Lattice(
        basis1=mp.Vector3(1, 0, 0),
        basis2=mp.Vector3(0.5, np.sqrt(3)/2, 0),
        size=mp.Vector3(1, 1, 0)
    )
    
    # Air hole in dielectric
    geometry = [mp.Cylinder(
        radius=0.48,
        material=mp.Medium(epsilon=1.0),
        height=mp.inf
    )]
    
    return mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=13.0),
        resolution=RESOLUTION,
        tolerance=1e-7,
        dimensions=2,
    ), polarization

def run_single_job(solver, polarization: str) -> float:
    """Run a single MPB job and return elapsed time."""
    start = time.perf_counter()
    with suppress_output():
        if polarization == "tm":
            solver.run_tm()
        else:
            solver.run_te()
    return time.perf_counter() - start

def run_benchmark_config(config_name: str, create_solver_func, polarization: str,
                         num_runs: int, num_iterations: int) -> dict:
    """Run benchmark for a specific config and polarization."""
    
    results = {
        "config": config_name,
        "polarization": polarization,
        "runs_per_iteration": num_runs,
        "num_iterations": num_iterations,
        "times": [],
    }
    
    all_times = []
    
    for iteration in range(num_iterations):
        iteration_times = []
        
        print(f"    Iteration {iteration + 1}/{num_iterations}: ", end="", flush=True)
        iter_start = time.perf_counter()
        
        for run in range(num_runs):
            solver, _ = create_solver_func(polarization)
            elapsed = run_single_job(solver, polarization)
            iteration_times.append(elapsed)
            
            if (run + 1) % 100 == 0:
                print(".", end="", flush=True)
        
        iter_total = time.perf_counter() - iter_start
        iter_mean = np.mean(iteration_times)
        all_times.extend(iteration_times)
        
        print(f" {iter_total:.1f}s (mean: {iter_mean*1000:.2f}ms/job)")
    
    results["times"] = all_times
    results["mean_ms"] = float(np.mean(all_times) * 1000)
    results["std_ms"] = float(np.std(all_times) * 1000)
    results["min_ms"] = float(np.min(all_times) * 1000)
    results["max_ms"] = float(np.max(all_times) * 1000)
    results["total_runs"] = len(all_times)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="MPB Speed Benchmark")
    parser.add_argument("--runs", type=int, default=None,
                        help="Runs per iteration (default: 10 for single-core, 20 for multi)")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS,
                        help=f"Number of iterations (default: {NUM_ITERATIONS})")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--cores", type=int, default=None,
                        help="Number of cores (1 for single-core, None for all)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Explicit filename tag (overrides 'single'/'multi')")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5 runs × 2 iterations)")
    args = parser.parse_args()
    
    # Determine core mode (env vars already set at top of file before imports)
    if args.cores == 1:
        core_mode = "single"
        cores_desc = "1 (single-core, env vars set before import)"
        default_runs = SINGLE_CORE_RUNS
    else:
        core_mode = "multi"
        num_cores = args.cores if args.cores else os.cpu_count()
        cores_desc = f"{num_cores} (multi-core)"
        default_runs = MULTI_CORE_RUNS
    
    # Override filename tag if provided
    file_tag = args.tag if args.tag else core_mode
    
    # Set run count: explicit arg > quick mode > mode-based default
    if args.quick:
        num_runs = 5
        args.iterations = 2
    elif args.runs is not None:
        num_runs = args.runs
    else:
        num_runs = default_runs
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 70)
    print(f"MPB Speed Benchmark - {core_mode.upper()}-CORE MODE")
    print("=" * 70)
    print(f"Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"Bands: {NUM_BANDS}")
    print(f"K-points: {K_POINTS_PER_SEGMENT * 3 + 1} ({K_POINTS_PER_SEGMENT} per segment)")
    print(f"Runs per iteration: {num_runs}")
    print(f"Iterations: {args.iterations}")
    print(f"Total runs per config: {num_runs * args.iterations}")
    print(f"Cores: {cores_desc}")
    print("=" * 70)
    
    all_results = {
        "benchmark": f"mpb_speed_{core_mode}",
        "timestamp": datetime.now().isoformat(),
        "solver": "MPB",
        "core_mode": core_mode,
        "resolution": RESOLUTION,
        "num_bands": NUM_BANDS,
        "k_points_per_segment": K_POINTS_PER_SEGMENT,
        "configs": {}
    }
    
    # All configs and polarizations
    configs = [
        ("config_a", "Square lattice, air bg, ε=8.9 rods, r=0.2a", create_solver_config_a),
        ("config_b", "Hex lattice, ε=13 bg, air rods, r=0.48a", create_solver_config_b),
    ]
    polarizations = ["tm", "te"]
    
    for config_key, config_desc, create_func in configs:
        print(f"\n[{config_key.upper()}] {config_desc}")
        all_results["configs"][config_key] = {"description": config_desc, "polarizations": {}}
        
        for pol in polarizations:
            print(f"\n  Polarization: {pol.upper()}")
            results = run_benchmark_config(
                config_key, create_func, pol, num_runs, args.iterations
            )
            all_results["configs"][config_key]["polarizations"][pol] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for config_key, config_data in all_results["configs"].items():
        print(f"\n{config_data['description']}:")
        for pol, pol_data in config_data["polarizations"].items():
            print(f"  {pol.upper()}: {pol_data['mean_ms']:.2f} ± {pol_data['std_ms']:.2f} ms/job")
    
    # Save results
    output_file = os.path.join(args.output, f"mpb_speed_{file_tag}_results.json")
    
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # CSV summary
    csv_file = os.path.join(args.output, f"mpb_speed_{file_tag}_summary.csv")
    with open(csv_file, 'w') as f:
        f.write("config,polarization,mean_ms,std_ms,min_ms,max_ms,total_runs,core_mode\n")
        for config_key, config_data in all_results["configs"].items():
            for pol, pol_data in config_data["polarizations"].items():
                f.write(f"{config_key},{pol},{pol_data['mean_ms']:.4f},"
                        f"{pol_data['std_ms']:.4f},{pol_data['min_ms']:.4f},"
                        f"{pol_data['max_ms']:.4f},{pol_data['total_runs']},{core_mode}\n")
    
    print(f"Summary CSV saved to: {csv_file}")

if __name__ == "__main__":
    main()
