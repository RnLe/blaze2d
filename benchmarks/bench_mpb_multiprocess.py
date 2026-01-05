#!/usr/bin/env python3
"""
MPB Multiprocessing Benchmark - Fair comparison with Blaze2D bulk-driver

Uses Python multiprocessing to run N independent single-threaded MPB jobs.
This is the optimal way to parallelize MPB for bulk calculations.

Config A TM only, 64×64 resolution, tol=1e-7
"""

import os
# Set single-threaded BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import argparse
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import numpy as np

# ============================================================================
# Configuration
# ============================================================================
RESOLUTION = 64
NUM_BANDS = 8
K_POINTS_PER_SEGMENT = 20
TOLERANCE = 1e-7

NUM_WORKERS = 16
NUM_JOBS = 100
NUM_ITERATIONS = 5


def init_worker():
    """Initialize worker process with single-threaded settings."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_single_mpb_job(job_id: int) -> float:
    """
    Run a single MPB job and return elapsed time in seconds.
    
    Each worker process runs this function independently.
    """
    # Import inside worker to ensure clean state
    import meep as mp_meep
    from meep import mpb
    
    # Build k-path for square lattice: Γ → X → M → Γ
    def build_k_path(density):
        nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
        vectors = []
        for seg in range(len(nodes) - 1):
            start = np.array(nodes[seg])
            end = np.array(nodes[seg + 1])
            if seg == 0:
                vectors.append(mp_meep.Vector3(start[0], start[1], 0.0))
            for step in range(1, density + 1):
                t = step / density
                point = (1.0 - t) * start + t * end
                vectors.append(mp_meep.Vector3(point[0], point[1], 0.0))
        return vectors
    
    k_pts = build_k_path(K_POINTS_PER_SEGMENT)
    lattice = mp_meep.Lattice(size=mp_meep.Vector3(1, 1, 0))
    
    # Config A: Square lattice, air bg, ε=8.9 rods, r=0.2a
    geometry = [mp_meep.Cylinder(
        radius=0.2,
        material=mp_meep.Medium(epsilon=8.9),
        height=mp_meep.inf
    )]
    
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp_meep.Medium(epsilon=1.0),
        resolution=RESOLUTION,
        tolerance=TOLERANCE,
        dimensions=2,
    )
    
    # Suppress output and run
    start = time.perf_counter()
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        solver.run_tm()
    elapsed = time.perf_counter() - start
    
    return elapsed


def run_benchmark_iteration(num_jobs: int, num_workers: int) -> list:
    """Run one iteration of the benchmark with multiprocessing."""
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        job_times = pool.map(run_single_mpb_job, range(num_jobs))
    return job_times


def main():
    parser = argparse.ArgumentParser(
        description="MPB Multiprocessing Benchmark (fair comparison with Blaze2D)"
    )
    parser.add_argument("--jobs", "-j", type=int, default=NUM_JOBS,
                        help=f"Number of jobs per iteration (default: {NUM_JOBS})")
    parser.add_argument("--workers", "-w", type=int, default=NUM_WORKERS,
                        help=f"Number of worker processes (default: {NUM_WORKERS})")
    parser.add_argument("--iterations", "-i", type=int, default=NUM_ITERATIONS,
                        help=f"Number of iterations (default: {NUM_ITERATIONS})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (16 jobs, 2 iterations)")
    args = parser.parse_args()
    
    num_jobs = args.jobs
    num_workers = args.workers
    num_iterations = args.iterations
    
    if args.quick:
        num_jobs = 16
        num_iterations = 2
    
    print("=" * 70)
    print("MPB Multiprocessing Benchmark - Config A TM")
    print("=" * 70)
    print(f"Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"Bands: {NUM_BANDS}")
    print(f"K-points: {K_POINTS_PER_SEGMENT * 3 + 1} (20 per segment)")
    print(f"Tolerance: {TOLERANCE}")
    print(f"Jobs per iteration: {num_jobs}")
    print(f"Worker processes: {num_workers}")
    print(f"Iterations: {num_iterations}")
    print(f"Total jobs: {num_jobs * num_iterations}")
    print("=" * 70)
    
    all_iteration_times = []
    all_job_times = []
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}:", end=" ", flush=True)
        
        iter_start = time.perf_counter()
        job_times = run_benchmark_iteration(num_jobs, num_workers)
        iter_elapsed = time.perf_counter() - iter_start
        
        all_iteration_times.append(iter_elapsed)
        all_job_times.extend(job_times)
        
        throughput = num_jobs / iter_elapsed
        mean_per_job = iter_elapsed / num_jobs * 1000
        
        print(f"{iter_elapsed:.1f}s ({throughput:.1f} jobs/s, {mean_per_job:.2f}ms/job)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Config A TM (Multiprocessing)")
    print("=" * 70)
    
    # Wall-clock time per iteration
    mean_iter_time = np.mean(all_iteration_times)
    std_iter_time = np.std(all_iteration_times)
    
    # Throughput
    throughputs = [num_jobs / t for t in all_iteration_times]
    mean_throughput = np.mean(throughputs)
    std_throughput = np.std(throughputs)
    
    # Effective time per job (wall-clock / jobs)
    effective_ms_per_job = mean_iter_time / num_jobs * 1000
    effective_ms_std = std_iter_time / num_jobs * 1000
    
    # Individual job times (actual solver time)
    mean_job_time = np.mean(all_job_times) * 1000
    std_job_time = np.std(all_job_times) * 1000
    
    print(f"Effective time per job: {effective_ms_per_job:.2f} ± {effective_ms_std:.2f} ms")
    print(f"Throughput: {mean_throughput:.1f} ± {std_throughput:.1f} jobs/s")
    print(f"Mean iteration time: {mean_iter_time:.2f} ± {std_iter_time:.2f} s")
    print(f"Individual solver time: {mean_job_time:.1f} ± {std_job_time:.1f} ms")
    print(f"Total elapsed: {sum(all_iteration_times):.1f}s")
    
    return {
        "effective_ms_per_job": effective_ms_per_job,
        "effective_ms_std": effective_ms_std,
        "throughput": mean_throughput,
        "throughput_std": std_throughput,
    }


if __name__ == "__main__":
    # Use spawn to ensure clean process state
    mp.set_start_method('spawn', force=True)
    main()
