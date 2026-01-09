#!/usr/bin/env python3
"""
Benchmark Series 7: Multi-Threading Scaling

This benchmark measures how performance scales with thread count for:
1. Blaze2D bulk-driver (Rust rayon threads)
2. MPB with OMP native multi-threading

Two problem sizes:
- Low resolution: 16×16
- High resolution: 128×128

Thread counts: 1, 2, 4, 8, 12, 16
Jobs: 2× thread count (to ensure parallelism is exercised)
Runs: 2 iterations per config (for averaging)

Output: results/series7_scaling/
"""

import subprocess
import time
import os
import sys
import json
import argparse
import tempfile
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import numpy as np

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Thread counts to test
THREAD_COUNTS = [1, 2, 4, 8, 12, 16]

# Problem sizes
RESOLUTIONS = {
    "low": 16,
    "high": 128,
}

# Fixed parameters
NUM_BANDS = 8
K_POINTS_PER_SEGMENT = 15
TOLERANCE_MPB = 1e-7
TOLERANCE_BLAZE = 1e-4

# Benchmark sizing (small!)
JOBS_PER_THREAD = 2  # 2× threads = num jobs
NUM_ITERATIONS = 2   # 2 runs for averaging

MPB_PYTHON = "/home/renlephy/.local/share/mamba/envs/mpb-reference/bin/python"


@dataclass
class ScalingResult:
    """Result for a single (solver, threads, resolution) combination."""
    solver: str
    resolution: int
    threads: int
    jobs: int
    iterations: int
    wall_times: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    
    @property
    def mean_time(self) -> float:
        return float(np.mean(self.wall_times)) if self.wall_times else 0
    
    @property
    def std_time(self) -> float:
        return float(np.std(self.wall_times)) if len(self.wall_times) > 1 else 0
    
    @property
    def mean_throughput(self) -> float:
        return float(np.mean(self.throughputs)) if self.throughputs else 0
    
    @property 
    def std_throughput(self) -> float:
        return float(np.std(self.throughputs)) if len(self.throughputs) > 1 else 0


# ============================================================================
# Blaze2D Bulk Driver
# ============================================================================
def create_blaze_config(resolution: int, num_jobs: int) -> str:
    """Create bulk config for Blaze2D."""
    step = 0.002 / (num_jobs - 1) if num_jobs > 1 else 0.001
    return f'''# Blaze2D Series 7 scaling benchmark
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
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = {K_POINTS_PER_SEGMENT}

[eigensolver]
n_bands = {NUM_BANDS}
tol = {TOLERANCE_BLAZE}

[output]
mode = "full"
directory = "./benchmark_output"

[ranges]
eps_bg = {{ min = 0.999, max = 1.001, step = {step:.10f} }}
'''


def build_blaze_bulk_driver():
    """Build blaze2d-bulk-driver."""
    print("  Building blaze2d-bulk-driver...", end=" ", flush=True)
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "blaze2d-bulk-driver"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build: {result.stderr}")
    print("done")


def run_blaze_benchmark(
    resolution: int,
    threads: int,
    num_jobs: int,
    num_iterations: int,
) -> ScalingResult:
    """Run Blaze2D bulk-driver benchmark."""
    
    result = ScalingResult(
        solver="Blaze2D",
        resolution=resolution,
        threads=threads,
        jobs=num_jobs,
        iterations=num_iterations,
    )
    
    binary = PROJECT_ROOT / "target" / "release" / "blaze2d-bulk-driver"
    config_content = create_blaze_config(resolution, num_jobs)
    
    for iteration in range(num_iterations):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)
        
        try:
            cmd = [
                str(binary),
                "--config", str(config_path),
                "--benchmark",
                "-j", str(threads),
            ]
            
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.perf_counter() - start
            
            if proc.returncode != 0:
                print(f"\n  Warning: non-zero exit: {proc.stderr[:200]}")
            
            result.wall_times.append(elapsed)
            result.throughputs.append(num_jobs / elapsed)
        finally:
            config_path.unlink(missing_ok=True)
    
    return result


# ============================================================================
# MPB with OMP (native multi-threading)
# ============================================================================
def run_mpb_omp_benchmark(
    resolution: int,
    threads: int,
    num_jobs: int,
    num_iterations: int,
) -> ScalingResult:
    """Run MPB with OMP multi-threading (one job at a time, threaded)."""
    
    result = ScalingResult(
        solver="MPB-OMP",
        resolution=resolution,
        threads=threads,
        jobs=num_jobs,
        iterations=num_iterations,
    )
    
    # Create worker script that runs jobs sequentially with OMP threads
    script = f'''
import os
os.environ["OMP_NUM_THREADS"] = "{threads}"
os.environ["OPENBLAS_NUM_THREADS"] = "{threads}"
os.environ["MKL_NUM_THREADS"] = "{threads}"

import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import meep as mp
from meep import mpb
import numpy as np

RESOLUTION = {resolution}
NUM_BANDS = {NUM_BANDS}
K_PER_SEG = {K_POINTS_PER_SEGMENT}
TOLERANCE = {TOLERANCE_MPB}
NUM_JOBS = {num_jobs}

def build_k_path(density):
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
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

k_pts = build_k_path(K_PER_SEG)
lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
geometry = [mp.Cylinder(radius=0.2, material=mp.Medium(epsilon=8.9), height=mp.inf)]

start = time.perf_counter()
for job_id in range(NUM_JOBS):
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=1.0),
        resolution=RESOLUTION,
        tolerance=TOLERANCE,
        dimensions=2,
    )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        solver.run_tm()

elapsed = time.perf_counter() - start
print(f"TIME:{{elapsed}}")
'''
    
    for iteration in range(num_iterations):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = Path(f.name)
        
        try:
            proc = subprocess.run(
                [MPB_PYTHON, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            output = proc.stdout + proc.stderr
            for line in output.split('\n'):
                if line.startswith("TIME:"):
                    elapsed = float(line.split(":")[1])
                    result.wall_times.append(elapsed)
                    result.throughputs.append(num_jobs / elapsed)
                    break
            else:
                print(f"\n  Warning: couldn't parse MPB-OMP output")
        finally:
            script_path.unlink(missing_ok=True)
    
    return result


# ============================================================================
# MPB with Python Multiprocessing (OMP=1 per worker)
# ============================================================================
def run_mpb_multiproc_benchmark(
    resolution: int,
    threads: int,
    num_jobs: int,
    num_iterations: int,
) -> ScalingResult:
    """Run MPB with Python multiprocessing (parallel single-threaded workers)."""
    
    result = ScalingResult(
        solver="MPB-Multiproc",
        resolution=resolution,
        threads=threads,
        jobs=num_jobs,
        iterations=num_iterations,
    )
    
    # Create worker script that uses multiprocessing
    script = f'''
import os
# CRITICAL: Set before any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import multiprocessing as mp_pool
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import meep as mp
from meep import mpb
import numpy as np

RESOLUTION = {resolution}
NUM_BANDS = {NUM_BANDS}
K_PER_SEG = {K_POINTS_PER_SEGMENT}
TOLERANCE = {TOLERANCE_MPB}
NUM_JOBS = {num_jobs}
NUM_WORKERS = {threads}

def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def run_job(job_id):
    import meep as mp
    from meep import mpb
    import numpy as np
    
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
    vectors = []
    for seg in range(len(nodes) - 1):
        start = np.array(nodes[seg])
        end = np.array(nodes[seg + 1])
        if seg == 0:
            vectors.append(mp.Vector3(start[0], start[1], 0.0))
        for step in range(1, K_PER_SEG + 1):
            t = step / K_PER_SEG
            point = (1.0 - t) * start + t * end
            vectors.append(mp.Vector3(point[0], point[1], 0.0))
    
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=vectors,
        geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1, 0)),
        geometry=[mp.Cylinder(radius=0.2, material=mp.Medium(epsilon=8.9), height=mp.inf)],
        default_material=mp.Medium(epsilon=1.0),
        resolution=RESOLUTION,
        tolerance=TOLERANCE,
        dimensions=2,
    )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        solver.run_tm()
    return 1

if __name__ == "__main__":
    start = time.perf_counter()
    with mp_pool.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        pool.map(run_job, range(NUM_JOBS))
    elapsed = time.perf_counter() - start
    print(f"TIME:{{elapsed}}")
'''
    
    for iteration in range(num_iterations):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = Path(f.name)
        
        try:
            proc = subprocess.run(
                [MPB_PYTHON, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            output = proc.stdout + proc.stderr
            for line in output.split('\n'):
                if line.startswith("TIME:"):
                    elapsed = float(line.split(":")[1])
                    result.wall_times.append(elapsed)
                    result.throughputs.append(num_jobs / elapsed)
                    break
            else:
                print(f"\n  Warning: couldn't parse MPB-Multiproc output")
        finally:
            script_path.unlink(missing_ok=True)
    
    return result


# ============================================================================
# Main Benchmark Runner
# ============================================================================
def run_series(
    output_dir: Path,
    resolutions: Optional[List[str]] = None,
    thread_counts: Optional[List[int]] = None,
):
    """Run the full Series 7 scaling benchmark."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if resolutions is None:
        resolutions = list(RESOLUTIONS.keys())
    if thread_counts is None:
        thread_counts = THREAD_COUNTS
    
    print("=" * 70)
    print("Benchmark Series 7: Multi-Threading Scaling")
    print("=" * 70)
    print(f"Resolutions: {[RESOLUTIONS[r] for r in resolutions]}")
    print(f"Thread counts: {thread_counts}")
    print(f"Jobs per run: 2× thread count")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Solvers: Blaze2D, MPB-OMP")
    print("=" * 70)
    
    # Build Blaze
    build_blaze_bulk_driver()
    
    all_results = {
        "series": "series7_scaling",
        "description": "Multi-threading scaling benchmark",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "thread_counts": thread_counts,
            "resolutions": {k: RESOLUTIONS[k] for k in resolutions},
            "jobs_per_thread": JOBS_PER_THREAD,
            "num_iterations": NUM_ITERATIONS,
            "num_bands": NUM_BANDS,
            "k_points_per_segment": K_POINTS_PER_SEGMENT,
        },
        "results": {},
    }
    
    for res_name in resolutions:
        res_val = RESOLUTIONS[res_name]
        print(f"\n{'=' * 70}")
        print(f"Resolution: {res_val}×{res_val} ({res_name})")
        print("=" * 70)
        
        all_results["results"][res_name] = {
            "resolution": res_val,
            "blaze": [],
            "mpb_omp": [],
        }
        
        for threads in thread_counts:
            num_jobs = threads * JOBS_PER_THREAD
            print(f"\n[{threads} threads, {num_jobs} jobs]")
            print("-" * 40)
            
            # Blaze2D
            print(f"  Blaze2D...", end=" ", flush=True)
            blaze_res = run_blaze_benchmark(res_val, threads, num_jobs, NUM_ITERATIONS)
            print(f"{blaze_res.mean_throughput:.1f} ± {blaze_res.std_throughput:.1f} jobs/s")
            all_results["results"][res_name]["blaze"].append({
                "threads": threads,
                "jobs": num_jobs,
                "mean_time": blaze_res.mean_time,
                "std_time": blaze_res.std_time,
                "mean_throughput": blaze_res.mean_throughput,
                "std_throughput": blaze_res.std_throughput,
                "wall_times": blaze_res.wall_times,
            })
            
            # MPB with OMP
            print(f"  MPB-OMP...", end=" ", flush=True)
            omp_res = run_mpb_omp_benchmark(res_val, threads, num_jobs, NUM_ITERATIONS)
            print(f"{omp_res.mean_throughput:.1f} ± {omp_res.std_throughput:.1f} jobs/s")
            all_results["results"][res_name]["mpb_omp"].append({
                "threads": threads,
                "jobs": num_jobs,
                "mean_time": omp_res.mean_time,
                "std_time": omp_res.std_time,
                "mean_throughput": omp_res.mean_throughput,
                "std_throughput": omp_res.std_throughput,
                "wall_times": omp_res.wall_times,
            })
    
    # Save results
    results_file = output_dir / "series7_scaling_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_file}")
    print("=" * 70)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Series 7: Multi-threading Scaling")
    parser.add_argument("--output", type=str, default="results/series7_scaling",
                        help="Output directory")
    parser.add_argument("--low-only", action="store_true",
                        help="Only run low resolution (16)")
    parser.add_argument("--high-only", action="store_true",
                        help="Only run high resolution (128)")
    args = parser.parse_args()
    
    output_dir = SCRIPT_DIR / args.output
    
    resolutions = None
    if args.low_only:
        resolutions = ["low"]
    elif args.high_only:
        resolutions = ["high"]
    
    run_series(output_dir, resolutions=resolutions)


if __name__ == "__main__":
    main()
