#!/usr/bin/env python3
"""
Direct benchmark comparison: MPB (Python/Scheme) vs Rust solver

Runs the SAME configurations through both solvers and times them.
Configuration: Matching the original 20000-job study parameters.
"""

import time
import sys
import os
from contextlib import contextmanager

# Suppress MPB's verbose output
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

# Import MPB
try:
    import meep as mp
    from meep import mpb
    import numpy as np
except ImportError:
    print("ERROR: This script requires the mpb-reference environment.")
    print("Run: mamba activate mpb-reference")
    sys.exit(1)

# ============================================================================
# Configuration matching your original study
# ============================================================================
RESOLUTION = 32
NUM_BANDS = 10
K_POINTS_PER_SEGMENT = 40  # 3 segments × 40 = 120 k-points

# Test parameters (subset for benchmark)
EPS_BG_VALUES = [2.0, 5.0, 8.0, 11.0, 14.0]  # 5 values (subset of 1.8-14.0)
RADIUS_VALUES = [0.10, 0.20, 0.30, 0.40]      # 4 values (subset of 0.10-0.48)
LATTICE_TYPES = ["square", "hex"]
POLARIZATIONS = ["tm", "te"]

# Total jobs: 5 × 4 × 2 × 2 = 80 jobs

def build_k_path_square(density: int):
    """Γ → X → M → Γ for square lattice"""
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
    return interpolate_path(nodes, density)

def build_k_path_hex(density: int):
    """Γ → M → K → Γ for triangular/hex lattice (60° convention)"""
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

def run_mpb_job(eps_bg, radius, lattice_type, polarization):
    """Run a single MPB job and return elapsed time in seconds."""
    
    # Build k-path
    if lattice_type == "square":
        k_pts = build_k_path_square(K_POINTS_PER_SEGMENT)
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    else:  # hex/triangular
        k_pts = build_k_path_hex(K_POINTS_PER_SEGMENT)
        # Triangular lattice: 60° angle between basis vectors
        lattice = mp.Lattice(
            basis1=mp.Vector3(1, 0, 0),
            basis2=mp.Vector3(0.5, np.sqrt(3)/2, 0),
            size=mp.Vector3(1, 1, 0)
        )
    
    # Air hole geometry
    geometry = [mp.Cylinder(radius=radius, material=mp.Medium(epsilon=1.0), height=mp.inf)]
    
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=eps_bg),
        resolution=RESOLUTION,
        dimensions=2,
    )
    
    start = time.perf_counter()
    with suppress_output():
        if polarization == "tm":
            solver.run_tm()
        else:
            solver.run_te()
    elapsed = time.perf_counter() - start
    
    return elapsed

def main():
    print("=" * 70)
    print("MPB (Python/Scheme) Benchmark")
    print("=" * 70)
    print(f"Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"Bands: {NUM_BANDS}")
    print(f"K-points: {K_POINTS_PER_SEGMENT * 3} (40 per segment × 3 segments)")
    print(f"Epsilon values: {EPS_BG_VALUES}")
    print(f"Radius values: {RADIUS_VALUES}")
    print(f"Lattices: {LATTICE_TYPES}")
    print(f"Polarizations: {POLARIZATIONS}")
    
    total_jobs = len(EPS_BG_VALUES) * len(RADIUS_VALUES) * len(LATTICE_TYPES) * len(POLARIZATIONS)
    print(f"\nTotal jobs: {total_jobs}")
    print("=" * 70)
    
    times = []
    job_count = 0
    
    start_total = time.perf_counter()
    
    for eps_bg in EPS_BG_VALUES:
        for radius in RADIUS_VALUES:
            for lattice in LATTICE_TYPES:
                for pol in POLARIZATIONS:
                    job_count += 1
                    print(f"[{job_count:3d}/{total_jobs}] eps={eps_bg:5.1f}, r={radius:.2f}, {lattice:6s}, {pol.upper():2s}...", end=" ", flush=True)
                    
                    elapsed = run_mpb_job(eps_bg, radius, lattice, pol)
                    times.append(elapsed)
                    
                    print(f"{elapsed:.2f}s")
    
    total_time = time.perf_counter() - start_total
    
    print("=" * 70)
    print(f"RESULTS:")
    print(f"  Total jobs:     {total_jobs}")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Average/job:    {np.mean(times):.2f}s")
    print(f"  Min/Max:        {np.min(times):.2f}s / {np.max(times):.2f}s")
    print(f"  Jobs/second:    {total_jobs / total_time:.2f}")
    print("=" * 70)
    
    # Extrapolate to 1000 jobs
    projected_1000 = (np.mean(times) * 1000)
    print(f"\nProjected time for 1000 jobs: {projected_1000:.1f}s ({projected_1000/60:.1f} minutes)")
    print(f"Projected time for 20000 jobs: {projected_1000 * 20 / 3600:.1f} hours")

if __name__ == "__main__":
    main()
