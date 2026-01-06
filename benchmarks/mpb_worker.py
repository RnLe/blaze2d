#!/usr/bin/env python3
"""
MPB Worker Script - Single-threaded worker for multiprocessing benchmarks

This script is invoked as a subprocess to ensure threading env vars are set
BEFORE any libraries (numpy, OpenBLAS, MKL) are loaded.

Usage:
    python mpb_worker.py <config_json> <num_jobs>
    
Returns: JSON array of job times (in seconds)
"""
import sys
import os

# CRITICAL: Set threading env vars BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["OMP_PROC_BIND"] = "false"

# Save real stdout/stderr
_real_stdout_fd = os.dup(sys.stdout.fileno())
_real_stderr_fd = os.dup(sys.stderr.fileno())
_real_stdout = os.fdopen(_real_stdout_fd, 'w')
_real_stderr = os.fdopen(_real_stderr_fd, 'w')

# Redirect stdout/stderr to devnull at file descriptor level
# This catches C library output (like MPB's "Elapsed run time")
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, sys.stdout.fileno())
os.dup2(_devnull, sys.stderr.fileno())

# Now safe to import
import json
import time

import numpy as np
import meep as mp
from meep import mpb

# Constants
RESOLUTION = 64
NUM_BANDS = 8
K_POINTS_PER_SEGMENT = 20
TOLERANCE = 1e-7


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


def build_k_path_square(density):
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
    return interpolate_path(nodes, density)


def build_k_path_hex(density):
    nodes = [(0.0, 0.0), (0.5, 0.0), (1/3, 1/3), (0.0, 0.0)]
    return interpolate_path(nodes, density)


def create_solver(config):
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
        tolerance=TOLERANCE,
        dimensions=2,
    )


def run_job(config):
    """Run a single MPB job and return elapsed time."""
    solver = create_solver(config)
    start = time.perf_counter()
    # Output already suppressed globally
    if config["polarization"] == "tm":
        solver.run_tm()
    else:
        solver.run_te()
    return time.perf_counter() - start


def main():
    if len(sys.argv) != 3:
        _real_stderr.write("Usage: python mpb_worker.py <config_json> <num_jobs>\n")
        sys.exit(1)
    
    config = json.loads(sys.argv[1])
    num_jobs = int(sys.argv[2])
    
    times = []
    for _ in range(num_jobs):
        elapsed = run_job(config)
        times.append(elapsed)
    
    # Output JSON array of times to the real stdout
    _real_stdout.write(json.dumps(times))
    _real_stdout.write('\n')
    _real_stdout.flush()


if __name__ == "__main__":
    main()
