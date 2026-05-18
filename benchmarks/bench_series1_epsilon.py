#!/usr/bin/env python3
"""
Benchmark Series 1: Epsilon Sweep on Square Lattice

This benchmark varies the dielectric constant (epsilon) from 2 to 13 in 0.5 steps
for both TM and TE polarizations on a square lattice with fixed radius r=0.2a.

Single-core only benchmark comparing MPB vs Blaze2D.

Output: results/series1_epsilon/
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
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Epsilon range: 2.0 to 13.0 in 0.5 steps
EPSILON_VALUES = np.arange(2.0, 13.5, 0.5)

# Fixed parameters
RADIUS = 0.2
RESOLUTION = 64
NUM_BANDS = 8
K_POINTS_PER_SEGMENT = 20

# Benchmark parameters
NUM_ITERATIONS = 2  # Iterations per epsilon value
NUM_RUNS = 1  # Single run per iteration (we vary epsilon, not repeat same config)

# Single-core environment
SINGLE_CORE_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
}

# ============================================================================
# Blaze2D Configuration Template
# ============================================================================
BLAZE_TEMPLATE = """# Series 1: Square lattice, air bg, ε={eps} rods, r=0.2a - {pol}
polarization = "{pol}"

[bulk]
dry_run = false

[geometry]
eps_bg = 1.0

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {radius}
eps_inside = {eps}

[grid]
nx = {res}
ny = {res}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = {k_per_seg}

[eigensolver]
n_bands = {bands}

[output]
mode = "full"
directory = "./series1_output"

[ranges]
# Single job - no sweep needed
eps_bg = {{ min = 0.999, max = 1.001, step = 0.01 }}
"""


def check_mpb_available():
    """Check if MPB is available."""
    try:
        import meep
        from meep import mpb
        return True
    except ImportError:
        return False


def run_mpb_benchmark(epsilon: float, polarization: str) -> float:
    """Run MPB benchmark for a single epsilon value."""
    import meep as mp
    from meep import mpb
    
    # Build k-path
    k_pts = []
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
    for seg in range(len(nodes) - 1):
        start = np.array(nodes[seg])
        end = np.array(nodes[seg + 1])
        if seg == 0:
            k_pts.append(mp.Vector3(start[0], start[1], 0.0))
        for step in range(1, K_POINTS_PER_SEGMENT + 1):
            t = step / K_POINTS_PER_SEGMENT
            point = (1.0 - t) * start + t * end
            k_pts.append(mp.Vector3(point[0], point[1], 0.0))
    
    # Create solver
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    geometry = [mp.Cylinder(
        radius=RADIUS,
        material=mp.Medium(epsilon=epsilon),
        height=mp.inf
    )]
    
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=1.0),
        resolution=RESOLUTION,
        tolerance=1e-7,
        dimensions=2,
    )
    
    # Suppress output
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    start = time.perf_counter()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        if polarization == "TM":
            solver.run_tm()
        else:
            solver.run_te()
    elapsed = time.perf_counter() - start
    
    return elapsed * 1000  # Return ms


def run_blaze_benchmark(epsilon: float, polarization: str, binary: Path) -> float:
    """Run Blaze2D benchmark for a single epsilon value."""
    config = BLAZE_TEMPLATE.format(
        eps=epsilon,
        pol=polarization,
        radius=RADIUS,
        res=RESOLUTION,
        k_per_seg=K_POINTS_PER_SEGMENT,
        bands=NUM_BANDS,
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config)
        config_path = Path(f.name)
    
    try:
        cmd = [
            str(binary),
            "--config", str(config_path),
            "--benchmark",
            "-j", "1",
        ]
        
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"  WARNING: Blaze2D returned non-zero: {result.stderr[:200]}")
            return float('nan')
        
        return elapsed * 1000  # Return ms
    finally:
        config_path.unlink(missing_ok=True)


def find_blaze_binary() -> Path:
    """Find the Blaze2D bulk driver binary."""
    binary = PROJECT_ROOT / "target" / "release" / "blaze2d-bulk-driver"
    if not binary.exists():
        raise FileNotFoundError(
            f"Blaze2D binary not found: {binary}\n"
            "Run: RUSTFLAGS=\"-C target-cpu=native\" cargo build --release -p blaze2d-bulk-driver"
        )
    return binary


def run_series(output_dir: Path, quick: bool = False):
    """Run the full epsilon sweep benchmark."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    iterations = 1 if quick else NUM_ITERATIONS
    epsilons = EPSILON_VALUES[::4] if quick else EPSILON_VALUES  # Every 4th for quick mode
    
    print("=" * 70)
    print("Benchmark Series 1: Epsilon Sweep")
    print("=" * 70)
    print(f"Epsilon values: {len(epsilons)} ({epsilons[0]:.1f} to {epsilons[-1]:.1f})")
    print(f"Polarizations: TM, TE")
    print(f"Iterations per value: {iterations}")
    print(f"Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"Bands: {NUM_BANDS}")
    print("=" * 70)
    
    # Check solver availability
    mpb_available = check_mpb_available()
    if not mpb_available:
        print("WARNING: MPB not available. Install meep/mpb.")
    else:
        print("MPB: available")
    
    # Find Blaze2D binary
    try:
        blaze_binary = find_blaze_binary()
        blaze_available = True
        print(f"Blaze2D: {blaze_binary}")
    except FileNotFoundError as e:
        print(f"WARNING: Blaze2D not available - {e}")
        blaze_available = False
    
    if not mpb_available and not blaze_available:
        print("ERROR: No solvers available!")
        sys.exit(1)
    
    # Results storage
    results = {
        "series": "series1_epsilon",
        "description": "Epsilon sweep on square lattice",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "radius": RADIUS,
            "resolution": RESOLUTION,
            "num_bands": NUM_BANDS,
            "k_points_per_segment": K_POINTS_PER_SEGMENT,
            "iterations": iterations,
        },
        "epsilon_values": list(epsilons),
        "TM": {"mpb": [], "blaze": [], "epsilon": []},
        "TE": {"mpb": [], "blaze": [], "epsilon": []},
    }
    
    # Run benchmarks
    for pol in ["TM", "TE"]:
        print(f"\n[{pol} Polarization]")
        
        for eps in epsilons:
            mpb_times = []
            blaze_times = []
            
            print(f"  ε = {eps:.1f}: ", end="", flush=True)
            
            for it in range(iterations):
                # MPB
                if mpb_available:
                    t = run_mpb_benchmark(eps, pol)
                    mpb_times.append(t)
                    print("M", end="", flush=True)
                
                # Blaze2D
                if blaze_available:
                    t = run_blaze_benchmark(eps, pol, blaze_binary)
                    blaze_times.append(t)
                    print("B", end="", flush=True)
            
            # Store results
            results[pol]["epsilon"].append(float(eps))
            
            if mpb_times:
                results[pol]["mpb"].append({
                    "mean": float(np.mean(mpb_times)),
                    "std": float(np.std(mpb_times)),
                    "values": mpb_times,
                })
            else:
                results[pol]["mpb"].append(None)
            
            if blaze_times:
                results[pol]["blaze"].append({
                    "mean": float(np.mean(blaze_times)),
                    "std": float(np.std(blaze_times)),
                    "values": blaze_times,
                })
            else:
                results[pol]["blaze"].append(None)
            
            # Print summary
            parts = []
            if mpb_times:
                parts.append(f"MPB={np.mean(mpb_times):.1f}ms")
            if blaze_times:
                parts.append(f"Blaze={np.mean(blaze_times):.1f}ms")
            print(f" → {', '.join(parts)}")
    
    # Save results
    results_file = output_dir / "series1_epsilon_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Series 1: Epsilon Sweep")
    parser.add_argument("--output", type=str, default="results/series1_epsilon",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer epsilon values, 1 iteration)")
    args = parser.parse_args()
    
    # Set single-core environment
    os.environ.update(SINGLE_CORE_ENV)
    
    output_dir = SCRIPT_DIR / args.output
    run_series(output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
