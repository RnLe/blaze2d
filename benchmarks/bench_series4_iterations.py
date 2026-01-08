#!/usr/bin/env python3
"""
Benchmark Series 4: Iteration Count Comparison

This benchmark compares the number of LOBPCG iterations needed per k-point
between MPB and Blaze2D. This is a fundamental measure of eigensolver efficiency
that is independent of implementation details like BLAS optimizations.

The benchmark produces bar plots showing iterations per k-point for both
TM and TE polarizations on a square lattice (Config A: ε=8.9 rods, r=0.2a).

Output: results/series4_iterations/
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

# Import extraction modules
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from extract_mpb_iterations import run_mpb_with_capture, result_to_dict as mpb_to_dict
from extract_blaze_iterations import run_blaze_with_diagnostics, result_to_dict as blaze_to_dict

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = SCRIPT_DIR.parent

# Fixed parameters (Config A: Square lattice with ε=8.9 rods)
EPSILON = 8.9
RADIUS = 0.2
EPS_BG = 1.0

# Default benchmark parameters
DEFAULT_RESOLUTION = 32
DEFAULT_NUM_BANDS = 8
DEFAULT_K_POINTS_PER_SEGMENT = 20
# MPB uses f64 and can achieve 1e-7 easily
# Blaze uses mixed-precision (f32 storage) so 1e-4 is the practical limit
DEFAULT_MPB_TOLERANCE = 1e-7
DEFAULT_BLAZE_TOLERANCE = 1e-4


def run_series(
    output_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    num_bands: int = DEFAULT_NUM_BANDS,
    k_points_per_segment: int = DEFAULT_K_POINTS_PER_SEGMENT,
    mpb_tolerance: float = DEFAULT_MPB_TOLERANCE,
    blaze_tolerance: float = DEFAULT_BLAZE_TOLERANCE,
):
    """Run the iteration count benchmark."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Benchmark Series 4: Iteration Count Comparison")
    print("=" * 70)
    print(f"Resolution: {resolution}×{resolution}")
    print(f"Bands: {num_bands}")
    print(f"K-points per segment: {k_points_per_segment}")
    print(f"Total k-points: {3 * k_points_per_segment + 1}")
    print(f"MPB Tolerance: {mpb_tolerance} (f64)")
    print(f"Blaze Tolerance: {blaze_tolerance} (mixed-precision f32)")
    print(f"Config: Square lattice, ε={EPSILON} rods, r={RADIUS}a")
    print("=" * 70)
    
    # Results storage
    results = {
        "series": "series4_iterations",
        "description": "Iteration count per k-point comparison",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "resolution": resolution,
            "num_bands": num_bands,
            "k_points_per_segment": k_points_per_segment,
            "total_k_points": 3 * k_points_per_segment + 1,
            "epsilon": EPSILON,
            "radius": RADIUS,
            "eps_bg": EPS_BG,
            "mpb_tolerance": mpb_tolerance,
            "blaze_tolerance": blaze_tolerance,
        },
        "TM": {"mpb": None, "blaze": None},
        "TE": {"mpb": None, "blaze": None},
    }
    
    # Run benchmarks for both polarizations
    for pol in ["TM", "TE"]:
        print(f"\n[{pol} Polarization]")
        
        # MPB
        print(f"  Running MPB... ", end="", flush=True)
        try:
            mpb_result = run_mpb_with_capture(
                resolution=resolution,
                num_bands=num_bands,
                polarization=pol,
                k_points_per_segment=k_points_per_segment,
                epsilon=EPSILON,
                radius=RADIUS,
                eps_bg=EPS_BG,
                tolerance=mpb_tolerance,
            )
            results[pol]["mpb"] = mpb_to_dict(mpb_result)
            total_mpb = sum(kp.iterations for kp in mpb_result.k_point_data)
            avg_mpb = total_mpb / len(mpb_result.k_point_data) if mpb_result.k_point_data else 0
            print(f"Done! {len(mpb_result.k_point_data)} k-pts, "
                  f"total {total_mpb} iters, avg {avg_mpb:.1f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results[pol]["mpb"] = None
        
        # Blaze2D
        print(f"  Running Blaze2D... ", end="", flush=True)
        try:
            blaze_result = run_blaze_with_diagnostics(
                resolution=resolution,
                num_bands=num_bands,
                polarization=pol,
                k_points_per_segment=k_points_per_segment,
                epsilon=EPSILON,
                radius=RADIUS,
                eps_bg=EPS_BG,
                tolerance=blaze_tolerance,
            )
            results[pol]["blaze"] = blaze_to_dict(blaze_result)
            total_blaze = sum(kp.iterations for kp in blaze_result.k_point_data)
            avg_blaze = total_blaze / len(blaze_result.k_point_data) if blaze_result.k_point_data else 0
            print(f"Done! {len(blaze_result.k_point_data)} k-pts, "
                  f"total {total_blaze} iters, avg {avg_blaze:.1f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results[pol]["blaze"] = None
    
    # Save results
    results_file = output_dir / "series4_iterations_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for pol in ["TM", "TE"]:
        print(f"\n{pol} Polarization:")
        if results[pol]["mpb"]:
            mpb_iters = [kp["iterations"] for kp in results[pol]["mpb"]["k_points"]]
            print(f"  MPB:     total={sum(mpb_iters):4d}, avg={np.mean(mpb_iters):5.1f}, "
                  f"min={min(mpb_iters):3d}, max={max(mpb_iters):3d}")
        else:
            print(f"  MPB:     (failed)")
        
        if results[pol]["blaze"]:
            blaze_iters = [kp["iterations"] for kp in results[pol]["blaze"]["k_points"]]
            print(f"  Blaze2D: total={sum(blaze_iters):4d}, avg={np.mean(blaze_iters):5.1f}, "
                  f"min={min(blaze_iters):3d}, max={max(blaze_iters):3d}")
        else:
            print(f"  Blaze2D: (failed)")
        
        # Ratio
        if results[pol]["mpb"] and results[pol]["blaze"]:
            mpb_total = sum(kp["iterations"] for kp in results[pol]["mpb"]["k_points"])
            blaze_total = sum(kp["iterations"] for kp in results[pol]["blaze"]["k_points"])
            if blaze_total > 0:
                ratio = mpb_total / blaze_total
                print(f"  Ratio (MPB/Blaze): {ratio:.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Series 4: Iteration Count Comparison")
    parser.add_argument("--output", type=str, default="results/series4_iterations",
                        help="Output directory")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help=f"Grid resolution (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS,
                        help=f"Number of bands (default: {DEFAULT_NUM_BANDS})")
    parser.add_argument("--k-points-per-segment", type=int, default=DEFAULT_K_POINTS_PER_SEGMENT,
                        help=f"K-points per path segment (default: {DEFAULT_K_POINTS_PER_SEGMENT})")
    parser.add_argument("--mpb-tolerance", type=float, default=DEFAULT_MPB_TOLERANCE,
                        help=f"MPB convergence tolerance (default: {DEFAULT_MPB_TOLERANCE})")
    parser.add_argument("--blaze-tolerance", type=float, default=DEFAULT_BLAZE_TOLERANCE,
                        help=f"Blaze convergence tolerance (default: {DEFAULT_BLAZE_TOLERANCE})")
    args = parser.parse_args()
    
    output_dir = SCRIPT_DIR / args.output
    run_series(
        output_dir,
        resolution=args.resolution,
        num_bands=args.num_bands,
        k_points_per_segment=args.k_points_per_segment,
        mpb_tolerance=args.mpb_tolerance,
        blaze_tolerance=args.blaze_tolerance,
    )


if __name__ == "__main__":
    main()
