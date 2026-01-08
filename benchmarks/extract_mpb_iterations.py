#!/usr/bin/env python3
"""
MPB Iteration Count Extractor

This script runs MPB and extracts the iteration count per k-point from the
verbose text output. MPB outputs iteration information like:

    solve_kpoint (0.1,0,0):
    Solving for bands 1 to 4...
        iteration    1: trace = ...
        iteration    2: trace = ...
    Finished solving for bands 1 to 4 after 6 iterations.

We parse this output to extract:
- k-point coordinates
- Number of iterations to convergence
- Time per k-point (if available)
"""

import re
import sys
import json
import argparse
import tempfile
from pathlib import Path
from io import StringIO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import subprocess


@dataclass
class KPointIterationData:
    """Data for a single k-point solve."""
    k_index: int
    k_frac: Tuple[float, float, float]
    iterations: int
    elapsed_seconds: float = 0.0
    bands_range: Tuple[int, int] = (0, 0)


@dataclass
class MPBIterationResult:
    """Complete iteration result from an MPB run."""
    polarization: str
    resolution: int
    num_bands: int
    num_k_points: int
    k_point_data: List[KPointIterationData] = field(default_factory=list)
    total_elapsed: float = 0.0


def parse_mpb_output(output: str) -> List[KPointIterationData]:
    """
    Parse MPB output text and extract iteration counts per k-point.
    
    Args:
        output: The full stdout/stderr from an MPB run
        
    Returns:
        List of KPointIterationData for each k-point solved
    """
    results = []
    
    # Pattern for k-point start: "solve_kpoint (0.1,0,0):"
    kpoint_pattern = re.compile(r'solve_kpoint\s+\(([^)]+)\):')
    
    # Pattern for iteration finish: "Finished solving for bands X to Y after N iterations."
    finish_pattern = re.compile(
        r'Finished solving for bands\s+(\d+)\s+to\s+(\d+)\s+after\s+(\d+)\s+iterations\.'
    )
    
    # Pattern for elapsed time: "elapsed time for k point: 0.123456"
    elapsed_pattern = re.compile(r'elapsed time for k point:\s+([\d.e+-]+)')
    
    # Parse line by line
    current_k = None
    k_index = 0
    
    for line in output.split('\n'):
        # Check for k-point start
        match = kpoint_pattern.search(line)
        if match:
            k_str = match.group(1)
            # Parse "0.1,0,0" or similar
            k_parts = [float(x.strip()) for x in k_str.split(',')]
            while len(k_parts) < 3:
                k_parts.append(0.0)
            current_k = {
                'k_index': k_index,
                'k_frac': tuple(k_parts[:3]),
                'iterations': 0,
                'elapsed': 0.0,
                'bands_range': (0, 0),
            }
            k_index += 1
            continue
        
        # Check for finish line (iteration count)
        if current_k is not None:
            match = finish_pattern.search(line)
            if match:
                band_start = int(match.group(1))
                band_end = int(match.group(2))
                iters = int(match.group(3))
                # Accumulate iterations (there may be multiple solve phases per k-point)
                current_k['iterations'] += iters
                current_k['bands_range'] = (band_start, band_end)
                continue
            
            # Check for elapsed time
            match = elapsed_pattern.search(line)
            if match:
                current_k['elapsed'] = float(match.group(1))
                # This marks end of k-point, save it
                results.append(KPointIterationData(
                    k_index=current_k['k_index'],
                    k_frac=current_k['k_frac'],
                    iterations=current_k['iterations'],
                    elapsed_seconds=current_k['elapsed'],
                    bands_range=current_k['bands_range'],
                ))
                current_k = None
    
    return results


def run_mpb_with_capture(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int = 20,
    epsilon: float = 8.9,
    radius: float = 0.2,
    eps_bg: float = 1.0,
    tolerance: float = 1e-7,
) -> MPBIterationResult:
    """
    Run MPB and capture iteration data.
    
    This function runs MPB in a subprocess to capture its verbose output,
    which is normally not accessible when running from Python.
    
    Args:
        resolution: Grid resolution (NxN)
        num_bands: Number of bands to compute
        polarization: "TM" or "TE"
        k_points_per_segment: K-points per path segment
        epsilon: Dielectric constant of rods
        radius: Rod radius in lattice units
        eps_bg: Background dielectric
        tolerance: Convergence tolerance
        
    Returns:
        MPBIterationResult with per-k-point iteration data
    """
    # Create a Python script that runs MPB
    script = f'''
import meep as mp
from meep import mpb
import numpy as np

def build_k_path(density):
    """Build Gamma-X-M-Gamma path for square lattice."""
    nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
    k_pts = []
    for seg in range(len(nodes) - 1):
        start = np.array(nodes[seg])
        end = np.array(nodes[seg + 1])
        if seg == 0:
            k_pts.append(mp.Vector3(start[0], start[1], 0.0))
        for step in range(1, density + 1):
            t = step / density
            point = (1.0 - t) * start + t * end
            k_pts.append(mp.Vector3(point[0], point[1], 0.0))
    return k_pts

k_pts = build_k_path({k_points_per_segment})
lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
geometry = [mp.Cylinder(
    radius={radius},
    material=mp.Medium(epsilon={epsilon}),
    height=mp.inf
)]

solver = mpb.ModeSolver(
    num_bands={num_bands},
    k_points=k_pts,
    geometry_lattice=lattice,
    geometry=geometry,
    default_material=mp.Medium(epsilon={eps_bg}),
    resolution={resolution},
    tolerance={tolerance},
    dimensions=2,
)

if "{polarization}" == "TM":
    solver.run_tm()
else:
    solver.run_te()
'''
    
    # Write to temp file and run
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = Path(f.name)
    
    try:
        # Run with explicit Python from mpb-reference environment
        mpb_python = "/home/renlephy/.local/share/mamba/envs/mpb-reference/bin/python"
        result = subprocess.run(
            [mpb_python, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        
        # Combine stdout and stderr (MPB prints to both)
        output = result.stdout + result.stderr
        
        # Parse the output
        k_data = parse_mpb_output(output)
        
        # Calculate total k-points
        total_k = 3 * k_points_per_segment + 1  # Gamma-X-M-Gamma path
        
        return MPBIterationResult(
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            num_k_points=total_k,
            k_point_data=k_data,
            total_elapsed=sum(kp.elapsed_seconds for kp in k_data),
        )
    finally:
        script_path.unlink(missing_ok=True)


def run_mpb_inline(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int = 20,
    epsilon: float = 8.9,
    radius: float = 0.2,
    eps_bg: float = 1.0,
    tolerance: float = 1e-7,
) -> MPBIterationResult:
    """
    Run MPB inline and capture iteration data by redirecting output.
    
    This is an alternative approach that runs MPB directly but redirects
    the C-level output to capture iteration counts.
    """
    import os
    import meep as mp
    from meep import mpb
    import numpy as np
    
    def build_k_path(density):
        nodes = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
        k_pts = []
        for seg in range(len(nodes) - 1):
            start = np.array(nodes[seg])
            end = np.array(nodes[seg + 1])
            if seg == 0:
                k_pts.append(mp.Vector3(start[0], start[1], 0.0))
            for step in range(1, density + 1):
                t = step / density
                point = (1.0 - t) * start + t * end
                k_pts.append(mp.Vector3(point[0], point[1], 0.0))
        return k_pts
    
    k_pts = build_k_path(k_points_per_segment)
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    geometry = [mp.Cylinder(
        radius=radius,
        material=mp.Medium(epsilon=epsilon),
        height=mp.inf
    )]
    
    solver = mpb.ModeSolver(
        num_bands=num_bands,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=eps_bg),
        resolution=resolution,
        tolerance=tolerance,
        dimensions=2,
    )
    
    # Capture output using file descriptor redirection
    import tempfile
    
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    stdout_copy = os.dup(stdout_fd)
    stderr_copy = os.dup(stderr_fd)
    
    # Create temp files to capture output
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as stdout_file:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as stderr_file:
            stdout_path = stdout_file.name
            stderr_path = stderr_file.name
    
    try:
        # Redirect stdout and stderr to temp files
        with open(stdout_path, 'w') as stdout_file:
            with open(stderr_path, 'w') as stderr_file:
                os.dup2(stdout_file.fileno(), stdout_fd)
                os.dup2(stderr_file.fileno(), stderr_fd)
                
                try:
                    if polarization == "TM":
                        solver.run_tm()
                    else:
                        solver.run_te()
                finally:
                    # Restore original file descriptors
                    os.dup2(stdout_copy, stdout_fd)
                    os.dup2(stderr_copy, stderr_fd)
        
        # Read captured output
        with open(stdout_path) as f:
            stdout_output = f.read()
        with open(stderr_path) as f:
            stderr_output = f.read()
        
        output = stdout_output + stderr_output
    finally:
        os.close(stdout_copy)
        os.close(stderr_copy)
        Path(stdout_path).unlink(missing_ok=True)
        Path(stderr_path).unlink(missing_ok=True)
    
    # Parse the output
    k_data = parse_mpb_output(output)
    
    return MPBIterationResult(
        polarization=polarization,
        resolution=resolution,
        num_bands=num_bands,
        num_k_points=len(k_pts),
        k_point_data=k_data,
        total_elapsed=sum(kp.elapsed_seconds for kp in k_data),
    )


def result_to_dict(result: MPBIterationResult) -> dict:
    """Convert MPBIterationResult to JSON-serializable dict."""
    return {
        "solver": "MPB",
        "polarization": result.polarization,
        "resolution": result.resolution,
        "num_bands": result.num_bands,
        "num_k_points": result.num_k_points,
        "total_elapsed": result.total_elapsed,
        "k_points": [
            {
                "k_index": kp.k_index,
                "k_frac": list(kp.k_frac),
                "iterations": kp.iterations,
                "elapsed_seconds": kp.elapsed_seconds,
                "bands_range": list(kp.bands_range),
            }
            for kp in result.k_point_data
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Extract MPB iteration counts per k-point")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution")
    parser.add_argument("--num-bands", type=int, default=8, help="Number of bands")
    parser.add_argument("--polarization", type=str, default="TM", choices=["TM", "TE"])
    parser.add_argument("--k-points-per-segment", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8.9, help="Rod dielectric")
    parser.add_argument("--radius", type=float, default=0.2, help="Rod radius")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="Convergence tolerance")
    parser.add_argument("--output", type=str, help="Output JSON file (default: stdout)")
    parser.add_argument("--method", type=str, default="subprocess", 
                        choices=["subprocess", "inline"],
                        help="Method to capture output")
    args = parser.parse_args()
    
    print(f"Running MPB with resolution={args.resolution}, bands={args.num_bands}, "
          f"pol={args.polarization}...", file=sys.stderr)
    
    if args.method == "subprocess":
        result = run_mpb_with_capture(
            resolution=args.resolution,
            num_bands=args.num_bands,
            polarization=args.polarization,
            k_points_per_segment=args.k_points_per_segment,
            epsilon=args.epsilon,
            radius=args.radius,
            tolerance=args.tolerance,
        )
    else:
        result = run_mpb_inline(
            resolution=args.resolution,
            num_bands=args.num_bands,
            polarization=args.polarization,
            k_points_per_segment=args.k_points_per_segment,
            epsilon=args.epsilon,
            radius=args.radius,
            tolerance=args.tolerance,
        )
    
    # Convert to dict
    data = result_to_dict(result)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(data, indent=2))
    
    # Summary
    total_iters = sum(kp.iterations for kp in result.k_point_data)
    avg_iters = total_iters / len(result.k_point_data) if result.k_point_data else 0
    print(f"\nSummary: {len(result.k_point_data)} k-points, "
          f"total {total_iters} iterations, "
          f"avg {avg_iters:.1f} iters/k-point", file=sys.stderr)


if __name__ == "__main__":
    main()
