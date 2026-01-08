#!/usr/bin/env python3
"""
Blaze2D Iteration Count Extractor

This script runs Blaze2D with diagnostics enabled and extracts the iteration
count per k-point from the diagnostics JSON output.

The Blaze2D CLI with --record-diagnostics produces a JSON file containing
a ConvergenceStudy with ConvergenceRun entries for each k-point. Each run
has a `final_iteration` field that gives us the iteration count.
"""

import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import sys


@dataclass
class KPointIterationData:
    """Data for a single k-point solve."""
    k_index: int
    k_frac: Tuple[float, float]
    iterations: int
    elapsed_seconds: float = 0.0
    converged: bool = True


@dataclass
class BlazeIterationResult:
    """Complete iteration result from a Blaze2D run."""
    polarization: str
    resolution: int
    num_bands: int
    num_k_points: int
    k_point_data: List[KPointIterationData] = field(default_factory=list)
    total_elapsed: float = 0.0


def parse_blaze_diagnostics(diag_path: Path) -> List[KPointIterationData]:
    """
    Parse Blaze2D diagnostics JSON and extract iteration counts per k-point.
    
    Args:
        diag_path: Path to the diagnostics JSON file
        
    Returns:
        List of KPointIterationData for each k-point solved
    """
    with open(diag_path) as f:
        study = json.load(f)
    
    results = []
    
    for run in study.get("runs", []):
        config = run.get("config", {})
        k_index = config.get("k_index")
        k_point = config.get("k_point")
        
        if k_index is None:
            continue
        
        k_frac = tuple(k_point) if k_point else (0.0, 0.0)
        iterations = run.get("final_iteration", 0)
        elapsed = run.get("total_elapsed_secs", 0.0)
        converged = run.get("converged", True)
        
        results.append(KPointIterationData(
            k_index=k_index,
            k_frac=k_frac,
            iterations=iterations,
            elapsed_seconds=elapsed,
            converged=converged,
        ))
    
    # Sort by k_index
    results.sort(key=lambda x: x.k_index)
    
    return results


def create_blaze_config(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int = 20,
    epsilon: float = 8.9,
    radius: float = 0.2,
    eps_bg: float = 1.0,
    tolerance: float = 1e-4,
) -> str:
    """Create a TOML configuration for Blaze2D."""
    config = f'''# Blaze2D iteration extraction config
polarization = "{polarization}"

[geometry]
eps_bg = {eps_bg}

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {radius}
eps_inside = {epsilon}

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = {k_points_per_segment}

[eigensolver]
n_bands = {num_bands}
tol = {tolerance}
max_iter = 200
record_diagnostics = true

[output]
mode = "full"
'''
    return config


def run_blaze_with_diagnostics(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int = 20,
    epsilon: float = 8.9,
    radius: float = 0.2,
    eps_bg: float = 1.0,
    tolerance: float = 1e-4,
) -> BlazeIterationResult:
    """
    Run Blaze2D with diagnostics and extract iteration data.
    
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
        BlazeIterationResult with per-k-point iteration data
    """
    # Find Blaze2D CLI binary
    project_root = Path(__file__).parent.parent
    blaze_binary = project_root / "target" / "release" / "blaze2d-cli"
    
    if not blaze_binary.exists():
        # Try building it
        print(f"Building Blaze2D CLI...", file=sys.stderr)
        result = subprocess.run(
            ["cargo", "build", "--release", "-p", "blaze2d-cli"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build blaze2d-cli: {result.stderr}")
    
    # Create config file
    config_content = create_blaze_config(
        resolution=resolution,
        num_bands=num_bands,
        polarization=polarization,
        k_points_per_segment=k_points_per_segment,
        epsilon=epsilon,
        radius=radius,
        eps_bg=eps_bg,
        tolerance=tolerance,
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        config_path = Path(f.name)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        diag_path = Path(f.name)
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        # Run Blaze2D with diagnostics
        cmd = [
            str(blaze_binary),
            "--config", str(config_path),
            "--record-diagnostics",
            "--diagnostics-output", str(diag_path),
            "--output", str(output_path),
            "--quiet",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            env={
                **{k: v for k, v in __import__('os').environ.items()},
                "RUST_LOG": "warn",  # Reduce log noise
            },
        )
        
        if result.returncode != 0:
            print(f"Blaze2D error: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"Blaze2D failed: {result.stderr}")
        
        # Parse diagnostics
        if not diag_path.exists():
            raise RuntimeError(f"Diagnostics file not created: {diag_path}")
        
        k_data = parse_blaze_diagnostics(diag_path)
        
        # Calculate total k-points
        total_k = 3 * k_points_per_segment + 1  # Gamma-X-M-Gamma path
        
        return BlazeIterationResult(
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            num_k_points=total_k,
            k_point_data=k_data,
            total_elapsed=sum(kp.elapsed_seconds for kp in k_data),
        )
    finally:
        config_path.unlink(missing_ok=True)
        diag_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def result_to_dict(result: BlazeIterationResult) -> dict:
    """Convert BlazeIterationResult to JSON-serializable dict."""
    return {
        "solver": "Blaze2D",
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
                "converged": kp.converged,
            }
            for kp in result.k_point_data
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Blaze2D iteration counts per k-point")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution")
    parser.add_argument("--num-bands", type=int, default=8, help="Number of bands")
    parser.add_argument("--polarization", type=str, default="TM", choices=["TM", "TE"])
    parser.add_argument("--k-points-per-segment", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8.9, help="Rod dielectric")
    parser.add_argument("--radius", type=float, default=0.2, help="Rod radius")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Convergence tolerance (1e-4 for mixed-precision blaze)")
    parser.add_argument("--output", type=str, help="Output JSON file (default: stdout)")
    args = parser.parse_args()
    
    print(f"Running Blaze2D with resolution={args.resolution}, bands={args.num_bands}, "
          f"pol={args.polarization}...", file=sys.stderr)
    
    result = run_blaze_with_diagnostics(
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
    converged = sum(1 for kp in result.k_point_data if kp.converged)
    print(f"\nSummary: {len(result.k_point_data)} k-points, "
          f"total {total_iters} iterations, "
          f"avg {avg_iters:.1f} iters/k-point, "
          f"{converged}/{len(result.k_point_data)} converged", file=sys.stderr)


if __name__ == "__main__":
    main()
