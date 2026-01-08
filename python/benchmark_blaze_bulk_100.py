#!/usr/bin/env python3
"""Benchmark blaze bulk driver for 1000 band diagram calculations.

This script measures the wall-clock time for 1000 band structure calculations
using the blaze bulk driver with a square lattice configuration
(eps=13, r/a=0.3, res=32, TM polarization).

Runs in batch mode with full output mode but benchmark=true (no file output).

Usage:
    python benchmark_blaze_bulk_100.py
    python benchmark_blaze_bulk_100.py --jobs 50  # Run fewer jobs
    python benchmark_blaze_bulk_100.py --polarization te
    python benchmark_blaze_bulk_100.py --threads 4  # Limit threads
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path


def generate_bulk_config(
    num_jobs: int,
    polarization: str,
) -> str:
    """Generate TOML config for bulk driver benchmark.
    
    Creates a parameter sweep that generates exactly `num_jobs` jobs by
    varying eps_bg slightly (which doesn't significantly affect computation time).
    """
    # Calculate step to get exactly num_jobs
    eps_min = 12.0
    eps_max = 14.0
    eps_step = (eps_max - eps_min) / (num_jobs - 1) if num_jobs > 1 else 1.0
    
    return f"""# Auto-generated benchmark config: {num_jobs} jobs, {polarization.upper()} polarization
# Square lattice: eps_bg≈13, r/a=0.3, res=32

polarization = "{polarization.upper()}"

[bulk]
dry_run = false

[geometry]
eps_bg = 13.0  # Will be swept

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.3
eps_inside = 1.0

[grid]
nx = 32
ny = 32
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = 8

[eigensolver]
n_bands = 8
max_iter = 1000
tol = 1e-4

[output]
mode = "full"
directory = "./benchmark_bulk_output"

# Parameter sweep to generate {num_jobs} jobs
[ranges]
eps_bg = {{ min = {eps_min}, max = {eps_max}, step = {eps_step:.10f} }}
"""


def find_workspace_root() -> Path:
    """Find workspace root by looking for Cargo.toml."""
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "Cargo.toml").exists():
            return candidate
    raise RuntimeError("Could not find workspace root (missing Cargo.toml)")


def build_bulk_driver(workspace: Path, quiet: bool = False) -> Path:
    """Build the bulk driver CLI in release mode."""
    print("Building blaze-bulk (release mode)...")
    
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "blaze2d-bulk-driver"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        raise SystemExit(1)
    
    cli_path = workspace / "target" / "release" / "blaze2d-bulk-driver"
    if not cli_path.exists():
        raise RuntimeError(f"Built binary not found: {cli_path}")
    
    print(f"Build complete: {cli_path}")
    return cli_path


def run_bulk_benchmark(
    cli_path: Path,
    config_path: Path,
    threads: int,
    verbose: bool = False,
) -> tuple[float, str]:
    """Run bulk driver and return (elapsed_seconds, stdout)."""
    
    cmd = [
        str(cli_path),
        "--config", str(config_path),
        "--benchmark",  # Real solves, no file output
        "-j", str(threads),  # Thread count
    ]
    if verbose:
        cmd.append("--verbose")
    
    start = time.perf_counter()
    # Don't capture output - let the progress bar show through
    result = subprocess.run(
        cmd,
        timeout=1800,  # 30 minute timeout for 1000 jobs
    )
    elapsed = time.perf_counter() - start
    
    if result.returncode != 0:
        print(f"Bulk driver failed with exit code {result.returncode}")
        raise SystemExit(1)
    
    return elapsed, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", "-n",
        type=int,
        default=1000,
        help="Number of band diagrams to compute (default: 1000)",
    )
    parser.add_argument(
        "--polarization", "-p",
        choices=["tm", "te"],
        default="tm",
        help="Polarization mode (default: tm)",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=16,
        help="Number of threads (default: 16)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output from bulk driver",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building (use existing binary)",
    )
    args = parser.parse_args()
    
    workspace = find_workspace_root()
    
    # Build or find binary
    if args.skip_build:
        cli_path = workspace / "target" / "release" / "blaze2d-bulk-driver"
        if not cli_path.exists():
            print(f"Binary not found: {cli_path}")
            print("Run without --skip-build to build first.")
            raise SystemExit(1)
    else:
        cli_path = build_bulk_driver(workspace)
    
    # Print configuration
    print()
    print("=" * 60)
    print("BLAZE BULK DRIVER BENCHMARK")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Square lattice: ε_bg≈13, r/a=0.3, res=32")
    print(f"  Polarization: {args.polarization.upper()}")
    print(f"  Bands: 8")
    print(f"  k-path: Γ → X → M → Γ (8 points/segment = 25 k-points)")
    print(f"  Jobs: {args.jobs}")
    print(f"  Threads: {args.threads}")
    print(f"  Mode: batch, benchmark (no output)")
    print("=" * 60)
    print()
    
    # Generate config
    config_content = generate_bulk_config(
        num_jobs=args.jobs,
        polarization=args.polarization,
    )
    
    # Write to temp file and run
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".toml",
        delete=False,
        prefix="benchmark_bulk_",
    ) as f:
        f.write(config_content)
        config_path = Path(f.name)
    
    try:
        print(f"Running bulk driver with {args.jobs} jobs...")
        print()
        
        elapsed, _ = run_bulk_benchmark(cli_path, config_path, args.threads, args.verbose)
        
        # Results
        print()
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  Total jobs:      {args.jobs}")
        print(f"  Total time:      {elapsed:.2f} s")
        print(f"  Mean time/job:   {elapsed / args.jobs * 1000:.2f} ms")
        print(f"  Throughput:      {args.jobs / elapsed:.2f} jobs/s")
        print("=" * 60)
        print()
        print(f"✓ Completed {args.jobs} band diagrams in {elapsed:.2f}s")
        print(f"  Average: {elapsed / args.jobs * 1000:.2f} ms per diagram")
        
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
