#!/usr/bin/env python3
"""Benchmark MPB solver for 100 band diagram calculations.

This script measures the wall-clock time for 100 MPB band structure calculations
using a square lattice configuration (eps=13, r/a=0.3, res=32, TM polarization).

No output files are generated - this is a pure performance benchmark.
Uses rich progress bar for visual feedback.

Usage:
    python benchmark_mpb_100.py
    python benchmark_mpb_100.py --jobs 50  # Run fewer jobs
    python benchmark_mpb_100.py --polarization te  # Run TE instead
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from contextlib import contextmanager


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _load_runtime_modules():
    """Load meep/mpb modules."""
    try:
        np_mod = importlib.import_module("numpy")
        mp_mod = importlib.import_module("meep")
        mpb_mod = importlib.import_module("meep.mpb")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires the mpb-reference environment (pymeep/mpb).\n"
            "Activate with: conda activate mpb-reference"
        ) from exc
    return np_mod, mp_mod, mpb_mod


def _load_rich():
    """Load rich for progress bar."""
    try:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            MofNCompleteColumn,
        )
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.live import Live
        return Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, Console, Table, Panel, Live
    except ModuleNotFoundError:
        raise SystemExit(
            "This script requires the 'rich' library.\n"
            "Install with: pip install rich"
        )


# Configuration matching square_eps13_r0p3_tm_res24.toml but with res=32
RESOLUTION = 32
NUM_BANDS = 8
K_DENSITY = 8  # Points per k-path segment
EPS_BG = 13.0
EPS_HOLE = 1.0
RADIUS = 0.3


def build_k_path(np, mp, density: int):
    """Build Γ → X → M → Γ k-path for square lattice."""
    nodes = [
        (0.0, 0.0),   # Γ
        (0.5, 0.0),   # X
        (0.5, 0.5),   # M
        (0.0, 0.0),   # Γ
    ]
    
    vectors = []
    for seg in range(len(nodes) - 1):
        start = np.array(nodes[seg], dtype=float)
        end = np.array(nodes[seg + 1], dtype=float)
        
        if seg == 0:
            vectors.append(mp.Vector3(start[0], start[1], 0.0))
        
        for step in range(1, density + 1):
            t = step / density
            point = (1.0 - t) * start + t * end
            vectors.append(mp.Vector3(point[0], point[1], 0.0))
    
    return vectors


def run_single_band_calculation(np, mp, mpb, polarization: str) -> float:
    """Run a single MPB band calculation and return elapsed time in seconds."""
    k_pts = build_k_path(np, mp, K_DENSITY)
    geometry = [mp.Cylinder(radius=RADIUS, material=mp.Medium(epsilon=EPS_HOLE), height=mp.inf)]
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=EPS_BG),
        resolution=RESOLUTION,
        dimensions=2,
    )
    
    start = time.perf_counter()
    # Redirect at file descriptor level to handle MPI output
    with suppress_output():
        if polarization.lower() == "tm":
            solver.run_tm()
        else:
            solver.run_te()
        # Force flush any MPI buffers
        sys.stdout.flush()
        sys.stderr.flush()
    elapsed = time.perf_counter() - start
    
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", "-n",
        type=int,
        default=100,
        help="Number of band diagrams to compute (default: 100)",
    )
    parser.add_argument(
        "--polarization", "-p",
        choices=["tm", "te"],
        default="tm",
        help="Polarization mode (default: tm)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple text progress (no fancy progress bar, better MPI compatibility)",
    )
    args = parser.parse_args()
    
    # Load modules
    np, mp, mpb = _load_runtime_modules()
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, Console, Table, Panel, Live = _load_rich()
    
    console = Console(force_terminal=True)
    
    # Print header
    console.print(Panel.fit(
        f"[bold blue]MPB Benchmark[/bold blue]\n"
        f"Square lattice: ε_bg={EPS_BG}, r/a={RADIUS}, res={RESOLUTION}\n"
        f"Polarization: [yellow]{args.polarization.upper()}[/yellow], Bands: {NUM_BANDS}\n"
        f"k-path: Γ → X → M → Γ ({K_DENSITY} points/segment = {K_DENSITY * 3 + 1} k-points)\n"
        f"Jobs: [green]{args.jobs}[/green]",
        title="Configuration",
        border_style="blue",
    ))
    
    # Warmup run
    console.print("\n[dim]Warmup run...[/dim]")
    _ = run_single_band_calculation(np, mp, mpb, args.polarization)
    console.print("[dim]Warmup complete.[/dim]\n")
    
    # Benchmark runs with progress bar
    times = []
    start_total = time.perf_counter()
    
    if args.simple:
        # Simple text-based progress for MPI compatibility
        console.print(f"[cyan]Computing {args.polarization.upper()} bands...[/cyan]")
        for i in range(args.jobs):
            elapsed = run_single_band_calculation(np, mp, mpb, args.polarization)
            times.append(elapsed)
            # Print progress every 10 jobs or on last job
            if (i + 1) % 10 == 0 or i == args.jobs - 1:
                pct = (i + 1) / args.jobs * 100
                elapsed_total = time.perf_counter() - start_total
                rate = (i + 1) / elapsed_total
                eta = (args.jobs - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:3d}/{args.jobs}] {pct:5.1f}% | {elapsed_total:.1f}s elapsed | ~{eta:.1f}s remaining", flush=True)
    else:
        # Rich progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
        )
        
        with progress:
            task = progress.add_task(
                f"[cyan]Computing {args.polarization.upper()} bands...",
                total=args.jobs,
            )
            
            for i in range(args.jobs):
                elapsed = run_single_band_calculation(np, mp, mpb, args.polarization)
                times.append(elapsed)
                progress.update(task, advance=1, refresh=True)
    
    # Statistics
    total_time = sum(times)
    mean_time = total_time / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Results table
    table = Table(title="Benchmark Results", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Total jobs", f"{args.jobs}")
    table.add_row("Total time", f"{total_time:.2f} s")
    table.add_row("Mean time/job", f"{mean_time * 1000:.2f} ms")
    table.add_row("Std dev", f"{std_time * 1000:.2f} ms")
    table.add_row("Min time", f"{min_time * 1000:.2f} ms")
    table.add_row("Max time", f"{max_time * 1000:.2f} ms")
    table.add_row("Throughput", f"{args.jobs / total_time:.2f} jobs/s")
    
    console.print()
    console.print(table)
    
    # Summary
    console.print(Panel.fit(
        f"[bold green]✓ Completed {args.jobs} band diagrams in {total_time:.2f}s[/bold green]\n"
        f"Average: [yellow]{mean_time * 1000:.2f} ms[/yellow] per diagram",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
