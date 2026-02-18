#!/usr/bin/env python3
"""Benchmark MPB solver and combine with Rust criterion results.

This script measures the wall-clock time for MPB band structure calculations
using the same configuration as the Rust resolution_scaling benchmark:
- Square lattice with eps=13 background and r/a=0.3 air holes
- Resolutions: 24, 32, 48, 64, 128
- Polarizations: TM and TE
- 8 bands
- k-path: Γ → X → M → Γ with 4 points per segment

It also reads criterion benchmark results from the Rust backend if available.

Output: CSV file with columns: resolution, polarization, source, time_ms
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, List, Optional


def _load_runtime_modules():
    try:
        np_mod = importlib.import_module("numpy")
        mp_mod = importlib.import_module("meep")
        mpb_mod = importlib.import_module("meep.mpb")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires the mpb-reference environment (pymeep/mpb)."
        ) from exc
    return np_mod, mp_mod, mpb_mod


def _load_plotting_modules():
    try:
        plt_mod = importlib.import_module("matplotlib.pyplot")
        return plt_mod
    except ModuleNotFoundError:
        return None


np, mp, mpb = _load_runtime_modules()
plt = _load_plotting_modules()


# Configuration matching Rust benchmark
RESOLUTIONS = [24, 32, 48, 64, 128]
POLARIZATIONS = ["tm", "te"]
NUM_BANDS = 8
K_DENSITY = 4  # Points per k-path segment (Γ→X→M→Γ = 3 segments × 4 = 12 points + 1)
EPS_BG = 13.0
EPS_HOLE = 1.0
RADIUS = 0.3


def build_k_path(density: int) -> List[Any]:
    """Build Γ → X → M → Γ k-path for square lattice."""
    nodes = [
        (0.0, 0.0),   # Γ
        (0.5, 0.0),   # X
        (0.5, 0.5),   # M
        (0.0, 0.0),   # Γ
    ]
    
    vectors: List[Any] = []
    prev = None
    
    for seg in range(len(nodes) - 1):
        start = np.array(nodes[seg], dtype=float)
        end = np.array(nodes[seg + 1], dtype=float)
        
        # Include start point only for first segment
        if seg == 0:
            vectors.append(mp.Vector3(start[0], start[1], 0.0))
        
        # Add interpolated points
        for step in range(1, density + 1):
            t = step / density
            point = (1.0 - t) * start + t * end
            vectors.append(mp.Vector3(point[0], point[1], 0.0))
    
    return vectors


def build_geometry() -> List[Any]:
    """Build geometry with air hole (r=0.3) in eps=13 background."""
    return [mp.Cylinder(radius=RADIUS, material=mp.Medium(epsilon=EPS_HOLE), height=mp.inf)]


def run_benchmark(resolution: int, polarization: str) -> float:
    """Run a single MPB calculation and return elapsed time in milliseconds."""
    k_pts = build_k_path(K_DENSITY)
    geometry = build_geometry()
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    
    solver = mpb.ModeSolver(
        num_bands=NUM_BANDS,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=EPS_BG),
        resolution=resolution,
        dimensions=2,
    )
    
    # Time the actual solve
    start = time.perf_counter()
    
    if polarization == "tm":
        solver.run_tm()
    else:
        solver.run_te()
    
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000.0
    return elapsed_ms


def read_criterion_results(criterion_dir: Path, resolutions: List[int]) -> List[dict]:
    """Read criterion benchmark results from the target directory.
    
    Criterion stores results in:
      target/criterion/<group>/<benchmark_id>/<parameter>/new/estimates.json
    
    For our benchmark:
      target/criterion/resolution_scaling/CPU_TM/24/new/estimates.json
      target/criterion/resolution_scaling/GPU_TE/128/new/estimates.json
    
    The estimates.json contains timing in nanoseconds.
    """
    results = []
    
    # Check if criterion directory exists
    base_dir = criterion_dir / "resolution_scaling"
    if not base_dir.exists():
        return results
    
    # Map criterion benchmark names to our source names
    source_map = {
        "CPU_TM": ("mpb2d CPU", "TM"),
        "CPU_TE": ("mpb2d CPU", "TE"),
        "GPU_TM": ("mpb2d GPU", "TM"),
        "GPU_TE": ("mpb2d GPU", "TE"),
    }
    
    for bench_name, (source, polarization) in source_map.items():
        bench_dir = base_dir / bench_name
        if not bench_dir.exists():
            continue
        
        for resolution in resolutions:
            estimates_file = bench_dir / str(resolution) / "new" / "estimates.json"
            if not estimates_file.exists():
                # Try "base" if "new" doesn't exist
                estimates_file = bench_dir / str(resolution) / "base" / "estimates.json"
            
            if estimates_file.exists():
                try:
                    with open(estimates_file) as f:
                        data = json.load(f)
                    
                    # Get mean time in nanoseconds, convert to milliseconds
                    mean_ns = data["mean"]["point_estimate"]
                    time_ms = mean_ns / 1_000_000.0
                    
                    results.append({
                        "resolution": resolution,
                        "polarization": polarization,
                        "source": source,
                        "time_ms": time_ms,
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    warnings.warn(f"Failed to parse {estimates_file}: {e}")
    
    return results


def create_benchmark_plot(results: List[dict], output_path: Path) -> None:
    """Create a benchmark comparison plot with two subplots.
    
    Left: Linear y-scale for absolute comparison
    Right: Log y-scale for relative scaling behavior
    
    Colors:
    - mpb: Blues (dark blue TM, light blue TE)
    - mpb2d CPU: Greens (dark green TM, light green TE)  
    - mpb2d GPU: Oranges (dark orange TM, light orange TE)
    """
    if plt is None:
        warnings.warn("matplotlib not available, skipping plot generation")
        return
    
    # Color scheme: (TM color, TE color) for each source
    color_scheme = {
        "mpb": ("#1f4e79", "#5b9bd5"),           # Dark blue, light blue
        "mpb2d CPU": ("#2e7d32", "#81c784"),     # Dark green, light green
        "mpb2d GPU": ("#e65100", "#ffb74d"),     # Dark orange, light orange
    }
    
    # Marker scheme
    marker_scheme = {
        "mpb": "s",          # Square
        "mpb2d CPU": "o",    # Circle
        "mpb2d GPU": "^",    # Triangle
    }
    
    # Line style: solid for TM, dashed for TE
    linestyle_scheme = {
        "TM": "-",
        "TE": "--",
    }
    
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group results by (source, polarization)
    from collections import defaultdict
    grouped = defaultdict(list)
    all_resolutions = set()
    for r in results:
        key = (r["source"], r["polarization"])
        grouped[key].append((r["resolution"], r["time_ms"]))
        all_resolutions.add(r["resolution"])
    
    # Sort resolutions for x-ticks
    sorted_resolutions = sorted(all_resolutions)
    min_res = min(sorted_resolutions) if sorted_resolutions else 24
    
    # Plot each group on both axes
    for (source, pol), data in sorted(grouped.items()):
        data.sort(key=lambda x: x[0])  # Sort by resolution
        resolutions = [d[0] for d in data]
        times_ms = [d[1] for d in data]
        times_s = [d[1] / 1000.0 for d in data]  # Convert to seconds for linear plot
        
        # Get colors based on source and polarization
        tm_color, te_color = color_scheme.get(source, ("#888888", "#cccccc"))
        color = tm_color if pol == "TM" else te_color
        marker = marker_scheme.get(source, "o")
        linestyle = linestyle_scheme.get(pol, "-")
        
        label = f"{source} {pol}"
        
        # Plot on linear axis (in seconds)
        ax_lin.plot(resolutions, times_s, 
                    color=color, 
                    marker=marker, 
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=8,
                    label=label)
        
        # Plot on log axis (in milliseconds)
        ax_log.plot(resolutions, times_ms, 
                    color=color, 
                    marker=marker, 
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=8,
                    label=label)
    
    # Helper function to add axis break markers
    def add_axis_break(ax, break_pos, y_max):
        """Add zig-zag break markers on x-axis."""
        # Break marker parameters
        d = 0.015  # Size of diagonal lines
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
        
        # Calculate x position in axes coordinates
        xlim = ax.get_xlim()
        x_axes = (break_pos - xlim[0]) / (xlim[1] - xlim[0])
        
        # Draw the break marks (two small diagonal lines)
        ax.plot((x_axes - d, x_axes + d), (-d, +d), **kwargs)
        ax.plot((x_axes - d, x_axes + d), (-d - 0.01, +d - 0.01), **kwargs)
    
    # Configure linear plot (left) - in seconds
    ax_lin.set_xlabel("Resolution", fontsize=12)
    ax_lin.set_ylabel("Time (s)", fontsize=12)
    ax_lin.set_title("Linear Scale", fontsize=14)
    # Start x-axis just before the first resolution with a small gap for the break
    x_start = min_res - 4
    ax_lin.set_xlim(left=x_start)
    ax_lin.set_ylim(bottom=0)
    ax_lin.set_xticks(sorted_resolutions)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc="upper left", fontsize=9)
    # Add break marker
    add_axis_break(ax_lin, x_start + 2, ax_lin.get_ylim()[1])
    
    # Configure log plot (right) - in milliseconds
    ax_log.set_xlabel("Resolution", fontsize=12)
    ax_log.set_ylabel("Time (ms)", fontsize=12)
    ax_log.set_title("Log Scale", fontsize=14)
    ax_log.set_yscale("log")
    ax_log.set_xlim(left=x_start)
    ax_log.set_xticks(sorted_resolutions)
    ax_log.grid(True, alpha=0.3, which='both')
    ax_log.legend(loc="upper left", fontsize=9)
    # Add 1 second reference line (1000 ms)
    ax_log.axhline(y=1000, color='#404040', linewidth=2.5, linestyle='-', zorder=1)
    ax_log.text(sorted_resolutions[-1] + 2, 1000, '1 s', fontsize=10, va='center', color='#404040', fontweight='bold')
    # Add break marker
    add_axis_break(ax_log, x_start + 2, ax_log.get_ylim()[1])
    
    fig.suptitle("MPB Benchmark: Resolution Scaling Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/mpb_benchmark.csv"),
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=RESOLUTIONS,
        help="Resolutions to benchmark (default: 24 32 48 64 128).",
    )
    parser.add_argument(
        "--polarizations",
        nargs="+",
        choices=["tm", "te"],
        default=POLARIZATIONS,
        help="Polarizations to benchmark (default: tm te).",
    )
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=None,
        help="Path to criterion results directory (default: auto-detect from workspace).",
    )
    parser.add_argument(
        "--skip-mpb",
        action="store_true",
        help="Skip running MPB benchmarks (only read criterion results).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Output PNG plot path (default: same as CSV but with .png extension).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the plot.",
    )
    args = parser.parse_args()
    
    output = args.output
    if not output.is_absolute():
        output = Path(__file__).parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Run MPB benchmarks or read existing data
    if not args.skip_mpb:
        print(f"MPB Benchmark: Square lattice, eps={EPS_BG}, r/a={RADIUS}, {NUM_BANDS} bands")
        print(f"k-path: Γ → X → M → Γ ({K_DENSITY} points/segment)")
        print("-" * 60)
        
        for resolution in args.resolutions:
            for polarization in args.polarizations:
                print(f"Running: resolution={resolution}, polarization={polarization.upper()}...", end=" ", flush=True)
                
                elapsed_ms = run_benchmark(resolution, polarization)
                
                results.append({
                    "resolution": resolution,
                    "polarization": polarization.upper(),
                    "source": "mpb",
                    "time_ms": elapsed_ms,
                })
                
                print(f"{elapsed_ms:.2f} ms")
        
        print("-" * 60)
    else:
        # Read existing MPB data from CSV if available
        if output.exists():
            print(f"Reading existing MPB data from: {output}")
            try:
                with open(output, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["source"] == "mpb":
                            results.append({
                                "resolution": int(row["resolution"]),
                                "polarization": row["polarization"],
                                "source": row["source"],
                                "time_ms": float(row["time_ms"]),
                            })
                if results:
                    print(f"Loaded {len(results)} existing MPB results")
                else:
                    warnings.warn(
                        f"No MPB data found in {output}.\n"
                        "Run without --skip-mpb first to generate MPB benchmark data."
                    )
            except Exception as e:
                warnings.warn(f"Failed to read existing MPB data: {e}")
    
    # Read criterion results
    criterion_dir = args.criterion_dir
    if criterion_dir is None:
        # Auto-detect: look for target/criterion relative to script
        workspace_root = Path(__file__).parent.parent
        criterion_dir = workspace_root / "target" / "criterion"
    
    if criterion_dir.exists():
        print(f"Reading criterion results from: {criterion_dir}")
        criterion_results = read_criterion_results(criterion_dir, args.resolutions)
        
        if criterion_results:
            # Count unique sources
            sources = set(r["source"] for r in criterion_results)
            print(f"Found {len(criterion_results)} criterion results from: {', '.join(sorted(sources))}")
            results.extend(criterion_results)
        else:
            warnings.warn(f"No criterion benchmark data found in {criterion_dir}/resolution_scaling/")
    else:
        warnings.warn(
            f"Criterion results directory not found: {criterion_dir}\n"
            "Run 'cargo bench --bench resolution_scaling -p mpb2d-backend-cpu --features cuda' first."
        )
    
    # Sort results by resolution, then polarization, then source
    source_order = {"mpb": 0, "mpb2d CPU": 1, "mpb2d GPU": 2}
    results.sort(key=lambda r: (r["resolution"], r["polarization"], source_order.get(r["source"], 99)))
    
    # Write CSV
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["resolution", "polarization", "source", "time_ms"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults written to: {output}")
    print(f"Total rows: {len(results)}")
    
    # Generate plot
    if not args.no_plot and results:
        plot_output = args.plot_output
        if plot_output is None:
            plot_output = output.with_suffix(".png")
        elif not plot_output.is_absolute():
            plot_output = Path(__file__).parent / plot_output
        
        create_benchmark_plot(results, plot_output)


if __name__ == "__main__":
    main()
