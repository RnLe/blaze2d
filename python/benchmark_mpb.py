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

Additionally, it runs mpb2d-cli for comparison and computes relative eigenvalue
errors between MPB and mpb2d solvers using Hungarian matching.

Output: CSV file with columns: resolution, polarization, source, time_ms
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Save copies of the original fds
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        # Open /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect stdout and stderr to /dev/null
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


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


def _load_scipy_optimize():
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment
    except ModuleNotFoundError:
        return None


np, mp, mpb = _load_runtime_modules()
plt = _load_plotting_modules()
linear_sum_assignment = _load_scipy_optimize()


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


def run_benchmark(resolution: int, polarization: str, quiet: bool = True) -> Tuple[float, List[List[float]]]:
    """Run a single MPB calculation and return elapsed time in milliseconds and eigenvalues.
    
    Args:
        resolution: Grid resolution.
        polarization: "tm" or "te".
        quiet: If True, suppress MPB's verbose output.
    
    Returns:
        Tuple of (elapsed_ms, bands) where bands is a list of lists of frequencies.
        bands[k_idx][band_idx] = frequency at k-point k_idx for band band_idx.
    """
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
    
    # Time the actual solve (with optional output suppression)
    start = time.perf_counter()
    
    if quiet:
        with suppress_output():
            if polarization == "tm":
                solver.run_tm()
            else:
                solver.run_te()
    else:
        if polarization == "tm":
            solver.run_tm()
        else:
            solver.run_te()
    
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000.0
    
    # Extract eigenvalues - solver.all_freqs is list of k-points, each with list of bands
    # Note: MPB returns normalized frequencies ω/(2πc/a)
    bands = []
    for freqs in solver.all_freqs:
        bands.append([float(f) for f in freqs])
    
    return elapsed_ms, bands


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


# ============================================================================
# mpb2d Integration Functions
# ============================================================================

def generate_toml_config(resolution: int, polarization: str) -> str:
    """Generate a TOML configuration file for mpb2d-cli."""
    return f"""# Auto-generated benchmark config: resolution={resolution}, polarization={polarization.upper()}
polarization = "{polarization.upper()}"

[geometry]
eps_bg = {EPS_BG}

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {RADIUS}
eps_inside = {EPS_HOLE}

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = {K_DENSITY}

[eigensolver]
n_bands = {NUM_BANDS}
max_iter = 1000
tol = 1e-6
"""


def run_mpb2d(resolution: int, polarization: str, workspace_root: Path) -> Tuple[Optional[List[List[float]]], float]:
    """Run mpb2d-cli for a given configuration and return eigenvalues and timing.
    
    Returns:
        Tuple of (bands, elapsed_ms) where bands is a list of lists of frequencies,
        or (None, 0.0) if mpb2d-cli fails.
        bands[k_idx][band_idx] = frequency at k-point k_idx for band band_idx.
    """
    cli_path = workspace_root / "target" / "release" / "mpb2d-cli"
    if not cli_path.exists():
        cli_path = workspace_root / "target" / "debug" / "mpb2d-cli"
    
    if not cli_path.exists():
        warnings.warn(f"mpb2d-cli not found at {cli_path}")
        return None, 0.0
    
    # Create temporary files for config and output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml_content = generate_toml_config(resolution, polarization)
        f.write(toml_content)
        config_path = Path(f.name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        # Run mpb2d-cli with timing
        start = time.perf_counter()
        result = subprocess.run(
            [str(cli_path), "--config", str(config_path), "--output", str(output_path), "--quiet"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        if result.returncode != 0:
            warnings.warn(f"mpb2d-cli failed for res={resolution} pol={polarization}: {result.stderr}")
            return None, elapsed_ms
        
        # Parse output CSV
        bands = []
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                warnings.warn(f"Empty CSV output for res={resolution} pol={polarization}")
                return None, elapsed_ms
            band_columns = [name for name in reader.fieldnames if name.startswith("band")]
            for row in reader:
                k_bands = []
                for col in band_columns:
                    value = row[col]
                    k_bands.append(float(value) if value else float("nan"))
                bands.append(k_bands)
        
        return bands, elapsed_ms
    
    except subprocess.TimeoutExpired:
        warnings.warn(f"mpb2d-cli timed out for res={resolution} pol={polarization}")
        return None, 0.0
    except Exception as e:
        warnings.warn(f"mpb2d-cli error for res={resolution} pol={polarization}: {e}")
        return None, 0.0
    finally:
        # Cleanup
        config_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def compute_hungarian_deviations(mpb_bands: List[List[float]], mpb2d_bands: List[List[float]]) -> List[float]:
    """Compute per-band deviations using Hungarian matching algorithm.
    
    For each k-point, finds the optimal assignment of mpb2d bands to mpb bands
    that minimizes total deviation, then returns all individual deviations.
    
    Args:
        mpb_bands: bands[k_idx][band_idx] = frequency for MPB reference
        mpb2d_bands: bands[k_idx][band_idx] = frequency for mpb2d solver
    
    Returns:
        List of all deviations |mpb2d - mpb| after optimal matching.
    """
    if linear_sum_assignment is None:
        warnings.warn("scipy not available, cannot compute Hungarian matching")
        return []
    
    all_deviations = []
    
    for k_idx, (mpb_k, mpb2d_k) in enumerate(zip(mpb_bands, mpb2d_bands)):
        # Convert to numpy arrays
        mpb_vals = np.array(mpb_k, dtype=float)
        mpb2d_vals = np.array(mpb2d_k, dtype=float)
        
        # Skip if any NaN
        if np.any(np.isnan(mpb_vals)) or np.any(np.isnan(mpb2d_vals)):
            continue
        
        # Skip if all values are near zero (Gamma point)
        if np.max(np.abs(mpb_vals)) < 1e-6:
            continue
        
        # Build cost matrix: |mpb[i] - mpb2d[j]|
        cost = np.abs(mpb_vals[:, None] - mpb2d_vals[None, :])
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Collect deviations after optimal matching
        for i, j in zip(row_ind, col_ind):
            deviation = abs(mpb_vals[i] - mpb2d_vals[j])
            all_deviations.append(deviation)
    
    return all_deviations


def compute_relative_error(mpb_bands: List[List[float]], mpb2d_bands: List[List[float]], debug: bool = False) -> float:
    """Compute mean relative error using trace comparison.
    
    For each k-point, computes the trace (sum of all eigenvalues) and compares.
    This is robust against band reordering issues.
    
    Returns:
        Mean relative error of traces across all k-points.
    """
    if debug:
        print(f"  DEBUG: mpb_bands has {len(mpb_bands)} k-points, first k-point has {len(mpb_bands[0]) if mpb_bands else 0} bands")
        print(f"  DEBUG: mpb2d_bands has {len(mpb2d_bands)} k-points, first k-point has {len(mpb2d_bands[0]) if mpb2d_bands else 0} bands")
        if mpb_bands and mpb2d_bands:
            print(f"  DEBUG: mpb_bands[0] = {mpb_bands[0][:4]}...")
            print(f"  DEBUG: mpb2d_bands[0] = {mpb2d_bands[0][:4]}...")
    
    errors = []
    
    for k_idx, (mpb_k, mpb2d_k) in enumerate(zip(mpb_bands, mpb2d_bands)):
        # Compute trace (sum of eigenvalues) at each k-point
        mpb_trace = sum(v for v in mpb_k if not (v != v))  # skip NaN
        mpb2d_trace = sum(v for v in mpb2d_k if not (v != v))  # skip NaN
        
        if debug and k_idx < 3:
            print(f"  DEBUG: k={k_idx}: mpb_trace={mpb_trace:.6f}, mpb2d_trace={mpb2d_trace:.6f}")
        
        # Skip if trace is too small (near Gamma point with zero modes)
        if mpb_trace > 1e-6:
            rel_err = abs(mpb2d_trace - mpb_trace) / mpb_trace
            errors.append(rel_err)
    
    return float(np.mean(errors)) if errors else float('nan')


def create_benchmark_plot(results: List[dict], error_results: List[dict], output_path: Path) -> None:
    """Create a benchmark comparison plot with 2x3 subplots.
    
    Top row: TM polarization (linear left, log middle, deviation distribution right)
    Bottom row: TE polarization (linear left, log middle, deviation distribution right)
    
    The error plots show Hungarian-matched deviation distributions as sideways histograms
    for each resolution, with equal spacing between resolutions.
    
    Colors:
    - mpb: Blue
    - mpb2d CPU: Green
    - mpb2d GPU: Orange
    """
    if plt is None:
        warnings.warn("matplotlib not available, skipping plot generation")
        return
    
    # Color scheme for each source
    color_scheme = {
        "mpb": "#1f4e79",           # Dark blue
        "mpb2d CPU": "#2e7d32",     # Dark green
        "mpb2d GPU": "#e65100",     # Dark orange
    }
    
    # Marker scheme
    marker_scheme = {
        "mpb": "s",          # Square
        "mpb2d CPU": "o",    # Circle
        "mpb2d GPU": "^",    # Triangle
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_tm_lin, ax_tm_log, ax_tm_err = axes[0]
    ax_te_lin, ax_te_log, ax_te_err = axes[1]
    
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
    
    # Plot timing data - use categorical x-axis (equal spacing)
    x_positions = list(range(len(sorted_resolutions)))
    res_to_x = {res: i for i, res in enumerate(sorted_resolutions)}
    
    for (source, pol), data in sorted(grouped.items()):
        data.sort(key=lambda x: x[0])  # Sort by resolution
        x_vals = [res_to_x[d[0]] for d in data]
        times_ms = [d[1] for d in data]
        times_s = [d[1] / 1000.0 for d in data]  # Convert to seconds for linear plot
        
        color = color_scheme.get(source, "#888888")
        marker = marker_scheme.get(source, "o")
        label = source
        
        # Select axes based on polarization
        if pol == "TM":
            ax_lin, ax_log = ax_tm_lin, ax_tm_log
        else:
            ax_lin, ax_log = ax_te_lin, ax_te_log
        
        # Plot on linear axis (in seconds)
        ax_lin.plot(x_vals, times_s, 
                    color=color, 
                    marker=marker, 
                    linestyle="-",
                    linewidth=2,
                    markersize=8,
                    label=label)
        
        # Plot on log axis (in milliseconds)
        ax_log.plot(x_vals, times_ms, 
                    color=color, 
                    marker=marker, 
                    linestyle="-",
                    linewidth=2,
                    markersize=8,
                    label=label)
    
    # Group error data by polarization and resolution
    error_grouped = defaultdict(dict)  # pol -> {resolution: deviations}
    for r in error_results:
        pol = r["polarization"]
        res = r["resolution"]
        if "deviations" in r:
            error_grouped[pol][res] = r["deviations"]
    
    # Plot deviation distributions as sideways scatter/violin for each resolution
    error_color = "#7b1fa2"  # Purple for deviations
    
    for pol in ["TM", "TE"]:
        ax_err = ax_tm_err if pol == "TM" else ax_te_err
        
        if pol in error_grouped:
            for res, deviations in error_grouped[pol].items():
                if res not in res_to_x or not deviations:
                    continue
                
                x_pos = res_to_x[res]
                devs = np.array(deviations)
                
                # Filter out zeros for log scale
                devs = devs[devs > 0]
                if len(devs) == 0:
                    continue
                
                # Create a horizontal scatter with slight jitter for visibility
                jitter = np.random.uniform(-0.25, 0.25, len(devs))
                ax_err.scatter(x_pos + jitter, devs, 
                              color=error_color, 
                              alpha=0.4, 
                              s=15, 
                              edgecolors='none')
                
                # Add box plot statistics (median, quartiles)
                median = np.median(devs)
                q1, q3 = np.percentile(devs, [25, 75])
                
                # Draw median line
                ax_err.hlines(median, x_pos - 0.3, x_pos + 0.3, 
                             colors=error_color, linewidth=2, zorder=5)
                # Draw IQR box
                ax_err.vlines(x_pos, q1, q3, colors=error_color, linewidth=4, alpha=0.6, zorder=4)
    
    # Configure all axes
    for ax_lin, ax_log, ax_err, pol_label in [(ax_tm_lin, ax_tm_log, ax_tm_err, "TM"), 
                                               (ax_te_lin, ax_te_log, ax_te_err, "TE")]:
        # Linear plot
        ax_lin.set_xlabel("Resolution", fontsize=12)
        ax_lin.set_ylabel("Time (s)", fontsize=12)
        ax_lin.set_title(f"{pol_label} Polarization - Linear Scale", fontsize=14)
        ax_lin.set_xticks(x_positions)
        ax_lin.set_xticklabels([str(r) for r in sorted_resolutions])
        ax_lin.set_ylim(bottom=0)
        ax_lin.grid(True, alpha=0.3)
        ax_lin.legend(loc="upper left", fontsize=9)
        
        # Log plot
        ax_log.set_xlabel("Resolution", fontsize=12)
        ax_log.set_ylabel("Time (ms)", fontsize=12)
        ax_log.set_title(f"{pol_label} Polarization - Log Scale", fontsize=14)
        ax_log.set_yscale("log")
        ax_log.set_xticks(x_positions)
        ax_log.set_xticklabels([str(r) for r in sorted_resolutions])
        ax_log.grid(True, alpha=0.3, which='both')
        ax_log.legend(loc="upper left", fontsize=9)
        # Add 1 second reference line (1000 ms) with attached label
        ax_log.axhline(y=1000, color='#909090', linewidth=1.5, linestyle='-', zorder=1)
        if x_positions:
            x_center = (x_positions[0] + x_positions[-1]) / 2
            ax_log.text(x_center, 1000, '1 s', fontsize=10, ha='center', va='center', 
                       color='#606060', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Error plot - Hungarian deviation distribution
        ax_err.set_xlabel("Resolution", fontsize=12)
        ax_err.set_ylabel("Deviation |ω_mpb2d - ω_mpb|", fontsize=12)
        ax_err.set_title(f"{pol_label} Polarization - Eigenvalue Deviation", fontsize=14)
        ax_err.set_yscale("log")
        ax_err.set_xticks(x_positions)
        ax_err.set_xticklabels([str(r) for r in sorted_resolutions])
        ax_err.grid(True, alpha=0.3, which='both')
        
        # Add reference lines at key deviation thresholds with attached labels
        ax_err.axhline(y=1e-2, color='#a0a0a0', linewidth=1.5, linestyle='-', zorder=1)
        ax_err.axhline(y=1e-3, color='#a0a0a0', linewidth=1.5, linestyle='-', zorder=1)
        ax_err.axhline(y=1e-4, color='#a0a0a0', linewidth=1.5, linestyle='-', zorder=1)
        
        # Add centered labels on the reference lines with white background
        if x_positions:
            x_center = (x_positions[0] + x_positions[-1]) / 2
            ax_err.text(x_center, 1e-2, '1 %', fontsize=9, ha='center', va='center', 
                       color='#707070', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
            ax_err.text(x_center, 1e-3, '0.1 %', fontsize=9, ha='center', va='center', 
                       color='#707070', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
            ax_err.text(x_center, 1e-4, '0.01 %', fontsize=9, ha='center', va='center', 
                       color='#707070', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
    
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
        "--skip-error-analysis",
        action="store_true",
        help="Skip eigenvalue error analysis (only do timing benchmarks).",
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
    
    # Workspace root for finding mpb2d-cli
    workspace_root = Path(__file__).parent.parent
    
    # Build mpb2d-cli (CPU backend - CUDA FFT not yet implemented)
    # NOTE: The CUDA backend has BLAS operations but FFT is not implemented yet,
    # so eigensolving doesn't work with CUDA. Using CPU backend for now.
    print("Building mpb2d-cli (CPU backend)...")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "-p", "mpb2d-cli"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )
    if build_result.returncode != 0:
        print(f"Error building mpb2d-cli:\n{build_result.stderr}")
        raise SystemExit(1)
    print("Build complete.")
    
    results = []
    mpb_eigenvalues = {}  # (resolution, polarization) -> bands
    
    # Run MPB benchmarks or read existing data
    if not args.skip_mpb:
        print(f"MPB Benchmark: Square lattice, eps={EPS_BG}, r/a={RADIUS}, {NUM_BANDS} bands")
        print(f"k-path: Γ → X → M → Γ ({K_DENSITY} points/segment)")
        
        # Build task list
        mpb_tasks = [(res, pol) for res in args.resolutions for pol in args.polarizations]
        
        print("-" * 60)
        for resolution, polarization in mpb_tasks:
            print(f"Running: resolution={resolution}, polarization={polarization.upper()}...", end=" ", flush=True)
            
            elapsed_ms, bands = run_benchmark(resolution, polarization)
            
            results.append({
                "resolution": resolution,
                "polarization": polarization.upper(),
                "source": "mpb",
                "time_ms": elapsed_ms,
            })
            
            mpb_eigenvalues[(resolution, polarization.upper())] = bands
            
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
    
    # ========================================================================
    # Eigenvalue Error Analysis: Run mpb2d and compare with MPB
    # ========================================================================
    error_results = []
    
    if not args.skip_error_analysis and mpb_eigenvalues:
        print("\n" + "=" * 60)
        print("Eigenvalue Error Analysis: mpb2d vs MPB")
        print("=" * 60)
        
        # Build task list for mpb2d runs
        mpb2d_tasks = [
            (res, pol) for res in args.resolutions for pol in args.polarizations
            if (res, pol.upper()) in mpb_eigenvalues
        ]
        
        for resolution, polarization in mpb2d_tasks:
            pol_upper = polarization.upper()
            
            print(f"Running mpb2d: resolution={resolution}, polarization={pol_upper}...", end=" ", flush=True)
            
            mpb2d_bands, elapsed_ms = run_mpb2d(resolution, polarization, workspace_root)
            
            if mpb2d_bands is None:
                print("FAILED")
                continue
            
            mpb_bands = mpb_eigenvalues[(resolution, pol_upper)]
            
            # Compute relative error - debug first comparison
            is_first = len(error_results) == 0
            mean_rel_error = compute_relative_error(mpb_bands, mpb2d_bands, debug=is_first)
            
            # Compute Hungarian-matched deviations for detailed analysis
            deviations = compute_hungarian_deviations(mpb_bands, mpb2d_bands)
            
            error_results.append({
                "resolution": resolution,
                "polarization": pol_upper,
                "mean_rel_error": mean_rel_error,
                "time_ms": elapsed_ms,
                "deviations": deviations,  # Store raw deviations for plotting
            })
            
            if deviations:
                median_dev = np.median(deviations)
                max_dev = np.max(deviations)
                print(f"median dev = {median_dev:.2e}, max dev = {max_dev:.2e}, time = {elapsed_ms:.2f} ms")
            else:
                print(f"mean relative error = {mean_rel_error:.2e}, time = {elapsed_ms:.2f} ms")
        
        print("=" * 60)
        
        # Save error results to CSV (without deviations array)
        error_csv_path = output.with_name(output.stem + "_errors.csv")
        with open(error_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["resolution", "polarization", "mean_rel_error", "time_ms"])
            writer.writeheader()
            for r in error_results:
                writer.writerow({k: v for k, v in r.items() if k != "deviations"})
        print(f"Error results written to: {error_csv_path}")
    
    elif args.skip_error_analysis:
        print("\nSkipping eigenvalue error analysis (--skip-error-analysis)")
    elif not mpb_eigenvalues:
        print("\nNo MPB eigenvalue data available for error analysis.")
        print("Run without --skip-mpb to enable error analysis.")
    
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
        
        create_benchmark_plot(results, error_results, plot_output)


if __name__ == "__main__":
    main()
