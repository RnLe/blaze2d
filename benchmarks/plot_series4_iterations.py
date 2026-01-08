#!/usr/bin/env python3
"""
Plot Series 4: Iteration Count Comparison

Generates bar plots comparing MPB vs Blaze2D iteration counts per k-point.
Creates two plots: one for TM and one for TE polarization.

This plot is crucial for understanding the fundamental eigensolver efficiency
difference between MPB and Blaze2D, independent of BLAS/implementation details.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# Plot Configuration
# ============================================================================
COLORS = {
    "mpb": "#E74C3C",      # Red
    "blaze": "#3498DB",    # Blue
}

LABELS = {
    "mpb": "MPB",
    "blaze": "Blaze2D",
}

# High-symmetry point labels for square lattice
K_PATH_LABELS = {
    0.0: "Γ",
    0.5: "X",
    (np.sqrt(2) / 2): "M",
}

# Style configuration
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_results(results_file: Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def get_high_symmetry_positions(k_points: list, k_per_seg: int) -> list:
    """Get positions of high-symmetry points on the k-path."""
    # For square lattice: Γ(0,0) → X(0.5,0) → M(0.5,0.5) → Γ(0,0)
    # Indices: 0, k_per_seg, 2*k_per_seg, 3*k_per_seg
    total = len(k_points)
    if total == 3 * k_per_seg + 1:
        return [
            (0, "Γ"),
            (k_per_seg, "X"),
            (2 * k_per_seg, "M"),
            (3 * k_per_seg, "Γ"),
        ]
    return []


def plot_iteration_bars(results: dict, polarization: str, ax: plt.Axes):
    """Create bar plot showing iterations per k-point for a given polarization."""
    data = results[polarization]
    
    mpb_data = data.get("mpb")
    blaze_data = data.get("blaze")
    
    if not mpb_data and not blaze_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        return
    
    # Determine the number of k-points from each solver
    n_mpb = len(mpb_data["k_points"]) if mpb_data else 0
    n_blaze = len(blaze_data["k_points"]) if blaze_data else 0
    n_kpts = max(n_mpb, n_blaze)
    
    # Bar width and positions
    width = 0.35
    
    # Extract iterations and plot each solver with its own k-indices
    if mpb_data:
        mpb_iters = np.array([kp["iterations"] for kp in mpb_data["k_points"]])
        mpb_indices = np.arange(len(mpb_iters))
        ax.bar(mpb_indices - width/2, mpb_iters, width, 
               label=LABELS["mpb"], color=COLORS["mpb"], alpha=0.85,
               edgecolor='white', linewidth=0.5)
    
    if blaze_data:
        blaze_iters = np.array([kp["iterations"] for kp in blaze_data["k_points"]])
        blaze_indices = np.arange(len(blaze_iters))
        ax.bar(blaze_indices + width/2, blaze_iters, width,
               label=LABELS["blaze"], color=COLORS["blaze"], alpha=0.85,
               edgecolor='white', linewidth=0.5)
    
    # Use the solver with more k-points as reference for axis
    ref_data = mpb_data if n_mpb >= n_blaze else blaze_data
    k_indices = np.arange(len(ref_data["k_points"]))
    
    # Labels and formatting
    ax.set_xlabel("K-point index")
    ax.set_ylabel("Iterations to convergence")
    ax.set_title(f"{polarization} Polarization")
    ax.legend(loc='upper right')
    
    # X-axis: show every 5th or 10th label depending on count
    step = 10 if n_kpts > 40 else 5
    ax.set_xticks(k_indices[::step])
    ax.set_xticklabels([str(i) for i in k_indices[::step]])
    
    # Add high-symmetry labels if we have the parameters
    params = results.get("parameters", {})
    k_per_seg = params.get("k_points_per_segment", 20)
    hs_positions = get_high_symmetry_positions(ref_data["k_points"], k_per_seg)
    
    if hs_positions:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([pos for pos, _ in hs_positions])
        ax2.set_xticklabels([label for _, label in hs_positions])
        ax2.tick_params(axis='x', length=0)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_time_per_iteration_bars(results: dict, polarization: str, ax: plt.Axes):
    """Create bar plot showing time per iteration for each k-point."""
    data = results[polarization]
    
    mpb_data = data.get("mpb")
    blaze_data = data.get("blaze")
    
    if not mpb_data and not blaze_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        return
    
    # Determine the number of k-points from each solver
    n_mpb = len(mpb_data["k_points"]) if mpb_data else 0
    n_blaze = len(blaze_data["k_points"]) if blaze_data else 0
    n_kpts = max(n_mpb, n_blaze)
    
    # Bar width and positions
    width = 0.35
    
    # Calculate time per iteration (in ms) and plot each solver
    if mpb_data:
        mpb_times = []
        for kp in mpb_data["k_points"]:
            elapsed = kp.get("elapsed_seconds", 0.0)
            iters = kp["iterations"]
            time_per_iter = (elapsed / iters * 1000) if iters > 0 else 0.0  # ms
            mpb_times.append(time_per_iter)
        mpb_times = np.array(mpb_times)
        mpb_indices = np.arange(len(mpb_times))
        ax.bar(mpb_indices - width/2, mpb_times, width,
               label=LABELS["mpb"], color=COLORS["mpb"], alpha=0.85,
               edgecolor='white', linewidth=0.5)
    
    if blaze_data:
        blaze_times = []
        for kp in blaze_data["k_points"]:
            elapsed = kp.get("elapsed_seconds", 0.0)
            iters = kp["iterations"]
            time_per_iter = (elapsed / iters * 1000) if iters > 0 else 0.0  # ms
            blaze_times.append(time_per_iter)
        blaze_times = np.array(blaze_times)
        blaze_indices = np.arange(len(blaze_times))
        ax.bar(blaze_indices + width/2, blaze_times, width,
               label=LABELS["blaze"], color=COLORS["blaze"], alpha=0.85,
               edgecolor='white', linewidth=0.5)
    
    # Use the solver with more k-points as reference for axis
    ref_data = mpb_data if n_mpb >= n_blaze else blaze_data
    k_indices = np.arange(len(ref_data["k_points"]))
    
    # Labels and formatting
    ax.set_xlabel("K-point index")
    ax.set_ylabel("Time per iteration (ms)")
    ax.set_title(f"{polarization} Polarization - Timing")
    ax.legend(loc='upper right')
    
    # X-axis: show every 5th or 10th label depending on count
    step = 10 if n_kpts > 40 else 5
    ax.set_xticks(k_indices[::step])
    ax.set_xticklabels([str(i) for i in k_indices[::step]])
    
    # Add high-symmetry labels if we have the parameters
    params = results.get("parameters", {})
    k_per_seg = params.get("k_points_per_segment", 20)
    hs_positions = get_high_symmetry_positions(ref_data["k_points"], k_per_seg)
    
    if hs_positions:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([pos for pos, _ in hs_positions])
        ax2.set_xticklabels([label for _, label in hs_positions])
        ax2.tick_params(axis='x', length=0)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def create_report_figure(results: dict) -> plt.Figure:
    """Create the full report figure with all panels."""
    fig = plt.figure(figsize=(14, 12))
    
    # Title
    params = results.get("parameters", {})
    res = params.get("resolution", "?")
    bands = params.get("num_bands", "?")
    eps = params.get("epsilon", "?")
    radius = params.get("radius", "?")
    k_per_seg = params.get("k_points_per_segment", "?")
    total_k = params.get("total_k_points", "?")
    
    fig.suptitle(
        f"Series 4: Iteration Count per K-Point\n"
        f"Square Lattice, ε={eps} rods, r={radius}a, {res}×{res}, {bands} bands, {total_k} k-points",
        fontsize=14, fontweight='bold'
    )
    
    # Create grid: 2 rows, 2 columns
    ax1 = fig.add_subplot(2, 2, 1)  # TM iteration bars
    ax2 = fig.add_subplot(2, 2, 2)  # TE iteration bars
    ax3 = fig.add_subplot(2, 2, 3)  # TM time per iteration bars
    ax4 = fig.add_subplot(2, 2, 4)  # TE time per iteration bars
    
    plot_iteration_bars(results, "TM", ax1)
    plot_iteration_bars(results, "TE", ax2)
    plot_time_per_iteration_bars(results, "TM", ax3)
    plot_time_per_iteration_bars(results, "TE", ax4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    return fig


def create_summary_figure(results: dict) -> plt.Figure:
    """Create a summary figure with comparison statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of iterations for each solver
    for idx, pol in enumerate(["TM", "TE"]):
        ax = axes[idx]
        data = results[pol]
        
        mpb_data = data.get("mpb")
        blaze_data = data.get("blaze")
        
        if mpb_data:
            mpb_iters = [kp["iterations"] for kp in mpb_data["k_points"]]
            ax.hist(mpb_iters, bins=20, alpha=0.6, label=LABELS["mpb"], 
                    color=COLORS["mpb"], edgecolor='white')
        
        if blaze_data:
            blaze_iters = [kp["iterations"] for kp in blaze_data["k_points"]]
            ax.hist(blaze_iters, bins=20, alpha=0.6, label=LABELS["blaze"],
                    color=COLORS["blaze"], edgecolor='white')
        
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Count")
        ax.set_title(f"{pol} Polarization - Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    params = results.get("parameters", {})
    fig.suptitle(
        f"Series 4: Iteration Distribution\n"
        f"Square Lattice, ε={params.get('epsilon', '?')} rods, "
        f"{params.get('resolution', '?')}×{params.get('resolution', '?')}",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def print_statistics(results: dict):
    """Print summary statistics."""
    params = results.get("parameters", {})
    
    print("\n" + "=" * 70)
    print("Series 4: Iteration Count Statistics")
    print("=" * 70)
    print(f"Resolution: {params.get('resolution')}×{params.get('resolution')}")
    print(f"Bands: {params.get('num_bands')}")
    print(f"K-points: {params.get('total_k_points')}")
    print(f"Config: ε={params.get('epsilon')}, r={params.get('radius')}a")
    print(f"Tolerances: MPB={params.get('mpb_tolerance', 'N/A')}, Blaze={params.get('blaze_tolerance', 'N/A')}")
    print("=" * 70)
    
    for pol in ["TM", "TE"]:
        data = results[pol]
        print(f"\n{pol} Polarization:")
        print("-" * 50)
        
        for solver in ["mpb", "blaze"]:
            solver_data = data.get(solver)
            if solver_data:
                iters = np.array([kp["iterations"] for kp in solver_data["k_points"]])
                times_ms = np.array([
                    (kp.get("elapsed_seconds", 0.0) / kp["iterations"] * 1000) if kp["iterations"] > 0 else 0.0
                    for kp in solver_data["k_points"]
                ])
                total_time = solver_data.get("total_elapsed", 0.0)
                print(f"  {LABELS[solver]:>10}: iters: total={np.sum(iters):5d}, "
                      f"mean={np.mean(iters):5.1f}, std={np.std(iters):5.1f}")
                print(f"  {' ':>10}  time/iter: mean={np.mean(times_ms):5.2f}ms, "
                      f"total={total_time:.3f}s")
            else:
                print(f"  {LABELS[solver]:>10}: (no data)")
        
        # Ratio
        if data.get("mpb") and data.get("blaze"):
            mpb_total = sum(kp["iterations"] for kp in data["mpb"]["k_points"])
            blaze_total = sum(kp["iterations"] for kp in data["blaze"]["k_points"])
            mpb_time = data["mpb"].get("total_elapsed", 0.0)
            blaze_time = data["blaze"].get("total_elapsed", 0.0)
            if blaze_total > 0:
                iter_ratio = mpb_total / blaze_total
                print(f"  {'Ratio':>10}: MPB uses {iter_ratio:.2f}× the iterations of Blaze2D")
            if blaze_time > 0:
                time_ratio = mpb_time / blaze_time
                print(f"  {' ':>10}  MPB takes {time_ratio:.2f}× the total time of Blaze2D")


def main():
    parser = argparse.ArgumentParser(description="Plot Series 4 results")
    parser.add_argument("--input", type=str, default="results/series4_iterations",
                        help="Input directory with results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    input_dir = script_dir / args.input
    output_dir = script_dir / (args.output or args.input)
    
    results_file = input_dir / "series4_iterations_results.json"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Run bench_series4_iterations.py first.")
        return 1
    
    results = load_results(results_file)
    print(f"Loaded results from: {results_file}")
    
    # Print statistics
    print_statistics(results)
    
    # Create figures
    fig_report = create_report_figure(results)
    fig_summary = create_summary_figure(results)
    
    # Save PNGs
    png_report = output_dir / "series4_iterations_comparison.png"
    png_summary = output_dir / "series4_iterations_distribution.png"
    
    fig_report.savefig(png_report, dpi=150, bbox_inches='tight')
    fig_summary.savefig(png_summary, dpi=150, bbox_inches='tight')
    
    print(f"\nSaved: {png_report}")
    print(f"Saved: {png_summary}")
    
    # Create PDF report
    pdf_file = output_dir / "series4_iterations_report.pdf"
    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig_report, bbox_inches='tight')
        pdf.savefig(fig_summary, bbox_inches='tight')
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Benchmark Series 4: Iteration Count Comparison'
        d['Author'] = 'Blaze2D Benchmark Suite'
        d['Subject'] = 'MPB vs Blaze2D LOBPCG Iteration Efficiency'
        d['CreationDate'] = results.get('timestamp', '')
    
    print(f"Saved PDF report: {pdf_file}")
    
    plt.close('all')
    return 0


if __name__ == "__main__":
    exit(main())
