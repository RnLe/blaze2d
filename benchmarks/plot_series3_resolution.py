#!/usr/bin/env python3
"""
Plot Series 3: Resolution Sweep Results

Generates bar plots comparing MPB vs Blaze2D across resolution values (16-512).
Outputs both PNG and PDF for inclusion in reports.
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


def plot_bar_comparison(results: dict, polarization: str, ax: plt.Axes):
    """Create bar plot comparing solvers for a given polarization."""
    data = results[polarization]
    resolutions = data["resolution"]
    n_res = len(resolutions)
    
    # Extract times
    mpb_times = []
    mpb_errs = []
    blaze_times = []
    blaze_errs = []
    
    for i in range(n_res):
        if data["mpb"][i]:
            mpb_times.append(data["mpb"][i]["mean"])
            mpb_errs.append(data["mpb"][i]["std"])
        else:
            mpb_times.append(0)
            mpb_errs.append(0)
        
        if data["blaze"][i]:
            blaze_times.append(data["blaze"][i]["mean"])
            blaze_errs.append(data["blaze"][i]["std"])
        else:
            blaze_times.append(0)
            blaze_errs.append(0)
    
    # Bar positions
    x = np.arange(n_res)
    width = 0.35
    
    # Create bars
    bars_mpb = ax.bar(
        x - width/2, mpb_times, width,
        yerr=mpb_errs, capsize=2,
        label=LABELS["mpb"], color=COLORS["mpb"], alpha=0.85,
        edgecolor='white', linewidth=0.5,
    )
    bars_blaze = ax.bar(
        x + width/2, blaze_times, width,
        yerr=blaze_errs, capsize=2,
        label=LABELS["blaze"], color=COLORS["blaze"], alpha=0.85,
        edgecolor='white', linewidth=0.5,
    )
    
    # Labels and formatting
    ax.set_xlabel("Resolution (N×N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{polarization} Polarization")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}" for r in resolutions], rotation=45, ha='right')
    ax.legend(loc='upper left')
    
    # Linear scale for y-axis (clearer comparison)
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_speedup(results: dict, polarization: str, ax: plt.Axes):
    """Plot speedup of Blaze2D over MPB."""
    data = results[polarization]
    resolutions = data["resolution"]
    
    speedups = []
    for i in range(len(resolutions)):
        mpb_data = data["mpb"][i]
        blaze_data = data["blaze"][i]
        
        if mpb_data and blaze_data and blaze_data["mean"] > 0:
            speedup = mpb_data["mean"] / blaze_data["mean"]
            speedups.append(speedup)
        else:
            speedups.append(0)
    
    x = np.arange(len(resolutions))
    
    # Color bars by speedup value
    colors = ['#2ECC71' if s > 1 else '#E74C3C' for s in speedups]
    
    bars = ax.bar(x, speedups, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Add horizontal line at 1x
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel("Resolution (N×N)")
    ax.set_ylabel("Speedup (MPB / Blaze2D)")
    ax.set_title(f"{polarization} Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}" for r in resolutions], rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        if val > 0:
            ax.annotate(
                f'{val:.1f}×',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=7,
            )
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def create_report_figure(results: dict) -> plt.Figure:
    """Create the full report figure with all panels."""
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    params = results.get("parameters", {})
    eps = params.get("epsilon", "?")
    radius = params.get("radius", "?")
    bands = params.get("num_bands", "?")
    
    fig.suptitle(
        f"Series 3: Resolution Sweep\n"
        f"Square Lattice, ε={eps} rods, r={radius}a, {bands} bands",
        fontsize=14, fontweight='bold'
    )
    
    # Create grid: 2 rows, 2 columns
    ax1 = fig.add_subplot(2, 2, 1)  # TM bars
    ax2 = fig.add_subplot(2, 2, 2)  # TE bars
    ax3 = fig.add_subplot(2, 2, 3)  # TM speedup
    ax4 = fig.add_subplot(2, 2, 4)  # TE speedup
    
    plot_bar_comparison(results, "TM", ax1)
    plot_bar_comparison(results, "TE", ax2)
    plot_speedup(results, "TM", ax3)
    plot_speedup(results, "TE", ax4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    return fig


def create_scaling_figure(results: dict) -> plt.Figure:
    """Create a figure showing scaling behavior."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, pol in enumerate(["TM", "TE"]):
        ax = axes[idx]
        data = results[pol]
        resolutions = np.array(data["resolution"])
        
        mpb_times = []
        blaze_times = []
        
        for i in range(len(resolutions)):
            mpb_times.append(data["mpb"][i]["mean"] if data["mpb"][i] else np.nan)
            blaze_times.append(data["blaze"][i]["mean"] if data["blaze"][i] else np.nan)
        
        mpb_times = np.array(mpb_times)
        blaze_times = np.array(blaze_times)
        
        # Plot
        ax.loglog(resolutions, mpb_times, 'o-', color=COLORS["mpb"], 
                  label=LABELS["mpb"], linewidth=2, markersize=6)
        ax.loglog(resolutions, blaze_times, 's-', color=COLORS["blaze"],
                  label=LABELS["blaze"], linewidth=2, markersize=6)
        
        ax.set_xlabel("Resolution (N)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{pol} Polarization - Scaling Behavior")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Add reference lines for O(N²) and O(N³)
        res_ref = resolutions[len(resolutions)//2]
        time_ref = (mpb_times[len(resolutions)//2] + blaze_times[len(resolutions)//2]) / 2
        
        # O(N²) reference
        n2_line = time_ref * (resolutions / res_ref) ** 2
        ax.loglog(resolutions, n2_line, '--', color='gray', alpha=0.5, label='O(N²)')
        
        # O(N³) reference
        n3_line = time_ref * (resolutions / res_ref) ** 3 / 10
        ax.loglog(resolutions, n3_line, ':', color='gray', alpha=0.5, label='O(N³)')
    
    params = results.get("parameters", {})
    fig.suptitle(
        f"Series 3: Resolution Scaling Analysis\n"
        f"Square Lattice, ε={params.get('epsilon', '?')} rods, "
        f"r={params.get('radius', '?')}a, {params.get('num_bands', '?')} bands",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot Series 3 results")
    parser.add_argument("--input", type=str, default="results/series3_resolution",
                        help="Input directory with results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    input_dir = script_dir / args.input
    output_dir = script_dir / (args.output or args.input)
    
    results_file = input_dir / "series3_resolution_results.json"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Run bench_series3_resolution.py first.")
        return 1
    
    results = load_results(results_file)
    print(f"Loaded results from: {results_file}")
    
    # Create figures
    fig_report = create_report_figure(results)
    fig_scaling = create_scaling_figure(results)
    
    # Save PNGs
    png_report = output_dir / "series3_resolution_comparison.png"
    png_scaling = output_dir / "series3_resolution_scaling.png"
    
    fig_report.savefig(png_report, dpi=150, bbox_inches='tight')
    fig_scaling.savefig(png_scaling, dpi=150, bbox_inches='tight')
    
    print(f"Saved: {png_report}")
    print(f"Saved: {png_scaling}")
    
    # Create PDF report
    pdf_file = output_dir / "series3_resolution_report.pdf"
    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig_report, bbox_inches='tight')
        pdf.savefig(fig_scaling, bbox_inches='tight')
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Benchmark Series 3: Resolution Sweep'
        d['Author'] = 'Blaze2D Benchmark Suite'
        d['Subject'] = 'MPB vs Blaze2D Resolution Scaling Comparison'
        d['CreationDate'] = results.get('timestamp', '')
    
    print(f"Saved PDF report: {pdf_file}")
    
    plt.close('all')
    return 0


if __name__ == "__main__":
    exit(main())
