#!/usr/bin/env python3
"""
Plot Series 5: Memory Usage Comparison

Generates plots comparing MPB vs Blaze2D memory usage (Peak RSS) across
three parameter sweeps: resolution, number of bands, and k-points per segment.

Creates:
1. Main comparison figure with 6 panels (3 sweeps × 2 polarizations)
2. Summary figure with memory efficiency ratios
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
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

MARKERS = {
    "mpb": "o",
    "blaze": "s",
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


def extract_sweep_data(sweep_data: dict, polarization: str) -> Dict[str, Dict]:
    """
    Extract memory data for a specific polarization from sweep results.
    
    Returns dict with 'mpb' and 'blaze' keys, each containing:
        - values: list of sweep variable values
        - memory_mb: list of peak RSS in MB (mean if multiple runs)
        - memory_mb_std: list of std deviations (0 if single run)
        - elapsed: list of elapsed times (mean if multiple runs)
    
    Supports both old format (single run) and new format (multiple runs with mean/std).
    """
    values = sweep_data["values"]
    results = sweep_data["results"]
    
    data = {
        "mpb": {"values": [], "memory_mb": [], "memory_mb_std": [], "elapsed": []},
        "blaze": {"values": [], "memory_mb": [], "memory_mb_std": [], "elapsed": []},
    }
    
    for r in results:
        if r["polarization"] != polarization:
            continue
        if not r["success"]:
            continue
        
        solver_key = "mpb" if r["solver"] == "MPB" else "blaze"
        sweep_var = sweep_data["variable"]
        value = r[sweep_var]
        
        data[solver_key]["values"].append(value)
        
        # Check if new format with mean/std or old format
        if "peak_rss_mb_mean" in r:
            data[solver_key]["memory_mb"].append(r["peak_rss_mb_mean"])
            data[solver_key]["memory_mb_std"].append(r["peak_rss_mb_std"])
            data[solver_key]["elapsed"].append(r["elapsed_seconds_mean"])
        else:
            data[solver_key]["memory_mb"].append(r["peak_rss_mb"])
            data[solver_key]["memory_mb_std"].append(0.0)
            data[solver_key]["elapsed"].append(r["elapsed_seconds"])
    
    return data


def plot_memory_sweep(ax: plt.Axes, sweep_data: dict, polarization: str, 
                      xlabel: str, title_suffix: str = ""):
    """Plot memory usage as bar chart for a single sweep and polarization with error bars."""
    
    data = extract_sweep_data(sweep_data, polarization)
    
    if not data["mpb"]["values"] and not data["blaze"]["values"]:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes)
        return
    
    # Get unique sweep values (use MPB or Blaze, whichever has data)
    sweep_values = data["mpb"]["values"] if data["mpb"]["values"] else data["blaze"]["values"]
    n_vals = len(sweep_values)
    x = np.arange(n_vals)
    width = 0.35
    
    # Create dicts for easy lookup
    mpb_dict = dict(zip(data["mpb"]["values"], data["mpb"]["memory_mb"]))
    blaze_dict = dict(zip(data["blaze"]["values"], data["blaze"]["memory_mb"]))
    mpb_std_dict = dict(zip(data["mpb"]["values"], data["mpb"]["memory_mb_std"]))
    blaze_std_dict = dict(zip(data["blaze"]["values"], data["blaze"]["memory_mb_std"]))
    
    mpb_mem = [mpb_dict.get(v, 0) for v in sweep_values]
    blaze_mem = [blaze_dict.get(v, 0) for v in sweep_values]
    mpb_std = [mpb_std_dict.get(v, 0) for v in sweep_values]
    blaze_std = [blaze_std_dict.get(v, 0) for v in sweep_values]
    
    # Create bars with error bars
    ax.bar(x - width/2, mpb_mem, width, label=LABELS["mpb"], 
           color=COLORS["mpb"], alpha=0.85, edgecolor='white', linewidth=0.5,
           yerr=mpb_std, capsize=3)
    ax.bar(x + width/2, blaze_mem, width, label=LABELS["blaze"],
           color=COLORS["blaze"], alpha=0.85, edgecolor='white', linewidth=0.5,
           yerr=blaze_std, capsize=3)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(f"{polarization} Polarization{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in sweep_values])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def plot_memory_ratio(ax: plt.Axes, sweep_data: dict, xlabel: str):
    """Plot memory ratio (MPB / Blaze2D) for both polarizations."""
    
    for pol in ["TM", "TE"]:
        data = extract_sweep_data(sweep_data, pol)
        
        if not data["mpb"]["values"] or not data["blaze"]["values"]:
            continue
        
        # Match values between solvers
        mpb_dict = dict(zip(data["mpb"]["values"], data["mpb"]["memory_mb"]))
        blaze_dict = dict(zip(data["blaze"]["values"], data["blaze"]["memory_mb"]))
        
        common_values = sorted(set(mpb_dict.keys()) & set(blaze_dict.keys()))
        ratios = [mpb_dict[v] / blaze_dict[v] if blaze_dict[v] > 0 else 0 
                  for v in common_values]
        
        ax.plot(common_values, ratios, marker='o', label=pol, linewidth=2, 
                markersize=5, alpha=0.85)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Memory Ratio (MPB / Blaze2D)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def create_main_figure(results: dict) -> plt.Figure:
    """Create the main comparison figure with all sweeps."""
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    config = results.get("config", {})
    
    fig.suptitle(
        f"Series 5: Memory Usage Comparison (Peak RSS)\n"
        f"Square Lattice, ε={config.get('epsilon', '?')} rods, r={config.get('radius', '?')}a",
        fontsize=14, fontweight='bold'
    )
    
    sweeps = results.get("sweeps", {})
    
    # Row 1: Resolution sweep
    if "resolution" in sweeps:
        sweep = sweeps["resolution"]
        fixed = sweep.get("fixed", {})
        title_suffix = f"\n(bands={fixed.get('num_bands')}, k/seg={fixed.get('k_points_per_segment')})"
        plot_memory_sweep(axes[0, 0], sweep, "TM", "Resolution", title_suffix)
        plot_memory_sweep(axes[0, 1], sweep, "TE", "Resolution", title_suffix)
    
    # Row 2: Bands sweep
    if "num_bands" in sweeps:
        sweep = sweeps["num_bands"]
        fixed = sweep.get("fixed", {})
        title_suffix = f"\n(res={fixed.get('resolution')}, k/seg={fixed.get('k_points_per_segment')})"
        plot_memory_sweep(axes[1, 0], sweep, "TM", "Number of Bands", title_suffix)
        plot_memory_sweep(axes[1, 1], sweep, "TE", "Number of Bands", title_suffix)
    
    # Row 3: K-points sweep
    if "k_points_per_segment" in sweeps:
        sweep = sweeps["k_points_per_segment"]
        fixed = sweep.get("fixed", {})
        title_suffix = f"\n(res={fixed.get('resolution')}, bands={fixed.get('num_bands')})"
        plot_memory_sweep(axes[2, 0], sweep, "TM", "K-points per Segment", title_suffix)
        plot_memory_sweep(axes[2, 1], sweep, "TE", "K-points per Segment", title_suffix)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_ratio_figure(results: dict) -> plt.Figure:
    """Create a figure showing memory ratios for all sweeps."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    config = results.get("config", {})
    
    fig.suptitle(
        f"Series 5: Memory Efficiency Ratio (MPB / Blaze2D)\n"
        f"Values > 1 mean Blaze2D uses less memory",
        fontsize=12, fontweight='bold'
    )
    
    sweeps = results.get("sweeps", {})
    
    sweep_configs = [
        ("resolution", "Resolution", axes[0]),
        ("num_bands", "Number of Bands", axes[1]),
        ("k_points_per_segment", "K-points per Segment", axes[2]),
    ]
    
    for sweep_key, xlabel, ax in sweep_configs:
        if sweep_key in sweeps:
            plot_memory_ratio(ax, sweeps[sweep_key], xlabel)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def create_scaling_figure(results: dict) -> plt.Figure:
    """Create a figure showing memory scaling (log-log) for resolution sweep."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    config = results.get("config", {})
    sweeps = results.get("sweeps", {})
    
    fig.suptitle(
        f"Series 5: Memory Scaling with Resolution (Log-Log)\n"
        f"Expected: O(N²) where N = resolution",
        fontsize=12, fontweight='bold'
    )
    
    if "resolution" not in sweeps:
        return fig
    
    sweep = sweeps["resolution"]
    
    for idx, pol in enumerate(["TM", "TE"]):
        ax = axes[idx]
        data = extract_sweep_data(sweep, pol)
        
        for solver in ["mpb", "blaze"]:
            if data[solver]["values"]:
                x = np.array(data[solver]["values"])
                y = np.array(data[solver]["memory_mb"])
                
                ax.loglog(
                    x, y,
                    marker=MARKERS[solver],
                    color=COLORS[solver],
                    label=LABELS[solver],
                    linewidth=2,
                    markersize=6,
                    alpha=0.85,
                )
                
                # Fit power law: y = a * x^b
                if len(x) >= 2:
                    log_x = np.log(x)
                    log_y = np.log(y)
                    b, log_a = np.polyfit(log_x, log_y, 1)
                    
                    # Plot fit line
                    x_fit = np.linspace(x.min(), x.max(), 50)
                    y_fit = np.exp(log_a) * x_fit ** b
                    ax.loglog(x_fit, y_fit, '--', color=COLORS[solver], alpha=0.5,
                              label=f'{LABELS[solver]} fit: O(N^{b:.2f})')
        
        ax.set_xlabel("Resolution (N)")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title(f"{pol} Polarization")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def print_statistics(results: dict):
    """Print summary statistics."""
    
    config = results.get("config", {})
    sweeps = results.get("sweeps", {})
    
    print("\n" + "=" * 70)
    print("Series 5: Memory Usage Statistics")
    print("=" * 70)
    print(f"Config: ε={config.get('epsilon')}, r={config.get('radius')}a")
    print("=" * 70)
    
    for sweep_name, sweep_data in sweeps.items():
        print(f"\n{sweep_name.upper()} Sweep:")
        print(f"  Values: {sweep_data['values']}")
        print(f"  Fixed: {sweep_data['fixed']}")
        print("-" * 50)
        
        for pol in ["TM", "TE"]:
            data = extract_sweep_data(sweep_data, pol)
            
            print(f"  {pol}:")
            for solver in ["mpb", "blaze"]:
                if data[solver]["memory_mb"]:
                    mem = data[solver]["memory_mb"]
                    print(f"    {LABELS[solver]:>10}: "
                          f"min={min(mem):.1f}MB, max={max(mem):.1f}MB, "
                          f"range={max(mem)-min(mem):.1f}MB")
            
            # Calculate average ratio
            if data["mpb"]["memory_mb"] and data["blaze"]["memory_mb"]:
                mpb_dict = dict(zip(data["mpb"]["values"], data["mpb"]["memory_mb"]))
                blaze_dict = dict(zip(data["blaze"]["values"], data["blaze"]["memory_mb"]))
                common = set(mpb_dict.keys()) & set(blaze_dict.keys())
                if common:
                    ratios = [mpb_dict[v] / blaze_dict[v] for v in common if blaze_dict[v] > 0]
                    if ratios:
                        print(f"    {'Avg Ratio':>10}: {np.mean(ratios):.2f}× (MPB/Blaze)")


def main():
    parser = argparse.ArgumentParser(description="Plot Series 5 results")
    parser.add_argument("--input", type=str, default="results/series5_memory",
                        help="Input directory with results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    input_dir = script_dir / args.input
    output_dir = script_dir / (args.output or args.input)
    
    results_file = input_dir / "series5_memory_results.json"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Run bench_series5_memory.py first.")
        return 1
    
    results = load_results(results_file)
    print(f"Loaded results from: {results_file}")
    
    # Print statistics
    print_statistics(results)
    
    # Create figures
    fig_main = create_main_figure(results)
    fig_ratio = create_ratio_figure(results)
    fig_scaling = create_scaling_figure(results)
    
    # Save PNGs
    png_main = output_dir / "series5_memory_comparison.png"
    png_ratio = output_dir / "series5_memory_ratio.png"
    png_scaling = output_dir / "series5_memory_scaling.png"
    
    fig_main.savefig(png_main, dpi=150, bbox_inches='tight')
    fig_ratio.savefig(png_ratio, dpi=150, bbox_inches='tight')
    fig_scaling.savefig(png_scaling, dpi=150, bbox_inches='tight')
    
    print(f"\nSaved: {png_main}")
    print(f"Saved: {png_ratio}")
    print(f"Saved: {png_scaling}")
    
    # Create PDF report
    pdf_file = output_dir / "series5_memory_report.pdf"
    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig_main, bbox_inches='tight')
        pdf.savefig(fig_ratio, bbox_inches='tight')
        pdf.savefig(fig_scaling, bbox_inches='tight')
        
        d = pdf.infodict()
        d['Title'] = 'Benchmark Series 5: Memory Usage Comparison'
        d['Author'] = 'Blaze2D Benchmark Suite'
        d['Subject'] = 'MPB vs Blaze2D Peak RSS Memory Comparison'
        d['CreationDate'] = results.get('timestamp', '')
    
    print(f"Saved PDF report: {pdf_file}")
    
    plt.close('all')
    return 0


if __name__ == "__main__":
    exit(main())
