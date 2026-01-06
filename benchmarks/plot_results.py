#!/usr/bin/env python3
"""
Generate Benchmark Plots - MPB vs Blaze2D Speed Comparison

Creates:
1. Bar chart comparing single-core performance
2. Bar chart comparing multi-core performance
3. Speedup comparison chart (both modes)
4. Summary figure with all comparisons
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).parent.absolute()

# Color scheme
COLORS = {
    "mpb": "#E74C3C",      # Red
    "blaze2d": "#3498DB",  # Blue
    "speedup": "#2ECC71",  # Green
}

def load_comparison_results(results_dir: Path) -> dict:
    """Load comparison results."""
    results_file = results_dir / "speed_comparison.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}\nRun analyze_speed.py first.")
    
    with open(results_file) as f:
        return json.load(f)

def plot_time_comparison(ax, mode_data: dict, title: str):
    """Plot time comparison bar chart for a single core mode."""
    comparisons = mode_data["comparisons"]
    core_mode = mode_data["core_mode"]
    
    config_labels = [f"{c['config']}\n{c['polarization']}" for c in comparisons]
    mpb_times = [c["mpb_mean_ms"] for c in comparisons]
    mpb_errs = [c["mpb_std_ms"] for c in comparisons]
    
    mpb_proc_times = [c.get("mpb_proc_mean_ms", 0.0) for c in comparisons]
    mpb_proc_errs = [c.get("mpb_proc_std_ms", 0.0) for c in comparisons]
    
    blaze_times = [c["blaze_mean_ms"] for c in comparisons]
    blaze_errs = [c["blaze_std_ms"] for c in comparisons]
    
    x = np.arange(len(config_labels))
    
    if core_mode == "multi":
        width = 0.25
        bars1 = ax.bar(x - width, mpb_times, width, yerr=mpb_errs,
                       label='MPB (Native)', color=COLORS["mpb"], capsize=3, alpha=0.8)
        bars2 = ax.bar(x, mpb_proc_times, width, yerr=mpb_proc_errs,
                       label='MPB (Process)', color="#8E44AD", capsize=3, alpha=0.8) # Purple
        bars3 = ax.bar(x + width, blaze_times, width, yerr=blaze_errs,
                       label='Blaze2D', color=COLORS["blaze2d"], capsize=3, alpha=0.8)
    else:
        width = 0.35
        bars1 = ax.bar(x - width/2, mpb_times, width, yerr=mpb_errs,
                       label='MPB', color=COLORS["mpb"], capsize=3, alpha=0.8)
        bars3 = ax.bar(x + width/2, blaze_times, width, yerr=blaze_errs,
                       label='Blaze2D', color=COLORS["blaze2d"], capsize=3, alpha=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time per job (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            label = f'{val:.0f}' if val > 0 else "N/A"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)

    add_labels(bars1, mpb_times)
    if core_mode == "multi":
        add_labels(bars2, mpb_proc_times)
    add_labels(bars3, blaze_times)

def plot_speedup_comparison(ax, results: dict):
    """Plot speedup comparison for both core modes."""
    
    modes = list(results["modes"].keys())
    if not modes:
        return
    
    # Get config labels from first mode
    first_mode = results["modes"][modes[0]]
    labels = [f"{c['config']}\n{c['polarization']}" for c in first_mode["comparisons"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    for i, mode in enumerate(modes):
        mode_data = results["modes"][mode]
        speedups = [c["speedup"] for c in mode_data["comparisons"]]
        speedup_errs = [c["speedup_err"] for c in mode_data["comparisons"]]
        
        offset = (i - len(modes)/2 + 0.5) * width
        color = COLORS["blaze2d"] if mode == "single" else COLORS["speedup"]
        
        bars = ax.bar(x + offset, speedups, width, yerr=speedup_errs,
                      label=f'{mode.capitalize()}-core', color=color, 
                      capsize=3, alpha=0.8)
        
        # Value labels
        for bar, val in zip(bars, speedups):
            label = f'{val:.1f}×' if val > 0 else "N/A"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal performance')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup (MPB time / Blaze2D time)')
    ax.set_title('Blaze2D Speedup vs MPB')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

def plot_summary_bars(ax, results: dict):
    """Plot summary bar chart with average speedups."""
    
    modes = list(results["modes"].keys())
    if not modes:
        return
    
    categories = []
    speedups = []
    errors = []
    colors = []
    
    for mode in modes:
        mode_data = results["modes"][mode]
        summary = mode_data["summary"]
        categories.append(f'{mode.capitalize()}-core')
        speedups.append(summary["mean_speedup"])
        errors.append(summary["combined_error"])
        colors.append(COLORS["blaze2d"] if mode == "single" else COLORS["speedup"])
    
    x = np.arange(len(categories))
    bars = ax.bar(x, speedups, yerr=errors, color=colors, capsize=5, alpha=0.8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mode')
    ax.set_ylabel('Average Speedup')
    ax.set_title('Average Blaze2D Speedup vs MPB')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis='y', alpha=0.3)
    
    # Value labels
    for bar, val, err in zip(bars, speedups, errors):
        label = f'{val:.1f}× ± {err:.1f}' if val > 0 else "N/A"
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

def create_full_report(results: dict, output_dir: Path):
    """Create comprehensive figure with all plots."""
    
    modes = list(results["modes"].keys())
    n_modes = len(modes)
    
    if n_modes == 0:
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    if n_modes == 2:
        # 2x2 layout: single-core, multi-core, speedup comparison, summary
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        plot_time_comparison(ax1, results["modes"]["single"], "Single-Core: Time per Job")
        plot_time_comparison(ax2, results["modes"]["multi"], "Multi-Core (16 threads): Time per Job")
        plot_speedup_comparison(ax3, results)
        plot_summary_bars(ax4, results)
    else:
        # Single mode
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        mode = modes[0]
        plot_time_comparison(ax1, results["modes"][mode], f"{mode.capitalize()}-Core: Time per Job")
        plot_speedup_comparison(ax2, results)
    
    plt.suptitle('Blaze2D vs MPB Speed Benchmark\n(Joannopoulos 1997 configurations, 32×32 resolution, 12 bands)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "speed_benchmark_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Report saved to: {output_file}")
    
    # Also save as PDF
    pdf_file = output_dir / "speed_benchmark_report.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF saved to: {pdf_file}")
    
    plt.close()

def create_individual_plots(results: dict, output_dir: Path):
    """Create individual plots for each comparison."""
    
    modes = list(results["modes"].keys())
    
    for mode in modes:
        mode_data = results["modes"][mode]
        
        # Time comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_time_comparison(ax, mode_data, f'{mode.capitalize()}-Core: Time per Job (ms)')
        plt.tight_layout()
        plt.savefig(output_dir / f"time_comparison_{mode}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Speedup comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_speedup_comparison(ax, results)
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_summary_bars(ax, results)
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Individual plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Plots")
    parser.add_argument("--output", type=str, default="results",
                        help="Results directory")
    parser.add_argument("--individual", action="store_true",
                        help="Also generate individual plots")
    args = parser.parse_args()
    
    results_dir = SCRIPT_DIR / args.output
    
    print("=" * 70)
    print("Generating Benchmark Plots")
    print("=" * 70)
    
    try:
        results = load_comparison_results(results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Create full report
    create_full_report(results, results_dir)
    
    # Optionally create individual plots
    if args.individual:
        create_individual_plots(results, results_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
