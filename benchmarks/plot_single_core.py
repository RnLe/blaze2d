#!/usr/bin/env python3
"""
Plot Single-Core Benchmark Results

Generates comparison plots for single-core benchmarks between MPB and Blaze2D.

Output:
  results/single_core/plots/
    single_core_comparison.png     - Bar chart comparing times
    single_core_speedup.png        - Speedup factors
    single_core_summary.png        - Combined summary figure
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'mpb': '#E74C3C',       # Red
    'blaze2d': '#3498DB',   # Blue
    'speedup': '#2ECC71',   # Green
}

SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / "results" / "single_core"
PLOTS_DIR = RESULTS_DIR / "plots"

CONFIG_LABELS = {
    'config_a_tm': 'Square TM',
    'config_a_te': 'Square TE',
    'config_b_tm': 'Hex TM',
    'config_b_te': 'Hex TE',
}

CONFIG_ORDER = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te']


def load_results(results_dir: Path) -> dict:
    """Load all single-core benchmark results."""
    results = {'mpb': {}, 'blaze2d': {}}
    
    for config in CONFIG_ORDER:
        # Load MPB results
        mpb_file = results_dir / f"mpb_{config}.json"
        if mpb_file.exists():
            with open(mpb_file) as f:
                results['mpb'][config] = json.load(f)
        
        # Load Blaze2D results
        blaze_file = results_dir / f"blaze2d_{config}.json"
        if blaze_file.exists():
            with open(blaze_file) as f:
                results['blaze2d'][config] = json.load(f)
    
    return results


def compute_speedup(mpb_mean: float, mpb_std: float, 
                    blaze_mean: float, blaze_std: float) -> tuple:
    """Compute speedup with error propagation."""
    if blaze_mean <= 0:
        return float('inf'), 0
    
    speedup = mpb_mean / blaze_mean
    rel_err_mpb = mpb_std / mpb_mean if mpb_mean > 0 else 0
    rel_err_blaze = blaze_std / blaze_mean if blaze_mean > 0 else 0
    speedup_err = speedup * np.sqrt(rel_err_mpb**2 + rel_err_blaze**2)
    
    return speedup, speedup_err


def plot_time_comparison(results: dict, output_dir: Path):
    """Create bar chart comparing execution times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = [c for c in CONFIG_ORDER if c in results['mpb'] and c in results['blaze2d']]
    if not configs:
        print("No matching results found for time comparison")
        return
    
    x = np.arange(len(configs))
    width = 0.35
    
    mpb_means = [results['mpb'][c]['mean_ms'] for c in configs]
    mpb_stds = [results['mpb'][c]['std_ms'] for c in configs]
    blaze_means = [results['blaze2d'][c]['mean_ms'] for c in configs]
    blaze_stds = [results['blaze2d'][c]['std_ms'] for c in configs]
    
    bars1 = ax.bar(x - width/2, mpb_means, width, yerr=mpb_stds, 
                   label='MPB', color=COLORS['mpb'], capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, blaze_means, width, yerr=blaze_stds,
                   label='Blaze2D', color=COLORS['blaze2d'], capsize=5, alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time per Job (ms)', fontsize=12)
    ax.set_title('Single-Core Performance Comparison: MPB vs Blaze2D', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.legend()
    
    # Add value labels on bars
    def autolabel(bars, stds):
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1, mpb_stds)
    autolabel(bars2, blaze_stds)
    
    plt.tight_layout()
    output_file = output_dir / "single_core_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_speedup(results: dict, output_dir: Path):
    """Create speedup bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = [c for c in CONFIG_ORDER if c in results['mpb'] and c in results['blaze2d']]
    if not configs:
        print("No matching results found for speedup plot")
        return
    
    speedups = []
    speedup_errs = []
    
    for config in configs:
        mpb = results['mpb'][config]
        blaze = results['blaze2d'][config]
        sp, sp_err = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                      blaze['mean_ms'], blaze['std_ms'])
        speedups.append(sp)
        speedup_errs.append(sp_err)
    
    x = np.arange(len(configs))
    bars = ax.bar(x, speedups, yerr=speedup_errs, color=COLORS['speedup'], 
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add horizontal line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Speedup (MPB time / Blaze2D time)', fontsize=12)
    ax.set_title('Blaze2D Speedup over MPB (Single-Core)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    
    # Add value labels
    for bar, sp, err in zip(bars, speedups, speedup_errs):
        height = bar.get_height()
        ax.annotate(f'{sp:.2f}×',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = output_dir / "single_core_speedup.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_summary(results: dict, output_dir: Path):
    """Create combined summary figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    configs = [c for c in CONFIG_ORDER if c in results['mpb'] and c in results['blaze2d']]
    if not configs:
        print("No matching results found for summary plot")
        return
    
    # Left: Time comparison
    ax1 = axes[0]
    x = np.arange(len(configs))
    width = 0.35
    
    mpb_means = [results['mpb'][c]['mean_ms'] for c in configs]
    mpb_stds = [results['mpb'][c]['std_ms'] for c in configs]
    blaze_means = [results['blaze2d'][c]['mean_ms'] for c in configs]
    blaze_stds = [results['blaze2d'][c]['std_ms'] for c in configs]
    
    ax1.bar(x - width/2, mpb_means, width, yerr=mpb_stds,
            label='MPB', color=COLORS['mpb'], capsize=4, alpha=0.8)
    ax1.bar(x + width/2, blaze_means, width, yerr=blaze_stds,
            label='Blaze2D', color=COLORS['blaze2d'], capsize=4, alpha=0.8)
    
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Time per Job (ms)', fontsize=11)
    ax1.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=10)
    ax1.legend(loc='upper right')
    
    # Right: Speedup
    ax2 = axes[1]
    speedups = []
    speedup_errs = []
    
    for config in configs:
        mpb = results['mpb'][config]
        blaze = results['blaze2d'][config]
        sp, sp_err = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                      blaze['mean_ms'], blaze['std_ms'])
        speedups.append(sp)
        speedup_errs.append(sp_err)
    
    bars = ax2.bar(x, speedups, yerr=speedup_errs, color=COLORS['speedup'],
                   capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Speedup Factor', fontsize=11)
    ax2.set_title('Blaze2D Speedup over MPB', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=10)
    ax2.set_ylim(bottom=0)
    
    # Add speedup labels
    for bar, sp in zip(bars, speedups):
        ax2.annotate(f'{sp:.2f}×', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle('Single-Core Benchmark: MPB vs Blaze2D', fontsize=14, fontweight='bold', y=1.02)
    
    # Add metadata
    avg_speedup = np.mean(speedups)
    fig.text(0.5, -0.02, f'Average Speedup: {avg_speedup:.2f}× | Resolution: 64×64 | Bands: 8 | K-points: 61',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    output_file = output_dir / "single_core_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_summary_table(results: dict):
    """Print a summary table to console."""
    configs = [c for c in CONFIG_ORDER if c in results['mpb'] and c in results['blaze2d']]
    
    print("\n" + "=" * 80)
    print("SINGLE-CORE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Config':<12} {'MPB (ms)':<18} {'Blaze2D (ms)':<18} {'Speedup':<12}")
    print("-" * 80)
    
    speedups = []
    for config in configs:
        mpb = results['mpb'][config]
        blaze = results['blaze2d'][config]
        sp, sp_err = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                      blaze['mean_ms'], blaze['std_ms'])
        speedups.append(sp)
        
        print(f"{CONFIG_LABELS[config]:<12} "
              f"{mpb['mean_ms']:>7.2f} ± {mpb['std_ms']:<7.2f} "
              f"{blaze['mean_ms']:>7.2f} ± {blaze['std_ms']:<7.2f} "
              f"{sp:>5.2f}× ± {sp_err:.2f}")
    
    print("-" * 80)
    print(f"{'Average':<12} {'':<18} {'':<18} {np.mean(speedups):>5.2f}×")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Plot Single-Core Benchmark Results")
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory with results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()
    
    results_dir = Path(args.input) if args.input else RESULTS_DIR
    output_dir = Path(args.output) if args.output else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    # Check what we have
    mpb_count = len(results['mpb'])
    blaze_count = len(results['blaze2d'])
    print(f"Found: {mpb_count} MPB results, {blaze_count} Blaze2D results")
    
    if mpb_count == 0 or blaze_count == 0:
        print("ERROR: Need both MPB and Blaze2D results to create plots")
        return
    
    # Generate plots
    print(f"\nGenerating plots to: {output_dir}")
    plot_time_comparison(results, output_dir)
    plot_speedup(results, output_dir)
    plot_summary(results, output_dir)
    
    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
