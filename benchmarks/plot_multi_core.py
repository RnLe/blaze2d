#!/usr/bin/env python3
"""
Plot Multi-Core Benchmark Results

Generates comparison plots for multi-core benchmarks:
- Blaze2D (Rayon parallelism, 16 threads)
- MPB Multiprocess (16 subprocess workers, each single-threaded)
- MPB OpenBLAS (default OpenBLAS multi-threading)

Output:
  results/multi_core/plots/
    multi_core_comparison.png      - Bar chart comparing times
    multi_core_throughput.png      - Throughput comparison
    multi_core_speedup.png         - Speedup vs MPB OpenBLAS
    multi_core_summary.png         - Combined summary figure
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
    'blaze2d': '#3498DB',           # Blue
    'mpb_multiprocess': '#E74C3C',  # Red
    'mpb_openblas': '#F39C12',      # Orange
    'speedup': '#2ECC71',           # Green
}

SOLVER_LABELS = {
    'blaze2d': 'Blaze2D (Rayon)',
    'mpb_multiprocess': 'MPB (Multiprocess)',
    'mpb_openblas': 'MPB (OpenBLAS)',
}

SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / "results" / "multi_core"
PLOTS_DIR = RESULTS_DIR / "plots"

CONFIG_LABELS = {
    'config_a_tm': 'Square TM',
    'config_a_te': 'Square TE',
    'config_b_tm': 'Hex TM',
    'config_b_te': 'Hex TE',
}

CONFIG_ORDER = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te']


def load_results(results_dir: Path) -> dict:
    """Load all multi-core benchmark results."""
    results = {
        'blaze2d': {},
        'mpb_multiprocess': {},
        'mpb_openblas': {},
    }
    
    for config in CONFIG_ORDER:
        # Load Blaze2D results
        blaze_file = results_dir / f"blaze2d_{config}.json"
        if blaze_file.exists():
            with open(blaze_file) as f:
                results['blaze2d'][config] = json.load(f)
        
        # Load MPB Multiprocess results
        mp_file = results_dir / f"mpb_multiprocess_{config}.json"
        if mp_file.exists():
            with open(mp_file) as f:
                results['mpb_multiprocess'][config] = json.load(f)
        
        # Load MPB OpenBLAS results
        ob_file = results_dir / f"mpb_openblas_{config}.json"
        if ob_file.exists():
            with open(ob_file) as f:
                results['mpb_openblas'][config] = json.load(f)
    
    return results


def compute_speedup(baseline_mean: float, baseline_std: float,
                    target_mean: float, target_std: float) -> tuple:
    """Compute speedup with error propagation."""
    if target_mean <= 0:
        return float('inf'), 0
    
    speedup = baseline_mean / target_mean
    rel_err_base = baseline_std / baseline_mean if baseline_mean > 0 else 0
    rel_err_target = target_std / target_mean if target_mean > 0 else 0
    speedup_err = speedup * np.sqrt(rel_err_base**2 + rel_err_target**2)
    
    return speedup, speedup_err


def get_available_configs(results: dict) -> list:
    """Get configs that have at least two solvers for comparison."""
    available = []
    for config in CONFIG_ORDER:
        solvers_with_data = sum(1 for s in results if config in results[s])
        if solvers_with_data >= 2:
            available.append(config)
    return available


def plot_time_comparison(results: dict, output_dir: Path):
    """Create grouped bar chart comparing execution times."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = get_available_configs(results)
    if not configs:
        print("No matching results found for time comparison")
        return
    
    solvers = [s for s in ['blaze2d', 'mpb_multiprocess', 'mpb_openblas'] 
               if any(c in results[s] for c in configs)]
    
    x = np.arange(len(configs))
    width = 0.25
    offsets = np.linspace(-width, width, len(solvers))
    
    for i, solver in enumerate(solvers):
        means = []
        stds = []
        for config in configs:
            if config in results[solver]:
                means.append(results[solver][config]['mean_ms'])
                stds.append(results[solver][config]['std_ms'])
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x + offsets[i], means, width, yerr=stds,
                     label=SOLVER_LABELS[solver], color=COLORS[solver],
                     capsize=4, alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Effective Time per Job (ms)', fontsize=12)
    ax.set_title('Multi-Core Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_file = output_dir / "multi_core_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_throughput(results: dict, output_dir: Path):
    """Create throughput comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = get_available_configs(results)
    if not configs:
        print("No matching results found for throughput plot")
        return
    
    solvers = [s for s in ['blaze2d', 'mpb_multiprocess', 'mpb_openblas']
               if any(c in results[s] for c in configs)]
    
    x = np.arange(len(configs))
    width = 0.25
    offsets = np.linspace(-width, width, len(solvers))
    
    for i, solver in enumerate(solvers):
        throughputs = []
        throughput_stds = []
        for config in configs:
            if config in results[solver]:
                data = results[solver][config]
                throughputs.append(data.get('mean_throughput_jobs_s', 1000/data['mean_ms']))
                throughput_stds.append(data.get('std_throughput_jobs_s', 0))
            else:
                throughputs.append(0)
                throughput_stds.append(0)
        
        bars = ax.bar(x + offsets[i], throughputs, width, yerr=throughput_stds,
                     label=SOLVER_LABELS[solver], color=COLORS[solver],
                     capsize=4, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Throughput (jobs/second)', fontsize=12)
    ax.set_title('Multi-Core Throughput Comparison (16 threads/workers)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = output_dir / "multi_core_throughput.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_speedup(results: dict, output_dir: Path):
    """Create speedup bar chart (Blaze2D speedup over both MPB modes)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = get_available_configs(results)
    configs_with_all = [c for c in configs 
                        if c in results['blaze2d'] and 
                        (c in results['mpb_multiprocess'] or c in results['mpb_openblas'])]
    
    if not configs_with_all:
        print("No matching results found for speedup plot")
        return
    
    x = np.arange(len(configs_with_all))
    width = 0.35
    
    # Speedup vs MPB Multiprocess
    speedups_mp = []
    speedup_errs_mp = []
    for config in configs_with_all:
        if config in results['mpb_multiprocess'] and config in results['blaze2d']:
            mpb = results['mpb_multiprocess'][config]
            blaze = results['blaze2d'][config]
            sp, sp_err = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                         blaze['mean_ms'], blaze['std_ms'])
            speedups_mp.append(sp)
            speedup_errs_mp.append(sp_err)
        else:
            speedups_mp.append(0)
            speedup_errs_mp.append(0)
    
    # Speedup vs MPB OpenBLAS
    speedups_ob = []
    speedup_errs_ob = []
    for config in configs_with_all:
        if config in results['mpb_openblas'] and config in results['blaze2d']:
            mpb = results['mpb_openblas'][config]
            blaze = results['blaze2d'][config]
            sp, sp_err = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                         blaze['mean_ms'], blaze['std_ms'])
            speedups_ob.append(sp)
            speedup_errs_ob.append(sp_err)
        else:
            speedups_ob.append(0)
            speedup_errs_ob.append(0)
    
    # Plot bars
    if any(s > 0 for s in speedups_mp):
        bars1 = ax.bar(x - width/2, speedups_mp, width, yerr=speedup_errs_mp,
                      label='vs MPB Multiprocess', color=COLORS['mpb_multiprocess'],
                      capsize=4, alpha=0.8)
        for bar, sp in zip(bars1, speedups_mp):
            if sp > 0:
                ax.annotate(f'{sp:.2f}×',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if any(s > 0 for s in speedups_ob):
        bars2 = ax.bar(x + width/2, speedups_ob, width, yerr=speedup_errs_ob,
                      label='vs MPB OpenBLAS', color=COLORS['mpb_openblas'],
                      capsize=4, alpha=0.8)
        for bar, sp in zip(bars2, speedups_ob):
            if sp > 0:
                ax.annotate(f'{sp:.2f}×',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Blaze2D Speedup over MPB (Multi-Core)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs_with_all])
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = output_dir / "multi_core_speedup.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_summary(results: dict, output_dir: Path):
    """Create combined summary figure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    configs = get_available_configs(results)
    if not configs:
        print("No matching results found for summary plot")
        return
    
    solvers = [s for s in ['blaze2d', 'mpb_multiprocess', 'mpb_openblas']
               if any(c in results[s] for c in configs)]
    
    # Left: Throughput comparison
    ax1 = axes[0]
    x = np.arange(len(configs))
    width = 0.25
    offsets = np.linspace(-width, width, len(solvers))
    
    for i, solver in enumerate(solvers):
        throughputs = []
        for config in configs:
            if config in results[solver]:
                data = results[solver][config]
                throughputs.append(data.get('mean_throughput_jobs_s', 1000/data['mean_ms']))
            else:
                throughputs.append(0)
        
        bars = ax1.bar(x + offsets[i], throughputs, width,
                      label=SOLVER_LABELS[solver], color=COLORS[solver], alpha=0.8)
    
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Throughput (jobs/s)', fontsize=11)
    ax1.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(bottom=0)
    
    # Right: Speedup
    ax2 = axes[1]
    
    # Calculate average speedups
    avg_speedups = {}
    for baseline in ['mpb_multiprocess', 'mpb_openblas']:
        speedups = []
        for config in configs:
            if config in results['blaze2d'] and config in results[baseline]:
                blaze = results['blaze2d'][config]
                mpb = results[baseline][config]
                sp, _ = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                        blaze['mean_ms'], blaze['std_ms'])
                if sp > 0 and sp < float('inf'):
                    speedups.append(sp)
        if speedups:
            avg_speedups[baseline] = np.mean(speedups)
    
    if avg_speedups:
        baselines = list(avg_speedups.keys())
        speedup_vals = [avg_speedups[b] for b in baselines]
        colors = [COLORS[b] for b in baselines]
        labels = ['vs ' + SOLVER_LABELS[b].split('(')[1].rstrip(')') for b in baselines]
        
        bars = ax2.bar(range(len(baselines)), speedup_vals, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1)
        ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        ax2.set_xticks(range(len(baselines)))
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel('Average Speedup', fontsize=11)
        ax2.set_title('Blaze2D Average Speedup', fontsize=12, fontweight='bold')
        ax2.set_ylim(bottom=0)
        
        for bar, sp in zip(bars, speedup_vals):
            ax2.annotate(f'{sp:.2f}×',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    fig.suptitle('Multi-Core Benchmark: Blaze2D vs MPB (16 threads/workers)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_file = output_dir / "multi_core_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_summary_table(results: dict):
    """Print a summary table to console."""
    configs = get_available_configs(results)
    
    print("\n" + "=" * 100)
    print("MULTI-CORE BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Config':<12} {'Blaze2D':<18} {'MPB Multiproc':<18} {'MPB OpenBLAS':<18} {'Speedup (vs MP)':<15}")
    print("-" * 100)
    
    all_speedups_mp = []
    all_speedups_ob = []
    
    for config in configs:
        blaze_str = "N/A"
        mp_str = "N/A"
        ob_str = "N/A"
        sp_str = "N/A"
        
        if config in results['blaze2d']:
            d = results['blaze2d'][config]
            blaze_str = f"{d['mean_ms']:>6.1f} ± {d['std_ms']:<5.1f}"
        
        if config in results['mpb_multiprocess']:
            d = results['mpb_multiprocess'][config]
            mp_str = f"{d['mean_ms']:>6.1f} ± {d['std_ms']:<5.1f}"
        
        if config in results['mpb_openblas']:
            d = results['mpb_openblas'][config]
            ob_str = f"{d['mean_ms']:>6.1f} ± {d['std_ms']:<5.1f}"
        
        # Calculate speedup vs multiprocess
        if config in results['blaze2d'] and config in results['mpb_multiprocess']:
            blaze = results['blaze2d'][config]
            mpb = results['mpb_multiprocess'][config]
            sp, sp_err = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                         blaze['mean_ms'], blaze['std_ms'])
            sp_str = f"{sp:>5.2f}×"
            all_speedups_mp.append(sp)
        
        if config in results['blaze2d'] and config in results['mpb_openblas']:
            blaze = results['blaze2d'][config]
            mpb = results['mpb_openblas'][config]
            sp, _ = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                    blaze['mean_ms'], blaze['std_ms'])
            all_speedups_ob.append(sp)
        
        print(f"{CONFIG_LABELS[config]:<12} {blaze_str:<18} {mp_str:<18} {ob_str:<18} {sp_str:<15}")
    
    print("-" * 100)
    avg_mp = np.mean(all_speedups_mp) if all_speedups_mp else 0
    avg_ob = np.mean(all_speedups_ob) if all_speedups_ob else 0
    print(f"{'Average':<12} {'':<18} {'':<18} {'':<18} {avg_mp:>5.2f}×")
    if all_speedups_ob:
        print(f"{'(vs OpenBLAS)':<12} {'':<18} {'':<18} {'':<18} {avg_ob:>5.2f}×")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Plot Multi-Core Benchmark Results")
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
    for solver, data in results.items():
        print(f"  {solver}: {len(data)} configs")
    
    total = sum(len(d) for d in results.values())
    if total == 0:
        print("ERROR: No results found")
        return
    
    # Generate plots
    print(f"\nGenerating plots to: {output_dir}")
    plot_time_comparison(results, output_dir)
    plot_throughput(results, output_dir)
    plot_speedup(results, output_dir)
    plot_summary(results, output_dir)
    
    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
