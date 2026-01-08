#!/usr/bin/env python3
"""
Plot Series 7: Multi-Threading Scaling

This script generates:
1. Throughput vs Thread Count (line plots)
2. Speedup vs Thread Count (relative to single-threaded)
3. Efficiency vs Thread Count (speedup / threads)
4. Bar comparison at max threads

Output: results/series7_scaling/plots/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================================
# Style Configuration
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Solver colors
SOLVER_COLORS = {
    "Blaze2D": "#2ca02c",        # Green
    "MPB-OMP": "#1f77b4",        # Blue
    "MPB-Multiproc": "#ff7f0e",  # Orange
}

SOLVER_MARKERS = {
    "Blaze2D": "o",
    "MPB-OMP": "s",
    "MPB-Multiproc": "^",
}


def load_results(results_path: Path) -> Dict:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def plot_throughput_scaling(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot throughput vs thread count for each resolution."""
    
    res_names = list(results["results"].keys())
    fig, axes = plt.subplots(1, len(res_names), figsize=(6 * len(res_names), 5))
    
    if len(res_names) == 1:
        axes = [axes]
    
    for ax, res_name in zip(axes, res_names):
        res_data = results["results"][res_name]
        res_val = res_data["resolution"]
        
        for solver_key, solver_name in [
            ("blaze", "Blaze2D"),
            ("mpb_omp", "MPB-OMP"),
            ("mpb_multiproc", "MPB-Multiproc"),
        ]:
            data = res_data[solver_key]
            threads = [d["threads"] for d in data]
            throughputs = [d["mean_throughput"] for d in data]
            errors = [d["std_throughput"] for d in data]
            
            ax.errorbar(
                threads, throughputs, yerr=errors,
                label=solver_name,
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name],
                markersize=8,
                linewidth=2,
                capsize=4,
                capthick=1.5,
            )
        
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Throughput (jobs/s)")
        ax.set_title(f"Resolution: {res_val}×{res_val}", fontweight='bold')
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max(threads) + 1)
        ax.set_ylim(bottom=0)
    
    fig.suptitle("Throughput Scaling with Thread Count", fontsize=14)
    fig.tight_layout()
    
    output_path = output_dir / "throughput_scaling.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_speedup(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot speedup vs thread count (relative to 1 thread)."""
    
    res_names = list(results["results"].keys())
    fig, axes = plt.subplots(1, len(res_names), figsize=(6 * len(res_names), 5))
    
    if len(res_names) == 1:
        axes = [axes]
    
    for ax, res_name in zip(axes, res_names):
        res_data = results["results"][res_name]
        res_val = res_data["resolution"]
        
        # Get max threads for ideal line
        all_threads = []
        
        for solver_key, solver_name in [
            ("blaze", "Blaze2D"),
            ("mpb_omp", "MPB-OMP"),
            ("mpb_multiproc", "MPB-Multiproc"),
        ]:
            data = res_data[solver_key]
            threads = [d["threads"] for d in data]
            throughputs = [d["mean_throughput"] for d in data]
            
            all_threads.extend(threads)
            
            # Speedup = throughput(N) / throughput(1)
            base_throughput = throughputs[0]  # 1 thread
            speedups = [t / base_throughput for t in throughputs]
            
            ax.plot(
                threads, speedups,
                label=solver_name,
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name],
                markersize=8,
                linewidth=2,
            )
        
        # Ideal linear speedup line
        max_threads = max(all_threads)
        ax.plot(
            [1, max_threads], [1, max_threads],
            'k--', alpha=0.5, linewidth=1.5,
            label="Ideal (linear)"
        )
        
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Speedup (×)")
        ax.set_title(f"Resolution: {res_val}×{res_val}", fontweight='bold')
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max_threads + 1)
        ax.set_ylim(bottom=0)
    
    fig.suptitle("Speedup vs Thread Count (relative to 1 thread)", fontsize=14)
    fig.tight_layout()
    
    output_path = output_dir / "speedup.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_efficiency(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot parallel efficiency (speedup / threads) vs thread count."""
    
    res_names = list(results["results"].keys())
    fig, axes = plt.subplots(1, len(res_names), figsize=(6 * len(res_names), 5))
    
    if len(res_names) == 1:
        axes = [axes]
    
    for ax, res_name in zip(axes, res_names):
        res_data = results["results"][res_name]
        res_val = res_data["resolution"]
        
        for solver_key, solver_name in [
            ("blaze", "Blaze2D"),
            ("mpb_omp", "MPB-OMP"),
            ("mpb_multiproc", "MPB-Multiproc"),
        ]:
            data = res_data[solver_key]
            threads = [d["threads"] for d in data]
            throughputs = [d["mean_throughput"] for d in data]
            
            base_throughput = throughputs[0]  # 1 thread
            efficiency = [(t / base_throughput) / n for t, n in zip(throughputs, threads)]
            
            ax.plot(
                threads, efficiency,
                label=solver_name,
                color=SOLVER_COLORS[solver_name],
                marker=SOLVER_MARKERS[solver_name],
                markersize=8,
                linewidth=2,
            )
        
        # Ideal 100% efficiency line
        max_threads = max(threads)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5,
                   label="Ideal (100%)")
        
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Parallel Efficiency")
        ax.set_title(f"Resolution: {res_val}×{res_val}", fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, max_threads + 1)
        ax.set_ylim(0, 1.2)
    
    fig.suptitle("Parallel Efficiency (Speedup / Threads)", fontsize=14)
    fig.tight_layout()
    
    output_path = output_dir / "efficiency.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_max_threads_comparison(
    results: Dict,
    output_dir: Path,
) -> None:
    """Bar plot comparing throughput at maximum thread count."""
    
    res_names = list(results["results"].keys())
    fig, axes = plt.subplots(1, len(res_names), figsize=(6 * len(res_names), 5))
    
    if len(res_names) == 1:
        axes = [axes]
    
    solver_order = [
        ("blaze", "Blaze2D"),
        ("mpb_omp", "MPB-OMP"),
        ("mpb_multiproc", "MPB-Multiproc"),
    ]
    
    for ax, res_name in zip(axes, res_names):
        res_data = results["results"][res_name]
        res_val = res_data["resolution"]
        
        labels = []
        values = []
        errors = []
        colors = []
        
        for solver_key, solver_name in solver_order:
            data = res_data[solver_key]
            # Get max threads entry
            max_entry = max(data, key=lambda x: x["threads"])
            
            labels.append(solver_name)
            values.append(max_entry["mean_throughput"])
            errors.append(max_entry["std_throughput"])
            colors.append(SOLVER_COLORS[solver_name])
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, yerr=errors, color=colors, alpha=0.8,
                      capsize=5, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        max_threads = max(d["threads"] for d in res_data["blaze"])
        ax.set_ylabel("Throughput (jobs/s)")
        ax.set_title(f"Resolution: {res_val}×{res_val}\n({max_threads} threads)", fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Peak Multi-Threaded Throughput Comparison", fontsize=14)
    fig.tight_layout()
    
    output_path = output_dir / "max_threads_comparison.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_combined_summary(
    results: Dict,
    output_dir: Path,
) -> None:
    """Combined summary plot with all metrics."""
    
    res_names = list(results["results"].keys())
    n_res = len(res_names)
    
    fig, axes = plt.subplots(3, n_res, figsize=(6 * n_res, 12))
    
    if n_res == 1:
        axes = axes.reshape(-1, 1)
    
    solver_order = [
        ("blaze", "Blaze2D"),
        ("mpb_omp", "MPB-OMP"),
        ("mpb_multiproc", "MPB-Multiproc"),
    ]
    
    for col, res_name in enumerate(res_names):
        res_data = results["results"][res_name]
        res_val = res_data["resolution"]
        
        # Row 0: Throughput
        ax = axes[0, col]
        for solver_key, solver_name in solver_order:
            data = res_data[solver_key]
            threads = [d["threads"] for d in data]
            throughputs = [d["mean_throughput"] for d in data]
            errors = [d["std_throughput"] for d in data]
            
            ax.errorbar(threads, throughputs, yerr=errors,
                       label=solver_name, color=SOLVER_COLORS[solver_name],
                       marker=SOLVER_MARKERS[solver_name], markersize=7,
                       linewidth=2, capsize=3)
        
        ax.set_ylabel("Throughput (jobs/s)")
        ax.set_title(f"Resolution: {res_val}×{res_val}", fontweight='bold')
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Row 1: Speedup
        ax = axes[1, col]
        max_threads = 0
        for solver_key, solver_name in solver_order:
            data = res_data[solver_key]
            threads = [d["threads"] for d in data]
            throughputs = [d["mean_throughput"] for d in data]
            max_threads = max(max_threads, max(threads))
            
            base = throughputs[0]
            speedups = [t / base for t in throughputs]
            
            ax.plot(threads, speedups, label=solver_name,
                   color=SOLVER_COLORS[solver_name],
                   marker=SOLVER_MARKERS[solver_name], markersize=7, linewidth=2)
        
        ax.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5,
               label="Ideal", linewidth=1.5)
        ax.set_ylabel("Speedup (×)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Row 2: Efficiency
        ax = axes[2, col]
        for solver_key, solver_name in solver_order:
            data = res_data[solver_key]
            threads = [d["threads"] for d in data]
            throughputs = [d["mean_throughput"] for d in data]
            
            base = throughputs[0]
            efficiency = [(t / base) / n for t, n in zip(throughputs, threads)]
            
            ax.plot(threads, efficiency, label=solver_name,
                   color=SOLVER_COLORS[solver_name],
                   marker=SOLVER_MARKERS[solver_name], markersize=7, linewidth=2)
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel("Thread Count")
        ax.set_ylabel("Efficiency")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.3)
    
    fig.suptitle("Multi-Threading Scaling Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    output_path = output_dir / "combined_summary.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def generate_all_plots(results_path: Path, output_dir: Path):
    """Generate all Series 7 plots."""
    
    results = load_results(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating Series 7 Scaling Plots...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    plot_throughput_scaling(results, output_dir)
    plot_speedup(results, output_dir)
    plot_efficiency(results, output_dir)
    plot_max_threads_comparison(results, output_dir)
    plot_combined_summary(results, output_dir)
    
    print("-" * 50)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Plot Series 7 scaling results")
    parser.add_argument("--results", type=str,
                        default="results/series7_scaling/series7_scaling_results.json",
                        help="Path to results JSON file")
    parser.add_argument("--output", type=str,
                        default="results/series7_scaling/plots",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    results_path = script_dir / args.results
    output_dir = script_dir / args.output
    
    generate_all_plots(results_path, output_dir)


if __name__ == "__main__":
    main()
