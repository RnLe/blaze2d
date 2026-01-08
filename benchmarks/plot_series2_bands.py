#!/usr/bin/env python3
"""
Plot benchmark results for Series 2: Number of Bands Sweep

Creates comparison plots showing performance vs. number of bands for
MPB and Blaze2D solvers on both TM and TE polarizations.

Output: results/series2_bands/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Color scheme (consistent across all benchmarks)
COLORS = {
    "mpb": "#E74C3C",       # Red
    "blaze": "#3498DB",     # Blue
}

LABELS = {
    "mpb": "MPB",
    "blaze": "Blaze2D",
}

SCRIPT_DIR = Path(__file__).parent.absolute()


def load_results(results_dir: Path) -> dict:
    """Load Series 2 results from JSON file."""
    results_file = results_dir / "series2_bands_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    with open(results_file) as f:
        return json.load(f)


def plot_series2_combined(results: dict, output_dir: Path):
    """
    Create a combined figure with TM and TE side by side.
    Shows runtime vs number of bands for both solvers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, pol in zip(axes, ["TM", "TE"]):
        data = results[pol]
        bands = np.array(data["bands"])
        
        # Determine which solvers have data
        solvers = []
        for solver in ["mpb", "blaze"]:
            if data[solver] and any(d is not None for d in data[solver]):
                solvers.append(solver)
        
        # Bar width and positions
        n_solvers = len(solvers)
        bar_width = 0.8 / n_solvers
        
        for i, solver in enumerate(solvers):
            offset = (i - (n_solvers - 1) / 2) * bar_width
            
            means = []
            stds = []
            valid_bands = []
            
            for j, entry in enumerate(data[solver]):
                if entry is not None:
                    means.append(entry["mean"])
                    stds.append(entry["std"])
                    valid_bands.append(bands[j])
            
            if means:
                x_pos = np.arange(len(valid_bands)) + offset
                ax.bar(x_pos, means, bar_width * 0.9, yerr=stds,
                       color=COLORS[solver], label=LABELS[solver],
                       capsize=2, alpha=0.85)
        
        ax.set_xlabel("Number of Bands", fontsize=12)
        ax.set_title(f"{pol} Polarization", fontsize=14)
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xticklabels([str(b) for b in bands], fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel("Runtime (ms)", fontsize=12)
    
    # Shared legend
    handles = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) 
               for s in ["mpb", "blaze"]]
    fig.legend(handles=handles, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 0.02), fontsize=11)
    
    # Title with parameters
    params = results["parameters"]
    title = (f"Series 2: Runtime vs. Number of Bands\n"
             f"(Square lattice, ε={params['epsilon']}, r={params['radius']}a, "
             f"res={params['resolution']})")
    fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_file = output_dir / "series2_bands_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_series2_line(results: dict, output_dir: Path):
    """
    Create a line plot showing runtime scaling with number of bands.
    Better for seeing trends in how runtime grows.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, pol in zip(axes, ["TM", "TE"]):
        data = results[pol]
        bands = np.array(data["bands"])
        
        for solver in ["mpb", "blaze"]:
            if data[solver] and any(d is not None for d in data[solver]):
                means = []
                stds = []
                valid_bands = []
                
                for j, entry in enumerate(data[solver]):
                    if entry is not None:
                        means.append(entry["mean"])
                        stds.append(entry["std"])
                        valid_bands.append(bands[j])
                
                if means:
                    ax.errorbar(valid_bands, means, yerr=stds,
                               color=COLORS[solver], label=LABELS[solver],
                               marker='o', markersize=6, linewidth=2,
                               capsize=3, alpha=0.85)
        
        ax.set_xlabel("Number of Bands", fontsize=12)
        ax.set_title(f"{pol} Polarization", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    
    axes[0].set_ylabel("Runtime (ms)", fontsize=12)
    
    # Title with parameters
    params = results["parameters"]
    title = (f"Series 2: Runtime Scaling with Number of Bands\n"
             f"(Square lattice, ε={params['epsilon']}, r={params['radius']}a, "
             f"res={params['resolution']})")
    fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_file = output_dir / "series2_bands_line.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_series2_speedup(results: dict, output_dir: Path):
    """
    Create a speedup plot showing Blaze2D speedup over MPB.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, pol in zip(axes, ["TM", "TE"]):
        data = results[pol]
        bands = np.array(data["bands"])
        
        # Need MPB as baseline
        if not data["mpb"] or not any(d is not None for d in data["mpb"]):
            ax.text(0.5, 0.5, "MPB data required\nfor speedup", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        mpb_means = {bands[j]: entry["mean"] for j, entry in enumerate(data["mpb"]) 
                    if entry is not None}
        
        if data["blaze"] and any(d is not None for d in data["blaze"]):
            speedups = []
            valid_bands = []
            
            for j, entry in enumerate(data["blaze"]):
                if entry is not None and bands[j] in mpb_means:
                    speedup = mpb_means[bands[j]] / entry["mean"]
                    speedups.append(speedup)
                    valid_bands.append(bands[j])
            
            if speedups:
                ax.plot(valid_bands, speedups, 
                       color=COLORS["blaze"], label=LABELS["blaze"],
                       marker='o', markersize=6, linewidth=2, alpha=0.85)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel("Number of Bands", fontsize=12)
        ax.set_title(f"{pol} Polarization", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    
    axes[0].set_ylabel("Speedup over MPB", fontsize=12)
    
    # Title
    params = results["parameters"]
    title = (f"Series 2: Blaze2D Speedup vs. Number of Bands\n"
             f"(Square lattice, ε={params['epsilon']}, r={params['radius']}a, "
             f"res={params['resolution']})")
    fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_file = output_dir / "series2_bands_speedup.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_summary_table(results: dict):
    """Print a summary table of the results."""
    params = results["parameters"]
    print("\n" + "=" * 70)
    print("Series 2: Number of Bands Benchmark Summary")
    print("=" * 70)
    print(f"Parameters: ε={params['epsilon']}, r={params['radius']}a, "
          f"res={params['resolution']}")
    print("=" * 70)
    
    for pol in ["TM", "TE"]:
        print(f"\n{pol} Polarization:")
        print("-" * 60)
        print(f"{'Bands':>6} | {'MPB (ms)':>12} | {'Blaze2D (ms)':>12} | {'Speedup':>12}")
        print("-" * 60)
        
        data = results[pol]
        bands = data["bands"]
        
        for j in range(len(bands)):
            mpb_val = data["mpb"][j]["mean"] if data["mpb"][j] else float('nan')
            blaze_val = data["blaze"][j]["mean"] if data["blaze"][j] else float('nan')
            
            speedup = mpb_val / blaze_val if blaze_val and mpb_val else float('nan')
            
            print(f"{bands[j]:>6} | {mpb_val:>12.2f} | {blaze_val:>12.2f} | {speedup:>12.2f}x")


def create_pdf_report(results: dict, output_dir: Path):
    """Create a multi-page PDF report with all figures."""
    pdf_file = output_dir / "series2_bands_report.pdf"
    params = results["parameters"]
    
    with PdfPages(pdf_file) as pdf:
        # Page 1: Combined bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        for ax, pol in zip(axes, ["TM", "TE"]):
            data = results[pol]
            bands = np.array(data["bands"])
            
            solvers = []
            for solver in ["mpb", "blaze"]:
                if data[solver] and any(d is not None for d in data[solver]):
                    solvers.append(solver)
            
            n_solvers = len(solvers)
            bar_width = 0.8 / n_solvers
            
            for i, solver in enumerate(solvers):
                offset = (i - (n_solvers - 1) / 2) * bar_width
                
                means = []
                stds = []
                valid_bands = []
                
                for j, entry in enumerate(data[solver]):
                    if entry is not None:
                        means.append(entry["mean"])
                        stds.append(entry["std"])
                        valid_bands.append(bands[j])
                
                if means:
                    x_pos = np.arange(len(valid_bands)) + offset
                    ax.bar(x_pos, means, bar_width * 0.9, yerr=stds,
                           color=COLORS[solver], label=LABELS[solver],
                           capsize=2, alpha=0.85)
            
            ax.set_xlabel("Number of Bands", fontsize=12)
            ax.set_title(f"{pol} Polarization", fontsize=14)
            ax.set_xticks(np.arange(len(bands)))
            ax.set_xticklabels([str(b) for b in bands], fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        axes[0].set_ylabel("Runtime (ms)", fontsize=12)
        handles = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) 
                   for s in ["mpb", "blaze"]]
        fig.legend(handles=handles, loc='upper center', ncol=2, 
                   bbox_to_anchor=(0.5, 0.02), fontsize=11)
        
        title = (f"Series 2: Runtime vs. Number of Bands\n"
                 f"(Square lattice, ε={params['epsilon']}, r={params['radius']}a, "
                 f"res={params['resolution']})")
        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Line plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        for ax, pol in zip(axes, ["TM", "TE"]):
            data = results[pol]
            bands = np.array(data["bands"])
            
            for solver in ["mpb", "blaze"]:
                if data[solver] and any(d is not None for d in data[solver]):
                    means = []
                    stds = []
                    valid_bands = []
                    
                    for j, entry in enumerate(data[solver]):
                        if entry is not None:
                            means.append(entry["mean"])
                            stds.append(entry["std"])
                            valid_bands.append(bands[j])
                    
                    if means:
                        ax.errorbar(valid_bands, means, yerr=stds,
                                   color=COLORS[solver], label=LABELS[solver],
                                   marker='o', markersize=6, linewidth=2,
                                   capsize=3, alpha=0.85)
            
            ax.set_xlabel("Number of Bands", fontsize=12)
            ax.set_title(f"{pol} Polarization", fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
        
        axes[0].set_ylabel("Runtime (ms)", fontsize=12)
        title = (f"Series 2: Runtime Scaling with Number of Bands\n"
                 f"(Square lattice, ε={params['epsilon']}, r={params['radius']}a, "
                 f"res={params['resolution']})")
        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Speedup plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        for ax, pol in zip(axes, ["TM", "TE"]):
            data = results[pol]
            bands = np.array(data["bands"])
            
            if not data["mpb"] or not any(d is not None for d in data["mpb"]):
                ax.text(0.5, 0.5, "MPB data required\nfor speedup", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            mpb_means = {bands[j]: entry["mean"] for j, entry in enumerate(data["mpb"]) 
                        if entry is not None}
            
            if data["blaze"] and any(d is not None for d in data["blaze"]):
                speedups = []
                valid_bands = []
                
                for j, entry in enumerate(data["blaze"]):
                    if entry is not None and bands[j] in mpb_means:
                        speedup = mpb_means[bands[j]] / entry["mean"]
                        speedups.append(speedup)
                        valid_bands.append(bands[j])
                
                if speedups:
                    ax.plot(valid_bands, speedups, 
                           color=COLORS["blaze"], label=LABELS["blaze"],
                           marker='o', markersize=6, linewidth=2, alpha=0.85)
            
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel("Number of Bands", fontsize=12)
            ax.set_title(f"{pol} Polarization", fontsize=14)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
        
        axes[0].set_ylabel("Speedup over MPB", fontsize=12)
        title = (f"Series 2: Blaze2D Speedup vs. Number of Bands\n"
                 f"(Square lattice, ε={params['epsilon']}, r={params['radius']}a, "
                 f"res={params['resolution']})")
        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Benchmark Series 2: Number of Bands Sweep'
        d['Author'] = 'Blaze2D Benchmark Suite'
        d['Subject'] = 'MPB vs Blaze2D Bands Scaling Comparison'
        d['CreationDate'] = results.get('timestamp', '')
    
    print(f"Saved PDF report: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot Series 2 Benchmark Results")
    parser.add_argument("--results-dir", type=str, default="results/series2_bands",
                        help="Directory containing results JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as results)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table to console")
    args = parser.parse_args()
    
    results_dir = SCRIPT_DIR / args.results_dir
    output_dir = SCRIPT_DIR / (args.output_dir or args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    # Generate plots
    plot_series2_combined(results, output_dir)
    plot_series2_line(results, output_dir)
    plot_series2_speedup(results, output_dir)
    
    if args.summary:
        print_summary_table(results)
    
    # Generate PDF report
    create_pdf_report(results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
