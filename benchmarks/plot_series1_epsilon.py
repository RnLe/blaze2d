#!/usr/bin/env python3
"""
Plot Benchmark Series 1: Epsilon Sweep Results

Creates bar plots showing timing vs epsilon for TM and TE polarizations.
Compares MPB and Blaze2D.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).parent.absolute()

# Color scheme
COLORS = {
    "mpb": "#E74C3C",         # Red
    "blaze2d": "#3498DB",     # Blue
}


def load_results(results_dir: Path) -> dict:
    """Load series 1 results."""
    results_file = results_dir / "series1_epsilon_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}\nRun bench_series1_epsilon.py first.")
    
    with open(results_file) as f:
        return json.load(f)


def plot_polarization(ax, pol_data: dict, polarization: str, results: dict):
    """Plot timing vs epsilon for a single polarization."""
    
    epsilons = pol_data["epsilon"]
    mpb_data = pol_data["mpb"]
    blaze_data = pol_data["blaze"]
    
    # Determine which solvers have data
    has_mpb = any(d is not None for d in mpb_data)
    has_blaze = any(d is not None for d in blaze_data)
    
    n_solvers = sum([has_mpb, has_blaze])
    if n_solvers == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    x = np.arange(len(epsilons))
    width = 0.8 / n_solvers  # Distribute bar width
    
    # Extract means and stds for each solver
    def extract_data(data_list):
        means, stds = [], []
        for d in data_list:
            if d is not None:
                means.append(d["mean"])
                stds.append(d["std"])
            else:
                means.append(0)
                stds.append(0)
        return means, stds
    
    mpb_means, mpb_stds = extract_data(mpb_data)
    blaze_means, blaze_stds = extract_data(blaze_data)
    
    # Calculate bar positions
    bar_idx = 0
    offset_start = -width * (n_solvers - 1) / 2
    
    # Plot bars
    if has_mpb:
        offset = offset_start + bar_idx * width
        ax.bar(x + offset, mpb_means, width, yerr=mpb_stds,
               label='MPB', color=COLORS["mpb"], capsize=2, alpha=0.8)
        bar_idx += 1
    
    if has_blaze:
        offset = offset_start + bar_idx * width
        ax.bar(x + offset, blaze_means, width, yerr=blaze_stds,
               label='Blaze2D', color=COLORS["blaze2d"], capsize=2, alpha=0.8)
        bar_idx += 1
    
    # Labels and formatting
    ax.set_xlabel('Dielectric Constant (ε)')
    ax.set_ylabel('Time per job (ms)')
    ax.set_title(f'{polarization} Polarization: Time vs Epsilon')
    ax.set_xticks(x)
    # Show every other epsilon label to avoid crowding
    labels = [f'{e:.1f}' if i % 2 == 0 else '' for i, e in enumerate(epsilons)]
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_speedup(ax, pol_data: dict, polarization: str):
    """Plot speedup vs epsilon for a single polarization."""
    
    epsilons = pol_data["epsilon"]
    mpb_data = pol_data["mpb"]
    blaze_data = pol_data["blaze"]
    
    speedups = []
    valid_eps = []
    
    for i, (mpb, blaze) in enumerate(zip(mpb_data, blaze_data)):
        if mpb is not None and blaze is not None and blaze["mean"] > 0:
            speedups.append(mpb["mean"] / blaze["mean"])
            valid_eps.append(epsilons[i])
    
    if not speedups:
        ax.text(0.5, 0.5, 'No speedup data', ha='center', va='center', transform=ax.transAxes)
        return
    
    ax.plot(valid_eps, speedups, 'o-', color=COLORS["blaze2d"], linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dielectric Constant (ε)')
    ax.set_ylabel('Speedup (MPB / Blaze2D)')
    ax.set_title(f'{polarization}: Blaze2D Speedup vs Epsilon')
    ax.grid(alpha=0.3)


def create_combined_plot(results: dict, output_dir: Path):
    """Create combined plot with TM and TE side by side."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_polarization(axes[0], results["TM"], "TM", results)
    plot_polarization(axes[1], results["TE"], "TE", results)
    
    params = results["parameters"]
    plt.suptitle(f'Series 1: Epsilon Sweep on Square Lattice\n'
                 f'(r={params["radius"]}a, {params["resolution"]}×{params["resolution"]}, '
                 f'{params["num_bands"]} bands)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "series1_epsilon_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_speedup_plot(results: dict, output_dir: Path):
    """Create speedup plot."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_speedup(axes[0], results["TM"], "TM")
    plot_speedup(axes[1], results["TE"], "TE")
    
    params = results["parameters"]
    plt.suptitle(f'Series 1: Blaze2D Speedup vs Epsilon\n'
                 f'(Square lattice, r={params["radius"]}a, {params["resolution"]}×{params["resolution"]})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "series1_epsilon_speedup.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_individual_plots(results: dict, output_dir: Path):
    """Create individual plots for each polarization."""
    
    for pol in ["TM", "TE"]:
        # Timing plot
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_polarization(ax, results[pol], pol, results)
        plt.tight_layout()
        plt.savefig(output_dir / f"series1_epsilon_{pol.lower()}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Speedup plot
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_speedup(ax, results[pol], pol)
        plt.tight_layout()
        plt.savefig(output_dir / f"series1_epsilon_{pol.lower()}_speedup.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Individual plots saved to: {output_dir}")


def create_pdf_report(results: dict, output_dir: Path):
    """Create a multi-page PDF report with all figures."""
    pdf_file = output_dir / "series1_epsilon_report.pdf"
    params = results["parameters"]
    
    with PdfPages(pdf_file) as pdf:
        # Page 1: Combined timing comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_polarization(axes[0], results["TM"], "TM", results)
        plot_polarization(axes[1], results["TE"], "TE", results)
        plt.suptitle(f'Series 1: Epsilon Sweep on Square Lattice\n'
                     f'(r={params["radius"]}a, {params["resolution"]}×{params["resolution"]}, '
                     f'{params["num_bands"]} bands)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Speedup plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_speedup(axes[0], results["TM"], "TM")
        plot_speedup(axes[1], results["TE"], "TE")
        plt.suptitle(f'Series 1: Blaze2D Speedup vs Epsilon\n'
                     f'(Square lattice, r={params["radius"]}a, {params["resolution"]}×{params["resolution"]})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Benchmark Series 1: Epsilon Sweep'
        d['Author'] = 'Blaze2D Benchmark Suite'
        d['Subject'] = 'MPB vs Blaze2D Epsilon Scaling Comparison'
        d['CreationDate'] = results.get('timestamp', '')
    
    print(f"Saved PDF report: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot Series 1 Benchmark Results")
    parser.add_argument("--output", type=str, default="results/series1_epsilon",
                        help="Results directory")
    parser.add_argument("--individual", action="store_true",
                        help="Also generate individual plots")
    args = parser.parse_args()
    
    results_dir = SCRIPT_DIR / args.output
    
    print("=" * 70)
    print("Plotting Series 1: Epsilon Sweep Results")
    print("=" * 70)
    
    try:
        results = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Create plots
    create_combined_plot(results, results_dir)
    create_speedup_plot(results, results_dir)
    
    if args.individual:
        create_individual_plots(results, results_dir)
    
    # Generate PDF report
    create_pdf_report(results, results_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
