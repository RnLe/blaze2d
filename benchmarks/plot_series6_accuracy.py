#!/usr/bin/env python3
"""
Plot Series 6: Accuracy Comparison Plots

This script generates:
1. Band diagrams: MPB (lines), Blaze-f64 (big dots), Blaze-f32 (small dots on top)
2. Deviation statistics: histograms and boxplots
3. Per-k-point deviation profiles

Output: results/series6_accuracy/plots/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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

# Colors
POL_COLORS = {
    "TE": "#e41a1c",  # Red
    "TM": "#377eb8",  # Blue
}

MPB_COLOR = "#666666"  # Gray
F64_COLOR = "#2ca02c"  # Green
F32_COLOR = "#ff7f0e"  # Orange

# Markers
F64_MARKER_SIZE = 40
F32_MARKER_SIZE = 15


def load_results(results_path: Path) -> Dict:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def get_node_positions_square(k_points: List[Dict]) -> Tuple[List[float], List[str]]:
    """
    Get high-symmetry node positions for square lattice path.
    Path: Γ → X → M → Γ
    """
    labels = ["Γ", "X", "M", "Γ"]
    
    # Find k-points near high-symmetry points
    node_positions = []
    prev_dist = 0.0
    
    for i, kp in enumerate(k_points):
        kx, ky = kp["k_frac"][0], kp["k_frac"][1]
        dist = kp["k_distance"]
        
        is_gamma = abs(kx) < 0.01 and abs(ky) < 0.01
        is_x = abs(kx - 0.5) < 0.01 and abs(ky) < 0.01
        is_m = abs(kx - 0.5) < 0.01 and abs(ky - 0.5) < 0.01
        
        if i == 0:
            node_positions.append(dist)
        elif is_x and len(node_positions) == 1:
            node_positions.append(dist)
        elif is_m and len(node_positions) == 2:
            node_positions.append(dist)
        elif i == len(k_points) - 1:
            node_positions.append(dist)
    
    return node_positions, labels


def plot_band_comparison(
    results: Dict,
    output_dir: Path,
) -> None:
    """
    Plot band diagrams with layered precision markers.
    
    MPB: lines (gray)
    Blaze-f64: big dots (green)
    Blaze-f32: small dots on top (orange)
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for row_idx, pol in enumerate(["TM", "TE"]):
        ax = axes[row_idx]
        pol_results = results["results"].get(pol, {})
        
        if not pol_results:
            ax.text(0.5, 0.5, f"No data for {pol}", transform=ax.transAxes,
                   ha='center', va='center')
            continue
        
        # Get k-point data
        mpb_kpts = pol_results.get("mpb", {}).get("k_points", [])
        f32_kpts = pol_results.get("blaze_f32", {}).get("k_points", [])
        f64_kpts = pol_results.get("blaze_f64", {}).get("k_points", [])
        
        if not mpb_kpts:
            continue
        
        # Get high-symmetry point positions
        node_positions, node_labels = get_node_positions_square(mpb_kpts)
        
        # Extract arrays
        mpb_dists = [kp["k_distance"] for kp in mpb_kpts]
        mpb_bands = np.array([kp["frequencies"] for kp in mpb_kpts])  # (num_k, num_bands)
        
        # Plot MPB as gray lines (reference)
        for band_idx in range(mpb_bands.shape[1]):
            label = "MPB (reference)" if band_idx == 0 else None
            ax.plot(
                mpb_dists,
                mpb_bands[:, band_idx],
                color=MPB_COLOR,
                linewidth=1.5,
                alpha=0.8,
                label=label,
                zorder=1,
            )
        
        # Plot Blaze-f64 as big green dots
        if f64_kpts:
            f64_dists = [kp["k_distance"] for kp in f64_kpts]
            f64_bands = np.array([kp["frequencies"] for kp in f64_kpts])
            
            for band_idx in range(f64_bands.shape[1]):
                label = "Blaze f64 (tol=1e-7)" if band_idx == 0 else None
                ax.scatter(
                    f64_dists,
                    f64_bands[:, band_idx],
                    color=F64_COLOR,
                    s=F64_MARKER_SIZE,
                    marker="o",
                    alpha=0.7,
                    label=label,
                    zorder=2,
                )
        
        # Plot Blaze-f32 as small orange dots on top
        if f32_kpts:
            f32_dists = [kp["k_distance"] for kp in f32_kpts]
            f32_bands = np.array([kp["frequencies"] for kp in f32_kpts])
            
            for band_idx in range(f32_bands.shape[1]):
                label = "Blaze f32 (tol=1e-4)" if band_idx == 0 else None
                ax.scatter(
                    f32_dists,
                    f32_bands[:, band_idx],
                    color=F32_COLOR,
                    s=F32_MARKER_SIZE,
                    marker="o",
                    alpha=0.9,
                    edgecolors='none',
                    label=label,
                    zorder=3,
                )
        
        # Add high-symmetry markers
        if node_positions:
            ax.set_xticks(node_positions)
            ax.set_xticklabels(node_labels, fontsize=11)
            for pos in node_positions:
                ax.axvline(pos, color="0.85", linewidth=0.8, zorder=0)
        
        ax.set_ylabel("ωa / 2πc")
        ax.set_title(f"{pol} Polarization", fontsize=12, fontweight='bold')
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    axes[-1].set_xlabel("k-path")
    fig.suptitle(
        "Band Diagram: MPB (lines) vs Blaze (dots)\n"
        "Large green dots = f64 precision, small orange dots = mixed f32 precision",
        fontsize=13
    )
    fig.tight_layout()
    
    output_path = output_dir / "band_comparison.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_deviation_histograms(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot histograms of relative deviations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for row_idx, pol in enumerate(["TM", "TE"]):
        pol_results = results["results"].get(pol, {})
        
        # f32 vs MPB
        ax_f32 = axes[row_idx, 0]
        f32_devs = pol_results.get("f32_vs_mpb", [])
        if f32_devs:
            rel_devs = [d["rel_deviation"] for d in f32_devs]
            ax_f32.hist(rel_devs, bins=50, color=F32_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_f32.axvline(np.mean(rel_devs), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(rel_devs):.2e}')
            ax_f32.axvline(np.max(rel_devs), color='darkred', linestyle=':', linewidth=2,
                          label=f'Max: {np.max(rel_devs):.2e}')
        ax_f32.set_xlabel("Relative Deviation")
        ax_f32.set_ylabel("Count")
        ax_f32.set_title(f"{pol}: Blaze-f32 vs MPB")
        ax_f32.legend(fontsize=8)
        ax_f32.set_yscale('log')
        
        # f64 vs MPB
        ax_f64 = axes[row_idx, 1]
        f64_devs = pol_results.get("f64_vs_mpb", [])
        if f64_devs:
            rel_devs = [d["rel_deviation"] for d in f64_devs]
            ax_f64.hist(rel_devs, bins=50, color=F64_COLOR, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_f64.axvline(np.mean(rel_devs), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(rel_devs):.2e}')
            ax_f64.axvline(np.max(rel_devs), color='darkred', linestyle=':', linewidth=2,
                          label=f'Max: {np.max(rel_devs):.2e}')
        ax_f64.set_xlabel("Relative Deviation")
        ax_f64.set_ylabel("Count")
        ax_f64.set_title(f"{pol}: Blaze-f64 vs MPB")
        ax_f64.legend(fontsize=8)
        ax_f64.set_yscale('log')
    
    fig.suptitle("Eigenvalue Deviation Histograms (after Hungarian band matching)", fontsize=13)
    fig.tight_layout()
    
    output_path = output_dir / "deviation_histograms.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_deviation_boxplots(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot boxplots comparing f32 vs f64 deviation distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for col_idx, pol in enumerate(["TM", "TE"]):
        ax = axes[col_idx]
        pol_results = results["results"].get(pol, {})
        
        data = []
        labels = []
        colors = []
        
        # f32 vs MPB
        f32_devs = pol_results.get("f32_vs_mpb", [])
        if f32_devs:
            data.append([d["rel_deviation"] for d in f32_devs])
            labels.append("f32 vs MPB")
            colors.append(F32_COLOR)
        
        # f64 vs MPB
        f64_devs = pol_results.get("f64_vs_mpb", [])
        if f64_devs:
            data.append([d["rel_deviation"] for d in f64_devs])
            labels.append("f64 vs MPB")
            colors.append(F64_COLOR)
        
        # f32 vs f64
        f32_vs_f64 = pol_results.get("f32_vs_f64", [])
        if f32_vs_f64:
            data.append([d["rel_deviation"] for d in f32_vs_f64])
            labels.append("f32 vs f64")
            colors.append("#9467bd")  # Purple
        
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel("Relative Deviation")
        ax.set_title(f"{pol} Polarization", fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Relative Deviation Distribution Comparison", fontsize=13)
    fig.tight_layout()
    
    output_path = output_dir / "deviation_boxplots.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_per_kpoint_deviation(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot per-k-point average deviation."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for row_idx, pol in enumerate(["TM", "TE"]):
        ax = axes[row_idx]
        pol_results = results["results"].get(pol, {})
        
        mpb_kpts = pol_results.get("mpb", {}).get("k_points", [])
        if not mpb_kpts:
            continue
        
        # Get k-distances
        k_dists = [kp["k_distance"] for kp in mpb_kpts]
        num_k = len(k_dists)
        
        # Aggregate deviations per k-point
        f32_per_k = {i: [] for i in range(num_k)}
        f64_per_k = {i: [] for i in range(num_k)}
        
        for d in pol_results.get("f32_vs_mpb", []):
            k_idx = d["k_index"]
            if k_idx < num_k:
                f32_per_k[k_idx].append(d["rel_deviation"])
        
        for d in pol_results.get("f64_vs_mpb", []):
            k_idx = d["k_index"]
            if k_idx < num_k:
                f64_per_k[k_idx].append(d["rel_deviation"])
        
        # Compute means
        f32_means = [np.mean(f32_per_k[i]) if f32_per_k[i] else 0 for i in range(num_k)]
        f64_means = [np.mean(f64_per_k[i]) if f64_per_k[i] else 0 for i in range(num_k)]
        
        # Plot
        if any(f32_means):
            ax.semilogy(k_dists, f32_means, '-o', color=F32_COLOR, markersize=4,
                       label='Blaze-f32 vs MPB', alpha=0.8)
        if any(f64_means):
            ax.semilogy(k_dists, f64_means, '-s', color=F64_COLOR, markersize=4,
                       label='Blaze-f64 vs MPB', alpha=0.8)
        
        # High-symmetry markers
        node_positions, node_labels = get_node_positions_square(mpb_kpts)
        if node_positions:
            ax.set_xticks(node_positions)
            ax.set_xticklabels(node_labels, fontsize=11)
            for pos in node_positions:
                ax.axvline(pos, color="0.85", linewidth=0.8, zorder=0)
        
        ax.set_ylabel("Mean Relative Deviation")
        ax.set_title(f"{pol} Polarization", fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which='both')
    
    axes[-1].set_xlabel("k-path")
    fig.suptitle("Per-k-point Mean Relative Deviation", fontsize=13)
    fig.tight_layout()
    
    output_path = output_dir / "per_kpoint_deviation.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_summary_bars(
    results: Dict,
    output_dir: Path,
) -> None:
    """Plot summary bar chart of mean and max deviations."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for col_idx, metric in enumerate(["mean", "max"]):
        ax = axes[col_idx]
        
        data = {"TM": {}, "TE": {}}
        
        for pol in ["TM", "TE"]:
            pol_results = results["results"].get(pol, {})
            
            f32_devs = [d["rel_deviation"] for d in pol_results.get("f32_vs_mpb", [])]
            f64_devs = [d["rel_deviation"] for d in pol_results.get("f64_vs_mpb", [])]
            
            if metric == "mean":
                data[pol]["f32"] = np.mean(f32_devs) if f32_devs else 0
                data[pol]["f64"] = np.mean(f64_devs) if f64_devs else 0
            else:
                data[pol]["f32"] = np.max(f32_devs) if f32_devs else 0
                data[pol]["f64"] = np.max(f64_devs) if f64_devs else 0
        
        # Bar positions
        x = np.arange(2)
        width = 0.35
        
        f32_vals = [data["TM"]["f32"], data["TE"]["f32"]]
        f64_vals = [data["TM"]["f64"], data["TE"]["f64"]]
        
        bars1 = ax.bar(x - width/2, f32_vals, width, label='Blaze-f32', color=F32_COLOR, alpha=0.8)
        bars2 = ax.bar(x + width/2, f64_vals, width, label='Blaze-f64', color=F64_COLOR, alpha=0.8)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xticks(x)
        ax.set_xticklabels(["TM", "TE"])
        ax.set_ylabel("Relative Deviation")
        ax.set_title(f"{metric.capitalize()} Relative Deviation vs MPB", fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Accuracy Summary: Blaze vs MPB Reference", fontsize=13)
    fig.tight_layout()
    
    output_path = output_dir / "summary_bars.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def generate_all_plots(results_path: Path, output_dir: Path):
    """Generate all Series 6 plots."""
    
    results = load_results(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating Series 6 Accuracy Plots...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    plot_band_comparison(results, output_dir)
    plot_deviation_histograms(results, output_dir)
    plot_deviation_boxplots(results, output_dir)
    plot_per_kpoint_deviation(results, output_dir)
    plot_summary_bars(results, output_dir)
    
    print("-" * 50)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Plot Series 6 accuracy results")
    parser.add_argument("--results", type=str, 
                        default="results/series6_accuracy/series6_accuracy_results.json",
                        help="Path to results JSON file")
    parser.add_argument("--output", type=str,
                        default="results/series6_accuracy/plots",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    results_path = script_dir / args.results
    output_dir = script_dir / args.output
    
    generate_all_plots(results_path, output_dir)


if __name__ == "__main__":
    main()
