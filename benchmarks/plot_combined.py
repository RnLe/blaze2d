#!/usr/bin/env python3
"""
Combined Benchmark Analysis and Plotting

Generates comprehensive analysis plots combining single-core and multi-core results.

Output:
  results/plots/
    benchmark_overview.png         - Complete overview figure
    scaling_comparison.png         - Single vs multi-core scaling
    detailed_report.png            - Detailed per-config analysis
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'blaze2d': '#3498DB',           # Blue
    'mpb': '#E74C3C',               # Red (generic MPB)
    'mpb_single': '#E74C3C',        # Red
    'mpb_multiprocess': '#C0392B',  # Dark Red
    'mpb_openblas': '#F39C12',      # Orange
    'speedup': '#2ECC71',           # Green
}

SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

CONFIG_LABELS = {
    'config_a_tm': 'Square TM',
    'config_a_te': 'Square TE',
    'config_b_tm': 'Hex TM',
    'config_b_te': 'Hex TE',
}

CONFIG_ORDER = ['config_a_tm', 'config_a_te', 'config_b_tm', 'config_b_te']


def load_all_results(results_dir: Path) -> dict:
    """Load results from both single-core and multi-core directories."""
    results = {
        'single_core': {
            'mpb': {},
            'blaze2d': {},
        },
        'multi_core': {
            'blaze2d': {},
            'mpb_multiprocess': {},
            'mpb_openblas': {},
        }
    }
    
    # Load single-core results
    single_dir = results_dir / "single_core"
    if single_dir.exists():
        for config in CONFIG_ORDER:
            mpb_file = single_dir / f"mpb_{config}.json"
            if mpb_file.exists():
                with open(mpb_file) as f:
                    results['single_core']['mpb'][config] = json.load(f)
            
            blaze_file = single_dir / f"blaze2d_{config}.json"
            if blaze_file.exists():
                with open(blaze_file) as f:
                    results['single_core']['blaze2d'][config] = json.load(f)
    
    # Load multi-core results
    multi_dir = results_dir / "multi_core"
    if multi_dir.exists():
        for config in CONFIG_ORDER:
            blaze_file = multi_dir / f"blaze2d_{config}.json"
            if blaze_file.exists():
                with open(blaze_file) as f:
                    results['multi_core']['blaze2d'][config] = json.load(f)
            
            mp_file = multi_dir / f"mpb_multiprocess_{config}.json"
            if mp_file.exists():
                with open(mp_file) as f:
                    results['multi_core']['mpb_multiprocess'][config] = json.load(f)
            
            ob_file = multi_dir / f"mpb_openblas_{config}.json"
            if ob_file.exists():
                with open(ob_file) as f:
                    results['multi_core']['mpb_openblas'][config] = json.load(f)
    
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


def plot_overview(results: dict, output_dir: Path):
    """Create comprehensive overview figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get available configs for single-core
    sc_mpb = results['single_core']['mpb']
    sc_blaze = results['single_core']['blaze2d']
    single_configs = [c for c in CONFIG_ORDER if c in sc_mpb and c in sc_blaze]
    
    # Get available configs for multi-core
    mc_blaze = results['multi_core']['blaze2d']
    mc_mp = results['multi_core']['mpb_multiprocess']
    mc_ob = results['multi_core']['mpb_openblas']
    multi_configs = [c for c in CONFIG_ORDER 
                     if c in mc_blaze and (c in mc_mp or c in mc_ob)]
    
    # Panel 1: Single-core time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if single_configs:
        x = np.arange(len(single_configs))
        width = 0.35
        
        mpb_means = [sc_mpb[c]['mean_ms'] for c in single_configs]
        blaze_means = [sc_blaze[c]['mean_ms'] for c in single_configs]
        
        ax1.bar(x - width/2, mpb_means, width, label='MPB', color=COLORS['mpb'], alpha=0.8)
        ax1.bar(x + width/2, blaze_means, width, label='Blaze2D', color=COLORS['blaze2d'], alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Single-Core: Time per Job', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([CONFIG_LABELS[c] for c in single_configs], fontsize=9)
        ax1.legend(fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No single-core data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Single-Core: Time per Job', fontweight='bold')
    
    # Panel 2: Single-core speedup
    ax2 = fig.add_subplot(gs[0, 1])
    if single_configs:
        speedups = []
        for c in single_configs:
            sp, _ = compute_speedup(sc_mpb[c]['mean_ms'], sc_mpb[c]['std_ms'],
                                    sc_blaze[c]['mean_ms'], sc_blaze[c]['std_ms'])
            speedups.append(sp)
        
        bars = ax2.bar(range(len(single_configs)), speedups, color=COLORS['speedup'], alpha=0.8)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xticks(range(len(single_configs)))
        ax2.set_xticklabels([CONFIG_LABELS[c] for c in single_configs], fontsize=9)
        ax2.set_ylabel('Speedup')
        ax2.set_title('Single-Core: Blaze2D Speedup', fontweight='bold')
        ax2.set_ylim(bottom=0)
        
        for bar, sp in zip(bars, speedups):
            ax2.annotate(f'{sp:.2f}×', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No single-core data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Single-Core: Blaze2D Speedup', fontweight='bold')
    
    # Panel 3: Multi-core throughput
    ax3 = fig.add_subplot(gs[0, 2])
    if multi_configs:
        x = np.arange(len(multi_configs))
        width = 0.25
        
        solvers = []
        if any(c in mc_blaze for c in multi_configs):
            solvers.append(('blaze2d', mc_blaze, 'Blaze2D'))
        if any(c in mc_mp for c in multi_configs):
            solvers.append(('mpb_multiprocess', mc_mp, 'MPB MP'))
        if any(c in mc_ob for c in multi_configs):
            solvers.append(('mpb_openblas', mc_ob, 'MPB OB'))
        
        offsets = np.linspace(-width, width, len(solvers))
        
        for i, (key, data, label) in enumerate(solvers):
            throughputs = []
            for c in multi_configs:
                if c in data:
                    d = data[c]
                    throughputs.append(d.get('mean_throughput_jobs_s', 1000/d['mean_ms']))
                else:
                    throughputs.append(0)
            ax3.bar(x + offsets[i], throughputs, width, label=label, color=COLORS[key], alpha=0.8)
        
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Throughput (jobs/s)')
        ax3.set_title('Multi-Core: Throughput', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([CONFIG_LABELS[c] for c in multi_configs], fontsize=9)
        ax3.legend(fontsize=8)
        ax3.set_ylim(bottom=0)
    else:
        ax3.text(0.5, 0.5, 'No multi-core data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Multi-Core: Throughput', fontweight='bold')
    
    # Panel 4: Average speedups summary
    ax4 = fig.add_subplot(gs[1, 0])
    
    avg_speedups = []
    labels = []
    colors = []
    
    # Single-core speedup
    if single_configs:
        speedups = []
        for c in single_configs:
            sp, _ = compute_speedup(sc_mpb[c]['mean_ms'], sc_mpb[c]['std_ms'],
                                    sc_blaze[c]['mean_ms'], sc_blaze[c]['std_ms'])
            speedups.append(sp)
        avg_speedups.append(np.mean(speedups))
        labels.append('Single-Core\nvs MPB')
        colors.append(COLORS['mpb'])
    
    # Multi-core speedup vs multiprocess
    if multi_configs and mc_mp:
        speedups = []
        for c in multi_configs:
            if c in mc_blaze and c in mc_mp:
                sp, _ = compute_speedup(mc_mp[c]['mean_ms'], mc_mp[c]['std_ms'],
                                        mc_blaze[c]['mean_ms'], mc_blaze[c]['std_ms'])
                speedups.append(sp)
        if speedups:
            avg_speedups.append(np.mean(speedups))
            labels.append('Multi-Core\nvs MPB MP')
            colors.append(COLORS['mpb_multiprocess'])
    
    # Multi-core speedup vs openblas
    if multi_configs and mc_ob:
        speedups = []
        for c in multi_configs:
            if c in mc_blaze and c in mc_ob:
                sp, _ = compute_speedup(mc_ob[c]['mean_ms'], mc_ob[c]['std_ms'],
                                        mc_blaze[c]['mean_ms'], mc_blaze[c]['std_ms'])
                speedups.append(sp)
        if speedups:
            avg_speedups.append(np.mean(speedups))
            labels.append('Multi-Core\nvs MPB OB')
            colors.append(COLORS['mpb_openblas'])
    
    if avg_speedups:
        bars = ax4.bar(range(len(avg_speedups)), avg_speedups, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1)
        ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, fontsize=9)
        ax4.set_ylabel('Average Speedup')
        ax4.set_title('Blaze2D Average Speedup Summary', fontweight='bold')
        ax4.set_ylim(bottom=0)
        
        for bar, sp in zip(bars, avg_speedups):
            ax4.annotate(f'{sp:.2f}×', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Blaze2D Average Speedup Summary', fontweight='bold')
    
    # Panel 5: Scaling (single vs multi for blaze2d)
    ax5 = fig.add_subplot(gs[1, 1])
    
    scaling_configs = [c for c in CONFIG_ORDER 
                       if c in sc_blaze and c in mc_blaze]
    
    if scaling_configs:
        x = np.arange(len(scaling_configs))
        width = 0.35
        
        single_times = [sc_blaze[c]['mean_ms'] for c in scaling_configs]
        multi_times = [mc_blaze[c]['mean_ms'] for c in scaling_configs]
        
        ax5.bar(x - width/2, single_times, width, label='Single-Core', color='#85C1E9', alpha=0.8)
        ax5.bar(x + width/2, multi_times, width, label='Multi-Core (16)', color=COLORS['blaze2d'], alpha=0.8)
        
        ax5.set_xlabel('Configuration')
        ax5.set_ylabel('Effective Time per Job (ms)')
        ax5.set_title('Blaze2D: Single vs Multi-Core', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([CONFIG_LABELS[c] for c in scaling_configs], fontsize=9)
        ax5.legend(fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No scaling data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Blaze2D: Single vs Multi-Core', fontweight='bold')
    
    # Panel 6: Info/stats box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    info_text = "Benchmark Configuration\n" + "=" * 30 + "\n"
    info_text += f"Resolution: 64×64\n"
    info_text += f"Bands: 8\n"
    info_text += f"K-points: 61 (20/segment)\n"
    info_text += f"Tolerance: 1e-7\n\n"
    
    info_text += "Solver Modes\n" + "=" * 30 + "\n"
    info_text += "• Blaze2D: Rayon parallel (16 threads)\n"
    info_text += "• MPB MP: 16 subprocesses (single-threaded)\n"
    info_text += "• MPB OB: OpenBLAS multi-threading\n\n"
    
    # Add timestamp
    info_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Blaze2D vs MPB: Comprehensive Benchmark Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_file = output_dir / "benchmark_overview.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_per_config_detail(results: dict, output_dir: Path):
    """Create detailed per-configuration comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    sc_mpb = results['single_core']['mpb']
    sc_blaze = results['single_core']['blaze2d']
    mc_blaze = results['multi_core']['blaze2d']
    mc_mp = results['multi_core']['mpb_multiprocess']
    mc_ob = results['multi_core']['mpb_openblas']
    
    for idx, config in enumerate(CONFIG_ORDER):
        ax = axes[idx]
        
        data = []
        labels = []
        colors = []
        
        # Single-core
        if config in sc_mpb:
            data.append(sc_mpb[config]['mean_ms'])
            labels.append('MPB\n(1 core)')
            colors.append(COLORS['mpb'])
        
        if config in sc_blaze:
            data.append(sc_blaze[config]['mean_ms'])
            labels.append('Blaze2D\n(1 core)')
            colors.append('#85C1E9')
        
        # Multi-core
        if config in mc_ob:
            data.append(mc_ob[config]['mean_ms'])
            labels.append('MPB OB\n(multi)')
            colors.append(COLORS['mpb_openblas'])
        
        if config in mc_mp:
            data.append(mc_mp[config]['mean_ms'])
            labels.append('MPB MP\n(16 proc)')
            colors.append(COLORS['mpb_multiprocess'])
        
        if config in mc_blaze:
            data.append(mc_blaze[config]['mean_ms'])
            labels.append('Blaze2D\n(16 thr)')
            colors.append(COLORS['blaze2d'])
        
        if data:
            bars = ax.bar(range(len(data)), data, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Time per Job (ms)')
            
            # Add value labels
            for bar, val in zip(bars, data):
                ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        
        ax.set_title(f'{CONFIG_LABELS[config]}', fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)
    
    fig.suptitle('Per-Configuration Benchmark Details', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / "detailed_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def print_full_summary(results: dict):
    """Print comprehensive summary to console."""
    sc_mpb = results['single_core']['mpb']
    sc_blaze = results['single_core']['blaze2d']
    mc_blaze = results['multi_core']['blaze2d']
    mc_mp = results['multi_core']['mpb_multiprocess']
    mc_ob = results['multi_core']['mpb_openblas']
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Single-core section
    print("\n--- SINGLE-CORE COMPARISON ---")
    print(f"{'Config':<12} {'MPB (ms)':<16} {'Blaze2D (ms)':<16} {'Speedup':<10}")
    print("-" * 60)
    
    single_speedups = []
    for config in CONFIG_ORDER:
        if config in sc_mpb and config in sc_blaze:
            mpb = sc_mpb[config]
            blaze = sc_blaze[config]
            sp, _ = compute_speedup(mpb['mean_ms'], mpb['std_ms'],
                                    blaze['mean_ms'], blaze['std_ms'])
            single_speedups.append(sp)
            print(f"{CONFIG_LABELS[config]:<12} {mpb['mean_ms']:>6.1f} ± {mpb['std_ms']:<6.1f} "
                  f"{blaze['mean_ms']:>6.1f} ± {blaze['std_ms']:<6.1f} {sp:>5.2f}×")
    
    if single_speedups:
        print(f"{'Average':<12} {'':<16} {'':<16} {np.mean(single_speedups):>5.2f}×")
    
    # Multi-core section
    print("\n--- MULTI-CORE COMPARISON ---")
    print(f"{'Config':<12} {'Blaze2D':<14} {'MPB MP':<14} {'MPB OB':<14} {'Speedup vs MP':<12}")
    print("-" * 70)
    
    multi_speedups = []
    for config in CONFIG_ORDER:
        blaze_str = "N/A"
        mp_str = "N/A"
        ob_str = "N/A"
        sp_str = "N/A"
        
        if config in mc_blaze:
            blaze_str = f"{mc_blaze[config]['mean_ms']:.1f}"
        if config in mc_mp:
            mp_str = f"{mc_mp[config]['mean_ms']:.1f}"
        if config in mc_ob:
            ob_str = f"{mc_ob[config]['mean_ms']:.1f}"
        
        if config in mc_blaze and config in mc_mp:
            sp, _ = compute_speedup(mc_mp[config]['mean_ms'], mc_mp[config]['std_ms'],
                                    mc_blaze[config]['mean_ms'], mc_blaze[config]['std_ms'])
            sp_str = f"{sp:.2f}×"
            multi_speedups.append(sp)
        
        print(f"{CONFIG_LABELS[config]:<12} {blaze_str:<14} {mp_str:<14} {ob_str:<14} {sp_str:<12}")
    
    if multi_speedups:
        print(f"{'Average':<12} {'':<14} {'':<14} {'':<14} {np.mean(multi_speedups):.2f}×")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Combined Benchmark Analysis")
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory with results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()
    
    results_dir = Path(args.input) if args.input else RESULTS_DIR
    output_dir = Path(args.output) if args.output else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    
    # Summary of loaded data
    print("\nLoaded data:")
    print(f"  Single-core MPB: {len(results['single_core']['mpb'])} configs")
    print(f"  Single-core Blaze2D: {len(results['single_core']['blaze2d'])} configs")
    print(f"  Multi-core Blaze2D: {len(results['multi_core']['blaze2d'])} configs")
    print(f"  Multi-core MPB Multiprocess: {len(results['multi_core']['mpb_multiprocess'])} configs")
    print(f"  Multi-core MPB OpenBLAS: {len(results['multi_core']['mpb_openblas'])} configs")
    
    # Generate plots
    print(f"\nGenerating plots to: {output_dir}")
    plot_overview(results, output_dir)
    plot_per_config_detail(results, output_dir)
    
    # Print summary
    print_full_summary(results)


if __name__ == "__main__":
    main()
