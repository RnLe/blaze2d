#!/usr/bin/env python3
"""
Analyze Speed Benchmark Results - Compare MPB vs Blaze2D

Supports both single-core and multi-core benchmark results.
Computes speedup factors with error propagation.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()

def load_results(results_dir: Path, solver: str, core_mode: str) -> dict:
    """Load results JSON for a solver and core mode."""
    results_file = results_dir / f"{solver}_speed_{core_mode}_results.json"
    if not results_file.exists():
        return None
    
    with open(results_file) as f:
        return json.load(f)

def compute_speedup(time1_mean: float, time1_std: float,
                    time2_mean: float, time2_std: float) -> tuple:
    """
    Compute speedup factor with error propagation.
    
    Speedup = time1 / time2
    Error propagation for ratio: σ(A/B) = (A/B) * sqrt((σA/A)² + (σB/B)²)
    """
    if time2_mean <= 0:
        return float('inf'), 0
    
    speedup = time1_mean / time2_mean
    
    rel_err_1 = time1_std / time1_mean if time1_mean > 0 else 0
    rel_err_2 = time2_std / time2_mean if time2_mean > 0 else 0
    
    rel_err_speedup = np.sqrt(rel_err_1**2 + rel_err_2**2)
    speedup_err = speedup * rel_err_speedup
    
    return speedup, speedup_err

def analyze_core_mode(results_dir: Path, core_mode: str) -> dict:
    """Analyze results for a specific core mode."""
    
    mpb_data = load_results(results_dir, "mpb", core_mode)
    blaze_data = load_results(results_dir, "blaze2d", core_mode)
    
    if mpb_data is None:
        print(f"  WARNING: MPB {core_mode}-core results not found")
        return None
    if blaze_data is None:
        print(f"  WARNING: Blaze2D {core_mode}-core results not found")
        return None
    
    print(f"\n  MPB timestamp: {mpb_data['timestamp']}")
    print(f"  Blaze2D timestamp: {blaze_data['timestamp']}")
    
    # Map configs between MPB and Blaze2D
    # MPB: configs["config_a"]["polarizations"]["tm"]
    # Blaze: configs["config_a_tm"]
    
    config_mappings = [
        ("config_a", "tm", "config_a_tm", "Square, ε=8.9 rods"),
        ("config_a", "te", "config_a_te", "Square, ε=8.9 rods"),
        ("config_b", "tm", "config_b_tm", "Hex, air holes"),
        ("config_b", "te", "config_b_te", "Hex, air holes"),
    ]
    
    comparisons = []
    
    print(f"\n  {'Config':<22} {'MPB (ms)':<15} {'Blaze2D (ms)':<15} {'Speedup':<12}")
    print("  " + "-" * 65)
    
    for mpb_config, pol, blaze_config, desc in config_mappings:
        # Get MPB data
        mpb_pol_data = mpb_data["configs"][mpb_config]["polarizations"][pol]
        mpb_mean = mpb_pol_data["mean_ms"]
        mpb_std = mpb_pol_data["std_ms"]
        
        # Get Blaze2D data
        blaze_cfg_data = blaze_data["configs"][blaze_config]
        blaze_mean = blaze_cfg_data["mean_ms"]
        blaze_std = blaze_cfg_data["std_ms"]
        
        # Compute speedup
        speedup, speedup_err = compute_speedup(mpb_mean, mpb_std, blaze_mean, blaze_std)
        
        comparisons.append({
            "config": desc,
            "polarization": pol.upper(),
            "mpb_mean_ms": mpb_mean,
            "mpb_std_ms": mpb_std,
            "blaze_mean_ms": blaze_mean,
            "blaze_std_ms": blaze_std,
            "speedup": speedup,
            "speedup_err": speedup_err,
        })
        
        print(f"  {desc + ' ' + pol.upper():<22} {mpb_mean:>6.1f} ± {mpb_std:<5.1f} "
              f"{blaze_mean:>6.1f} ± {blaze_std:<5.1f} "
              f"{speedup:>5.1f}× ± {speedup_err:.1f}")
    
    # Summary stats
    speedups = [c["speedup"] for c in comparisons]
    speedup_errs = [c["speedup_err"] for c in comparisons]
    
    return {
        "core_mode": core_mode,
        "mpb_timestamp": mpb_data["timestamp"],
        "blaze_timestamp": blaze_data["timestamp"],
        "comparisons": comparisons,
        "summary": {
            "mean_speedup": float(np.mean(speedups)),
            "min_speedup": float(np.min(speedups)),
            "max_speedup": float(np.max(speedups)),
            "combined_error": float(np.sqrt(np.sum(np.array(speedup_errs)**2)) / len(speedup_errs)),
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze Speed Benchmark Results")
    parser.add_argument("--output", type=str, default="results",
                        help="Results directory")
    args = parser.parse_args()
    
    results_dir = SCRIPT_DIR / args.output
    
    print("=" * 70)
    print("Speed Benchmark Analysis: MPB vs Blaze2D")
    print("=" * 70)
    
    all_results = {
        "benchmark": "speed_comparison",
        "timestamp": datetime.now().isoformat(),
        "modes": {}
    }
    
    # Analyze both core modes
    for core_mode in ["single", "multi"]:
        print(f"\n{'='*70}")
        print(f"{core_mode.upper()}-CORE MODE")
        print("=" * 70)
        
        mode_results = analyze_core_mode(results_dir, core_mode)
        if mode_results:
            all_results["modes"][core_mode] = mode_results
            
            summary = mode_results["summary"]
            print(f"\n  SUMMARY:")
            print(f"    Average Speedup: {summary['mean_speedup']:.1f}× ± {summary['combined_error']:.1f}")
            print(f"    Speedup Range:   {summary['min_speedup']:.1f}× - {summary['max_speedup']:.1f}×")
    
    # Save combined results
    if all_results["modes"]:
        output_file = results_dir / "speed_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nComparison saved to: {output_file}")
        
        # CSV summary
        csv_file = results_dir / "speed_comparison.csv"
        with open(csv_file, 'w') as f:
            f.write("core_mode,config,polarization,mpb_mean_ms,mpb_std_ms,blaze_mean_ms,blaze_std_ms,speedup,speedup_err\n")
            for mode, mode_data in all_results["modes"].items():
                for c in mode_data["comparisons"]:
                    f.write(f"{mode},{c['config']},{c['polarization']},"
                            f"{c['mpb_mean_ms']:.4f},{c['mpb_std_ms']:.4f},"
                            f"{c['blaze_mean_ms']:.4f},{c['blaze_std_ms']:.4f},"
                            f"{c['speedup']:.2f},{c['speedup_err']:.2f}\n")
        print(f"CSV saved to: {csv_file}")
    
    # Print markdown table
    print("\n" + "=" * 70)
    print("MARKDOWN TABLES")
    print("=" * 70)
    
    for mode, mode_data in all_results["modes"].items():
        print(f"\n### {mode.capitalize()}-Core Comparison\n")
        print("| Configuration | MPB (ms) | Blaze2D (ms) | Speedup |")
        print("|---------------|----------|--------------|---------|")
        for c in mode_data["comparisons"]:
            print(f"| {c['config']} {c['polarization']} | {c['mpb_mean_ms']:.1f} ± {c['mpb_std_ms']:.1f} | "
                  f"{c['blaze_mean_ms']:.1f} ± {c['blaze_std_ms']:.1f} | "
                  f"**{c['speedup']:.1f}×** |")
        s = mode_data["summary"]
        print(f"| **Average** | - | - | **{s['mean_speedup']:.1f}×** ± {s['combined_error']:.1f} |")

if __name__ == "__main__":
    main()
