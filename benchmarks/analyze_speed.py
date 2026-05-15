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
    """Analyze results for a specific core mode.
    
    If core_mode is 'multi', it attempts to load both 'multi-native' and 'multi-process'
    results for MPB.
    """
    
    mpb_data = None
    mpb_process_data = None
    
    if core_mode == "single":
        mpb_data = load_results(results_dir, "mpb", "single")
    elif core_mode == "multi":
        # Load both native and process if available
        # Note: bench_mpb_speed.py writes with tag if provided.
        # Check standard names first
        mpb_data = load_results(results_dir, "mpb", "multi-native")
        if not mpb_data:
             mpb_data = load_results(results_dir, "mpb", "multi") # Fallback
             
        mpb_process_data = load_results(results_dir, "mpb", "multi-process")
    
    blaze_data = load_results(results_dir, "blaze2d", core_mode)
    
    if mpb_data is None and mpb_process_data is None and blaze_data is None:
        print(f"  No results found for {core_mode}-core mode.")
        return None
    
    mpb_ts = mpb_data['timestamp'] if mpb_data else "N/A"
    mpb_proc_ts = mpb_process_data['timestamp'] if mpb_process_data else "N/A"
    blaze_ts = blaze_data['timestamp'] if blaze_data else "N/A"
    
    print(f"\n  MPB (Native) timestamp:  {mpb_ts}")
    if core_mode == "multi":
        print(f"  MPB (Process) timestamp: {mpb_proc_ts}")
    print(f"  Blaze2D timestamp:       {blaze_ts}")
    
    # Map configs
    config_mappings = [
        ("config_a", "tm", "config_a_tm", "Square, ε=8.9 rods"),
        ("config_a", "te", "config_a_te", "Square, ε=8.9 rods"),
        ("config_b", "tm", "config_b_tm", "Hex, air holes"),
        ("config_b", "te", "config_b_te", "Hex, air holes"),
    ]
    
    comparisons = []
    
    # Adjust header based on mode
    if core_mode == "single":
         print(f"\n  {'Config':<22} {'MPB (ms)':<15} {'Blaze2D (ms)':<15} {'Speedup':<12}")
    else:
         print(f"\n  {'Config':<22} {'MPB Nat(ms)':<15} {'MPB Proc(ms)':<15} {'Blaze2D(ms)':<15} {'Speedup(Proc)':<15}")

    print("  " + "-" * 85)
    
    for mpb_config, pol, blaze_config, desc in config_mappings:
        # Get MPB Native data
        mpb_mean, mpb_std = 0.0, 0.0
        if mpb_data:
            try:
                mpb_pol_data = mpb_data["configs"][mpb_config]["polarizations"][pol]
                mpb_mean = mpb_pol_data["mean_ms"]
                mpb_std = mpb_pol_data["std_ms"]
            except (KeyError, TypeError):
                pass

        # Get MPB Process data (Multi only)
        mpb_proc_mean, mpb_proc_std = 0.0, 0.0
        if mpb_process_data:
             try:
                mpb_pol_data = mpb_process_data["configs"][mpb_config]["polarizations"][pol]
                mpb_proc_mean = mpb_pol_data["mean_ms"]
                mpb_proc_std = mpb_pol_data["std_ms"]
             except (KeyError, TypeError):
                pass
        
        # Get Blaze2D data
        blaze_mean, blaze_std = 0.0, 0.0
        if blaze_data:
            try:
                blaze_cfg_data = blaze_data["configs"][blaze_config]
                blaze_mean = blaze_cfg_data["mean_ms"]
                blaze_std = blaze_cfg_data["std_ms"]
            except (KeyError, TypeError):
                pass
        
        # Compute speedups
        # Single core: Blaze vs MPB (Native/Single)
        # Multi core: Blaze vs MPB Process (Fair) AND Blaze vs MPB Native
        
        speedup_native, speedup_native_err = 0.0, 0.0
        if mpb_mean > 0 and blaze_mean > 0:
            speedup_native, speedup_native_err = compute_speedup(mpb_mean, mpb_std, blaze_mean, blaze_std)
            
        speedup_proc, speedup_proc_err = 0.0, 0.0
        if mpb_proc_mean > 0 and blaze_mean > 0:
            speedup_proc, speedup_proc_err = compute_speedup(mpb_proc_mean, mpb_proc_std, blaze_mean, blaze_std)
        
        # Primary speedup for summary
        if core_mode == "single":
            primary_speedup = speedup_native
            primary_err = speedup_native_err
        else:
            primary_speedup = speedup_proc # Fair comparison
            primary_err = speedup_proc_err

        comparisons.append({
            "config": desc,
            "polarization": pol.upper(),
            "mpb_mean_ms": mpb_mean,
            "mpb_std_ms": mpb_std,
            "mpb_proc_mean_ms": mpb_proc_mean,
            "mpb_proc_std_ms": mpb_proc_std,
            "blaze_mean_ms": blaze_mean,
            "blaze_std_ms": blaze_std,
            "speedup": primary_speedup,
            "speedup_err": primary_err,
            "speedup_native": speedup_native, # Extra info
            "speedup_native_err": speedup_native_err
        })
        
        mpb_str = f"{mpb_mean:>6.1f} ± {mpb_std:<5.1f}" if mpb_mean > 0 else "N/A"
        mpb_proc_str = f"{mpb_proc_mean:>6.1f} ± {mpb_proc_std:<5.1f}" if mpb_proc_mean > 0 else "N/A"
        blaze_str = f"{blaze_mean:>6.1f} ± {blaze_std:<5.1f}" if blaze_mean > 0 else "N/A"
        
        if core_mode == "single":
             speedup_str = f"{primary_speedup:>5.1f}× ± {primary_err:.1f}" if primary_speedup > 0 else "N/A"
             print(f"  {desc + ' ' + pol.upper():<22} {mpb_str:<15} {blaze_str:<15} {speedup_str}")
        else:
             # For multi, show speedup vs Process
             speedup_str = f"{primary_speedup:>5.1f}× ± {primary_err:.1f}" if primary_speedup > 0 else "N/A"
             print(f"  {desc + ' ' + pol.upper():<22} {mpb_str:<15} {mpb_proc_str:<15} {blaze_str:<15} {speedup_str}")

    # Summary stats
    valid_speedups = [c["speedup"] for c in comparisons if c["speedup"] > 0]
    valid_errs = [c["speedup_err"] for c in comparisons if c["speedup"] > 0]
    
    if valid_speedups:
        mean_speedup = float(np.mean(valid_speedups))
        min_speedup = float(np.min(valid_speedups))
        max_speedup = float(np.max(valid_speedups))
        combined_err = float(np.sqrt(np.sum(np.array(valid_errs)**2)) / len(valid_errs))
    else:
        mean_speedup = 0.0
        min_speedup = 0.0
        max_speedup = 0.0
        combined_err = 0.0
    
    return {
        "core_mode": core_mode,
        "mpb_timestamp": mpb_ts,
        "blaze_timestamp": blaze_ts,
        "comparisons": comparisons,
        "summary": {
            "mean_speedup": mean_speedup,
            "min_speedup": min_speedup,
            "max_speedup": max_speedup,
            "combined_error": combined_err,
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
        if mode == "single":
            print("| Configuration | MPB (ms) | Blaze2D (ms) | Speedup |")
            print("|---------------|----------|--------------|---------|")
            for c in mode_data["comparisons"]:
                print(f"| {c['config']} {c['polarization']} | {c['mpb_mean_ms']:.1f} ± {c['mpb_std_ms']:.1f} | "
                      f"{c['blaze_mean_ms']:.1f} ± {c['blaze_std_ms']:.1f} | "
                      f"**{c['speedup']:.1f}×** |")
        else:
            print("| Configuration | MPB Nat (ms) | MPB Proc (ms) | Blaze2D (ms) | Speedup (vs Proc) |")
            print("|---------------|--------------|---------------|--------------|-------------------|")
            for c in mode_data["comparisons"]:
                print(f"| {c['config']} {c['polarization']} | {c['mpb_mean_ms']:.1f} ± {c['mpb_std_ms']:.1f} | "
                      f"{c['mpb_proc_mean_ms']:.1f} ± {c['mpb_proc_std_ms']:.1f} | "
                      f"{c['blaze_mean_ms']:.1f} ± {c['blaze_std_ms']:.1f} | "
                      f"**{c['speedup']:.1f}×** |")

        s = mode_data["summary"]
        print(f"| **Average** | - | - | - | **{s['mean_speedup']:.1f}×** ± {s['combined_error']:.1f} |")

if __name__ == "__main__":
    main()
