#!/usr/bin/env python3
"""
Run multiple profiling iterations and aggregate the results.

This script runs the blaze2d CLI multiple times with profiling enabled,
then combines the results into a single averaged profile JSON.

Usage:
    python profile_aggregate.py --config config.toml --output avg.json --iterations 10
"""

import json
import subprocess
import argparse
import tempfile
import os
from pathlib import Path
from typing import Dict, List
import statistics


def run_profile(cli_path: str, config: str, profile_output: str, extra_args: List[str] = None) -> Dict:
    """Run a single profiling iteration and return the profile data."""
    cmd = [cli_path, "--config", config, "--output", "/dev/null", "--profile", profile_output]
    if extra_args:
        cmd.extend(extra_args)
    
    # Run silently
    subprocess.run(cmd, check=True, capture_output=True)
    
    with open(profile_output, 'r') as f:
        return json.load(f)


def aggregate_profiles(profiles: List[Dict]) -> Dict:
    """
    Aggregate multiple profile runs into a single averaged profile.
    
    Returns:
        Dict with averaged times, proper min/max, and statistics.
    """
    if not profiles:
        return {}
    
    n = len(profiles)
    
    # Collect session totals
    session_totals = [p.get('session_total_ms', 0) for p in profiles]
    
    # Collect all method names
    all_methods = set()
    for p in profiles:
        all_methods.update(p.get('methods', {}).keys())
    
    # Aggregate each method
    aggregated_methods = {}
    for method in all_methods:
        total_ms_list = []
        calls_list = []
        min_us_list = []
        max_us_list = []
        
        for p in profiles:
            method_data = p.get('methods', {}).get(method)
            if method_data:
                total_ms_list.append(method_data.get('total_ms', 0))
                calls_list.append(method_data.get('calls', 0))
                min_us_list.append(method_data.get('min_us', float('inf')))
                max_us_list.append(method_data.get('max_us', 0))
        
        if total_ms_list:
            avg_total_ms = statistics.mean(total_ms_list)
            avg_calls = statistics.mean(calls_list)
            
            # Use the true min/max across all runs
            true_min_us = min(min_us_list) if min_us_list else 0
            true_max_us = max(max_us_list) if max_us_list else 0
            
            # Calculate average per-call time
            avg_us = (avg_total_ms * 1000 / avg_calls) if avg_calls > 0 else 0
            
            # Calculate standard deviation of total time
            std_ms = statistics.stdev(total_ms_list) if len(total_ms_list) > 1 else 0
            
            aggregated_methods[method] = {
                "calls": int(round(avg_calls)),
                "total_ms": round(avg_total_ms, 3),
                "avg_us": round(avg_us, 3),
                "min_us": round(true_min_us, 3),
                "max_us": round(true_max_us, 3),
                "std_ms": round(std_ms, 3),
                "pct": 0  # Will be recalculated
            }
    
    # Calculate session average
    avg_session_ms = statistics.mean(session_totals)
    std_session_ms = statistics.stdev(session_totals) if len(session_totals) > 1 else 0
    
    # Recalculate percentages based on averaged session total
    for method, data in aggregated_methods.items():
        data["pct"] = round(data["total_ms"] / avg_session_ms * 100, 2) if avg_session_ms > 0 else 0
    
    # Build final profile
    return {
        "session_total_ms": round(avg_session_ms, 3),
        "session_std_ms": round(std_session_ms, 3),
        "iterations": n,
        "timestamp": profiles[-1].get('timestamp', ''),
        "methods": aggregated_methods
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple profiling iterations and aggregate results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', '-c', required=True,
                        help='Path to the TOML configuration file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output the aggregated profile JSON')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help='Number of iterations to run (default: 10)')
    parser.add_argument('--cli', default=None,
                        help='Path to blaze2d-cli (default: auto-detect)')
    parser.add_argument('--extra-args', nargs='*', default=[],
                        help='Extra arguments to pass to blaze2d-cli')
    parser.add_argument('--symmetry', action='store_true',
                        help='Enable symmetry (pass --symmetry to CLI)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress information')
    
    args = parser.parse_args()
    
    # Handle --symmetry explicitly
    if args.symmetry:
        args.extra_args.append('--symmetry')
    
    # Find CLI
    if args.cli:
        cli_path = args.cli
    else:
        # Try to find it relative to this script
        script_dir = Path(__file__).parent.parent.parent
        cli_path = script_dir / "target" / "release" / "blaze2d-cli"
        if not cli_path.exists():
            print(f"Error: CLI not found at {cli_path}")
            print("Build with: cargo build --release -p blaze2d-cli --features profiling")
            return 1
        cli_path = str(cli_path)
    
    profiles = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(args.iterations):
            if args.verbose:
                print(f"Running iteration {i+1}/{args.iterations}...", end=" ", flush=True)
            
            tmp_profile = os.path.join(tmpdir, f"profile_{i}.json")
            try:
                profile = run_profile(cli_path, args.config, tmp_profile, args.extra_args)
                profiles.append(profile)
                if args.verbose:
                    print(f"done ({profile.get('session_total_ms', 0):.1f}ms)")
            except subprocess.CalledProcessError as e:
                print(f"\nError running iteration {i+1}: {e}")
                return 1
    
    if args.verbose:
        print(f"\nAggregating {len(profiles)} profiles...")
    
    aggregated = aggregate_profiles(profiles)
    
    # Write output
    with open(args.output, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    if args.verbose:
        session_avg = aggregated.get('session_total_ms', 0)
        session_std = aggregated.get('session_std_ms', 0)
        print(f"Session average: {session_avg:.2f}ms ± {session_std:.2f}ms ({session_std/session_avg*100:.1f}% variance)")
        print(f"Output written to: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
