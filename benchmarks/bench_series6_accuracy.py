#!/usr/bin/env python3
"""
Benchmark Series 6: Accuracy Comparison

This benchmark compares the numerical accuracy of eigenvalues between:
1. MPB (f64 reference - "ground truth")
2. Blaze2D mixed-precision (f32 storage, f64 accumulation) - default mode
3. Blaze2D full-precision (f64 throughout) - requires --no-default-features

Key insight: Blaze2D converges to the TRUE lowest eigenvalues, while MPB may
converge to different local minima. Therefore, we use Hungarian algorithm
(optimal assignment) to match bands between solvers before computing errors.

Metrics:
- Per-band relative error after optimal matching
- Maximum deviation across all k-points and bands
- RMS error distribution

Output: results/series6_accuracy/
"""

import subprocess
import os
import sys
import json
import csv
import argparse
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Fixed parameters (Config A: Square lattice with ε=8.9 rods)
EPSILON = 8.9
RADIUS = 0.2
EPS_BG = 1.0

# Tolerances
MPB_TOLERANCE = 1e-7      # MPB f64 reference
BLAZE_F64_TOLERANCE = 1e-7  # Blaze full precision
BLAZE_F32_TOLERANCE = 1e-4  # Blaze mixed precision

# Benchmark parameters
DEFAULT_RESOLUTION = 32
DEFAULT_BANDS = 8
DEFAULT_K_PER_SEG = 15


@dataclass
class BandData:
    """Band structure data for a single polarization."""
    solver: str
    polarization: str
    resolution: int
    num_bands: int
    k_points: List[Dict]  # Each has k_frac, k_distance, frequencies
    tolerance: float


@dataclass
class AccuracyResult:
    """Accuracy comparison results."""
    polarization: str
    mpb_data: Optional[BandData]
    blaze_f32_data: Optional[BandData]
    blaze_f64_data: Optional[BandData]
    
    # After Hungarian matching
    f32_deviations: List[Dict] = field(default_factory=list)
    f64_deviations: List[Dict] = field(default_factory=list)
    f32_vs_f64_deviations: List[Dict] = field(default_factory=list)


def run_mpb(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int,
    tolerance: float = MPB_TOLERANCE,
) -> BandData:
    """Run MPB and extract band data."""
    pol_lower = polarization.lower()
    
    script = f'''
import meep as mp
from meep import mpb
import json

geometry = [
    mp.Cylinder(
        radius={RADIUS},
        material=mp.Medium(epsilon={EPSILON}),
        center=mp.Vector3(0, 0, 0),
    )
]

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))

k_points = mp.interpolate({k_points_per_segment - 1}, [
    mp.Vector3(0, 0, 0),      # Gamma
    mp.Vector3(0.5, 0, 0),    # X
    mp.Vector3(0.5, 0.5, 0),  # M
    mp.Vector3(0, 0, 0),      # Gamma
])

solver = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    resolution={resolution},
    num_bands={num_bands},
    default_material=mp.Medium(epsilon={EPS_BG}),
    tolerance={tolerance},
)

solver.run_{pol_lower}()

# Extract band data
results = []
freqs = solver.all_freqs
k_distances = []
dist = 0.0
prev_k = None
for i, k in enumerate(k_points):
    if prev_k is not None:
        dist += ((k.x - prev_k.x)**2 + (k.y - prev_k.y)**2)**0.5
    k_distances.append(dist)
    prev_k = k
    
    results.append({{
        "k_index": i,
        "k_frac": [k.x, k.y, k.z],
        "k_distance": dist,
        "frequencies": list(freqs[i]) if i < len(freqs) else [],
    }})

print("JSON_OUTPUT_START")
print(json.dumps(results))
print("JSON_OUTPUT_END")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = Path(f.name)
    
    try:
        mpb_python = "/home/renlephy/.local/share/mamba/envs/mpb-reference/bin/python"
        result = subprocess.run(
            [mpb_python, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        output = result.stdout + result.stderr
        
        # Extract JSON
        match = re.search(r'JSON_OUTPUT_START\n(.*?)\nJSON_OUTPUT_END', output, re.DOTALL)
        if not match:
            raise RuntimeError(f"Failed to extract MPB output: {output[:500]}")
        
        k_points_data = json.loads(match.group(1))
        
        return BandData(
            solver="MPB",
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            k_points=k_points_data,
            tolerance=tolerance,
        )
    finally:
        script_path.unlink(missing_ok=True)


def create_blaze_config(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int,
    tolerance: float,
) -> str:
    """Create TOML config for Blaze2D."""
    return f'''# Blaze2D accuracy test config
polarization = "{polarization}"

[geometry]
eps_bg = {EPS_BG}

[geometry.lattice]
a1 = [1.0, 0.0]
a2 = [0.0, 1.0]

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {RADIUS}
eps_inside = {EPSILON}

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[path]
preset = "square"
segments_per_leg = {k_points_per_segment}

[eigensolver]
num_bands = {num_bands}
tol = {tolerance}
'''


def run_blaze(
    resolution: int,
    num_bands: int,
    polarization: str,
    k_points_per_segment: int,
    tolerance: float,
    use_mixed_precision: bool = True,
) -> BandData:
    """Run Blaze2D and extract band data."""
    
    config = create_blaze_config(resolution, num_bands, polarization, 
                                  k_points_per_segment, tolerance)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config)
        config_path = Path(f.name)
    
    output_csv = tempfile.mktemp(suffix='.csv')
    
    try:
        # Build command
        if use_mixed_precision:
            # Default features include mixed-precision
            build_cmd = ["cargo", "build", "--release", "-p", "blaze2d-cli"]
        else:
            # Disable default features to get full f64 precision
            build_cmd = ["cargo", "build", "--release", "-p", "blaze2d-cli", 
                        "--no-default-features", "--features", "batched-fft"]
        
        # Build
        subprocess.run(
            build_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            env={**os.environ, "RUSTFLAGS": "-C target-cpu=native"},
        )
        
        blaze_cli = PROJECT_ROOT / "target" / "release" / "blaze2d-cli"
        
        result = subprocess.run(
            [str(blaze_cli), "--config", str(config_path), "--output", output_csv],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Blaze2D failed: {result.stderr[:500]}")
        
        # Parse CSV output
        k_points_data = []
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            band_cols = [c for c in reader.fieldnames if c.startswith("band")]
            
            for row in reader:
                freqs = []
                for col in band_cols:
                    val = row[col]
                    freqs.append(float(val) if val else float('nan'))
                
                k_points_data.append({
                    "k_index": int(row["k_index"]),
                    "k_frac": [float(row["kx"]), float(row["ky"])],
                    "k_distance": float(row["k_distance"]),
                    "frequencies": freqs,
                })
        
        solver_name = "Blaze2D-f32" if use_mixed_precision else "Blaze2D-f64"
        
        return BandData(
            solver=solver_name,
            polarization=polarization,
            resolution=resolution,
            num_bands=num_bands,
            k_points=k_points_data,
            tolerance=tolerance,
        )
    finally:
        config_path.unlink(missing_ok=True)
        Path(output_csv).unlink(missing_ok=True)


def hungarian_match(ref_freqs: np.ndarray, test_freqs: np.ndarray) -> List[Dict]:
    """
    Match bands using Hungarian algorithm for optimal assignment.
    
    Args:
        ref_freqs: Reference frequencies, shape (num_bands, num_k)
        test_freqs: Test frequencies, shape (num_bands, num_k)
    
    Returns:
        List of deviation dicts for each (k, band) pair after optimal matching.
    """
    num_bands, num_k = ref_freqs.shape
    deviations = []
    
    for k_idx in range(num_k):
        ref_vals = ref_freqs[:, k_idx]
        test_vals = test_freqs[:, k_idx]
        
        # Skip if NaN
        if np.any(np.isnan(ref_vals)) or np.any(np.isnan(test_vals)):
            continue
        
        # Build cost matrix
        cost = np.abs(ref_vals[:, None] - test_vals[None, :])
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)
        
        for ref_band, test_band in zip(row_ind, col_ind):
            ref_val = ref_vals[ref_band]
            test_val = test_vals[test_band]
            abs_dev = abs(ref_val - test_val)
            rel_dev = abs_dev / abs(ref_val) if ref_val != 0 else 0
            
            deviations.append({
                "k_index": k_idx,
                "ref_band": int(ref_band),
                "test_band": int(test_band),
                "ref_freq": float(ref_val),
                "test_freq": float(test_val),
                "abs_deviation": float(abs_dev),
                "rel_deviation": float(rel_dev),
            })
    
    return deviations


def compute_accuracy(
    mpb_data: BandData,
    blaze_f32_data: Optional[BandData],
    blaze_f64_data: Optional[BandData],
    polarization: str,
) -> AccuracyResult:
    """Compute accuracy metrics using Hungarian matching."""
    
    result = AccuracyResult(
        polarization=polarization,
        mpb_data=mpb_data,
        blaze_f32_data=blaze_f32_data,
        blaze_f64_data=blaze_f64_data,
    )
    
    # Extract frequency arrays
    num_k = len(mpb_data.k_points)
    num_bands = mpb_data.num_bands
    
    mpb_freqs = np.zeros((num_bands, num_k))
    for i, kp in enumerate(mpb_data.k_points):
        mpb_freqs[:, i] = kp["frequencies"][:num_bands]
    
    # Compare f32 vs MPB
    if blaze_f32_data:
        f32_freqs = np.zeros((num_bands, num_k))
        for i, kp in enumerate(blaze_f32_data.k_points):
            f32_freqs[:, i] = kp["frequencies"][:num_bands]
        result.f32_deviations = hungarian_match(mpb_freqs, f32_freqs)
    
    # Compare f64 vs MPB
    if blaze_f64_data:
        f64_freqs = np.zeros((num_bands, num_k))
        for i, kp in enumerate(blaze_f64_data.k_points):
            f64_freqs[:, i] = kp["frequencies"][:num_bands]
        result.f64_deviations = hungarian_match(mpb_freqs, f64_freqs)
    
    # Compare f32 vs f64 (if both available)
    if blaze_f32_data and blaze_f64_data:
        result.f32_vs_f64_deviations = hungarian_match(f64_freqs, f32_freqs)
    
    return result


def run_series(
    output_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
    num_bands: int = DEFAULT_BANDS,
    k_points_per_segment: int = DEFAULT_K_PER_SEG,
    skip_f64: bool = False,
):
    """Run the full accuracy benchmark."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Benchmark Series 6: Accuracy Comparison")
    print("=" * 70)
    print(f"Resolution: {resolution}×{resolution}")
    print(f"Bands: {num_bands}")
    print(f"K-points per segment: {k_points_per_segment}")
    print(f"Total k-points: {3 * k_points_per_segment + 1}")
    print(f"Config: Square lattice, ε={EPSILON} rods, r={RADIUS}a")
    print(f"Tolerances: MPB={MPB_TOLERANCE}, Blaze-f64={BLAZE_F64_TOLERANCE}, Blaze-f32={BLAZE_F32_TOLERANCE}")
    print("=" * 70)
    
    all_results = {
        "series": "series6_accuracy",
        "description": "Numerical accuracy comparison",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "resolution": resolution,
            "num_bands": num_bands,
            "k_points_per_segment": k_points_per_segment,
            "total_k_points": 3 * k_points_per_segment + 1,
            "epsilon": EPSILON,
            "radius": RADIUS,
            "mpb_tolerance": MPB_TOLERANCE,
            "blaze_f64_tolerance": BLAZE_F64_TOLERANCE,
            "blaze_f32_tolerance": BLAZE_F32_TOLERANCE,
        },
        "results": {},
    }
    
    for pol in ["TM", "TE"]:
        print(f"\n[{pol} Polarization]")
        print("-" * 50)
        
        # Run MPB reference
        print("  Running MPB (f64 reference)...", end=" ", flush=True)
        mpb_data = run_mpb(resolution, num_bands, pol, k_points_per_segment)
        print("done")
        
        # Run Blaze f32 (mixed precision)
        print("  Running Blaze2D (mixed-precision f32)...", end=" ", flush=True)
        blaze_f32_data = run_blaze(resolution, num_bands, pol, k_points_per_segment,
                                    BLAZE_F32_TOLERANCE, use_mixed_precision=True)
        print("done")
        
        # Run Blaze f64 (full precision)
        blaze_f64_data = None
        if not skip_f64:
            print("  Running Blaze2D (full-precision f64)...", end=" ", flush=True)
            blaze_f64_data = run_blaze(resolution, num_bands, pol, k_points_per_segment,
                                        BLAZE_F64_TOLERANCE, use_mixed_precision=False)
            print("done")
        
        # Compute accuracy
        accuracy = compute_accuracy(mpb_data, blaze_f32_data, blaze_f64_data, pol)
        
        # Store results
        pol_results = {
            "mpb": {
                "solver": mpb_data.solver,
                "tolerance": mpb_data.tolerance,
                "k_points": mpb_data.k_points,
            },
            "blaze_f32": {
                "solver": blaze_f32_data.solver if blaze_f32_data else None,
                "tolerance": blaze_f32_data.tolerance if blaze_f32_data else None,
                "k_points": blaze_f32_data.k_points if blaze_f32_data else None,
            },
            "f32_vs_mpb": accuracy.f32_deviations,
        }
        
        if blaze_f64_data:
            pol_results["blaze_f64"] = {
                "solver": blaze_f64_data.solver,
                "tolerance": blaze_f64_data.tolerance,
                "k_points": blaze_f64_data.k_points,
            }
            pol_results["f64_vs_mpb"] = accuracy.f64_deviations
            pol_results["f32_vs_f64"] = accuracy.f32_vs_f64_deviations
        
        all_results["results"][pol] = pol_results
        
        # Print summary
        if accuracy.f32_deviations:
            rel_devs = [d["rel_deviation"] for d in accuracy.f32_deviations]
            print(f"  f32 vs MPB: mean={np.mean(rel_devs):.2e}, max={np.max(rel_devs):.2e}")
        
        if accuracy.f64_deviations:
            rel_devs = [d["rel_deviation"] for d in accuracy.f64_deviations]
            print(f"  f64 vs MPB: mean={np.mean(rel_devs):.2e}, max={np.max(rel_devs):.2e}")
        
        if accuracy.f32_vs_f64_deviations:
            rel_devs = [d["rel_deviation"] for d in accuracy.f32_vs_f64_deviations]
            print(f"  f32 vs f64: mean={np.mean(rel_devs):.2e}, max={np.max(rel_devs):.2e}")
    
    # Save results
    results_file = output_dir / "series6_accuracy_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_file}")
    print("=" * 70)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Series 6: Accuracy")
    parser.add_argument("--output", type=str, default="results/series6_accuracy",
                        help="Output directory")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument("--k-per-seg", type=int, default=DEFAULT_K_PER_SEG)
    parser.add_argument("--skip-f64", action="store_true",
                        help="Skip full-precision f64 run (faster)")
    args = parser.parse_args()
    
    output_dir = SCRIPT_DIR / args.output
    run_series(
        output_dir,
        resolution=args.resolution,
        num_bands=args.bands,
        k_points_per_segment=args.k_per_seg,
        skip_f64=args.skip_f64,
    )


if __name__ == "__main__":
    main()
