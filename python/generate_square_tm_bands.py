#!/usr/bin/env python3
"""Generate 2D MPB band reference data (square/hex lattices)."""

from __future__ import annotations

import argparse
import json
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

def _load_runtime_modules():
    try:
        np_mod = importlib.import_module("numpy")
        mp_mod = importlib.import_module("meep")
        mpb_mod = importlib.import_module("meep.mpb")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(
            "This script requires the mpb-reference environment (pymeep/mpb)."
        ) from exc
    return np_mod, mp_mod, mpb_mod


np, mp, mpb = _load_runtime_modules()


@dataclass
class KPoint:
    label: str
    coords: Tuple[float, float]


SQUARE_PATH: Sequence[KPoint] = (
    KPoint("G", (0.0, 0.0)),
    KPoint("X", (0.5, 0.0)),
    KPoint("M", (0.5, 0.5)),
    KPoint("G", (0.0, 0.0)),
)

HEX_PATH: Sequence[KPoint] = (
    KPoint("G", (0.0, 0.0)),
    KPoint("M", (0.5, 0.0)),
    KPoint("K", (1.0 / 3.0, 1.0 / 3.0)),
    KPoint("G", (0.0, 0.0)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference-data/square_tm_uniform_mpb.json"),
        help="Destination JSON file.",
    )
    parser.add_argument("--resolution", type=int, default=32, help="Fourier grid resolution.")
    parser.add_argument("--num-bands", type=int, default=4, help="Number of bands to compute.")
    parser.add_argument("--k-density", type=int, default=8, help="Interpolation points per leg.")
    parser.add_argument("--radius", type=float, default=0.0, help="Air-hole radius in units of a.")
    parser.add_argument(
        "--eps-bg",
        type=float,
        default=12.0,
        help="Background dielectric constant (default silicon).",
    )
    parser.add_argument(
        "--eps-hole",
        type=float,
        default=1.0,
        help="Dielectric constant inside holes (air by default).",
    )
    parser.add_argument(
        "--polarization",
        choices=["tm", "te"],
        default="tm",
        help="Which polarization to solve (TM or TE).",
    )
    parser.add_argument(
        "--lattice",
        choices=["square", "hexagonal"],
        default="square",
        help="Which Bravais lattice / high-symmetry path to sample.",
    )
    parser.add_argument(
        "--export-epsilon",
        type=Path,
        default=None,
        help="Export epsilon(r) grid to CSV file.",
    )
    return parser.parse_args()


def build_k_path(nodes: Sequence[KPoint], density: int) -> Tuple[List[Any], List[int], List[dict]]:
    assert density > 0, "k-point density must be positive"
    samples: List[dict] = []
    vectors: List[Any] = []
    node_indices: List[int] = [0]
    total_dist = 0.0
    prev = np.array(nodes[0].coords, dtype=float)
    samples.append({"kx": float(prev[0]), "ky": float(prev[1]), "distance": total_dist})
    vectors.append(mp.Vector3(prev[0], prev[1], 0.0))
    for seg in range(len(nodes) - 1):
        start = np.array(nodes[seg].coords, dtype=float)
        end = np.array(nodes[seg + 1].coords, dtype=float)
        for step in range(1, density + 1):
            t = step / density
            point = (1.0 - t) * start + t * end
            total_dist += np.linalg.norm(point - prev)
            samples.append(
                {"kx": float(point[0]), "ky": float(point[1]), "distance": total_dist}
            )
            vectors.append(mp.Vector3(point[0], point[1], 0.0))
            prev = point
        node_indices.append(len(samples) - 1)
    return vectors, node_indices, samples


def build_geometry(radius: float, eps_hole: float) -> List[Any]:
    if radius <= 0.0:
        return []
    return [mp.Cylinder(radius=radius, material=mp.Medium(epsilon=eps_hole), height=mp.inf)]


def build_lattice(kind: str) -> Any:
    if kind == "square":
        return mp.Lattice(size=mp.Vector3(1, 1, 0))
    if kind == "hexagonal":
        return mp.Lattice(
            size=mp.Vector3(1, 1, 0),
            basis1=mp.Vector3(1, 0, 0),
            basis2=mp.Vector3(0.5, math.sqrt(3) / 2.0, 0),
        )
    raise ValueError(f"unsupported lattice kind: {kind}")


def select_path(kind: str) -> Sequence[KPoint]:
    if kind == "square":
        return SQUARE_PATH
    if kind == "hexagonal":
        return HEX_PATH
    raise ValueError(f"unsupported lattice kind: {kind}")


def run_solver(args: argparse.Namespace) -> Tuple[dict, Any]:
    """Run the MPB solver and return (result_dict, solver) for optional epsilon export."""
    nodes = select_path(args.lattice)
    k_pts, node_indices, k_samples = build_k_path(nodes, args.k_density)
    geometry = build_geometry(args.radius, args.eps_hole)
    lattice = build_lattice(args.lattice)
    solver = mpb.ModeSolver(
        num_bands=args.num_bands,
        k_points=k_pts,
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=args.eps_bg),
        resolution=args.resolution,
        dimensions=2,
    )
    if args.polarization.lower() == "tm":
        solver.run_tm()
    else:
        solver.run_te()

    freqs = np.array(solver.all_freqs)
    nodes = [
        {
            "label": nodes[i].label,
            "index": node_indices[i],
        }
        for i in range(len(nodes))
    ]

    result = {
        "metadata": {
            "num_bands": args.num_bands,
            "resolution": args.resolution,
            "radius": args.radius,
            "eps_bg": args.eps_bg,
            "eps_hole": args.eps_hole,
            "k_density": args.k_density,
            "polarization": args.polarization.lower(),
            "lattice": args.lattice.lower(),
        },
        "k_path": k_samples,
        "k_nodes": nodes,
        "bands": freqs.tolist(),
    }
    return result, solver


def export_epsilon_csv(solver: Any, output_path: Path, resolution: int) -> None:
    """Export MPB's epsilon grid to CSV format matching mpb2d output."""
    # Get epsilon array from MPB (this is the smoothed/effective epsilon)
    eps_data = solver.get_epsilon()
    
    # eps_data is a 2D array for 2D simulations
    # Shape should be (resolution, resolution) for square lattice
    eps_array = np.array(eps_data)
    
    # Handle potential 3D array (squeeze z dimension if present)
    if eps_array.ndim == 3:
        eps_array = eps_array[:, :, 0]
    
    ny, nx = eps_array.shape
    
    # NOTE: mpb2d samples at cell centers while MPB reports epsilon on a grid that
    # behaves as if it were offset by half a pixel. We continue to export the data
    # using center-based coordinates (i + 0.5) / n so downstream tooling can treat
    # both files consistently, and apply the necessary shift during analysis.
    with open(output_path, 'w') as f:
        f.write("ix,iy,frac_x,frac_y,eps_mpb\n")
        for iy in range(ny):
            for ix in range(nx):
                frac_x = (ix + 0.5) / nx
                frac_y = (iy + 0.5) / ny
                eps_val = eps_array[iy, ix]
                f.write(f"{ix},{iy},{frac_x},{frac_y},{eps_val}\n")
    
    print(f"Wrote epsilon grid ({nx}x{ny}) to {output_path}")


def main() -> None:
    args = parse_args()
    payload, solver = run_solver(args)
    output = args.output
    if not output.is_absolute():
        output = Path(__file__).parent.joinpath(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output} with {len(payload['bands'])} k-points")
    
    # Export epsilon if requested
    if args.export_epsilon is not None:
        eps_output = args.export_epsilon
        if not eps_output.is_absolute():
            eps_output = Path(__file__).parent.joinpath(eps_output)
        eps_output.parent.mkdir(parents=True, exist_ok=True)
        export_epsilon_csv(solver, eps_output, args.resolution)


if __name__ == "__main__":
    main()
