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
    parser.add_argument(
        "--export-epsilon-inverse",
        type=Path,
        default=None,
        help="Export inverse epsilon tensor grid to CSV file (uses interpolation).",
    )
    parser.add_argument(
        "--export-epsilon-inverse-h5",
        type=Path,
        default=None,
        help="Export RAW inverse epsilon tensor from HDF5 (no interpolation, matches eigensolver).",
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
    """Export MPB's epsilon grid to CSV format matching blaze output."""
    # Get epsilon array from MPB (this is the smoothed/effective epsilon)
    eps_data = solver.get_epsilon()
    
    # eps_data is a 2D array for 2D simulations
    # Shape should be (resolution, resolution) for square lattice
    eps_array = np.array(eps_data)
    
    # Handle potential 3D array (squeeze z dimension if present)
    if eps_array.ndim == 3:
        eps_array = eps_array[:, :, 0]
    
    ny, nx = eps_array.shape
    
    # NOTE: blaze samples at cell centers while MPB reports epsilon on a grid that
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


def export_epsilon_inverse_csv(solver: Any, output_path: Path, resolution: int) -> None:
    """Export MPB's inverse epsilon tensor grid to CSV format matching blaze output.
    
    MPB uses get_epsilon_inverse_tensor_point(r) to get the 3x3 inverse epsilon
    tensor at each point. For 2D, we extract the 2x2 in-plane components:
    [xx, xy, yx, yy] which corresponds to row/column indices (0,0), (0,1), (1,0), (1,1).
    
    The returned Matrix type has c1, c2, c3 as column vectors.
    Element (i,j) can be accessed as matrix.row(i)[j] or matrix.c{j+1}[i].
    """
    nx = resolution
    ny = resolution
    
    with open(output_path, 'w') as f:
        f.write("ix,iy,frac_x,frac_y,inv_eps_xx,inv_eps_xy,inv_eps_yx,inv_eps_yy\n")
        for iy in range(ny):
            for ix in range(nx):
                # Use cell-centered fractional coordinates, same convention as epsilon export
                frac_x = (ix + 0.5) / nx
                frac_y = (iy + 0.5) / ny
                
                # MPB lattice coordinates: convert fractional to [-0.5, 0.5) range
                lat_x = frac_x - 0.5
                lat_y = frac_y - 0.5
                
                # Get inverse epsilon tensor at this point (3x3 Matrix)
                # Matrix has columns c1, c2, c3 which are Vector3 objects
                inv_eps_tensor = solver.get_epsilon_inverse_tensor_point(
                    mp.Vector3(lat_x, lat_y, 0)
                )
                
                # Extract 2D in-plane components (xx, xy, yx, yy)
                # Column vectors: c1 = [xx, yx, zx], c2 = [xy, yy, zy], c3 = [xz, yz, zz]
                # So: xx = c1.x, xy = c2.x, yx = c1.y, yy = c2.y
                # Values may be complex for general materials, but for real dielectrics
                # the imaginary part is zero. Take the real part.
                inv_eps_xx = float(inv_eps_tensor.c1.x.real)
                inv_eps_xy = float(inv_eps_tensor.c2.x.real)
                inv_eps_yx = float(inv_eps_tensor.c1.y.real)
                inv_eps_yy = float(inv_eps_tensor.c2.y.real)
                
                f.write(f"{ix},{iy},{frac_x:.12e},{frac_y:.12e},"
                        f"{inv_eps_xx:.12e},{inv_eps_xy:.12e},"
                        f"{inv_eps_yx:.12e},{inv_eps_yy:.12e}\n")
    
    print(f"Wrote inverse epsilon tensor grid ({nx}x{ny}) to {output_path}")


def export_epsilon_inverse_h5_csv(h5_path: Path, output_path: Path) -> None:
    """Export MPB's RAW inverse epsilon tensor from HDF5 file.
    
    This reads the epsilon_inverse.xx, epsilon_inverse.xy, epsilon_inverse.yy
    datasets directly from the HDF5 file that MPB uses internally for the
    eigensolver. This is the TRUE grid data with NO interpolation.
    
    Args:
        h5_path: Path to epsilon.h5 file generated by MPB's output_epsilon
        output_path: Path to write CSV output
    """
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        # List available datasets for debugging
        print(f"HDF5 datasets: {list(f.keys())}")
        
        # Check for inverse epsilon datasets
        has_inv = 'epsilon_inverse.xx' in f
        if not has_inv:
            # Try alternative naming
            print("Available keys:", list(f.keys()))
            raise ValueError("epsilon_inverse.xx not found in HDF5 file. "
                           "Make sure output-epsilon-inverse is enabled in MPB.")
        
        # Read the raw tensor components
        inv_xx = np.array(f['epsilon_inverse.xx'])
        inv_xy = np.array(f['epsilon_inverse.xy']) if 'epsilon_inverse.xy' in f else None
        inv_yy = np.array(f['epsilon_inverse.yy'])
        
        # For 2D simulations, these should be 2D arrays (or 3D with z=1)
        if inv_xx.ndim == 3:
            inv_xx = inv_xx[:, :, 0]
            if inv_xy is not None:
                inv_xy = inv_xy[:, :, 0]
            inv_yy = inv_yy[:, :, 0]
        
        ny, nx = inv_xx.shape
        print(f"Raw HDF5 grid size: {nx} x {ny}")
        
        # If no off-diagonal, it's zero
        if inv_xy is None:
            inv_xy = np.zeros_like(inv_xx)
        
        with open(output_path, 'w') as out:
            out.write("ix,iy,frac_x,frac_y,inv_eps_xx,inv_eps_xy,inv_eps_yx,inv_eps_yy\n")
            for iy in range(ny):
                for ix in range(nx):
                    # MPB stores data with node-based coordinates: i/N for index i
                    # This matches the grid the eigensolver uses
                    frac_x = ix / nx
                    frac_y = iy / ny
                    
                    # For symmetric tensor: xy = yx
                    xx = float(inv_xx[iy, ix])
                    xy = float(inv_xy[iy, ix])
                    yy = float(inv_yy[iy, ix])
                    
                    out.write(f"{ix},{iy},{frac_x:.12e},{frac_y:.12e},"
                              f"{xx:.12e},{xy:.12e},{xy:.12e},{yy:.12e}\n")
    
    print(f"Wrote RAW inverse epsilon tensor from HDF5 ({nx}x{ny}) to {output_path}")


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
    
    # Export inverse epsilon tensor if requested (interpolated version)
    if args.export_epsilon_inverse is not None:
        inv_eps_output = args.export_epsilon_inverse
        if not inv_eps_output.is_absolute():
            inv_eps_output = Path(__file__).parent.joinpath(inv_eps_output)
        inv_eps_output.parent.mkdir(parents=True, exist_ok=True)
        export_epsilon_inverse_csv(solver, inv_eps_output, args.resolution)
    
    # Export RAW inverse epsilon from HDF5 (no interpolation)
    if args.export_epsilon_inverse_h5 is not None:
        # First, we need to output the epsilon file from MPB
        # This writes epsilon.h5 which contains epsilon_inverse datasets
        solver.output_epsilon()
        
        # Find the HDF5 file (MPB writes to current directory with prefix)
        # The filename depends on the run prefix, typically "epsilon.h5"
        h5_candidates = list(Path('.').glob('*epsilon*.h5'))
        if not h5_candidates:
            print("Warning: No epsilon HDF5 file found. Run output_epsilon first.")
        else:
            h5_path = h5_candidates[0]  # Take the first match
            print(f"Reading from {h5_path}")
            
            inv_h5_output = args.export_epsilon_inverse_h5
            if not inv_h5_output.is_absolute():
                inv_h5_output = Path(__file__).parent.joinpath(inv_h5_output)
            inv_h5_output.parent.mkdir(parents=True, exist_ok=True)
            export_epsilon_inverse_h5_csv(h5_path, inv_h5_output)


if __name__ == "__main__":
    main()
