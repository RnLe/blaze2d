#!/usr/bin/env python3
"""
Convert phase1_band_data.h5 to compact binary format for web visualization.

This script reads HDF5 files and outputs:
  - phase1_fields.bin: Binary file with all field data as Float32
  - phase1_fields_meta.json: Metadata (shape, extent, min/max for colorscaling)

Binary format (all Float32, little-endian):
  [V_data: NxN] [vg_norm_data: NxN] [M_inv_eig1_data: NxN] [M_inv_eig2_data: NxN]

Usage:
  python convert_phase1_data.py <input_h5_path> <output_dir>
"""

import sys
import json
import struct
from pathlib import Path

import h5py
import numpy as np


def compute_vg_norm(vg: np.ndarray) -> np.ndarray:
    """Compute magnitude of group velocity."""
    return np.linalg.norm(vg, axis=-1)


def compute_M_inv_eigenvalues(M_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues of 2x2 inverse mass tensor at each grid point."""
    # M_inv has shape (Nx, Ny, 2, 2)
    # Eigenvalues of 2x2 matrix [[a,b],[c,d]]:
    # λ = (trace ± sqrt(trace² - 4*det)) / 2
    
    Nx, Ny = M_inv.shape[:2]
    eig1 = np.zeros((Nx, Ny), dtype=np.float64)
    eig2 = np.zeros((Nx, Ny), dtype=np.float64)
    
    for i in range(Nx):
        for j in range(Ny):
            mat = M_inv[i, j]
            eigenvalues = np.linalg.eigvalsh(mat)  # Hermitian eigenvalues, sorted
            eig1[i, j] = eigenvalues[0]  # smaller
            eig2[i, j] = eigenvalues[1]  # larger
    
    return eig1, eig2


def convert_phase1_data(input_path: Path, output_dir: Path) -> dict:
    """Convert phase1_band_data.h5 to binary format."""
    
    with h5py.File(input_path, 'r') as f:
        V = np.array(f['V'])
        vg = np.array(f['vg'])
        M_inv = np.array(f['M_inv'])
        R_grid = np.array(f['R_grid'])
    
    # Compute derived fields
    vg_norm = compute_vg_norm(vg)
    M_inv_eig1, M_inv_eig2 = compute_M_inv_eigenvalues(M_inv)
    
    # Get grid extent from R_grid
    x_coords = R_grid[:, 0, 0]
    y_coords = R_grid[0, :, 1]
    extent = {
        'xMin': float(x_coords.min()),
        'xMax': float(x_coords.max()),
        'yMin': float(y_coords.min()),
        'yMax': float(y_coords.max()),
    }
    
    # Convert all to float32 for compact storage
    V_f32 = V.astype(np.float32)
    vg_norm_f32 = vg_norm.astype(np.float32)
    M_inv_eig1_f32 = M_inv_eig1.astype(np.float32)
    M_inv_eig2_f32 = M_inv_eig2.astype(np.float32)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write binary file (concatenated Float32 arrays)
    bin_path = output_dir / 'phase1_fields.bin'
    with open(bin_path, 'wb') as f:
        f.write(V_f32.tobytes())
        f.write(vg_norm_f32.tobytes())
        f.write(M_inv_eig1_f32.tobytes())
        f.write(M_inv_eig2_f32.tobytes())
    
    # Create metadata
    shape = list(V.shape)
    meta = {
        'shape': shape,
        'extent': extent,
        'fields': {
            'V': {
                'offset': 0,
                'min': float(V.min()),
                'max': float(V.max()),
            },
            'vg_norm': {
                'offset': V_f32.nbytes,
                'min': float(vg_norm.min()),
                'max': float(vg_norm.max()),
            },
            'M_inv_eig1': {
                'offset': V_f32.nbytes + vg_norm_f32.nbytes,
                'min': float(M_inv_eig1.min()),
                'max': float(M_inv_eig1.max()),
            },
            'M_inv_eig2': {
                'offset': V_f32.nbytes + vg_norm_f32.nbytes + M_inv_eig1_f32.nbytes,
                'min': float(M_inv_eig2.min()),
                'max': float(M_inv_eig2.max()),
            },
        },
        'totalBytes': V_f32.nbytes * 4,
    }
    
    # Write metadata
    meta_path = output_dir / 'phase1_fields_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return meta


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_h5_path> <output_dir>", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    meta = convert_phase1_data(input_path, output_dir)
    print(f"Converted {input_path.name}: {meta['shape'][0]}x{meta['shape'][1]} grid, {meta['totalBytes']} bytes")


if __name__ == '__main__':
    main()
