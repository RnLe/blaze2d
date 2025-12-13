#!/usr/bin/env python3
"""Produce an analytic TM reference dataset for a uniform square lattice.

This script mirrors the k-path used by the MPB tooling but avoids requiring
pymeep at build/test time by evaluating the |k + G| dispersion analytically.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class KPoint:
    label: str
    coords: Tuple[float, float]


def lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def build_k_path(path: Sequence[KPoint], density: int) -> Tuple[List[dict], List[int]]:
    assert density > 0
    samples: List[dict] = [
        {"kx": path[0].coords[0], "ky": path[0].coords[1], "distance": 0.0}
    ]
    node_indices: List[int] = [0]
    total_dist = 0.0
    for segment_idx in range(len(path) - 1):
        start = path[segment_idx]
        end = path[segment_idx + 1]
        prev = samples[-1]
        for step in range(1, density + 1):
            t = step / density
            kx = lerp(start.coords[0], end.coords[0], t)
            ky = lerp(start.coords[1], end.coords[1], t)
            dx = kx - prev["kx"]
            dy = ky - prev["ky"]
            total_dist += math.hypot(dx, dy)
            samples.append({"kx": kx, "ky": ky, "distance": total_dist})
            prev = samples[-1]
        node_indices.append(len(samples) - 1)
    return samples, node_indices


def enumerate_freqs(kx: float, ky: float, eps_bg: float, max_g: int) -> List[float]:
    inv_sqrt_eps = 1.0 / math.sqrt(eps_bg)
    two_pi = 2.0 * math.pi
    freqs: List[float] = []
    for gx in range(-max_g, max_g + 1):
        for gy in range(-max_g, max_g + 1):
            total_kx = kx + gx
            total_ky = ky + gy
            mag = math.hypot(total_kx, total_ky)
            freqs.append(two_pi * mag * inv_sqrt_eps)
    freqs.sort()
    return freqs


def build_bands(k_path: Sequence[dict], num_bands: int, eps_bg: float, max_g: int) -> List[List[float]]:
    bands: List[List[float]] = []
    for kp in k_path:
        freqs = enumerate_freqs(kp["kx"], kp["ky"], eps_bg, max_g)
        bands.append(freqs[:num_bands])
    return bands


def main() -> None:
    output = Path(__file__).parent / "reference-data" / "square_tm_uniform.json"
    num_bands = 5
    eps_bg = 12.0
    eps_hole = 1.0
    radius = 0.0
    k_density = 8
    max_g = 3
    path = (
        KPoint("G", (0.0, 0.0)),
        KPoint("X", (0.5, 0.0)),
        KPoint("M", (0.5, 0.5)),
        KPoint("G", (0.0, 0.0)),
    )
    k_path, node_indices = build_k_path(path, k_density)
    bands = build_bands(k_path, num_bands, eps_bg, max_g)
    payload = {
        "metadata": {
            "num_bands": num_bands,
            "radius": radius,
            "eps_bg": eps_bg,
            "eps_hole": eps_hole,
            "k_density": k_density,
            "resolution": "analytic",
            "max_g_index": max_g,
            "source": "uniform_plane_wave_enumeration",
        },
        "k_path": k_path,
        "k_nodes": [
            {"label": path[i].label, "index": node_indices[i]}
            for i in range(len(path))
        ],
        "bands": bands,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output} with {len(k_path)} k-points")


if __name__ == "__main__":
    main()
