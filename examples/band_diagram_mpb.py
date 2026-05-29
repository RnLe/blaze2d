"""
Band diagrams for 2D photonic crystals using MPB (via meep).

Install:
  conda install -c conda-forge pymeep=*=mpi_mpich_*
  # or: conda install -c conda-forge pymeep=*=mpi_openmpi_*

Computes TM and TE band structures for two lattices:
  1) Square lattice — dielectric rods (ε=8.9, r=0.2a) in air
  2) Hexagonal lattice — air holes (r=0.48a) in dielectric (ε=13)

Usage:
    python band_diagram_mpb.py
"""

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from meep import mpb

# ── Settings ──────────────────────────────────────────────────────────────────
RESOLUTION = 32        # Grid points per lattice constant
NUM_BANDS = 8          # Number of bands to compute
K_INTERP = 15          # Interpolated points between each high-symmetry pair

# ── 1) Square lattice: ε=8.9 rods in air ─────────────────────────────────────
square_lattice = mp.Lattice(size=mp.Vector3(1, 1))
square_geometry = [
    mp.Cylinder(radius=0.2, material=mp.Medium(epsilon=8.9), height=mp.inf)
]
# Path: Γ → X → M → Γ
square_kpoints = mp.interpolate(K_INTERP, [
    mp.Vector3(0, 0),       # Γ
    mp.Vector3(0.5, 0),     # X
    mp.Vector3(0.5, 0.5),   # M
    mp.Vector3(0, 0),       # Γ
])

# ── 2) Hex lattice: air holes in ε=13 ────────────────────────────────────────
hex_lattice = mp.Lattice(
    basis1=mp.Vector3(1, 0),
    basis2=mp.Vector3(0.5, np.sqrt(3)/2),
    size=mp.Vector3(1, 1),
)
hex_geometry = [
    mp.Cylinder(radius=0.48, material=mp.Medium(epsilon=1.0), height=mp.inf)
]
# Path: Γ → M → K → Γ
hex_kpoints = mp.interpolate(K_INTERP, [
    mp.Vector3(0, 0),       # Γ
    mp.Vector3(0.5, 0),     # M
    mp.Vector3(1/3, 1/3),   # K
    mp.Vector3(0, 0),       # Γ
])

# ── Helper: run solver and extract data ───────────────────────────────────────
def run_mpb(lattice, geometry, kpoints, eps_bg, polarization):
    """
    Run MPB for a given polarization ("tm" or "te").

    Returns:
        freqs    – np.array of shape (n_kpoints, NUM_BANDS)
        k_dists  – np.array of cumulative k-path distance (n_kpoints,)
    """
    solver = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=kpoints,
        default_material=mp.Medium(epsilon=eps_bg),
        resolution=RESOLUTION,
        num_bands=NUM_BANDS,
        tolerance=1e-7,
    )

    # Run TM or TE
    if polarization == "tm":
        solver.run_tm()
    else:
        solver.run_te()

    # solver.all_freqs: list of arrays, one per k-point
    freqs = np.array(solver.all_freqs)  # shape (n_k, n_bands)

    # Build cumulative k-path distance
    k_dists = [0.0]
    for i in range(1, len(kpoints)):
        prev, cur = kpoints[i-1], kpoints[i]
        k_dists.append(k_dists[-1] + np.sqrt(
            (cur.x - prev.x)**2 + (cur.y - prev.y)**2))
    k_dists = np.array(k_dists)

    return freqs, k_dists


# ── Run all four calculations ─────────────────────────────────────────────────
print("=== Square lattice, TM ===")
sq_tm_freqs, sq_kdist = run_mpb(
    square_lattice, square_geometry, square_kpoints, eps_bg=1.0, polarization="tm")

print("=== Square lattice, TE ===")
sq_te_freqs, _ = run_mpb(
    square_lattice, square_geometry, square_kpoints, eps_bg=1.0, polarization="te")

print("=== Hex lattice, TM ===")
hex_tm_freqs, hex_kdist = run_mpb(
    hex_lattice, hex_geometry, hex_kpoints, eps_bg=13.0, polarization="tm")

print("=== Hex lattice, TE ===")
hex_te_freqs, _ = run_mpb(
    hex_lattice, hex_geometry, hex_kpoints, eps_bg=13.0, polarization="te")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Square lattice
ax = axes[0]
for b in range(NUM_BANDS):
    ax.plot(sq_kdist, sq_tm_freqs[:, b], color="C0", linewidth=0.8)
    ax.plot(sq_kdist, sq_te_freqs[:, b], color="C1", linewidth=0.8)
ax.set_title("Square: ε=8.9 rods in air")
ax.set_ylabel("Frequency (c/a)")
ax.set_xlabel("k-path")
# Mark high-symmetry points (indices: 0, K_INTERP+1, 2*(K_INTERP+1), 3*(K_INTERP+1))
seg = K_INTERP + 1
ticks = [sq_kdist[i * seg] for i in range(3)] + [sq_kdist[-1]]
ax.set_xticks(ticks)
ax.set_xticklabels(["Γ", "X", "M", "Γ"])

# Hex lattice
ax = axes[1]
for b in range(NUM_BANDS):
    ax.plot(hex_kdist, hex_tm_freqs[:, b], color="C0", linewidth=0.8)
    ax.plot(hex_kdist, hex_te_freqs[:, b], color="C1", linewidth=0.8)
ax.set_title("Hex: air holes in ε=13")
ax.set_ylabel("Frequency (c/a)")
ax.set_xlabel("k-path")
ticks = [hex_kdist[i * seg] for i in range(3)] + [hex_kdist[-1]]
ax.set_xticks(ticks)
ax.set_xticklabels(["Γ", "M", "K", "Γ"])

# Legend (one entry per polarization)
axes[0].plot([], [], color="C0", label="TM")
axes[0].plot([], [], color="C1", label="TE")
axes[0].legend()

plt.tight_layout()
plt.savefig("band_diagram_mpb.svg")
plt.show()
