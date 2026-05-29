"""
Band diagrams for 2D photonic crystals using blaze.solve().

Install:  pip install blaze2d matplotlib

Two lattices, TM + TE each:
  1) Square — dielectric rods (ε=8.9, r=0.2a) in air
  2) Hexagonal — air holes (r=0.48a) in dielectric (ε=13)

Usage:  python band_diagram_blaze.py
"""

import matplotlib.pyplot as plt
import blaze

# ── Solve (polarization sweep gives [TM, TE] per lattice) ────────────────────
square = blaze.solve(lattice_type="square", resolution=32, epsilon_background=1.0,
                 epsilon_atoms=8.9, radius_atom=0.2, polarization=["TM", "TE"], n_bands=8)

hexagonal = blaze.solve(lattice_type="hexagonal", resolution=32, epsilon_background=13.0,
                 epsilon_atoms=1.0, radius_atom=0.48, polarization=["TM", "TE"], n_bands=8)




# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, results, title in [
    (axes[0], square, "Square: ε=8.9 rods in air"),
    (axes[1], hexagonal, "Hex: air holes in ε=13"),
]:
    for r, color, ls in zip(results, ["C0", "C1"], ["-", "-"]):
        for b in range(r.n_bands):
            ax.plot(r.distances, r.freqs[:, b], color=color, ls=ls, lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Frequency (c/a)")
    ax.set_xticks(results[0].k_label_distances)
    ax.set_xticklabels(results[0].k_labels)

axes[0].plot([], [], "C0-", label="TM")
axes[0].plot([], [], "C1-", label="TE")
axes[0].legend()
plt.tight_layout()
plt.savefig("band_diagram_blaze.svg")
plt.show()
