"""Recompute the thesis TM mode comparison in a common Blaze/MPB basis.

Requires NumPy, Matplotlib, and Meep with MPB. Weighted unitary Procrustes
alignment changes only the Blaze basis within degenerate eigenspaces.
All displayed fields come from fresh solves; raw fields are also exported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GROUPS = [(0,), (1, 2), (3,), (4, 5), (6,), (7,)]
DARK_STEEL = "#4F5F6B"
CMAP = LinearSegmentedColormap.from_list("thesis_field", [
    (0.00, "#080E18"), (0.12, "#2A3744"), (0.25, DARK_STEEL),
    (0.40, "#4D7B9E"), (0.55, "#4E9AE1"), (0.70, "#A5C6DF"),
    (0.80, "#E3D5BF"), (0.90, "#E3B064"), (0.97, "#EBA538"),
    (1.00, "#FFF3DC"),
], N=256)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array(record, key):
    value = record["arrays"][key]
    data = np.asarray(value["data"], dtype=np.float64)
    if value["dtype"] == "complex128":
        data = data.view(np.complex128)
    return data.reshape(value["shape"])


def intensity(fields):
    values = abs(fields) ** 2
    return values / values.max(axis=(-2, -1), keepdims=True)


def solve(blaze, output):
    config = output / "config.toml"
    shutil.copyfile(HERE / "config.toml", config)
    with (output / "blaze-result.json").open("w") as stdout, \
            (output / "blaze-run.log").open("w") as stderr:
        subprocess.run([str(blaze), "run", str(config), "--threads", "1"],
                       stdout=stdout, stderr=stderr, check=True)
    record = json.loads((output / "blaze-result.json").read_text())["results"][0]
    if not record["metadata"]["converged"]:
        raise RuntimeError("Blaze did not converge")
    blaze_fields = array(record, "eigenvectors")[0, :8]
    blaze_freqs = array(record, "frequencies")[0, :8]

    import meep as mp
    from meep import mpb
    solver = mpb.ModeSolver(
        num_bands=8, k_points=[mp.Vector3(.5, .5)],
        geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1)),
        geometry=[mp.Cylinder(radius=.2, material=mp.Medium(epsilon=8.9),
                              center=mp.Vector3(), height=mp.inf)],
        default_material=mp.Medium(epsilon=1), resolution=64, dimensions=2,
        tolerance=1e-12, deterministic=True,
    )
    solver.run_tm()
    mpb_fields = np.stack([
        np.asarray(solver.get_efield(band, bloch_phase=False))[:, :, 0, 2]
        for band in range(1, 9)
    ])
    mpb_freqs = np.asarray(solver.all_freqs[-1])
    epsilon = np.asarray(solver.get_epsilon())
    np.savez_compressed(output / "raw-fields.npz", blaze=blaze_fields,
                        blaze_frequencies=blaze_freqs, mpb=mpb_fields,
                        mpb_frequencies=mpb_freqs, mpb_epsilon=epsilon)
    return record, blaze_fields, blaze_freqs, mpb_fields, mpb_freqs, epsilon, {
        "meep": mp.__version__, "mpb": list(mpb.__version__),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
    }


def align_to_mpb(blaze, mpb, epsilon, frequencies):
    """Align physical periodic fields using a common epsilon-weighted metric."""
    reference = np.fft.ifftshift(mpb.transpose(0, 2, 1), axes=(-2, -1))
    weight = np.sqrt(np.fft.ifftshift(epsilon.T))
    transform = np.eye(8, dtype=np.complex128)
    matched = blaze.copy()
    records = []
    for group in GROUPS:
        indices = list(group)
        split = np.ptp(frequencies[indices]) / np.mean(frequencies[indices])
        if split > 1e-10:
            raise ValueError(f"Bands {indices} are not numerically degenerate")
        x = (blaze[indices] * weight).reshape(len(group), -1).T
        y = (reference[indices] * weight).reshape(len(group), -1).T
        x /= np.linalg.norm(x, axis=0)
        y /= np.linalg.norm(y, axis=0)
        left, _, right = np.linalg.svd(x.conj().T @ y)
        # Fields are stored as rows; Procrustes acts on column eigenvectors.
        matrix = (left @ right).T
        transform[np.ix_(group, group)] = matrix
        matched[indices] = np.einsum("ab,bij->aij", matrix, blaze[indices])
        total_before = np.sum(abs(blaze[indices]) ** 2, axis=0)
        total_after = np.sum(abs(matched[indices]) ** 2, axis=0)
        invariant_error = np.max(abs(total_after - total_before)) / total_before.max()
        if invariant_error > 1e-12:
            raise RuntimeError("Basis change did not preserve the subspace intensity")
        records.append({"bands": [i + 1 for i in group],
                        "relative_frequency_split": float(split),
                        "unitarity_defect": float(np.linalg.norm(matrix @ matrix.conj().T - np.eye(len(group)))),
                        "summed_intensity_relative_error": float(invariant_error)})
    overlaps = []
    for x, y in zip(matched * weight, reference * weight):
        overlaps.append(float(abs(np.vdot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))))
    return matched, reference, transform, records, overlaps


def diagnostics(record, blaze, bf, mpb, mf, epsilon, transform):
    # MPB exports (x,y) on a centered cell; Blaze exports (y,x) with the rod
    # at the periodic origin. Use the MPB epsilon map as a common positive metric.
    reference = np.fft.ifftshift(mpb.transpose(0, 2, 1), axes=(-2, -1))
    weight = np.sqrt(np.fft.ifftshift(epsilon.T))
    overlaps = []
    for indices in GROUPS:
        group = list(indices)
        x = (blaze[group] * weight).reshape(len(group), -1).T
        y = (reference[group] * weight).reshape(len(group), -1).T
        qx, qy = np.linalg.qr(x)[0], np.linalg.qr(y)[0]
        overlaps.append({"bands": [i + 1 for i in group],
                         "principal_overlaps": np.linalg.svd(qx.conj().T @ qy, compute_uv=False).tolist()})

    # Bound the displayed residuals using the measured source residuals,
    # the triangle inequality, and 1 <= epsilon <= 8.9. No visual fit enters
    # this certificate. Assigned eigenvalues are the coefficient-weighted means.
    eigenvalues = (2 * np.pi * bf) ** 2
    norms = np.linalg.norm(blaze.reshape(8, -1), axis=1)
    absolute = array(record, "absolute_residuals")[0, :8]
    mixed = np.einsum("ab,bij->aij", transform, blaze)
    bounds = []
    for i, coefficients in enumerate(transform):
        eigenvalue = np.sum(abs(coefficients) ** 2 * eigenvalues)
        bound = np.sum(abs(coefficients) * norms * (absolute + abs(eigenvalues - eigenvalue) * 8.9))
        bound /= np.linalg.norm(mixed[i])
        bounds.append(float(bound / (2 * eigenvalue - bound)))
    return {"blaze_frequencies": bf.tolist(), "mpb_frequencies": mf.tolist(),
            "relative_frequency_differences": (abs(bf / mf - 1)).tolist(),
            "max_relative_frequency_difference": float(max(abs(bf / mf - 1))),
            "displayed_source_residual_max": float(array(record, "residuals")[0, :8].max()),
            "displayed_residual_upper_bounds": bounds,
            "b_orthogonality_defect": record["metadata"]["certification"]["b_orthogonality_defect"],
            "weighted_subspace_comparisons": overlaps}


def plot(rows, frequencies, destination):
    fig = plt.figure(figsize=(14, 4.2), facecolor="white")
    grid = fig.add_gridspec(2, 8, hspace=.08, wspace=.05,
                           top=.84, bottom=.04, left=.06, right=.92)
    for row, fields in enumerate(rows):
        for col, field in enumerate(intensity(fields)):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(field, origin="lower", interpolation="gaussian",
                      cmap=CMAP, norm=Normalize(0, 1), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(DARK_STEEL)
                spine.set_linewidth(.6)
            if col == 0:
                ax.set_ylabel(["Blaze2D", "MPB"][row], fontsize=14,
                              fontweight="bold", labelpad=8, color=DARK_STEEL)
            if row == 0:
                ax.set_title(f"Band {col + 1}\n$\\omega = {frequencies[col]:.3f}$",
                             fontsize=11, fontweight="bold", pad=6, color=DARK_STEEL)
    cax = fig.add_axes([.935, .04, .013, .80])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1)), cax=cax)
    cb.set_label("$|\\mathbf{u}|^2$  (normalized per band)",
                 fontsize=12, labelpad=10, color=DARK_STEEL)
    cax.tick_params(labelsize=9, colors=DARK_STEEL)
    cb.outline.set_edgecolor(DARK_STEEL)
    cb.outline.set_linewidth(.6)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(destination.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blaze", type=Path, default=ROOT / "target/release/blaze2d")
    parser.add_argument("--reference", type=Path,
                        default=ROOT.parent / "master-thesis/assets/figures/blaze2d/u2_mode_comparison.svg")
    parser.add_argument("--output", type=Path, default=ROOT / "target/thesis-modes")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record, blaze, bf, mpb, mf, epsilon, versions = solve(args.blaze.resolve(), output)
    centered = np.fft.fftshift(blaze, axes=(-2, -1))
    matched, reference, transform, fits, overlaps = align_to_mpb(blaze, mpb, epsilon, bf)
    mpb_display = np.fft.fftshift(reference, axes=(-2, -1))
    blaze_display = np.fft.fftshift(matched, axes=(-2, -1))
    plot([centered, mpb_display], bf, output / "modes-raw")
    plot([blaze_display, mpb_display], bf, output / "u2-mode-comparison")
    np.savez_compressed(output / "display-fields.npz", blaze=blaze_display, mpb=mpb_display,
                        blaze_transform=transform)
    report = diagnostics(record, blaze, bf, mpb, mf, epsilon, transform)
    report.update({"reference_svg": str(args.reference.resolve()),
                   "reference_sha256": sha256(args.reference),
                   "binary_sha256": sha256(args.blaze),
                   "config_sha256": sha256(HERE / "config.toml"),
                   "blaze_build": record["metadata"]["build"], "versions": versions,
                   "basis_matching": fits, "aligned_mode_overlaps": overlaps,
                   "normalized_intensity_rms_differences": np.sqrt(np.mean((intensity(blaze_display) - intensity(mpb_display)) ** 2, axis=(-2, -1))).tolist(),
                   "matching_scope": "epsilon-weighted unitary Procrustes within Blaze doublets; MPB eigenvectors unchanged",
                   "plot_coordinates": "both rows use (y,x) on the centered unit cell",
                   "normalization": "per displayed field peak; colors show abs(periodic_Ez)**2",
                   "frequency_label": "thesis omega notation retained; values are omega*a/(2*pi*c)",
                   "differences_from_thesis": ["Band 8 rounds to 0.989 instead of 0.990",
                       "Fresh f64 solutions; original numerical errors are not reinstated",
                       "Both solvers show the same physical basis, as requested, rather than independent doublet representatives",
                       "The saved plotting script had later changed to 128 pixels and six panels; the thesis caption and SVG specify 64 pixels and eight panels"]})
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved reproduction to {output}")
    print(f"Maximum relative frequency difference: {report['max_relative_frequency_difference']:.6g}")
    print(f"Displayed source residual maximum: {report['displayed_source_residual_max']:.6g}")


if __name__ == "__main__":
    main()
