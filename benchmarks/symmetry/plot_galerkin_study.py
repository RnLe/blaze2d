#!/usr/bin/env python3
"""Render D04 frequency convergence without treating cutoff changes as speedups."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


def main():
    root = Path(__file__).resolve().parent
    audit = json.loads((root / "D04-audit.json").read_text())
    if audit["issues"]:
        raise SystemExit("Resolve audit issues before rendering accepted evidence")
    rows = [row for row in audit["rows"] if row["study"] == "convergence"]
    colors = {17: "#8A6428", 33: "#416BA9", 65: "#087F75"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "D04",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for i, polarization in enumerate(("TM", "TE")):
        for j, (point, label) in enumerate((("hex-gamma", "Γ"), ("hex-k", "K"))):
            ax = axes[i, j]
            for n, color in colors.items():
                for material, style in (("analytic", "-"), ("raw", "--")):
                    series = sorted((r for r in rows if r["key"][0] == point
                        and r["key"][1] == n and r["key"][3] == polarization
                        and r["key"][-1] == material), key=lambda r: r["key"][5])
                    ax.loglog([r["key"][5] for r in series],
                              [r["max_scaled_frequency_error"] for r in series],
                              style, color=color, marker="o", markersize=3, linewidth=1.7)
            ax.axhline(1e-3, color="#B13D4B", linestyle=":", linewidth=1.5)
            ax.set_title(f"{polarization} · {label}", loc="left", fontweight="bold")
            ax.set_ylim(3e-6, 1)
            ax.set_xlim(2.2, 32)
            ax.grid(which="major", alpha=.18)
            ax.set_xticks([2.5, 5, 10, 20, 28], ["2.5", "5", "10", "20", "28"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            if i == 1:
                ax.set_xlabel("Reciprocal-metric cutoff R (2π units)")
            if j == 0:
                ax.set_ylabel("Maximum scaled frequency error")
    handles = [Line2D([], [], color=c, label=f"{n}² grid", linewidth=2)
               for n, c in colors.items()]
    handles += [Line2D([], [], color="#172B3A", linestyle=s, label=l, linewidth=2)
                for s, l in (("-", "Analytic material"), ("--", "Raw material"))]
    handles.append(Line2D([], [], color="#B13D4B", linestyle=":", label="Frequency target 1e-3"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.53, 1.005))
    fig.subplots_adjust(top=.86, bottom=.14, hspace=.23, wspace=.11,
                        left=.095, right=.985)
    fig.text(.095, .025,
        "Eight bands; error = |ω − ωref| / max(1, |ωref|), against each grid/material's full rectangle.\n"
        "Changed finite discretization. No symmetry speedup or continuum convergence order is inferred.",
        fontsize=10, color="#516675")
    fig.savefig(root / "D04-convergence.png", dpi=180)
    svg = root / "D04-convergence.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
