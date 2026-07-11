"""
High-level solve interface for Blaze2D band structure calculations.

Provides :func:`solve` — a single entry point for computing 2D photonic crystal
band diagrams, supporting both individual runs and multi-parameter sweeps.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from blaze._native import OperatorDataExtractor, BulkDriver

console = Console()

# ---------------------------------------------------------------------------
# Lattice presets
# ---------------------------------------------------------------------------
_LATTICE_VECTORS = {
    "square":      [[1.0, 0.0], [0.0, 1.0]],
    "hexagonal":   [[1.0, 0.0], [0.5, math.sqrt(3) / 2]],
    "rectangular": [[1.0, 0.0], [0.0, 1.5]],
}

_K_PATH_CORNERS = {
    "square":      [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0]],
    "hexagonal":   [[0, 0], [0.5, 0], [1/3, 1/3], [0, 0]],
    "rectangular": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0, 0]],
}

_K_PATH_LABELS = {
    "square":      ["Γ", "X", "M", "Γ"],
    "hexagonal":   ["Γ", "M", "K", "Γ"],
    "rectangular": ["Γ", "X", "S", "Y", "Γ"],
}

_PATH_PRESETS = {
    "square":      "square",
    "hexagonal":   "hexagonal",
    "rectangular": "rectangular",
}

# Allowed lattice names (with common aliases)
_LATTICE_ALIASES = {
    "square": "square",
    "hex": "hexagonal",
    "hexagonal": "hexagonal",
    "triangular": "hexagonal",
    "rectangular": "rectangular",
    "rect": "rectangular",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_lattice(name: str) -> str:
    """Resolve lattice name aliases → canonical name."""
    key = name.strip().lower()
    if key not in _LATTICE_ALIASES:
        allowed = ", ".join(sorted(_LATTICE_ALIASES.keys()))
        raise ValueError(f"Unknown lattice_type '{name}'. Choose from: {allowed}")
    return _LATTICE_ALIASES[key]


def _make_kpath(corners: list[list[float]], n_per_seg: int) -> list[list[float]]:
    """Linearly interpolate between high-symmetry corner points.

    If the path is closed (first corner == last corner, e.g. Γ→…→Γ), the
    final point is nudged by a tiny offset so the Rust solver recomputes it
    with warm-start from the penultimate k-point instead of reusing the
    first Γ result.  This preserves band continuity near degenerate points.
    """
    pts: list[list[float]] = []
    for i in range(len(corners) - 1):
        s, e = np.array(corners[i]), np.array(corners[i + 1])
        for j in range(n_per_seg):
            pts.append((s + j / n_per_seg * (e - s)).tolist())

    last = list(corners[-1])
    # Nudge closed-path endpoint so Rust doesn't reuse the cached Γ result
    if np.allclose(corners[0], corners[-1], atol=1e-12):
        last = [c + 1e-10 for c in last]
    pts.append(last)
    return pts


def _build_range(spec, name: str) -> list[float]:
    """
    Convert a sweep specification [start, stop, step] → list of values.

    *start* and *stop* are both inclusive.  If *step* does not evenly divide
    the interval, a note is printed and *stop* is appended.
    """
    if len(spec) != 3:
        raise ValueError(f"{name}: expected [start, stop, step], got {spec}")

    start, stop, step = float(spec[0]), float(spec[1]), float(spec[2])

    # Sanity checks
    if step == 0:
        raise ValueError(f"{name}: step cannot be zero")
    if (stop - start) != 0 and math.copysign(1, stop - start) != math.copysign(1, step):
        raise ValueError(
            f"{name}: step sign ({step:+g}) does not match direction "
            f"start={start} → stop={stop}"
        )

    values: list[float] = []
    current = start
    if step > 0:
        while current <= stop + 1e-12:
            values.append(round(current, 10))
            current += step
    else:
        while current >= stop - 1e-12:
            values.append(round(current, 10))
            current += step

    # Ensure stop is included
    if abs(values[-1] - stop) > 1e-12:
        values.append(round(stop, 10))
        console.print(
            f"  [yellow]ℹ[/yellow]  {name}: step {step} does not evenly divide "
            f"[{start}, {stop}]. Using: {values}"
        )

    return values


def _validate_radius(values: list[float]) -> None:
    for v in values:
        if v <= 0 or v >= 0.5:
            raise ValueError(
                f"radius_atom={v} out of range. Must be in (0, 0.5) "
                f"(units of lattice constant a)."
            )


def _validate_epsilon(values: list[float], name: str) -> None:
    for v in values:
        if v < 1.0:
            raise ValueError(f"{name}={v} invalid. Permittivity must be ≥ 1.")


def _validate_resolution(values: list[int]) -> None:
    for v in values:
        if v < 4:
            raise ValueError(f"resolution={v} too small. Must be ≥ 4.")
        if v > 512:
            raise ValueError(f"resolution={v} too large. Must be ≤ 512.")


def _is_sweep(val) -> bool:
    """True if *val* is a list/tuple with 3 numeric elements (start, stop, step)."""
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 3:
        return any(not isinstance(v, str) for v in val)
    return False


def _sort_bands(freqs: np.ndarray) -> np.ndarray:
    """Re-index bands so each band varies smoothly along the k-path.

    Uses nearest-neighbour matching: at each k-point, the permutation of
    band indices is chosen to minimise the total squared difference to the
    previous k-point.  This removes the jumps that occur when the
    eigensolver returns degenerate eigenvalues in a different order.

    Parameters
    ----------
    freqs : np.ndarray, shape (n_k, n_bands)

    Returns
    -------
    np.ndarray, shape (n_k, n_bands) — reordered copy.
    """
    from scipy.optimize import linear_sum_assignment

    out = freqs.copy()
    n_k, n_b = out.shape
    for i in range(1, n_k):
        # Cost matrix: C[old_band, new_band] = (f_prev[old] - f_cur[new])²
        cost = (out[i - 1, :, None] - out[i, None, :]) ** 2
        _, col_ind = linear_sum_assignment(cost)
        out[i] = out[i, col_ind]
    return out


def _is_pol_list(val) -> bool:
    """True if *val* is a list of polarisation strings."""
    return isinstance(val, (list, tuple)) and all(isinstance(v, str) for v in val)


# ---------------------------------------------------------------------------
# TOML generation for the bulk driver
# ---------------------------------------------------------------------------

def _generate_sweep_toml(
    lattice: str,
    base_eps_bg: float,
    base_eps_atom: float,
    base_radius: float,
    base_pol: str,
    sweeps: list[dict],
    n_bands: int,
    resolution: int,
    points_per_segment: int,
    k_path: list[list[float]] | None,
    threads: int,
) -> str:
    """Build a TOML config string for the bulk driver."""
    lines = [
        '[bulk]',
        f'threads = {threads}',
        'verbose = false',
        '',
        '[solver]',
        'type = "maxwell"',
        '',
        '[defaults]',
        f'eps_bg = {base_eps_bg}',
        f'resolution = {resolution}',
        f'polarization = "{base_pol}"',
        '',
    ]

    # Sweeps
    for sw in sweeps:
        lines.append('[[sweeps]]')
        lines.append(f'parameter = "{sw["parameter"]}"')
        if "values" in sw:
            vals = sw["values"]
            if isinstance(vals[0], str):
                vals_str = ", ".join(f'"{v}"' for v in vals)
            else:
                vals_str = ", ".join(str(v) for v in vals)
            lines.append(f'values = [{vals_str}]')
        else:
            lines.append(f'min = {sw["min"]}')
            lines.append(f'max = {sw["max"]}')
            lines.append(f'step = {sw["step"]}')
        lines.append('')

    # Geometry
    lines.append('[geometry]')
    lines.append(f'eps_bg = {base_eps_bg}')
    lines.append('')
    lines.append('[geometry.lattice]')
    lines.append(f'type = "{lattice}"')
    lines.append('a = 1.0')
    lines.append('')
    lines.append('[[geometry.atoms]]')
    lines.append('pos = [0.0, 0.0]')
    lines.append(f'radius = {base_radius}')
    lines.append(f'eps_inside = {base_eps_atom}')
    lines.append('')

    # Grid
    lines.append('[grid]')
    lines.append(f'nx = {resolution}')
    lines.append(f'ny = {resolution}')
    lines.append('lx = 1.0')
    lines.append('ly = 1.0')
    lines.append('')

    # Path
    lines.append('[path]')
    if k_path is not None:
        # Custom k-path: write explicit points
        pts_str = ", ".join(f"[{p[0]}, {p[1]}]" for p in k_path)
        lines.append(f'points = [{pts_str}]')
    else:
        lines.append(f'preset = "{_PATH_PRESETS[lattice]}"')
        lines.append(f'segments_per_leg = {points_per_segment}')
    lines.append('')

    # Eigensolver
    lines.append('[eigensolver]')
    lines.append(f'n_bands = {n_bands}')
    lines.append('max_iter = 200')
    lines.append('tol = 1e-4')
    lines.append('')

    # Output — in-memory, no disk writes
    lines.append('[output]')
    lines.append('mode = "full"')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class BandResult:
    """Result of a single band structure calculation.

    Attributes
    ----------
    freqs : np.ndarray
        Frequencies, shape ``(n_kpoints, n_bands)``, in units of *c/a*.
    distances : np.ndarray
        Cumulative k-path distance for plotting, shape ``(n_kpoints,)``.
    k_points : np.ndarray
        Fractional reciprocal-space coordinates, shape ``(n_kpoints, 2)``.
    lattice_type : str
        Canonical lattice name (``"square"``, ``"hexagonal"``, …).
    polarization : str
        ``"TM"`` or ``"TE"``.
    epsilon_background : float
        Background permittivity used.
    epsilon_atoms : float
        Atom (inclusion) permittivity used.
    radius_atom : float
        Atom radius in units of *a*.
    resolution : int
        Grid resolution used for this calculation.
    k_labels : list[str]
        High-symmetry-point labels for tick marks, e.g. ``["Γ", "X", "M", "Γ"]``.
    k_label_distances : list[float]
        k-path distances at each high-symmetry point (for ``ax.set_xticks``).
    """

    def __init__(self, freqs, distances, k_points, lattice_type, polarization,
                 eps_bg, eps_atom, radius, resolution, points_per_segment,
                 k_labels, k_label_distances):
        self.freqs = np.asarray(freqs)
        self.distances = np.asarray(distances)
        self.k_points = np.asarray(k_points)
        self.lattice_type = lattice_type
        self.polarization = polarization
        self.epsilon_background = eps_bg
        self.epsilon_atoms = eps_atom
        self.radius_atom = radius
        self.resolution = resolution
        self.n_bands = self.freqs.shape[1] if self.freqs.ndim == 2 else 0
        self.n_kpoints = len(self.distances)
        self.k_labels = k_labels
        self.k_label_distances = k_label_distances
        self._points_per_segment = points_per_segment

    def __repr__(self):
        return (
            f"BandResult(lattice={self.lattice_type!r}, pol={self.polarization!r}, "
            f"ε_bg={self.epsilon_background}, ε_atom={self.epsilon_atoms}, "
            f"r={self.radius_atom}, res={self.resolution}, "
            f"bands={self.n_bands}, k={self.n_kpoints})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(
    *,
    lattice_type: str = "square",
    resolution: Union[int, list] = 32,
    epsilon_background: Union[float, list] = 1.0,
    epsilon_atoms: Union[float, list] = 8.9,
    radius_atom: Union[float, list] = 0.2,
    polarization: Union[str, list] = "TM",
    n_bands: int = 8,
    points_per_segment: int = 15,
    k_path: list | None = None,
    threads: int = 0,
) -> Union[BandResult, list[BandResult]]:
    """Compute 2D photonic crystal band structures.

    This is the main entry point for Blaze2D.  Pass scalar parameters for a
    single band diagram, or pass ``[start, stop, step]`` lists to sweep over
    one or more parameters (executed in parallel via the Rust bulk driver).

    Parameters
    ----------
    lattice_type : str
        Lattice geometry.  One of:

        * ``"square"`` — square lattice, path Γ → X → M → Γ
        * ``"hexagonal"`` (or ``"hex"``, ``"triangular"``) — triangular
          lattice, path Γ → M → K → Γ
        * ``"rectangular"`` (or ``"rect"``) — rectangular lattice,
          path Γ → X → S → Y → Γ

    resolution : int or [start, stop, step]
        Real-space grid resolution per lattice constant (default: 32).
        Pass a 3-element list of integers to sweep, e.g. ``[16, 64, 16]``.
    epsilon_background : float or [start, stop, step]
        Relative permittivity of the background medium (≥ 1).
        Pass a 3-element list to sweep.
    epsilon_atoms : float or [start, stop, step]
        Relative permittivity of the cylindrical inclusions (≥ 1).
        Pass a 3-element list to sweep.
    radius_atom : float or [start, stop, step]
        Radius of each inclusion in units of the lattice constant *a*.
        Must be in ``(0, 0.5)``.  Pass a 3-element list to sweep.
    polarization : str or list[str]
        ``"TM"`` or ``"TE"``.  Pass ``["TM", "TE"]`` to compute both.
    n_bands : int
        Number of bands to compute (default: 8).
    points_per_segment : int
        Number of k-points per high-symmetry segment (default: 15).
    k_path : list of [kx, ky] or None
        Custom k-path in fractional reciprocal coordinates.  Overrides
        *points_per_segment* and the lattice preset.  If given, high-symmetry
        tick labels are omitted.
    threads : int
        Thread count for parallel sweeps (0 = auto-detect, default).

    Returns
    -------
    BandResult or list[BandResult]
        A single :class:`BandResult` when no sweep parameters are used,
        or a list of results (one per parameter combination) for sweeps.

        Each :class:`BandResult` has:

        * ``freqs`` — ``np.ndarray`` of shape ``(n_kpoints, n_bands)``
          in units of *c/a* (same convention as MPB).
        * ``distances`` — cumulative k-path distance for the x-axis.
        * ``k_labels`` / ``k_label_distances`` — tick labels and positions.
        * ``polarization``, ``epsilon_background``, ``epsilon_atoms``,
          ``radius_atom`` — the parameters used for this run.

    Raises
    ------
    ValueError
        If parameters are out of range or inconsistent.

    Examples
    --------
    Single band diagram::

        result = blaze.solve(lattice_type="square", epsilon_background=12.0,
                             epsilon_atoms=1.0, radius_atom=0.2,
                             polarization="TM", n_bands=8)
        print(result.freqs.shape)  # (n_kpoints, 8)

    Radius sweep with both polarisations::

        results = blaze.solve(lattice_type="hexagonal",
                              epsilon_background=13.0, epsilon_atoms=1.0,
                              radius_atom=[0.2, 0.4, 0.05],
                              polarization=["TM", "TE"])
        for r in results:
            print(r)

    Resolution convergence study::

        results = blaze.solve(lattice_type="square",
                              resolution=[16, 64, 16],
                              epsilon_background=12.0, epsilon_atoms=1.0,
                              radius_atom=0.2, polarization="TM")
        for r in results:
            print(f"res={r.resolution}: band 1 max = {r.freqs[:, 0].max():.6f}")
    """
    # ------------------------------------------------------------------
    # 1. Resolve lattice
    # ------------------------------------------------------------------
    lattice = _resolve_lattice(lattice_type)

    # ------------------------------------------------------------------
    # 2. Parse sweep specifications
    # ------------------------------------------------------------------
    has_sweep = False
    sweep_defs: list[dict] = []

    # resolution
    if _is_sweep(resolution):
        res_vals = [int(v) for v in _build_range(resolution, "resolution")]
        _validate_resolution(res_vals)
        sweep_defs.append({
            "parameter": "resolution",
            "values": res_vals,
        })
        base_resolution = res_vals[0]
        has_sweep = True
    else:
        base_resolution = int(resolution)
        _validate_resolution([base_resolution])

    # epsilon_background
    if _is_sweep(epsilon_background):
        eps_bg_vals = _build_range(epsilon_background, "epsilon_background")
        _validate_epsilon(eps_bg_vals, "epsilon_background")
        sweep_defs.append({
            "parameter": "eps_bg",
            "values": eps_bg_vals,
        })
        base_eps_bg = eps_bg_vals[0]
        has_sweep = True
    else:
        base_eps_bg = float(epsilon_background)
        _validate_epsilon([base_eps_bg], "epsilon_background")

    # epsilon_atoms
    if _is_sweep(epsilon_atoms):
        eps_atom_vals = _build_range(epsilon_atoms, "epsilon_atoms")
        _validate_epsilon(eps_atom_vals, "epsilon_atoms")
        sweep_defs.append({
            "parameter": "atom0.eps_inside",
            "values": eps_atom_vals,
        })
        base_eps_atom = eps_atom_vals[0]
        has_sweep = True
    else:
        base_eps_atom = float(epsilon_atoms)
        _validate_epsilon([base_eps_atom], "epsilon_atoms")

    # radius_atom
    if _is_sweep(radius_atom):
        radius_vals = _build_range(radius_atom, "radius_atom")
        _validate_radius(radius_vals)
        sweep_defs.append({
            "parameter": "atom0.radius",
            "values": radius_vals,
        })
        base_radius = radius_vals[0]
        has_sweep = True
    else:
        base_radius = float(radius_atom)
        _validate_radius([base_radius])

    # polarization
    if _is_pol_list(polarization):
        for p in polarization:
            if p.upper() not in ("TM", "TE"):
                raise ValueError(f"Unknown polarization '{p}'. Use 'TM' or 'TE'.")
        sweep_defs.append({
            "parameter": "polarization",
            "values": [p.upper() for p in polarization],
        })
        base_pol = polarization[0].upper()
        has_sweep = True
    else:
        base_pol = str(polarization).upper()
        if base_pol not in ("TM", "TE"):
            raise ValueError(f"Unknown polarization '{polarization}'. Use 'TM' or 'TE'.")

    # ------------------------------------------------------------------
    # 3. Compute k-path and label info
    # ------------------------------------------------------------------
    if k_path is not None:
        kpts = k_path
        k_labels = []
        k_label_dists = []
    else:
        corners = _K_PATH_CORNERS[lattice]
        kpts = _make_kpath(corners, points_per_segment)
        k_labels = _K_PATH_LABELS[lattice]
        # Label positions: first point of each segment + last point
        dists = _kpath_distances(kpts)
        k_label_dists = [dists[i * points_per_segment] for i in range(len(corners) - 1)]
        k_label_dists.append(dists[-1])

    # ------------------------------------------------------------------
    # 4. Dispatch: single solve vs sweep
    # ------------------------------------------------------------------
    if not has_sweep:
        return _run_single(
            lattice, base_eps_bg, base_eps_atom, base_radius, base_pol,
            n_bands, base_resolution, kpts, points_per_segment, k_labels, k_label_dists,
        )
    else:
        return _run_sweep(
            lattice, base_eps_bg, base_eps_atom, base_radius, base_pol,
            sweep_defs, n_bands, base_resolution, points_per_segment,
            kpts if k_path is not None else None,
            k_labels, k_label_dists, threads,
        )


def _kpath_distances(kpts: list) -> np.ndarray:
    """Cumulative Euclidean distance along a k-path."""
    pts = np.array(kpts)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


# ---------------------------------------------------------------------------
# Single solve (direct Rust call)
# ---------------------------------------------------------------------------

def _run_single(lattice, eps_bg, eps_atom, radius, pol,
                n_bands, resolution, kpts, pps, k_labels, k_label_dists):
    """Run a single band structure calculation via the Rust eigensolver."""
    console.print(Panel(
        f"[bold]Blaze2D[/bold]  ·  {lattice} lattice  ·  {pol}\n"
        f"ε_bg={eps_bg}  ε_atom={eps_atom}  r/a={radius}  "
        f"res={resolution}  bands={n_bands}  k-pts={len(kpts)}",
        title="Band Structure", border_style="blue",
    ))

    lattice_vecs = _LATTICE_VECTORS[lattice]
    atoms = [{"pos": [0.0, 0.0], "radius": radius, "eps_inside": eps_atom}]

    result = OperatorDataExtractor.solve_k_path(
        lattice_vectors=lattice_vecs,
        atoms=atoms,
        eps_bg=eps_bg,
        k_points=kpts,
        polarization=pol,
        resolution=resolution,
        n_bands=n_bands,
    )

    freqs = _sort_bands(np.array(result["freqs"]))
    distances = np.array(result["distances"])
    k_points = np.array(result["k_points"])

    # Summary
    table = Table(title="Computed Bands", show_lines=False)
    table.add_column("Band", style="cyan", justify="right")
    table.add_column("f_min (c/a)", justify="right")
    table.add_column("f_max (c/a)", justify="right")
    for b in range(min(n_bands, 10)):
        band = freqs[:, b]
        table.add_row(str(b + 1), f"{band.min():.6f}", f"{band.max():.6f}")
    if n_bands > 10:
        table.add_row("…", "…", "…")
    console.print(table)

    return BandResult(
        freqs=freqs, distances=distances, k_points=k_points,
        lattice_type=lattice, polarization=pol,
        eps_bg=eps_bg, eps_atom=eps_atom, radius=radius,
        resolution=resolution, points_per_segment=pps,
        k_labels=k_labels, k_label_distances=k_label_dists,
    )


# ---------------------------------------------------------------------------
# Sweep solve (bulk driver)
# ---------------------------------------------------------------------------

def _run_sweep(lattice, base_eps_bg, base_eps_atom, base_radius, base_pol,
               sweep_defs, n_bands, resolution, pps, k_path_override,
               k_labels, k_label_dists, threads):
    """Run a parameter sweep via the Rust bulk driver."""

    # Count total jobs
    total = 1
    for sw in sweep_defs:
        total *= len(sw["values"])

    # Print sweep summary
    table = Table(title="Parameter Sweep", show_lines=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Values", style="green")
    table.add_column("Count", justify="right")
    for sw in sweep_defs:
        name = sw["parameter"]
        vals = sw["values"]
        if len(vals) <= 8:
            vals_str = ", ".join(str(v) for v in vals)
        else:
            vals_str = f"{vals[0]} … {vals[-1]}"
        table.add_row(name, vals_str, str(len(vals)))
    console.print(table)
    console.print(f"  Total jobs: [bold]{total}[/bold]  ·  {lattice} lattice  ·  "
                  f"res={resolution}  bands={n_bands}")

    # Generate TOML config
    toml_str = _generate_sweep_toml(
        lattice=lattice,
        base_eps_bg=base_eps_bg,
        base_eps_atom=base_eps_atom,
        base_radius=base_radius,
        base_pol=base_pol,
        sweeps=sweep_defs,
        n_bands=n_bands,
        resolution=resolution,
        points_per_segment=pps,
        k_path=k_path_override,
        threads=threads,
    )

    # Write to temp file and run
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_str)
        toml_path = f.name

    try:
        driver = BulkDriver(toml_path, threads=threads)

        # Suppress Rust-side stdout (banner + indicatif progress bar) so only
        # the Python Rich output is shown.  We redirect fd 1 to /dev/null
        # during execution, then restore it.
        sys.stdout.flush()
        saved_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)

        raw_results = []
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Computing…", total=total)
                for result in driver.run_streaming():
                    raw_results.append(result)
                    progress.advance(task)
        finally:
            # Restore stdout
            os.dup2(saved_fd, 1)
            os.close(saved_fd)

    finally:
        Path(toml_path).unlink(missing_ok=True)

    # Convert to BandResult objects
    band_results = []
    for r in raw_results:
        if r.get("result_type") != "maxwell":
            continue

        params = r.get("sweep_values", {})
        eps_bg = params.get("eps_bg", base_eps_bg)
        eps_atom = params.get("atom0.eps_inside", base_eps_atom)
        radius = params.get("atom0.radius", base_radius)
        pol = params.get("polarization", base_pol)
        res = int(params.get("resolution", resolution))

        # bands: list[list[float]] — bands[k_idx][band_idx]
        raw_bands = r["bands"]
        freqs = _sort_bands(np.array(raw_bands))
        distances = np.array(r["distances"])
        kpoints = np.array(r["k_path"])

        band_results.append(BandResult(
            freqs=freqs, distances=distances, k_points=kpoints,
            lattice_type=lattice, polarization=pol,
            eps_bg=eps_bg, eps_atom=eps_atom, radius=radius,
            resolution=res, points_per_segment=pps,
            k_labels=k_labels, k_label_distances=k_label_dists,
        ))

    # Summary
    console.print(f"\n  [green]✓[/green]  {len(band_results)} band structures computed.")
    return band_results
