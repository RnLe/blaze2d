#!/usr/bin/env python3
"""Summarize pipeline inspection dumps and surface potential anomalies.

This script scans reference-data pipeline directories produced by `mpb2d-cli --dump-pipeline`
and prints a compact report with statistics for:
  * dielectric real/Fourier snapshots
  * operator snapshot CSVs (real space & spectra)
    * residual snapshot CSVs (raw/projected/preconditioned residuals)
  * operator iteration traces
  * FFT workspace raw/report dumps

It emits warning hints when data is missing or suspicious (e.g., gradients absent,
residuals stagnating, |k+G|^2 touching zero, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Small utilities


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    rank = (pct / 100) * (len(values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    fraction = rank - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


@dataclass
class Summary:
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(value)

    def describe(self) -> Optional[Dict[str, float]]:
        if not self.values:
            return None
        sorted_vals = sorted(self.values)
        return {
            "count": len(sorted_vals),
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": statistics.fmean(sorted_vals),
            "median": percentile(sorted_vals, 50),
            "p90": percentile(sorted_vals, 90),
            "p99": percentile(sorted_vals, 99),
        }


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        # CSV fields may appear as floats (e.g., "3.0"); coerce via float before int.
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> Iterable[Dict[str, str]]:
    record_file_timestamp(path)
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


INPUT_FILE_TIMESTAMPS: Dict[str, float] = {}


def record_file_timestamp(path: Path) -> None:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        return
    try:
        INPUT_FILE_TIMESTAMPS[str(resolved)] = resolved.stat().st_mtime
    except FileNotFoundError:
        return


POL_TOKENS = {"te", "tm"}


def sanitize_group_label(name: str) -> str:
    cleaned = [ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name]
    sanitized = "".join(cleaned).strip("_")
    return sanitized or "config"


def infer_config_group(directory: Path) -> str:
    name = directory.stem
    if name.endswith("_pipeline"):
        name = name[: -len("_pipeline")]
    tokens = []
    for token in name.split("_"):
        if token.lower() in POL_TOKENS:
            continue
        tokens.append(token)
    if not tokens:
        tokens = [name]
    return "_".join(tokens)


def timestamps_for_directories(directories: Sequence[Path]) -> Dict[str, float]:
    if not directories:
        return {}
    resolved_dirs = [directory.resolve() for directory in directories]
    subset: Dict[str, float] = {}
    for path_str, mtime in INPUT_FILE_TIMESTAMPS.items():
        candidate = Path(path_str)
        for directory in resolved_dirs:
            try:
                candidate.relative_to(directory)
            except ValueError:
                continue
            subset[path_str] = mtime
            break
    return subset


# -----------------------------------------------------------------------------
# Snapshot analysis

SNAPSHOT_REGEX = "operator_snapshot_"


def parse_snapshot_ident(path: Path) -> str:
    name = path.stem.replace("operator_snapshot_", "")
    return name


@dataclass
class SnapshotReport:
    identifier: str
    path: str
    field: Optional[Dict[str, float]]
    theta: Optional[Dict[str, float]]
    grad: Optional[Dict[str, float]]
    grad_present: bool
    eps_grad_present: bool
    k_frac: Optional[Tuple[float, float]]


def analyze_operator_snapshot(path: Path) -> SnapshotReport:
    field_stats = Summary()
    theta_stats = Summary()
    grad_stats = Summary()
    grad_columns: Tuple[str, ...] = (
        "re_grad_x",
        "im_grad_x",
        "re_grad_y",
        "im_grad_y",
    )
    eps_grad_columns: Tuple[str, ...] = (
        "re_eps_grad_x",
        "im_eps_grad_x",
        "re_eps_grad_y",
        "im_eps_grad_y",
    )
    grad_present = False
    eps_grad_present = False
    first_row: Optional[Dict[str, str]] = None
    for row in load_csv(path):
        if first_row is None:
            first_row = row
        field_stats.add(math.hypot(safe_float(row.get("re_field", "nan")), safe_float(row.get("im_field", "nan"))))
        theta_stats.add(math.hypot(safe_float(row.get("re_theta", "nan")), safe_float(row.get("im_theta", "nan"))))
        if all(col in row for col in grad_columns):
            grad_present = True
            grad_x = complex(safe_float(row["re_grad_x"]), safe_float(row["im_grad_x"]))
            grad_y = complex(safe_float(row["re_grad_y"]), safe_float(row["im_grad_y"]))
            grad_stats.add(math.hypot(abs(grad_x), abs(grad_y)))
        if all(col in row for col in eps_grad_columns):
            eps_grad_present = True
    k_frac: Optional[Tuple[float, float]] = None
    if first_row:
        k_frac = (
            safe_float(first_row.get("k_frac_x", "nan")),
            safe_float(first_row.get("k_frac_y", "nan")),
        )
    return SnapshotReport(
        identifier=parse_snapshot_ident(path),
        path=str(path),
        field=field_stats.describe(),
        theta=theta_stats.describe(),
        grad=grad_stats.describe() if grad_present else None,
        grad_present=grad_present,
        eps_grad_present=eps_grad_present,
        k_frac=k_frac,
    )


@dataclass
class ResidualSnapshotReport:
    identifier: str
    path: str
    stage: str
    iteration: Optional[int]
    band_index: Optional[int]
    k_index: Optional[int]
    stats: Optional[Dict[str, float]]


def analyze_residual_snapshot(path: Path) -> ResidualSnapshotReport:
    magnitude = Summary()
    first_row: Optional[Dict[str, str]] = None
    for row in load_csv(path):
        if first_row is None:
            first_row = row
        magnitude.add(
            math.hypot(
                safe_float(row.get("re_field", "nan")),
                safe_float(row.get("im_field", "nan")),
            )
        )
    identifier = path.stem.replace("residual_snapshot_", "")
    iteration = safe_int(first_row.get("iteration")) if first_row else None
    band_index = safe_int(first_row.get("band_index")) if first_row else None
    k_index = safe_int(first_row.get("k_index")) if first_row else None
    stage = (first_row.get("stage") if first_row else "unknown") or "unknown"
    stage = stage.lower()
    return ResidualSnapshotReport(
        identifier=identifier,
        path=str(path),
        stage=stage,
        iteration=iteration,
        band_index=band_index,
        k_index=k_index,
        stats=magnitude.describe(),
    )


@dataclass
class SpectrumReport:
    identifier: str
    path: str
    field_hat: Optional[Dict[str, float]]
    theta_hat: Optional[Dict[str, float]]


def analyze_operator_spectrum(path: Path) -> SpectrumReport:
    field_stats = Summary()
    theta_stats = Summary()
    for row in load_csv(path):
        field_stats.add(safe_float(row.get("field_hat_mag", "nan")))
        theta_stats.add(safe_float(row.get("theta_hat_mag", "nan")))
    return SpectrumReport(
        identifier=parse_snapshot_ident(path).replace("_spectrum", ""),
        path=str(path),
        field_hat=field_stats.describe(),
        theta_hat=theta_stats.describe(),
    )


# -----------------------------------------------------------------------------
# Iteration trace analysis

THRESHOLDS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]


def iterations_to_threshold(sequence: Sequence[float], threshold: float) -> Optional[int]:
    for idx, value in enumerate(sequence):
        if value <= threshold:
            return idx
    return None


@dataclass
class IterationReport:
    path: str
    iterations: int
    final_max_residual: float
    final_avg_residual: float
    final_max_relative: float
    final_avg_relative: float
    final_relative_scale: float
    zig_zag_ratio: float
    threshold_hits: Dict[float, Optional[int]]
    preconditioner_trials: int
    projection_dims: Optional[Dict[str, Dict[str, float]]] = None
    projection_condition: Optional[Dict[str, float]] = None
    projection_fallbacks: int = 0


@dataclass
class FFTWorkspaceRawAnalysis:
    clamped: Optional[Dict[str, float]]
    raw: Optional[Dict[str, float]]
    clamped_bins: int = 0
    total_bins: int = 0

    @property
    def clamp_fraction(self) -> float:
        if self.total_bins == 0:
            return 0.0
        return self.clamped_bins / self.total_bins


def analyze_iteration_trace(path: Path) -> IterationReport:
    iterations: List[int] = []
    max_residuals: List[float] = []
    avg_residuals: List[float] = []
    max_relative: List[float] = []
    avg_relative: List[float] = []
    relative_scales: List[float] = []
    projection_orig = Summary()
    projection_reduced = Summary()
    projection_condition = Summary()
    projection_fallbacks = 0
    preconditioner_trials = 0
    for row in load_csv(path):
        iterations.append(int(row["iteration"]))
        max_residuals.append(abs(safe_float(row.get("max_residual", "nan"))))
        avg_residuals.append(abs(safe_float(row.get("avg_residual", "nan"))))
        max_relative.append(abs(safe_float(row.get("max_relative_residual", "nan"))))
        avg_relative.append(abs(safe_float(row.get("avg_relative_residual", "nan"))))
        relative_scales.append(abs(safe_float(row.get("max_relative_scale", "nan"))))
        preconditioner_trials += int(row.get("preconditioner_trials", 0))
        projection_orig.add(safe_float(row.get("projection_original_dim")))
        projection_reduced.add(safe_float(row.get("projection_reduced_dim")))
        projection_condition.add(safe_float(row.get("projection_condition_estimate")))
        if row.get("projection_fallback_used") not in {None, "", "0", "False", "false"}:
            projection_fallbacks += 1
    zig_zag = 0
    for prev, curr in zip(max_residuals, max_residuals[1:]):
        if curr > prev * 1.02:  # 2% uptick counts as zig-zag
            zig_zag += 1
    thresholds = {thr: iterations_to_threshold(max_residuals, thr) for thr in THRESHOLDS}
    return IterationReport(
        path=str(path),
        iterations=len(iterations),
        final_max_residual=max_residuals[-1] if max_residuals else math.nan,
        final_avg_residual=avg_residuals[-1] if avg_residuals else math.nan,
        final_max_relative=max_relative[-1] if max_relative else math.nan,
        final_avg_relative=avg_relative[-1] if avg_relative else math.nan,
        final_relative_scale=relative_scales[-1] if relative_scales else math.nan,
        zig_zag_ratio=zig_zag / max(1, len(max_residuals) - 1),
        threshold_hits=thresholds,
        preconditioner_trials=preconditioner_trials,
        projection_dims={
            "original": projection_orig.describe() or {},
            "reduced": projection_reduced.describe() or {},
        },
        projection_condition=projection_condition.describe(),
        projection_fallbacks=projection_fallbacks,
    )


# -----------------------------------------------------------------------------
# FFT workspace & dielectric stats


def analyze_fft_workspace_raw(path: Path) -> Optional[FFTWorkspaceRawAnalysis]:
    clamped_stats = Summary()
    raw_stats = Summary()
    clamped_bins = 0
    total_bins = 0
    saw_clamp_flag = False
    for row in load_csv(path):
        total_bins += 1
        clamped_stats.add(safe_float(row.get("k_plus_g_sq", "nan")))
        if "k_plus_g_sq_raw" in row:
            raw_val = safe_float(row.get("k_plus_g_sq_raw", "nan"))
        else:
            kx = safe_float(row.get("kx_plus_g", "nan"))
            ky = safe_float(row.get("ky_plus_g", "nan"))
            raw_val = kx * kx + ky * ky
        raw_stats.add(raw_val)
        clamp_flag = row.get("clamped")
        if clamp_flag is not None:
            saw_clamp_flag = True
            if safe_float(clamp_flag) > 0.5:
                clamped_bins += 1
    if total_bins == 0:
        return None
    if not saw_clamp_flag:
        clamped_bins = 0
    return FFTWorkspaceRawAnalysis(
        clamped=clamped_stats.describe(),
        raw=raw_stats.describe(),
        clamped_bins=clamped_bins,
        total_bins=total_bins,
    )


def analyze_fft_workspace_report(path: Path) -> Dict[str, object]:
    record_file_timestamp(path)
    data = json.loads(path.read_text())
    return {
        "bloch_norm": data.get("bloch_vector", {}).get("norm"),
        "k_plus_g_sq": data.get("k_plus_g_sq", {}),
        "workspace": data.get("workspace_buffers", {}),
        "mesh_size": data.get("dielectric_mesh_size"),
    }


def analyze_eps_real(path: Path) -> Optional[Dict[str, float]]:
    stats = Summary()
    for row in load_csv(path):
        stats.add(safe_float(row.get("epsilon", "nan")))
    return stats.describe()


def analyze_eps_fourier(path: Path) -> Optional[Dict[str, float]]:
    stats = Summary()
    for row in load_csv(path):
        real = safe_float(row.get("real", "nan"))
        imag = safe_float(row.get("imag", "nan"))
        stats.add(math.hypot(real, imag))
    return stats.describe()


# -----------------------------------------------------------------------------
# Aggregation logic

@dataclass
class PipelineReport:
    name: str
    pol: str
    snapshots: List[SnapshotReport] = field(default_factory=list)
    spectra: List[SpectrumReport] = field(default_factory=list)
    iterations: List[IterationReport] = field(default_factory=list)
    residual_snapshots: List[ResidualSnapshotReport] = field(default_factory=list)
    fft_raw: Optional[FFTWorkspaceRawAnalysis] = None
    fft_report: Optional[Dict[str, object]] = None
    eps_real: Optional[Dict[str, float]] = None
    eps_real_raw: Optional[Dict[str, float]] = None
    eps_fourier: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)

    def finalize(self) -> None:
        # Missing gradient data?
        missing_grad = [snap for snap in self.snapshots if not snap.grad_present]
        if missing_grad:
            ids = ", ".join(snap.identifier for snap in missing_grad)
            self.warnings.append(f"∇u columns absent for: {ids}")
        # Slow convergence heuristics
        slow_iters = [trace for trace in self.iterations if trace.iterations > 80]
        for trace in slow_iters:
            self.warnings.append(
                f"Slow convergence ({trace.iterations} iters) -> {Path(trace.path).name}"
            )
        # Residual floor warnings
        noisy = [trace for trace in self.iterations if trace.final_max_residual > 1e-3]
        for trace in noisy:
            self.warnings.append(
                f"Residual stuck at {trace.final_max_residual:.2e} in {Path(trace.path).name}"
            )
        low_scale = [
            trace
            for trace in self.iterations
            if math.isfinite(trace.final_relative_scale)
            and trace.final_relative_scale < 1e-2
        ]
        for trace in low_scale:
            self.warnings.append(
                f"Relative scale {trace.final_relative_scale:.2e} (check tol) in {Path(trace.path).name}"
            )
        ziggy = [trace for trace in self.iterations if trace.zig_zag_ratio > 0.3]
        for trace in ziggy:
            self.warnings.append(
                f"Residual zig-zag ratio {trace.zig_zag_ratio:.2f} ({Path(trace.path).name})"
            )
        noisy_projection = [
            trace
            for trace in self.iterations
            if trace.projection_condition
            and trace.projection_condition.get("max", math.inf) > 1e10
        ]
        for trace in noisy_projection:
            cond = trace.projection_condition.get("max", math.nan)
            self.warnings.append(
                f"Ill-conditioned Rayleigh–Ritz mass matrix (cond≈{cond:.2e}) in {Path(trace.path).name}"
            )
        fallback_traces = [trace for trace in self.iterations if trace.projection_fallbacks > 0]
        for trace in fallback_traces:
            self.warnings.append(
                f"Rayleigh–Ritz fallback used {trace.projection_fallbacks}× in {Path(trace.path).name}"
            )
        if self.fft_raw:
            raw_min = (self.fft_raw.raw or {}).get("min") if self.fft_raw.raw else None
            if raw_min is not None and raw_min <= 0:
                self.warnings.append("|k+G|^2 touches zero (pre-clamp) in fft_workspace_raw")
            if self.fft_raw.clamp_fraction > 0:
                percent = self.fft_raw.clamp_fraction * 100
                self.warnings.append(
                    f"{self.fft_raw.clamped_bins} Fourier bins clamped ({percent:.2f}% of grid)"
                )
        if self.residual_snapshots:
            stages_present = {snap.stage for snap in self.residual_snapshots}
            expected_stages = {"raw", "projected", "preconditioned"}
            missing = sorted(expected_stages.difference(stages_present))
            if missing:
                self.warnings.append(
                    "Residual snapshots missing stages: " + ", ".join(missing)
                )


# -----------------------------------------------------------------------------
# Reporting helpers


def format_stats(title: str, stats: Optional[Dict[str, float]]) -> str:
    if not stats:
        return f"  {title}: (missing)"
    return (
        f"  {title}: min={stats['min']:.3e}, max={stats['max']:.3e}, "
        f"mean={stats['mean']:.3e}, p99={stats['p99']:.3e}"
    )


def render_report(report: PipelineReport) -> str:
    lines: List[str] = []
    lines.append(f"=== {report.name} ({report.pol.upper()}) ===")
    if report.warnings:
        lines.append("Warnings:")
        for warn in report.warnings:
            lines.append(f"  ! {warn}")
    if report.eps_real:
        lines.append(format_stats("ε(x,y) smoothed", report.eps_real))
    if report.eps_real_raw:
        lines.append(format_stats("ε(x,y) raw", report.eps_real_raw))
    if report.eps_fourier:
        lines.append(format_stats("|ε(G)|", report.eps_fourier))
    if report.fft_raw:
        if report.fft_raw.clamped:
            lines.append(format_stats("|k+G|^2 (clamped)", report.fft_raw.clamped))
        if report.fft_raw.raw:
            lines.append(format_stats("|k+G|^2 (raw)", report.fft_raw.raw))
        if report.fft_raw.total_bins:
            frac = report.fft_raw.clamp_fraction * 100
            lines.append(
                f"  Clamped bins: {report.fft_raw.clamped_bins}/{report.fft_raw.total_bins} ({frac:.2f}%)"
            )
    if report.fft_report:
        bloch = report.fft_report.get("bloch_norm")
        mesh = report.fft_report.get("mesh_size")
        if bloch is not None:
            lines.append(f"  Bloch |k| = {bloch:.4f}  (mesh={mesh})")
    if report.snapshots:
        lines.append(f"  Operator snapshots: {len(report.snapshots)} files")
        for snap in report.snapshots:
            ident = snap.identifier
            field = format_stats(f"    {ident} |u|", snap.field)
            theta = format_stats("    |Θu|", snap.theta)
            grad_state = "present" if snap.grad_present else "missing"
            lines.append(field)
            lines.append(theta)
            lines.append(f"    ∇u data: {grad_state}")
    if report.spectra:
        lines.append(f"  Operator spectra: {len(report.spectra)} files")
        for spec in report.spectra:
            ident = spec.identifier
            lines.append(format_stats(f"    {ident} |û|", spec.field_hat))
            lines.append(format_stats("    |Θû|", spec.theta_hat))
    if report.residual_snapshots:
        lines.append(f"  Residual snapshots: {len(report.residual_snapshots)} files")
        stage_groups: Dict[str, List[ResidualSnapshotReport]] = {}
        for snapshot in report.residual_snapshots:
            stage_groups.setdefault(snapshot.stage, []).append(snapshot)
        for stage in sorted(stage_groups):
            group = stage_groups[stage]
            iter_values = [snap.iteration for snap in group if snap.iteration is not None]
            if iter_values:
                it_min = min(iter_values)
                it_max = max(iter_values)
                iter_desc = (
                    f"iter {it_min}"
                    if it_min == it_max
                    else f"iters {it_min}-{it_max}"
                )
            else:
                iter_desc = "iters ?"
            band_values = [snap.band_index for snap in group if snap.band_index is not None]
            if band_values:
                b_min = min(band_values) + 1
                b_max = max(band_values) + 1
                band_desc = (
                    f"band {b_min}"
                    if b_min == b_max
                    else f"bands {b_min}-{b_max}"
                )
            else:
                band_desc = "bands ?"
            peaks: List[float] = []
            for snap in group:
                if snap.stats and "max" in snap.stats:
                    peak = snap.stats["max"]
                    if math.isfinite(peak):
                        peaks.append(peak)
            peak_desc = f"max |r|={max(peaks):.2e}" if peaks else "max |r|=n/a"
            stage_label = stage.replace("_", " ")
            lines.append(
                f"    {stage_label}: {len(group)} files ({iter_desc}, {band_desc}), {peak_desc}"
            )
    if report.iterations:
        lines.append(f"  Iteration traces: {len(report.iterations)} files")
        for trace in report.iterations:
            final_res = trace.final_max_residual
            zig = trace.zig_zag_ratio
            iter_count = trace.iterations
            thresh_hits = trace.threshold_hits
            hit_parts = []
            for thr in THRESHOLDS:
                hit = thresh_hits.get(thr)
                hit_parts.append(f"≤{thr:.0e}:{hit if hit is not None else '—'}")
            hit_str = ", ".join(hit_parts)
            rel_bits: List[str] = []
            if math.isfinite(trace.final_max_relative):
                rel_bits.append(f"rel={trace.final_max_relative:.2e}")
            if math.isfinite(trace.final_relative_scale):
                rel_bits.append(f"scale={trace.final_relative_scale:.2e}")
            rel_suffix = f", {' '.join(rel_bits)}" if rel_bits else ""
            lines.append(
                f"    {Path(trace.path).name}: iters={iter_count}, final residual={final_res:.2e}, "
                f"zig-zag={zig:.2f}{rel_suffix}"
            )
            lines.append(f"      thresholds: {hit_str}")
            if trace.projection_condition:
                cond = trace.projection_condition
                lines.append(
                    f"      projection cond: min={cond['min']:.2e}, max={cond['max']:.2e}"
                )
            if trace.projection_dims:
                orig = trace.projection_dims.get("original", {})
                reduced = trace.projection_dims.get("reduced", {})
                if orig and reduced:
                    lines.append(
                        f"      projection dims: orig~{orig.get('median', orig.get('mean', math.nan)):.1f} "
                        f"→ reduced~{reduced.get('median', reduced.get('mean', math.nan)):.1f}"
                    )
            if trace.projection_fallbacks:
                lines.append(f"      projection fallbacks: {trace.projection_fallbacks}")
    lines.append("")
    return "\n".join(lines)


def format_file_timestamp(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat()


def load_previous_metadata(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# -----------------------------------------------------------------------------
# Discovery logic

EXPECTED_FILES = {
    "epsilon_real": "epsilon_real.csv",
    "epsilon_real_raw": "epsilon_real_raw.csv",
    "epsilon_fourier": "epsilon_fourier.csv",
    "fft_workspace_raw": "fft_workspace_raw_k000.csv",
    "fft_workspace_report": "fft_workspace_report_k000.json",
}


def discover_pipeline_dirs(reference_dir: Path) -> List[Path]:
    return sorted(reference_dir.glob("*_pipeline"))


def infer_pol(name: str) -> str:
    if "_tm_" in name:
        return "tm"
    if "_te_" in name:
        return "te"
    return "unknown"


def collect_report(directory: Path) -> PipelineReport:
    report = PipelineReport(name=directory.stem.replace("_pipeline", ""), pol=infer_pol(directory.name))
    # Dielectric/FFT files
    eps_real = directory / EXPECTED_FILES["epsilon_real"]
    if eps_real.exists():
        report.eps_real = analyze_eps_real(eps_real)
    eps_real_raw = directory / EXPECTED_FILES["epsilon_real_raw"]
    if eps_real_raw.exists():
        report.eps_real_raw = analyze_eps_real(eps_real_raw)
    eps_fourier = directory / EXPECTED_FILES["epsilon_fourier"]
    if eps_fourier.exists():
        report.eps_fourier = analyze_eps_fourier(eps_fourier)
    fft_raw = directory / EXPECTED_FILES["fft_workspace_raw"]
    if fft_raw.exists():
        report.fft_raw = analyze_fft_workspace_raw(fft_raw)
    fft_report = directory / EXPECTED_FILES["fft_workspace_report"]
    if fft_report.exists():
        report.fft_report = analyze_fft_workspace_report(fft_report)
    # Operator snapshots
    for snapshot in sorted(directory.glob("operator_snapshot_*_mode??.csv")):
        if snapshot.name.endswith("_spectrum.csv"):
            continue
        report.snapshots.append(analyze_operator_snapshot(snapshot))
    for spectrum in sorted(directory.glob("operator_snapshot_*_mode??_spectrum.csv")):
        report.spectra.append(analyze_operator_spectrum(spectrum))
    for residual in sorted(directory.glob("residual_snapshot_*.csv")):
        report.residual_snapshots.append(analyze_residual_snapshot(residual))
    # Iteration traces
    for iteration_csv in sorted(directory.glob("operator_iteration_trace_*.csv")):
        report.iterations.append(analyze_iteration_trace(iteration_csv))
    report.finalize()
    return report


# -----------------------------------------------------------------------------
# Entry point


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mpb2d pipeline dumps.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reference-data",
        help="Directory containing *_pipeline folders (default: reference-data next to this script)",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Path to write a single consolidated report; omitting this enables per-config reports",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Metadata path for single-report mode (default: reference-data/pipeline_report_meta.json)",
    )
    parser.add_argument(
        "--single-report",
        action="store_true",
        help="Force single consolidated output even when --report-file is omitted",
    )
    args = parser.parse_args()
    reference_dir = args.reference_dir.resolve()
    if not reference_dir.exists():
        raise SystemExit(f"Reference directory {reference_dir} does not exist")
    pipeline_dirs = [path.resolve() for path in discover_pipeline_dirs(reference_dir)]
    if not pipeline_dirs:
        raise SystemExit("No *_pipeline directories found")
    reports_by_dir: Dict[Path, PipelineReport] = {}
    grouped_dirs: Dict[str, List[Path]] = {}
    for directory in pipeline_dirs:
        reports_by_dir[directory] = collect_report(directory)
        group = infer_config_group(directory)
        grouped_dirs.setdefault(group, []).append(directory)

    single_report_mode = args.single_report or args.report_file is not None
    if single_report_mode:
        report_path = (args.report_file or (reference_dir / "pipeline_report.txt")).resolve()
        metadata_path = (args.metadata_file or (reference_dir / "pipeline_report_meta.json")).resolve()
        previous_meta = load_previous_metadata(metadata_path)
        report_sections = [render_report(reports_by_dir[directory]) for directory in pipeline_dirs]
        run_timestamp = datetime.now(timezone.utc)
        header_lines = [
            "# mpb2d pipeline report",
            f"Generated at: {run_timestamp.isoformat()}",
            "",
            "Input file timestamps:",
        ]
        if INPUT_FILE_TIMESTAMPS:
            for path_str in sorted(INPUT_FILE_TIMESTAMPS):
                header_lines.append(
                    f"  {path_str}: {format_file_timestamp(INPUT_FILE_TIMESTAMPS[path_str])}"
                )
        else:
            header_lines.append("  (no files were read)")
        header_lines.append("")
        full_report = "\n".join(header_lines + report_sections)
        print(full_report, end="")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(full_report)
        metadata_payload = {
            "generated_at": run_timestamp.isoformat(),
            "file_timestamps": dict(INPUT_FILE_TIMESTAMPS),
            "report_file": str(report_path),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))
        if previous_meta and previous_meta.get("file_timestamps") == metadata_payload["file_timestamps"]:
            prev_time = previous_meta.get("generated_at", "unknown")
            print(
                f"[report] WARNING: Input file timestamps unchanged since {prev_time}",
                file=sys.stderr,
            )
        else:
            print(f"[report] Saved to {report_path}", file=sys.stderr)
        return

    # Per-config report mode
    for group_name in sorted(grouped_dirs):
        group_dirs = sorted(grouped_dirs[group_name])
        sanitized = sanitize_group_label(group_name)
        report_path = (reference_dir / f"pipeline_report_{sanitized}.txt").resolve()
        metadata_path = (reference_dir / f"pipeline_report_{sanitized}_meta.json").resolve()
        previous_meta = load_previous_metadata(metadata_path)
        run_timestamp = datetime.now(timezone.utc)
        sections = [render_report(reports_by_dir[directory]) for directory in group_dirs]
        timestamp_subset = timestamps_for_directories(group_dirs) or dict(INPUT_FILE_TIMESTAMPS)
        header_lines = [
            f"# mpb2d pipeline report ({group_name})",
            f"Generated at: {run_timestamp.isoformat()}",
            "",
            "Input file timestamps:",
        ]
        if timestamp_subset:
            for path_str in sorted(timestamp_subset):
                header_lines.append(
                    f"  {path_str}: {format_file_timestamp(timestamp_subset[path_str])}"
                )
        else:
            header_lines.append("  (no files were read)")
        header_lines.append("")
        full_report = "\n".join(header_lines + sections)
        print(full_report, end="")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(full_report)
        metadata_payload = {
            "generated_at": run_timestamp.isoformat(),
            "file_timestamps": timestamp_subset,
            "report_file": str(report_path),
            "config_group": group_name,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))
        if previous_meta and previous_meta.get("file_timestamps") == metadata_payload["file_timestamps"]:
            prev_time = previous_meta.get("generated_at", "unknown")
            print(
                f"[report:{group_name}] WARNING: Input file timestamps unchanged since {prev_time}",
                file=sys.stderr,
            )
        else:
            print(f"[report:{group_name}] Saved to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
