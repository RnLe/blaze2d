#!/usr/bin/env python3
"""Summarize pipeline inspection dumps and surface potential anomalies.

This script scans reference-data pipeline directories produced by `mpb2d-cli --dump-pipeline`
and prints a compact report with statistics for:
  * dielectric real/Fourier snapshots
  * operator snapshot CSVs (real space & spectra)
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
    zig_zag_ratio: float
    threshold_hits: Dict[float, Optional[int]]
    preconditioner_trials: int


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
    preconditioner_trials = 0
    for row in load_csv(path):
        iterations.append(int(row["iteration"]))
        max_residuals.append(abs(safe_float(row.get("max_residual", "nan"))))
        avg_residuals.append(abs(safe_float(row.get("avg_residual", "nan"))))
        preconditioner_trials += int(row.get("preconditioner_trials", 0))
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
        zig_zag_ratio=zig_zag / max(1, len(max_residuals) - 1),
        threshold_hits=thresholds,
        preconditioner_trials=preconditioner_trials,
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
        ziggy = [trace for trace in self.iterations if trace.zig_zag_ratio > 0.3]
        for trace in ziggy:
            self.warnings.append(
                f"Residual zig-zag ratio {trace.zig_zag_ratio:.2f} ({Path(trace.path).name})"
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
            lines.append(
                f"    {Path(trace.path).name}: iters={iter_count}, final residual={final_res:.2e}, "
                f"zig-zag={zig:.2f}"
            )
            lines.append(f"      thresholds: {hit_str}")
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
        help="Path to write the consolidated report (default: reference-data/pipeline_report.txt)",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Path to write report metadata (default: reference-data/pipeline_report_meta.json)",
    )
    args = parser.parse_args()
    reference_dir = args.reference_dir.resolve()
    if not reference_dir.exists():
        raise SystemExit(f"Reference directory {reference_dir} does not exist")
    report_path = (args.report_file or (reference_dir / "pipeline_report.txt")).resolve()
    metadata_path = (args.metadata_file or (reference_dir / "pipeline_report_meta.json")).resolve()
    previous_meta = load_previous_metadata(metadata_path)

    pipeline_dirs = discover_pipeline_dirs(reference_dir)
    if not pipeline_dirs:
        raise SystemExit("No *_pipeline directories found")
    report_sections: List[str] = []
    for directory in pipeline_dirs:
        report = collect_report(directory)
        report_sections.append(render_report(report))

    run_timestamp = datetime.now(timezone.utc)
    header_lines = [
        "# mpb2d pipeline report",
        f"Generated at: {run_timestamp.isoformat()}",
        "",
        "Input file timestamps:",
    ]
    if INPUT_FILE_TIMESTAMPS:
        for path_str in sorted(INPUT_FILE_TIMESTAMPS):
            header_lines.append(f"  {path_str}: {format_file_timestamp(INPUT_FILE_TIMESTAMPS[path_str])}")
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


if __name__ == "__main__":
    main()
