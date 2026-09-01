"""Single calculations and bounded native studies."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from time import perf_counter
from . import _native
from .config import Config, as_config
from .types import Result, StudyResult


def _events(config, *, threads=0, error_policy="stop", queue_capacity=0):
    config = as_config(config)
    return config._handle.events(json.dumps({"threads": threads, "error_policy": error_policy,
                                           "queue_capacity": queue_capacity}, allow_nan=False))


def _raise_failure(error):
    failure = _native.CalculationError(error["diagnostic"]["message"])
    failure.diagnostic = error["diagnostic"]
    failure.partial_result = error.get("partial_result")
    failure.job_index = error["job_index"]
    raise failure


def solve(config=None, *, lattice_type=None, epsilon_background=None, epsilon_atoms=None,
          radius_atom=None, polarization=None, resolution=None, n_bands=None, k_path=None,
          tolerance=None, max_iterations=None, precision=None, eigenvectors=None) -> Result:
    """Solve exactly one job and return named NumPy arrays.

    Scalar options construct the same Rust configuration as TOML. A numeric list
    is never interpreted as a sweep. Use Config and run or stream for studies.
    Frequencies have shape (k_point, band); fields use (k_point, band, y, x).
    """
    options = {k: v for k, v in locals().items() if k != "config" and v is not None}
    if config is None:
        config = Config(_native.simple_config(json.dumps(options, allow_nan=False)))
    elif options:
        raise TypeError("Use either a configuration or scalar calculation options")
    config = as_config(config)
    if config.summary["jobs"] != 1:
        raise ValueError("solve requires exactly one job; use run or stream for a study")
    return config._handle.solve()


def run(config, *, threads=0, error_policy="stop", queue_capacity=0,
        progress: Callable[[dict], None] | None = None) -> StudyResult:
    """Collect a study, including errors and completed results on failure.

    The default stops scheduling new jobs after a failure. Already running jobs
    finish and their results are retained. No output is printed by the library.
    """
    config = as_config(config)
    start = perf_counter()
    results, errors = [], []
    statistics = {"status": "failed", "completed": 0, "failed": 0}
    events = _events(config, threads=threads, error_policy=error_policy, queue_capacity=queue_capacity)
    try:
        for event in events:
            kind = event["event"]
            if kind == "result":
                results.append(event["result"])
            elif kind == "job_failure":
                errors.append(event["error"])
            elif kind == "terminal":
                statistics = {k: v for k, v in event.items() if k != "event"}
            if progress is not None:
                progress(event)
    finally:
        events.cancel()
    statistics["elapsed_seconds"] = perf_counter() - start
    return {"schema": "blaze2d/run/1", "config": config.to_dict(),
            "results": sorted(results, key=lambda r: r["job_index"]),
            "errors": errors, "statistics": statistics}


def stream(config, *, threads=0, error_policy="stop", queue_capacity=0,
           progress: Callable[[dict], None] | None = None) -> Iterator[Result]:
    """Yield results as jobs finish, without collecting the study in memory.

    A failure raises CalculationError by default. With error_policy='continue',
    it yields an error record with job_index and error keys and continues.
    Closing the generator stops scheduling further jobs.
    """
    events = _events(config, threads=threads, error_policy=error_policy, queue_capacity=queue_capacity)
    try:
        for event in events:
            if progress is not None:
                progress(event)
            if event["event"] == "result":
                yield event["result"]
            elif event["event"] == "job_failure":
                if error_policy == "stop":
                    _raise_failure(event["error"])
                yield {"schema": "blaze2d/error/1", "job_index": event["error"]["job_index"],
                       "error": event["error"]}
    finally:
        events.cancel()
