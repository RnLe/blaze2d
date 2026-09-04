"""Append-only checkpoints with schema, configuration, and build validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter, time_ns
from . import _native
from .config import as_config
from .io import _encode, _decode

SCHEMA = "blaze2d/checkpoint/1"


def load_checkpoint(path):
    """Load complete records; leave an interrupted final record untouched."""
    path = Path(path)
    results, errors = [], []
    valid_bytes = 0
    with path.open("rb") as file:
        first = file.readline()
        header = json.loads(first)
        if header.get("schema") != SCHEMA:
            raise ValueError("Unsupported checkpoint schema; the original file is unchanged")
        valid_bytes += len(first)
        seen = set()
        for raw in file:
            try:
                record = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                if not raw.endswith(b"\n") and not file.read(1):
                    break
                raise ValueError("Checkpoint contains a malformed record") from None
            if record.get("schema") == "blaze2d/result/1":
                record = _decode(record)
                if record["job_index"] in seen:
                    raise ValueError("Checkpoint contains duplicate completed jobs")
                seen.add(record["job_index"])
                results.append(record)
            elif record.get("schema") == "blaze2d/error/1":
                errors.append(_decode(record)["error"])
            else:
                raise ValueError("Unsupported checkpoint record schema")
            valid_bytes += len(raw)
    return {**header, "results":results, "errors":errors, "valid_bytes":valid_bytes}


def _append(file, record):
    payload = json.dumps(_encode(record),allow_nan=False,separators=(",",":")).encode("utf-8") + b"\n"
    file.write(payload)
    file.flush()
    os.fsync(file.fileno())


def run_checkpointed(config, checkpoint, *, resume=True, threads=0, error_policy="stop", progress=None):
    """Run pending jobs and flush each completed record before reporting it.

    Resume requires an identical normalized calculation and build revision.
    An interrupted final write is archived before the valid prefix is restored.
    The returned study includes results completed by earlier attempts.
    """
    config = as_config(config)
    path = Path(checkpoint)
    start = perf_counter()
    results, errors = [], []
    build = _native.build_info()
    if resume and path.exists():
        loaded = load_checkpoint(path)
        if loaded["config"] != config.to_dict():
            raise ValueError("Checkpoint configuration differs from this calculation")
        if loaded["build"] != build:
            raise ValueError("Checkpoint build differs from the current solver; use the recorded build or a new checkpoint")
        results = loaded["results"]
        if loaded["valid_bytes"] < path.stat().st_size:
            archive = path.with_name(f"{path.name}.interrupted-{time_ns()}")
            path.rename(archive)
            with archive.open("rb") as source, path.open("xb") as destination:
                remaining = loaded["valid_bytes"]
                while remaining:
                    chunk = source.read(min(remaining,1024*1024))
                    if not chunk:
                        raise OSError("Interrupted checkpoint changed during recovery")
                    destination.write(chunk); remaining -= len(chunk)
                destination.flush(); os.fsync(destination.fileno())
    else:
        with path.open("xb") as file:
            _append(file,{"schema":SCHEMA,"config":config.to_dict(),"build":build})
    completed = [r["job_index"] for r in results]
    with path.open("rb+") as file:
        file.seek(-1,os.SEEK_END)
        if file.read(1) != b"\n":
            file.write(b"\n"); file.flush(); os.fsync(file.fileno())
    if any(type(i) is not int or i < 0 or i >= config.summary["jobs"] for i in completed):
        raise ValueError("Checkpoint contains an out-of-range job index")
    resumed = len(completed)
    statistics = {"status":"completed","completed":resumed,"failed":0}
    events = config._handle.events(json.dumps({"threads":threads,"error_policy":error_policy,"completed_jobs":completed}))
    try:
        with path.open("ab") as file:
            for event in events:
                if event["event"] == "result":
                    results.append(event["result"])
                    _append(file,event["result"])
                elif event["event"] == "job_failure":
                    errors.append(event["error"])
                    _append(file,{"schema":"blaze2d/error/1","error":event["error"]})
                elif event["event"] == "terminal":
                    statistics = {k:v for k,v in event.items() if k != "event"}
                    statistics["completed"] += resumed
                if progress is not None:
                    progress(event)
    except OSError as error:
        error.completed_results = results
        raise
    finally:
        events.cancel()
    statistics["resumed"] = resumed
    statistics["elapsed_seconds"] = perf_counter()-start
    statistics["runtime"] = dict(events.options, progress=progress is not None, checkpoint=str(path))
    return {"schema":"blaze2d/run/1","config":config.to_dict(),
            "results":sorted(results,key=lambda r:r["job_index"]),"errors":errors,"statistics":statistics}
