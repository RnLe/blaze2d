"""Versioned JSON, NDJSON, and NPZ without pickle or precision loss."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable, Iterator
import numpy as np


def _encode(value, buffers=None):
    if isinstance(value, dict):
        if "array_info" not in value:
            return {k: _encode(v, buffers) for k, v in value.items()}
        out = {k: _encode(v, buffers) for k, v in value.items()
               if k != "array_info" and k not in value["array_info"]}
        out["arrays"] = {}
        for name, info in value["array_info"].items():
            array = np.asarray(value[name])
            if array.dtype not in (np.dtype("float64"), np.dtype("complex128")):
                raise ValueError(f"{name}: expected float64 or complex128")
            if list(array.shape) != info["shape"] or array.dtype.name != info["dtype"]:
                raise ValueError(f"{name}: array does not match its dimensions or dtype")
            if not np.isfinite(array).all():
                raise ValueError(f"{name}: non-finite values cannot be exported as lossless JSON")
            array = np.ascontiguousarray(array)
            descriptor = dict(info)
            if buffers is None:
                descriptor["data"] = array.reshape(-1).view(np.float64).tolist()
            else:
                key = f"array_{len(buffers):06d}"
                buffers[key] = array
                descriptor["buffer"] = key
            out["arrays"][name] = descriptor
        return out
    if isinstance(value, (list, tuple)):
        return [_encode(v, buffers) for v in value]
    return value


def _decode(value, buffers=None):
    if isinstance(value, list):
        return [_decode(v, buffers) for v in value]
    if not isinstance(value, dict):
        return value
    schema = value.get("schema")
    if schema is not None and schema not in ("blaze2d/result/1", "blaze2d/run/1", "blaze2d/error/1", "blaze2d/1"):
        raise ValueError(f"Unsupported result schema: {schema}")
    out = {k: _decode(v, buffers) for k, v in value.items() if k != "arrays"}
    if "arrays" in value:
        out["array_info"] = {}
        for name, descriptor in value["arrays"].items():
            dtype = descriptor["dtype"]
            shape = descriptor["shape"]
            if dtype not in ("float64", "complex128") or descriptor["order"] != "C":
                raise ValueError(f"{name}: unsupported array encoding")
            if len(shape) != len(descriptor["dimensions"]) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"{name}: invalid array dimensions")
            if buffers is not None:
                array = buffers[descriptor["buffer"]]
                if array.dtype.name != dtype or list(array.shape) != shape:
                    raise ValueError(f"{name}: NPZ buffer does not match its descriptor")
            else:
                data = np.asarray(descriptor["data"], dtype=np.float64)
                if data.ndim != 1:
                    raise ValueError(f"{name}: expected flat array data")
                array = data.view(np.complex128) if dtype == "complex128" else data
                array = array.reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError(f"{name}: non-finite array data")
            out[name] = np.ascontiguousarray(array)
            out["array_info"][name] = {k: descriptor[k] for k in ("dtype", "shape", "dimensions", "order")}
    return out


def save(result: dict, path: str | Path) -> Path:
    """Save a result or collected study. The extension selects JSON, NDJSON, or NPZ."""
    path = Path(path)
    if path.suffix == ".npz":
        buffers = {}
        manifest = json.dumps(_encode(result, buffers), allow_nan=False, separators=(",", ":"))
        with path.open("wb") as file:
            np.savez_compressed(file, manifest=np.frombuffer(manifest.encode("utf-8"), dtype=np.uint8), **buffers)
    elif path.suffix in (".json", ".ndjson", ".jsonl"):
        path.write_text(json.dumps(_encode(result), allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    else:
        raise ValueError("Use a .json, .ndjson, or .npz filename")
    return path


def load(path: str | Path) -> dict:
    """Load a result or collected study, validating array shapes and schema."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return _decode(json.loads(archive["manifest"].tobytes().decode("utf-8")), archive)
    return _decode(json.loads(path.read_text(encoding="utf-8")))


def write_ndjson(records: Iterable[dict], path: str | Path) -> Path:
    """Write an incremental stream without collecting its arrays in memory."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(_encode(record), allow_nan=False, separators=(",", ":")) + "\n")
            file.flush()
    return path


def read_ndjson(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield _decode(json.loads(line))
