"""Client for the matrix-free operator server in crates/interface/examples/precond_lab.rs.

The server applies the band job's operator A, mass B and preconditioner T to
blocks of complex128 vectors in the solver's spatial ordering (row-major ny x nx).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


class RustOperators:
    def __init__(self, binary: str | Path, case_toml: str | Path):
        self.proc = subprocess.Popen([str(binary), "serve", str(case_toml)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.info, _ = self.request({"op": "info"})
        self.nx, self.ny = self.info["nx"], self.info["ny"]
        self.n = self.nx * self.ny
        self.point: dict = {}

    def request(self, header: dict, payload: np.ndarray | None = None) -> tuple[dict, np.ndarray | None]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((json.dumps(header) + "\n").encode())
        if payload is not None:
            self.proc.stdin.write(np.ascontiguousarray(payload, dtype="<c16").tobytes())
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        if not reply.get("ok", False):
            raise RuntimeError(reply.get("error", "precond_lab request failed"))
        count = reply.get("count", 0) if header["op"] in ("apply", "solve") else 0
        data = None
        if count:
            raw = self.proc.stdout.read(count * self.n * 16)
            data = np.frombuffer(raw, dtype="<c16").reshape(count, self.n).copy()
        return reply, data

    def set_k(self, index: int | None = None, k: tuple[float, float] | None = None) -> dict:
        header = {"op": "set_k", "index": index} if index is not None else {"op": "set_k", "k": list(k)}
        self.point, _ = self.request(header)
        return self.point

    def set_spec(self, spec: dict, previous: list[float] | None = None) -> dict:
        self.point, _ = self.request({"op": "set_spec", "spec": spec, "previous": previous})
        return self.point

    def apply(self, which: str, vectors: np.ndarray, chunk: int = 256) -> np.ndarray:
        vectors = np.atleast_2d(vectors)
        out = np.empty_like(vectors, dtype=np.complex128)
        for start in range(0, len(vectors), chunk):
            block = vectors[start:start + chunk]
            _, data = self.request({"op": "apply", "which": which, "count": len(block)}, block)
            out[start:start + chunk] = data
        return out

    def dense(self, which: str) -> np.ndarray:
        """Dense matrix M with M[:, j] = which(e_j)."""
        return self.apply(which, np.eye(self.n, dtype=np.complex128)).T

    def solve(self, trace: bool = True, vectors: bool = False) -> tuple[dict, np.ndarray | None]:
        return self.request({"op": "solve", "trace": trace, "vectors": vectors})

    def close(self) -> None:
        if self.proc.poll() is None:
            assert self.proc.stdin is not None
            self.proc.stdin.write(b'{"op": "quit"}\n')
            self.proc.stdin.close()
            self.proc.wait(timeout=10)

    def __enter__(self) -> "RustOperators":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
