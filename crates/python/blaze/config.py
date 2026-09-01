"""Validated configurations backed by the shared Rust contract."""
from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping
from . import _native


class Config:
    """An immutable, validated calculation. Numerical defaults resolve in Rust."""

    __slots__ = ("_handle", "_source")

    def __init__(self, handle=None, source: str | None = None):
        self._handle = handle or _native.simple_config("{}")
        self._source = source

    @classmethod
    def from_file(cls, path: str | PathLike) -> Config:
        return cls.from_toml(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_toml(cls, source: str) -> Config:
        return cls(_native.Configuration(source, "toml"), source)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Config:
        return cls(_native.Configuration(json.dumps(value, allow_nan=False), "json"))

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration with its effective defaults."""
        return self._handle.to_dict()

    def to_toml(self) -> str:
        """Serialize the normalized configuration. Source comments are not copied."""
        return self._handle.to_toml()

    @property
    def source(self) -> str | None:
        """Original TOML, preserved without rewriting."""
        return self._source

    @property
    def summary(self) -> dict[str, Any]:
        return self._handle.summary

    @property
    def resolved(self) -> dict[str, Any]:
        """Effective lattice, rectangular grid, sampled path, and plot distances."""
        return self._handle.resolved()

    def __repr__(self) -> str:
        summary = self.summary
        return f"Config(task={summary['task']!r}, jobs={summary['jobs']})"


def as_config(value) -> Config:
    if isinstance(value, Config):
        return value
    if isinstance(value, (str, PathLike)):
        return Config.from_file(value)
    if isinstance(value, Mapping):
        return Config.from_dict(value)
    raise TypeError("Expected Config, a configuration dictionary, or a TOML file path")
