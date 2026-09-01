"""Blaze2D: photonic bands and projected operators from one calculation contract."""
from ._native import (CalculationError, ConfigurationError,
                      build_info, capabilities, __version__)
from .config import Config
from .api import solve, run, stream
from .io import save, load, write_ndjson, read_ndjson
from .plot import plot
from .research import OperatorDataExtractor

__all__ = ["Config", "solve", "run", "stream", "save", "load", "write_ndjson",
           "read_ndjson", "plot", "build_info", "capabilities", "OperatorDataExtractor",
           "CalculationError", "ConfigurationError", "__version__"]
