"""Array dimensions for the public dictionary API.

All arrays use C ordering. Rust Complex64 maps to NumPy complex128 regardless
of solver storage precision. Each result also carries per-array dimensions in
array_info. Operator derivatives retain the basis recorded in metadata.
"""
from typing import Any, TypedDict
import numpy as np
from numpy.typing import NDArray


class ArrayInfo(TypedDict):
    dtype: str
    shape: list[int]
    dimensions: list[str]
    order: str


class Sample(TypedDict, total=False):
    sample_index: int
    metadata: dict[str, Any]
    array_info: dict[str, ArrayInfo]
    eigenvalues: NDArray[np.float64]  # (solved_band,)
    eigenvectors: NDArray[np.complex128]  # (solved_band, ny, nx)
    residuals: NDArray[np.float64]  # (solved_band,)
    velocity_matrices: NDArray[np.complex128]  # (2, retained_band, solved_band)
    w_matrices: NDArray[np.complex128]  # (2, 2, retained_band, retained_band)
    mass_tensor_inv: NDArray[np.complex128]  # (2, 2, retained_band, retained_band)
    r_derivative_matrices: NDArray[np.complex128]  # (2, retained_band, solved_band)
    metric_derivative_matrices: NDArray[np.complex128]  # (2, retained_band, solved_band)


class Result(Sample, total=False):
    schema: str
    task: str
    job_index: int
    frequencies: NDArray[np.float64]  # (k_point, band)
    k_points: NDArray[np.float64]  # (k_point, 2), reciprocal fractional
    k_points_cartesian: NDArray[np.float64]  # (k_point, 2), angular wavevector
    distances: NDArray[np.float64]  # (k_point,), reciprocal Cartesian metric
    samples: list[Sample]  # center first, followed by canonical stencil neighbors


class StudyResult(TypedDict):
    schema: str
    config: dict[str, Any]
    results: list[Result]
    errors: list[dict[str, Any]]
    statistics: dict[str, Any]
