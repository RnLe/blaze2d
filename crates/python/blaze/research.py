"""Research entry points using the same configuration and result dictionaries."""
import numpy as np
from . import _native
from .api import solve, run
from .config import as_config


def _operators(config):
    config = as_config(config)
    if config.summary["task"] != "operators":
        raise ValueError("Use task = 'operators' for operator extraction")
    return config


class OperatorDataExtractor:
    """Projected operators, registry studies, external fields, and checkpoints.

    Every method accepts a Config, dictionary, or TOML file path. Reference
    blocks contain the complete solved band window, including lower remote bands.
    """

    @staticmethod
    def extract(config):
        return solve(_operators(config))

    @staticmethod
    def extract_registry_sweep(config, **runner_options):
        return run(_operators(config), **runner_options)

    @staticmethod
    def extract_k_stencil(config):
        config = _operators(config)
        if "k_stencil" not in config.to_dict()["operators"]:
            raise ValueError("Declare operators.k_stencil before extracting a stencil")
        return solve(config)

    @staticmethod
    def extract_with_reference(config, *, reference_eigenvectors=None, warmstart_eigenvectors=None):
        config = _operators(config)
        reference = None if reference_eigenvectors is None else np.ascontiguousarray(reference_eigenvectors,dtype=np.complex128)
        warm = None if warmstart_eigenvectors is None else np.ascontiguousarray(warmstart_eigenvectors,dtype=np.complex128)
        return _native._extract_with_reference(config._handle,reference,warm)

    @staticmethod
    def solve_external_map(config, epsilon, *, inverse_epsilon_tensors=None):
        """Solve one sampled band point with dielectric.source = 'external'."""
        config = as_config(config)
        tensors = None if inverse_epsilon_tensors is None else np.ascontiguousarray(inverse_epsilon_tensors,dtype=np.float64)
        return _native._solve_external_map(config._handle,np.ascontiguousarray(epsilon,dtype=np.float64),tensors)

    @staticmethod
    def solve_k_path(config):
        config = as_config(config)
        if config.summary["task"] != "bands":
            raise ValueError("Use task = 'bands' for a band path")
        return solve(config)

    @staticmethod
    def extract_registry_sweep_checkpointed(config, checkpoint, **options):
        from .checkpoint import run_checkpointed
        return run_checkpointed(_operators(config),checkpoint,**options)

    @staticmethod
    def load_checkpoint_row(path):
        from .checkpoint import load_checkpoint
        return load_checkpoint(path)["results"]
