"""
BLAZE — Band-structure LOBPCG Accelerated Zone Eigensolver for 2D Photonic Crystals.

Quick start::

    import blaze

    # Single band diagram
    result = blaze.solve(lattice_type="square", epsilon_background=12.0,
                         epsilon_atoms=1.0, radius_atom=0.2, polarization="TM")

    # Parameter sweep
    results = blaze.solve(lattice_type="hexagonal", epsilon_background=13.0,
                          epsilon_atoms=1.0, radius_atom=[0.2, 0.4, 0.05],
                          polarization=["TM", "TE"])

    # Reproducible TOML config without writing a file
    driver = blaze.BulkDriver.from_toml(config_toml)
"""

# Re-export native Rust bindings
from blaze._native import BulkDriver, BandResultIterator, OperatorDataExtractor  # noqa: F401
from blaze._native import __version__  # noqa: F401

# High-level API
from blaze.solve import solve  # noqa: F401

__all__ = [
    "solve",
    "BulkDriver",
    "BandResultIterator",
    "OperatorDataExtractor",
    "__version__",
]
