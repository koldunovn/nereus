"""MITgcm ocean model support for nereus.

This module provides functionality for working with MITgcm model output,
reading from the native MDS binary format (``.meta`` + ``.data`` file pairs).

Examples
--------
>>> import nereus as nr

# Load MITgcm mesh
>>> mesh = nr.mitgcm.load_mesh("/path/to/run/")

# Load diagnostic data
>>> ds = nr.mitgcm.open_dataset("/path/to/run/", prefix="diags2D",
...     delta_t=3600, ref_date="1710-1-1", mesh=mesh)
"""

from nereus.models.mitgcm.data import open_dataset
from nereus.models.mitgcm.mesh import load_mesh

__all__ = [
    "load_mesh",
    "open_dataset",
]
