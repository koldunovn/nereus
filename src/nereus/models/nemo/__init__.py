"""NEMO ocean model support for nereus.

This module provides functionality for working with NEMO ocean model meshes,
loading from mesh_mask.nc files.

Examples
--------
>>> import nereus as nr

# Load NEMO mesh
>>> mesh = nr.nemo.load_mesh("/path/to/mesh_mask.nc")

# Access flattened coordinates
>>> lon = mesh["lon"].values  # 1D array
>>> lat = mesh["lat"].values
>>> area = mesh["area"]

# The mesh includes original 2D shape for reshaping
>>> shape_2d = (mesh.attrs["nlat"], mesh.attrs["nlon"])
"""

from nereus.models.nemo.mesh import (
    flatten_structured,
    load_mesh,
)

__all__ = [
    "flatten_structured",
    "load_mesh",
]
