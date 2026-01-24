"""IFS TCO model support for nereus.

This module provides functionality for loading IFS TCO meshes as
xr.Dataset objects with standardized variable names.

Examples
--------
>>> import nereus as nr
>>> mesh = nr.ifs_tco.load_mesh("/path/to/grid.nc", "/path/to/areas.nc")
>>> mesh["lon"]
"""

from nereus.models.ifs_tco.mesh import load_mesh

__all__ = [
    "load_mesh",
]
