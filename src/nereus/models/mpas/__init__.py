"""MPAS ocean model support for nereus.

This module provides functionality for working with MPAS-Ocean model output.
MPAS uses an unstructured Voronoi mesh stored in standard NetCDF files,
with mesh information embedded alongside the data.

Examples
--------
>>> import nereus as nr

# Load MPAS mesh from any MPAS-Ocean NetCDF file
>>> mesh = nr.mpas.load_mesh("/path/to/mpas_output.nc")

# Open data file with mesh coordinates attached
>>> ds = nr.mpas.open_dataset("/path/to/mpas_output.nc", mesh=mesh)
"""

from nereus.models.mpas.mesh import load_mesh, open_dataset

__all__ = [
    "load_mesh",
    "open_dataset",
]
