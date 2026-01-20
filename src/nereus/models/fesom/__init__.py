"""FESOM2 model support for nereus.

This module provides functionality for working with FESOM2 ocean model data,
including mesh loading and data handling.

Examples
--------
>>> import nereus as nr

# Load a FESOM mesh
>>> mesh = nr.fesom.load_mesh("/path/to/mesh")
>>> print(f"Mesh has {mesh.n2d} nodes, {mesh.nlev} levels")

# Open data with mesh coordinates
>>> ds = nr.fesom.open_dataset("temp.fesom.2010.nc", mesh=mesh)

# Use with plotting
>>> fig, ax, _ = nr.plot(ds.temp[0, 0, :], mesh.lon, mesh.lat)
"""

from nereus.models.fesom.mesh import FesomMesh, load_mesh, open_dataset

__all__ = [
    "FesomMesh",
    "load_mesh",
    "open_dataset",
]
