"""FESOM2 model support for nereus.

This module provides functionality for working with FESOM2 ocean model data,
including mesh loading and data handling.

Examples
--------
>>> import nereus as nr

# Load a FESOM mesh (returns xr.Dataset)
>>> mesh = nr.fesom.load_mesh("/path/to/mesh")
>>> print(f"Mesh has {mesh.sizes['npoints']} nodes")

# Access mesh data
>>> lon = mesh["lon"]  # xr.DataArray
>>> lon_np = mesh["lon"].values  # numpy array
>>> area = mesh["area"]
>>> triangles = mesh["triangles"]

# Open data with mesh coordinates
>>> ds = nr.fesom.open_dataset("temp.fesom.2010.nc", mesh=mesh)

# Use with diagnostics
>>> ice_area = nr.ice_area(sic, mesh["area"], mask=mesh["lat"] > 0)

# Use with plotting
>>> fig, ax, _ = nr.plot(ds.temp[0, 0, :], mesh["lon"].values, mesh["lat"].values)
"""

from nereus.models.fesom.functions import (
    compute_element_centers,
    element_to_node,
    mask_by_depth,
    node_to_element,
)
from nereus.models.fesom.mesh import load_mesh, open_dataset

__all__ = [
    "compute_element_centers",
    "element_to_node",
    "load_mesh",
    "mask_by_depth",
    "node_to_element",
    "open_dataset",
]
