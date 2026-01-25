"""FESOM-specific functions.

This module provides functions specific to FESOM mesh operations,
such as computing element centers and node-to-element interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from nereus.core.coordinates import compute_element_centers as _compute_element_centers

if TYPE_CHECKING:
    pass


def compute_element_centers(mesh: xr.Dataset) -> xr.Dataset:
    """Compute element center coordinates.

    Adds lon_tri and lat_tri variables to the mesh dataset.

    Parameters
    ----------
    mesh : xr.Dataset
        FESOM mesh dataset with lon, lat, and triangles.

    Returns
    -------
    xr.Dataset
        Mesh with lon_tri and lat_tri added.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh(path)
    >>> mesh = nr.fesom.compute_element_centers(mesh)
    >>> lon_elem = mesh["lon_tri"]
    """
    if "lon_tri" in mesh and "lat_tri" in mesh:
        return mesh

    if "triangles" not in mesh:
        raise ValueError("Mesh does not contain triangles")

    lon = mesh["lon"].values
    lat = mesh["lat"].values
    triangles = mesh["triangles"].values

    lon_tri, lat_tri = _compute_element_centers(lon, lat, triangles)

    mesh = mesh.copy()
    mesh["lon_tri"] = xr.DataArray(
        lon_tri,
        dims=("nelem",),
        attrs={
            "units": "degrees_east",
            "long_name": "Element center longitude",
        },
    )
    mesh["lat_tri"] = xr.DataArray(
        lat_tri,
        dims=("nelem",),
        attrs={
            "units": "degrees_north",
            "long_name": "Element center latitude",
        },
    )

    return mesh


def node_to_element(
    data: xr.DataArray | NDArray[np.floating],
    mesh: xr.Dataset,
    method: Literal["mean", "median", "min", "max"] = "mean",
) -> xr.DataArray | NDArray[np.floating]:
    """Interpolate data from nodes to element centers.

    For each triangle, computes an aggregate of its three vertex values.

    Parameters
    ----------
    data : xr.DataArray or ndarray
        Node data with shape (..., npoints).
    mesh : xr.Dataset
        FESOM mesh dataset with triangles.
    method : {"mean", "median", "min", "max"}
        Aggregation method.

    Returns
    -------
    xr.DataArray or ndarray
        Element data with shape (..., nelem).

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh(path)
    >>> temp_elem = nr.fesom.node_to_element(temp_node, mesh)
    """
    if "triangles" not in mesh:
        raise ValueError("Mesh does not contain triangles")

    triangles = mesh["triangles"].values
    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        data_np = data.values
        orig_dims = data.dims
    else:
        data_np = np.asarray(data)
        orig_dims = None

    # Get the spatial dimension (last one)
    spatial_shape = data_np.shape[:-1]
    npoints = data_np.shape[-1]
    nelem = triangles.shape[0]

    # Get data at triangle vertices: shape (..., nelem, 3)
    vertex_data = data_np[..., triangles]

    # Aggregate
    if method == "mean":
        elem_data = np.mean(vertex_data, axis=-1)
    elif method == "median":
        elem_data = np.median(vertex_data, axis=-1)
    elif method == "min":
        elem_data = np.min(vertex_data, axis=-1)
    elif method == "max":
        elem_data = np.max(vertex_data, axis=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    if is_xarray:
        # Create new DataArray with element dimension
        new_dims = orig_dims[:-1] + ("nelem",)
        result = xr.DataArray(
            elem_data,
            dims=new_dims,
            attrs=data.attrs,
        )
        # Copy coordinates except the spatial one
        for coord, coord_data in data.coords.items():
            if coord not in data.dims[-1:]:
                result = result.assign_coords({coord: coord_data})
        return result
    else:
        return elem_data


def element_to_node(
    data: xr.DataArray | NDArray[np.floating],
    mesh: xr.Dataset,
    method: Literal["mean", "sum"] = "mean",
) -> xr.DataArray | NDArray[np.floating]:
    """Interpolate data from element centers to nodes.

    For each node, aggregates values from all connected elements.

    Parameters
    ----------
    data : xr.DataArray or ndarray
        Element data with shape (..., nelem).
    mesh : xr.Dataset
        FESOM mesh dataset with triangles.
    method : {"mean", "sum"}
        Aggregation method.

    Returns
    -------
    xr.DataArray or ndarray
        Node data with shape (..., npoints).

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh(path)
    >>> temp_node = nr.fesom.element_to_node(temp_elem, mesh)
    """
    if "triangles" not in mesh:
        raise ValueError("Mesh does not contain triangles")

    triangles = mesh["triangles"].values
    npoints = mesh.sizes["npoints"]
    nelem = triangles.shape[0]

    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        data_np = data.values
        orig_dims = data.dims
    else:
        data_np = np.asarray(data)
        orig_dims = None

    # Count how many elements each node belongs to
    node_count = np.zeros(npoints, dtype=np.float64)
    for tri in triangles:
        node_count[tri] += 1

    # Shape handling for broadcasting
    spatial_shape = data_np.shape[:-1]

    # Initialize output
    node_data = np.zeros(spatial_shape + (npoints,), dtype=data_np.dtype)

    # Accumulate element values to nodes
    for i, tri in enumerate(triangles):
        node_data[..., tri] += data_np[..., i:i + 1]

    if method == "mean":
        # Avoid division by zero
        node_count = np.where(node_count > 0, node_count, 1)
        node_data = node_data / node_count
    elif method == "sum":
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    if is_xarray:
        new_dims = orig_dims[:-1] + ("npoints",)
        result = xr.DataArray(
            node_data,
            dims=new_dims,
            attrs=data.attrs,
        )
        for coord, coord_data in data.coords.items():
            if coord not in data.dims[-1:]:
                result = result.assign_coords({coord: coord_data})
        return result
    else:
        return node_data


def mask_by_depth(
    data: xr.DataArray | NDArray[np.floating],
    mask: xr.DataArray | NDArray[np.bool_],
) -> NDArray[np.floating]:
    """Apply depth mask to data, setting invalid cells to NaN.

    This function masks ocean data based on bottom topography. Use with
    nod_area_mask (for node-based variables) or elem_area_mask (for
    element-based variables) from the FESOM mesh.

    Parameters
    ----------
    data : xr.DataArray or ndarray
        Data to mask. Can be:
        - 2D array (nz, npoints) or (nz, nelem) for 3D fields
        - 1D array (npoints,) or (nelem,) for 2D fields at a single level
    mask : xr.DataArray or ndarray
        Boolean mask where True = valid ocean cell, False = below bottom/land.
        Can be:
        - Full 3D mask (nz, npoints) or (nz, nelem) - must match data shape
        - Single level 1D mask (npoints,) or (nelem,) - for 2D data

    Returns
    -------
    ndarray
        Masked data as numpy array with NaN where mask is False.
        Returns float64 to support NaN values.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh(path)
    >>> # Mask 3D temperature field
    >>> temp_masked = nr.fesom.mask_by_depth(temp_3d, mesh.nod_area_mask)
    >>>
    >>> # Mask single level (e.g., level 10)
    >>> temp_lev10_masked = nr.fesom.mask_by_depth(
    ...     temp_3d[10, :], mesh.nod_area_mask[10, :]
    ... )
    >>>
    >>> # For element-based data (e.g., velocity)
    >>> u_masked = nr.fesom.mask_by_depth(u_3d, mesh.elem_area_mask)

    Notes
    -----
    The function is dimension-name-agnostic - it works purely by array shape.
    This avoids issues with inconsistent dimension names across different
    FESOM output files (nod2, npoints, ncells, etc.).
    """
    # Convert to numpy arrays
    if hasattr(data, "values"):
        data_np = data.values
    else:
        data_np = np.asarray(data)

    if hasattr(mask, "values"):
        mask_np = mask.values
    else:
        mask_np = np.asarray(mask)

    # Validate shapes match
    if data_np.shape != mask_np.shape:
        raise ValueError(
            f"Data shape {data_np.shape} does not match mask shape {mask_np.shape}. "
            "Ensure you're using the correct mask (nod_area_mask for node data, "
            "elem_area_mask for element data) and matching depth levels."
        )

    # Create output array (float64 to support NaN)
    result = data_np.astype(np.float64, copy=True)

    # Apply mask: set False locations to NaN
    result[~mask_np] = np.nan

    return result
