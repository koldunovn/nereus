"""NEMO ocean model mesh loading.

This module provides functionality for loading NEMO meshes from mesh_mask.nc
or coordinates files as xr.Dataset objects with standardized variable names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from nereus.core.coordinates import EARTH_RADIUS
from nereus.core.mesh import (
    add_mesh_metadata,
    normalize_lon,
    should_use_dask,
)

if TYPE_CHECKING:
    pass


def load_mesh(
    path: str | os.PathLike,
    *,
    use_dask: bool | None = None,
    mask_var: str = "tmask",
    lon_var: str | None = None,
    lat_var: str | None = None,
    area_var: str | None = None,
) -> xr.Dataset:
    """Load NEMO mesh from mesh_mask.nc or coordinates file.

    Flattens 2D coordinates to 1D for compatibility with nereus functions.
    Ocean points are identified using the mask variable.

    Parameters
    ----------
    path : str or Path
        Path to mesh_mask.nc or coordinates file.
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.
    mask_var : str
        Variable name for ocean mask (default: "tmask").
        Set to None to include all points.
    lon_var : str, optional
        Longitude variable name. Auto-detects if None.
    lat_var : str, optional
        Latitude variable name. Auto-detects if None.
    area_var : str, optional
        Cell area variable name. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Mesh dataset with:
        - lon, lat: Flattened coordinates (npoints,)
        - area: Cell area in m^2 (npoints,)
        - mask: Ocean mask (npoints,)
        Attributes include original 2D shape for reshaping.

    Examples
    --------
    >>> mesh = nr.nemo.load_mesh("/path/to/mesh_mask.nc")
    >>> print(f"Ocean points: {mesh.sizes['npoints']}")

    >>> # Include all points (land + ocean)
    >>> mesh = nr.nemo.load_mesh(path, mask_var=None)
    """
    path = Path(path)

    # Open dataset
    ds_orig = xr.open_dataset(path)

    # Auto-detect variable names
    if lon_var is None:
        lon_var = _find_var(ds_orig, ["glamt", "nav_lon", "lon", "longitude"])
    if lat_var is None:
        lat_var = _find_var(ds_orig, ["gphit", "nav_lat", "lat", "latitude"])
    if area_var is None:
        area_var = _find_var(ds_orig, ["e1t", "e2t", "area"], optional=True)

    if lon_var is None:
        raise ValueError("Could not find longitude variable")
    if lat_var is None:
        raise ValueError("Could not find latitude variable")

    # Get 2D coordinates
    lon_2d = ds_orig[lon_var].values.squeeze()
    lat_2d = ds_orig[lat_var].values.squeeze()

    # Ensure 2D
    if lon_2d.ndim != 2:
        raise ValueError(f"Expected 2D longitude, got shape {lon_2d.shape}")

    # Get mask
    mask_2d = None
    if mask_var and mask_var in ds_orig:
        mask_data = ds_orig[mask_var].values
        # Take surface level if 3D/4D
        while mask_data.ndim > 2:
            mask_data = mask_data[0]
        mask_2d = mask_data > 0

    # Flatten coordinates
    lon, lat, flat_indices = flatten_structured(lon_2d, lat_2d, mask=mask_2d)

    # Normalize longitude
    lon = normalize_lon(lon, "pm180")

    npoints = len(lon)
    use_dask_actual = should_use_dask(npoints, use_dask)

    # Compute area
    if area_var and "e1t" in ds_orig and "e2t" in ds_orig:
        # NEMO stores dx, dy separately
        e1t = ds_orig["e1t"].values.squeeze()
        e2t = ds_orig["e2t"].values.squeeze()
        area_2d = e1t * e2t
        if mask_2d is not None:
            area = area_2d[mask_2d]
        else:
            area = area_2d.ravel()
    elif area_var and area_var in ds_orig:
        area_2d = ds_orig[area_var].values.squeeze()
        if mask_2d is not None:
            area = area_2d[mask_2d]
        else:
            area = area_2d.ravel()
    else:
        # Estimate from grid spacing
        area = _estimate_structured_area(lon_2d, lat_2d, mask_2d)

    if use_dask_actual:
        import dask.array as da

        lon = da.from_array(lon, chunks=-1)
        lat = da.from_array(lat, chunks=-1)
        area = da.from_array(area, chunks=-1)

    ds = xr.Dataset(
        {
            "lon": (("npoints",), lon, {
                "units": "degrees_east",
                "long_name": "Longitude",
                "standard_name": "longitude",
            }),
            "lat": (("npoints",), lat, {
                "units": "degrees_north",
                "long_name": "Latitude",
                "standard_name": "latitude",
            }),
            "area": (("npoints",), area, {
                "units": "m2",
                "long_name": "Cell area",
            }),
        },
        attrs={
            "nlon": lon_2d.shape[1],
            "nlat": lon_2d.shape[0],
            "original_shape": lon_2d.shape,
        },
    )

    # Store depth levels if available
    if "gdept_1d" in ds_orig:
        depth = ds_orig["gdept_1d"].values.squeeze()
        ds["depth"] = xr.DataArray(
            depth,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Depth of layer centers",
                "positive": "down",
            },
        )

    if "gdepw_1d" in ds_orig:
        depth_w = ds_orig["gdepw_1d"].values.squeeze()
        if len(depth_w) > 1:
            layer_thickness = np.diff(depth_w)
            ds["layer_thickness"] = xr.DataArray(
                layer_thickness,
                dims=("depth_level",),
                attrs={
                    "units": "m",
                    "long_name": "Layer thickness",
                },
            )

    ds_orig.close()

    return add_mesh_metadata(ds, "nemo", path, use_dask=use_dask_actual)


def flatten_structured(
    lon_2d: NDArray[np.floating],
    lat_2d: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.intp]]:
    """Flatten 2D structured coordinates to 1D.

    Parameters
    ----------
    lon_2d : ndarray
        2D longitude array (nlat, nlon).
    lat_2d : ndarray
        2D latitude array (nlat, nlon).
    mask : ndarray, optional
        Boolean mask (True = include). If None, includes all points.

    Returns
    -------
    lon : ndarray
        Flattened longitude (npoints,).
    lat : ndarray
        Flattened latitude (npoints,).
    indices : ndarray
        Flat indices for reconstructing 2D arrays.

    Examples
    --------
    >>> lon, lat, indices = nr.nemo.flatten_structured(lon_2d, lat_2d, mask=ocean_mask)
    >>> # To reshape data back to 2D:
    >>> data_2d = np.full(lon_2d.shape, np.nan)
    >>> data_2d.flat[indices] = data_1d
    """
    lon_2d = np.asarray(lon_2d)
    lat_2d = np.asarray(lat_2d)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        indices = np.where(mask.ravel())[0]
        lon = lon_2d.ravel()[indices]
        lat = lat_2d.ravel()[indices]
    else:
        indices = np.arange(lon_2d.size)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()

    return lon.astype(np.float64), lat.astype(np.float64), indices


def _find_var(ds: xr.Dataset, candidates: list[str], optional: bool = False) -> str | None:
    """Find first matching variable name."""
    for name in candidates:
        if name in ds:
            return name
    if optional:
        return None
    return None


def _estimate_structured_area(
    lon_2d: NDArray[np.floating],
    lat_2d: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.floating]:
    """Estimate cell areas for structured grid.

    Uses finite differences to estimate dx, dy, then computes area.

    Parameters
    ----------
    lon_2d : ndarray
        2D longitude (nlat, nlon).
    lat_2d : ndarray
        2D latitude (nlat, nlon).
    mask : ndarray, optional
        Boolean mask.

    Returns
    -------
    ndarray
        Cell areas in m^2.
    """
    nlat, nlon = lon_2d.shape

    # Compute grid spacing using finite differences
    # dx: spacing in longitude direction
    dlon = np.zeros_like(lon_2d)
    dlon[:, 1:-1] = (lon_2d[:, 2:] - lon_2d[:, :-2]) / 2
    dlon[:, 0] = lon_2d[:, 1] - lon_2d[:, 0]
    dlon[:, -1] = lon_2d[:, -1] - lon_2d[:, -2]

    # Handle wraparound
    dlon = np.abs(dlon)
    dlon = np.where(dlon > 180, 360 - dlon, dlon)

    # dy: spacing in latitude direction
    dlat = np.zeros_like(lat_2d)
    dlat[1:-1, :] = (lat_2d[2:, :] - lat_2d[:-2, :]) / 2
    dlat[0, :] = lat_2d[1, :] - lat_2d[0, :]
    dlat[-1, :] = lat_2d[-1, :] - lat_2d[-2, :]
    dlat = np.abs(dlat)

    # Convert to radians
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)
    lat_rad = np.deg2rad(lat_2d)

    # Area = R^2 * cos(lat) * dlon * dlat
    area_2d = EARTH_RADIUS**2 * np.cos(lat_rad) * dlon_rad * dlat_rad

    if mask is not None:
        return area_2d[mask]
    else:
        return area_2d.ravel()
