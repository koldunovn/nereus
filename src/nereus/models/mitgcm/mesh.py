"""MITgcm mesh loading.

Loads MITgcm grid files (``XC.data``, ``YC.data``, etc.) and produces
a standardized nereus ``xr.Dataset`` with flattened 1D coordinates.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from nereus.core.coordinates import EARTH_RADIUS
from nereus.core.mesh import (
    add_mesh_metadata,
    normalize_lon,
    should_use_dask,
)
from nereus.models.mitgcm.io import read_mds


def load_mesh(
    path: str | os.PathLike,
    *,
    grid_type: str = "C",
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Load MITgcm mesh from a directory of MDS grid files.

    Parameters
    ----------
    path : str or Path
        Directory containing MITgcm grid files (``XC.data``, ``YC.data``, etc.).
    grid_type : {"C", "G"}
        Which grid point to use:
        - ``"C"``: Cell centers (reads ``XC``/``YC``).
        - ``"G"``: Cell corners (reads ``XG``/``YG``).
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Mesh dataset with ``lon``, ``lat``, ``area`` on ``npoints`` dimension.
        Optionally includes ``depth``, ``layer_thickness``, ``bathymetry``.
        Attributes include ``nx``, ``ny``, ``original_shape``.

    Examples
    --------
    >>> mesh = nr.mitgcm.load_mesh("/path/to/run/")
    >>> print(mesh)
    """
    path = Path(path)

    # Select coordinate file prefixes based on grid type
    if grid_type == "C":
        lon_prefix, lat_prefix = "XC", "YC"
    elif grid_type == "G":
        lon_prefix, lat_prefix = "XG", "YG"
    else:
        raise ValueError(f"Unknown grid_type: {grid_type!r}. Use 'C' or 'G'.")

    # Read 2D coordinate arrays
    _, lon_2d = read_mds(path / lon_prefix)
    _, lat_2d = read_mds(path / lat_prefix)

    lon_2d = lon_2d.astype(np.float64)
    lat_2d = lat_2d.astype(np.float64)

    ny, nx = lon_2d.shape

    # Read cell areas
    if (path / "RAC.meta").exists():
        _, area_2d = read_mds(path / "RAC")
        area_flat = area_2d.ravel().astype(np.float64)
    else:
        area_flat = _estimate_structured_area(lon_2d, lat_2d)

    # Flatten and normalize
    lon_flat = normalize_lon(lon_2d.ravel(), "pm180")
    lat_flat = lat_2d.ravel().astype(np.float64)

    npoints = len(lon_flat)
    use_dask_actual = should_use_dask(npoints, use_dask)

    if use_dask_actual:
        import dask.array as da

        lon_data = da.from_array(lon_flat, chunks=-1)
        lat_data = da.from_array(lat_flat, chunks=-1)
        area_data = da.from_array(area_flat, chunks=-1)
    else:
        lon_data = lon_flat
        lat_data = lat_flat
        area_data = area_flat

    ds = xr.Dataset(
        {
            "lon": (("npoints",), lon_data, {
                "units": "degrees_east",
                "long_name": "Longitude",
                "standard_name": "longitude",
            }),
            "lat": (("npoints",), lat_data, {
                "units": "degrees_north",
                "long_name": "Latitude",
                "standard_name": "latitude",
            }),
            "area": (("npoints",), area_data, {
                "units": "m2",
                "long_name": "Cell area",
            }),
        },
        attrs={
            "nx": nx,
            "ny": ny,
            "original_shape": (ny, nx),
        },
    )

    # Depth levels (RC.data = center depths, DRF.data = layer thicknesses)
    if (path / "RC.meta").exists():
        _, rc = read_mds(path / "RC")
        depth = np.abs(rc.ravel()).astype(np.float64)
        ds["depth"] = xr.DataArray(
            depth,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Depth of layer centers",
                "positive": "down",
            },
        )

    if (path / "DRF.meta").exists():
        _, drf = read_mds(path / "DRF")
        layer_thickness = drf.ravel().astype(np.float64)
        ds["layer_thickness"] = xr.DataArray(
            layer_thickness,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Layer thickness",
            },
        )

    # Bathymetry
    if (path / "Depth.meta").exists():
        _, bathy = read_mds(path / "Depth")
        bathy_flat = bathy.ravel().astype(np.float64)
        if use_dask_actual:
            import dask.array as da

            bathy_flat = da.from_array(bathy_flat, chunks=-1)
        ds["bathymetry"] = xr.DataArray(
            bathy_flat,
            dims=("npoints",),
            attrs={
                "units": "m",
                "long_name": "Bathymetry",
                "positive": "down",
            },
        )

    return add_mesh_metadata(ds, "mitgcm", path, use_dask=use_dask_actual)


def _estimate_structured_area(
    lon_2d: NDArray[np.floating],
    lat_2d: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Estimate cell areas from coordinate arrays using finite differences.

    Parameters
    ----------
    lon_2d : ndarray
        2D longitude array (ny, nx).
    lat_2d : ndarray
        2D latitude array (ny, nx).

    Returns
    -------
    ndarray
        Flattened cell areas in m^2.
    """
    nlat, nlon = lon_2d.shape

    # dx: spacing in longitude direction
    dlon = np.zeros_like(lon_2d)
    dlon[:, 1:-1] = (lon_2d[:, 2:] - lon_2d[:, :-2]) / 2
    dlon[:, 0] = lon_2d[:, 1] - lon_2d[:, 0]
    dlon[:, -1] = lon_2d[:, -1] - lon_2d[:, -2]
    dlon = np.abs(dlon)
    dlon = np.where(dlon > 180, 360 - dlon, dlon)

    # dy: spacing in latitude direction
    dlat = np.zeros_like(lat_2d)
    dlat[1:-1, :] = (lat_2d[2:, :] - lat_2d[:-2, :]) / 2
    dlat[0, :] = lat_2d[1, :] - lat_2d[0, :]
    dlat[-1, :] = lat_2d[-1, :] - lat_2d[-2, :]
    dlat = np.abs(dlat)

    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)
    lat_rad = np.deg2rad(lat_2d)

    area_2d = EARTH_RADIUS**2 * np.cos(lat_rad) * dlon_rad * dlat_rad
    return np.abs(area_2d).ravel()
