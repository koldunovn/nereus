"""MPAS-Ocean mesh loading.

Reads MPAS-Ocean NetCDF files and produces a standardized nereus
``xr.Dataset`` with flattened 1D coordinates.  MPAS stores mesh
information (coordinates, areas, connectivity) directly in every
output file, so any MPAS-Ocean NetCDF file can serve as the mesh source.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from nereus.core.mesh import (
    add_mesh_metadata,
    normalize_lon,
    should_use_dask,
)


def load_mesh(
    path: str | os.PathLike,
    *,
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Load MPAS-Ocean mesh from a NetCDF file.

    Any MPAS-Ocean NetCDF file can be used because mesh variables
    (``lonCell``, ``latCell``, ``areaCell``, etc.) are embedded in
    every output file.

    Parameters
    ----------
    path : str or Path
        Path to an MPAS-Ocean NetCDF file.
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Mesh dataset with ``lon``, ``lat``, ``area`` on ``npoints`` dimension.
        Also includes ``depth``, ``layer_thickness``, ``bathymetry``,
        ``maxLevelCell`` when available in the source file.

    Examples
    --------
    >>> mesh = nr.mpas.load_mesh("/path/to/mpas_output.nc")
    """
    path = Path(path)

    src = xr.open_dataset(path)

    # --- Coordinates (radians -> degrees) --------------------------------
    lon_rad = src["lonCell"].values
    lat_rad = src["latCell"].values

    lon_deg = np.degrees(lon_rad).astype(np.float64)
    lat_deg = np.degrees(lat_rad).astype(np.float64)

    lon_deg = normalize_lon(lon_deg, "pm180")

    # --- Cell areas (already in m²) --------------------------------------
    area = src["areaCell"].values.astype(np.float64)

    npoints = len(lon_deg)
    use_dask_actual = should_use_dask(npoints, use_dask)

    if use_dask_actual:
        import dask.array as da

        lon_data = da.from_array(lon_deg, chunks=-1)
        lat_data = da.from_array(lat_deg, chunks=-1)
        area_data = da.from_array(area, chunks=-1)
    else:
        lon_data = lon_deg
        lat_data = lat_deg
        area_data = area

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
    )

    # --- Depth levels from refBottomDepth --------------------------------
    if "refBottomDepth" in src:
        ref_bottom = src["refBottomDepth"].values.astype(np.float64)
        layer_thickness = np.diff(np.concatenate([[0.0], ref_bottom]))
        depth = ref_bottom - layer_thickness / 2.0

        ds["depth"] = xr.DataArray(
            depth,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Depth of layer centers",
                "positive": "down",
            },
        )
        ds["layer_thickness"] = xr.DataArray(
            layer_thickness,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Layer thickness",
            },
        )

    # --- Bathymetry ------------------------------------------------------
    if "bottomDepth" in src:
        bathy = src["bottomDepth"].values.astype(np.float64)
        if use_dask_actual:
            import dask.array as da

            bathy = da.from_array(bathy, chunks=-1)
        ds["bathymetry"] = xr.DataArray(
            bathy,
            dims=("npoints",),
            attrs={
                "units": "m",
                "long_name": "Bathymetry",
                "positive": "down",
            },
        )

    # --- Land masking via maxLevelCell -----------------------------------
    if "maxLevelCell" in src:
        max_level = src["maxLevelCell"].values
        if use_dask_actual:
            import dask.array as da

            max_level = da.from_array(max_level, chunks=-1)
        ds["maxLevelCell"] = xr.DataArray(
            max_level,
            dims=("npoints",),
            attrs={
                "units": "1",
                "long_name": "Index of deepest active ocean cell",
            },
        )

    src.close()

    return add_mesh_metadata(ds, "mpas", path, use_dask=use_dask_actual)


def open_dataset(
    data_path: str | os.PathLike,
    mesh: xr.Dataset | None = None,
    mesh_path: str | os.PathLike | None = None,
) -> xr.Dataset:
    """Open an MPAS-Ocean data file with mesh coordinates attached.

    Parameters
    ----------
    data_path : str or path-like
        Path to the data file (NetCDF).
    mesh : xr.Dataset, optional
        Pre-loaded mesh dataset. If not provided, mesh_path must be
        specified, or mesh will be loaded from the data file itself.
    mesh_path : str or path-like, optional
        Path to a file to load mesh from. Ignored if mesh is provided.

    Returns
    -------
    xr.Dataset
        Dataset with mesh coordinates attached.

    Examples
    --------
    >>> mesh = nr.mpas.load_mesh("/path/to/mpas_output.nc")
    >>> ds = nr.mpas.open_dataset("/path/to/another_output.nc", mesh=mesh)

    >>> # Or load mesh from the data file itself
    >>> ds = nr.mpas.open_dataset("/path/to/mpas_output.nc")
    """
    if mesh is None:
        if mesh_path is not None:
            mesh = load_mesh(mesh_path)
        else:
            mesh = load_mesh(data_path)

    ds = xr.open_dataset(data_path)

    lon_data = mesh["lon"].values
    lat_data = mesh["lat"].values

    if "nCells" in ds.dims:
        ds = ds.assign_coords(
            lon=("nCells", lon_data),
            lat=("nCells", lat_data),
        )

    if "depth" in mesh and "nVertLevels" in ds.dims:
        ds = ds.assign_coords(
            depth=("nVertLevels", mesh["depth"].values),
        )

    return ds
