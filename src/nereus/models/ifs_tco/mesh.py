"""IFS TCO mesh loading.

This module provides functionality for loading IFS TCO meshes as xr.Dataset
objects with standardized variable names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from nereus.core.mesh import (
    add_mesh_metadata,
    normalize_lon,
    should_use_dask,
)

if TYPE_CHECKING:
    pass


def load_mesh(
    grid_file: str | os.PathLike,
    area_file: str | os.PathLike,
    *,
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Load IFS TCO mesh from grid and area files.

    Parameters
    ----------
    grid_file : str or Path
        Path to the grid NetCDF file containing A*.lon and A*.lat.
    area_file : str or Path
        Path to the area NetCDF file containing A*.srf.
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Standardized mesh dataset with:
        - lon, lat: Coordinates (npoints,)
        - area: Cell area in m^2 (npoints,)
        Plus original variables with their native names.
    """
    grid_path = Path(grid_file)
    area_path = Path(area_file)

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    if not area_path.exists():
        raise FileNotFoundError(f"Area file not found: {area_path}")

    with xr.open_dataset(grid_path) as ds_grid_peek, xr.open_dataset(area_path) as ds_area_peek:
        prefix = _select_prefix(ds_grid_peek, ds_area_peek)
        lon_name = f"{prefix}.lon"
        npoints = ds_grid_peek[lon_name].size

    use_dask_actual = should_use_dask(npoints, use_dask)

    if use_dask_actual:
        ds_grid = xr.open_dataset(grid_path, chunks={})
        ds_area = xr.open_dataset(area_path, chunks={})
    else:
        ds_grid = xr.open_dataset(grid_path)
        ds_area = xr.open_dataset(area_path)

    try:
        lon_var = f"{prefix}.lon"
        lat_var = f"{prefix}.lat"
        area_var = f"{prefix}.srf"

        lon_da = ds_grid[lon_var]
        lat_da = ds_grid[lat_var]
        area_da = ds_area[area_var]

        if lon_da.dims != lat_da.dims:
            raise ValueError(
                f"lon/lat dimension mismatch: {lon_da.dims} vs {lat_da.dims}"
            )

        lon_data = lon_da.data if use_dask_actual else lon_da.values
        lat_data = lat_da.data if use_dask_actual else lat_da.values
        area_data = area_da.data if use_dask_actual else area_da.values

        if use_dask_actual:
            import dask.array as da

            lon_data = da.map_blocks(
                lambda x: normalize_lon(x, "pm180"),
                lon_data,
                dtype=np.float64,
            )
            lon_flat = da.ravel(lon_data)
            lat_flat = da.ravel(lat_data)
            area_flat = da.ravel(area_data)
        else:
            lon_flat = np.ravel(normalize_lon(lon_data, "pm180"))
            lat_flat = np.ravel(lat_data)
            area_flat = np.ravel(area_data)

        ds = xr.Dataset(
            {
                "lon": (("npoints",), lon_flat, {
                    "units": "degrees_east",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                }),
                "lat": (("npoints",), lat_flat, {
                    "units": "degrees_north",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                }),
                "area": (("npoints",), area_flat, {
                    "units": "m2",
                    "long_name": "Cell area",
                }),
            },
            attrs={
                "ifs_tco_prefix": prefix,
                "original_shape": lon_da.shape,
            },
        )

        _copy_variables(ds, ds_grid, use_dask_actual, skip_existing=True)
        _copy_variables(ds, ds_area, use_dask_actual, skip_existing=True)
    finally:
        if not use_dask_actual:
            ds_grid.close()
            ds_area.close()

    ds.attrs["ifs_tco_grid_file"] = str(grid_path)
    ds.attrs["ifs_tco_area_file"] = str(area_path)

    return add_mesh_metadata(ds, "ifs_tco", grid_path, use_dask=use_dask_actual)


def _select_prefix(ds_grid: xr.Dataset, ds_area: xr.Dataset) -> str:
    """Select A* prefix present in grid and area files."""
    lon_prefixes = _collect_prefixes(ds_grid, "lon")
    lat_prefixes = _collect_prefixes(ds_grid, "lat")
    area_prefixes = _collect_prefixes(ds_area, "srf")

    candidates = lon_prefixes & lat_prefixes & area_prefixes
    if not candidates:
        raise ValueError(
            "Could not find matching A* prefix for lon/lat and area. "
            f"lon prefixes: {sorted(lon_prefixes)}, "
            f"lat prefixes: {sorted(lat_prefixes)}, "
            f"area prefixes: {sorted(area_prefixes)}"
        )

    return sorted(candidates)[0]


def _collect_prefixes(ds: xr.Dataset, suffix: str) -> set[str]:
    """Collect prefixes for variables matching A*.suffix."""
    prefixes: set[str] = set()
    for name in ds.variables:
        if not name.startswith("A"):
            continue
        if not name.endswith(f".{suffix}"):
            continue
        prefix = name.split(".", 1)[0]
        if prefix:
            prefixes.add(prefix)
    return prefixes


def _copy_variables(
    dest: xr.Dataset,
    src: xr.Dataset,
    use_dask: bool,
    *,
    skip_existing: bool = False,
) -> None:
    """Copy variables and coordinates from source dataset to destination."""
    for name in src.data_vars:
        if skip_existing and name in dest:
            continue
        var = src[name]
        data = var.data if use_dask else var.values
        dest[name] = xr.DataArray(data, dims=var.dims, attrs=var.attrs)

    for name in src.coords:
        if skip_existing and (name in dest or name in dest.coords):
            continue
        coord = src[name]
        data = coord.data if use_dask else coord.values
        dims = coord.dims if coord.dims else ()
        dest.coords[name] = xr.DataArray(data, dims=dims, attrs=coord.attrs)
