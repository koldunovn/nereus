"""Core mesh utilities for nereus.

This module provides utilities for creating, validating, and normalizing
mesh datasets. Meshes are represented as xr.Dataset objects with
standardized variable names.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from nereus.core.coordinates import EARTH_RADIUS

if TYPE_CHECKING:
    pass

# Threshold for automatic dask usage
DASK_THRESHOLD_POINTS = 1_000_000

# Nereus mesh metadata version
MESH_VERSION = "1.0"


def should_use_dask(npoints: int, use_dask: bool | None = None) -> bool:
    """Determine whether to use dask arrays based on mesh size.

    Parameters
    ----------
    npoints : int
        Number of mesh points.
    use_dask : bool, optional
        Explicit setting. If None, auto-detects based on threshold.

    Returns
    -------
    bool
        Whether to use dask arrays.
    """
    if use_dask is not None:
        return use_dask
    return npoints > DASK_THRESHOLD_POINTS


def normalize_lon(
    lon: NDArray[np.floating],
    convention: Literal["pm180", "0360"] = "pm180",
) -> NDArray[np.floating]:
    """Normalize longitude to specified convention.

    Parameters
    ----------
    lon : array_like
        Longitude values in degrees.
    convention : {"pm180", "0360"}
        Target convention:
        - "pm180": [-180, 180]
        - "0360": [0, 360]

    Returns
    -------
    ndarray
        Normalized longitude values.
    """
    lon = np.asarray(lon, dtype=np.float64)
    if convention == "pm180":
        # Normalize to [-180, 180]
        lon = np.mod(lon + 180, 360) - 180
    elif convention == "0360":
        # Normalize to [0, 360]
        lon = np.mod(lon, 360)
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return lon


def ensure_lon_pm180(ds: xr.Dataset) -> xr.Dataset:
    """Ensure longitude is normalized to [-180, 180].

    Parameters
    ----------
    ds : xr.Dataset
        Mesh dataset with 'lon' variable.

    Returns
    -------
    xr.Dataset
        Dataset with normalized longitude.
    """
    if "lon" not in ds:
        return ds

    lon_data = ds["lon"].values
    if np.any(lon_data > 180) or np.any(lon_data < -180):
        ds = ds.copy()
        ds["lon"] = (ds["lon"].dims, normalize_lon(lon_data, "pm180"))
        ds["lon"].attrs = {
            "units": "degrees_east",
            "long_name": "Longitude",
            "standard_name": "longitude",
        }
    return ds


def validate_mesh(ds: xr.Dataset, strict: bool = False) -> list[str]:
    """Validate mesh dataset against nereus conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.
    strict : bool
        If True, raise ValueError on validation errors.

    Returns
    -------
    list of str
        List of validation warnings/errors.

    Raises
    ------
    ValueError
        If strict=True and validation fails.
    """
    errors = []

    # Check required variables
    required = ["lon", "lat", "area"]
    for var in required:
        if var not in ds:
            errors.append(f"Missing required variable: {var}")

    # Check lon/lat have same dimension
    if "lon" in ds and "lat" in ds:
        if ds["lon"].dims != ds["lat"].dims:
            errors.append(
                f"lon and lat have different dimensions: "
                f"{ds['lon'].dims} vs {ds['lat'].dims}"
            )

    # Check lon range
    if "lon" in ds:
        lon_min = float(ds["lon"].min())
        lon_max = float(ds["lon"].max())
        if lon_min < -180 or lon_max > 180:
            errors.append(
                f"Longitude out of [-180, 180] range: [{lon_min}, {lon_max}]"
            )

    # Check lat range
    if "lat" in ds:
        lat_min = float(ds["lat"].min())
        lat_max = float(ds["lat"].max())
        if lat_min < -90 or lat_max > 90:
            errors.append(
                f"Latitude out of [-90, 90] range: [{lat_min}, {lat_max}]"
            )

    # Check area is positive
    if "area" in ds:
        if float(ds["area"].min()) <= 0:
            errors.append("Area contains non-positive values")

    # Check triangles if present
    if "triangles" in ds:
        tri_min = int(ds["triangles"].min())
        if tri_min < 0:
            errors.append(f"Triangles contain negative indices: min={tri_min}")
        if "lon" in ds:
            npoints = ds.sizes.get("npoints", len(ds["lon"]))
            tri_max = int(ds["triangles"].max())
            if tri_max >= npoints:
                errors.append(
                    f"Triangle index {tri_max} exceeds npoints={npoints}"
                )

    if strict and errors:
        raise ValueError("Mesh validation failed:\n" + "\n".join(errors))

    return errors


def add_mesh_metadata(
    ds: xr.Dataset,
    mesh_type: str,
    source_path: str | Path | None = None,
    use_dask: bool = False,
) -> xr.Dataset:
    """Add nereus mesh metadata to dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Mesh dataset.
    mesh_type : str
        Mesh type: "fesom", "healpix", "nemo", "lonlat".
    source_path : str or Path, optional
        Path to original mesh files.
    use_dask : bool
        Whether dask arrays are used.

    Returns
    -------
    xr.Dataset
        Dataset with metadata attributes.
    """
    ds = ds.copy()
    ds.attrs["nereus_mesh_type"] = mesh_type
    ds.attrs["nereus_mesh_version"] = MESH_VERSION
    ds.attrs["nereus_dask_backend"] = use_dask
    if source_path is not None:
        ds.attrs["nereus_source_path"] = str(source_path)
    return ds


def is_nereus_mesh(ds: xr.Dataset) -> bool:
    """Check if dataset is a nereus mesh.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check.

    Returns
    -------
    bool
        True if dataset has nereus mesh metadata.
    """
    return "nereus_mesh_type" in ds.attrs


def get_mesh_type(ds: xr.Dataset) -> str | None:
    """Get mesh type from nereus mesh dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Mesh dataset.

    Returns
    -------
    str or None
        Mesh type, or None if not a nereus mesh.
    """
    return ds.attrs.get("nereus_mesh_type")


def create_lonlat_mesh(
    resolution: float | tuple[float, float],
    *,
    lon_bounds: tuple[float, float] = (-180, 180),
    lat_bounds: tuple[float, float] = (-90, 90),
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Create regular lon-lat mesh.

    Parameters
    ----------
    resolution : float or tuple
        Grid resolution in degrees. If tuple, (dlon, dlat).
    lon_bounds : tuple
        Longitude bounds (min, max).
    lat_bounds : tuple
        Latitude bounds (min, max).
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Mesh dataset with flattened lon, lat, area.
    """
    if isinstance(resolution, (int, float)):
        dlon = dlat = float(resolution)
    else:
        dlon, dlat = resolution

    # Create 1D coordinate arrays
    lon_1d = np.arange(lon_bounds[0] + dlon / 2, lon_bounds[1], dlon)
    lat_1d = np.arange(lat_bounds[0] + dlat / 2, lat_bounds[1], dlat)

    # Create 2D grid and flatten
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    lon_flat = lon_2d.ravel()
    lat_flat = lat_2d.ravel()

    # Compute cell areas
    area_flat = _compute_lonlat_cell_area(lat_flat, dlon, dlat)

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
            "resolution_lon": dlon,
            "resolution_lat": dlat,
            "nlon": len(lon_1d),
            "nlat": len(lat_1d),
        },
    )

    return add_mesh_metadata(ds, "lonlat", use_dask=use_dask_actual)


def mesh_from_arrays(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    area: NDArray[np.floating] | None = None,
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Create mesh from existing coordinate arrays.

    Handles both 1D and 2D input arrays (2D will be flattened).

    Parameters
    ----------
    lon : array_like
        Longitude coordinates in degrees.
    lat : array_like
        Latitude coordinates in degrees.
    area : array_like, optional
        Cell areas in m^2. If None, estimates from grid spacing.
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Mesh dataset with lon, lat, area.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    # Flatten if 2D
    if lon.ndim == 2:
        lon = lon.ravel()
        lat = lat.ravel()

    # Normalize longitude
    lon = normalize_lon(lon, "pm180")

    npoints = len(lon)
    use_dask_actual = should_use_dask(npoints, use_dask)

    # Estimate area if not provided
    if area is None:
        area = _estimate_area(lon, lat)
    else:
        area = np.asarray(area, dtype=np.float64)
        if area.ndim == 2:
            area = area.ravel()

    if use_dask_actual:
        import dask.array as da

        lon_data = da.from_array(lon, chunks=-1)
        lat_data = da.from_array(lat, chunks=-1)
        area_data = da.from_array(area, chunks=-1)
    else:
        lon_data = lon
        lat_data = lat
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

    return add_mesh_metadata(ds, "custom", use_dask=use_dask_actual)


def _compute_lonlat_cell_area(
    lat: NDArray[np.floating],
    dlon: float,
    dlat: float,
) -> NDArray[np.floating]:
    """Compute area of regular lon-lat grid cells.

    Parameters
    ----------
    lat : array_like
        Cell center latitudes in degrees.
    dlon : float
        Longitudinal grid spacing in degrees.
    dlat : float
        Latitudinal grid spacing in degrees.

    Returns
    -------
    ndarray
        Cell areas in m^2.
    """
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    # Area = R^2 * dlon * (sin(lat+dlat/2) - sin(lat-dlat/2))
    lat_min = lat_rad - dlat_rad / 2
    lat_max = lat_rad + dlat_rad / 2

    # Clip to valid range
    lat_min = np.clip(lat_min, -np.pi / 2, np.pi / 2)
    lat_max = np.clip(lat_max, -np.pi / 2, np.pi / 2)

    area = EARTH_RADIUS**2 * dlon_rad * (np.sin(lat_max) - np.sin(lat_min))
    return np.abs(area)


def _estimate_area(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Estimate cell area from irregular grid (rough approximation).

    Uses Earth surface area divided by number of points.

    Parameters
    ----------
    lon : array_like
        Longitude in degrees.
    lat : array_like
        Latitude in degrees.

    Returns
    -------
    ndarray
        Estimated cell areas in m^2.
    """
    npoints = len(lon)
    earth_area = 4 * np.pi * EARTH_RADIUS**2
    avg_area = earth_area / npoints

    # Simple latitude weighting
    lat_rad = np.deg2rad(lat)
    weights = np.cos(lat_rad)
    weights = weights / weights.mean()

    return avg_area * weights
