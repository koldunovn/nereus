"""Grid utilities for nereus.

Functions for creating regular grids for regridding and plotting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def create_regular_grid(
    resolution: float | tuple[int, int] = 1.0,
    lon_bounds: tuple[float, float] = (-180.0, 180.0),
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
    center: Literal["cell", "node"] = "cell",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Create a regular lon/lat grid.

    Parameters
    ----------
    resolution : float or tuple of int
        Grid resolution. If float, specifies degrees per grid cell.
        If tuple (nlon, nlat), specifies number of grid points.
    lon_bounds : tuple of float
        Longitude bounds (lon_min, lon_max) in degrees.
    lat_bounds : tuple of float
        Latitude bounds (lat_min, lat_max) in degrees.
    center : {"cell", "node"}
        Whether coordinates are at cell centers or nodes.
        "cell" means coordinates at center of grid boxes.
        "node" means coordinates at corners.

    Returns
    -------
    lon, lat : tuple of ndarrays
        2D arrays of longitude and latitude coordinates.

    Examples
    --------
    >>> lon, lat = create_regular_grid(1.0)  # 1 degree resolution
    >>> lon.shape
    (180, 360)

    >>> lon, lat = create_regular_grid((360, 180))  # 360x180 grid
    >>> lon.shape
    (180, 360)
    """
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    if isinstance(resolution, (list, tuple)):
        nlon, nlat = resolution
    else:
        nlon = int((lon_max - lon_min) / resolution)
        nlat = int((lat_max - lat_min) / resolution)

    if center == "cell":
        # Cell centers
        dlon = (lon_max - lon_min) / nlon
        dlat = (lat_max - lat_min) / nlat
        lon_1d = np.linspace(lon_min + dlon / 2, lon_max - dlon / 2, nlon)
        lat_1d = np.linspace(lat_min + dlat / 2, lat_max - dlat / 2, nlat)
    else:
        # Node positions
        lon_1d = np.linspace(lon_min, lon_max, nlon)
        lat_1d = np.linspace(lat_min, lat_max, nlat)

    lon, lat = np.meshgrid(lon_1d, lat_1d)

    return lon, lat


def grid_cell_area(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    radius: float = 6_371_000.0,
) -> NDArray[np.floating]:
    """Compute area of regular grid cells.

    Parameters
    ----------
    lon : ndarray
        2D array of longitude coordinates (cell centers).
    lat : ndarray
        2D array of latitude coordinates (cell centers).
    radius : float
        Earth radius in meters.

    Returns
    -------
    ndarray
        2D array of cell areas in square meters.

    Notes
    -----
    Assumes uniform spacing in lon and lat.
    Area of a spherical rectangle:
    A = R^2 * |sin(lat1) - sin(lat2)| * |lon2 - lon1|
    """
    # Get grid spacing
    if lon.ndim == 2:
        dlon = np.abs(lon[0, 1] - lon[0, 0])
        dlat = np.abs(lat[1, 0] - lat[0, 0])
    else:
        dlon = np.abs(lon[1] - lon[0])
        dlat = np.abs(lat[1] - lat[0])

    dlon_rad = np.deg2rad(dlon)
    lat_rad = np.deg2rad(lat)

    # Half grid spacing in lat
    dlat_rad = np.deg2rad(dlat / 2)

    # sin(lat + dlat/2) - sin(lat - dlat/2)
    sin_diff = np.sin(lat_rad + dlat_rad) - np.sin(lat_rad - dlat_rad)

    area = radius**2 * np.abs(sin_diff) * dlon_rad

    return area


def expand_bounds_for_polar(
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    factor: float = 1.414,  # sqrt(2)
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand bounding box for polar projections.

    Polar projections need a larger data extent to fill the circular
    plot area without gaps.

    Parameters
    ----------
    lon_bounds : tuple of float
        Original longitude bounds.
    lat_bounds : tuple of float
        Original latitude bounds.
    factor : float
        Expansion factor. Default is sqrt(2).

    Returns
    -------
    lon_bounds, lat_bounds : tuple of tuples
        Expanded bounds.
    """
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2

    lon_half = (lon_max - lon_min) / 2 * factor
    lat_half = (lat_max - lat_min) / 2 * factor

    new_lon_bounds = (
        max(-180.0, lon_center - lon_half),
        min(180.0, lon_center + lon_half),
    )
    new_lat_bounds = (
        max(-90.0, lat_center - lat_half),
        min(90.0, lat_center + lat_half),
    )

    return new_lon_bounds, new_lat_bounds
