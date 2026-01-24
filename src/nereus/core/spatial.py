"""Spatial query functions for nereus.

Standalone functions for spatial operations on coordinate arrays.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from nereus.core.coordinates import (
    EARTH_RADIUS,
    chord_to_meters,
    lonlat_to_cartesian,
)

if TYPE_CHECKING:
    pass


def find_nearest(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    query_lon: float | NDArray[np.floating],
    query_lat: float | NDArray[np.floating],
    k: int = 1,
    *,
    return_distance: bool = False,
) -> NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.floating]]:
    """Find nearest mesh points to query locations.

    Uses a KDTree built on Cartesian coordinates for efficient
    spherical nearest-neighbor search.

    Parameters
    ----------
    lon : array_like
        Longitude coordinates of mesh points in degrees.
    lat : array_like
        Latitude coordinates of mesh points in degrees.
    query_lon : float or array_like
        Query longitude(s) in degrees.
    query_lat : float or array_like
        Query latitude(s) in degrees.
    k : int
        Number of nearest neighbors to find.
    return_distance : bool
        If True, also return distances in meters.

    Returns
    -------
    indices : ndarray
        Indices of nearest mesh points. Shape depends on inputs:
        - Scalar query, k=1: scalar int
        - Scalar query, k>1: (k,) array
        - Array query, k=1: (n_queries,) array
        - Array query, k>1: (n_queries, k) array
    distances : ndarray, optional
        Distances in meters. Returned only if return_distance=True.
        Same shape as indices.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh(path)
    >>> idx = nr.find_nearest(mesh["lon"].values, mesh["lat"].values, -30.5, 60.2)
    >>> print(f"Nearest point: ({mesh['lon'].values[idx]}, {mesh['lat'].values[idx]})")

    >>> # Find 3 nearest neighbors with distances
    >>> indices, distances = nr.find_nearest(
    ...     mesh["lon"].values, mesh["lat"].values,
    ...     [-30, -31], [60, 61],
    ...     k=3,
    ...     return_distance=True,
    ... )
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    query_lon = np.atleast_1d(query_lon)
    query_lat = np.atleast_1d(query_lat)

    # Build KDTree on Cartesian coordinates
    xyz = np.column_stack(lonlat_to_cartesian(lon, lat))
    tree = cKDTree(xyz)

    # Query points
    query_xyz = np.column_stack(lonlat_to_cartesian(query_lon, query_lat))
    chord_distances, indices = tree.query(query_xyz, k=k)

    # Squeeze output for scalar queries
    scalar_query = query_lon.shape == (1,)
    if scalar_query and k == 1:
        indices = indices.item()
        chord_distances = chord_distances.item()
    elif scalar_query:
        indices = indices.squeeze()
        chord_distances = chord_distances.squeeze()
    elif k == 1:
        indices = indices.squeeze()
        chord_distances = chord_distances.squeeze()

    if return_distance:
        # Convert chord distance to meters
        if np.isscalar(chord_distances):
            distances = chord_to_meters(chord_distances)
        else:
            distances = np.array([chord_to_meters(d) for d in np.atleast_1d(chord_distances).ravel()])
            distances = distances.reshape(np.atleast_1d(chord_distances).shape)
            if scalar_query and k == 1:
                distances = distances.item()
            elif scalar_query:
                distances = distances.squeeze()
            elif k == 1:
                distances = distances.squeeze()
        return indices, distances

    return indices


def subset_by_bbox(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> NDArray[np.bool_]:
    """Get mask for points within bounding box.

    Parameters
    ----------
    lon : array_like
        Longitude coordinates in degrees.
    lat : array_like
        Latitude coordinates in degrees.
    lon_min, lon_max : float
        Longitude bounds in degrees.
    lat_min, lat_max : float
        Latitude bounds in degrees.

    Returns
    -------
    mask : ndarray
        Boolean mask of points within bounds.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh(path)
    >>> mask = nr.subset_by_bbox(
    ...     mesh["lon"].values, mesh["lat"].values,
    ...     lon_min=-10, lon_max=10,
    ...     lat_min=-5, lat_max=5,
    ... )
    >>> equatorial_area = mesh["area"].values[mask].sum()
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    return lon_mask & lat_mask


def points_in_polygon(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    polygon_lon: NDArray[np.floating],
    polygon_lat: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Get mask for points inside polygon.

    Uses matplotlib.path for point-in-polygon testing.

    Parameters
    ----------
    lon : array_like
        Longitude coordinates of mesh points in degrees.
    lat : array_like
        Latitude coordinates of mesh points in degrees.
    polygon_lon : array_like
        Longitude coordinates of polygon vertices.
    polygon_lat : array_like
        Latitude coordinates of polygon vertices.

    Returns
    -------
    mask : ndarray
        Boolean mask of points inside polygon.

    Notes
    -----
    The polygon is assumed to be defined in Cartesian lon-lat space
    (not geodesic). For large-scale polygons crossing the dateline,
    consider normalizing longitudes first.

    Examples
    --------
    >>> # Select points in a triangular region
    >>> poly_lon = [-10, 10, 0, -10]
    >>> poly_lat = [0, 0, 10, 0]
    >>> mask = nr.points_in_polygon(lon, lat, poly_lon, poly_lat)
    """
    from matplotlib.path import Path

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    polygon_lon = np.asarray(polygon_lon)
    polygon_lat = np.asarray(polygon_lat)

    # Create polygon path
    polygon_vertices = np.column_stack([polygon_lon, polygon_lat])
    polygon_path = Path(polygon_vertices)

    # Test points
    points = np.column_stack([lon.ravel(), lat.ravel()])
    mask = polygon_path.contains_points(points)

    return mask.reshape(lon.shape)


def haversine_distance(
    lon1: float | NDArray[np.floating],
    lat1: float | NDArray[np.floating],
    lon2: float | NDArray[np.floating],
    lat2: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Compute great-circle distance between points using Haversine formula.

    Parameters
    ----------
    lon1, lat1 : float or array_like
        First point(s) coordinates in degrees.
    lon2, lat2 : float or array_like
        Second point(s) coordinates in degrees.

    Returns
    -------
    distance : float or ndarray
        Distance in meters.

    Examples
    --------
    >>> d = nr.haversine_distance(-74.0, 40.7, 2.3, 48.9)  # NYC to Paris
    >>> print(f"Distance: {d/1000:.0f} km")
    """
    lon1 = np.deg2rad(np.asarray(lon1))
    lat1 = np.deg2rad(np.asarray(lat1))
    lon2 = np.deg2rad(np.asarray(lon2))
    lat2 = np.deg2rad(np.asarray(lat2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS * c
