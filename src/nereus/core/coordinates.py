"""Coordinate utilities for nereus.

Functions for converting between geographic and Cartesian coordinates,
and for computing distances on the sphere.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Earth's radius in meters (WGS84 mean radius)
EARTH_RADIUS = 6_371_000.0


def lonlat_to_cartesian(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    radius: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert longitude/latitude to Cartesian coordinates.

    Parameters
    ----------
    lon : array_like
        Longitude in degrees.
    lat : array_like
        Latitude in degrees.
    radius : float, optional
        Radius of the sphere. Default is 1.0 (unit sphere).

    Returns
    -------
    x, y, z : tuple of ndarrays
        Cartesian coordinates.

    Notes
    -----
    Uses the convention where:
    - x points towards (lon=0, lat=0)
    - y points towards (lon=90, lat=0)
    - z points towards the North Pole (lat=90)
    """
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    cos_lat = np.cos(lat_rad)
    x = radius * cos_lat * np.cos(lon_rad)
    y = radius * cos_lat * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return x, y, z


def cartesian_to_lonlat(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert Cartesian coordinates to longitude/latitude.

    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates.

    Returns
    -------
    lon, lat : tuple of ndarrays
        Longitude and latitude in degrees.
    """
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))

    return lon, lat


def meters_to_chord(meters: float, radius: float = EARTH_RADIUS) -> float:
    """Convert distance in meters to chord distance on unit sphere.

    The chord distance is the straight-line distance through the sphere
    between two points, normalized by the sphere radius. This is useful
    for KDTree queries on unit sphere coordinates.

    Parameters
    ----------
    meters : float
        Distance in meters along the surface of the Earth.
    radius : float, optional
        Earth radius in meters. Default is 6,371,000 m.

    Returns
    -------
    float
        Chord distance on unit sphere.

    Notes
    -----
    The formula converts arc length to chord length:
    chord = 2 * sin(arc_length / (2 * radius))

    For small distances, chord ≈ arc_length / radius.
    """
    # Arc length in radians
    arc_radians = meters / radius
    # Chord distance (normalized to unit sphere)
    chord = 2.0 * np.sin(arc_radians / 2.0)
    return float(chord)


def chord_to_meters(chord: float, radius: float = EARTH_RADIUS) -> float:
    """Convert chord distance on unit sphere to meters.

    Parameters
    ----------
    chord : float
        Chord distance on unit sphere.
    radius : float, optional
        Earth radius in meters. Default is 6,371,000 m.

    Returns
    -------
    float
        Distance in meters along the surface of the Earth.
    """
    # Chord to arc length in radians
    arc_radians = 2.0 * np.arcsin(chord / 2.0)
    # Arc length in meters
    meters = arc_radians * radius
    return float(meters)


def great_circle_distance(
    lon1: NDArray[np.floating],
    lat1: NDArray[np.floating],
    lon2: NDArray[np.floating],
    lat2: NDArray[np.floating],
    radius: float = EARTH_RADIUS,
) -> NDArray[np.floating]:
    """Compute great-circle distance between two points using Haversine formula.

    Parameters
    ----------
    lon1, lat1 : array_like
        Longitude and latitude of first point(s) in degrees.
    lon2, lat2 : array_like
        Longitude and latitude of second point(s) in degrees.
    radius : float, optional
        Earth radius in meters. Default is 6,371,000 m.

    Returns
    -------
    ndarray
        Distance in meters.
    """
    lon1_rad = np.deg2rad(lon1)
    lat1_rad = np.deg2rad(lat1)
    lon2_rad = np.deg2rad(lon2)
    lat2_rad = np.deg2rad(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c


def great_circle_path(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    n_points: int = 100,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate points along a great circle path.

    Parameters
    ----------
    start_lon, start_lat : float
        Start point in degrees.
    end_lon, end_lat : float
        End point in degrees.
    n_points : int, optional
        Number of points along the path. Default is 100.

    Returns
    -------
    lon, lat : tuple of ndarrays
        Coordinates of points along the great circle path.
    """
    # Convert to Cartesian
    x1, y1, z1 = lonlat_to_cartesian(np.array([start_lon]), np.array([start_lat]))
    x2, y2, z2 = lonlat_to_cartesian(np.array([end_lon]), np.array([end_lat]))

    # Interpolate in Cartesian space
    t = np.linspace(0, 1, n_points)
    x = x1[0] * (1 - t) + x2[0] * t
    y = y1[0] * (1 - t) + y2[0] * t
    z = z1[0] * (1 - t) + z2[0] * t

    # Normalize to unit sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    x = x / norm
    y = y / norm
    z = z / norm

    # Convert back to lon/lat
    lon, lat = cartesian_to_lonlat(x, y, z)

    return lon, lat


def compute_element_centers(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    triangles: NDArray[np.integer],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute element center coordinates, handling cyclic triangles.

    For triangular meshes, computes the center of each triangle. Handles
    triangles that cross the dateline (±180°) by shifting longitudes
    before averaging.

    Parameters
    ----------
    lon : array_like
        Longitude coordinates of mesh nodes in degrees.
    lat : array_like
        Latitude coordinates of mesh nodes in degrees.
    triangles : array_like
        Triangle connectivity array of shape (3, nelem) or (nelem, 3).
        Contains indices into lon/lat arrays.

    Returns
    -------
    lon_elem, lat_elem : tuple of ndarrays
        Longitude and latitude of element centers.

    Examples
    --------
    >>> lon_elem, lat_elem = compute_element_centers(mesh.lon, mesh.lat, mesh.face_nodes)
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    triangles = np.asarray(triangles)

    # Ensure triangles has shape (nelem, 3)
    if triangles.shape[0] == 3 and triangles.shape[1] != 3:
        triangles = triangles.T

    # Get vertex coordinates for each triangle
    tri_lon = lon[triangles]  # (nelem, 3)
    tri_lat = lat[triangles]  # (nelem, 3)

    # Simple mean for latitude (no cyclic issues)
    lat_elem = tri_lat.mean(axis=1)

    # For longitude, need to handle triangles crossing the dateline
    # First compute simple mean
    lon_mean = tri_lon.mean(axis=1)

    # Find cyclic triangles: where any vertex is far from the mean (>100°)
    max_diff = np.abs(tri_lon - lon_mean[:, np.newaxis]).max(axis=1)
    cyclic_mask = max_diff > 100

    if np.any(cyclic_mask):
        # For cyclic triangles, shift negative longitudes by +360 before averaging
        cyclic_lon = tri_lon[cyclic_mask].copy()
        cyclic_lon_shifted = np.where(cyclic_lon < 0, cyclic_lon + 360, cyclic_lon)
        new_means = cyclic_lon_shifted.mean(axis=1)
        # Shift back to [-180, 180] range
        new_means = np.where(new_means > 180, new_means - 360, new_means)
        lon_mean[cyclic_mask] = new_means

    return lon_mean, lat_elem
