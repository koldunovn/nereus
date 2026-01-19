"""Projection utilities for nereus plotting.

This module provides convenient aliases and configuration for Cartopy projections.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs

# Projection aliases mapping short names to Cartopy projection classes and defaults
PROJECTION_ALIASES: dict[str, dict[str, Any]] = {
    # PlateCarree
    "pc": {"class": ccrs.PlateCarree, "kwargs": {}},
    "platecarree": {"class": ccrs.PlateCarree, "kwargs": {}},
    # Robinson
    "rob": {"class": ccrs.Robinson, "kwargs": {}, "global": True},
    "robinson": {"class": ccrs.Robinson, "kwargs": {}, "global": True},
    # Mercator
    "merc": {"class": ccrs.Mercator, "kwargs": {}},
    "mercator": {"class": ccrs.Mercator, "kwargs": {}},
    # Mollweide
    "moll": {"class": ccrs.Mollweide, "kwargs": {}, "global": True},
    "mollweide": {"class": ccrs.Mollweide, "kwargs": {}, "global": True},
    # North Polar Stereographic
    "npstere": {
        "class": ccrs.NorthPolarStereo,
        "kwargs": {},
        "polar": "north",
    },
    "northpolarstereo": {
        "class": ccrs.NorthPolarStereo,
        "kwargs": {},
        "polar": "north",
    },
    # South Polar Stereographic
    "spstere": {
        "class": ccrs.SouthPolarStereo,
        "kwargs": {},
        "polar": "south",
    },
    "southpolarstereo": {
        "class": ccrs.SouthPolarStereo,
        "kwargs": {},
        "polar": "south",
    },
    # Orthographic
    "ortho": {"class": ccrs.Orthographic, "kwargs": {"central_longitude": 0, "central_latitude": 0}},
    "orthographic": {"class": ccrs.Orthographic, "kwargs": {"central_longitude": 0, "central_latitude": 0}},
    # Lambert Conformal
    "lcc": {
        "class": ccrs.LambertConformal,
        "kwargs": {"central_longitude": 0, "central_latitude": 45},
    },
    "lambertconformal": {
        "class": ccrs.LambertConformal,
        "kwargs": {"central_longitude": 0, "central_latitude": 45},
    },
}


def get_projection(
    name: str | ccrs.Projection,
    **kwargs: Any,
) -> ccrs.Projection:
    """Get a Cartopy projection from name or alias.

    Parameters
    ----------
    name : str or Projection
        Projection name/alias or an existing Cartopy Projection.
    **kwargs
        Additional keyword arguments passed to the projection constructor.

    Returns
    -------
    Projection
        A Cartopy projection instance.

    Raises
    ------
    ValueError
        If the projection name is not recognized.

    Examples
    --------
    >>> proj = get_projection("pc")
    >>> proj = get_projection("npstere")
    >>> proj = get_projection("ortho", central_longitude=10, central_latitude=50)
    """
    if isinstance(name, ccrs.Projection):
        return name

    name_lower = name.lower()
    if name_lower not in PROJECTION_ALIASES:
        valid = ", ".join(sorted(set(PROJECTION_ALIASES.keys())))
        raise ValueError(
            f"Unknown projection '{name}'. Valid options: {valid}"
        )

    config = PROJECTION_ALIASES[name_lower]
    proj_kwargs = {**config["kwargs"], **kwargs}

    return config["class"](**proj_kwargs)


def is_global_projection(name: str | ccrs.Projection) -> bool:
    """Check if a projection should use set_global().

    Parameters
    ----------
    name : str or Projection
        Projection name/alias or Cartopy Projection.

    Returns
    -------
    bool
        True if the projection is global.
    """
    if isinstance(name, ccrs.Projection):
        # Check by type
        return isinstance(name, (ccrs.Robinson, ccrs.Mollweide))

    name_lower = name.lower()
    if name_lower in PROJECTION_ALIASES:
        return PROJECTION_ALIASES[name_lower].get("global", False)
    return False


def is_polar_projection(name: str | ccrs.Projection) -> str | None:
    """Check if a projection is polar.

    Parameters
    ----------
    name : str or Projection
        Projection name/alias or Cartopy Projection.

    Returns
    -------
    str or None
        "north" for north polar, "south" for south polar, None otherwise.
    """
    if isinstance(name, ccrs.Projection):
        if isinstance(name, ccrs.NorthPolarStereo):
            return "north"
        elif isinstance(name, ccrs.SouthPolarStereo):
            return "south"
        return None

    name_lower = name.lower()
    if name_lower in PROJECTION_ALIASES:
        return PROJECTION_ALIASES[name_lower].get("polar")
    return None


def get_data_bounds_for_projection(
    projection: str | ccrs.Projection,
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get appropriate data bounds for a projection.

    For polar projections, expands the bounds to fill the circular plot area.

    Parameters
    ----------
    projection : str or Projection
        The projection being used.
    extent : tuple of float, optional
        Desired extent (lon_min, lon_max, lat_min, lat_max).

    Returns
    -------
    lon_bounds, lat_bounds : tuple of tuples
        Longitude and latitude bounds for data fetching.
    """
    polar = is_polar_projection(projection)

    if polar == "north":
        # North polar: full longitude, high latitudes
        lon_bounds = (-180.0, 180.0)
        if extent:
            lat_bounds = (max(0.0, extent[2] - 20), 90.0)
        else:
            lat_bounds = (0.0, 90.0)
    elif polar == "south":
        # South polar: full longitude, low latitudes
        lon_bounds = (-180.0, 180.0)
        if extent:
            lat_bounds = (-90.0, min(0.0, extent[3] + 20))
        else:
            lat_bounds = (-90.0, 0.0)
    elif extent:
        # Use provided extent with small buffer
        buffer = 5.0
        lon_bounds = (
            max(-180.0, extent[0] - buffer),
            min(180.0, extent[1] + buffer),
        )
        lat_bounds = (
            max(-90.0, extent[2] - buffer),
            min(90.0, extent[3] + buffer),
        )
    else:
        # Default: global
        lon_bounds = (-180.0, 180.0)
        lat_bounds = (-90.0, 90.0)

    return lon_bounds, lat_bounds
