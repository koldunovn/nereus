"""HEALPix grid mesh generation.

This module provides functionality for creating HEALPix grid meshes
as xr.Dataset objects using the healpy package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from nereus.core.coordinates import EARTH_RADIUS
from nereus.core.mesh import (
    add_mesh_metadata,
    normalize_lon,
    should_use_dask,
)

if TYPE_CHECKING:
    pass


def load_mesh(
    npoints: int,
    *,
    nest: bool = True,
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Create HEALPix mesh from number of points.

    Infers nside from npoints (npoints = 12 * nside^2).

    Parameters
    ----------
    npoints : int
        Number of HEALPix pixels. Must be 12 * nside^2 for valid nside.
    nest : bool
        Use NESTED ordering (True) or RING ordering (False).
        Default is NESTED (True).
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.

    Returns
    -------
    xr.Dataset
        Mesh dataset with:
        - lon: Pixel center longitudes (npoints,)
        - lat: Pixel center latitudes (npoints,)
        - area: Pixel area in m^2 (uniform for HEALPix) (npoints,)

    Examples
    --------
    >>> # Create HEALPix mesh with ~3 million pixels (nside=512)
    >>> mesh = nr.healpix.load_mesh(3145728)
    >>> print(f"Pixel area: {mesh['area'].values[0] / 1e6:.1f} km^2")

    >>> # Create smaller mesh for testing
    >>> mesh = nr.healpix.load_mesh(12 * 64**2)  # nside=64
    """
    try:
        import healpy as hp
    except ImportError as e:
        raise ImportError(
            "healpy is required for HEALPix mesh support. "
            "Install with: pip install healpy"
        ) from e

    # Infer nside from npoints
    nside = hp.npix2nside(npoints)

    # Verify
    expected_npix = 12 * nside**2
    if npoints != expected_npix:
        raise ValueError(
            f"npoints={npoints} is not valid. "
            f"For nside={nside}, expected {expected_npix} points."
        )

    use_dask_actual = should_use_dask(npoints, use_dask)

    # Get pixel center coordinates
    # healpy.pix2ang returns (theta, phi) in radians by default
    # with lonlat=True, returns (lon, lat) in degrees
    lon, lat = hp.pix2ang(nside, np.arange(npoints), nest=nest, lonlat=True)

    # Normalize longitude to [-180, 180]
    lon = normalize_lon(lon, "pm180")

    # Compute pixel area (uniform for all pixels)
    # hp.nside2pixarea returns area in steradians
    pixel_area_sr = hp.nside2pixarea(nside)
    pixel_area_m2 = pixel_area_sr * EARTH_RADIUS**2

    area = np.full(npoints, pixel_area_m2, dtype=np.float64)

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
                "long_name": "Pixel area",
            }),
        },
        attrs={
            "nside": nside,
            "nest": nest,
            "ordering": "NESTED" if nest else "RING",
        },
    )

    return add_mesh_metadata(ds, "healpix", use_dask=use_dask_actual)


def nside_to_npoints(nside: int) -> int:
    """Convert HEALPix nside parameter to number of points.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (must be power of 2).

    Returns
    -------
    int
        Number of pixels (12 * nside^2).

    Examples
    --------
    >>> nr.healpix.nside_to_npoints(512)
    3145728
    """
    return 12 * nside**2


def npoints_to_nside(npoints: int) -> int:
    """Convert number of points to HEALPix nside parameter.

    Parameters
    ----------
    npoints : int
        Number of HEALPix pixels.

    Returns
    -------
    int
        HEALPix nside parameter.

    Raises
    ------
    ValueError
        If npoints is not a valid HEALPix pixel count.

    Examples
    --------
    >>> nr.healpix.npoints_to_nside(3145728)
    512
    """
    try:
        import healpy as hp
    except ImportError as e:
        raise ImportError(
            "healpy is required for HEALPix mesh support. "
            "Install with: pip install healpy"
        ) from e

    return hp.npix2nside(npoints)


def resolution_to_nside(resolution_deg: float) -> int:
    """Get approximate nside for desired angular resolution.

    Parameters
    ----------
    resolution_deg : float
        Desired angular resolution in degrees.

    Returns
    -------
    int
        Recommended nside (power of 2).

    Examples
    --------
    >>> # Get nside for ~1 degree resolution
    >>> nside = nr.healpix.resolution_to_nside(1.0)
    >>> print(nside)
    64
    """
    try:
        import healpy as hp
    except ImportError as e:
        raise ImportError(
            "healpy is required for HEALPix mesh support. "
            "Install with: pip install healpy"
        ) from e

    # Resolution in arcmin
    resolution_arcmin = resolution_deg * 60
    # Get nside from resolution
    nside = hp.resolution_to_nside(resolution_arcmin, arcmin=True)
    return int(nside)
