"""HEALPix grid support for nereus.

This module provides functionality for working with HEALPix grids,
commonly used in climate models like ICON and IFS.

Examples
--------
>>> import nereus as nr

# Create HEALPix mesh from number of points
>>> mesh = nr.healpix.load_mesh(3145728)  # nside=512

# Or use nside directly
>>> npoints = nr.healpix.nside_to_npoints(512)
>>> mesh = nr.healpix.load_mesh(npoints)

# Access mesh data
>>> lon = mesh["lon"].values
>>> area = mesh["area"]  # Uniform for all pixels
"""

from nereus.models.healpix.mesh import (
    load_mesh,
    npoints_to_nside,
    nside_to_npoints,
    resolution_to_nside,
)

__all__ = [
    "load_mesh",
    "npoints_to_nside",
    "nside_to_npoints",
    "resolution_to_nside",
]
