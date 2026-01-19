"""Nereus - Python module for working with geophysical data.

Nereus provides tools for quick data exploration and analysis of
unstructured atmospheric and ocean model data in Jupyter notebooks.

Examples
--------
>>> import nereus as nr

# Plot unstructured data on a map
>>> fig, ax, interp = nr.plot(temperature, mesh.lon, mesh.lat)

# Reuse interpolator for another variable
>>> fig, ax, _ = nr.plot(salinity, mesh.lon, mesh.lat, interpolator=interp)

# Regrid data to regular grid
>>> regridded, interp = nr.regrid(data, lon, lat, resolution=0.5)

# Plot on different projections
>>> fig, ax, interp = nr.plot(data, lon, lat, projection="npstere")
"""

from nereus._version import __version__

# Regridding
from nereus.regrid import RegridInterpolator, regrid, set_cache_options

# Plotting
from nereus.plotting import plot, transect

# Model-specific modules (as namespaces)
from nereus.models import fesom, icono, icona, ifs, healpix

__all__ = [
    # Version
    "__version__",
    # Regridding
    "RegridInterpolator",
    "regrid",
    "set_cache_options",
    # Plotting
    "plot",
    "transect",
    # Model namespaces
    "fesom",
    "icono",
    "icona",
    "ifs",
    "healpix",
]
