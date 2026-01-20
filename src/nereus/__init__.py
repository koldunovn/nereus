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

# Compute sea ice diagnostics
>>> nh_ice_area = nr.ice_area(sic, mesh.area, mask=mesh.lat > 0)
>>> ice_vol = nr.ice_volume(sit, mesh.area, concentration=sic)

# Ocean diagnostics
>>> ohc = nr.heat_content(temp, mesh.area, mesh.layer_thickness)
>>> mean_temp = nr.volume_mean(temp, mesh.area, mesh.layer_thickness, depth_max=500)
"""

from nereus._version import __version__

# Regridding
from nereus.regrid import RegridInterpolator, regrid, set_cache_options

# Plotting
from nereus.plotting import plot, transect

# Diagnostics (exported at top level)
from nereus.diag import (
    get_region_mask,
    heat_content,
    hovmoller,
    ice_area,
    ice_extent,
    ice_volume,
    list_available_regions,
    load_geojson,
    plot_hovmoller,
    volume_mean,
)

# Model-specific modules (as namespaces)
from nereus.models import fesom, healpix, icona, icono, ifs

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
    # Diagnostics
    "ice_area",
    "ice_volume",
    "ice_extent",
    "volume_mean",
    "heat_content",
    "hovmoller",
    "plot_hovmoller",
    # Region masks
    "get_region_mask",
    "list_available_regions",
    "load_geojson",
    # Model namespaces
    "fesom",
    "icono",
    "icona",
    "ifs",
    "healpix",
]
