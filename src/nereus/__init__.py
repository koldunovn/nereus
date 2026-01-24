"""Nereus - Python module for working with geophysical data.

Nereus provides tools for quick data exploration and analysis of
unstructured atmospheric and ocean model data in Jupyter notebooks.

Examples
--------
>>> import nereus as nr

# Load mesh (model-specific)
>>> mesh = nr.fesom.load_mesh("/path/to/mesh")  # Returns xr.Dataset
>>> mesh = nr.healpix.load_mesh(3145728)
>>> mesh = nr.nemo.load_mesh("/path/to/mesh_mask.nc")

# Universal loader with auto-detection
>>> mesh = nr.load_mesh("/path/to/mesh")

# Access mesh data
>>> lon = mesh["lon"]  # xr.DataArray
>>> area = mesh["area"].values  # numpy array

# Plot unstructured data on a map
>>> fig, ax, interp = nr.plot(temperature, mesh["lon"].values, mesh["lat"].values)

# Reuse interpolator for another variable
>>> fig, ax, _ = nr.plot(salinity, mesh["lon"].values, mesh["lat"].values, interpolator=interp)

# Regrid data to regular grid
>>> regridded, interp = nr.regrid(data, lon, lat, resolution=0.5)

# Plot on different projections
>>> fig, ax, interp = nr.plot(data, lon, lat, projection="npstere")

# Compute sea ice diagnostics
>>> nh_ice_area = nr.ice_area(sic, mesh["area"], mask=mesh["lat"] > 0)
>>> sh_ice_extent = nr.ice_extent(sic, mesh["area"], mask=mesh["lat"] < 0)
>>> ice_vol = nr.ice_volume(sit, mesh["area"], concentration=sic)

# Spatial queries
>>> idx = nr.find_nearest(mesh["lon"].values, mesh["lat"].values, -30.5, 60.2)
>>> mask = nr.subset_by_bbox(mesh["lon"].values, mesh["lat"].values, -10, 10, -5, 5)

# Ocean diagnostics
>>> mean_sst = nr.surface_mean(sst, mesh["area"])
>>> ohc = nr.heat_content(temp, mesh["area"], mesh["layer_thickness"])
>>> mean_temp = nr.volume_mean(temp, mesh["area"], mesh["layer_thickness"], depth_max=500)
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
    ice_area_nh,
    ice_area_sh,
    ice_extent,
    ice_extent_nh,
    ice_extent_sh,
    ice_volume,
    ice_volume_nh,
    ice_volume_sh,
    list_available_regions,
    load_geojson,
    plot_hovmoller,
    surface_mean,
    volume_mean,
)

# Core utilities (exported at top level)
from nereus.core.mesh import (
    create_lonlat_mesh,
    mesh_from_arrays,
)
from nereus.core.spatial import (
    find_nearest,
    haversine_distance,
    points_in_polygon,
    subset_by_bbox,
)

# Model-specific modules (as namespaces)
from nereus.models import fesom, healpix, icona, icono, ifs, nemo

# Universal mesh loader
from nereus.models import load_mesh

__all__ = [
    # Version
    "__version__",
    # Mesh loading
    "load_mesh",
    "create_lonlat_mesh",
    "mesh_from_arrays",
    # Spatial queries
    "find_nearest",
    "subset_by_bbox",
    "points_in_polygon",
    "haversine_distance",
    # Regridding
    "RegridInterpolator",
    "regrid",
    "set_cache_options",
    # Plotting
    "plot",
    "transect",
    # Diagnostics - Ice
    "ice_area",
    "ice_area_nh",
    "ice_area_sh",
    "ice_volume",
    "ice_volume_nh",
    "ice_volume_sh",
    "ice_extent",
    "ice_extent_nh",
    "ice_extent_sh",
    # Diagnostics - Vertical
    "surface_mean",
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
    "healpix",
    "nemo",
    "icono",
    "icona",
    "ifs",
]
