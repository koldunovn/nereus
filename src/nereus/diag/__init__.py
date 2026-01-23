"""Diagnostics module for nereus.

This module provides functions for computing common geophysical diagnostics:

- Sea ice metrics (ice_area, ice_volume, ice_extent) with NH/SH convenience functions
- Vertical/ocean metrics (surface_mean, volume_mean, heat_content)
- Hovmoller diagrams (hovmoller, plot_hovmoller)
- Region masks (get_region_mask, list_available_regions, load_geojson)
"""

from nereus.diag.hovmoller import hovmoller, plot_hovmoller
from nereus.diag.ice import (
    ice_area,
    ice_area_nh,
    ice_area_sh,
    ice_extent,
    ice_extent_nh,
    ice_extent_sh,
    ice_volume,
    ice_volume_nh,
    ice_volume_sh,
)
from nereus.diag.regions import get_region_mask, list_available_regions, load_geojson
from nereus.diag.vertical import heat_content, surface_mean, volume_mean

__all__ = [
    # Ice diagnostics
    "ice_area",
    "ice_area_nh",
    "ice_area_sh",
    "ice_volume",
    "ice_volume_nh",
    "ice_volume_sh",
    "ice_extent",
    "ice_extent_nh",
    "ice_extent_sh",
    # Vertical diagnostics
    "surface_mean",
    "volume_mean",
    "heat_content",
    # Hovmoller
    "hovmoller",
    "plot_hovmoller",
    # Region masks
    "get_region_mask",
    "list_available_regions",
    "load_geojson",
]
