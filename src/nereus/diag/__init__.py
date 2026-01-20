"""Diagnostics module for nereus.

This module provides functions for computing common geophysical diagnostics:

- Sea ice metrics (ice_area, ice_volume, ice_extent)
- Vertical/ocean metrics (volume_mean, heat_content)
- Hovmoller diagrams (hovmoller, plot_hovmoller)
"""

from nereus.diag.hovmoller import hovmoller, plot_hovmoller
from nereus.diag.ice import ice_area, ice_extent, ice_volume
from nereus.diag.vertical import heat_content, volume_mean

__all__ = [
    # Ice diagnostics
    "ice_area",
    "ice_volume",
    "ice_extent",
    # Vertical diagnostics
    "volume_mean",
    "heat_content",
    # Hovmoller
    "hovmoller",
    "plot_hovmoller",
]
