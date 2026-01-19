"""Plotting module for nereus."""

from nereus.plotting.maps import plot, plot_polar
from nereus.plotting.projections import (
    PROJECTION_ALIASES,
    get_data_bounds_for_projection,
    get_projection,
    is_global_projection,
    is_polar_projection,
)
from nereus.plotting.transect import transect

__all__ = [
    "PROJECTION_ALIASES",
    "get_data_bounds_for_projection",
    "get_projection",
    "is_global_projection",
    "is_polar_projection",
    "plot",
    "plot_polar",
    "transect",
]
