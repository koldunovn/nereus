"""Core utilities for nereus."""

from nereus.core.coordinates import (
    EARTH_RADIUS,
    cartesian_to_lonlat,
    chord_to_meters,
    compute_element_centers,
    great_circle_distance,
    great_circle_path,
    lonlat_to_cartesian,
    meters_to_chord,
)
from nereus.core.grids import (
    create_regular_grid,
    expand_bounds_for_polar,
    grid_cell_area,
)
from nereus.core.types import (
    ArrayLike,
    BoolArray,
    FloatArray,
    HasArea,
    HasCoordinates,
    IntArray,
    MeshProtocol,
)

__all__ = [
    "EARTH_RADIUS",
    "ArrayLike",
    "BoolArray",
    "FloatArray",
    "HasArea",
    "HasCoordinates",
    "IntArray",
    "MeshProtocol",
    "cartesian_to_lonlat",
    "chord_to_meters",
    "compute_element_centers",
    "create_regular_grid",
    "expand_bounds_for_polar",
    "great_circle_distance",
    "great_circle_path",
    "grid_cell_area",
    "lonlat_to_cartesian",
    "meters_to_chord",
]
