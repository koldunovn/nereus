"""Type aliases and protocols for nereus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr

# Type aliases
ArrayLike = Union[NDArray[np.floating], "xr.DataArray"]
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


class HasCoordinates(Protocol):
    """Protocol for objects with lon/lat coordinates."""

    @property
    def lon(self) -> FloatArray:
        """Longitude array in degrees."""
        ...

    @property
    def lat(self) -> FloatArray:
        """Latitude array in degrees."""
        ...


class HasArea(Protocol):
    """Protocol for objects with cell area."""

    @property
    def area(self) -> FloatArray:
        """Cell area in square meters."""
        ...


class MeshProtocol(HasCoordinates, HasArea, Protocol):
    """Protocol for model mesh objects."""

    pass
