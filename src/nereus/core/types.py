"""Type aliases and protocols for nereus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr

# Type aliases
ArrayLike = Union[NDArray[np.floating], "xr.DataArray"]
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]

# Mesh type literals
MeshType = Literal["fesom", "healpix", "nemo", "lonlat", "custom"]
LonConvention = Literal["pm180", "0360"]


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


def is_dask_array(x) -> bool:
    """Check if array is a dask array.

    Parameters
    ----------
    x : array_like
        Input array (numpy, dask, or xarray).

    Returns
    -------
    bool
        True if the underlying data is a dask array.
    """
    # Check direct dask array
    if hasattr(x, "dask"):
        return True
    # Check xarray with dask backend
    if hasattr(x, "data") and hasattr(x.data, "dask"):
        return True
    return False


def get_array_data(x):
    """Extract underlying array data, preserving dask arrays.

    This function extracts the underlying array from xarray DataArrays
    while preserving dask arrays (no eager computation).

    Parameters
    ----------
    x : array_like
        Input array (numpy, dask, or xarray).

    Returns
    -------
    ndarray or dask.array
        The underlying array data.
    """
    # xarray with dask backend - get the dask array directly
    if hasattr(x, "data") and hasattr(x.data, "dask"):
        return x.data
    # xarray DataArray with numpy backend
    if hasattr(x, "values"):
        return x.values
    return x
