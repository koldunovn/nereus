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


def wrap_as_xarray(
    result: "NDArray | float",
    source_data: "NDArray | xr.DataArray",
    default_name: str,
    *,
    skip_dims: int = 1,
) -> "xr.DataArray":
    """Wrap a reduced result as an xarray DataArray.

    After a spatial reduction (e.g. nansum over trailing axes), the result
    has a subset of the input's dimensions.  This helper preserves dimension
    names, coordinates, variable name, and attributes when the input is an
    xarray DataArray.

    Parameters
    ----------
    result : float or ndarray
        The reduced result (scalar or array with leading dims).
    source_data : array_like
        The original input data (used to extract dimension metadata).
    default_name : str
        Variable name to use when the input has no name (e.g. numpy).
    skip_dims : int
        Number of *trailing* dimensions of ``source_data`` that were
        consumed by the reduction and should not appear in the output.
        Default 1 (spatial-only reduction like ice_area).  Use 2 for
        reductions over (nlevels, npoints) like volume_mean.
    """
    import xarray as xr

    result_arr = np.atleast_1d(np.asarray(result))

    if hasattr(source_data, "dims"):
        # Input is xarray – preserve leading dimension info
        n_leading = result_arr.ndim
        leading_dim_names = list(source_data.dims[:n_leading])
        leading_coords = {
            d: source_data.coords[d]
            for d in leading_dim_names
            if d in source_data.coords
        }
        var_name = source_data.name or default_name
        var_attrs = dict(source_data.attrs)
    else:
        n_leading = result_arr.ndim
        leading_dim_names = [f"dim_{i}" for i in range(n_leading)]
        leading_coords = {}
        var_name = default_name
        var_attrs = {}

    return xr.DataArray(
        result_arr.squeeze() if result_arr.size == 1 else result_arr,
        dims=leading_dim_names if result_arr.size > 1 else None,
        coords=leading_coords if result_arr.size > 1 else None,
        name=var_name,
        attrs=var_attrs,
    )


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
