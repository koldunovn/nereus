"""Sea ice diagnostics for nereus.

This module provides functions for computing sea ice metrics:
- ice_area: Total sea ice area
- ice_volume: Total sea ice volume
- ice_extent: Sea ice extent (area with concentration above threshold)

Hemisphere convenience functions are also provided:
- ice_area_nh, ice_area_sh: Northern/Southern Hemisphere ice area
- ice_volume_nh, ice_volume_sh: Northern/Southern Hemisphere ice volume
- ice_extent_nh, ice_extent_sh: Northern/Southern Hemisphere ice extent

All functions are dask-friendly: if inputs are dask arrays, the result
will be a lazy dask array that can be computed later with ``.compute()``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nereus.core.types import get_array_data, is_dask_array

if TYPE_CHECKING:
    import xarray as xr


def _wrap_as_xarray(
    result: float | NDArray,
    source_data: NDArray | "xr.DataArray",
    default_name: str,
) -> "xr.DataArray":
    """Wrap a reduced result as an xarray DataArray.

    After a spatial reduction (nansum over last axis), the result has
    the leading dimensions of the input.  This helper preserves those
    dimension names, coordinates, variable name, and attributes when
    the input is an xarray DataArray.

    Parameters
    ----------
    result : float or ndarray
        The reduced result (scalar or array with leading dims).
    source_data : array_like
        The original input data (used to extract dimension metadata).
    default_name : str
        Variable name to use when the input has no name (e.g. numpy).
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


def ice_area(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    *,
    mask: NDArray[np.bool_] | None = None,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute total sea ice area.

    Sea ice area is the sum of grid cell areas weighted by ice concentration.

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    concentration : array_like
        Sea ice concentration (fraction, 0-1). Can be 1D (npoints,) or
        ND with the last axis being npoints.
    area : array_like
        Grid cell areas in m^2.
    mask : array_like, optional
        Boolean mask (True = include). If None, all points are included.
    as_xarray : bool
        If True, return the result as an xarray DataArray with dimension
        names and coordinates preserved from the input (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Total sea ice area in m^2. Returns float for 1D numpy input,
        ndarray for ND numpy input, dask array if inputs are dask,
        or xr.DataArray if as_xarray=True.

    Examples
    --------
    >>> # 1D concentration array
    >>> total_area = nr.ice_area(sic, mesh.area)

    >>> # With time dimension (time, npoints)
    >>> area_timeseries = nr.ice_area(sic, mesh.area)

    >>> # With hemisphere mask
    >>> nh_area = nr.ice_area(sic, mesh.area, mask=mesh.lat > 0)

    >>> # With dask arrays (lazy computation)
    >>> area = nr.ice_area(sic_dask, mesh.area)
    >>> area.compute()  # triggers actual computation
    """
    # Extract arrays, preserving dask
    conc = get_array_data(concentration)
    area_arr = get_array_data(area)
    is_lazy = is_dask_array(concentration)

    # Warn if dask data is mixed with large numpy arrays (causes graph bloat)
    if is_lazy and not is_dask_array(area_arr) and area_arr.nbytes > 10_000_000:
        warnings.warn(
            f"Data is a dask array but area ({area_arr.nbytes / 1e6:.1f} MB) is numpy. "
            "This can cause very large dask graphs. Consider loading all "
            "large arrays with dask (e.g., xr.open_dataset(..., chunks='auto')).",
            UserWarning,
            stacklevel=2,
        )

    # Flatten area - ravel() works for both numpy and dask
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Apply mask by setting area to NaN where mask is False
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        area_arr = np.where(mask_arr, area_arr, np.nan)

    # Clip concentration to valid range (NaN values stay NaN)
    conc = np.clip(conc, 0.0, 1.0)

    # Compute ice area using nansum (NaN in conc or area automatically excluded)
    result = np.nansum(conc * area_arr, axis=-1)

    # Return appropriate type
    if as_xarray:
        return _wrap_as_xarray(result, concentration, "ice_area")
    elif is_lazy:
        return result  # Keep lazy, user calls .compute()
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result


def ice_volume(
    thickness: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    concentration: NDArray | "xr.DataArray" | None = None,
    *,
    mask: NDArray[np.bool_] | None = None,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute total sea ice volume.

    This function handles two types of thickness definitions:

    **Effective thickness** (default, when concentration=None):
        Thickness averaged over the entire grid cell, including open water.
        This is common in model output (e.g., CICE volume-per-area variables).
        Formula: V = sum(h_eff * cell_area)

    **Real thickness** (when concentration is provided):
        Thickness averaged only over ice-covered area (e.g., CMIP6 sithick).
        Formula: V = sum(h_ice * concentration * cell_area)

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    thickness : array_like
        Sea ice thickness in meters. Can be 1D (npoints,) or ND with the
        last axis being npoints.

        - If concentration is None: interpreted as effective thickness
          (grid-cell mean, already includes open water as zero).
        - If concentration is provided: interpreted as real thickness
          (ice-area mean, the physical thickness of the ice itself).

    area : array_like
        Grid cell areas in m^2.
    concentration : array_like, optional
        Sea ice concentration (fraction, 0-1). Provide this when thickness
        is "real thickness" (ice-area mean, like CMIP6 sithick). Leave as
        None when thickness is "effective thickness" (grid-cell mean).
    mask : array_like, optional
        Boolean mask (True = include).
    as_xarray : bool
        If True, return the result as an xarray DataArray with dimension
        names and coordinates preserved from the input (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Total sea ice volume in m^3. Returns a dask array if inputs are
        dask arrays (call ``.compute()`` to get the result), or
        xr.DataArray if as_xarray=True.

    Examples
    --------
    >>> # Effective thickness (e.g., model output with volume-per-area)
    >>> volume = nr.ice_volume(h_eff, mesh.area)

    >>> # Real thickness (e.g., CMIP6 sithick) - must provide concentration
    >>> volume = nr.ice_volume(sithick, mesh.area, concentration=siconc)

    >>> # With dask arrays (lazy computation)
    >>> volume = nr.ice_volume(h_eff_dask, mesh.area)
    >>> volume.compute()  # triggers actual computation

    Notes
    -----
    Common pitfall: Using real thickness (sithick) without concentration
    will overestimate volume by a factor of 1/concentration, because it
    assumes ice covers the entire grid cell.
    """
    # Extract arrays, preserving dask
    thick = get_array_data(thickness)
    area_arr = get_array_data(area)
    is_lazy = is_dask_array(thickness)

    # Warn if dask data is mixed with large numpy arrays (causes graph bloat)
    if is_lazy and not is_dask_array(area_arr) and area_arr.nbytes > 10_000_000:
        warnings.warn(
            f"Data is a dask array but area ({area_arr.nbytes / 1e6:.1f} MB) is numpy. "
            "This can cause very large dask graphs. Consider loading all "
            "large arrays with dask (e.g., xr.open_dataset(..., chunks='auto')).",
            UserWarning,
            stacklevel=2,
        )

    # Flatten area - ravel() works for both numpy and dask
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Apply mask by setting area to NaN where mask is False
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        area_arr = np.where(mask_arr, area_arr, np.nan)

    # Ensure no negative thickness (NaN values stay NaN)
    thick = np.maximum(thick, 0.0)

    # Compute volume based on thickness type using nansum
    # NaN in thick, conc, or area automatically excluded
    if concentration is not None:
        # Real thickness (ice-area mean): V = h_ice * a * A_cell
        conc = get_array_data(concentration)
        conc = np.clip(conc, 0.0, 1.0)
        result = np.nansum(thick * conc * area_arr, axis=-1)
    else:
        # Effective thickness (grid-cell mean): V = h_eff * A_cell
        result = np.nansum(thick * area_arr, axis=-1)

    # Return appropriate type
    if as_xarray:
        return _wrap_as_xarray(result, thickness, "ice_volume")
    elif is_lazy:
        return result  # Keep lazy, user calls .compute()
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result


def ice_area_nh(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute Northern Hemisphere sea ice area.

    Convenience function that calls ice_area with a Northern Hemisphere mask.

    Parameters
    ----------
    concentration : array_like
        Sea ice concentration (fraction, 0-1).
    area : array_like
        Grid cell areas in m^2.
    lat : array_like
        Latitude of grid points in degrees.
    as_xarray : bool
        If True, return the result as an xarray DataArray (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Northern Hemisphere sea ice area in m^2.

    Examples
    --------
    >>> nh_area = nr.ice_area_nh(sic, mesh.area, mesh.lat)
    """
    lat_arr = get_array_data(lat)
    if hasattr(lat_arr, "ravel"):
        lat_arr = lat_arr.ravel()
    else:
        lat_arr = np.asarray(lat_arr).ravel()
    return ice_area(concentration, area, mask=lat_arr > 0, as_xarray=as_xarray)


def ice_area_sh(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute Southern Hemisphere sea ice area.

    Convenience function that calls ice_area with a Southern Hemisphere mask.

    Parameters
    ----------
    concentration : array_like
        Sea ice concentration (fraction, 0-1).
    area : array_like
        Grid cell areas in m^2.
    lat : array_like
        Latitude of grid points in degrees.
    as_xarray : bool
        If True, return the result as an xarray DataArray (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Southern Hemisphere sea ice area in m^2.

    Examples
    --------
    >>> sh_area = nr.ice_area_sh(sic, mesh.area, mesh.lat)
    """
    lat_arr = get_array_data(lat)
    if hasattr(lat_arr, "ravel"):
        lat_arr = lat_arr.ravel()
    else:
        lat_arr = np.asarray(lat_arr).ravel()
    return ice_area(concentration, area, mask=lat_arr < 0, as_xarray=as_xarray)


def ice_extent(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    *,
    threshold: float = 0.15,
    mask: NDArray[np.bool_] | None = None,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute sea ice extent.

    Sea ice extent is the total area of grid cells where ice concentration
    exceeds a threshold (typically 15%).

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    concentration : array_like
        Sea ice concentration (fraction, 0-1). Can be 1D (npoints,) or
        ND with the last axis being npoints.
    area : array_like
        Grid cell areas in m^2.
    threshold : float
        Concentration threshold (default 0.15 = 15%).
    mask : array_like, optional
        Boolean mask (True = include).
    as_xarray : bool
        If True, return the result as an xarray DataArray with dimension
        names and coordinates preserved from the input (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Total sea ice extent in m^2. Returns a dask array if inputs are
        dask arrays (call ``.compute()`` to get the result), or
        xr.DataArray if as_xarray=True.

    Examples
    --------
    >>> extent = nr.ice_extent(sic, mesh.area)
    >>> extent_nh = nr.ice_extent(sic, mesh.area, mask=mesh.lat > 0)

    >>> # With dask arrays (lazy computation)
    >>> extent = nr.ice_extent(sic_dask, mesh.area)
    >>> extent.compute()  # triggers actual computation
    """
    # Extract arrays, preserving dask
    conc = get_array_data(concentration)
    area_arr = get_array_data(area)
    is_lazy = is_dask_array(concentration)

    # Warn if dask data is mixed with large numpy arrays (causes graph bloat)
    if is_lazy and not is_dask_array(area_arr) and area_arr.nbytes > 10_000_000:
        warnings.warn(
            f"Data is a dask array but area ({area_arr.nbytes / 1e6:.1f} MB) is numpy. "
            "This can cause very large dask graphs. Consider loading all "
            "large arrays with dask (e.g., xr.open_dataset(..., chunks='auto')).",
            UserWarning,
            stacklevel=2,
        )

    # Flatten area - ravel() works for both numpy and dask
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Apply mask by setting area to NaN where mask is False
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        area_arr = np.where(mask_arr, area_arr, np.nan)

    # Compute extent: sum(cell_area) where concentration >= threshold
    # NaN concentration treated as not meeting threshold (comparison with NaN is False)
    ice_mask = conc >= threshold
    # Use nansum to handle NaN in area (from mask)
    result = np.nansum(area_arr * ice_mask, axis=-1)

    # Return appropriate type
    if as_xarray:
        return _wrap_as_xarray(result, concentration, "ice_extent")
    elif is_lazy:
        return result  # Keep lazy, user calls .compute()
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result


def ice_volume_nh(
    thickness: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    lat: NDArray[np.floating],
    concentration: NDArray | "xr.DataArray" | None = None,
    *,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute Northern Hemisphere sea ice volume.

    Convenience function that calls ice_volume with a Northern Hemisphere mask.

    Parameters
    ----------
    thickness : array_like
        Sea ice thickness in meters.
    area : array_like
        Grid cell areas in m^2.
    lat : array_like
        Latitude of grid points in degrees.
    concentration : array_like, optional
        Sea ice concentration (fraction, 0-1). Required if thickness
        is "real thickness" (ice-area mean).
    as_xarray : bool
        If True, return the result as an xarray DataArray (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Northern Hemisphere sea ice volume in m^3.

    Examples
    --------
    >>> nh_volume = nr.ice_volume_nh(sit, mesh.area, mesh.lat)
    """
    lat_arr = get_array_data(lat)
    if hasattr(lat_arr, "ravel"):
        lat_arr = lat_arr.ravel()
    else:
        lat_arr = np.asarray(lat_arr).ravel()
    return ice_volume(
        thickness, area, concentration, mask=lat_arr > 0, as_xarray=as_xarray
    )


def ice_volume_sh(
    thickness: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    lat: NDArray[np.floating],
    concentration: NDArray | "xr.DataArray" | None = None,
    *,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute Southern Hemisphere sea ice volume.

    Convenience function that calls ice_volume with a Southern Hemisphere mask.

    Parameters
    ----------
    thickness : array_like
        Sea ice thickness in meters.
    area : array_like
        Grid cell areas in m^2.
    lat : array_like
        Latitude of grid points in degrees.
    concentration : array_like, optional
        Sea ice concentration (fraction, 0-1). Required if thickness
        is "real thickness" (ice-area mean).
    as_xarray : bool
        If True, return the result as an xarray DataArray (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Southern Hemisphere sea ice volume in m^3.

    Examples
    --------
    >>> sh_volume = nr.ice_volume_sh(sit, mesh.area, mesh.lat)
    """
    lat_arr = get_array_data(lat)
    if hasattr(lat_arr, "ravel"):
        lat_arr = lat_arr.ravel()
    else:
        lat_arr = np.asarray(lat_arr).ravel()
    return ice_volume(
        thickness, area, concentration, mask=lat_arr < 0, as_xarray=as_xarray
    )


def ice_extent_nh(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    threshold: float = 0.15,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute Northern Hemisphere sea ice extent.

    Convenience function that calls ice_extent with a Northern Hemisphere mask.

    Parameters
    ----------
    concentration : array_like
        Sea ice concentration (fraction, 0-1).
    area : array_like
        Grid cell areas in m^2.
    lat : array_like
        Latitude of grid points in degrees.
    threshold : float
        Concentration threshold (default 0.15 = 15%).
    as_xarray : bool
        If True, return the result as an xarray DataArray (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Northern Hemisphere sea ice extent in m^2.

    Examples
    --------
    >>> nh_extent = nr.ice_extent_nh(sic, mesh.area, mesh.lat)
    """
    lat_arr = get_array_data(lat)
    if hasattr(lat_arr, "ravel"):
        lat_arr = lat_arr.ravel()
    else:
        lat_arr = np.asarray(lat_arr).ravel()
    return ice_extent(
        concentration, area, threshold=threshold, mask=lat_arr > 0,
        as_xarray=as_xarray,
    )


def ice_extent_sh(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    threshold: float = 0.15,
    as_xarray: bool = False,
) -> float | NDArray | "xr.DataArray":
    """Compute Southern Hemisphere sea ice extent.

    Convenience function that calls ice_extent with a Southern Hemisphere mask.

    Parameters
    ----------
    concentration : array_like
        Sea ice concentration (fraction, 0-1).
    area : array_like
        Grid cell areas in m^2.
    lat : array_like
        Latitude of grid points in degrees.
    threshold : float
        Concentration threshold (default 0.15 = 15%).
    as_xarray : bool
        If True, return the result as an xarray DataArray (default False).

    Returns
    -------
    float or ndarray or dask.array or xr.DataArray
        Southern Hemisphere sea ice extent in m^2.

    Examples
    --------
    >>> sh_extent = nr.ice_extent_sh(sic, mesh.area, mesh.lat)
    """
    lat_arr = get_array_data(lat)
    if hasattr(lat_arr, "ravel"):
        lat_arr = lat_arr.ravel()
    else:
        lat_arr = np.asarray(lat_arr).ravel()
    return ice_extent(
        concentration, area, threshold=threshold, mask=lat_arr < 0,
        as_xarray=as_xarray,
    )