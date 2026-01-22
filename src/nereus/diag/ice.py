"""Sea ice diagnostics for nereus.

This module provides functions for computing sea ice metrics:
- ice_area: Total sea ice area
- ice_volume: Total sea ice volume
- ice_extent: Sea ice extent (area with concentration above threshold)

All functions are dask-friendly: if inputs are dask arrays, the result
will be a lazy dask array that can be computed later with ``.compute()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nereus.core.types import get_array_data, is_dask_array

if TYPE_CHECKING:
    import xarray as xr


def ice_area(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    *,
    mask: NDArray[np.bool_] | None = None,
) -> float | NDArray:
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

    Returns
    -------
    float or ndarray or dask.array
        Total sea ice area in m^2. Returns float for 1D numpy input,
        ndarray for ND numpy input, or dask array if inputs are dask.

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

    # Flatten area - ravel() works for both numpy and dask
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Apply mask (np.where works with dask arrays)
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        area_arr = np.where(mask_arr, area_arr, 0.0)

    # Handle NaN values - treat as zero concentration
    conc = np.where(np.isfinite(conc), conc, 0.0)

    # Clip concentration to valid range
    conc = np.clip(conc, 0.0, 1.0)

    # Compute ice area: sum(concentration * cell_area)
    result = np.sum(conc * area_arr, axis=-1)

    # Return appropriate type
    if is_lazy:
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
) -> float | NDArray:
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

    Returns
    -------
    float or ndarray or dask.array
        Total sea ice volume in m^3. Returns a dask array if inputs are
        dask arrays (call ``.compute()`` to get the result).

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

    # Flatten area - ravel() works for both numpy and dask
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Apply mask (np.where works with dask arrays)
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        area_arr = np.where(mask_arr, area_arr, 0.0)

    # Handle NaN values (np.where, np.isfinite, np.maximum work with dask)
    thick = np.where(np.isfinite(thick), thick, 0.0)
    thick = np.maximum(thick, 0.0)  # No negative thickness

    # Compute volume based on thickness type
    if concentration is not None:
        # Real thickness (ice-area mean): V = h_ice * a * A_cell
        conc = get_array_data(concentration)
        conc = np.where(np.isfinite(conc), conc, 0.0)
        conc = np.clip(conc, 0.0, 1.0)
        result = np.sum(thick * conc * area_arr, axis=-1)
    else:
        # Effective thickness (grid-cell mean): V = h_eff * A_cell
        result = np.sum(thick * area_arr, axis=-1)

    # Return appropriate type
    if is_lazy:
        return result  # Keep lazy, user calls .compute()
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result


def ice_extent(
    concentration: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    *,
    threshold: float = 0.15,
    mask: NDArray[np.bool_] | None = None,
) -> float | NDArray:
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

    Returns
    -------
    float or ndarray or dask.array
        Total sea ice extent in m^2. Returns a dask array if inputs are
        dask arrays (call ``.compute()`` to get the result).

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

    # Flatten area - ravel() works for both numpy and dask
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Apply mask (np.where works with dask arrays)
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        area_arr = np.where(mask_arr, area_arr, 0.0)

    # Handle NaN values
    conc = np.where(np.isfinite(conc), conc, 0.0)

    # Compute extent: sum(cell_area) where concentration >= threshold
    ice_mask = conc >= threshold
    result = np.sum(area_arr * ice_mask, axis=-1)

    # Return appropriate type
    if is_lazy:
        return result  # Keep lazy, user calls .compute()
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result
