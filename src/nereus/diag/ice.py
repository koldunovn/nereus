"""Sea ice diagnostics for nereus.

This module provides functions for computing sea ice metrics:
- ice_area: Total sea ice area
- ice_volume: Total sea ice volume
- ice_extent: Sea ice extent (area with concentration above threshold)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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
    float or ndarray
        Total sea ice area in m^2. Returns float for 1D input, ndarray
        for ND input (preserving leading dimensions).

    Examples
    --------
    >>> # 1D concentration array
    >>> total_area = nr.ice_area(sic, mesh.area)

    >>> # With time dimension (time, npoints)
    >>> area_timeseries = nr.ice_area(sic, mesh.area)

    >>> # With hemisphere mask
    >>> nh_area = nr.ice_area(sic, mesh.area, mask=mesh.lat > 0)
    """
    # Handle xarray DataArray
    if hasattr(concentration, "values"):
        concentration = concentration.values
    conc = np.asarray(concentration)
    area_arr = np.asarray(area).ravel()

    # Validate concentration values
    if conc.size > 0:
        valid_conc = conc[np.isfinite(conc)]
        if valid_conc.size > 0 and (valid_conc.min() < -0.01 or valid_conc.max() > 1.01):
            # Allow small numerical errors
            pass  # Could warn, but let's be permissive

    # Apply mask
    if mask is not None:
        mask = np.asarray(mask).ravel()
        area_arr = np.where(mask, area_arr, 0.0)

    # Handle NaN values - treat as zero concentration
    conc = np.where(np.isfinite(conc), conc, 0.0)

    # Clip concentration to valid range
    conc = np.clip(conc, 0.0, 1.0)

    # Compute ice area: sum(concentration * cell_area)
    if conc.ndim == 1:
        return float(np.sum(conc * area_arr))
    else:
        # ND array - sum over last axis
        return np.sum(conc * area_arr, axis=-1)


def ice_volume(
    thickness: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    concentration: NDArray | "xr.DataArray" | None = None,
    *,
    mask: NDArray[np.bool_] | None = None,
) -> float | NDArray:
    """Compute total sea ice volume.

    Sea ice volume is computed as the sum of thickness * area * concentration.
    If concentration is not provided, it is assumed to be 1 where thickness > 0.

    Parameters
    ----------
    thickness : array_like
        Sea ice thickness in meters. Can be 1D (npoints,) or ND with the
        last axis being npoints.
    area : array_like
        Grid cell areas in m^2.
    concentration : array_like, optional
        Sea ice concentration (fraction, 0-1). If None, assumed 1 where
        thickness > 0.
    mask : array_like, optional
        Boolean mask (True = include).

    Returns
    -------
    float or ndarray
        Total sea ice volume in m^3.

    Examples
    --------
    >>> volume = nr.ice_volume(sit, mesh.area)
    >>> volume = nr.ice_volume(sit, mesh.area, concentration=sic)
    """
    # Handle xarray DataArray
    if hasattr(thickness, "values"):
        thickness = thickness.values
    thick = np.asarray(thickness)
    area_arr = np.asarray(area).ravel()

    # Handle concentration
    if concentration is not None:
        if hasattr(concentration, "values"):
            concentration = concentration.values
        conc = np.asarray(concentration)
        conc = np.where(np.isfinite(conc), conc, 0.0)
        conc = np.clip(conc, 0.0, 1.0)
    else:
        # Assume full concentration where ice exists
        conc = np.where(thick > 0, 1.0, 0.0)

    # Apply mask
    if mask is not None:
        mask = np.asarray(mask).ravel()
        area_arr = np.where(mask, area_arr, 0.0)

    # Handle NaN values
    thick = np.where(np.isfinite(thick), thick, 0.0)
    thick = np.maximum(thick, 0.0)  # No negative thickness

    # Compute volume: sum(thickness * concentration * cell_area)
    if thick.ndim == 1:
        return float(np.sum(thick * conc * area_arr))
    else:
        return np.sum(thick * conc * area_arr, axis=-1)


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
    float or ndarray
        Total sea ice extent in m^2.

    Examples
    --------
    >>> extent = nr.ice_extent(sic, mesh.area)
    >>> extent_nh = nr.ice_extent(sic, mesh.area, mask=mesh.lat > 0)
    """
    # Handle xarray DataArray
    if hasattr(concentration, "values"):
        concentration = concentration.values
    conc = np.asarray(concentration)
    area_arr = np.asarray(area).ravel()

    # Apply mask
    if mask is not None:
        mask = np.asarray(mask).ravel()
        area_arr = np.where(mask, area_arr, 0.0)

    # Handle NaN values
    conc = np.where(np.isfinite(conc), conc, 0.0)

    # Compute extent: sum(cell_area) where concentration >= threshold
    ice_mask = conc >= threshold

    if conc.ndim == 1:
        return float(np.sum(area_arr * ice_mask))
    else:
        return np.sum(area_arr * ice_mask, axis=-1)
