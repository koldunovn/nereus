"""Vertical/ocean diagnostics for nereus.

This module provides functions for computing vertically-integrated ocean metrics:
- volume_mean: Volume-weighted mean in a depth range
- heat_content: Ocean heat content

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

# Physical constants
RHO_SEAWATER = 1025.0  # kg/m^3
CP_SEAWATER = 3985.0  # J/(kg·K)


def volume_mean(
    data: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    thickness: NDArray[np.floating],
    depth: NDArray[np.floating] | None = None,
    *,
    depth_min: float | None = None,
    depth_max: float | None = None,
    mask: NDArray[np.bool_] | None = None,
) -> float | NDArray:
    """Compute volume-weighted mean of a quantity.

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    data : array_like
        3D data with shape (nlevels, npoints) or higher-dimensional with
        the last two axes being (nlevels, npoints). For time series,
        shape would be (ntime, nlevels, npoints).
    area : array_like
        Grid cell areas in m^2. Can be either:
        - 1D array of shape (npoints,) for surface area (uniform across depth)
        - 2D array of shape (nlevels, npoints) for depth-dependent area
        If 2D and has one extra level compared to data layers, the extra
        level is dropped with a warning (levels vs layers).
    thickness : array_like
        Layer thicknesses in meters, shape (nlevels, npoints) or (nlevels,)
        if uniform across points.
    depth : array_like, optional
        Depth of layer centers in meters (positive downward), shape (nlevels,).
        Required if depth_min or depth_max are specified.
    depth_min : float, optional
        Minimum depth to include (meters, positive downward).
    depth_max : float, optional
        Maximum depth to include (meters, positive downward).
    mask : array_like, optional
        Boolean mask for horizontal points, shape (npoints,). True = include.

    Returns
    -------
    float or ndarray or dask.array
        Volume-weighted mean. Returns float for 2D numpy input (nlevels, npoints),
        ndarray for higher-dimensional numpy input, or dask array if inputs are dask.

    Examples
    --------
    >>> # Mean temperature in upper 500m
    >>> mean_temp = nr.volume_mean(
    ...     temp, mesh.area, mesh.layer_thickness, mesh.depth,
    ...     depth_max=500
    ... )

    >>> # Mean salinity over full depth
    >>> mean_sal = nr.volume_mean(sal, mesh.area, mesh.layer_thickness)

    >>> # With dask arrays (lazy computation)
    >>> mean_temp = nr.volume_mean(temp_dask, mesh.area, mesh.layer_thickness)
    >>> mean_temp.compute()  # triggers actual computation
    """
    # Extract arrays, preserving dask
    data_arr = get_array_data(data)
    area_arr = get_array_data(area)
    thick_arr = get_array_data(thickness)
    is_lazy = is_dask_array(data)

    # Get number of levels from data
    nlev_data = data_arr.shape[-2]
    npoints = data_arr.shape[-1]

    # Handle area: can be 1D (npoints,) or 2D (nlevels, npoints)
    if area_arr.ndim == 1:
        # Surface area only - will broadcast later
        area_is_2d = False
    elif area_arr.ndim == 2:
        nlev_area = area_arr.shape[0]
        area_is_2d = True
        # Check if area has one extra level (levels vs layers mismatch)
        if nlev_area != nlev_data:
            diff = nlev_area - nlev_data
            if diff != 1:
                raise ValueError(
                    f"area has {nlev_area} vertical levels but data has {nlev_data}; "
                    "only area having one extra level is supported (levels vs layers)."
                )
            warnings.warn(
                f"area has one more vertical level than data; "
                f"using the first {nlev_data} levels of area to match data "
                "(levels vs layers).",
                UserWarning,
                stacklevel=2,
            )
            area_arr = area_arr[:nlev_data, :]
    else:
        raise ValueError(f"area must be 1D or 2D, got {area_arr.ndim}D")

    # Handle thickness - need to broadcast if 1D
    if thick_arr.ndim == 1:
        nlevels = thick_arr.shape[0]
        # Use broadcasting instead of np.broadcast_to for dask compatibility
        thick_2d = thick_arr[:, np.newaxis]  # Shape: (nlevels, 1)
    else:
        nlevels = thick_arr.shape[0]
        thick_2d = thick_arr

    # Validate dimensions
    if nlevels != nlev_data:
        raise ValueError(
            f"thickness has {nlevels} levels but data has {nlev_data}"
        )

    # Build depth mask if needed (this is small, keep as numpy)
    level_mask = np.ones(nlevels, dtype=np.float64)
    if depth_min is not None or depth_max is not None:
        if depth is None:
            raise ValueError("depth array required when using depth_min/depth_max")
        depth_arr = np.asarray(get_array_data(depth)).ravel()
        if depth_min is not None:
            level_mask = level_mask * (depth_arr >= depth_min)
        if depth_max is not None:
            level_mask = level_mask * (depth_arr <= depth_max)

    # Build horizontal mask
    if mask is not None:
        horiz_mask = get_array_data(mask)
        if hasattr(horiz_mask, "ravel"):
            horiz_mask = horiz_mask.ravel()
        else:
            horiz_mask = np.asarray(horiz_mask).ravel()
        horiz_mask = horiz_mask.astype(np.float64)
    else:
        horiz_mask = 1.0  # Scalar, broadcasts everywhere

    # Compute cell volumes: thickness * area
    # Shape: (nlevels, npoints) or broadcasts to it
    if area_is_2d:
        volumes = thick_2d * area_arr
    else:
        volumes = thick_2d * area_arr[np.newaxis, :]

    # Apply masks - use multiplication for dask compatibility
    # level_mask: (nlevels,) -> (nlevels, 1)
    # horiz_mask: (npoints,) or scalar
    volumes = volumes * level_mask[:, np.newaxis] * horiz_mask

    # Handle NaN in data - set volume to 0 where data is NaN
    valid_mask = np.isfinite(data_arr)
    volumes_valid = np.where(valid_mask, volumes, 0.0)

    # Replace NaN with 0 for summation
    data_filled = np.where(valid_mask, data_arr, 0.0)

    # Compute weighted sum and total volume
    # Sum over last two axes (nlevels, npoints)
    weighted_sum = np.sum(data_filled * volumes_valid, axis=(-2, -1))
    total_volume = np.sum(volumes_valid, axis=(-2, -1))

    # Compute mean, handling zero volume
    result = np.where(total_volume > 0, weighted_sum / total_volume, np.nan)

    # Return appropriate type
    if is_lazy:
        return result
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result


def heat_content(
    temperature: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    thickness: NDArray[np.floating],
    depth: NDArray[np.floating] | None = None,
    *,
    depth_min: float | None = None,
    depth_max: float | None = None,
    reference_temp: float = 0.0,
    mask: NDArray[np.bool_] | None = None,
    rho: float = RHO_SEAWATER,
    cp: float = CP_SEAWATER,
) -> float | NDArray:
    """Compute ocean heat content.

    Heat content is computed as: OHC = rho * cp * sum(T * thickness * area)
    where the sum is over all grid cells in the specified depth range.

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    temperature : array_like
        Temperature in degrees Celsius, shape (nlevels, npoints) or higher
        dimensional with the last two axes being (nlevels, npoints).
    area : array_like
        Grid cell areas in m^2. Can be either:
        - 1D array of shape (npoints,) for surface area (uniform across depth)
        - 2D array of shape (nlevels, npoints) for depth-dependent area
        If 2D and has one extra level compared to data layers, the extra
        level is dropped with a warning (levels vs layers).
    thickness : array_like
        Layer thicknesses in meters, shape (nlevels, npoints) or (nlevels,).
    depth : array_like, optional
        Depth of layer centers in meters (positive downward).
        Required if depth_min or depth_max are specified.
    depth_min : float, optional
        Minimum depth to include (meters, positive downward).
    depth_max : float, optional
        Maximum depth to include (meters, positive downward).
    reference_temp : float
        Reference temperature for heat content calculation. Default 0°C.
    mask : array_like, optional
        Boolean mask for horizontal points. True = include.
    rho : float
        Seawater density in kg/m^3. Default 1025.
    cp : float
        Specific heat capacity in J/(kg·K). Default 3985.

    Returns
    -------
    float or ndarray or dask.array
        Ocean heat content in Joules. Returns a dask array if inputs are dask.

    Examples
    --------
    >>> # Total ocean heat content
    >>> ohc = nr.heat_content(temp, mesh.area, mesh.layer_thickness)

    >>> # Heat content in upper 700m
    >>> ohc_700 = nr.heat_content(
    ...     temp, mesh.area, mesh.layer_thickness, mesh.depth,
    ...     depth_max=700
    ... )

    >>> # With dask arrays (lazy computation)
    >>> ohc = nr.heat_content(temp_dask, mesh.area, mesh.layer_thickness)
    >>> ohc.compute()  # triggers actual computation
    """
    # Extract arrays, preserving dask
    temp_arr = get_array_data(temperature)
    area_arr = get_array_data(area)
    thick_arr = get_array_data(thickness)
    is_lazy = is_dask_array(temperature)

    # Get number of levels from data
    nlev_data = temp_arr.shape[-2]
    npoints = temp_arr.shape[-1]

    # Handle area: can be 1D (npoints,) or 2D (nlevels, npoints)
    if area_arr.ndim == 1:
        area_is_2d = False
    elif area_arr.ndim == 2:
        nlev_area = area_arr.shape[0]
        area_is_2d = True
        # Check if area has one extra level (levels vs layers mismatch)
        if nlev_area != nlev_data:
            diff = nlev_area - nlev_data
            if diff != 1:
                raise ValueError(
                    f"area has {nlev_area} vertical levels but data has {nlev_data}; "
                    "only area having one extra level is supported (levels vs layers)."
                )
            warnings.warn(
                f"area has one more vertical level than data; "
                f"using the first {nlev_data} levels of area to match data "
                "(levels vs layers).",
                UserWarning,
                stacklevel=2,
            )
            area_arr = area_arr[:nlev_data, :]
    else:
        raise ValueError(f"area must be 1D or 2D, got {area_arr.ndim}D")

    # Handle thickness - need to broadcast if 1D
    if thick_arr.ndim == 1:
        nlevels = thick_arr.shape[0]
        thick_2d = thick_arr[:, np.newaxis]  # Shape: (nlevels, 1)
    else:
        nlevels = thick_arr.shape[0]
        thick_2d = thick_arr

    # Validate dimensions
    if nlevels != nlev_data:
        raise ValueError(
            f"thickness has {nlevels} levels but data has {nlev_data}"
        )

    # Build depth mask if needed (this is small, keep as numpy)
    level_mask = np.ones(nlevels, dtype=np.float64)
    if depth_min is not None or depth_max is not None:
        if depth is None:
            raise ValueError("depth array required when using depth_min/depth_max")
        depth_arr = np.asarray(get_array_data(depth)).ravel()
        if depth_min is not None:
            level_mask = level_mask * (depth_arr >= depth_min)
        if depth_max is not None:
            level_mask = level_mask * (depth_arr <= depth_max)

    # Build horizontal mask
    if mask is not None:
        horiz_mask = get_array_data(mask)
        if hasattr(horiz_mask, "ravel"):
            horiz_mask = horiz_mask.ravel()
        else:
            horiz_mask = np.asarray(horiz_mask).ravel()
        horiz_mask = horiz_mask.astype(np.float64)
    else:
        horiz_mask = 1.0  # Scalar, broadcasts everywhere

    # Compute cell volumes: thickness * area
    if area_is_2d:
        volumes = thick_2d * area_arr
    else:
        volumes = thick_2d * area_arr[np.newaxis, :]

    # Apply masks
    volumes = volumes * level_mask[:, np.newaxis] * horiz_mask

    # Compute heat content: rho * cp * sum((T - T_ref) * volume)
    temp_anomaly = temp_arr - reference_temp

    # Handle NaN in data
    valid_mask = np.isfinite(temp_anomaly)
    volumes_valid = np.where(valid_mask, volumes, 0.0)
    temp_filled = np.where(valid_mask, temp_anomaly, 0.0)

    # Sum over last two axes (nlevels, npoints)
    heat = np.sum(temp_filled * volumes_valid, axis=(-2, -1))
    result = rho * cp * heat

    # Return appropriate type
    if is_lazy:
        return result
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result
