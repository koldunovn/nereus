"""Vertical/ocean diagnostics for nereus.

This module provides functions for computing vertically-integrated ocean metrics:
- volume_mean: Volume-weighted mean in a depth range
- heat_content: Ocean heat content
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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
    float or ndarray
        Volume-weighted mean. Returns float for 2D input (nlevels, npoints),
        ndarray for higher-dimensional input (preserving leading dimensions).

    Examples
    --------
    >>> # Mean temperature in upper 500m
    >>> mean_temp = nr.volume_mean(
    ...     temp, mesh.area, mesh.layer_thickness, mesh.depth,
    ...     depth_max=500
    ... )

    >>> # Mean salinity over full depth
    >>> mean_sal = nr.volume_mean(sal, mesh.area, mesh.layer_thickness)
    """
    # Handle xarray DataArray
    if hasattr(data, "values"):
        data = data.values
    data_arr = np.asarray(data)
    area_arr = np.asarray(area)
    thick_arr = np.asarray(thickness)

    # Get number of levels from data
    nlev_data = data_arr.shape[-2]

    # Handle area: can be 1D (npoints,) or 2D (nlevels, npoints)
    if area_arr.ndim == 1:
        # Surface area only - will broadcast later
        npoints = area_arr.shape[0]
        area_is_2d = False
    elif area_arr.ndim == 2:
        nlev_area, npoints = area_arr.shape
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

    # Ensure thickness is 2D (nlevels, npoints)
    if thick_arr.ndim == 1:
        nlevels = thick_arr.shape[0]
        thick_arr = np.broadcast_to(thick_arr[:, np.newaxis], (nlevels, npoints))
    else:
        nlevels = thick_arr.shape[0]

    # Validate dimensions
    if nlevels != nlev_data:
        raise ValueError(
            f"thickness has {nlevels} levels but data has {nlev_data}"
        )

    # Build depth mask if needed
    level_mask = np.ones(nlevels, dtype=bool)
    if depth_min is not None or depth_max is not None:
        if depth is None:
            raise ValueError("depth array required when using depth_min/depth_max")
        depth_arr = np.asarray(depth).ravel()
        if depth_min is not None:
            level_mask &= depth_arr >= depth_min
        if depth_max is not None:
            level_mask &= depth_arr <= depth_max

    # Build horizontal mask
    if mask is not None:
        horiz_mask = np.asarray(mask).ravel()
    else:
        horiz_mask = np.ones(npoints, dtype=bool)

    # Compute cell volumes: thickness * area
    if area_is_2d:
        # Area is (nlevels, npoints)
        volumes = thick_arr * area_arr
    else:
        # Area is (npoints,), broadcast to (nlevels, npoints)
        volumes = thick_arr * area_arr[np.newaxis, :]

    # Apply masks
    volumes = volumes * level_mask[:, np.newaxis] * horiz_mask[np.newaxis, :]

    # Handle NaN in data - set volume to 0 where data is NaN
    # For ND data, we need to handle this per timestep
    if data_arr.ndim == 2:
        # Shape: (nlevels, npoints)
        valid_mask = np.isfinite(data_arr)
        volumes_valid = np.where(valid_mask, volumes, 0.0)
        total_volume = np.sum(volumes_valid)
        if total_volume == 0:
            return np.nan
        return float(np.nansum(data_arr * volumes_valid) / total_volume)
    else:
        # Higher dimensional: (..., nlevels, npoints)
        # Process keeping leading dimensions
        leading_shape = data_arr.shape[:-2]
        result = np.zeros(leading_shape)

        # Flatten leading dimensions for iteration
        data_flat = data_arr.reshape(-1, nlevels, npoints)
        result_flat = result.ravel()

        for i in range(data_flat.shape[0]):
            slice_data = data_flat[i]
            valid_mask = np.isfinite(slice_data)
            volumes_valid = np.where(valid_mask, volumes, 0.0)
            total_volume = np.sum(volumes_valid)
            if total_volume == 0:
                result_flat[i] = np.nan
            else:
                result_flat[i] = np.nansum(slice_data * volumes_valid) / total_volume

        return result_flat.reshape(leading_shape)


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
    float or ndarray
        Ocean heat content in Joules.

    Examples
    --------
    >>> # Total ocean heat content
    >>> ohc = nr.heat_content(temp, mesh.area, mesh.layer_thickness)

    >>> # Heat content in upper 700m
    >>> ohc_700 = nr.heat_content(
    ...     temp, mesh.area, mesh.layer_thickness, mesh.depth,
    ...     depth_max=700
    ... )
    """
    # Handle xarray DataArray
    if hasattr(temperature, "values"):
        temperature = temperature.values
    temp_arr = np.asarray(temperature)
    area_arr = np.asarray(area)
    thick_arr = np.asarray(thickness)

    # Get number of levels from data
    nlev_data = temp_arr.shape[-2]

    # Handle area: can be 1D (npoints,) or 2D (nlevels, npoints)
    if area_arr.ndim == 1:
        # Surface area only - will broadcast later
        npoints = area_arr.shape[0]
        area_is_2d = False
    elif area_arr.ndim == 2:
        nlev_area, npoints = area_arr.shape
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

    # Ensure thickness is 2D (nlevels, npoints)
    if thick_arr.ndim == 1:
        nlevels = thick_arr.shape[0]
        thick_arr = np.broadcast_to(thick_arr[:, np.newaxis], (nlevels, npoints))
    else:
        nlevels = thick_arr.shape[0]

    # Validate dimensions
    if nlevels != nlev_data:
        raise ValueError(
            f"thickness has {nlevels} levels but data has {nlev_data}"
        )

    # Build depth mask if needed
    level_mask = np.ones(nlevels, dtype=bool)
    if depth_min is not None or depth_max is not None:
        if depth is None:
            raise ValueError("depth array required when using depth_min/depth_max")
        depth_arr = np.asarray(depth).ravel()
        if depth_min is not None:
            level_mask &= depth_arr >= depth_min
        if depth_max is not None:
            level_mask &= depth_arr <= depth_max

    # Build horizontal mask
    if mask is not None:
        horiz_mask = np.asarray(mask).ravel()
    else:
        horiz_mask = np.ones(npoints, dtype=bool)

    # Compute cell volumes: thickness * area
    if area_is_2d:
        # Area is (nlevels, npoints)
        volumes = thick_arr * area_arr
    else:
        # Area is (npoints,), broadcast to (nlevels, npoints)
        volumes = thick_arr * area_arr[np.newaxis, :]

    # Apply masks
    volumes = volumes * level_mask[:, np.newaxis] * horiz_mask[np.newaxis, :]

    # Compute heat content: rho * cp * sum((T - T_ref) * volume)
    temp_anomaly = temp_arr - reference_temp

    if temp_arr.ndim == 2:
        # Shape: (nlevels, npoints)
        valid_mask = np.isfinite(temp_anomaly)
        volumes_valid = np.where(valid_mask, volumes, 0.0)
        heat = np.nansum(temp_anomaly * volumes_valid)
        return float(rho * cp * heat)
    else:
        # Higher dimensional: (..., nlevels, npoints)
        leading_shape = temp_arr.shape[:-2]
        result = np.zeros(leading_shape)

        temp_flat = temp_anomaly.reshape(-1, nlevels, npoints)
        result_flat = result.ravel()

        for i in range(temp_flat.shape[0]):
            slice_temp = temp_flat[i]
            valid_mask = np.isfinite(slice_temp)
            volumes_valid = np.where(valid_mask, volumes, 0.0)
            heat = np.nansum(slice_temp * volumes_valid)
            result_flat[i] = rho * cp * heat

        return result_flat.reshape(leading_shape)
