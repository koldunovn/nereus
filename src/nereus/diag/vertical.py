"""Vertical/ocean diagnostics for nereus.

This module provides functions for computing ocean diagnostics:
- surface_mean: Area-weighted mean for 2D fields (SST, SSS, etc.)
- volume_mean: Volume-weighted mean in a depth range
- heat_content: Ocean heat content
- find_closest_depth: Find index and value of closest depth to target
- interpolate_to_depth: Interpolate 3D data to target depths

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
CP_SEAWATER = 3990.0  # J/(kg·K) - consistent with FESOM2


def surface_mean(
    data: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    *,
    mask: NDArray[np.bool_] | None = None,
) -> float | NDArray:
    """Compute area-weighted mean of a 2D field (single level).

    This is commonly used for surface fields like SST, SSS, or for
    analyzing a single depth level.

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    data : array_like
        2D data with shape (npoints,) or higher-dimensional with
        the last axis being npoints. For time series, shape would be
        (ntime, npoints).
    area : array_like
        Grid cell areas in m^2, shape (npoints,).
    mask : array_like, optional
        Boolean mask for horizontal points, shape (npoints,). True = include.

    Returns
    -------
    float or ndarray or dask.array
        Area-weighted mean. Returns float for 1D numpy input (npoints,),
        ndarray for higher-dimensional numpy input, or dask array if inputs
        are dask.

    Examples
    --------
    >>> # Mean SST
    >>> mean_sst = nr.surface_mean(sst, mesh.area)

    >>> # Mean SST in a region
    >>> mean_sst = nr.surface_mean(sst, mesh.area, mask=region_mask)

    >>> # With dask arrays (lazy computation)
    >>> mean_sst = nr.surface_mean(sst_dask, mesh.area)
    >>> mean_sst.compute()  # triggers actual computation
    """
    # Extract arrays, preserving dask
    data_arr = get_array_data(data)
    area_arr = get_array_data(area)
    is_lazy = is_dask_array(data)

    # Warn if dask data is mixed with large numpy arrays (causes graph bloat)
    if is_lazy and not is_dask_array(area_arr) and area_arr.nbytes > 10_000_000:
        warnings.warn(
            f"Data is a dask array but area ({area_arr.nbytes / 1e6:.1f} MB) is numpy. "
            "This can cause very large dask graphs. Consider loading all "
            "large arrays with dask (e.g., xr.open_dataset(..., chunks='auto')).",
            UserWarning,
            stacklevel=2,
        )

    # Flatten area
    if hasattr(area_arr, "ravel"):
        area_arr = area_arr.ravel()
    else:
        area_arr = np.asarray(area_arr).ravel()

    # Build weights from area, applying mask if provided
    if mask is not None:
        mask_arr = get_array_data(mask)
        if hasattr(mask_arr, "ravel"):
            mask_arr = mask_arr.ravel()
        else:
            mask_arr = np.asarray(mask_arr).ravel()
        # Set weights to NaN where mask is False (will be ignored by nansum)
        weights = np.where(mask_arr, area_arr, np.nan)
    else:
        weights = area_arr

    # Compute weighted mean using nansum
    # NaN values in data or weights are automatically excluded
    weighted_sum = np.nansum(data_arr * weights, axis=-1)

    # For total weight, only count weights where data is valid
    # Use indicator trick: data * 0 + 1 gives 1 where valid, NaN where NaN
    valid_indicator = data_arr * 0 + 1
    total_weight = np.nansum(weights * valid_indicator, axis=-1)

    # Compute mean, handling zero weight
    result = np.where(total_weight > 0, weighted_sum / total_weight, np.nan)

    # Return appropriate type
    if is_lazy:
        return result
    elif np.ndim(result) == 0:
        return float(result)
    else:
        return result


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

    # Warn if dask data is mixed with large numpy arrays (causes graph bloat)
    if is_lazy:
        large_numpy_arrays = []
        # Check area - threshold ~10MB (large enough to cause issues)
        if not is_dask_array(area_arr) and area_arr.nbytes > 10_000_000:
            large_numpy_arrays.append(f"area ({area_arr.nbytes / 1e6:.1f} MB)")
        # Check thickness
        if not is_dask_array(thick_arr) and thick_arr.nbytes > 10_000_000:
            large_numpy_arrays.append(f"thickness ({thick_arr.nbytes / 1e6:.1f} MB)")
        if large_numpy_arrays:
            warnings.warn(
                f"Data is a dask array but {', '.join(large_numpy_arrays)} "
                f"{'is' if len(large_numpy_arrays) == 1 else 'are'} numpy. "
                "This can cause very large dask graphs. Consider loading all "
                "large arrays with dask (e.g., xr.open_dataset(..., chunks='auto')).",
                UserWarning,
                stacklevel=2,
            )

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

    # For dask arrays, ensure thickness is also dask to avoid graph bloat
    # When numpy arrays are broadcast with dask arrays, they get embedded
    # in every task, causing massive graph sizes
    if is_lazy and not is_dask_array(thick_2d):
        import dask.array as da

        # Get chunks from data_arr for the last two axes (nlevels, npoints)
        data_chunks = data_arr.chunks
        level_chunks = data_chunks[-2]  # chunks along nlevels axis
        point_chunks = data_chunks[-1]  # chunks along npoints axis

        if thick_2d.shape[-1] == 1:
            # thick_2d is (nlevels, 1) - broadcast along points
            thick_2d = da.from_array(thick_2d, chunks=(level_chunks, 1))
        else:
            # thick_2d is (nlevels, npoints)
            thick_2d = da.from_array(thick_2d, chunks=(level_chunks, point_chunks))

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

    # Compute cell volumes: thickness * area
    # Shape: (nlevels, npoints) or broadcasts to it
    if area_is_2d:
        volumes = thick_2d * area_arr
    else:
        volumes = thick_2d * area_arr[np.newaxis, :]

    # Apply depth mask by setting excluded levels to NaN
    if depth_min is not None or depth_max is not None:
        level_mask_nan = np.where(level_mask, 1.0, np.nan)
        volumes = volumes * level_mask_nan[:, np.newaxis]

    # Apply horizontal mask by setting excluded points to NaN
    if mask is not None:
        horiz_mask = get_array_data(mask)
        if hasattr(horiz_mask, "ravel"):
            horiz_mask = horiz_mask.ravel()
        else:
            horiz_mask = np.asarray(horiz_mask).ravel()
        horiz_mask_nan = np.where(horiz_mask, 1.0, np.nan)
        volumes = volumes * horiz_mask_nan

    # Compute weighted mean using nansum
    # NaN values in data or volumes are automatically excluded
    weighted_sum = np.nansum(data_arr * volumes, axis=(-2, -1))

    # For total volume, only count volumes where data is valid
    # Use indicator trick: data * 0 + 1 gives 1 where valid, NaN where NaN
    valid_indicator = data_arr * 0 + 1
    total_volume = np.nansum(volumes * valid_indicator, axis=(-2, -1))

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
    output: str = "total",
) -> float | NDArray:
    """Compute ocean heat content.

    Heat content can be computed as either:
    - Total (default): OHC = rho * cp * sum(T * thickness * area) in Joules
    - Map: OHC = rho * cp * sum_z(T * thickness) in J/m² at each point

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
        Note: area is not used when output="map".
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
        Specific heat capacity in J/(kg·K). Default 3990.
    output : str
        Output type. One of:
        - "total": Total heat content in Joules (scalar per timestep)
        - "map": Heat content per unit area in J/m² (2D field at each point)
        Default is "total".

    Returns
    -------
    float or ndarray or dask.array
        If output="total": Ocean heat content in Joules.
        If output="map": Heat content per unit area in J/m², shape (npoints,)
        or (..., npoints) for higher-dimensional input.
        Returns a dask array if inputs are dask.

    Examples
    --------
    >>> # Total ocean heat content
    >>> ohc = nr.heat_content(temp, mesh.area, mesh.layer_thickness)

    >>> # Heat content in upper 700m
    >>> ohc_700 = nr.heat_content(
    ...     temp, mesh.area, mesh.layer_thickness, mesh.depth,
    ...     depth_max=700
    ... )

    >>> # Heat content map (J/m² at each point, like FESOM2 output)
    >>> ohc_map = nr.heat_content(
    ...     temp, mesh.area, mesh.layer_thickness,
    ...     output="map"
    ... )

    >>> # With dask arrays (lazy computation)
    >>> ohc = nr.heat_content(temp_dask, mesh.area, mesh.layer_thickness)
    >>> ohc.compute()  # triggers actual computation
    """
    # Validate output parameter
    if output not in ("total", "map"):
        raise ValueError(f"output must be 'total' or 'map', got '{output}'")

    # Extract arrays, preserving dask
    temp_arr = get_array_data(temperature)
    area_arr = get_array_data(area)
    thick_arr = get_array_data(thickness)
    is_lazy = is_dask_array(temperature)

    # Warn if dask data is mixed with large numpy arrays (causes graph bloat)
    if is_lazy:
        large_numpy_arrays = []
        # Check area - threshold ~10MB (large enough to cause issues)
        if not is_dask_array(area_arr) and area_arr.nbytes > 10_000_000:
            large_numpy_arrays.append(f"area ({area_arr.nbytes / 1e6:.1f} MB)")
        # Check thickness
        if not is_dask_array(thick_arr) and thick_arr.nbytes > 10_000_000:
            large_numpy_arrays.append(f"thickness ({thick_arr.nbytes / 1e6:.1f} MB)")
        if large_numpy_arrays:
            warnings.warn(
                f"Data is a dask array but {', '.join(large_numpy_arrays)} "
                f"{'is' if len(large_numpy_arrays) == 1 else 'are'} numpy. "
                "This can cause very large dask graphs. Consider loading all "
                "large arrays with dask (e.g., xr.open_dataset(..., chunks='auto')).",
                UserWarning,
                stacklevel=2,
            )

    # Get number of levels from data
    nlev_data = temp_arr.shape[-2]
    npoints = temp_arr.shape[-1]

    # Handle area: can be 1D (npoints,) or 2D (nlevels, npoints)
    # Only needed for output="total"
    area_is_2d = False
    if output == "total":
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

    # Compute heat content: rho * cp * sum((T - T_ref) * thickness [* area])
    temp_anomaly = temp_arr - reference_temp

    # For dask arrays, ensure thickness is also dask to avoid graph bloat
    # When numpy arrays are broadcast with dask arrays, they get embedded
    # in every task, causing massive graph sizes
    if is_lazy and not is_dask_array(thick_2d):
        import dask.array as da

        # Get chunks from temp_arr for the last two axes (nlevels, npoints)
        temp_chunks = temp_arr.chunks
        level_chunks = temp_chunks[-2]  # chunks along nlevels axis
        point_chunks = temp_chunks[-1]  # chunks along npoints axis

        if thick_2d.shape[-1] == 1:
            # thick_2d is (nlevels, 1) - broadcast along points
            thick_2d = da.from_array(thick_2d, chunks=(level_chunks, 1))
        else:
            # thick_2d is (nlevels, npoints)
            thick_2d = da.from_array(thick_2d, chunks=(level_chunks, point_chunks))

    if output == "total":
        # Compute cell volumes: thickness * area
        if area_is_2d:
            volumes = thick_2d * area_arr
        else:
            volumes = thick_2d * area_arr[np.newaxis, :]

        # Apply depth mask by setting excluded levels to NaN
        if depth_min is not None or depth_max is not None:
            level_mask_nan = np.where(level_mask, 1.0, np.nan)
            volumes = volumes * level_mask_nan[:, np.newaxis]

        # Apply horizontal mask by setting excluded points to NaN
        if mask is not None:
            horiz_mask = get_array_data(mask)
            if hasattr(horiz_mask, "ravel"):
                horiz_mask = horiz_mask.ravel()
            else:
                horiz_mask = np.asarray(horiz_mask).ravel()
            horiz_mask_nan = np.where(horiz_mask, 1.0, np.nan)
            volumes = volumes * horiz_mask_nan

        # Sum over last two axes (nlevels, npoints) using nansum
        # NaN values in temp_anomaly or volumes are automatically excluded
        heat = np.nansum(temp_anomaly * volumes, axis=(-2, -1))
        result = rho * cp * heat

        # Return appropriate type
        if is_lazy:
            return result
        elif np.ndim(result) == 0:
            return float(result)
        else:
            return result

    else:  # output == "map"
        # Compute heat content per unit area: rho * cp * sum_z(T * thickness)
        thick_masked = thick_2d

        # Apply depth mask by setting excluded levels to NaN
        if depth_min is not None or depth_max is not None:
            level_mask_nan = np.where(level_mask, 1.0, np.nan)
            thick_masked = thick_masked * level_mask_nan[:, np.newaxis]

        # Sum over vertical axis only (second to last axis) using nansum
        # NaN values in temp_anomaly or thickness are automatically excluded
        heat_per_area = np.nansum(temp_anomaly * thick_masked, axis=-2)

        # Apply horizontal mask (use 0 for masked points, not NaN, for map output)
        if mask is not None:
            horiz_mask = get_array_data(mask)
            if hasattr(horiz_mask, "ravel"):
                horiz_mask = horiz_mask.ravel()
            else:
                horiz_mask = np.asarray(horiz_mask).ravel()
            heat_per_area = heat_per_area * horiz_mask.astype(np.float64)

        result = rho * cp * heat_per_area

        # Return appropriate type
        if is_lazy:
            return result
        else:
            return np.asarray(result)


def find_closest_depth(
    depths: NDArray[np.floating] | list | "xr.DataArray",
    target: float,
) -> tuple[int, float]:
    """Find the index and value of the depth closest to a target depth.

    This is useful when comparing multiple models with different depth levels
    to find corresponding levels, and to assess how far model depths are
    from target depths.

    Parameters
    ----------
    depths : array_like
        1D array of depth values (typically positive downward in meters).
    target : float
        Target depth value to find the closest match for.

    Returns
    -------
    tuple[int, float]
        A tuple of (index, value) where index is the position of the closest
        depth in the input array, and value is the actual depth at that index.

    Examples
    --------
    >>> depths = [0, 10, 25, 50, 100, 200, 500, 1000]
    >>> idx, val = nr.find_closest_depth(depths, 100)
    >>> print(f"Index: {idx}, Depth: {val}m")
    Index: 4, Depth: 100.0m

    >>> # Check how far model depth is from target
    >>> idx, val = nr.find_closest_depth(depths, 75)
    >>> print(f"Closest depth: {val}m, difference: {abs(val - 75)}m")
    Closest depth: 50.0m, difference: 25.0m
    """
    # Extract array data
    depth_arr = get_array_data(depths)
    depth_arr = np.asarray(depth_arr).ravel()

    # Find index of minimum absolute difference
    idx = int(np.argmin(np.abs(depth_arr - target)))
    value = float(depth_arr[idx])

    return idx, value


def interpolate_to_depth(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating] | "xr.DataArray" | None,
    lat: NDArray[np.floating] | "xr.DataArray" | None,
    model_depths: NDArray[np.floating] | list | "xr.DataArray",
    target_depths: NDArray[np.floating] | list | float,
) -> NDArray | tuple[NDArray, NDArray, NDArray]:
    """Interpolate 3D data to target depth levels using linear interpolation.

    Performs column-wise linear interpolation from model depth levels to
    specified target depths. Values outside the model depth range are
    extrapolated (with a warning for significant extrapolation).

    This function is dask-friendly: if inputs are dask arrays, the result
    will be a lazy dask array that can be computed later with ``.compute()``.

    Parameters
    ----------
    data : array_like
        3D data with shape (nlevels, npoints) or higher-dimensional with
        the last two axes being (nlevels, npoints). For time series,
        shape would be (ntime, nlevels, npoints).
    lon : array_like or None
        Longitude coordinates, shape (npoints,). If provided along with lat,
        these are returned with the result for convenience. Pass None if
        not needed.
    lat : array_like or None
        Latitude coordinates, shape (npoints,). If provided along with lon,
        these are returned with the result for convenience. Pass None if
        not needed.
    model_depths : array_like
        Depth levels of the input data in meters (positive downward),
        shape (nlevels,).
    target_depths : array_like or float
        Target depth(s) to interpolate to in meters. Can be a single value
        or an array of depths.

    Returns
    -------
    ndarray or tuple
        If lon and lat are None:
            Interpolated data with shape (ntargets, npoints) or
            (..., ntargets, npoints) for higher-dimensional input.
            If target_depths is a scalar, ntargets=1.
        If lon and lat are provided:
            Tuple of (interpolated_data, lon, lat).

    Examples
    --------
    >>> # Interpolate temperature to 100m depth (without coordinates)
    >>> temp_100m = nr.interpolate_to_depth(temp, None, None, mesh.depth, 100)

    >>> # Interpolate to multiple standard depths
    >>> standard_depths = [10, 50, 100, 200, 500, 1000]
    >>> temp_interp = nr.interpolate_to_depth(temp, None, None, mesh.depth, standard_depths)

    >>> # With coordinates for plotting
    >>> temp_100m, lon, lat = nr.interpolate_to_depth(
    ...     temp, mesh.lon, mesh.lat, mesh.depth, 100
    ... )
    >>> nr.plot(temp_100m.squeeze(), lon, lat)

    >>> # Compare models at the same depth
    >>> temp_model1 = nr.interpolate_to_depth(temp1, None, None, depths1, 100)
    >>> temp_model2 = nr.interpolate_to_depth(temp2, None, None, depths2, 100)
    """
    # Extract arrays, preserving dask
    data_arr = get_array_data(data)
    is_lazy = is_dask_array(data)

    # Handle model depths
    depth_arr = get_array_data(model_depths)
    depth_arr = np.asarray(depth_arr).ravel()
    nlevels = len(depth_arr)

    # Validate data shape
    if data_arr.shape[-2] != nlevels:
        raise ValueError(
            f"data has {data_arr.shape[-2]} levels but model_depths has {nlevels}"
        )

    # Handle target depths - ensure array
    target_arr = np.atleast_1d(np.asarray(target_depths)).ravel()
    ntargets = len(target_arr)

    # Check for extrapolation
    depth_min, depth_max = depth_arr.min(), depth_arr.max()
    targets_below = target_arr[target_arr < depth_min]
    targets_above = target_arr[target_arr > depth_max]
    if len(targets_below) > 0 or len(targets_above) > 0:
        extrap_msg = []
        if len(targets_below) > 0:
            extrap_msg.append(
                f"{len(targets_below)} target(s) shallower than model minimum ({depth_min}m)"
            )
        if len(targets_above) > 0:
            extrap_msg.append(
                f"{len(targets_above)} target(s) deeper than model maximum ({depth_max}m)"
            )
        warnings.warn(
            f"Extrapolation required: {'; '.join(extrap_msg)}. "
            "Results may be unreliable outside model depth range.",
            UserWarning,
            stacklevel=2,
        )

    # Get shape information
    npoints = data_arr.shape[-1]
    leading_dims = data_arr.shape[:-2]  # e.g., (ntime,) or ()

    # Reshape data to (nbatch, nlevels, npoints) for uniform processing
    if len(leading_dims) == 0:
        # Shape: (nlevels, npoints)
        data_3d = data_arr[np.newaxis, :, :]  # (1, nlevels, npoints)
        nbatch = 1
    else:
        # Shape: (..., nlevels, npoints)
        nbatch = int(np.prod(leading_dims))
        data_3d = data_arr.reshape(nbatch, nlevels, npoints)

    # Perform linear interpolation column by column
    # For each target depth, find bracketing levels and interpolate

    if is_lazy:
        import dask.array as da

        # Interpolation requires all depth levels at once, so rechunk
        # to have a single chunk along the levels axis
        data_3d = data_3d.rechunk({1: -1})

        # For dask, use map_blocks for efficiency
        def _interp_chunk(data_chunk, depth_arr, target_arr):
            """Interpolate a chunk of data."""
            return _linear_interp_vectorized(data_chunk, depth_arr, target_arr)

        # Get chunks from data (after rechunking)
        data_chunks = data_3d.chunks
        result = da.map_blocks(
            _interp_chunk,
            data_3d,
            depth_arr,
            target_arr,
            dtype=data_3d.dtype,
            chunks=(data_chunks[0], (ntargets,), data_chunks[2]),
            drop_axis=None,
            new_axis=None,
        )
    else:
        result = _linear_interp_vectorized(data_3d, depth_arr, target_arr)

    # Reshape result to match input dimensions
    if len(leading_dims) == 0:
        result = result[0, :, :]  # Remove batch dimension: (ntargets, npoints)
    else:
        result = result.reshape(*leading_dims, ntargets, npoints)

    # Handle coordinate returns
    if lon is not None and lat is not None:
        lon_arr = get_array_data(lon)
        lat_arr = get_array_data(lat)
        if hasattr(lon_arr, "ravel"):
            lon_arr = lon_arr.ravel()
        if hasattr(lat_arr, "ravel"):
            lat_arr = lat_arr.ravel()
        return result, np.asarray(lon_arr), np.asarray(lat_arr)

    return result


def _linear_interp_vectorized(
    data: NDArray,
    depths: NDArray,
    targets: NDArray,
) -> NDArray:
    """Vectorized linear interpolation for depth profiles.

    Parameters
    ----------
    data : ndarray
        Shape (nbatch, nlevels, npoints).
    depths : ndarray
        Shape (nlevels,), must be monotonic.
    targets : ndarray
        Shape (ntargets,).

    Returns
    -------
    ndarray
        Shape (nbatch, ntargets, npoints).
    """
    nbatch, nlevels, npoints = data.shape
    ntargets = len(targets)

    # Check if depths are monotonically increasing or decreasing
    if depths[0] > depths[-1]:
        # Depths are decreasing, flip for interpolation
        depths = depths[::-1]
        data = data[:, ::-1, :]

    # Output array
    result = np.empty((nbatch, ntargets, npoints), dtype=data.dtype)

    for t_idx, target in enumerate(targets):
        # Find bracketing indices
        # np.searchsorted returns index where target would be inserted
        idx_upper = np.searchsorted(depths, target)

        if idx_upper == 0:
            # Target is above/at shallowest level - extrapolate using first two levels
            idx_lower, idx_upper = 0, 1
        elif idx_upper >= nlevels:
            # Target is below deepest level - extrapolate using last two levels
            idx_lower, idx_upper = nlevels - 2, nlevels - 1
        else:
            idx_lower = idx_upper - 1

        # Get bracketing depths and data
        z_lower = depths[idx_lower]
        z_upper = depths[idx_upper]
        data_lower = data[:, idx_lower, :]  # (nbatch, npoints)
        data_upper = data[:, idx_upper, :]  # (nbatch, npoints)

        # Linear interpolation weight
        if z_upper != z_lower:
            weight = (target - z_lower) / (z_upper - z_lower)
        else:
            weight = 0.0

        # Interpolate
        result[:, t_idx, :] = data_lower + weight * (data_upper - data_lower)

    return result
