"""Hovmoller diagram generation for nereus.

This module provides functions for computing and plotting Hovmoller diagrams
(time-depth or time-latitude plots).

The hovmoller function is dask-friendly: if inputs are dask arrays, the result
will be lazy dask arrays for both mode="depth" and mode="latitude".
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from nereus.core.types import get_array_data, is_dask_array

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _lat_bin_chunk(
    data_chunk: NDArray,
    area_1d: NDArray,
    bin_indices: NDArray,
    nlat: int,
) -> NDArray:
    """Process a chunk of time steps for latitude binning.

    Parameters
    ----------
    data_chunk : ndarray
        Data chunk with shape (ntime_chunk, npoints).
    area_1d : ndarray
        Area weights with shape (npoints,).
    bin_indices : ndarray
        Precomputed bin index for each point, shape (npoints,).
    nlat : int
        Number of latitude bins.

    Returns
    -------
    ndarray
        Binned results with shape (ntime_chunk, nlat).
    """
    ntime_chunk = data_chunk.shape[0]
    result = np.full((ntime_chunk, nlat), np.nan)

    for i in range(nlat):
        in_bin = bin_indices == i
        if not np.any(in_bin):
            continue

        # Extract data for this bin
        bin_data = data_chunk[:, in_bin]  # (ntime_chunk, npoints_in_bin)
        bin_area = area_1d[in_bin]

        # Compute valid mask
        valid = np.isfinite(bin_data)
        valid_area = np.where(valid, bin_area[np.newaxis, :], 0.0)
        total_area = np.sum(valid_area, axis=1)

        # Weighted sum
        data_filled = np.where(valid, bin_data, 0.0)
        weighted_sum = np.sum(data_filled * valid_area, axis=1)

        # Mean
        with np.errstate(divide="ignore", invalid="ignore"):
            result[:, i] = np.where(total_area > 0, weighted_sum / total_area, np.nan)

    return result


def hovmoller(
    data: NDArray | "xr.DataArray",
    area: NDArray[np.floating],
    time: NDArray | None = None,
    depth: NDArray[np.floating] | None = None,
    lat: NDArray[np.floating] | None = None,
    *,
    mode: Literal["depth", "latitude"] = "depth",
    lat_bins: NDArray[np.floating] | None = None,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute Hovmoller diagram data.

    Computes area-weighted means at each time step, binned by either depth
    level or latitude.

    This function is dask-friendly: if inputs are dask arrays, the result
    will be lazy dask arrays.

    Parameters
    ----------
    data : array_like
        Data array. For depth mode: shape (ntime, nlevels, npoints).
        For latitude mode: shape (ntime, npoints) or (ntime, nlevels, npoints).
    area : array_like
        Grid cell areas in m^2. Can be either:
        - 1D array of shape (npoints,) for surface area (uniform across depth)
        - 2D array of shape (nlevels, npoints) for depth-dependent area
        If 2D and has one extra level compared to data layers, the extra
        level is dropped with a warning (levels vs layers).
    time : array_like, optional
        Time coordinates. If None, uses integer indices.
    depth : array_like, optional
        Depth levels in meters. Required for mode="depth".
    lat : array_like, optional
        Latitude coordinates in degrees. Required for mode="latitude".
    mode : {"depth", "latitude"}
        Type of Hovmoller diagram.
    lat_bins : array_like, optional
        Latitude bin edges for mode="latitude". Default is 5-degree bins.
    mask : array_like, optional
        Boolean mask for horizontal points, shape (npoints,). True = include.

    Returns
    -------
    time_out : ndarray
        Time coordinates.
    y_out : ndarray
        Depth or latitude coordinates.
    data_out : ndarray or dask.array
        Hovmoller data array, shape (ntime, ny). For mode="depth" with dask
        input, returns a dask array.

    Examples
    --------
    >>> # Time-depth Hovmoller
    >>> t, z, hov = nr.hovmoller(temp, mesh.area, time, depth, mode="depth")

    >>> # Time-latitude Hovmoller
    >>> t, lat, hov = nr.hovmoller(sst, mesh.area, time, lat=mesh.lat, mode="latitude")

    >>> # With dask arrays (depth mode)
    >>> t, z, hov = nr.hovmoller(temp_dask, mesh.area, time, depth, mode="depth")
    >>> hov.compute()  # triggers actual computation
    """
    # Extract arrays, preserving dask for depth mode
    data_arr = get_array_data(data)
    area_arr = get_array_data(area)
    is_lazy = is_dask_array(data)

    # Apply mask to horizontal points
    if mask is not None:
        horiz_mask = get_array_data(mask)
        if hasattr(horiz_mask, "ravel"):
            horiz_mask = horiz_mask.ravel()
        else:
            horiz_mask = np.asarray(horiz_mask).ravel()
    else:
        horiz_mask = None

    if mode == "depth":
        if depth is None:
            raise ValueError("depth array required for mode='depth'")
        depth_arr = np.asarray(get_array_data(depth)).ravel()

        # Expect data shape: (ntime, nlevels, npoints)
        if data_arr.ndim == 2:
            # Assume (nlevels, npoints) - single timestep
            data_arr = data_arr[np.newaxis, :, :]

        ntime, nlevels, npoints = data_arr.shape

        # Handle area: can be 1D (npoints,) or 2D (nlevels, npoints)
        if area_arr.ndim == 1:
            if hasattr(area_arr, "ravel"):
                area_arr = area_arr.ravel()
            if area_arr.shape[0] != npoints:
                raise ValueError(
                    f"area has {area_arr.shape[0]} points but data has {npoints}"
                )
            area_is_2d = False
        elif area_arr.ndim == 2:
            nlev_area = area_arr.shape[0]
            area_is_2d = True
            # Check if area has one extra level (levels vs layers mismatch)
            if nlev_area != nlevels:
                diff = nlev_area - nlevels
                if diff != 1:
                    raise ValueError(
                        f"area has {nlev_area} vertical levels but data has {nlevels}; "
                        "only area having one extra level is supported (levels vs layers)."
                    )
                warnings.warn(
                    f"area has one more vertical level than data; "
                    f"using the first {nlevels} levels of area to match data "
                    "(levels vs layers).",
                    UserWarning,
                    stacklevel=2,
                )
                area_arr = area_arr[:nlevels, :]
        else:
            raise ValueError(f"area must be 1D or 2D, got {area_arr.ndim}D")

        # Apply horizontal mask if provided
        if horiz_mask is not None:
            horiz_mask_float = horiz_mask.astype(np.float64)
            if area_is_2d:
                area_arr = area_arr * horiz_mask_float[np.newaxis, :]
            else:
                area_arr = area_arr * horiz_mask_float

        # Compute area-weighted mean at each depth level for each time
        # Vectorized approach for dask compatibility
        # data_arr shape: (ntime, nlevels, npoints)
        # area_arr shape: (npoints,) or (nlevels, npoints)

        # Get valid mask
        valid = np.isfinite(data_arr)

        # Prepare area for broadcasting
        if area_is_2d:
            # area_arr: (nlevels, npoints) -> (1, nlevels, npoints)
            area_broadcast = area_arr[np.newaxis, :, :]
        else:
            # area_arr: (npoints,) -> (1, 1, npoints)
            area_broadcast = area_arr[np.newaxis, np.newaxis, :]

        # Compute valid area (zero where data is NaN)
        valid_area = np.where(valid, area_broadcast, 0.0)

        # Replace NaN with 0 for summation
        data_filled = np.where(valid, data_arr, 0.0)

        # Sum over points (last axis)
        weighted_sum = np.sum(data_filled * valid_area, axis=-1)  # (ntime, nlevels)
        total_area = np.sum(valid_area, axis=-1)  # (ntime, nlevels)

        # Compute mean, handling zero area
        result = np.where(total_area > 0, weighted_sum / total_area, np.nan)

        # Time array
        if time is None:
            time_out = np.arange(ntime)
        else:
            time_out = np.asarray(get_array_data(time))

        return time_out, depth_arr, result

    elif mode == "latitude":
        if lat is None:
            raise ValueError("lat array required for mode='latitude'")

        # Get lat array as numpy (small array, safe to compute)
        lat_arr = np.asarray(get_array_data(lat)).ravel()

        # Set up latitude bins
        if lat_bins is None:
            lat_bins = np.arange(-90, 95, 5)  # 5-degree bins
        lat_bins = np.asarray(lat_bins)
        lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
        nlat = len(lat_centers)

        # Precompute bin indices for each point (small numpy array)
        # digitize returns 1-based indices for bins, so subtract 1
        # Points outside the range get clipped to valid bin indices
        bin_indices = np.digitize(lat_arr, lat_bins) - 1
        bin_indices = np.clip(bin_indices, 0, nlat - 1)

        # Handle data shape - need to work with original array (possibly dask)
        if data_arr.ndim == 1:
            # Single timestep, single level (npoints,)
            data_arr = data_arr[np.newaxis, np.newaxis, :]
        elif data_arr.ndim == 2:
            # Could be (ntime, npoints) or (nlevels, npoints)
            # Assume (ntime, npoints) for latitude mode
            data_arr = data_arr[:, np.newaxis, :]

        ntime = data_arr.shape[0]
        npoints = data_arr.shape[-1]

        # For latitude mode, use surface area (first level if 2D)
        # Area needs to be numpy for the binning function
        if is_dask_array(area_arr):
            area_arr = np.asarray(area_arr)
        if area_arr.ndim == 2:
            area_1d = area_arr[0, :]
        else:
            area_1d = area_arr.ravel()

        # Apply horizontal mask if provided
        if horiz_mask is not None:
            area_1d = np.where(horiz_mask, area_1d, 0.0)

        # Vertically integrate first if 3D
        if data_arr.shape[1] > 1:
            # Vertical mean (simple average across levels)
            if is_lazy:
                import dask.array as da

                data_2d = da.nanmean(data_arr, axis=1)
            else:
                data_2d = np.nanmean(data_arr, axis=1)
        else:
            data_2d = data_arr[:, 0, :]

        # Process using dask or numpy
        if is_lazy:
            import dask.array as da

            # Try to use distributed client.scatter to avoid graph bloat
            # This sends the arrays to workers once, then tasks reference by future
            use_scatter = False
            try:
                from distributed import get_client

                client = get_client()
                # Scatter arrays to all workers (broadcast=True)
                area_future = client.scatter(area_1d, broadcast=True)
                bin_future = client.scatter(bin_indices, broadcast=True)
                use_scatter = True
            except (ImportError, ValueError):
                # No distributed client available, use regular approach
                pass

            if use_scatter:
                # With scattered data, use map_blocks with futures
                result = da.map_blocks(
                    _lat_bin_chunk,
                    data_2d,
                    area_1d=area_future,
                    bin_indices=bin_future,
                    nlat=nlat,
                    dtype=np.float64,
                    drop_axis=1,
                    new_axis=1,
                    chunks=(data_2d.chunks[0], (nlat,)),
                )
            else:
                # Fallback: use map_blocks with embedded arrays
                # This works fine for local schedulers
                result = da.map_blocks(
                    _lat_bin_chunk,
                    data_2d,
                    area_1d=area_1d,
                    bin_indices=bin_indices,
                    nlat=nlat,
                    dtype=np.float64,
                    drop_axis=1,
                    new_axis=1,
                    chunks=(data_2d.chunks[0], (nlat,)),
                )
        else:
            # Numpy path
            data_2d = np.asarray(data_2d)
            result = _lat_bin_chunk(data_2d, area_1d, bin_indices, nlat)

        # Time array
        if time is None:
            time_out = np.arange(ntime)
        else:
            time_out = np.asarray(get_array_data(time))

        return time_out, lat_centers, result

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'depth' or 'latitude'.")


def plot_hovmoller(
    time: NDArray,
    y: NDArray,
    data: NDArray,
    *,
    mode: Literal["depth", "latitude"] = "depth",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: "Axes | None" = None,
    invert_y: bool | None = None,
    anomaly: bool = False,
    log_y: bool = False,
    **kwargs: Any,
) -> tuple["Figure", "Axes"]:
    """Plot a Hovmoller diagram.

    Parameters
    ----------
    time : array_like
        Time coordinates.
    y : array_like
        Depth or latitude coordinates.
    data : array_like
        Hovmoller data, shape (ntime, ny).
    mode : {"depth", "latitude"}
        Type of diagram (affects axis labels and orientation).
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    colorbar : bool
        Whether to add a colorbar.
    colorbar_label : str, optional
        Label for the colorbar.
    title : str, optional
        Plot title.
    figsize : tuple of float, optional
        Figure size.
    ax : Axes, optional
        Existing axes to plot on.
    invert_y : bool, optional
        Whether to invert y-axis. Default True for depth, False for latitude.
    anomaly : bool
        If True and mode="depth", plot anomaly relative to first time step
        (data - data[0, :]). Default False.
    log_y : bool
        If True and mode="depth", use logarithmic scale for vertical axis.
        Useful for highlighting surface layers. Default False.
    **kwargs
        Additional arguments passed to pcolormesh.

    Returns
    -------
    fig : Figure
        The matplotlib Figure.
    ax : Axes
        The matplotlib Axes.
    """
    time = np.asarray(time)
    y = np.asarray(y)
    data = np.asarray(data)

    # Validate dimensions
    ntime_data, ny_data = data.shape
    if len(time) != ntime_data:
        raise ValueError(
            f"time array has {len(time)} elements but data has {ntime_data} "
            f"time steps (data shape: {data.shape})"
        )
    if len(y) != ny_data:
        raise ValueError(
            f"y array has {len(y)} elements but data has {ny_data} "
            f"y values (data shape: {data.shape}). "
            f"Make sure you're using the y coordinates returned by hovmoller(), "
            f"not the original mesh coordinates."
        )

    # Compute anomaly if requested (only for depth mode)
    if anomaly and mode == "depth":
        data = data - data[0, :]

    # Create figure if needed
    if ax is None:
        if figsize is None:
            figsize = (12, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot
    # Use shading='nearest' to match coordinate arrays with data dimensions
    im = ax.pcolormesh(
        time,
        y,
        data.T,  # Transpose so y is on vertical axis
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        **kwargs,
    )

    # Axis labels
    ax.set_xlabel("Time")
    if mode == "depth":
        ax.set_ylabel("Depth (m)")
        if invert_y is None:
            invert_y = True
    else:
        ax.set_ylabel("Latitude (°)")
        if invert_y is None:
            invert_y = False

    if invert_y:
        ax.invert_yaxis()

    # Logarithmic y-axis for depth mode
    if log_y and mode == "depth":
        ax.set_yscale("log")

    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if title:
        ax.set_title(title)

    return fig, ax
