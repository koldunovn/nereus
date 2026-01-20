"""Hovmoller diagram generation for nereus.

This module provides functions for computing and plotting Hovmoller diagrams
(time-depth or time-latitude plots).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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

    Parameters
    ----------
    data : array_like
        Data array. For depth mode: shape (ntime, nlevels, npoints).
        For latitude mode: shape (ntime, npoints) or (ntime, nlevels, npoints).
    area : array_like
        Grid cell areas in m^2, shape (npoints,).
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
    data_out : ndarray
        Hovmoller data array, shape (ntime, ny).

    Examples
    --------
    >>> # Time-depth Hovmoller
    >>> t, z, hov = nr.hovmoller(temp, mesh.area, time, depth, mode="depth")

    >>> # Time-latitude Hovmoller
    >>> t, lat, hov = nr.hovmoller(sst, mesh.area, time, lat=mesh.lat, mode="latitude")
    """
    # Handle xarray DataArray
    if hasattr(data, "values"):
        data = data.values
    data_arr = np.asarray(data)
    area_arr = np.asarray(area).ravel()
    npoints = area_arr.shape[0]

    # Apply mask
    if mask is not None:
        mask = np.asarray(mask).ravel()
        area_arr = np.where(mask, area_arr, 0.0)

    if mode == "depth":
        if depth is None:
            raise ValueError("depth array required for mode='depth'")
        depth_arr = np.asarray(depth).ravel()

        # Expect data shape: (ntime, nlevels, npoints)
        if data_arr.ndim == 2:
            # Assume (nlevels, npoints) - single timestep
            data_arr = data_arr[np.newaxis, :, :]

        ntime, nlevels, _ = data_arr.shape

        # Compute area-weighted mean at each depth level for each time
        result = np.zeros((ntime, nlevels))
        for t in range(ntime):
            for k in range(nlevels):
                layer_data = data_arr[t, k, :]
                valid = np.isfinite(layer_data)
                valid_area = np.where(valid, area_arr, 0.0)
                total_area = np.sum(valid_area)
                if total_area > 0:
                    result[t, k] = np.nansum(layer_data * valid_area) / total_area
                else:
                    result[t, k] = np.nan

        # Time array
        if time is None:
            time_out = np.arange(ntime)
        else:
            time_out = np.asarray(time)

        return time_out, depth_arr, result

    elif mode == "latitude":
        if lat is None:
            raise ValueError("lat array required for mode='latitude'")
        lat_arr = np.asarray(lat).ravel()

        # Set up latitude bins
        if lat_bins is None:
            lat_bins = np.arange(-90, 95, 5)  # 5-degree bins
        lat_bins = np.asarray(lat_bins)
        lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
        nlat = len(lat_centers)

        # Handle data shape
        if data_arr.ndim == 1:
            # Single timestep, single level (npoints,)
            data_arr = data_arr[np.newaxis, np.newaxis, :]
        elif data_arr.ndim == 2:
            # Could be (ntime, npoints) or (nlevels, npoints)
            # Assume (ntime, npoints) for latitude mode
            data_arr = data_arr[:, np.newaxis, :]

        ntime = data_arr.shape[0]

        # Compute area-weighted mean in each latitude bin
        # Vertically integrate first if 3D
        if data_arr.shape[1] > 1:
            # Vertical mean (simple average across levels)
            data_2d = np.nanmean(data_arr, axis=1)
        else:
            data_2d = data_arr[:, 0, :]

        result = np.zeros((ntime, nlat))
        for t in range(ntime):
            for i in range(nlat):
                in_bin = (lat_arr >= lat_bins[i]) & (lat_arr < lat_bins[i + 1])
                layer_data = data_2d[t, in_bin]
                bin_area = area_arr[in_bin]
                valid = np.isfinite(layer_data)
                valid_area = np.where(valid, bin_area, 0.0)
                total_area = np.sum(valid_area)
                if total_area > 0:
                    result[t, i] = np.nansum(layer_data * valid_area) / total_area
                else:
                    result[t, i] = np.nan

        # Time array
        if time is None:
            time_out = np.arange(ntime)
        else:
            time_out = np.asarray(time)

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

    # Create figure if needed
    if ax is None:
        if figsize is None:
            figsize = (12, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot
    im = ax.pcolormesh(
        time,
        y,
        data.T,  # Transpose so y is on vertical axis
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
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

    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if title:
        ax.set_title(title)

    return fig, ax
