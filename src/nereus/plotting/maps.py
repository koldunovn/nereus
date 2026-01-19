"""2D map plotting for unstructured data.

This module provides functions for plotting unstructured geophysical data
on various map projections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from nereus.plotting.projections import (
    get_data_bounds_for_projection,
    get_projection,
    is_global_projection,
    is_polar_projection,
)
from nereus.regrid.cache import get_cache
from nereus.regrid.interpolator import RegridInterpolator

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure


def plot(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    projection: str | ccrs.Projection = "pc",
    extent: tuple[float, float, float, float] | None = None,
    resolution: float | tuple[int, int] = 1.0,
    interpolator: RegridInterpolator | None = None,
    influence_radius: float = 500_000.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    coastlines: bool = True,
    land: bool = False,
    gridlines: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: "Axes | None" = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> tuple["Figure", "Axes", RegridInterpolator]:
    """Plot 2D map of unstructured data.

    This function regrids unstructured data to a regular grid and plots it
    on a map with the specified projection.

    Parameters
    ----------
    data : array_like
        1D array of data values at unstructured points.
    lon : array_like
        1D array of longitude coordinates.
    lat : array_like
        1D array of latitude coordinates.
    projection : str or Projection
        Map projection. Options: "pc", "rob", "merc", "npstere", "spstere",
        "moll", "ortho", "lcc", or a Cartopy Projection.
    extent : tuple of float, optional
        Map extent (lon_min, lon_max, lat_min, lat_max).
    resolution : float or tuple of int
        Grid resolution for regridding.
    interpolator : RegridInterpolator, optional
        Pre-computed interpolator. If None, one will be created.
    influence_radius : float
        Maximum influence radius in meters for interpolation.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    coastlines : bool
        Whether to draw coastlines.
    land : bool
        Whether to fill land areas.
    gridlines : bool
        Whether to draw gridlines.
    colorbar : bool
        Whether to add a colorbar.
    colorbar_label : str, optional
        Label for the colorbar.
    title : str, optional
        Plot title.
    figsize : tuple of float, optional
        Figure size (width, height) in inches.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    use_cache : bool
        Whether to use the interpolator cache.
    **kwargs
        Additional arguments passed to pcolormesh.

    Returns
    -------
    fig : Figure
        The matplotlib Figure.
    ax : Axes
        The matplotlib Axes (GeoAxes).
    interpolator : RegridInterpolator
        The interpolator used (can be reused).

    Examples
    --------
    >>> fig, ax, interp = nr.plot(temp, mesh.lon, mesh.lat)
    >>> fig, ax, _ = nr.plot(salinity, mesh.lon, mesh.lat, interpolator=interp)
    """
    # Handle xarray DataArray
    data_values = data.values if hasattr(data, "values") else np.asarray(data)
    lon_arr = np.asarray(lon).ravel()
    lat_arr = np.asarray(lat).ravel()

    # Get projection
    proj = get_projection(projection)
    data_crs = ccrs.PlateCarree()

    # Determine data bounds based on projection
    lon_bounds, lat_bounds = get_data_bounds_for_projection(projection, extent)

    # Get or create interpolator
    if interpolator is None:
        if use_cache:
            cache = get_cache()
            interpolator = cache.get_or_create(
                lon_arr,
                lat_arr,
                resolution=resolution,
                influence_radius=influence_radius,
                lon_bounds=lon_bounds,
                lat_bounds=lat_bounds,
            )
        else:
            interpolator = RegridInterpolator(
                source_lon=lon_arr,
                source_lat=lat_arr,
                resolution=resolution,
                influence_radius=influence_radius,
                lon_bounds=lon_bounds,
                lat_bounds=lat_bounds,
            )

    # Regrid data
    regridded = interpolator(data_values)

    # Create figure if needed
    if ax is None:
        if figsize is None:
            # Default figure size based on projection
            if is_polar_projection(projection):
                figsize = (8, 8)
            elif is_global_projection(projection):
                figsize = (12, 6)
            else:
                figsize = (10, 6)
        fig, ax = plt.subplots(
            1, 1,
            figsize=figsize,
            subplot_kw={"projection": proj},
        )
    else:
        fig = ax.get_figure()

    # Set up map
    if is_global_projection(projection):
        ax.set_global()
    elif extent:
        ax.set_extent(extent, crs=data_crs)

    # Add map features
    if land:
        ax.add_feature(
            cfeature.LAND,
            facecolor="lightgray",
            edgecolor="none",
            zorder=0,
        )

    # Plot data
    im = ax.pcolormesh(
        interpolator.target_lon,
        interpolator.target_lat,
        regridded,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=data_crs,
        **kwargs,
    )

    # Add coastlines on top
    if coastlines:
        ax.coastlines(linewidth=0.5, color="black", zorder=2)

    if gridlines:
        ax.gridlines(draw_labels=not is_polar_projection(projection), linewidth=0.5, alpha=0.5)

    # Add colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        elif hasattr(data, "name") and data.name:
            cbar.set_label(data.name)

    if title:
        ax.set_title(title)

    return fig, ax, interpolator


def plot_polar(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    *,
    hemisphere: str = "north",
    lat_limit: float = 50.0,
    **kwargs: Any,
) -> tuple["Figure", "Axes", RegridInterpolator]:
    """Plot data on a polar stereographic projection.

    Convenience function for polar plots with sensible defaults.

    Parameters
    ----------
    data : array_like
        1D array of data values.
    lon : array_like
        1D array of longitude coordinates.
    lat : array_like
        1D array of latitude coordinates.
    hemisphere : {"north", "south"}
        Which hemisphere to plot.
    lat_limit : float
        Latitude limit for the plot (equatorward boundary).
    **kwargs
        Additional arguments passed to plot().

    Returns
    -------
    fig, ax, interpolator
        Same as plot().
    """
    if hemisphere.lower() == "north":
        projection = "npstere"
        extent = (-180, 180, lat_limit, 90)
    else:
        projection = "spstere"
        extent = (-180, 180, -90, -lat_limit)

    return plot(
        data,
        lon,
        lat,
        projection=projection,
        extent=extent,
        **kwargs,
    )
