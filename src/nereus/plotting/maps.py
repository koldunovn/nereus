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

from nereus.core.grids import extract_coordinates, prepare_input_arrays
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
    lon: NDArray[np.floating] | None = None,
    lat: NDArray[np.floating] | None = None,
    *,
    projection: str | ccrs.Projection = "pc",
    extent: tuple[float, float, float, float] | None = None,
    resolution: float | tuple[int, int] = 1.0,
    interpolator: RegridInterpolator | None = None,
    influence_radius: float = 80_000.0,
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

    The function accepts various input formats and automatically transforms
    them to 1D arrays for plotting:

    - All 1D arrays of same size: used directly (no warning)
    - 2D data with 2D lon/lat (same shape): all raveled to 1D
    - 1D data with 2D lon/lat: lon/lat raveled to match data
    - 2D data with 1D lon/lat: meshgrid created, then all raveled

    A warning is issued whenever array transformations are applied.

    If lon/lat are not provided and data is an xarray DataArray, the function
    will attempt to extract coordinates automatically by looking for common
    coordinate names (lon/lat, longitude/latitude, x/y, etc.).

    Parameters
    ----------
    data : array_like
        Data values at unstructured points. Can be 1D or 2D array.
        If xarray DataArray, coordinates may be extracted automatically.
    lon : array_like, optional
        Longitude coordinates. Can be 1D or 2D array.
        If None, will attempt to extract from data (xarray only).
    lat : array_like, optional
        Latitude coordinates. Can be 1D or 2D array.
        If None, will attempt to extract from data (xarray only).
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
        Maximum influence radius in meters for interpolation. Default is 80 km.
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
    # Extract coordinates from xarray if not provided
    if lon is None or lat is None:
        extracted_lon, extracted_lat = extract_coordinates(data)
        if lon is None:
            lon = extracted_lon
        if lat is None:
            lat = extracted_lat

    # Validate that we have coordinates
    if lon is None or lat is None:
        raise ValueError(
            "lon and lat coordinates are required. Either provide them explicitly "
            "or use an xarray DataArray with recognizable coordinate names "
            "(lon/lat, longitude/latitude, x/y, etc.)."
        )

    # Prepare inputs: handle various array shapes and validate
    data_values, lon_arr, lat_arr = prepare_input_arrays(data, lon, lat)

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
            zorder=1,
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
        zorder=0,
        **kwargs,
    )

    # Add coastlines on top
    if coastlines:
        ax.coastlines(linewidth=0.5, color="black", zorder=2)

    if gridlines:
        ax.gridlines(draw_labels=not is_polar_projection(projection), linewidth=0.5, alpha=0.5)

    # Add colorbar (horizontal at bottom)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        elif hasattr(data, "name") and data.name:
            cbar.set_label(data.name)

    if title:
        ax.set_title(title)

    return fig, ax, interpolator
