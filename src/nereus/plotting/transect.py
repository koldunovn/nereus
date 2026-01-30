"""Vertical transect plotting for nereus.

This module provides functions for plotting vertical transects (cross-sections)
of 3D data along arbitrary paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from nereus.core.coordinates import great_circle_path, lonlat_to_cartesian
from nereus.core.grids import extract_coordinates, prepare_coordinates

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def transect(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating] | None = None,
    lat: NDArray[np.floating] | None = None,
    depth: NDArray[np.floating] | None = None,
    start: tuple[float, float] | None = None,
    end: tuple[float, float] | None = None,
    *,
    n_points: int = 100,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    depth_lim: tuple[float, float] | None = None,
    invert_depth: bool = True,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: "Axes | None" = None,
    **kwargs: Any,
) -> tuple["Figure", "Axes"]:
    """Plot vertical transect along a great circle path.

    The function accepts various coordinate formats and automatically transforms
    them to 1D arrays:

    - Both 1D with same size: used directly (no warning)
    - Both 2D with same shape: raveled to 1D
    - Both 1D with different sizes: meshgrid created, then raveled

    A warning is issued whenever coordinate transformations are applied.

    If lon/lat are not provided and data is an xarray DataArray, the function
    will attempt to extract coordinates automatically by looking for common
    coordinate names (lon/lat, longitude/latitude, x/y, etc.).

    Parameters
    ----------
    data : array_like
        Data values with shape (nlevels, npoints) for 2D, or (nlevels, nlat, nlon)
        for 3D regular grids. 3D data is automatically reshaped to 2D.
        If xarray DataArray, coordinates may be extracted automatically.
    lon : array_like, optional
        Longitude coordinates. Can be 1D or 2D array.
        If None, will attempt to extract from data (xarray only).
    lat : array_like, optional
        Latitude coordinates. Can be 1D or 2D array.
        If None, will attempt to extract from data (xarray only).
    depth : array_like
        1D array of depth levels (positive downward).
    start : tuple of float
        Start point (lon, lat).
    end : tuple of float
        End point (lon, lat).
    n_points : int
        Number of points along the transect.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    depth_lim : tuple of float, optional
        Depth/height limits (min, max). If None, uses data range.
    invert_depth : bool
        Whether to invert vertical axis. Default True for ocean (0 at top,
        depth increases downward). Set False for atmosphere (height increases upward).
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
    **kwargs
        Additional arguments passed to pcolormesh.

    Returns
    -------
    fig : Figure
        The matplotlib Figure.
    ax : Axes
        The matplotlib Axes.

    Examples
    --------
    >>> fig, ax = nr.transect(
    ...     temp, mesh.lon, mesh.lat, depth,
    ...     start=(-30, 60), end=(30, 60)
    ... )
    """
    # Extract coordinates from xarray if not provided
    if lon is None or lat is None:
        extracted_lon, extracted_lat = extract_coordinates(data)
        if lon is None:
            lon = extracted_lon
        if lat is None:
            lat = extracted_lat

    # Validate required parameters
    if lon is None or lat is None:
        raise ValueError(
            "lon and lat coordinates are required. Either provide them explicitly "
            "or use an xarray DataArray with recognizable coordinate names "
            "(lon/lat, longitude/latitude, x/y, etc.)."
        )
    if depth is None:
        raise ValueError("depth array is required for transect plots.")
    if start is None or end is None:
        raise ValueError("start and end points are required for transect plots.")

    # Handle xarray DataArray
    if hasattr(data, "values"):
        data = data.values
    data = np.asarray(data)

    # Prepare coordinates: handle various array shapes and validate
    lon_arr, lat_arr = prepare_coordinates(lon, lat)
    depth_arr = np.asarray(depth).ravel()

    # Handle 3D data on regular grids: (depth, lat, lon) -> (depth, lat*lon)
    # This ensures indexing is consistent with the flattened coordinates
    if data.ndim == 3:
        nlevels, nlat, nlon = data.shape
        # Reshape to (nlevels, npoints) where npoints = nlat * nlon
        data = data.reshape(nlevels, -1)

    # Generate transect path
    path_lon, path_lat = great_circle_path(
        start[0], start[1], end[0], end[1], n_points
    )

    # Build KDTree for source coordinates
    source_xyz = np.column_stack(lonlat_to_cartesian(lon_arr, lat_arr))
    tree = cKDTree(source_xyz)

    # Find nearest points along path
    path_xyz = np.column_stack(lonlat_to_cartesian(path_lon, path_lat))
    _, indices = tree.query(path_xyz, k=1)

    # Extract data along path
    if data.ndim == 1:
        # Single level
        transect_data = data[indices].reshape(1, -1)
    else:
        # Multiple levels (nlevels, npoints)
        transect_data = data[:, indices]

    # Compute distance along path (approximate)
    distance = np.zeros(n_points)
    for i in range(1, n_points):
        # Simple euclidean distance on path coordinates for display
        dlat = path_lat[i] - path_lat[i - 1]
        dlon = path_lon[i] - path_lon[i - 1]
        # Approximate km
        distance[i] = distance[i - 1] + np.sqrt(dlat**2 + (dlon * np.cos(np.deg2rad(path_lat[i])))**2) * 111

    # Create figure if needed
    if ax is None:
        if figsize is None:
            figsize = (12, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot
    im = ax.pcolormesh(
        distance,
        depth_arr,
        transect_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        **kwargs,
    )

    # Configure axes
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (m)" if invert_depth else "Height (m)")

    if depth_lim:
        if invert_depth:
            # For ocean: 0 at top, max depth at bottom
            ax.set_ylim(depth_lim[1], depth_lim[0])
        else:
            # For atmosphere: 0 at bottom, max height at top
            ax.set_ylim(depth_lim)
    elif invert_depth:
        ax.invert_yaxis()

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if title:
        ax.set_title(title)

    return fig, ax
