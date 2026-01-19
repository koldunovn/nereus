"""Regridding interpolator for unstructured to regular grid conversion.

This module provides the RegridInterpolator class for efficiently regridding
unstructured data (like FESOM, ICON) to regular lat/lon grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from nereus.core.coordinates import lonlat_to_cartesian, meters_to_chord
from nereus.core.grids import create_regular_grid

if TYPE_CHECKING:
    import xarray as xr


@dataclass
class RegridInterpolator:
    """Pre-computed interpolation for fast repeated regridding.

    This class computes and stores interpolation weights for regridding
    unstructured data to a regular grid. The computation is done once
    during initialization, allowing fast repeated application.

    Parameters
    ----------
    source_lon : array_like
        Source grid longitude coordinates in degrees.
    source_lat : array_like
        Source grid latitude coordinates in degrees.
    resolution : float or tuple of int
        Target grid resolution. If float, specifies degrees per cell.
        If tuple (nlon, nlat), specifies number of grid points.
    method : {"nearest"}
        Interpolation method. Currently only "nearest" is supported.
    influence_radius : float
        Maximum influence radius in meters. Points beyond this distance
        from any source point are masked. Default is 500 km.
    lon_bounds : tuple of float
        Target grid longitude bounds. Default is (-180, 180).
    lat_bounds : tuple of float
        Target grid latitude bounds. Default is (-90, 90).

    Attributes
    ----------
    target_lon : ndarray
        Target grid longitude coordinates (2D).
    target_lat : ndarray
        Target grid latitude coordinates (2D).
    indices : ndarray
        Source indices for each target point.
    distances : ndarray
        Distances from target to source points (in chord units).
    valid_mask : ndarray
        Boolean mask of valid target points within influence radius.

    Examples
    --------
    >>> interpolator = RegridInterpolator(mesh_lon, mesh_lat, resolution=1.0)
    >>> regridded = interpolator(data)
    >>> regridded.shape
    (180, 360)
    """

    source_lon: NDArray[np.floating]
    source_lat: NDArray[np.floating]
    resolution: float | tuple[int, int] = 1.0
    method: Literal["nearest"] = "nearest"
    influence_radius: float = 500_000.0
    lon_bounds: tuple[float, float] = (-180.0, 180.0)
    lat_bounds: tuple[float, float] = (-90.0, 90.0)

    # Computed attributes (initialized in __post_init__)
    target_lon: NDArray[np.floating] = field(init=False, repr=False)
    target_lat: NDArray[np.floating] = field(init=False, repr=False)
    indices: NDArray[np.intp] = field(init=False, repr=False)
    distances: NDArray[np.floating] = field(init=False, repr=False)
    valid_mask: NDArray[np.bool_] = field(init=False, repr=False)
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize interpolation weights."""
        # Ensure source coordinates are 1D numpy arrays
        self.source_lon = np.asarray(self.source_lon).ravel()
        self.source_lat = np.asarray(self.source_lat).ravel()

        # Create target grid
        self.target_lon, self.target_lat = create_regular_grid(
            self.resolution,
            lon_bounds=self.lon_bounds,
            lat_bounds=self.lat_bounds,
        )

        # Convert source coordinates to Cartesian (unit sphere)
        source_xyz = np.column_stack(
            lonlat_to_cartesian(self.source_lon, self.source_lat)
        )

        # Build KDTree
        self._tree = cKDTree(source_xyz)

        # Convert target coordinates to Cartesian
        target_xyz = np.column_stack(
            lonlat_to_cartesian(self.target_lon.ravel(), self.target_lat.ravel())
        )

        # Query nearest neighbors
        self.distances, self.indices = self._tree.query(target_xyz, k=1)

        # Reshape to target grid shape
        self.distances = self.distances.reshape(self.target_lon.shape)
        self.indices = self.indices.reshape(self.target_lon.shape)

        # Create valid mask based on influence radius
        max_chord = meters_to_chord(self.influence_radius)
        self.valid_mask = self.distances <= max_chord

    def __call__(
        self,
        data: NDArray | "xr.DataArray",
        fill_value: float = np.nan,
    ) -> NDArray[np.floating]:
        """Apply interpolation to data.

        Parameters
        ----------
        data : array_like
            Data to interpolate. Can be:
            - 1D array of shape (npoints,)
            - 2D array of shape (nlevels, npoints) or (ntime, npoints)
            - ND array with last axis = npoints
        fill_value : float
            Value for invalid points outside influence radius.

        Returns
        -------
        ndarray
            Regridded data. Shape depends on input:
            - 1D input: (nlat, nlon)
            - 2D input: (extra_dim, nlat, nlon)
            - ND input: (*leading_dims, nlat, nlon)
        """
        # Handle xarray DataArray
        if hasattr(data, "values"):
            data = data.values

        data = np.asarray(data)
        target_shape = self.target_lon.shape

        # Handle different input dimensions
        if data.ndim == 1:
            # Simple 1D case
            result = self._interpolate_1d(data, fill_value)
        elif data.ndim == 2:
            # 2D case: (extra_dim, npoints)
            n_extra = data.shape[0]
            result = np.empty((n_extra,) + target_shape, dtype=data.dtype)
            for i in range(n_extra):
                result[i] = self._interpolate_1d(data[i], fill_value)
        else:
            # ND case: (*leading_dims, npoints)
            leading_shape = data.shape[:-1]
            npoints = data.shape[-1]
            data_flat = data.reshape(-1, npoints)
            result_flat = np.empty((data_flat.shape[0],) + target_shape, dtype=data.dtype)
            for i in range(data_flat.shape[0]):
                result_flat[i] = self._interpolate_1d(data_flat[i], fill_value)
            result = result_flat.reshape(leading_shape + target_shape)

        return result

    def _interpolate_1d(
        self,
        data: NDArray[np.floating],
        fill_value: float,
    ) -> NDArray[np.floating]:
        """Interpolate 1D data array."""
        # Get values at nearest source points
        result = data[self.indices]

        # Apply mask for points outside influence radius
        if not np.isnan(fill_value):
            result = result.astype(np.float64)
        result[~self.valid_mask] = fill_value

        return result

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of target grid (nlat, nlon)."""
        return self.target_lon.shape


def regrid(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    resolution: float | tuple[int, int] = 1.0,
    method: Literal["nearest"] = "nearest",
    influence_radius: float = 500_000.0,
    fill_value: float = np.nan,
    lon_bounds: tuple[float, float] = (-180.0, 180.0),
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
) -> tuple[NDArray[np.floating], RegridInterpolator]:
    """Regrid unstructured data to regular grid.

    This is a convenience function that creates a RegridInterpolator and
    applies it. For repeated regridding with the same source grid, create
    a RegridInterpolator once and reuse it.

    Parameters
    ----------
    data : array_like
        Data to interpolate.
    lon : array_like
        Source grid longitude coordinates.
    lat : array_like
        Source grid latitude coordinates.
    resolution : float or tuple of int
        Target grid resolution.
    method : {"nearest"}
        Interpolation method.
    influence_radius : float
        Maximum influence radius in meters.
    fill_value : float
        Value for invalid points.
    lon_bounds : tuple of float
        Target grid longitude bounds.
    lat_bounds : tuple of float
        Target grid latitude bounds.

    Returns
    -------
    regridded : ndarray
        Regridded data.
    interpolator : RegridInterpolator
        The interpolator used (can be reused for other variables).
    """
    interpolator = RegridInterpolator(
        source_lon=lon,
        source_lat=lat,
        resolution=resolution,
        method=method,
        influence_radius=influence_radius,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
    )

    regridded = interpolator(data, fill_value=fill_value)

    return regridded, interpolator
