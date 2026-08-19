"""Regridding interpolator for unstructured to regular grid conversion.

This module provides the RegridInterpolator class for efficiently regridding
unstructured data (like FESOM, ICON) to regular lat/lon grids.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from nereus.core.coordinates import (
    lonlat_to_cartesian,
    meters_to_chord,
    normalize_longitude,
)
from nereus.core.grids import (
    create_regular_grid,
    extract_coordinates,
    flatten_spatial,
    grid_cell_area,
    prepare_coordinates,
)
from nereus.regrid.conservative import (
    build_conservative_weights,
    build_grid_box_cells,
    build_voronoi_cells,
    geometry_spherical_area,
)

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
    method : {"nearest", "idw", "linear", "cubic", "conservative"}
        Interpolation method. "nearest" uses nearest-neighbor lookup via
        KDTree (fast). "idw" uses inverse distance weighting with 8
        nearest neighbors (fast, smooth). "linear" uses Delaunay
        triangulation with barycentric interpolation (slower but
        smoother). "cubic" uses Clough-Tocher C1 interpolation for the
        smoothest results. "conservative" computes area-weighted overlap
        between source and target cell polygons (derived via a spherical
        Voronoi tessellation) so that an integrated quantity (e.g. total
        heat or mass) is preserved; it is the slowest method and its
        ``__call__`` accepts ``return_fraction=True`` to also get the
        fraction of each target cell actually covered by valid source
        data. Source longitudes are automatically normalized to match the
        target grid's ``lon_bounds`` so that any input convention (0-360
        or -180-180) works transparently.
    influence_radius : float
        Maximum influence radius in meters. Points beyond this distance
        from any source point are masked. Default is 80 km. Not used by
        "conservative" (coverage is determined by polygon overlap instead).
    lon_bounds : tuple of float
        Target grid longitude bounds. Default is (-180, 180). Ignored when
        ``target_lon``/``target_lat`` are provided.
    lat_bounds : tuple of float
        Target grid latitude bounds. Default is (-90, 90). Ignored when
        ``target_lon``/``target_lat`` are provided.
    target_lon : array_like, optional
        Arbitrary unstructured target point longitudes in degrees. Only
        valid with ``method="conservative"``; when provided (together with
        ``target_lat``) the target is this point cloud instead of a
        regular grid, enabling mesh-to-mesh conservative remapping.
    target_lat : array_like, optional
        Arbitrary unstructured target point latitudes in degrees. See
        ``target_lon``.

    Attributes
    ----------
    target_lon : ndarray
        Target coordinates. 2D (nlat, nlon) for a regular grid, or 1D
        (n_target,) when ``target_lon``/``target_lat`` were supplied.
    target_lat : ndarray
        Target coordinates, see ``target_lon``.
    indices : ndarray
        Source indices for each target point. Not set for "conservative".
    distances : ndarray
        Distances from target to source points (in chord units). Not set
        for "conservative".
    valid_mask : ndarray
        Boolean mask of valid target points within influence radius. Not
        set for "conservative" (see ``return_fraction`` on ``__call__``).

    Examples
    --------
    >>> interpolator = RegridInterpolator(mesh_lon, mesh_lat, resolution=1.0)
    >>> regridded = interpolator(data)
    >>> regridded.shape
    (180, 360)

    Use linear interpolation for smoother results:

    >>> interpolator = RegridInterpolator(
    ...     mesh_lon, mesh_lat, resolution=1.0, method="linear"
    ... )

    Conservative remapping onto a regular grid, with coverage fraction:

    >>> interpolator = RegridInterpolator(
    ...     mesh_lon, mesh_lat, resolution=1.0, method="conservative"
    ... )
    >>> regridded, valid_fraction = interpolator(data, return_fraction=True)

    Conservative remapping onto another unstructured mesh:

    >>> interpolator = RegridInterpolator(
    ...     mesh_lon, mesh_lat, method="conservative",
    ...     target_lon=other_mesh_lon, target_lat=other_mesh_lat,
    ... )
    """

    source_lon: NDArray[np.floating]
    source_lat: NDArray[np.floating]
    resolution: float | tuple[int, int] = 1.0
    method: Literal["nearest", "idw", "linear", "cubic", "conservative"] = "nearest"
    influence_radius: float = 80_000.0
    lon_bounds: tuple[float, float] = (-180.0, 180.0)
    lat_bounds: tuple[float, float] = (-90.0, 90.0)
    target_lon: NDArray[np.floating] | None = None
    target_lat: NDArray[np.floating] | None = None

    # Computed attributes (initialized in __post_init__)
    indices: NDArray[np.intp] = field(init=False, repr=False)
    distances: NDArray[np.floating] = field(init=False, repr=False)
    valid_mask: NDArray[np.bool_] = field(init=False, repr=False)
    _tree: cKDTree | None = field(init=False, repr=False, default=None)
    _delaunay: Any = field(init=False, repr=False, default=None)
    _source_2d: NDArray[np.floating] | None = field(
        init=False, repr=False, default=None
    )
    _idw_weights: NDArray[np.floating] | None = field(
        init=False, repr=False, default=None
    )
    _idw_indices: NDArray[np.intp] | None = field(
        init=False, repr=False, default=None
    )
    _target_is_grid: bool = field(init=False, repr=False, default=True)
    _weights: Any = field(init=False, repr=False, default=None)
    _target_area: NDArray[np.floating] | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self) -> None:
        """Initialize interpolation weights."""
        # Prepare source coordinates: handle 1D/2D and validate
        self.source_lon, self.source_lat = prepare_coordinates(
            self.source_lon, self.source_lat
        )

        has_target_lon = self.target_lon is not None
        has_target_lat = self.target_lat is not None
        if has_target_lon != has_target_lat:
            raise ValueError(
                "target_lon and target_lat must both be provided together."
            )
        if has_target_lon and self.method != "conservative":
            raise ValueError(
                "target_lon/target_lat (arbitrary unstructured target) are "
                "only supported for method='conservative'."
            )

        if has_target_lon:
            # Arbitrary unstructured target point cloud
            assert self.target_lon is not None and self.target_lat is not None
            self.target_lon, self.target_lat = prepare_coordinates(
                self.target_lon, self.target_lat
            )
            self._target_is_grid = False
        else:
            # Create regular target grid
            self.target_lon, self.target_lat = create_regular_grid(
                self.resolution,
                lon_bounds=self.lon_bounds,
                lat_bounds=self.lat_bounds,
            )
            self._target_is_grid = True

        # target_lon/target_lat are always concrete arrays from this point on.
        assert self.target_lon is not None and self.target_lat is not None

        if self.method == "conservative":
            self._build_conservative_weights()
            return

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

        # Pre-compute IDW weights
        if self.method == "idw":
            k = 8
            dists, idxs = self._tree.query(target_xyz, k=k)
            target_shape = self.target_lon.shape
            dists = dists.reshape(target_shape + (k,))
            idxs = idxs.reshape(target_shape + (k,))

            # Inverse distance squared weights
            # Handle exact matches (distance == 0)
            exact = dists == 0.0
            has_exact = exact.any(axis=-1)

            weights = np.zeros_like(dists)
            # For points with an exact match, set weight=1 for first exact neighbor
            weights[has_exact] = 0.0
            first_exact = exact & (np.cumsum(exact, axis=-1) == 1)
            weights[first_exact] = 1.0
            # For points without exact match, use 1/d^2
            with np.errstate(divide="ignore"):
                inv_d2 = np.where(
                    ~has_exact[..., np.newaxis], 1.0 / dists**2, weights
                )
            weights = np.where(has_exact[..., np.newaxis], weights, inv_d2)
            # Normalize so weights sum to 1
            weight_sum = weights.sum(axis=-1, keepdims=True)
            weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
            weights = weights / weight_sum

            self._idw_weights = weights
            self._idw_indices = idxs

        # Build Delaunay triangulation for linear/cubic interpolation
        if self.method in ("linear", "cubic"):
            from scipy.spatial import Delaunay

            # Normalize source longitudes to match the target grid range
            # so that e.g. 0-360 source works with -180-180 target.
            lon_center = (self.lon_bounds[0] + self.lon_bounds[1]) / 2
            source_lon_norm = normalize_longitude(self.source_lon, lon_center)

            self._source_2d = np.column_stack([source_lon_norm, self.source_lat])
            self._delaunay = Delaunay(self._source_2d)

    def _build_conservative_weights(self) -> None:
        """Build the area-overlap weight matrix for conservative remapping."""
        assert self.target_lon is not None and self.target_lat is not None
        source_polys = build_voronoi_cells(self.source_lon, self.source_lat)

        if self._target_is_grid:
            target_polys = build_grid_box_cells(
                self.resolution,
                lon_bounds=self.lon_bounds,
                lat_bounds=self.lat_bounds,
            )
            self._target_area = grid_cell_area(
                self.target_lon, self.target_lat
            ).ravel()
        else:
            target_polys = build_voronoi_cells(self.target_lon, self.target_lat)
            self._target_area = np.array(
                [geometry_spherical_area(poly) for poly in target_polys]
            )

        self._weights = build_conservative_weights(source_polys, target_polys)

    @overload
    def __call__(
        self,
        data: NDArray | "xr.DataArray",
        fill_value: float = ...,
        return_fraction: Literal[False] = ...,
    ) -> NDArray[np.floating]: ...

    @overload
    def __call__(
        self,
        data: NDArray | "xr.DataArray",
        fill_value: float = ...,
        *,
        return_fraction: Literal[True],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

    def __call__(
        self,
        data: NDArray | "xr.DataArray",
        fill_value: float = np.nan,
        return_fraction: bool = False,
    ) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Apply interpolation to data.

        Parameters
        ----------
        data : array_like
            Data to interpolate. Can be:
            - 1D array of shape (npoints,)
            - 2D array of shape (nlevels, npoints) or (ntime, npoints)
            - ND array with last axis = npoints
        fill_value : float
            Value for invalid points outside influence radius (or, for
            "conservative", target cells with no valid overlapping source
            data).
        return_fraction : bool, default False
            If True, also return a ``valid_fraction`` array giving the
            fraction of each target cell's area covered by valid
            (non-NaN) overlapping source data. Only valid for
            ``method="conservative"``.

        Returns
        -------
        ndarray
            Regridded data. Shape depends on input and target:
            - 1D input, regular-grid target: (nlat, nlon)
            - 1D input, arbitrary target: (n_target,)
            - ND input: (*leading_dims, *target_shape)
        ndarray
            Only returned when ``return_fraction=True``: ``valid_fraction``
            with the same shape as the regridded data.
        """
        if return_fraction and self.method != "conservative":
            raise ValueError(
                "return_fraction=True is only supported for "
                "method='conservative'."
            )

        # Handle xarray DataArray
        if hasattr(data, "values"):
            data = data.values

        data = np.asarray(data)

        if self.method == "conservative":
            result, fraction = self._interpolate_conservative(data, fill_value)
            if return_fraction:
                return result, fraction
            return result

        assert self.target_lon is not None and self.target_lat is not None
        target_shape = self.target_lon.shape

        # Handle different input dimensions
        if data.ndim == 1:
            # Simple 1D case
            result = self._interpolate_1d(data, fill_value)
        elif data.ndim == 2:
            # 2D case: (extra_dim, npoints)
            n_extra = data.shape[0]
            result = np.empty((n_extra,) + target_shape, dtype=np.float64)
            for i in range(n_extra):
                result[i] = self._interpolate_1d(data[i], fill_value)
        else:
            # ND case: (*leading_dims, npoints)
            leading_shape = data.shape[:-1]
            npoints = data.shape[-1]
            data_flat = data.reshape(-1, npoints)
            result_flat = np.empty(
                (data_flat.shape[0],) + target_shape, dtype=np.float64
            )
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
        assert self.target_lon is not None and self.target_lat is not None
        if self.method == "nearest":
            result = data[self.indices]
            if not np.isnan(fill_value):
                result = result.astype(np.float64)
            result[~self.valid_mask] = fill_value
        elif self.method == "idw":
            result = np.sum(
                self._idw_weights * data[self._idw_indices], axis=-1
            )
            result[~self.valid_mask] = fill_value
        elif self.method == "linear":
            from scipy.interpolate import LinearNDInterpolator

            interp = LinearNDInterpolator(
                self._delaunay, data, fill_value=fill_value
            )
            target_2d = np.column_stack(
                [self.target_lon.ravel(), self.target_lat.ravel()]
            )
            result = interp(target_2d).reshape(self.target_lon.shape)
            # Apply distance-based valid_mask on top
            result[~self.valid_mask] = fill_value
        elif self.method == "cubic":
            from scipy.interpolate import CloughTocher2DInterpolator

            valid_src = np.isfinite(data)
            if valid_src.all():
                interp = CloughTocher2DInterpolator(
                    self._delaunay, data, fill_value=fill_value
                )
            else:
                interp = CloughTocher2DInterpolator(
                    self._source_2d[valid_src],
                    data[valid_src],
                    fill_value=fill_value,
                )
            target_2d = np.column_stack(
                [self.target_lon.ravel(), self.target_lat.ravel()]
            )
            result = interp(target_2d).reshape(self.target_lon.shape)
            # Apply distance-based valid_mask on top
            result[~self.valid_mask] = fill_value
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        return result

    def _interpolate_conservative(
        self,
        data: NDArray[np.floating],
        fill_value: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Area-weighted conservative interpolation, batched over all levels.

        Unlike the other methods, this applies the precomputed sparse
        overlap-weight matrix to the whole (possibly multi-level) input at
        once via a single sparse-dense matrix multiply, rather than looping
        per level.
        """
        assert self.target_lon is not None
        assert self._target_area is not None
        npoints = self.source_lon.shape[0]
        leading_shape = data.shape[:-1]
        data_flat = data.reshape(-1, npoints).astype(np.float64)

        valid = np.isfinite(data_flat)
        data_filled = np.where(valid, data_flat, 0.0)

        # (n_target, n_leading)
        numerator = self._weights @ data_filled.T
        denominator = self._weights @ valid.T.astype(np.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(denominator > 0, numerator / denominator, fill_value)
        fraction = denominator / self._target_area[:, np.newaxis]
        # The chord-based spherical area approximation (see conservative.py)
        # isn't perfectly additive under polygon clipping, so summed overlap
        # areas can drift slightly past a target cell's own computed area.
        # Clip so the reported fraction stays interpretable as a fraction.
        fraction = np.clip(fraction, 0.0, 1.0)

        out_shape = leading_shape + self.target_lon.shape

        value = np.asarray(value).T.reshape(out_shape)
        fraction = np.asarray(fraction).T.reshape(out_shape)

        return value, fraction

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the target: (nlat, nlon) for a regular grid, or
        (n_target,) for an arbitrary unstructured target."""
        assert self.target_lon is not None
        return self.target_lon.shape


def regrid(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating] | None = None,
    lat: NDArray[np.floating] | None = None,
    resolution: float | tuple[int, int] = 1.0,
    method: Literal["nearest", "idw", "linear", "cubic", "conservative"] = "nearest",
    influence_radius: float = 80_000.0,
    fill_value: float = np.nan,
    lon_bounds: tuple[float, float] = (-180.0, 180.0),
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
    as_xarray: bool = False,
) -> tuple[NDArray[np.floating], RegridInterpolator]:
    """Regrid unstructured data to regular grid.

    This convenience function only targets a regular lat/lon grid (built
    from ``resolution``/``lon_bounds``/``lat_bounds``). For conservative
    remapping onto an arbitrary unstructured target mesh, or to retrieve
    the coverage fraction of ``method="conservative"``, construct a
    :class:`RegridInterpolator` directly with ``target_lon``/``target_lat``
    and call it with ``return_fraction=True``.

    This is a convenience function that creates a RegridInterpolator and
    applies it. For repeated regridding with the same source grid, create
    a RegridInterpolator once and reuse it.

    Supports multi-dimensional data where the last axis contains the spatial
    points. For example:

    - 1D data (npoints,): single field
    - 2D data (nlevels, npoints): multi-level unstructured data (e.g., FESOM, ICON)
    - ND data (*dims, npoints): arbitrary leading dimensions

    Coordinate arrays can be:

    - 1D arrays of same size: unstructured mesh coordinates (used directly)
    - 1D arrays of different sizes: regular grid side coordinates (meshgrid created)
    - 2D arrays of same shape: full coordinate arrays (raveled to 1D)

    A warning is issued whenever coordinate transformations are applied.

    If lon/lat are not provided and data is an xarray DataArray, the function
    will attempt to extract coordinates automatically by looking for common
    coordinate names (lon/lat, longitude/latitude, x/y, etc.).

    Parameters
    ----------
    data : array_like
        Data to interpolate. Last axis must be npoints (matching coordinates).
        Can be 1D (npoints,), 2D (nlevels, npoints), or ND (*dims, npoints).
        If xarray DataArray, coordinates may be extracted automatically.
    lon : array_like, optional
        Source grid longitude coordinates. Can be 1D or 2D array.
        If None, will attempt to extract from data (xarray only).
    lat : array_like, optional
        Source grid latitude coordinates. Can be 1D or 2D array.
        If None, will attempt to extract from data (xarray only).
    resolution : float or tuple of int
        Target grid resolution.
    method : {"nearest", "idw", "linear", "cubic", "conservative"}
        Interpolation method. "nearest" uses nearest-neighbor lookup.
        "idw" uses inverse distance weighting (fast, smooth). "linear"
        uses Delaunay triangulation with barycentric interpolation.
        "cubic" uses Clough-Tocher C1 interpolation (smoothest).
        "conservative" uses area-weighted polygon overlap so an
        integrated quantity is preserved (slowest; see the class
        docstring of :class:`RegridInterpolator` for details).
    influence_radius : float
        Maximum influence radius in meters.
    fill_value : float
        Value for invalid points.
    lon_bounds : tuple of float
        Target grid longitude bounds.
    lat_bounds : tuple of float
        Target grid latitude bounds.
    as_xarray : bool, default False
        If True, wrap the regridded array in an ``xr.DataArray`` with
        ``lat`` and ``lon`` as 1-D dimension coordinates.  Leading
        dimensions (e.g. time, depth) and their coordinates are
        preserved when the input is an ``xr.DataArray``.  The return
        type of the tuple's first element changes from ``NDArray`` to
        ``xr.DataArray``.

    Returns
    -------
    regridded : ndarray or xr.DataArray
        Regridded data. Returns ``xr.DataArray`` when ``as_xarray=True``,
        otherwise ``ndarray``.
    interpolator : RegridInterpolator
        The interpolator used (can be reused for other variables).
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

    # Handle xarray DataArray
    if hasattr(data, "values"):
        data_values = data.values
    else:
        data_values = np.asarray(data)

    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)

    # Determine the data/coordinate format and prepare accordingly
    # Key insight: for unstructured data, lon and lat have SAME size matching data's last dim
    # For regular grids, lon and lat have DIFFERENT sizes matching data's last two dims

    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        if lon_arr.size == lat_arr.size:
            # Case: Unstructured mesh coordinates (both 1D, same size)
            # Data can be 1D (npoints,) or multi-level (nlevels, npoints)
            npoints = data_values.shape[-1] if data_values.ndim >= 1 else data_values.size
            if lon_arr.size != npoints:
                raise ValueError(
                    f"Coordinate size ({lon_arr.size}) must match data's last dimension ({npoints}). "
                    f"Data shape: {data_values.shape}"
                )
            # Coordinates are ready, data stays as-is
        else:
            # Case: Regular grid with side coordinates (1D lon, 1D lat, different sizes)
            # Data shape should be (..., nlat, nlon) or (nlat, nlon)
            if data_values.ndim < 2:
                raise ValueError(
                    f"For regular grid coordinates (lon size {lon_arr.size}, lat size {lat_arr.size}), "
                    f"data must be at least 2D, got shape {data_values.shape}"
                )
            nlat, nlon = data_values.shape[-2], data_values.shape[-1]
            if lon_arr.size != nlon or lat_arr.size != nlat:
                raise ValueError(
                    f"Coordinate sizes (lon: {lon_arr.size}, lat: {lat_arr.size}) must match "
                    f"data's last two dimensions (nlat: {nlat}, nlon: {nlon}). "
                    f"Data shape: {data_values.shape}"
                )
            # Create meshgrid and ravel
            warnings.warn(
                f"Creating meshgrid from 1D lon ({lon_arr.size}) and lat ({lat_arr.size}) "
                f"for data (shape {data_values.shape}), then raveling spatial dimensions.",
                stacklevel=2,
            )
            lon_arr, lat_arr = np.meshgrid(lon_arr, lat_arr)
            lon_arr = lon_arr.ravel()
            lat_arr = lat_arr.ravel()
            data_values = flatten_spatial(data_values)
    else:
        # 2D coordinates - prepare them and handle data accordingly
        lon_arr, lat_arr = prepare_coordinates(lon_arr, lat_arr)
        # Ravel data if it matches the original 2D coordinate shape
        if data_values.ndim >= 2 and data_values.shape[-2:] == np.asarray(lon).shape:
            data_values = flatten_spatial(data_values)

    interpolator = RegridInterpolator(
        source_lon=lon_arr,
        source_lat=lat_arr,
        resolution=resolution,
        method=method,
        influence_radius=influence_radius,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
    )

    regridded = interpolator(data_values, fill_value=fill_value)

    if as_xarray:
        import xarray as xr

        assert interpolator.target_lon is not None
        assert interpolator.target_lat is not None
        lat_1d = interpolator.target_lat[:, 0]
        lon_1d = interpolator.target_lon[0, :]

        if hasattr(data, "dims"):
            n_leading = regridded.ndim - 2
            leading_dim_names = list(data.dims[:n_leading])
            leading_coords = {d: data.coords[d] for d in leading_dim_names if d in data.coords}
            var_name = data.name or "data"
            var_attrs = dict(data.attrs)
        else:
            n_leading = regridded.ndim - 2
            leading_dim_names = [f"dim_{i}" for i in range(n_leading)]
            leading_coords = {}
            var_name = "data"
            var_attrs = {}

        dims = (*leading_dim_names, "lat", "lon")
        coords = {
            **leading_coords,
            "lat": ("lat", lat_1d, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", lon_1d, {"units": "degrees_east", "standard_name": "longitude"}),
        }

        regridded = xr.DataArray(
            regridded,
            dims=dims,
            coords=coords,
            name=var_name,
            attrs=var_attrs,
        )

    return regridded, interpolator
