"""Grid utilities for nereus.

Functions for creating regular grids for regridding and plotting.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import xarray as xr

# Common coordinate name patterns for longitude
LON_NAMES = (
    "lon",
    "longitude",
    "x",
    "nav_lon",
    "glon",
    "xt_ocean",
    "xu_ocean",
    "xh",
    "xq",
    "nod2d_lon",
)

# Common coordinate name patterns for latitude
LAT_NAMES = (
    "lat",
    "latitude",
    "y",
    "nav_lat",
    "glat",
    "yt_ocean",
    "yu_ocean",
    "yh",
    "yq",
    "nod2d_lat",
)


def _find_coord_by_names(
    data: "xr.DataArray",
    names: tuple[str, ...],
    coord_type: str,
) -> NDArray | None:
    """Find a coordinate in DataArray by trying common names.

    Parameters
    ----------
    data : xr.DataArray
        The DataArray to search for coordinates.
    names : tuple of str
        Tuple of possible coordinate names to try.
    coord_type : str
        Type of coordinate for error messages ("longitude" or "latitude").

    Returns
    -------
    ndarray or None
        The coordinate values if found, None otherwise.
    """
    # First check coordinates (dims + coords)
    all_coords = set(data.dims) | set(data.coords.keys())

    for name in names:
        # Try exact match (case-insensitive)
        for coord_name in all_coords:
            if coord_name.lower() == name.lower():
                return np.asarray(data.coords[coord_name].values)

    # Try attributes that might contain coordinate info
    if hasattr(data, "attrs"):
        for attr_name in ("coordinates", "grid_mapping"):
            if attr_name in data.attrs:
                coord_str = data.attrs[attr_name]
                for name in names:
                    if name in coord_str.lower():
                        # Found reference, try to get from coords
                        for coord_name in all_coords:
                            if coord_name.lower() == name.lower():
                                return np.asarray(data.coords[coord_name].values)

    return None


def extract_coordinates(
    data: "xr.DataArray | NDArray",
) -> tuple[NDArray | None, NDArray | None]:
    """Extract longitude and latitude coordinates from an xarray DataArray.

    This function attempts to automatically detect coordinate variables by
    looking for common naming conventions used in geophysical data.

    Parameters
    ----------
    data : xr.DataArray or array_like
        The data array to extract coordinates from. If not an xarray DataArray,
        returns (None, None).

    Returns
    -------
    lon : ndarray or None
        Longitude coordinates if found, None otherwise.
    lat : ndarray or None
        Latitude coordinates if found, None otherwise.

    Notes
    -----
    The function looks for coordinates with these common names:

    Longitude: lon, longitude, x, nav_lon, glon, xt_ocean, xu_ocean, xh, xq, nod2d_lon
    Latitude: lat, latitude, y, nav_lat, glat, yt_ocean, yu_ocean, yh, yq, nod2d_lat

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.DataArray(
    ...     np.random.rand(10, 20),
    ...     coords={"lat": np.arange(10), "lon": np.arange(20)},
    ...     dims=["lat", "lon"]
    ... )
    >>> lon, lat = extract_coordinates(data)
    >>> lon.shape, lat.shape
    ((20,), (10,))
    """
    # Check if it's an xarray DataArray
    if not hasattr(data, "coords") or not hasattr(data, "dims"):
        return None, None

    lon = _find_coord_by_names(data, LON_NAMES, "longitude")
    lat = _find_coord_by_names(data, LAT_NAMES, "latitude")

    return lon, lat


def prepare_input_arrays(
    data: NDArray | "xr.DataArray",
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
) -> tuple[NDArray, NDArray[np.floating], NDArray[np.floating]]:
    """Prepare and validate input arrays, converting to 1D if necessary.

    This function handles various input formats and transforms them to 1D arrays
    suitable for plotting and regridding. It issues warnings when transformations
    are applied.

    Parameters
    ----------
    data : array_like
        Data values, can be 1D or 2D.
    lon : array_like
        Longitude coordinates, can be 1D or 2D.
    lat : array_like
        Latitude coordinates, can be 1D or 2D.

    Returns
    -------
    data_1d : ndarray
        1D array of data values.
    lon_1d : ndarray
        1D array of longitude coordinates.
    lat_1d : ndarray
        1D array of latitude coordinates.

    Raises
    ------
    ValueError
        If array dimensions are incompatible or sizes don't match.

    Notes
    -----
    Supported input combinations:

    - All 1D arrays of same size: used directly (no warning)
    - 2D data with 2D lon/lat (same shape): all raveled to 1D
    - 1D data with 2D lon/lat: lon/lat raveled to match data
    - 2D data with 1D lon/lat: meshgrid created, then all raveled

    Examples
    --------
    >>> # 2D data with 1D coordinates (regular grid)
    >>> data = np.random.rand(180, 360)
    >>> lon = np.linspace(-179.5, 179.5, 360)
    >>> lat = np.linspace(-89.5, 89.5, 180)
    >>> data_1d, lon_1d, lat_1d = prepare_input_arrays(data, lon, lat)
    >>> data_1d.shape
    (64800,)
    """
    # Handle xarray DataArray
    data_arr = data.values if hasattr(data, "values") else np.asarray(data)
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)

    data_ndim = data_arr.ndim
    lon_ndim = lon_arr.ndim
    lat_ndim = lat_arr.ndim

    # Case 1: All 1D arrays of the same size - no transformation needed
    if data_ndim == 1 and lon_ndim == 1 and lat_ndim == 1:
        if not (data_arr.size == lon_arr.size == lat_arr.size):
            raise ValueError(
                f"1D arrays must have the same size. Got data: {data_arr.size}, "
                f"lon: {lon_arr.size}, lat: {lat_arr.size}"
            )
        return data_arr, lon_arr, lat_arr

    # Case 2: 2D data with 2D lon/lat of matching shape
    if data_ndim == 2 and lon_ndim == 2 and lat_ndim == 2:
        if not (data_arr.shape == lon_arr.shape == lat_arr.shape):
            raise ValueError(
                f"2D arrays must have the same shape. Got data: {data_arr.shape}, "
                f"lon: {lon_arr.shape}, lat: {lat_arr.shape}"
            )
        warnings.warn(
            f"Raveling 2D arrays (shape {data_arr.shape}) to 1D for plotting.",
            stacklevel=3,
        )
        return data_arr.ravel(), lon_arr.ravel(), lat_arr.ravel()

    # Case 3: 1D data with 2D lon/lat (pre-ravelled data)
    if data_ndim == 1 and lon_ndim == 2 and lat_ndim == 2:
        if lon_arr.shape != lat_arr.shape:
            raise ValueError(
                f"2D lon and lat arrays must have the same shape. "
                f"Got lon: {lon_arr.shape}, lat: {lat_arr.shape}"
            )
        if data_arr.size != lon_arr.size:
            raise ValueError(
                f"Data size ({data_arr.size}) must match lon/lat size ({lon_arr.size})."
            )
        warnings.warn(
            f"Raveling 2D lon/lat arrays (shape {lon_arr.shape}) to match 1D data.",
            stacklevel=3,
        )
        return data_arr, lon_arr.ravel(), lat_arr.ravel()

    # Case 4: 2D data with 1D lon/lat (side coordinates - need meshgrid)
    if data_ndim == 2 and lon_ndim == 1 and lat_ndim == 1:
        ny, nx = data_arr.shape
        if lon_arr.size != nx:
            raise ValueError(
                f"1D lon array size ({lon_arr.size}) must match data columns ({nx})."
            )
        if lat_arr.size != ny:
            raise ValueError(
                f"1D lat array size ({lat_arr.size}) must match data rows ({ny})."
            )
        warnings.warn(
            f"Creating meshgrid from 1D lon ({lon_arr.size}) and lat ({lat_arr.size}) "
            f"for 2D data (shape {data_arr.shape}), then raveling to 1D.",
            stacklevel=3,
        )
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
        return data_arr.ravel(), lon_2d.ravel(), lat_2d.ravel()

    # Case 5: 1D data with 1D lon/lat but different sizes (invalid)
    # Already handled above in Case 1

    # Case 6: 2D data with mixed 1D/2D lon/lat
    if data_ndim == 2 and (lon_ndim == 1) != (lat_ndim == 1):
        raise ValueError(
            "When data is 2D, lon and lat must both be 1D (side coordinates) "
            f"or both be 2D (full coordinates). Got lon: {lon_ndim}D, lat: {lat_ndim}D"
        )

    # Case 7: 1D data with mixed 1D/2D lon/lat
    if data_ndim == 1 and (lon_ndim == 1) != (lat_ndim == 1):
        raise ValueError(
            "When data is 1D, lon and lat must both be 1D or both be 2D. "
            f"Got lon: {lon_ndim}D, lat: {lat_ndim}D"
        )

    # Catch-all for unsupported dimensions
    raise ValueError(
        f"Unsupported array dimensions: data {data_ndim}D, lon {lon_ndim}D, lat {lat_ndim}D. "
        "Supported combinations: all 1D, all 2D, 2D data with 1D lon/lat, "
        "or 1D data with 2D lon/lat."
    )


def prepare_coordinates(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    data_shape: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Prepare and validate coordinate arrays, converting to 1D if necessary.

    This function handles various coordinate formats and transforms them to 1D
    arrays. It issues warnings when transformations are applied.

    Parameters
    ----------
    lon : array_like
        Longitude coordinates, can be 1D or 2D.
    lat : array_like
        Latitude coordinates, can be 1D or 2D.
    data_shape : tuple of int, optional
        Expected data shape for validation. If provided with 1D lon/lat,
        used to create a meshgrid matching the data dimensions.

    Returns
    -------
    lon_1d : ndarray
        1D array of longitude coordinates.
    lat_1d : ndarray
        1D array of latitude coordinates.

    Raises
    ------
    ValueError
        If array dimensions are incompatible or sizes don't match.

    Examples
    --------
    >>> # 2D coordinates
    >>> lon_2d, lat_2d = np.meshgrid(np.arange(10), np.arange(5))
    >>> lon_1d, lat_1d = prepare_coordinates(lon_2d, lat_2d)
    >>> lon_1d.shape
    (50,)

    >>> # 1D coordinates with data shape
    >>> lon = np.arange(10)
    >>> lat = np.arange(5)
    >>> lon_1d, lat_1d = prepare_coordinates(lon, lat, data_shape=(5, 10))
    >>> lon_1d.shape
    (50,)
    """
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)

    lon_ndim = lon_arr.ndim
    lat_ndim = lat_arr.ndim

    # Case 1: Both 1D with same size - already flat coordinates
    if lon_ndim == 1 and lat_ndim == 1 and lon_arr.size == lat_arr.size:
        return lon_arr, lat_arr

    # Case 2: Both 2D with same shape - ravel them
    if lon_ndim == 2 and lat_ndim == 2:
        if lon_arr.shape != lat_arr.shape:
            raise ValueError(
                f"2D lon and lat arrays must have the same shape. "
                f"Got lon: {lon_arr.shape}, lat: {lat_arr.shape}"
            )
        warnings.warn(
            f"Raveling 2D lon/lat arrays (shape {lon_arr.shape}) to 1D.",
            stacklevel=3,
        )
        return lon_arr.ravel(), lat_arr.ravel()

    # Case 3: Both 1D with different sizes - need meshgrid
    if lon_ndim == 1 and lat_ndim == 1 and lon_arr.size != lat_arr.size:
        # This is the side-coordinates case
        if data_shape is not None:
            if len(data_shape) != 2:
                raise ValueError(
                    f"data_shape must be 2D for meshgrid creation, got {len(data_shape)}D"
                )
            ny, nx = data_shape
            if lon_arr.size != nx:
                raise ValueError(
                    f"1D lon array size ({lon_arr.size}) must match data columns ({nx})."
                )
            if lat_arr.size != ny:
                raise ValueError(
                    f"1D lat array size ({lat_arr.size}) must match data rows ({ny})."
                )
        warnings.warn(
            f"Creating meshgrid from 1D lon ({lon_arr.size}) and lat ({lat_arr.size}), "
            "then raveling to 1D.",
            stacklevel=3,
        )
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
        return lon_2d.ravel(), lat_2d.ravel()

    # Case 4: Mixed 1D/2D
    if (lon_ndim == 1) != (lat_ndim == 1):
        raise ValueError(
            f"lon and lat must both be 1D or both be 2D. "
            f"Got lon: {lon_ndim}D, lat: {lat_ndim}D"
        )

    # Catch-all for unsupported dimensions
    raise ValueError(
        f"Unsupported coordinate dimensions: lon {lon_ndim}D, lat {lat_ndim}D. "
        "Supported: both 1D (same or different sizes) or both 2D (same shape)."
    )


def create_regular_grid(
    resolution: float | tuple[int, int] = 1.0,
    lon_bounds: tuple[float, float] = (-180.0, 180.0),
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
    center: Literal["cell", "node"] = "cell",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Create a regular lon/lat grid.

    Parameters
    ----------
    resolution : float or tuple of int
        Grid resolution. If float, specifies degrees per grid cell.
        If tuple (nlon, nlat), specifies number of grid points.
    lon_bounds : tuple of float
        Longitude bounds (lon_min, lon_max) in degrees.
    lat_bounds : tuple of float
        Latitude bounds (lat_min, lat_max) in degrees.
    center : {"cell", "node"}
        Whether coordinates are at cell centers or nodes.
        "cell" means coordinates at center of grid boxes.
        "node" means coordinates at corners.

    Returns
    -------
    lon, lat : tuple of ndarrays
        2D arrays of longitude and latitude coordinates.

    Examples
    --------
    >>> lon, lat = create_regular_grid(1.0)  # 1 degree resolution
    >>> lon.shape
    (180, 360)

    >>> lon, lat = create_regular_grid((360, 180))  # 360x180 grid
    >>> lon.shape
    (180, 360)
    """
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    if isinstance(resolution, (list, tuple)):
        nlon, nlat = resolution
    else:
        nlon = int((lon_max - lon_min) / resolution)
        nlat = int((lat_max - lat_min) / resolution)

    if center == "cell":
        # Cell centers
        dlon = (lon_max - lon_min) / nlon
        dlat = (lat_max - lat_min) / nlat
        lon_1d = np.linspace(lon_min + dlon / 2, lon_max - dlon / 2, nlon)
        lat_1d = np.linspace(lat_min + dlat / 2, lat_max - dlat / 2, nlat)
    else:
        # Node positions
        lon_1d = np.linspace(lon_min, lon_max, nlon)
        lat_1d = np.linspace(lat_min, lat_max, nlat)

    lon, lat = np.meshgrid(lon_1d, lat_1d)

    return lon, lat


def grid_cell_area(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    radius: float = 6_371_000.0,
) -> NDArray[np.floating]:
    """Compute area of regular grid cells.

    Parameters
    ----------
    lon : ndarray
        2D array of longitude coordinates (cell centers).
    lat : ndarray
        2D array of latitude coordinates (cell centers).
    radius : float
        Earth radius in meters.

    Returns
    -------
    ndarray
        2D array of cell areas in square meters.

    Notes
    -----
    Assumes uniform spacing in lon and lat.
    Area of a spherical rectangle:
    A = R^2 * |sin(lat1) - sin(lat2)| * |lon2 - lon1|
    """
    # Get grid spacing
    if lon.ndim == 2:
        dlon = np.abs(lon[0, 1] - lon[0, 0])
        dlat = np.abs(lat[1, 0] - lat[0, 0])
    else:
        dlon = np.abs(lon[1] - lon[0])
        dlat = np.abs(lat[1] - lat[0])

    dlon_rad = np.deg2rad(dlon)
    lat_rad = np.deg2rad(lat)

    # Half grid spacing in lat
    dlat_rad = np.deg2rad(dlat / 2)

    # sin(lat + dlat/2) - sin(lat - dlat/2)
    sin_diff = np.sin(lat_rad + dlat_rad) - np.sin(lat_rad - dlat_rad)

    area = radius**2 * np.abs(sin_diff) * dlon_rad

    return area


def expand_bounds_for_polar(
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    factor: float = 1.414,  # sqrt(2)
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand bounding box for polar projections.

    Polar projections need a larger data extent to fill the circular
    plot area without gaps.

    Parameters
    ----------
    lon_bounds : tuple of float
        Original longitude bounds.
    lat_bounds : tuple of float
        Original latitude bounds.
    factor : float
        Expansion factor. Default is sqrt(2).

    Returns
    -------
    lon_bounds, lat_bounds : tuple of tuples
        Expanded bounds.
    """
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2

    lon_half = (lon_max - lon_min) / 2 * factor
    lat_half = (lat_max - lat_min) / 2 * factor

    new_lon_bounds = (
        max(-180.0, lon_center - lon_half),
        min(180.0, lon_center + lon_half),
    )
    new_lat_bounds = (
        max(-90.0, lat_center - lat_half),
        min(90.0, lat_center + lat_half),
    )

    return new_lon_bounds, new_lat_bounds
