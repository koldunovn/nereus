Regridding Guide
================

Nereus provides efficient tools for regridding unstructured mesh data to regular latitude-longitude grids.

Basic Regridding
----------------

The simplest way to regrid data:

.. code-block:: python

   import nereus as nr

   # Regrid to 1-degree resolution
   regridded, interpolator = nr.regrid(data, lon, lat, resolution=1.0)

   print(regridded.shape)  # (180, 360) for global 1-degree

The function returns:

* ``regridded``: 2D numpy array on the regular grid
* ``interpolator``: The :class:`~nereus.RegridInterpolator` for reuse

Flexible Input Formats
----------------------

Nereus accepts various input formats and handles the conversion automatically.
The key distinction is whether lon/lat have the **same size** (unstructured mesh)
or **different sizes** (regular grid side coordinates):

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Data Shape
     - Lon/Lat
     - Behavior
   * - ``(npoints,)``
     - 1D, same size
     - Unstructured mesh, used directly
   * - ``(nlevels, npoints)``
     - 1D, same size as npoints
     - Multi-level unstructured (e.g., FESOM, ICON)
   * - ``(nlat, nlon)``
     - 1D, different sizes
     - Regular grid: meshgrid created, data raveled
   * - ``(nlevels, nlat, nlon)``
     - 1D, different sizes
     - Multi-level regular grid: spatial dims raveled
   * - ``(ny, nx)``
     - 2D, same shape
     - All raveled to 1D (warning issued)

Example with multi-level unstructured data (FESOM/ICON style):

.. code-block:: python

   # Multi-level unstructured mesh data
   # data shape: (42, 196608) = (nlevels, npoints)
   # lon/lat shape: (196608,) = (npoints,)

   regridded, interp = nr.regrid(
       fesom_data,           # (42, 196608)
       mesh.longitude,       # (196608,)
       mesh.latitude,        # (196608,)
       resolution=1.0
   )
   # Result shape: (42, 180, 360) = (nlevels, nlat, nlon)

Example with 2D regular grid data:

.. code-block:: python

   # 2D data with 1D coordinates (like from NetCDF)
   # data shape: (180, 360) = (nlat, nlon)
   # lon shape: (360,), lat shape: (180,)

   data_2d = np.random.rand(180, 360)
   lon_1d = np.linspace(-179.5, 179.5, 360)
   lat_1d = np.linspace(-89.5, 89.5, 180)

   # Nereus automatically creates meshgrid internally
   regridded, _ = nr.regrid(data_2d, lon_1d, lat_1d, resolution=0.5)

Automatic Coordinate Extraction
-------------------------------

When working with xarray DataArrays, coordinates can be extracted automatically:

.. code-block:: python

   import xarray as xr

   # Load data with coordinates
   ds = xr.open_dataset("ocean_data.nc")
   temp = ds.temperature.isel(time=0, depth=0)

   # No need to specify lon/lat - extracted automatically
   regridded, interp = nr.regrid(temp, resolution=0.5)

Nereus recognizes common coordinate names:

- **Longitude**: ``lon``, ``longitude``, ``x``, ``nav_lon``, ``glon``, ``xt_ocean``, ``xu_ocean``, ``xh``, ``xq``, ``nod2d_lon``
- **Latitude**: ``lat``, ``latitude``, ``y``, ``nav_lat``, ``glat``, ``yt_ocean``, ``yu_ocean``, ``yh``, ``yq``, ``nod2d_lat``

Coordinate names are matched case-insensitively.

You can also override one coordinate while extracting the other:

.. code-block:: python

   # Use custom lon, extract lat from xarray
   regridded, _ = nr.regrid(temp, lon=custom_lon, resolution=0.5)

xarray Output
-------------

Pass ``as_xarray=True`` to receive the regridded result as an
:class:`xarray.DataArray` with ``lat`` and ``lon`` already attached as
1-D dimension coordinates, instead of a bare numpy array:

.. code-block:: python

   regridded, interp = nr.regrid(temp, resolution=0.5, as_xarray=True)

   print(type(regridded))   # <class 'xarray.core.dataarray.DataArray'>
   print(regridded.dims)    # ('lat', 'lon')
   print(regridded.lat)     # 1-D coordinate in degrees_north
   print(regridded.lon)     # 1-D coordinate in degrees_east

When the input is an :class:`xarray.DataArray`, leading dimensions
(e.g. ``time``, ``depth``) and their coordinate values are preserved
automatically, and the variable name and attributes are copied across:

.. code-block:: python

   # Multi-level xarray input
   temp_da = ds["temperature"]        # dims: (time, depth, npoints)

   regridded, _ = nr.regrid(temp_da, resolution=0.5, as_xarray=True)

   print(regridded.dims)             # ('time', 'depth', 'lat', 'lon')
   print(regridded.name)             # 'temperature'
   print(regridded.attrs)            # original variable attributes

For plain numpy input with leading dimensions, auto-generated names
``dim_0``, ``dim_1``, … are used for those axes.

Interpolation Methods
---------------------

Nereus supports five interpolation methods:

**Nearest neighbor** (default):

.. code-block:: python

   regridded, _ = nr.regrid(data, lon, lat, method="nearest")

Uses KDTree lookup in 3D Cartesian space. Fast and preserves original data
values, but produces blocky patterns when the source grid is much coarser
than the target.

**IDW** (Inverse Distance Weighting):

.. code-block:: python

   regridded, _ = nr.regrid(data, lon, lat, method="idw")

Uses the 8 nearest neighbors weighted by inverse squared distance.
Fast like nearest neighbor but produces smooth results. Weights are
pre-computed so repeated application is very efficient.

**Linear** (Delaunay-based):

.. code-block:: python

   regridded, _ = nr.regrid(data, lon, lat, method="linear")

Builds a Delaunay triangulation of the source points and interpolates
using barycentric coordinates. Produces smooth results. Points outside the
convex hull of source data are automatically masked with ``fill_value``.

**Cubic** (Clough-Tocher C1):

.. code-block:: python

   regridded, _ = nr.regrid(data, lon, lat, method="cubic")

Uses Clough-Tocher piecewise cubic interpolation on the Delaunay
triangulation. Produces the smoothest results with C1 continuity
(continuous first derivatives). Like linear, points outside the convex
hull are masked.

**Conservative** (area-weighted overlap):

.. code-block:: python

   regridded, _ = nr.regrid(data, lon, lat, method="conservative")

Preserves an integrated quantity (e.g. total heat or mass) rather than
interpolating point values: source and target cell polygons are derived
from a spherical Voronoi tessellation of the points, and each target
cell's value is the area-weighted average of every source cell it
overlaps. See :ref:`conservative-remapping` below for details, including
how to get the fraction of each target cell actually covered by data and
how to remap directly onto another unstructured mesh.

.. note::

   Linear and cubic interpolation are slower than nearest/IDW because a
   new interpolator object is created for each data field. Conservative
   remapping is slower still (it builds a Voronoi tessellation and does
   polygon-overlap geometry), but its weights are precomputed once, so
   repeated application via :class:`~nereus.RegridInterpolator` (rather
   than the ``nr.regrid()`` convenience function) is still efficient. All
   three are best suited for visualization, exploration, or budget
   diagnostics rather than the fastest possible bulk processing.

Source longitudes in any convention (0-360, -180-180, or mixed) are
automatically normalized to match the target grid, so data from models
like EN4 (0-360) works transparently with the default -180-180 target.

When to choose which method:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Method
     - Best for
     - Drawbacks
   * - ``"nearest"``
     - Fast exploration, high-res source data
     - Blocky patterns with coarse source data
   * - ``"idw"``
     - Fast smooth results, general purpose
     - Less accurate than triangulation-based methods
   * - ``"linear"``
     - Smooth visualization, coarse source data
     - Slower, may create long triangles over land
   * - ``"cubic"``
     - Smoothest results, presentation-quality plots
     - Slowest, may overshoot near sharp gradients
   * - ``"conservative"``
     - Budget/flux diagnostics where totals must be preserved
     - Slowest to build; boundaries are a planar approximation of true geodesics

The method parameter is also available in :func:`~nereus.plot`:

.. code-block:: python

   fig, ax, _ = nr.plot(data, lon, lat, method="linear")

.. _conservative-remapping:

Conservative Remapping
-----------------------

Conservative remapping answers a different question than the other
methods: instead of "what's the value at this point?" it asks "how much
of each source cell's area falls inside each target cell?", so that
``sum(target_value * target_area) ~= sum(source_value * source_area)``.
This matters for budget-style diagnostics (heat, freshwater, mass) where
regridding must not create or destroy the quantity being summed.

Source cell polygons (and, for mesh-to-mesh remapping, target cell
polygons) are derived generically from the point cloud via a spherical
Voronoi tessellation -- no mesh-specific connectivity is required, so this
works for FESOM nodes today and for any other unstructured mesh without
extra code. Overlap areas are computed with `shapely
<https://shapely.readthedocs.io/>`_ polygon intersection.

.. code-block:: python

   interpolator = nr.RegridInterpolator(
       mesh_lon, mesh_lat, resolution=1.0, method="conservative"
   )
   regridded, valid_fraction = interpolator(data, return_fraction=True)

``valid_fraction`` gives, for every target cell, the fraction of its area
that was actually covered by valid (non-NaN) overlapping source data
(similar to ESMF's ``frac_b``). Use it to mask or threshold cells with
poor coverage:

.. code-block:: python

   regridded[valid_fraction < 0.5] = np.nan

Unlike the other methods, conservative remapping can also target another
unstructured mesh directly (mesh-to-mesh remapping), by passing
``target_lon``/``target_lat`` instead of ``resolution``:

.. code-block:: python

   interpolator = nr.RegridInterpolator(
       fesom_lon, fesom_lat, method="conservative",
       target_lon=icon_lon, target_lat=icon_lat,
   )
   regridded, valid_fraction = interpolator(data, return_fraction=True)
   # regridded.shape == icon_lon.shape

``target_lon``/``target_lat`` and ``return_fraction=True`` both raise
``ValueError`` for any method other than ``"conservative"``. The
``nr.regrid()`` convenience function only supports the regular-grid
target case; use :class:`~nereus.RegridInterpolator` directly for
mesh-to-mesh remapping or to retrieve ``valid_fraction``.

.. note::

   Cell boundaries are built and clipped in the lon/lat plane (like the
   Delaunay triangulation used by ``"linear"``/``"cubic"``), and cell
   areas use the same chord-based spherical-area approximation as
   FESOM's mesh-area calculations elsewhere in nereus. This is a good
   approximation away from the poles but is not exact; ``valid_fraction``
   is clipped to ``[0, 1]`` to stay interpretable despite the small
   resulting approximation noise.

Resolution Options
------------------

The ``resolution`` parameter accepts:

**Single number (degrees)**:

.. code-block:: python

   # 1-degree grid (default)
   regridded, _ = nr.regrid(data, lon, lat, resolution=1.0)  # 360x180

   # Half-degree grid (higher resolution)
   regridded, _ = nr.regrid(data, lon, lat, resolution=0.5)  # 720x360

   # Quarter-degree grid
   regridded, _ = nr.regrid(data, lon, lat, resolution=0.25)  # 1440x720

**Tuple (nlon, nlat)**:

.. code-block:: python

   # Custom grid dimensions
   regridded, _ = nr.regrid(data, lon, lat, resolution=(720, 360))

Grid Bounds
-----------

By default, regridding covers the full globe. Customize with bounds:

.. code-block:: python

   # North Atlantic only
   regridded, _ = nr.regrid(
       data, lon, lat,
       resolution=0.5,
       lon_bounds=(-80, 0),
       lat_bounds=(0, 65)
   )

   # Arctic region
   regridded, _ = nr.regrid(
       data, lon, lat,
       resolution=0.25,
       lat_bounds=(60, 90)
   )

The Influence Radius
--------------------

The ``influence_radius`` parameter controls the maximum distance (in meters) from a target grid point to valid source data:

.. code-block:: python

   # Strict: only interpolate very close to data points
   regridded, _ = nr.regrid(data, lon, lat, influence_radius=50000)  # 50 km

   # Default: reasonable for most meshes
   regridded, _ = nr.regrid(data, lon, lat, influence_radius=80000)  # 80 km

   # Permissive: fill larger gaps
   regridded, _ = nr.regrid(data, lon, lat, influence_radius=200000)  # 200 km

Points outside the influence radius are filled with ``fill_value`` (default: ``np.nan``).

.. note::

   For coarse meshes, increase the influence radius. For high-resolution meshes, you can decrease it for sharper boundaries.

Using RegridInterpolator Directly
---------------------------------

For repeated regridding operations on the same mesh, create an interpolator once:

.. code-block:: python

   # Create interpolator (slow - builds KD-tree / Delaunay)
   interpolator = nr.RegridInterpolator(
       lon, lat,
       resolution=0.5,
       method="nearest",         # or "idw", "linear", "cubic", "conservative"
       influence_radius=80000.0,
       lon_bounds=(-180, 180),
       lat_bounds=(-90, 90)
   )

   # Apply to multiple fields (fast - reuses weights)
   temp_regridded = interpolator(temp_data)
   salt_regridded = interpolator(salt_data)
   ssh_regridded = interpolator(ssh_data)

Interpolator Properties
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   interpolator = nr.RegridInterpolator(lon, lat, resolution=1.0)

   # Target grid coordinates
   target_lon = interpolator.target_lon  # 2D array
   target_lat = interpolator.target_lat  # 2D array

   # Grid shape
   print(interpolator.shape)  # (180, 360)

   # Valid mask (True where data is available)
   valid = interpolator.valid_mask  # 2D boolean array

Handling Multi-Dimensional Data
-------------------------------

The interpolator handles arrays with additional dimensions:

.. code-block:: python

   interpolator = nr.RegridInterpolator(lon, lat, resolution=1.0)

   # 1D: (npoints,) -> (nlat, nlon)
   surface_temp = interpolator(temp_2d)

   # 2D: (nz, npoints) -> (nz, nlat, nlon)
   full_3d = interpolator(temp_3d)

   # 3D: (time, nz, npoints) -> (time, nz, nlat, nlon)
   timeseries = interpolator(temp_4d)

The last axis is always assumed to be the spatial dimension (npoints).

Automatic Caching
-----------------

Nereus maintains a cache of interpolators:

.. code-block:: python

   # First call: builds interpolator
   regridded1, interp = nr.regrid(data1, lon, lat, resolution=0.5)

   # Second call: retrieves from cache (same coordinates + parameters)
   regridded2, interp = nr.regrid(data2, lon, lat, resolution=0.5)

Configure caching behavior:

.. code-block:: python

   # Increase cache size
   nr.set_cache_options(max_memory_items=50)

   # Enable disk persistence
   nr.set_cache_options(disk_path="/path/to/cache")

   # Clear cache
   from nereus.regrid.cache import clear_cache
   clear_cache()

Fill Values
-----------

Customize how missing data is handled:

.. code-block:: python

   # Default: NaN for missing
   regridded = interpolator(data, fill_value=np.nan)

   # Use specific value
   regridded = interpolator(data, fill_value=-999)

   # Use zero (useful for some applications)
   regridded = interpolator(data, fill_value=0.0)

Saving Regridded Data
---------------------

The quickest way to export regridded data is with ``as_xarray=True``,
which returns a :class:`xarray.DataArray` that is ready to save:

.. code-block:: python

   temp_da, interp = nr.regrid(temp, lon, lat, resolution=0.5, as_xarray=True)
   salt_da, _      = nr.regrid(salt, lon, lat, resolution=0.5, as_xarray=True)

   ds_regridded = xr.Dataset({"temp": temp_da, "salt": salt_da})
   ds_regridded.to_netcdf("regridded_output.nc")

If you are using :class:`~nereus.RegridInterpolator` directly and need to
attach coordinates manually, the target grid arrays are available on the
interpolator:

.. code-block:: python

   import xarray as xr

   interpolator = nr.RegridInterpolator(lon, lat, resolution=0.5)

   temp_reg = interpolator(temp)
   salt_reg = interpolator(salt)

   ds_regridded = xr.Dataset(
       {
           "temp": (["lat", "lon"], temp_reg),
           "salt": (["lat", "lon"], salt_reg),
       },
       coords={
           "lon": interpolator.target_lon[0, :],  # 1D longitude
           "lat": interpolator.target_lat[:, 0],  # 1D latitude
       }
   )

   ds_regridded.to_netcdf("regridded_output.nc")

Performance Tips
----------------

1. **Reuse interpolators**: The KD-tree construction is expensive; reuse interpolators when possible.

2. **Use appropriate resolution**: Higher resolution = more memory and slower computation.

3. **Limit bounds**: Only regrid the region you need.

4. **Batch operations**: Regrid multiple time steps in a loop rather than rebuilding interpolators.

.. code-block:: python

   # Efficient: create once, use many times
   interp = nr.RegridInterpolator(lon, lat, resolution=0.5)

   results = []
   for t in range(n_times):
       regridded = interp(data[t])
       results.append(regridded)

   # Stack results
   all_regridded = np.stack(results, axis=0)

Comparison with Other Tools
---------------------------

Nereus regridding is optimized for quick exploration:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tool
     - Strength
     - When to use
   * - Nereus (nearest)
     - Speed, simplicity
     - Quick exploration, high-res source data
   * - Nereus (idw)
     - Fast smooth output
     - General-purpose visualization
   * - Nereus (linear)
     - Smooth output, no extra deps
     - Visualization of coarse source data
   * - Nereus (cubic)
     - Smoothest output (C1)
     - Presentation-quality plots
   * - xESMF
     - Conservation, accuracy
     - Production workflows, budget-closing
   * - CDO
     - Flexibility, formats
     - Command-line processing

For publication-quality results requiring conservative remapping, consider using xESMF or CDO after initial exploration with Nereus.
