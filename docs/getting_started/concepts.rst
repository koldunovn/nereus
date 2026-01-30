Core Concepts
=============

This page explains the key concepts underlying Nereus to help you use it effectively.

Unstructured vs. Regular Grids
------------------------------

Atmospheric and ocean models increasingly use **unstructured grids** (also called triangular meshes or irregular grids) instead of regular latitude-longitude grids. Examples include:

* **FESOM2**: Finite Element Sea ice-Ocean Model
* **ICON**: Icosahedral Nonhydrostatic model
* **MPAS**: Model for Prediction Across Scales

Unstructured grids offer advantages:

* Variable resolution (high resolution where needed)
* Better representation of coastlines and topography
* No convergence of grid lines at poles

However, they require special handling for visualization and analysis since most plotting and analysis tools expect regular grids.

.. note::

   Unstructured grids use triangular elements that can vary in size, allowing
   higher resolution in areas of interest (like coastlines) while using larger
   elements in the open ocean to reduce computational cost.

The Regridding Process
----------------------

Nereus converts unstructured data to regular grids using **nearest neighbor interpolation** based on a KD-tree spatial index.

The process involves:

1. **Coordinate transformation**: Convert lon/lat to 3D Cartesian coordinates on a unit sphere
2. **KD-tree construction**: Build a spatial index of source points
3. **Nearest neighbor lookup**: For each target grid point, find the nearest source point
4. **Distance validation**: Mask points that are too far from any source data

.. note::

   Nearest neighbor interpolation is chosen for speed and simplicity. It preserves the original data values without smoothing, making it ideal for quick exploration. More sophisticated interpolation methods (IDW, linear) may be added in future versions.

The RegridInterpolator
----------------------

The :class:`~nereus.RegridInterpolator` is the core class for regridding:

.. code-block:: python

   interpolator = nr.RegridInterpolator(
       source_lon, source_lat,
       resolution=1.0,           # Target resolution in degrees
       method="nearest",         # Interpolation method
       influence_radius=80000.0, # Max distance in meters
   )

   # Apply to any data on the same source grid
   regridded = interpolator(data)

Key parameters:

**resolution**
   Can be a single number (e.g., ``1.0`` for 1-degree) or a tuple ``(nlon, nlat)`` for explicit grid dimensions.

**influence_radius**
   Points on the target grid farther than this distance (in meters) from any source point are masked. This prevents extrapolation into data-void regions.

**lon_bounds / lat_bounds**
   Customize the target grid extent. Defaults to global coverage (-180 to 180, -90 to 90).

Caching and Performance
-----------------------

Building the interpolation weights (KD-tree and nearest neighbor lookup) is the expensive part. Nereus provides automatic caching:

.. code-block:: python

   # First call: builds interpolator (~1-2 seconds for large meshes)
   fig, ax, interp = nr.plot(data1, lon, lat)

   # Second call: reuses cached interpolator (instant)
   fig, ax, interp = nr.plot(data2, lon, lat)

The cache is keyed by:

* Source coordinates (lon, lat arrays)
* Resolution
* Influence radius
* Bounds

You can control caching behavior:

.. code-block:: python

   # Configure cache size
   nr.set_cache_options(max_memory_items=20)

   # Disable caching for a specific plot
   nr.plot(data, lon, lat, use_cache=False)

   # Manually reuse an interpolator
   nr.plot(data, lon, lat, interpolator=interp)

Flexible Input Handling
-----------------------

Nereus is designed to work with data in various formats without requiring manual preprocessing.

**Automatic Array Reshaping**

Functions like ``plot()``, ``regrid()``, and ``transect()`` accept data and coordinates in multiple formats:

- **1D arrays**: Traditional unstructured mesh format
- **2D arrays**: Regular grid format (automatically raveled)
- **Mixed formats**: 2D data with 1D coordinates (meshgrid created automatically)

.. code-block:: python

   # All of these work:
   nr.plot(data_1d, lon_1d, lat_1d)           # Unstructured mesh
   nr.plot(data_2d, lon_2d, lat_2d)           # Curvilinear grid
   nr.plot(data_2d, lon_1d, lat_1d)           # Regular grid

Warnings are issued when transformations are applied, so you know exactly what's happening.

**Automatic Coordinate Extraction**

When using xarray DataArrays, coordinates can be extracted automatically:

.. code-block:: python

   import xarray as xr

   ds = xr.open_dataset("model_output.nc")
   temp = ds.temperature.isel(time=0)

   # Coordinates extracted from xarray metadata
   nr.plot(temp)
   nr.regrid(temp, resolution=0.5)

Nereus recognizes common coordinate names used by various models:

- FESOM/ICON: ``lon``, ``lat``, ``nod2d_lon``, ``nod2d_lat``
- NEMO: ``nav_lon``, ``nav_lat``
- MOM/GFDL: ``glon``, ``glat``, ``xt_ocean``, ``yt_ocean``
- Generic: ``longitude``, ``latitude``, ``x``, ``y``

Coordinate Systems
------------------

Nereus uses standard geographic conventions:

**Longitude**
   -180 to 180 degrees (or 0 to 360, automatically handled)

**Latitude**
   -90 to 90 degrees

**Depth**
   Positive downward in meters (following oceanographic convention)

**Area**
   In square meters (m²)

Internally, Nereus converts lon/lat to Cartesian coordinates on a unit sphere for spatial operations:

.. math::

   x &= \cos(\text{lat}) \cdot \cos(\text{lon}) \\
   y &= \cos(\text{lat}) \cdot \sin(\text{lon}) \\
   z &= \sin(\text{lat})

This avoids issues with the pole singularity and dateline crossings.

Area-Weighted Operations
------------------------

Many diagnostics require proper area weighting. For unstructured meshes, each grid point has an associated area (often called "cluster area" or "dual cell area").

.. code-block:: python

   # Total sea ice area
   ice_area = sum(concentration * cell_area)

   # Area-weighted mean
   mean_temp = sum(temp * area) / sum(area)

   # Volume-weighted mean (3D)
   mean_temp = sum(temp * area * thickness) / sum(area * thickness)

Nereus handles these calculations correctly:

.. code-block:: python

   # Sea ice area
   total_ice = nr.ice_area(concentration, area)

   # Volume mean for upper ocean
   upper_mean = nr.volume_mean(
       temp, area, thickness, depth,
       depth_max=500  # Upper 500m only
   )

Model-Specific Support
----------------------

Each model has its own mesh format and conventions. Nereus provides model-specific submodules:

.. code-block:: python

   # FESOM2
   mesh = nr.fesom.load_mesh("/path/to/mesh/")

   # Future: ICON-Ocean
   mesh = nr.icono.load_mesh("icon_mesh.nc")

   # Future: ICON-Atmosphere
   mesh = nr.icona.load_mesh("icon_atmo_mesh.nc")

Each mesh object provides a consistent interface:

* ``mesh.lon``, ``mesh.lat``: Coordinates
* ``mesh.area``: Cell areas
* ``mesh.find_nearest()``: Spatial queries
* ``mesh.subset_by_bbox()``: Geographic subsetting

The MeshProtocol
~~~~~~~~~~~~~~~~

All mesh classes follow the :class:`~nereus.models._base.MeshBase` protocol:

.. code-block:: python

   class MeshProtocol:
       @property
       def lon(self) -> NDArray: ...

       @property
       def lat(self) -> NDArray: ...

       @property
       def area(self) -> NDArray: ...

       def find_nearest(self, lon, lat, k=1): ...
       def subset_by_bbox(self, lon_min, lon_max, lat_min, lat_max): ...

This allows writing generic code that works with any supported model.

Dask Integration
----------------

Nereus is designed to work with Dask arrays for out-of-core and parallel computation:

.. code-block:: python

   import dask.array as da

   # Load data lazily with Dask
   ds = xr.open_mfdataset("output_*.nc", chunks={"time": 10})

   # Diagnostics work with Dask arrays
   ice_area_timeseries = nr.ice_area(ds.a_ice, mesh.area)
   # Returns a Dask array - computation is deferred

   # Trigger computation
   result = ice_area_timeseries.compute()

When working with Dask:

* Regridding and plotting trigger immediate computation (needed for visualization)
* Diagnostics preserve lazy evaluation where possible
* Use ``.compute()`` or ``.values`` to trigger computation

Physical Constants
------------------

Nereus uses standard physical constants for ocean calculations:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Constant
     - Value
     - Description
   * - Earth radius
     - 6,371,000 m
     - WGS84 mean radius
   * - Seawater density
     - 1,025 kg/m³
     - Reference density
   * - Specific heat
     - 3,985 J/(kg·K)
     - Seawater heat capacity

These are used in calculations like ocean heat content:

.. math::

   OHC = \rho \cdot c_p \cdot \sum (T - T_{ref}) \cdot \Delta z \cdot A

Where:
- :math:`\rho` = seawater density
- :math:`c_p` = specific heat capacity
- :math:`T` = temperature
- :math:`T_{ref}` = reference temperature
- :math:`\Delta z` = layer thickness
- :math:`A` = cell area
