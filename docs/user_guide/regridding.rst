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

   # Create interpolator (slow - builds KD-tree)
   interpolator = nr.RegridInterpolator(
       lon, lat,
       resolution=0.5,
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

Export regridded data to NetCDF with xarray:

.. code-block:: python

   import xarray as xr

   # Create interpolator
   interpolator = nr.RegridInterpolator(lon, lat, resolution=0.5)

   # Regrid data
   temp_reg = interpolator(temp)
   salt_reg = interpolator(salt)

   # Create xarray Dataset
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

   # Save to NetCDF
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
   * - Nereus
     - Speed, simplicity
     - Quick exploration, visualization
   * - xESMF
     - Conservation, accuracy
     - Production workflows, budget-closing
   * - CDO
     - Flexibility, formats
     - Command-line processing

For publication-quality results requiring conservative remapping, consider using xESMF or CDO after initial exploration with Nereus.
