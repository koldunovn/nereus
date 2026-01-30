Quickstart
==========

This guide will help you get started with Nereus in just a few minutes.

Basic Import
------------

Import Nereus with the conventional alias:

.. code-block:: python

   import nereus as nr
   import numpy as np
   import xarray as xr

Plotting Unstructured Data
--------------------------

The most common use case is plotting data from an unstructured mesh:

.. code-block:: python

   # Load your data (example with xarray)
   ds = xr.open_dataset("model_output.nc")

   # Get temperature data
   temp = ds.temperature.isel(time=0, depth=0)

   # Create a map plot - coordinates extracted automatically from xarray
   fig, ax, interpolator = nr.plot(
       temp,
       projection="rob",      # Robinson projection
       cmap="RdBu_r",         # Colormap
       vmin=-2, vmax=30,      # Color limits
       coastlines=True,       # Add coastlines
       colorbar=True,         # Add colorbar
       title="Sea Surface Temperature"
   )

You can also explicitly provide coordinates if needed:

.. code-block:: python

   # With explicit coordinates
   fig, ax, interpolator = nr.plot(
       temp.values, ds.lon.values, ds.lat.values,
       projection="rob"
   )

The ``plot`` function returns:

* ``fig``: The matplotlib Figure object
* ``ax``: The Cartopy GeoAxes object
* ``interpolator``: A :class:`~nereus.RegridInterpolator` for reuse

Available Projections
~~~~~~~~~~~~~~~~~~~~~

Nereus supports several map projections:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Alias
     - Name
     - Best for
   * - ``"pc"``
     - PlateCarree
     - Regional maps, default choice
   * - ``"rob"``
     - Robinson
     - Global maps
   * - ``"merc"``
     - Mercator
     - Navigation, mid-latitudes
   * - ``"moll"``
     - Mollweide
     - Global equal-area maps
   * - ``"np"``
     - North Polar Stereographic
     - Arctic region
   * - ``"sp"``
     - South Polar Stereographic
     - Antarctic region
   * - ``"ortho"``
     - Orthographic
     - Globe view from space

Regridding Data
---------------

To regrid data from an unstructured mesh to a regular grid:

.. code-block:: python

   # Simple regridding - coordinates extracted from xarray
   regridded, interpolator = nr.regrid(temp, resolution=1.0)

   # Or with explicit coordinates
   regridded, interpolator = nr.regrid(
       data, lon, lat,
       resolution=1.0,  # 1-degree resolution
   )

   # regridded is now a 2D array on a regular lon-lat grid
   print(regridded.shape)  # e.g., (180, 360)

For repeated operations on the same source grid, reuse the interpolator:

.. code-block:: python

   # Create interpolator once
   interpolator = nr.RegridInterpolator(lon, lat, resolution=0.5)

   # Use it multiple times
   sst_regridded = interpolator(sst_data)
   sss_regridded = interpolator(sss_data)
   mld_regridded = interpolator(mld_data)

Sea Ice Diagnostics
-------------------

Compute sea ice metrics:

.. code-block:: python

   # Load mesh with area information
   mesh = nr.fesom.load_mesh("/path/to/mesh")

   # Sea ice area (total area covered by ice)
   total_area = nr.ice_area(
       ds.a_ice,    # Ice concentration (0-1)
       mesh.area    # Cell areas in m²
   )
   print(f"Sea ice area: {total_area / 1e12:.2f} million km²")

   # Sea ice extent (area where concentration > 15%)
   extent = nr.ice_extent(
       ds.a_ice,
       mesh.area,
       threshold=0.15
   )

   # Sea ice volume
   volume = nr.ice_volume(
       ds.m_ice,    # Ice thickness (m)
       mesh.area,
       ds.a_ice     # Optional: weight by concentration
   )

Ocean Heat Content
------------------

Calculate volume-integrated quantities:

.. code-block:: python

   # Volume-weighted mean temperature
   mean_temp = nr.volume_mean(
       ds.temp,           # 3D temperature field
       mesh.area,         # Cell areas
       ds.thickness,      # Layer thicknesses
       ds.depth,          # Depth coordinates
       depth_min=0,       # Upper depth limit
       depth_max=700,     # Lower depth limit (m)
   )

   # Ocean heat content (in Joules)
   ohc = nr.heat_content(
       ds.temp,
       mesh.area,
       ds.thickness,
       ds.depth,
       depth_max=2000,
       reference_temp=0.0  # Reference temperature
   )

Vertical Transects
------------------

Create vertical cross-sections along a path:

.. code-block:: python

   # Define transect endpoints
   start = (-30, 60)   # (lon, lat) - North Atlantic
   end = (-30, -60)    # South Atlantic

   fig, ax = nr.transect(
       ds.temp,         # 3D data
       ds.lon, ds.lat,
       ds.depth,
       start, end,
       n_points=200,    # Number of points along path
       cmap="RdBu_r",
       depth_lim=(0, 5000),
       invert_depth=True
   )

Working with FESOM2 Meshes
--------------------------

Load and use FESOM2 mesh files:

.. code-block:: python

   # Load mesh
   mesh = nr.fesom.load_mesh("/path/to/mesh/")

   # Mesh properties
   print(f"Nodes: {mesh.n2d}")
   print(f"Depth levels: {mesh.nlev}")
   print(f"Depths: {mesh.depth}")

   # Open data with mesh coordinates
   ds = nr.fesom.open_dataset("fesom_output.nc", mesh=mesh)

   # Find nearest points to a location
   distances, indices = mesh.find_nearest(lon=-30, lat=45, k=5)

   # Subset by bounding box
   mask = mesh.subset_by_bbox(
       lon_min=-80, lon_max=0,
       lat_min=0, lat_max=60
   )
   atlantic_temp = ds.temp.where(mask)

Caching for Performance
-----------------------

Nereus automatically caches interpolation weights. For advanced control:

.. code-block:: python

   # Configure cache (optional)
   nr.set_cache_options(
       max_memory_items=20,    # Keep 20 interpolators in memory
       disk_path="/tmp/nereus_cache"  # Optional disk persistence
   )

   # Subsequent plots with same source coordinates are fast
   for t in range(100):
       data = ds.temp.isel(time=t).values
       fig, ax, _ = nr.plot(data, lon, lat, use_cache=True)
       plt.savefig(f"frame_{t:03d}.png")
       plt.close()

Next Steps
----------

* :doc:`concepts` - Understand the core concepts
* :doc:`../user_guide/plotting` - Detailed plotting guide
* :doc:`../user_guide/regridding` - Advanced regridding options
* :doc:`../user_guide/diagnostics` - All diagnostic functions
* :doc:`../api/top_level` - Full API reference
