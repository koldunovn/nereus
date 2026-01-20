Plotting Guide
==============

Nereus provides powerful plotting capabilities for visualizing unstructured model data on maps and as vertical transects.

2D Map Plotting
---------------

The :func:`~nereus.plot` function is the primary tool for creating map visualizations:

.. code-block:: python

   import nereus as nr

   fig, ax, interpolator = nr.plot(
       data, lon, lat,
       projection="rob",
       cmap="viridis"
   )

Basic Parameters
~~~~~~~~~~~~~~~~

**data** : array-like
   1D array of values at each mesh point.

**lon, lat** : array-like
   1D arrays of longitude and latitude coordinates (in degrees).

**projection** : str, default "pc"
   Map projection to use. See :ref:`projections` below.

**extent** : tuple, optional
   Map extent as ``(lon_min, lon_max, lat_min, lat_max)``.

**resolution** : float or tuple, default 1.0
   Target grid resolution. Either degrees (float) or ``(nlon, nlat)``.

Appearance Parameters
~~~~~~~~~~~~~~~~~~~~~

**cmap** : str, default "viridis"
   Matplotlib colormap name.

**vmin, vmax** : float, optional
   Color scale limits. If not specified, uses data min/max.

**coastlines** : bool, default True
   Draw coastlines on the map.

**land** : bool, default False
   Fill land areas with gray color.

**gridlines** : bool, default False
   Draw latitude/longitude gridlines.

**colorbar** : bool, default True
   Add a colorbar to the plot.

**colorbar_label** : str, optional
   Label for the colorbar.

**title** : str, optional
   Plot title.

**figsize** : tuple, optional
   Figure size as ``(width, height)`` in inches.

Advanced Parameters
~~~~~~~~~~~~~~~~~~~

**ax** : matplotlib Axes, optional
   Existing axes to plot on. Must be a Cartopy GeoAxes.

**interpolator** : RegridInterpolator, optional
   Pre-computed interpolator to reuse.

**influence_radius** : float, default 80000.0
   Maximum distance (meters) from data points for interpolation.

**use_cache** : bool, default True
   Use cached interpolator if available.

.. _projections:

Map Projections
~~~~~~~~~~~~~~~

Nereus supports multiple map projections via short aliases:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Alias
     - Projection
     - Description
   * - ``"pc"``
     - PlateCarree
     - Simple equirectangular projection. Good default for regional maps.
   * - ``"rob"``
     - Robinson
     - Compromise projection for global maps. Good for world maps.
   * - ``"merc"``
     - Mercator
     - Conformal projection. Preserves angles but distorts area at high latitudes.
   * - ``"moll"``
     - Mollweide
     - Equal-area projection. Good for showing global distributions.
   * - ``"np"``
     - North Polar Stereographic
     - Centered on North Pole. Ideal for Arctic visualization.
   * - ``"sp"``
     - South Polar Stereographic
     - Centered on South Pole. Ideal for Antarctic visualization.
   * - ``"ortho"``
     - Orthographic
     - View of Earth from space. Requires ``central_longitude`` and ``central_latitude``.
   * - ``"lcc"``
     - Lambert Conformal
     - Good for mid-latitude regions with east-west extent.

Examples with different projections:

.. code-block:: python

   # Global Robinson projection
   nr.plot(data, lon, lat, projection="rob")

   # Arctic view
   nr.plot(data, lon, lat, projection="np")

   # Antarctic view
   nr.plot(data, lon, lat, projection="sp")

   # Regional PlateCarree with extent
   nr.plot(data, lon, lat, projection="pc",
           extent=(-80, 0, 0, 60))  # North Atlantic

   # Orthographic (globe view)
   nr.plot(data, lon, lat, projection="ortho",
           central_longitude=-30, central_latitude=45)

Polar Projections
~~~~~~~~~~~~~~~~~

For polar projections (``"np"`` and ``"sp"``), Nereus automatically:

* Expands the data bounds to avoid edge artifacts
* Uses circular map boundary
* Sets appropriate extent

.. code-block:: python

   # Arctic sea ice
   fig, ax, _ = nr.plot(
       ice_concentration, lon, lat,
       projection="np",
       cmap="Blues",
       vmin=0, vmax=1,
       title="Arctic Sea Ice Concentration"
   )

Reusing Interpolators
~~~~~~~~~~~~~~~~~~~~~

For efficiency when plotting multiple fields on the same mesh:

.. code-block:: python

   # First plot creates the interpolator
   fig1, ax1, interp = nr.plot(temp, lon, lat, projection="rob")

   # Subsequent plots reuse it (much faster)
   fig2, ax2, _ = nr.plot(salt, lon, lat, projection="rob", interpolator=interp)
   fig3, ax3, _ = nr.plot(speed, lon, lat, projection="rob", interpolator=interp)

Creating Animations
~~~~~~~~~~~~~~~~~~~

Efficiently create frame sequences:

.. code-block:: python

   import matplotlib.pyplot as plt

   interpolator = None
   for t in range(len(ds.time)):
       data = ds.temp.isel(time=t, nz1=0).values

       fig, ax, interpolator = nr.plot(
           data, lon, lat,
           projection="rob",
           interpolator=interpolator,  # Reuse after first frame
           vmin=-2, vmax=30,
           cmap="RdBu_r",
           title=f"Time: {t}"
       )

       plt.savefig(f"frame_{t:04d}.png", dpi=150, bbox_inches="tight")
       plt.close()

Vertical Transects
------------------

The :func:`~nereus.transect` function creates vertical cross-sections along a great circle path:

.. code-block:: python

   fig, ax = nr.transect(
       data_3d, lon, lat, depth,
       start=(lon1, lat1),
       end=(lon2, lat2),
       n_points=200
   )

Parameters
~~~~~~~~~~

**data** : array-like
   2D array with shape ``(nz, npoints)`` or 3D with time dimension.

**lon, lat** : array-like
   1D coordinate arrays.

**depth** : array-like
   1D array of depth levels (positive downward).

**start, end** : tuple
   Start and end points as ``(lon, lat)`` tuples.

**n_points** : int, default 100
   Number of points along the transect.

**depth_lim** : tuple, optional
   Depth limits as ``(min, max)``.

**invert_depth** : bool, default True
   Invert y-axis so depth increases downward.

Example Transects
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Atlantic meridional transect
   fig, ax = nr.transect(
       ds.temp.isel(time=0),
       ds.lon, ds.lat, ds.depth,
       start=(-30, 60),    # North Atlantic
       end=(-30, -60),     # South Atlantic
       n_points=300,
       cmap="RdBu_r",
       vmin=-2, vmax=25,
       depth_lim=(0, 5000),
       colorbar_label="Temperature (°C)",
       title="Atlantic Temperature Transect"
   )

   # Drake Passage transect
   fig, ax = nr.transect(
       ds.u.isel(time=0),
       ds.lon, ds.lat, ds.depth,
       start=(-70, -55),
       end=(-55, -62),
       cmap="RdBu_r",
       depth_lim=(0, 3000),
       colorbar_label="Zonal Velocity (m/s)"
   )

Multi-Panel Figures
-------------------

Combining multiple plots:

.. code-block:: python

   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs

   fig = plt.figure(figsize=(14, 6))

   # Temperature map
   ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
   nr.plot(temp, lon, lat, ax=ax1, cmap="RdBu_r",
           title="Temperature", colorbar_label="°C")

   # Salinity map
   ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())
   nr.plot(salt, lon, lat, ax=ax2, cmap="viridis",
           title="Salinity", colorbar_label="PSU")

   plt.tight_layout()

Customizing Plots
-----------------

The returned axes can be further customized:

.. code-block:: python

   fig, ax, _ = nr.plot(data, lon, lat, projection="rob")

   # Add custom features
   ax.set_title("Custom Title", fontsize=14, fontweight="bold")

   # Add markers
   ax.plot(-30, 45, 'r*', markersize=15, transform=ccrs.PlateCarree())

   # Add text
   ax.text(-30, 40, "Point A", transform=ccrs.PlateCarree(),
           fontsize=10, ha="center")

   # Customize gridlines
   gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
   gl.top_labels = False
   gl.right_labels = False

Handling Missing Data
---------------------

Nereus handles NaN values automatically:

.. code-block:: python

   # NaN values in data are preserved in output
   data_with_nans = np.where(land_mask, np.nan, data)
   nr.plot(data_with_nans, lon, lat)

The ``influence_radius`` parameter also creates a natural mask where no source data exists:

.. code-block:: python

   # Smaller influence radius = tighter mask around data
   nr.plot(data, lon, lat, influence_radius=50000)  # 50 km

   # Larger influence radius = fills more gaps
   nr.plot(data, lon, lat, influence_radius=200000)  # 200 km
