Diagnostics Guide
=================

Nereus provides a suite of diagnostic functions for analyzing ocean and sea ice model output.

Sea Ice Diagnostics
-------------------

Sea Ice Area
~~~~~~~~~~~~

Compute the total area covered by sea ice:

.. code-block:: python

   import nereus as nr

   # Total sea ice area (m²)
   area = nr.ice_area(concentration, cell_area)

   # Convert to million km²
   area_mkm2 = area / 1e12
   print(f"Sea ice area: {area_mkm2:.2f} million km²")

**Parameters:**

- ``concentration``: Ice concentration (0-1)
- ``area``: Cell area in m² (from mesh)
- ``mask``: Optional boolean mask to limit region

**Formula:**

.. math::

   A_{ice} = \sum_i c_i \cdot A_i

where :math:`c_i` is ice concentration and :math:`A_i` is cell area.

Sea Ice Extent
~~~~~~~~~~~~~~

Compute the area where ice concentration exceeds a threshold:

.. code-block:: python

   # Sea ice extent with standard 15% threshold
   extent = nr.ice_extent(concentration, cell_area, threshold=0.15)

   # Alternative thresholds
   extent_strict = nr.ice_extent(concentration, cell_area, threshold=0.30)

**Formula:**

.. math::

   E_{ice} = \sum_i A_i \cdot \mathbf{1}_{c_i \geq \tau}

where :math:`\tau` is the threshold (default 0.15).

Sea Ice Volume
~~~~~~~~~~~~~~

Compute total sea ice volume:

.. code-block:: python

   # Simple: thickness × area
   volume = nr.ice_volume(thickness, cell_area)

   # With concentration weighting
   volume = nr.ice_volume(thickness, cell_area, concentration=conc)

   # Convert to thousand km³
   volume_kkm3 = volume / 1e12
   print(f"Sea ice volume: {volume_kkm3:.1f} thousand km³")

**Formula:**

.. math::

   V_{ice} = \sum_i h_i \cdot c_i \cdot A_i

Hemisphere Convenience Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the common case of computing ice metrics for Northern or Southern Hemisphere:

.. code-block:: python

   # Northern Hemisphere sea ice
   nh_area = nr.ice_area_nh(conc, area, lat)
   nh_extent = nr.ice_extent_nh(conc, area, lat)
   nh_volume = nr.ice_volume_nh(thick, area, lat)

   # Southern Hemisphere sea ice
   sh_area = nr.ice_area_sh(conc, area, lat)
   sh_extent = nr.ice_extent_sh(conc, area, lat)
   sh_volume = nr.ice_volume_sh(thick, area, lat)

   # With concentration for real thickness
   nh_volume = nr.ice_volume_nh(sithick, area, lat, concentration=siconc)

These functions automatically create the hemisphere mask from latitude, saving you
from having to define ``mask=lat > 0`` or ``mask=lat < 0`` each time.

Regional Analysis
~~~~~~~~~~~~~~~~~

Use masks to compute regional statistics for custom regions:

.. code-block:: python

   # Arctic only (lat > 60°N)
   arctic_mask = lat > 60

   arctic_area = nr.ice_area(conc, area, mask=arctic_mask)
   arctic_extent = nr.ice_extent(conc, area, mask=arctic_mask)
   arctic_volume = nr.ice_volume(thick, area, conc, mask=arctic_mask)

   # Antarctic only (lat < -60°S)
   antarctic_mask = lat < -60
   antarctic_area = nr.ice_area(conc, area, mask=antarctic_mask)

Ocean Diagnostics
-----------------

Surface Mean (SST, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~

Compute area-weighted mean of a 2D field (single level):

.. code-block:: python

   # Global mean SST
   mean_sst = nr.surface_mean(sst, area)

   # Mean SST in a region
   tropical_mask = np.abs(lat) < 23.5
   tropical_sst = nr.surface_mean(sst, area, mask=tropical_mask)

   # Mean at a specific depth level
   temp_500m = ds.temp.isel(nz1=10)  # Select 500m level
   mean_temp_500m = nr.surface_mean(temp_500m, area)

**Formula:**

.. math::

   \bar{X} = \frac{\sum_i X_i \cdot A_i}{\sum_i A_i}

where :math:`X_i` is the value at cell :math:`i` and :math:`A_i` is cell area.

Volume Mean
~~~~~~~~~~~

Compute volume-weighted mean of any quantity:

.. code-block:: python

   # Mean temperature in upper 700m
   mean_temp = nr.volume_mean(
       temp,           # 2D array (nz, npoints)
       area,           # Cell areas
       thickness,      # Layer thicknesses
       depth,          # Depth levels
       depth_max=700   # Depth limit (m)
   )

   # Mean temperature between 200-2000m
   mid_temp = nr.volume_mean(
       temp, area, thickness, depth,
       depth_min=200,
       depth_max=2000
   )

**Formula:**

.. math::

   \bar{T} = \frac{\sum_{i,k} T_{i,k} \cdot A_i \cdot \Delta z_k}{\sum_{i,k} A_i \cdot \Delta z_k}

Ocean Heat Content
~~~~~~~~~~~~~~~~~~

Compute integrated ocean heat content:

.. code-block:: python

   # Total OHC in upper 2000m (scalar value)
   ohc = nr.heat_content(
       temp, area, thickness, depth,
       depth_max=2000,
       reference_temp=0.0  # Reference temperature
   )

   # Convert to ZJ (zettajoules)
   ohc_zj = ohc / 1e21
   print(f"Ocean heat content: {ohc_zj:.1f} ZJ")

   # OHC map (J/m² at each point, like FESOM2 output)
   ohc_map = nr.heat_content(
       temp, area, thickness, depth,
       depth_max=2000,
       output="map"
   )

   # Plot the OHC map
   fig, ax, _ = nr.plot(ohc_map, lon, lat, cmap="Reds")

**Parameters:**

- ``temperature``: Temperature field (°C)
- ``area``: Cell areas (m²)
- ``thickness``: Layer thicknesses (m)
- ``depth``: Depth coordinates (m)
- ``reference_temp``: Reference temperature (default 0°C)
- ``rho``: Seawater density (default 1025 kg/m³)
- ``cp``: Specific heat capacity (default 3990 J/(kg·K))
- ``output``: Output type - ``"total"`` for scalar (Joules) or ``"map"`` for per-point (J/m²)

**Formulas:**

Total heat content (``output="total"``):

.. math::

   OHC = \rho \cdot c_p \cdot \sum_{i,k} (T_{i,k} - T_{ref}) \cdot A_i \cdot \Delta z_k

Heat content map (``output="map"``):

.. math::

   OHC_i = \rho \cdot c_p \cdot \sum_{k} (T_{i,k} - T_{ref}) \cdot \Delta z_k

Depth Filtering
~~~~~~~~~~~~~~~

Both ``volume_mean`` and ``heat_content`` support depth filtering:

.. code-block:: python

   # Full water column
   full_mean = nr.volume_mean(temp, area, thickness)

   # Upper ocean (0-300m)
   upper_mean = nr.volume_mean(temp, area, thickness, depth, depth_max=300)

   # Intermediate water (300-1500m)
   inter_mean = nr.volume_mean(temp, area, thickness, depth,
                                depth_min=300, depth_max=1500)

   # Deep ocean (>1500m)
   deep_mean = nr.volume_mean(temp, area, thickness, depth, depth_min=1500)

Hovmoller Diagrams
------------------

Compute and plot Hovmoller diagrams showing evolution over time:

Computing Hovmoller Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Depth-time Hovmoller
   time, depth_out, hov_data = nr.hovmoller(
       temp,           # (time, depth, npoints)
       area,           # Cell areas for weighting
       time=time_coord,
       depth=depth_coord,
       mode="depth"
   )

   # Latitude-time Hovmoller
   time, lat_out, hov_data = nr.hovmoller(
       sst,            # (time, npoints)
       area,
       time=time_coord,
       lat=lat_coord,
       mode="latitude",
       lat_bins=np.arange(-90, 91, 5)  # 5° bins
   )

Plotting Hovmoller
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot depth-time Hovmoller
   fig, ax = nr.plot_hovmoller(
       time, depth_out, hov_data,
       mode="depth",
       cmap="RdBu_r",
       vmin=-2, vmax=25,
       colorbar_label="Temperature (°C)",
       title="Temperature Evolution"
   )

   # Plot latitude-time Hovmoller
   fig, ax = nr.plot_hovmoller(
       time, lat_out, hov_data,
       mode="latitude",
       cmap="RdBu_r",
       invert_y=False,  # Don't invert for latitude
       title="SST Zonal Mean"
   )

Working with Time Series
------------------------

Time Series of Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~

For time-varying data:

.. code-block:: python

   import numpy as np

   # Compute time series of sea ice area
   n_times = len(ds.time)
   ice_area_ts = np.zeros(n_times)

   for t in range(n_times):
       ice_area_ts[t] = nr.ice_area(
           ds.a_ice.isel(time=t).values,
           mesh.area
       )

   # Plot time series
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 4))
   plt.plot(ds.time, ice_area_ts / 1e12)
   plt.ylabel("Sea Ice Area (million km²)")
   plt.xlabel("Time")

Dask Integration
~~~~~~~~~~~~~~~~

Diagnostics work with Dask arrays:

.. code-block:: python

   import dask.array as da

   # Load data lazily
   ds = xr.open_mfdataset("output_*.nc", chunks={"time": 10})

   # Compute diagnostics (stays lazy)
   ice_area_lazy = nr.ice_area(ds.a_ice, mesh.area)

   # Trigger computation
   ice_area_values = ice_area_lazy.compute()

Regional Masks
--------------

Simple Masks
~~~~~~~~~~~~

Creating and using simple regional masks:

.. code-block:: python

   # Simple latitude-based mask
   tropical_mask = np.abs(lat) < 23.5
   arctic_mask = lat > 66.5
   antarctic_mask = lat < -66.5

   # Bounding box mask
   north_atlantic = (
       (lon > -80) & (lon < 0) &
       (lat > 0) & (lat < 65)
   )

   # Using mesh method
   bbox_mask = mesh.subset_by_bbox(
       lon_min=-80, lon_max=0,
       lat_min=0, lat_max=65
   )

   # Apply to diagnostics
   na_ohc = nr.heat_content(
       temp, area, thickness, depth,
       depth_max=2000,
       mask=north_atlantic
   )

GeoJSON Region Masks
~~~~~~~~~~~~~~~~~~~~

Nereus includes built-in GeoJSON files with predefined ocean basins and regions.
These can be used to create masks for standard oceanographic regions.

**Available GeoJSON Files:**

- ``MOCBasins``: Regions for Meridional Overturning Circulation analysis
- ``oceanBasins``: Major ocean basins (Atlantic, Pacific, Indian, etc.)
- ``NinoRegions``: El Nino monitoring regions (Nino 3, Nino 3.4, Nino 4)

**Listing Available Regions:**

.. code-block:: python

   import nereus as nr

   # List regions in MOCBasins
   nr.list_available_regions("MOCBasins")
   # ['Atlantic_MOC', 'IndoPacific_MOC', 'Pacific_MOC', 'Indian_MOC']

   # List regions in oceanBasins
   nr.list_available_regions("oceanBasins")
   # ['Atlantic_Basin', 'Pacific_Basin', 'Indian_Basin', 'Arctic_Basin',
   #  'Southern_Ocean_Basin', 'Mediterranean_Basin', 'Global Ocean', ...]

   # List Nino regions
   nr.list_available_regions("NinoRegions")
   # ['Nino 3.4', 'Nino 3', 'Nino 4']

**Creating Region Masks:**

.. code-block:: python

   import nereus as nr
   import numpy as np

   # Get coordinates from your data/mesh
   lon = mesh.lon  # 1D array of longitudes
   lat = mesh.lat  # 1D array of latitudes

   # Create mask for Atlantic basin
   atlantic_mask = nr.get_region_mask(lon, lat, "Atlantic_Basin", "oceanBasins")

   # Create mask for Atlantic MOC region
   amoc_mask = nr.get_region_mask(lon, lat, "Atlantic_MOC", "MOCBasins")

   # Create mask for Nino 3.4 region
   nino34_mask = nr.get_region_mask(lon, lat, "Nino 3.4", "NinoRegions")

**Using Region Masks with Diagnostics:**

.. code-block:: python

   # Compute Atlantic Ocean heat content
   atlantic_mask = nr.get_region_mask(lon, lat, "Atlantic_Basin", "oceanBasins")

   atlantic_ohc = nr.heat_content(
       temp, area, thickness, depth,
       depth_max=2000,
       mask=atlantic_mask
   )

   # Compute sea ice area in the Arctic basin
   arctic_mask = nr.get_region_mask(lon, lat, "Arctic_Basin", "oceanBasins")
   arctic_ice_area = nr.ice_area(conc, area, mask=arctic_mask)

   # Compute mean SST in Nino 3.4 region
   nino34_mask = nr.get_region_mask(lon, lat, "Nino 3.4", "NinoRegions")
   nino34_sst = np.average(sst[nino34_mask], weights=area[nino34_mask])

**Loading Raw GeoJSON Data:**

For advanced use cases, you can load the raw GeoJSON data:

.. code-block:: python

   geojson = nr.load_geojson("oceanBasins")

   # Access features
   for feature in geojson["features"]:
       name = feature["properties"]["name"]
       geometry = feature["geometry"]
       print(f"{name}: {geometry['type']}")

.. note::

   The ``get_region_mask`` function requires ``shapely>=2.0`` to be installed.
   Install it with: ``pip install shapely>=2.0``

Anomalies and Trends
--------------------

Computing anomalies relative to climatology:

.. code-block:: python

   # Compute climatological mean
   clim_mean = np.zeros(ds.temp.shape[1:])  # (nz, npoints)
   for t in range(len(ds.time)):
       clim_mean += ds.temp.isel(time=t).values
   clim_mean /= len(ds.time)

   # Compute anomaly time series
   anomalies = []
   for t in range(len(ds.time)):
       anom = ds.temp.isel(time=t).values - clim_mean
       mean_anom = nr.volume_mean(anom, mesh.area, thickness, depth)
       anomalies.append(mean_anom)

   anomalies = np.array(anomalies)

Summary Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Function
     - Purpose
     - Output Unit
   * - ``ice_area``
     - Total ice-covered area
     - m² (→ million km²)
   * - ``ice_area_nh``, ``ice_area_sh``
     - Hemisphere ice area
     - m² (→ million km²)
   * - ``ice_extent``
     - Area with ice > threshold
     - m² (→ million km²)
   * - ``ice_extent_nh``, ``ice_extent_sh``
     - Hemisphere ice extent
     - m² (→ million km²)
   * - ``ice_volume``
     - Total ice volume
     - m³ (→ thousand km³)
   * - ``ice_volume_nh``, ``ice_volume_sh``
     - Hemisphere ice volume
     - m³ (→ thousand km³)
   * - ``surface_mean``
     - Area-weighted mean (2D field)
     - Same as input
   * - ``volume_mean``
     - Volume-weighted mean (3D field)
     - Same as input
   * - ``heat_content``
     - Integrated OHC (total or map)
     - J (→ ZJ) or J/m²
   * - ``hovmoller``
     - Time-depth/lat arrays
     - Tuple of arrays
   * - ``plot_hovmoller``
     - Hovmoller visualization
     - (fig, ax)
   * - ``get_region_mask``
     - Boolean mask from GeoJSON region
     - np.ndarray (bool)
   * - ``list_available_regions``
     - List region names in GeoJSON file
     - list[str]
   * - ``load_geojson``
     - Load raw GeoJSON data
     - dict
