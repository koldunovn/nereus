Model Support Guide
===================

Nereus provides specialized support for various atmospheric and ocean models with unstructured grids.

Unified Mesh System
-------------------

All meshes in nereus are represented as **xarray Datasets** with standardized variable names. This provides a consistent interface across all supported models.

**Key design principles:**

- **Mesh IS an xr.Dataset**: No special classes, just standard xarray
- **Standardized names**: ``lon``, ``lat``, ``area``, ``triangles``, ``depth``
- **Standalone spatial functions**: ``nr.find_nearest(lon, lat, ...)``
- **Auto dask detection**: Large meshes (>1M points) automatically use dask arrays

Standard Variable Names
~~~~~~~~~~~~~~~~~~~~~~~

All mesh datasets contain these standardized variables:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Dimension
     - Description
   * - ``lon``
     - ``(npoints,)``
     - Longitude, normalized to [-180, 180] degrees
   * - ``lat``
     - ``(npoints,)``
     - Latitude in degrees
   * - ``area``
     - ``(npoints,)``
     - Cell/node area in m²

Optional variables (model-dependent):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Dimension
     - Description
   * - ``triangles``
     - ``(nelem, 3)``
     - 0-indexed triangle connectivity
   * - ``lon_tri``, ``lat_tri``
     - ``(nelem,)``
     - Element center coordinates
   * - ``depth``
     - ``(depth_level,)``
     - Layer center depths (positive down)
   * - ``depth_bounds``
     - ``(depth_level, 2)``
     - Layer interfaces
   * - ``layer_thickness``
     - ``(depth_level,)``
     - Layer thickness in meters

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Module
     - Model
     - Status
   * - ``nr.fesom``
     - FESOM2
     - Fully implemented
   * - ``nr.healpix``
     - HEALPix grids
     - Fully implemented
   * - ``nr.nemo``
     - NEMO
     - Fully implemented
   * - ``nr.ifs_tco``
     - IFS TCO
     - Fully implemented
   * - ``nr.icono``
     - ICON-Ocean
     - Planned
   * - ``nr.icona``
     - ICON-Atmosphere
     - Planned
   * - ``nr.ifs``
     - IFS (ECMWF)
     - Planned

FESOM2
------

The Finite Element Sea ice-Ocean Model version 2 uses an unstructured triangular mesh.

Loading a FESOM2 Mesh
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nereus as nr

   # Load mesh from directory (returns xr.Dataset)
   mesh = nr.fesom.load_mesh("/path/to/mesh/")

   # Check what we got
   print(type(mesh))  # <class 'xarray.core.dataset.Dataset'>
   print(mesh.sizes)  # {'npoints': 126858, 'nelem': 244659, 'nz': 47, ...}

The mesh directory should contain:

* ``nod2d.out``: 2D node coordinates
* ``elem2d.out``: Element (triangle) connectivity
* ``mesh.diag.nc`` or ``fesom.mesh.diag.nc``: Areas and depth info
* ``aux3d.out`` (optional): Vertical level information

Accessing Mesh Data
~~~~~~~~~~~~~~~~~~~

Use standard xarray syntax to access mesh variables:

.. code-block:: python

   mesh = nr.fesom.load_mesh("/path/to/mesh/")

   # As xr.DataArray
   lon = mesh["lon"]
   lat = mesh["lat"]
   area = mesh["area"]

   # As numpy arrays (for use with other functions)
   lon_np = mesh["lon"].values
   lat_np = mesh["lat"].values
   area_np = mesh["area"].values

   # Check mesh attributes
   print(mesh.attrs["nereus_mesh_type"])  # 'fesom'
   print(mesh.attrs["nereus_dask_backend"])  # False (or True for large meshes)

Triangles and Element Centers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Triangle connectivity (0-indexed)
   triangles = mesh["triangles"].values  # Shape: (nelem, 3)

   # Element center coordinates (pre-computed)
   lon_tri = mesh["lon_tri"].values
   lat_tri = mesh["lat_tri"].values

Vertical Structure
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Layer center depths
   depth = mesh["depth"].values  # Shape: (nz,)

   # Layer bounds
   depth_bounds = mesh["depth_bounds"].values  # Shape: (nz, 2)

   # Layer thickness
   thickness = mesh["layer_thickness"].values  # Shape: (nz,)

Opening FESOM2 Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Open data file with mesh coordinates
   ds = nr.fesom.open_dataset("fesom_output.nc", mesh=mesh)

   # The dataset now has lon/lat as coordinates
   print(ds)

   # Plot directly
   fig, ax, _ = nr.plot(ds.temp.isel(time=0, nz1=0), mesh["lon"].values, mesh["lat"].values)

Spatial Queries
~~~~~~~~~~~~~~~

Use standalone functions with coordinate arrays:

.. code-block:: python

   # Find nearest mesh point to a location
   idx = nr.find_nearest(mesh["lon"].values, mesh["lat"].values, -30, 45)
   print(f"Nearest node index: {idx}")

   # With distance
   idx, dist = nr.find_nearest(mesh["lon"].values, mesh["lat"].values, -30, 45, return_distance=True)
   print(f"Distance: {dist/1000:.1f} km")

   # Find k nearest neighbors
   indices = nr.find_nearest(mesh["lon"].values, mesh["lat"].values, -30, 45, k=10)
   print(f"10 nearest nodes: {indices}")

   # Query multiple locations at once
   query_lons = np.array([-30, -40, -50])
   query_lats = np.array([45, 50, 55])
   indices = nr.find_nearest(mesh["lon"].values, mesh["lat"].values, query_lons, query_lats)

Geographic Subsetting
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get mask for bounding box
   mask = nr.subset_by_bbox(
       mesh["lon"].values, mesh["lat"].values,
       lon_min=-80, lon_max=0,
       lat_min=0, lat_max=65
   )

   # Apply mask to data
   atlantic_temp = temp.values[mask]
   atlantic_area = mesh["area"].values[mask]

   # Use mask in diagnostics
   atlantic_ice = nr.ice_area(conc, mesh["area"], mask=mask)

FESOM-Specific Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Interpolate node data to element centers
   temp_elements = nr.fesom.node_to_element(temp_nodes, mesh)

   # Interpolate element data back to nodes
   temp_nodes = nr.fesom.element_to_node(temp_elements, mesh)

   # Compute element centers (if not already present)
   mesh = nr.fesom.compute_element_centers(mesh)

HEALPix
-------

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) grids are used by ICON and other models.

Creating a HEALPix Mesh
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nereus as nr

   # Create mesh from number of points
   mesh = nr.healpix.load_mesh(3145728)  # nside=512

   # Or use helper function
   npoints = nr.healpix.nside_to_npoints(512)  # 3145728
   mesh = nr.healpix.load_mesh(npoints)

   # Find appropriate nside for desired resolution
   nside = nr.healpix.resolution_to_nside(1.0)  # ~1 degree -> nside=64

HEALPix Properties
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # All pixels have equal area
   pixel_area = mesh["area"].values[0]  # Same for all pixels
   print(f"Pixel area: {pixel_area/1e6:.1f} km²")

   # Access coordinates
   lon = mesh["lon"].values
   lat = mesh["lat"].values

   # Mesh attributes
   print(mesh.attrs["nside"])  # 512
   print(mesh.attrs["ordering"])  # 'NESTED' or 'RING'

NEMO
----

NEMO (Nucleus for European Modelling of the Ocean) uses structured grids that are flattened for compatibility.

Loading a NEMO Mesh
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nereus as nr

   # Load from mesh_mask.nc file
   mesh = nr.nemo.load_mesh("/path/to/mesh_mask.nc")

   # Coordinates are flattened from 2D to 1D
   print(mesh.sizes["npoints"])  # Total ocean points

   # Original 2D shape is stored in attributes
   print(f"Original shape: {mesh.attrs['nlon']} x {mesh.attrs['nlat']}")

IFS TCO
-------

IFS TCO meshes use separate grid and area files with ``A*`` variables for
longitude, latitude, and cell surface area.

Loading an IFS TCO Mesh
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nereus as nr

   mesh = nr.ifs_tco.load_mesh(
       "/path/to/tco_grid.nc",
       "/path/to/tco_areas.nc",
   )

   print(mesh.sizes["npoints"])
   print(mesh["lon"].values[:5])

Universal Mesh Loader
---------------------

The ``nr.load_mesh()`` function auto-detects mesh type:

.. code-block:: python

   # Auto-detect FESOM mesh
   mesh = nr.load_mesh("/path/to/fesom/mesh/")

   # Auto-detect NEMO mesh
   mesh = nr.load_mesh("/path/to/mesh_mask.nc")

   # Explicit type
   mesh = nr.load_mesh("/path/to/mesh/", mesh_type="fesom")

   # HEALPix from npoints
   mesh = nr.load_mesh(3145728, mesh_type="healpix")

Creating Regular Lon-Lat Grids
------------------------------

Create regular grids as mesh datasets for comparison or regridding targets:

.. code-block:: python

   # 1-degree global grid
   mesh_1deg = nr.create_lonlat_mesh(1.0)

   # Different resolution in lon/lat
   mesh = nr.create_lonlat_mesh((0.5, 0.25))  # 0.5° lon, 0.25° lat

   # Regional grid
   mesh_regional = nr.create_lonlat_mesh(
       0.25,
       lon_bounds=(-80, 0),
       lat_bounds=(20, 70)
   )

Creating Meshes from Arrays
---------------------------

Create mesh datasets from existing coordinate arrays:

.. code-block:: python

   import numpy as np

   # From 1D arrays
   lon = np.array([...])
   lat = np.array([...])
   area = np.array([...])  # Optional

   mesh = nr.mesh_from_arrays(lon, lat, area=area)

   # 2D arrays are automatically flattened
   lon_2d = np.array([[...]])
   lat_2d = np.array([[...]])

   mesh = nr.mesh_from_arrays(lon_2d, lat_2d)

Using Meshes with Diagnostics
-----------------------------

Mesh area integrates directly with diagnostic functions:

.. code-block:: python

   # Sea ice area
   ice_area = nr.ice_area(sic, mesh["area"], mask=mesh["lat"].values > 0)

   # Ice extent
   ice_extent = nr.ice_extent(sic, mesh["area"], threshold=0.15)

   # Surface mean
   sst_mean = nr.surface_mean(sst, mesh["area"])

   # Volume mean (3D data)
   temp_mean = nr.volume_mean(
       temp_3d,
       mesh["area"],
       mesh["layer_thickness"],
       mesh["depth"],
       depth_max=500
   )

Mesh Metadata
-------------

All nereus meshes include metadata attributes:

.. code-block:: python

   mesh.attrs["nereus_mesh_type"]     # 'fesom', 'healpix', 'nemo', 'lonlat'
   mesh.attrs["nereus_mesh_version"]  # '1.0'
   mesh.attrs["nereus_dask_backend"]  # True/False
   mesh.attrs["nereus_source_path"]   # Original file path (if applicable)

Check if a dataset is a nereus mesh:

.. code-block:: python

   from nereus.core.mesh import is_nereus_mesh, get_mesh_type

   if is_nereus_mesh(ds):
       print(f"Mesh type: {get_mesh_type(ds)}")

Dask Support
------------

Large meshes (>1M points) automatically use dask arrays:

.. code-block:: python

   # Auto-detect (>1M points uses dask)
   mesh = nr.fesom.load_mesh("/path/to/large/mesh/")
   print(mesh.attrs["nereus_dask_backend"])  # True

   # Force dask for smaller meshes
   mesh = nr.fesom.load_mesh("/path/to/mesh/", use_dask=True)

   # Disable dask for large meshes
   mesh = nr.fesom.load_mesh("/path/to/mesh/", use_dask=False)

   # Check threshold
   from nereus.core.mesh import DASK_THRESHOLD_POINTS
   print(f"Dask threshold: {DASK_THRESHOLD_POINTS:,} points")  # 1,000,000
