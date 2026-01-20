Model Support Guide
===================

Nereus provides specialized support for various atmospheric and ocean models with unstructured grids.

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
   * - ``nr.icono``
     - ICON-Ocean
     - Planned
   * - ``nr.icona``
     - ICON-Atmosphere
     - Planned
   * - ``nr.ifs``
     - IFS (ECMWF)
     - Planned
   * - ``nr.healpix``
     - HEALPix grids
     - Planned

FESOM2
------

The Finite Element Sea ice-Ocean Model version 2 uses an unstructured triangular mesh.

Loading a FESOM2 Mesh
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nereus as nr

   # Load mesh from directory
   mesh = nr.fesom.load_mesh("/path/to/mesh/")

The mesh directory should contain:

* ``nod2d.out``: 2D node coordinates
* ``elem2d.out``: Element (triangle) connectivity
* ``mesh.diag.nc`` or ``fesom.mesh.diag.nc``: Cluster areas and optional depth info
* ``aux3d.out`` (optional): Vertical level information

Mesh Properties
~~~~~~~~~~~~~~~

.. code-block:: python

   mesh = nr.fesom.load_mesh("/path/to/mesh/")

   # Basic properties
   print(f"2D nodes: {mesh.n2d}")
   print(f"3D nodes: {mesh.n3d}")
   print(f"Vertical levels: {mesh.nlev}")

   # Coordinate arrays
   lon = mesh.lon  # 1D array of longitudes
   lat = mesh.lat  # 1D array of latitudes
   area = mesh.area  # Cluster areas in m²

   # Vertical structure
   depth = mesh.depth  # Mid-level depths
   depth_lev = mesh.depth_lev  # Level interfaces

   # Element connectivity
   elem = mesh.elem  # Triangle indices (n_elem, 3)

Opening FESOM2 Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Open data file with mesh coordinates
   ds = nr.fesom.open_dataset("fesom_output.nc", mesh=mesh)

   # The dataset now has lon/lat as coordinates
   print(ds)

   # Access data
   temp = ds.temp.isel(time=0, nz1=0)

   # Plot directly
   fig, ax, _ = nr.plot(temp, ds.lon, ds.lat, projection="rob")

Spatial Queries
~~~~~~~~~~~~~~~

Find nearest mesh points to a location:

.. code-block:: python

   # Find single nearest point
   distances, indices = mesh.find_nearest(lon=-30, lat=45)
   print(f"Nearest node index: {indices[0]}")
   print(f"Distance: {distances[0]/1000:.1f} km")

   # Find k nearest neighbors
   distances, indices = mesh.find_nearest(lon=-30, lat=45, k=10)
   print(f"10 nearest nodes: {indices}")

Geographic Subsetting
~~~~~~~~~~~~~~~~~~~~~

Create masks for geographic regions:

.. code-block:: python

   # Get mask for bounding box
   mask = mesh.subset_by_bbox(
       lon_min=-80, lon_max=0,
       lat_min=0, lat_max=65
   )

   # Apply mask to data
   atlantic_temp = temp.values[mask]
   atlantic_lon = mesh.lon[mask]
   atlantic_lat = mesh.lat[mask]
   atlantic_area = mesh.area[mask]

   # Use mask in diagnostics
   atlantic_ice = nr.ice_area(conc, mesh.area, mask=mask)

Working with 3D Data
~~~~~~~~~~~~~~~~~~~~

FESOM2 uses a z-level coordinate system:

.. code-block:: python

   # Access 3D temperature
   temp_3d = ds.temp.isel(time=0)  # Shape: (nz, npoints)

   # Volume mean for specific depth range
   mean_temp = nr.volume_mean(
       temp_3d,
       mesh.area,
       ds.thickness,  # Layer thicknesses
       mesh.depth,
       depth_max=500
   )

   # Vertical transect
   fig, ax = nr.transect(
       temp_3d, mesh.lon, mesh.lat, mesh.depth,
       start=(-30, 60), end=(-30, -60)
   )

FESOM2 File Formats
~~~~~~~~~~~~~~~~~~~

FESOM2 mesh files can be in different formats:

**ASCII format** (original):

.. code-block:: text

   # nod2d.out - each line: node_id lon lat
   1 -180.0 -80.0
   2 -179.5 -80.0
   ...

   # elem2d.out - each line: elem_id node1 node2 node3
   1 1 2 100
   2 2 3 101
   ...

**NetCDF format** (mesh.diag.nc):

.. code-block:: text

   Dimensions:
       nod2: number of 2D nodes
       elem: number of elements

   Variables:
       cluster_area(nod2): cell areas in m²
       ...

The ``load_mesh`` function automatically detects and handles both formats.

The Mesh Protocol
-----------------

All model meshes follow a common interface defined by :class:`~nereus.models._base.MeshBase`:

.. code-block:: python

   class MeshBase(ABC):
       """Abstract base class for model meshes."""

       @property
       def lon(self) -> NDArray:
           """Longitude coordinates in degrees."""

       @property
       def lat(self) -> NDArray:
           """Latitude coordinates in degrees."""

       @property
       def area(self) -> NDArray:
           """Cell areas in square meters."""

       @property
       def npoints(self) -> int:
           """Number of mesh points."""

       def find_nearest(self, lon, lat, k=1):
           """Find k nearest mesh points."""

       def subset_by_bbox(self, lon_min, lon_max, lat_min, lat_max):
           """Get mask for points in bounding box."""

This allows writing generic code that works with any supported model:

.. code-block:: python

   def compute_regional_stats(data, mesh, lon_min, lon_max, lat_min, lat_max):
       """Works with any mesh that follows MeshBase."""
       mask = mesh.subset_by_bbox(lon_min, lon_max, lat_min, lat_max)
       return nr.volume_mean(data, mesh.area[mask], ...)

Future Model Support
--------------------

ICON-Ocean (planned)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Planned API
   mesh = nr.icono.load_mesh("icon_ocean_mesh.nc")

   # Similar interface to FESOM
   fig, ax, _ = nr.plot(data, mesh.lon, mesh.lat)

ICON-Atmosphere (planned)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Planned API
   mesh = nr.icona.load_mesh("icon_atmo_mesh.nc")

IFS/ECMWF (planned)
~~~~~~~~~~~~~~~~~~~

Support for ECMWF's Tco spectral transform grids:

.. code-block:: python

   # Planned API
   mesh = nr.ifs.load_grid("ifs_Tco1279.nc")

HEALPix (planned)
~~~~~~~~~~~~~~~~~

Support for HEALPix hierarchical grids:

.. code-block:: python

   # Planned API
   mesh = nr.healpix.load_grid(nside=64)

Using Custom Meshes
-------------------

For models not yet supported, you can work directly with coordinates:

.. code-block:: python

   import xarray as xr

   # Load your mesh data
   ds = xr.open_dataset("custom_mesh.nc")

   lon = ds.longitude.values
   lat = ds.latitude.values
   area = ds.cell_area.values  # If available

   # Use Nereus functions directly
   fig, ax, interp = nr.plot(data, lon, lat, projection="rob")

   # For diagnostics requiring area, provide it explicitly
   ice_area = nr.ice_area(concentration, area)

You can also create a simple mesh wrapper:

.. code-block:: python

   class CustomMesh:
       def __init__(self, lon, lat, area):
           self._lon = lon
           self._lat = lat
           self._area = area

       @property
       def lon(self):
           return self._lon

       @property
       def lat(self):
           return self._lat

       @property
       def area(self):
           return self._area

   # Use with Nereus
   mesh = CustomMesh(lon, lat, area)
   ice_area = nr.ice_area(conc, mesh.area)
