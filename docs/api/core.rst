Core Utilities
==============

The ``nereus.core`` module provides fundamental utilities for coordinate transformations, grid operations, mesh handling, spatial queries, and type definitions.

.. module:: nereus.core

Type Definitions
----------------

.. automodule:: nereus.core.types
   :members:
   :undoc-members:
   :show-inheritance:

Type Aliases
~~~~~~~~~~~~

The following type aliases are defined for convenience:

- **ArrayLike**: ``Union[NDArray[np.floating], xr.DataArray]`` - Array-like data type
- **FloatArray**: ``NDArray[np.floating]`` - Float array type
- **IntArray**: ``NDArray[np.integer]`` - Integer array type
- **BoolArray**: ``NDArray[np.bool_]`` - Boolean array type

Mesh Utilities
--------------

.. automodule:: nereus.core.mesh
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Creation
~~~~~~~~~~~~~

.. autofunction:: nereus.core.mesh.create_lonlat_mesh

.. autofunction:: nereus.core.mesh.mesh_from_arrays

Mesh Validation
~~~~~~~~~~~~~~~

.. autofunction:: nereus.core.mesh.validate_mesh

.. autofunction:: nereus.core.mesh.is_nereus_mesh

.. autofunction:: nereus.core.mesh.get_mesh_type

Coordinate Normalization
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.core.mesh.normalize_lon

.. autofunction:: nereus.core.mesh.ensure_lon_pm180

Dask Support
~~~~~~~~~~~~

.. autofunction:: nereus.core.mesh.should_use_dask

.. py:data:: nereus.core.mesh.DASK_THRESHOLD_POINTS

   Number of points above which dask arrays are automatically used: ``1_000_000``

Spatial Functions
-----------------

.. automodule:: nereus.core.spatial
   :members:
   :undoc-members:
   :show-inheritance:

Point Queries
~~~~~~~~~~~~~

.. autofunction:: nereus.core.spatial.find_nearest

.. autofunction:: nereus.core.spatial.haversine_distance

Geographic Subsetting
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.core.spatial.subset_by_bbox

.. autofunction:: nereus.core.spatial.points_in_polygon

Coordinate Functions
--------------------

.. automodule:: nereus.core.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

Constants
~~~~~~~~~

- **EARTH_RADIUS**: Earth's mean radius in meters (WGS84): ``6_371_000.0``

Grid Functions
--------------

.. automodule:: nereus.core.grids
   :members:
   :undoc-members:
   :show-inheritance:
