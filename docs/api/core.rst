Core Utilities
==============

The ``nereus.core`` module provides fundamental utilities for coordinate transformations, grid operations, and type definitions.

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
