Models Module
=============

The ``nereus.models`` module provides support for specific atmospheric and ocean models.
All mesh loaders return standardized ``xr.Dataset`` objects with consistent variable naming.

.. module:: nereus.models

Universal Mesh Loader
---------------------

.. autofunction:: nereus.models.load_mesh

.. autofunction:: nereus.models.detect_mesh_type

.. _fesom:

FESOM2
------

The Finite Element Sea ice-Ocean Model version 2 uses an unstructured triangular mesh.

.. automodule:: nereus.models.fesom
   :members:
   :undoc-members:
   :show-inheritance:

FESOM2 Mesh Loading
~~~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.models.fesom.load_mesh

The ``load_mesh`` function expects a directory containing FESOM2 mesh files:

**Required files:**

* ``nod2d.out``: ASCII file with 2D node coordinates

  .. code-block:: text

     # node_id longitude latitude boundary_flag
     1 -180.0 -80.0 0
     2 -179.5 -80.0 0
     ...

* ``elem2d.out``: ASCII file with element connectivity

  .. code-block:: text

     # elem_id node1 node2 node3
     1 1 2 100
     2 2 3 101
     ...

* ``mesh.diag.nc`` or ``fesom.mesh.diag.nc``: NetCDF file with node areas

  .. code-block:: text

     Dimensions: nod2, nz
     Variables: nod_area(nz, nod2)

**Optional files:**

* ``aux3d.out``: ASCII file with vertical level information

  .. code-block:: text

     # depth levels (one per line)
     0.0
     10.0
     25.0
     ...

FESOM2 Functions
~~~~~~~~~~~~~~~~

.. autofunction:: nereus.models.fesom.compute_element_centers

.. autofunction:: nereus.models.fesom.node_to_element

.. autofunction:: nereus.models.fesom.element_to_node

.. autofunction:: nereus.models.fesom.open_dataset

HEALPix
-------

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) grids are used by ICON and other models.

.. automodule:: nereus.models.healpix
   :members:
   :undoc-members:
   :show-inheritance:

HEALPix Mesh Loading
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.models.healpix.load_mesh

HEALPix Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.models.healpix.nside_to_npoints

.. autofunction:: nereus.models.healpix.npoints_to_nside

.. autofunction:: nereus.models.healpix.resolution_to_nside

NEMO
----

NEMO (Nucleus for European Modelling of the Ocean) uses structured grids.

.. automodule:: nereus.models.nemo
   :members:
   :undoc-members:
   :show-inheritance:

NEMO Mesh Loading
~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.models.nemo.load_mesh

The NEMO mesh loader expects a ``mesh_mask.nc`` file and flattens 2D coordinates
to 1D for compatibility with nereus functions. Original 2D shape information is
preserved in mesh attributes.

IFS TCO
-------

.. automodule:: nereus.models.ifs_tco
   :members:
   :undoc-members:
   :show-inheritance:

IFS TCO Mesh Loading
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nereus.models.ifs_tco.load_mesh

The IFS TCO mesh loader expects separate grid and area NetCDF files. It selects
the ``A*`` prefix for ``lon``/``lat`` and ``A*.srf`` for area, then flattens the
coordinates to a 1D mesh compatible with nereus utilities.

ICON-Ocean (Planned)
--------------------

.. note::

   ICON-Ocean support is planned for a future release.

ICON-Atmosphere (Planned)
-------------------------

.. note::

   ICON-Atmosphere support is planned for a future release.

IFS (Planned)
-------------

.. note::

   IFS/ECMWF support is planned for a future release.
