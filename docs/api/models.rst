Models Module
=============

The ``nereus.models`` module provides support for specific atmospheric and ocean models.

.. module:: nereus.models

Base Classes
------------

.. automodule:: nereus.models._base
   :members:
   :undoc-members:
   :show-inheritance:

.. _fesom:

FESOM2
------

The Finite Element Sea ice-Ocean Model version 2 uses an unstructured triangular mesh.

.. automodule:: nereus.models.fesom
   :members:
   :undoc-members:
   :show-inheritance:

FESOM2 File Formats
~~~~~~~~~~~~~~~~~~~

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

* ``mesh.diag.nc`` or ``fesom.mesh.diag.nc``: NetCDF file with cluster areas

  .. code-block:: text

     Dimensions: nod2
     Variables: cluster_area(nod2)

**Optional files:**

* ``aux3d.out``: ASCII file with vertical level information

  .. code-block:: text

     # depth levels (one per line)
     0.0
     10.0
     25.0
     ...

ICON-Ocean (Planned)
--------------------

.. automodule:: nereus.models.icono
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   ICON-Ocean support is planned for a future release.

ICON-Atmosphere (Planned)
-------------------------

.. automodule:: nereus.models.icona
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   ICON-Atmosphere support is planned for a future release.

IFS (Planned)
-------------

.. automodule:: nereus.models.ifs
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   IFS/ECMWF support is planned for a future release.

HEALPix (Planned)
-----------------

.. automodule:: nereus.models.healpix
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   HEALPix grid support is planned for a future release.
