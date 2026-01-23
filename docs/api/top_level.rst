Top-Level API
=============

The top-level ``nereus`` module provides the main user-facing API for quick data exploration and analysis.

.. code-block:: python

   import nereus as nr

Plotting Functions
------------------

.. autofunction:: nereus.plot

.. autofunction:: nereus.transect

Regridding Functions
--------------------

.. autofunction:: nereus.regrid

.. autoclass:: nereus.RegridInterpolator
   :members:
   :special-members: __init__, __call__

Sea Ice Diagnostics
-------------------

.. autofunction:: nereus.ice_area

.. autofunction:: nereus.ice_area_nh

.. autofunction:: nereus.ice_area_sh

.. autofunction:: nereus.ice_volume

.. autofunction:: nereus.ice_volume_nh

.. autofunction:: nereus.ice_volume_sh

.. autofunction:: nereus.ice_extent

.. autofunction:: nereus.ice_extent_nh

.. autofunction:: nereus.ice_extent_sh

Ocean Diagnostics
-----------------

.. autofunction:: nereus.surface_mean

.. autofunction:: nereus.volume_mean

.. autofunction:: nereus.heat_content

Hovmoller Diagrams
------------------

.. autofunction:: nereus.hovmoller

.. autofunction:: nereus.plot_hovmoller

Cache Configuration
-------------------

.. autofunction:: nereus.set_cache_options

Model Submodules
----------------

Access model-specific functionality through these namespaces:

* ``nr.fesom`` - FESOM2 ocean model (:doc:`models`)
* ``nr.icono`` - ICON-Ocean model (planned)
* ``nr.icona`` - ICON-Atmosphere model (planned)
* ``nr.ifs`` - IFS/ECMWF model (planned)
* ``nr.healpix`` - HEALPix grids (planned)

Version
-------

.. py:data:: nereus.__version__

   The current version of Nereus.
