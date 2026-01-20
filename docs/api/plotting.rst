Plotting Module
===============

The ``nereus.plotting`` module provides visualization tools for unstructured mesh data.

.. module:: nereus.plotting

Map Plotting
------------

.. automodule:: nereus.plotting.maps
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: nereus.plotting.maps.plot

Projections
-----------

.. automodule:: nereus.plotting.projections
   :members:
   :undoc-members:
   :show-inheritance:

Supported Projections
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Alias(es)
     - Projection Class
     - Description
   * - ``"pc"``, ``"platecarree"``
     - ``ccrs.PlateCarree``
     - Equirectangular projection
   * - ``"rob"``, ``"robinson"``
     - ``ccrs.Robinson``
     - Robinson projection (global)
   * - ``"merc"``, ``"mercator"``
     - ``ccrs.Mercator``
     - Mercator projection
   * - ``"moll"``, ``"mollweide"``
     - ``ccrs.Mollweide``
     - Mollweide projection (global, equal-area)
   * - ``"np"``, ``"npstere"``, ``"northpolarstereo"``
     - ``ccrs.NorthPolarStereo``
     - North Polar Stereographic
   * - ``"sp"``, ``"spstere"``, ``"southpolarstereo"``
     - ``ccrs.SouthPolarStereo``
     - South Polar Stereographic
   * - ``"ortho"``, ``"orthographic"``
     - ``ccrs.Orthographic``
     - Orthographic projection
   * - ``"lcc"``, ``"lambertconformal"``
     - ``ccrs.LambertConformal``
     - Lambert Conformal Conic

Functions
~~~~~~~~~

.. autofunction:: nereus.plotting.projections.get_projection

.. autofunction:: nereus.plotting.projections.is_global_projection

.. autofunction:: nereus.plotting.projections.is_polar_projection

.. autofunction:: nereus.plotting.projections.get_data_bounds_for_projection

Constants
~~~~~~~~~

.. py:data:: nereus.plotting.projections.PROJECTION_ALIASES

   Dictionary mapping projection aliases to their configurations.

   Example structure:

   .. code-block:: python

      {
          "pc": {"class": "PlateCarree", "global": False},
          "rob": {"class": "Robinson", "global": True},
          "np": {"class": "NorthPolarStereo", "global": False, "polar": "north"},
          ...
      }

Transect Plotting
-----------------

.. automodule:: nereus.plotting.transect
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: nereus.plotting.transect.transect
