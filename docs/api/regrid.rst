Regridding Module
=================

The ``nereus.regrid`` module provides tools for regridding unstructured mesh data to regular grids.

.. module:: nereus.regrid

Interpolator
------------

.. automodule:: nereus.regrid.interpolator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nereus.regrid.interpolator.RegridInterpolator
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

   .. rubric:: Attributes

   .. py:attribute:: source_lon
      :type: NDArray[np.floating]

      Source grid longitude coordinates.

   .. py:attribute:: source_lat
      :type: NDArray[np.floating]

      Source grid latitude coordinates.

   .. py:attribute:: resolution
      :type: float | tuple[int, int]

      Target grid resolution (degrees or grid dimensions).

   .. py:attribute:: method
      :type: Literal["nearest"]

      Interpolation method. Currently only "nearest" is supported.

   .. py:attribute:: influence_radius
      :type: float

      Maximum influence radius in meters.

   .. py:attribute:: lon_bounds
      :type: tuple[float, float]

      Target grid longitude bounds.

   .. py:attribute:: lat_bounds
      :type: tuple[float, float]

      Target grid latitude bounds.

   .. py:attribute:: target_lon
      :type: NDArray

      2D array of target grid longitudes.

   .. py:attribute:: target_lat
      :type: NDArray

      2D array of target grid latitudes.

   .. py:attribute:: indices
      :type: NDArray

      Pre-computed source indices for each target point.

   .. py:attribute:: distances
      :type: NDArray

      Pre-computed distances from target to source points.

   .. py:attribute:: valid_mask
      :type: NDArray

      Boolean mask indicating valid target points.

.. autofunction:: nereus.regrid.interpolator.regrid

Cache
-----

.. automodule:: nereus.regrid.cache
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nereus.regrid.cache.InterpolatorCache
   :members:
   :special-members: __init__, __len__
   :show-inheritance:

   .. rubric:: Methods

   .. automethod:: get_or_create

   .. automethod:: clear

.. autofunction:: nereus.regrid.cache.get_cache

.. autofunction:: nereus.regrid.cache.set_cache_options

.. autofunction:: nereus.regrid.cache.clear_cache
