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
      :type: Literal["nearest", "idw", "linear", "cubic", "conservative"]

      Interpolation method. ``"nearest"`` uses KDTree nearest-neighbor
      lookup (fast).  ``"idw"`` uses inverse distance weighting with 8
      nearest neighbors (fast, smooth).  ``"linear"`` uses Delaunay
      triangulation with barycentric interpolation (slower but smoother).
      ``"cubic"`` uses Clough-Tocher C1 interpolation (smoothest).
      ``"conservative"`` uses area-weighted polygon overlap so an
      integrated quantity is preserved (slowest); see
      :ref:`conservative-remapping`.

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
      :type: NDArray | None

      Longitudes of an arbitrary unstructured target (constructor input,
      ``method="conservative"`` only), or the resulting target
      coordinates after ``__post_init__`` -- 2D ``(nlat, nlon)`` for a
      regular grid, 1D ``(n_target,)`` for an arbitrary target.

   .. py:attribute:: target_lat
      :type: NDArray | None

      See ``target_lon``.

   .. py:attribute:: indices
      :type: NDArray

      Pre-computed source indices for each target point. Not set for
      ``method="conservative"``.

   .. py:attribute:: distances
      :type: NDArray

      Pre-computed distances from target to source points. Not set for
      ``method="conservative"``.

   .. py:attribute:: valid_mask
      :type: NDArray

      Boolean mask indicating valid target points. Not set for
      ``method="conservative"`` (use ``return_fraction=True`` on
      ``__call__`` instead).

.. autofunction:: nereus.regrid.interpolator.regrid

Conservative Remapping
-----------------------

.. automodule:: nereus.regrid.conservative
   :members:
   :undoc-members:
   :show-inheritance:

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
