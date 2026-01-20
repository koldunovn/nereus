Changelog
=========

All notable changes to Nereus will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~

- Initial public release
- Core regridding functionality with KD-tree nearest neighbor interpolation
- Map plotting with Cartopy projections (PlateCarree, Robinson, Mercator, Mollweide, Polar Stereographic, Orthographic, Lambert Conformal)
- Vertical transect plotting
- Sea ice diagnostics: ``ice_area``, ``ice_volume``, ``ice_extent``
- Ocean diagnostics: ``volume_mean``, ``heat_content``
- Hovmoller diagram computation and plotting
- FESOM2 mesh support with ``load_mesh`` and ``open_dataset``
- Interpolator caching with LRU eviction
- Dask array support for lazy computation
- Comprehensive documentation with Sphinx

[0.1.0] - 2024-XX-XX
--------------------

Initial release.

Added
~~~~~

**Core**

- ``lonlat_to_cartesian``: Convert geographic to Cartesian coordinates
- ``cartesian_to_lonlat``: Convert Cartesian to geographic coordinates
- ``meters_to_chord``: Convert meters to chord distance on unit sphere
- ``great_circle_distance``: Haversine distance calculation
- ``great_circle_path``: Generate points along great circle
- ``create_regular_grid``: Create regular lon/lat grids
- ``grid_cell_area``: Compute areas of regular grid cells

**Regridding**

- ``RegridInterpolator``: Pre-computed interpolation weights class
- ``regrid``: Convenience function for one-shot regridding
- ``InterpolatorCache``: LRU cache for interpolators
- ``set_cache_options``: Configure global cache behavior

**Plotting**

- ``plot``: 2D map plotting with automatic regridding
- ``transect``: Vertical cross-section plotting
- Support for 8 map projections with short aliases

**Diagnostics**

- ``ice_area``: Total sea ice area
- ``ice_volume``: Total sea ice volume
- ``ice_extent``: Sea ice extent with threshold
- ``volume_mean``: Volume-weighted mean
- ``heat_content``: Ocean heat content
- ``hovmoller``: Compute Hovmoller diagram data
- ``plot_hovmoller``: Plot Hovmoller diagrams

**Models**

- ``MeshBase``: Abstract base class for model meshes
- ``FesomMesh``: FESOM2 mesh class
- ``fesom.load_mesh``: Load FESOM2 mesh from directory
- ``fesom.open_dataset``: Open FESOM2 data with mesh coordinates
- Stub modules for ICON-Ocean, ICON-Atmosphere, IFS, HEALPix

Migration Guide
---------------

From Other Tools
~~~~~~~~~~~~~~~~

**From pyfesom2**

Nereus provides similar functionality with a cleaner API:

.. code-block:: python

   # pyfesom2
   from pyfesom2 import load_mesh, plot
   mesh = load_mesh("/path/to/mesh")
   plot(mesh, data)

   # Nereus
   import nereus as nr
   mesh = nr.fesom.load_mesh("/path/to/mesh")
   nr.plot(data, mesh.lon, mesh.lat)

**From xESMF**

Nereus is designed for quick exploration; use xESMF for production regridding:

.. code-block:: python

   # For exploration: Nereus
   regridded, _ = nr.regrid(data, lon, lat, resolution=0.5)

   # For production: xESMF
   import xesmf as xe
   regridder = xe.Regridder(ds_in, ds_out, "bilinear")
   regridded = regridder(data)
