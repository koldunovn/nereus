Changelog
=========

All notable changes to Nereus will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.4.0] - 2026-02-27
---------------------

Added
~~~~~

**New model support**

- **MITgcm**: Native MDS binary I/O (``.meta`` + ``.data`` file pairs) without
  external dependencies. Includes ``load_mesh``, ``open_dataset``, and optional
  land masking via ``hFacC`` (3D per-level) or ``Depth`` (2D fallback).
- **MPAS-Ocean**: Mesh loading from standard NetCDF files with automatic
  coordinate conversion (radians to degrees) and ``open_dataset`` support.

**Diagnostics**

- ``as_xarray`` option for all diagnostic functions to return results as
  ``xr.DataArray`` with preserved dimension names, coordinates, and
  attributes from the input:

  - Sea ice: ``ice_area``, ``ice_volume``, ``ice_extent``, and their
    ``_nh`` / ``_sh`` hemisphere variants
  - Ocean: ``surface_mean``, ``volume_mean``, ``heat_content``
    (both ``"total"`` and ``"map"`` modes)
  - Hovmoller: ``hovmoller`` (returns DataArray with time + depth/latitude
    coordinates instead of a 3-tuple)

**Regridding**

- ``as_xarray`` option for ``nr.regrid`` to return results as ``xr.DataArray``.

**Core**

- ``flatten_spatial`` utility for shared ``(..., nlat, nlon) -> (..., npoints)``
  reshape logic, used by regrid, transect, and hovmoller.
- ``mesh_from_arrays`` now handles 1D lon/lat arrays of different sizes
  (e.g., ``lon(360)`` and ``lat(173)``) by expanding via meshgrid, with
  proper spherical cell area calculation.

Changed
~~~~~~~

- ``load_mesh`` auto-detection now supports MITgcm and MPAS-Ocean meshes.

Fixed
~~~~~

- ``plot_hovmoller`` y-axis artefact with non-linear scales: now computes
  explicit cell edges clamped to 0 at the surface instead of relying on
  ``pcolormesh(shading="nearest")``.
- ``hovmoller`` failing on 4D regular-grid input (e.g., EN4) by
  auto-flattening the spatial dimensions in both depth and latitude modes.

[0.3.0] - 2026-02-04
------------------

Added
~~~~~

**Mesh system**

- Unified mesh representation as standardized ``xr.Dataset`` with mesh metadata utilities.
- ``load_mesh`` auto-detection plus new model support for HEALPix, NEMO, and IFS TCO.
- Spatial utilities: ``find_nearest``, ``subset_by_bbox``, ``points_in_polygon``,
  ``haversine_distance``.
- FESOM helpers: element center computation, node/element conversions, ``mask_by_depth``,
  and ``nod_area_nans`` for depth-aware masking.

**Diagnostics**

- ``find_closest_depth``: Return index/value of closest depth to a target.
- ``interpolate_to_depth``: Interpolate 3D data to target depths.

Changed
~~~~~~~

- ``plot_hovmoller``: Added ``anomaly`` mode and replaced ``log_y`` with ``y_scale``
  (``"linear"``, ``"sqrt"``, ``"power"``, ``"symlog"``) plus ``y_scale_kw`` controls.
- Regridding/plotting coordinate handling: automatic lon/lat extraction and robust 1D/2D
  preparation via ``extract_coordinates`` and ``prepare_coordinates``.
- Dask handling improvements in vertical diagnostics and hovmoller (reduced graph bloat,
  better support for distributed execution).

Fixed
~~~~~

- Transect plotting now handles 3D data shaped ``(nlevels, nlat, nlon)`` by reshaping
  before KDTree indexing.
- Regridding/plotting fixes for mismatched coordinate shapes and 3D inputs.

[0.2.1] - 2026-01-23
--------------------

Added
~~~~~

**Diagnostics**

- ``surface_mean``: Area-weighted mean of 2D fields at a single level
- ``ice_area_nh``, ``ice_area_sh``: Northern/Southern Hemisphere ice area
- ``ice_volume_nh``, ``ice_volume_sh``: Northern/Southern Hemisphere ice volume
- ``ice_extent_nh``, ``ice_extent_sh``: Northern/Southern Hemisphere ice extent

**Core**

- ``get_array_data``: Extract underlying array while preserving dask arrays
- ``is_dask_array``: Identify dask array inputs

Changed
~~~~~~~

- ``heat_content``: Added ``output`` parameter supporting "total" (Joules) and "map" (J/m²) modes
- ``heat_content``: Updated specific heat capacity constant from 3985.0 to 3990.0 J/(kg·K) to match FESOM2 standards
- Dask support for ``ice_area``, ``ice_volume``, ``ice_extent``, ``volume_mean``, ``heat_content``, and ``hovmoller`` (depth mode)

[0.2.0] - 2026-01-20
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

[0.1.0] - 2024-01-19
--------------------

Initial release.

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
