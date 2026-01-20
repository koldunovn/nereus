Nereus Documentation
====================

**Nereus** is a Python library for quick data exploration and analysis of unstructured atmospheric and ocean model data in Jupyter notebooks.

.. image:: https://readthedocs.org/projects/nereus/badge/?version=latest
   :target: https://nereus.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/nereus.svg
   :target: https://pypi.org/project/nereus/
   :alt: PyPI

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://python.org
   :alt: Python 3.10+

Key Features
------------

- **Fast regridding** from unstructured to regular grids using KD-tree based nearest neighbor interpolation
- **Interactive plotting** with Cartopy projections for quick data visualization
- **Model-specific support** for FESOM2, ICON, IFS, and HEALPix grids
- **Diagnostic functions** for sea ice, ocean heat content, and Hovmoller diagrams
- **Dask integration** for efficient handling of large datasets
- **Automatic caching** of interpolation weights for repeated operations

Quick Example
-------------

.. code-block:: python

   import nereus as nr
   import xarray as xr

   # Load your unstructured data
   ds = xr.open_dataset("fesom_output.nc")

   # Plot with a single line
   fig, ax, interp = nr.plot(
       ds.temp.isel(time=0, nz1=0),
       ds.lon, ds.lat,
       projection="rob",
       cmap="RdBu_r"
   )

   # Compute sea ice area
   ice_area = nr.ice_area(ds.a_ice, mesh.area)

Installation
------------

.. code-block:: bash

   pip install nereus

   # Or with conda/mamba
   conda install -c conda-forge nereus

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/plotting
   user_guide/regridding
   user_guide/diagnostics
   user_guide/models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/top_level
   api/core
   api/regrid
   api/plotting
   api/diag
   api/models

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
