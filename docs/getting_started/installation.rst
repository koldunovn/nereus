Installation
============

Requirements
------------

Nereus requires Python 3.10 or later and depends on the following packages:

* NumPy (>=1.22)
* SciPy (>=1.9)
* xarray (>=2022.6)
* netCDF4 (>=1.6)
* Matplotlib (>=3.5)
* Cartopy (>=0.21)
* Dask (>=2022.6) - for parallel and out-of-core computation

Installing with pip
-------------------

The simplest way to install Nereus is using pip:

.. code-block:: bash

   pip install nereus

Installing with conda
---------------------

You can also install Nereus using conda or mamba from conda-forge:

.. code-block:: bash

   conda install -c conda-forge nereus

   # or with mamba (faster)
   mamba install -c conda-forge nereus

Installing from source
----------------------

For the latest development version, you can install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/koldunovn/nereus.git

Or clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/koldunovn/nereus.git
   cd nereus
   pip install -e ".[dev]"

This installs Nereus in editable mode along with development dependencies (pytest, ruff, mypy).

Optional Dependencies
---------------------

Nereus has several optional dependency groups:

**Dask support** (included by default):

.. code-block:: bash

   pip install nereus[dask]

**Development tools**:

.. code-block:: bash

   pip install nereus[dev]

**All optional dependencies**:

.. code-block:: bash

   pip install nereus[all]

Verifying Installation
----------------------

To verify that Nereus is installed correctly, run:

.. code-block:: python

   import nereus as nr
   print(nr.__version__)

You should see the version number printed (e.g., ``0.2.0``).

Cartopy Installation Notes
--------------------------

Cartopy requires GEOS and PROJ libraries. If you encounter installation issues with Cartopy, we recommend installing it via conda:

.. code-block:: bash

   conda install -c conda-forge cartopy

Then install Nereus with pip:

.. code-block:: bash

   pip install nereus --no-deps
   pip install numpy scipy xarray netCDF4 matplotlib dask

Environment Setup
-----------------

For reproducible environments, we recommend using conda/mamba with an environment file:

.. code-block:: yaml

   # environment.yml
   name: nereus-env
   channels:
     - conda-forge
   dependencies:
     - python>=3.10
     - nereus
     - jupyterlab  # for interactive analysis

Create and activate the environment:

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate nereus-env
