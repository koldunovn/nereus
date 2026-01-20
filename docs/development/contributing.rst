Contributing
============

Thank you for your interest in contributing to Nereus! This guide will help you get started.

Development Setup
-----------------

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/koldunovn/nereus.git
      cd nereus

2. **Create a development environment:**

   Using conda/mamba (recommended):

   .. code-block:: bash

      mamba create -n nereus-dev python=3.11
      mamba activate nereus-dev
      mamba install -c conda-forge numpy scipy xarray netcdf4 matplotlib cartopy dask

   Or using pip:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install in development mode:**

   .. code-block:: bash

      pip install -e ".[dev]"

   This installs Nereus in editable mode with development dependencies (pytest, ruff, mypy).

Running Tests
-------------

Run the full test suite:

.. code-block:: bash

   pytest tests/

Run tests with coverage:

.. code-block:: bash

   pytest tests/ --cov=nereus --cov-report=html

Run a specific test file:

.. code-block:: bash

   pytest tests/test_core/test_coordinates.py

Run tests matching a pattern:

.. code-block:: bash

   pytest -k "test_regrid"

Code Style
----------

We use ``ruff`` for linting and formatting:

.. code-block:: bash

   # Check for issues
   ruff check src/

   # Auto-fix issues
   ruff check src/ --fix

   # Format code
   ruff format src/

Configuration is in ``pyproject.toml``:

.. code-block:: toml

   [tool.ruff]
   line-length = 100
   target-version = "py310"

   [tool.ruff.lint]
   select = ["E", "F", "I", "UP", "B", "SIM"]

Type Checking
-------------

We use ``mypy`` for static type checking:

.. code-block:: bash

   mypy src/nereus/

All public functions should have type annotations.

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

View the built documentation:

.. code-block:: bash

   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Docstring Format
~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def my_function(param1, param2):
       """Short description of the function.

       Longer description if needed. Can span multiple lines.

       Parameters
       ----------
       param1 : type
           Description of param1.
       param2 : type, optional
           Description of param2. Defaults to X.

       Returns
       -------
       type
           Description of return value.

       Examples
       --------
       >>> my_function(1, 2)
       3

       Notes
       -----
       Additional notes about the implementation.
       """

Pull Request Process
--------------------

1. **Fork the repository** and create a new branch:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. **Make your changes** and ensure:

   - All tests pass: ``pytest tests/``
   - Code is formatted: ``ruff format src/``
   - No linting errors: ``ruff check src/``
   - Type hints are correct: ``mypy src/nereus/``

3. **Write tests** for new functionality.

4. **Update documentation** if needed.

5. **Commit with clear messages:**

   .. code-block:: bash

      git commit -m "Add feature X for doing Y"

6. **Push and create a pull request:**

   .. code-block:: bash

      git push origin feature/my-new-feature

7. **Address review feedback** and update as needed.

Adding New Features
-------------------

Adding a New Diagnostic
~~~~~~~~~~~~~~~~~~~~~~~

1. Create the function in the appropriate module (e.g., ``src/nereus/diag/ice.py``).

2. Add docstrings with examples.

3. Export from ``src/nereus/diag/__init__.py``.

4. Export from ``src/nereus/__init__.py`` if it should be top-level.

5. Add tests in ``tests/test_diag/``.

6. Update documentation in ``docs/api/diag.rst``.

Adding Model Support
~~~~~~~~~~~~~~~~~~~~

1. Create a new directory under ``src/nereus/models/`` (e.g., ``models/newmodel/``).

2. Create ``__init__.py`` with:

   - A ``Mesh`` class inheriting from ``MeshBase``
   - A ``load_mesh()`` function
   - Any model-specific utilities

3. Export from ``src/nereus/models/__init__.py``.

4. Add namespace export in ``src/nereus/__init__.py``.

5. Add tests and documentation.

Example:

.. code-block:: python

   # src/nereus/models/newmodel/__init__.py
   from nereus.models._base import MeshBase

   class NewModelMesh(MeshBase):
       def __init__(self, ...):
           ...

       @property
       def lon(self):
           return self._lon

       @property
       def lat(self):
           return self._lat

       @property
       def area(self):
           return self._area

   def load_mesh(path):
       """Load NewModel mesh from file."""
       ...
       return NewModelMesh(...)

Reporting Issues
----------------

When reporting bugs, please include:

1. **Python version** and **Nereus version**
2. **Operating system**
3. **Minimal reproducible example**
4. **Expected behavior** vs **actual behavior**
5. **Full error traceback** if applicable

Feature requests are also welcome! Please describe:

1. **Use case**: What are you trying to accomplish?
2. **Current workaround**: How do you handle it now?
3. **Proposed solution**: What would the ideal API look like?

Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

Thank you for contributing!
