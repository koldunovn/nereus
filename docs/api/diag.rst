Diagnostics Module
==================

The ``nereus.diag`` module provides diagnostic functions for analyzing ocean and sea ice model output.

.. module:: nereus.diag

Sea Ice Diagnostics
-------------------

.. automodule:: nereus.diag.ice
   :members:
   :undoc-members:
   :show-inheritance:

Vertical/Ocean Diagnostics
--------------------------

.. automodule:: nereus.diag.vertical
   :members:
   :undoc-members:
   :show-inheritance:

Constants
~~~~~~~~~

- **RHO_SEAWATER**: Reference seawater density: ``1025.0`` kg/m³
- **CP_SEAWATER**: Seawater specific heat capacity: ``3985.0`` J/(kg·K)

Hovmoller Diagrams
------------------

.. automodule:: nereus.diag.hovmoller
   :members:
   :undoc-members:
   :show-inheritance:

Region Masks
------------

.. automodule:: nereus.diag.regions
   :members:
   :undoc-members:
   :show-inheritance:

Built-in GeoJSON Files
~~~~~~~~~~~~~~~~~~~~~~

Nereus includes the following GeoJSON files with predefined regions:

**MOCBasins** - Meridional Overturning Circulation regions:

- Atlantic_MOC
- IndoPacific_MOC
- Pacific_MOC
- Indian_MOC

**oceanBasins** - Major ocean basins:

- Atlantic_Basin
- Pacific_Basin
- Indian_Basin
- Arctic_Basin
- Southern_Ocean_Basin
- Mediterranean_Basin
- Global Ocean
- Global Ocean 65N to 65S
- Global Ocean 15S to 15N

**NinoRegions** - El Nino monitoring regions:

- Nino 3.4
- Nino 3
- Nino 4

Formulas
--------

This section documents the mathematical formulas used in the diagnostic calculations.

Sea Ice Area
~~~~~~~~~~~~

.. math::

   A_{ice} = \sum_i c_i \cdot A_i

where:

- :math:`c_i` = ice concentration at cell :math:`i` (fraction, 0-1)
- :math:`A_i` = area of cell :math:`i` (m²)

Sea Ice Extent
~~~~~~~~~~~~~~

.. math::

   E_{ice} = \sum_i A_i \cdot \mathbf{1}_{c_i \geq \tau}

where:

- :math:`\tau` = concentration threshold (default 0.15)
- :math:`\mathbf{1}_{(\cdot)}` = indicator function

Sea Ice Volume
~~~~~~~~~~~~~~

.. math::

   V_{ice} = \sum_i h_i \cdot c_i \cdot A_i

where:

- :math:`h_i` = ice thickness at cell :math:`i` (m)
- :math:`c_i` = ice concentration (optional weighting)
- :math:`A_i` = cell area (m²)

Volume Mean
~~~~~~~~~~~

.. math::

   \bar{X} = \frac{\sum_{i,k} X_{i,k} \cdot A_i \cdot \Delta z_k}{\sum_{i,k} A_i \cdot \Delta z_k}

where:

- :math:`X_{i,k}` = quantity at horizontal cell :math:`i`, vertical level :math:`k`
- :math:`A_i` = horizontal cell area (m²)
- :math:`\Delta z_k` = layer thickness at level :math:`k` (m)

Ocean Heat Content
~~~~~~~~~~~~~~~~~~

.. math::

   OHC = \rho \cdot c_p \cdot \sum_{i,k} (T_{i,k} - T_{ref}) \cdot A_i \cdot \Delta z_k

where:

- :math:`\rho` = seawater density (default 1025 kg/m³)
- :math:`c_p` = specific heat capacity (default 3985 J/(kg·K))
- :math:`T_{i,k}` = temperature at cell :math:`i`, level :math:`k` (°C)
- :math:`T_{ref}` = reference temperature (default 0°C)
- :math:`A_i` = cell area (m²)
- :math:`\Delta z_k` = layer thickness (m)
