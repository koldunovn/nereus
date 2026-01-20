# Nereus

[![CI](https://github.com/koldunovn/nereus/actions/workflows/ci.yml/badge.svg)](https://github.com/koldunovn/nereus/actions/workflows/ci.yml)

Python module for working with geophysical data from atmospheric and ocean models.

## Installation

```bash
pip install nereus
```

## Quick Start

```python
import nereus as nr

# Plot unstructured data on a map
fig, ax, interp = nr.plot(temperature, mesh.lon, mesh.lat)

# Reuse interpolator for another variable
fig, ax, _ = nr.plot(salinity, mesh.lon, mesh.lat, interpolator=interp)

# Regrid data to regular grid
regridded, interp = nr.regrid(data, lon, lat, resolution=0.5)

# Plot on different projections
fig, ax, interp = nr.plot(data, lon, lat, projection="npstere")
```

## Features

- Fast regridding of unstructured data to regular grids
- Support for multiple map projections
- Interpolator caching for repeated operations
- Dask support for large datasets

## License

MIT
