# Nereus

**Fast visualization and diagnostics for unstructured climate model data**

[![CI](https://github.com/koldunovn/nereus/actions/workflows/ci.yml/badge.svg)](https://github.com/koldunovn/nereus/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/nereus.svg)](https://badge.fury.io/py/nereus)
[![Documentation Status](https://readthedocs.org/projects/nereus/badge/?version=latest)](https://nereus.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Nereus is a Python library for quick data exploration of unstructured atmospheric and ocean model output in Jupyter notebooks. Plot global maps, compute diagnostics, and visualize transects from models like FESOM, ICON, and others — all with minimal code.

## ✨ Features

- **One-line plotting** — Visualize unstructured mesh data on publication-quality maps
- **Fast regridding** — KD-tree based interpolation with automatic caching
- **Multiple projections** — Robinson, Mollweide, stereographic, orthographic, and more
- **Ocean & sea ice diagnostics** — Ice area/volume/extent, heat content, volume-weighted means
- **Vertical transects** — Cross-section plots along great circle paths
- **Hovmöller diagrams** — Time-depth and time-latitude visualizations
- **Region masking** — Built-in support for ocean basins, NINO regions, MOC basins

## Installation

```bash
pip install nereus
```

## Quick Start

```python
import nereus as nr
import xarray as xr

# Load your unstructured model output
ds = xr.open_dataset("model_output.nc")
mesh = xr.open_dataset('model_grid.nc')

# Plot sea surface temperature in one line
fig, ax, interp = nr.plot(
    ds.temp.isel(time=0, nz1=0),
    mesh.lon, mesh.lat,
    projection="rob",
    cmap="RdBu_r",
    vmin=-2, vmax=30
)

# Reuse interpolator for another variable
fig, ax, _ = nr.plot(ds.salt.isel(time=0, nz1=0), ds.lon, ds.lat, interpolator=interp)
```

### Regridding

```python
# Regrid to a regular 0.5° grid
regridded, interp = nr.regrid(data, lon, lat, resolution=0.5)

# Reuse interpolator for efficiency
regridded_salt, _ = nr.regrid(salinity, lon, lat, interpolator=interp)
```

### Working with FESOM

```python
# Load mesh once, reuse for multiple operations
mesh = nr.fesom.load_mesh("/path/to/mesh/")

# Compute sea ice extent
extent = nr.ice_extent(ds.a_ice, mesh.area, threshold=0.15)

# Ocean heat content
ohc = nr.heat_content(ds.temp, mesh.area, mesh.layer_thickness)

# Volume-weighted mean temperature
mean_temp = nr.volume_mean(ds.temp, mesh.area, mesh.layer_thickness)
```

### Vertical Transects

```python
# Plot a transect from point A to point B
nr.transect(
    ds.temp,
    mesh.lon, mesh.lat, mesh.depth,
    start=(-30, -60),  # (lon, lat)
    end=(30, 60),
    cmap="thermal"
)
```

## Supported Models

| Model/mesh | Status |
|------------|--------|
| FESOM2 | ✅ Full support |
| ICON-O | 🔧 In development |
| ICON-A | 🔧 In development |
| HEALPix | 📋 Planned |

## Documentation

📖 Full documentation: [nereus.readthedocs.io](https://nereus.readthedocs.io)

- [Installation Guide](https://nereus.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start Tutorial](https://nereus.readthedocs.io/en/latest/getting_started/quickstart.html)
- [API Reference](https://nereus.readthedocs.io/en/latest/api/index.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Clone and install in development mode
git clone https://github.com/koldunovn/nereus.git
cd nereus
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Named after the Greek god of the sea, the "Old Man of the Sea."*
