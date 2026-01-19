"""Shared test fixtures for nereus tests."""

import numpy as np
import pytest


@pytest.fixture
def random_mesh_small():
    """Create a small random mesh for testing."""
    rng = np.random.default_rng(42)
    n_points = 1000

    lon = rng.uniform(-180, 180, n_points)
    lat = rng.uniform(-90, 90, n_points)

    return lon, lat


@pytest.fixture
def random_mesh_medium():
    """Create a medium-sized random mesh for testing."""
    rng = np.random.default_rng(42)
    n_points = 10000

    lon = rng.uniform(-180, 180, n_points)
    lat = rng.uniform(-90, 90, n_points)

    return lon, lat


@pytest.fixture
def synthetic_data(random_mesh_small):
    """Create synthetic data on the small mesh."""
    lon, lat = random_mesh_small
    # Create data that varies smoothly with lat/lon
    data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    return data


@pytest.fixture
def synthetic_3d_data(random_mesh_small):
    """Create synthetic 3D data (levels, npoints)."""
    lon, lat = random_mesh_small
    n_levels = 10

    # Create data that varies with depth and position
    depths = np.linspace(0, 1000, n_levels)
    data = np.zeros((n_levels, len(lon)))
    for i, d in enumerate(depths):
        data[i] = np.sin(np.deg2rad(lat)) * np.exp(-d / 500)

    return data, depths
