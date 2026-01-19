"""Tests for nereus.core.grids."""

import numpy as np
import pytest

from nereus.core.grids import create_regular_grid, grid_cell_area


class TestCreateRegularGrid:
    """Tests for create_regular_grid function."""

    def test_1_degree_resolution(self):
        """Test 1 degree resolution grid."""
        lon, lat = create_regular_grid(1.0)

        assert lon.shape == (180, 360)
        assert lat.shape == (180, 360)

    def test_resolution_tuple(self):
        """Test resolution specified as tuple."""
        lon, lat = create_regular_grid((720, 360))

        assert lon.shape == (360, 720)
        assert lat.shape == (360, 720)

    def test_lon_bounds(self):
        """Test that longitude bounds are respected."""
        lon, lat = create_regular_grid(1.0, lon_bounds=(-180, 180))

        assert lon.min() > -180
        assert lon.max() < 180

    def test_lat_bounds(self):
        """Test that latitude bounds are respected."""
        lon, lat = create_regular_grid(1.0, lat_bounds=(-90, 90))

        assert lat.min() > -90
        assert lat.max() < 90

    def test_cell_center_mode(self):
        """Test cell center positioning."""
        lon, lat = create_regular_grid(10.0, center="cell")

        # First cell center should be at half resolution from bound
        np.testing.assert_allclose(lon[0, 0], -175.0)
        np.testing.assert_allclose(lat[0, 0], -85.0)

    def test_node_mode(self):
        """Test node positioning."""
        lon, lat = create_regular_grid(10.0, center="node")

        # First node should be at bound
        np.testing.assert_allclose(lon[0, 0], -180.0)
        np.testing.assert_allclose(lat[0, 0], -90.0)

    def test_custom_bounds(self):
        """Test custom geographic bounds."""
        lon, lat = create_regular_grid(
            1.0,
            lon_bounds=(0, 30),
            lat_bounds=(40, 60)
        )

        assert lon.shape[1] == 30  # 30 degrees / 1 degree
        assert lon.shape[0] == 20  # 20 degrees / 1 degree
        assert lon.min() >= 0
        assert lon.max() <= 30
        assert lat.min() >= 40
        assert lat.max() <= 60


class TestGridCellArea:
    """Tests for grid_cell_area function."""

    def test_area_sum(self):
        """Test that total area is approximately Earth's surface area."""
        lon, lat = create_regular_grid(1.0)
        area = grid_cell_area(lon, lat)

        # Earth's surface area ≈ 5.1e14 m²
        earth_area = 4 * np.pi * 6_371_000**2
        total_area = area.sum()

        # Should be within 1% due to discretization
        np.testing.assert_allclose(total_area, earth_area, rtol=0.01)

    def test_area_symmetry(self):
        """Test that area is symmetric about equator."""
        lon, lat = create_regular_grid(1.0)
        area = grid_cell_area(lon, lat)

        # Sum of northern hemisphere should equal southern
        nlat = area.shape[0]
        north_area = area[nlat // 2:, :].sum()
        south_area = area[: nlat // 2, :].sum()

        np.testing.assert_allclose(north_area, south_area, rtol=1e-10)

    def test_polar_area_smaller(self):
        """Test that polar cells are smaller than equatorial cells."""
        lon, lat = create_regular_grid(1.0)
        area = grid_cell_area(lon, lat)

        equator_area = area[area.shape[0] // 2, 0]
        polar_area = area[0, 0]

        assert polar_area < equator_area
