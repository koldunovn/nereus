"""Tests for nereus.regrid.interpolator."""

import numpy as np
import pytest

from nereus.regrid.interpolator import RegridInterpolator, regrid


class TestRegridInterpolator:
    """Tests for RegridInterpolator class."""

    def test_basic_creation(self, random_mesh_small):
        """Test basic interpolator creation."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=5.0)

        assert interp.target_lon.shape == interp.target_lat.shape
        assert interp.indices.shape == interp.target_lon.shape
        assert interp.valid_mask.shape == interp.target_lon.shape

    def test_resolution_float(self, random_mesh_small):
        """Test resolution specified as float."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=2.0)

        # 360/2 x 180/2 = 180 x 90
        assert interp.target_lon.shape == (90, 180)

    def test_resolution_tuple(self, random_mesh_small):
        """Test resolution specified as tuple."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=(100, 50))

        assert interp.target_lon.shape == (50, 100)

    def test_interpolate_1d(self, random_mesh_small, synthetic_data):
        """Test interpolation of 1D data."""
        lon, lat = random_mesh_small
        data = synthetic_data

        interp = RegridInterpolator(lon, lat, resolution=5.0)
        result = interp(data)

        assert result.shape == interp.target_lon.shape
        assert np.isfinite(result[interp.valid_mask]).all()

    def test_interpolate_2d(self, random_mesh_small):
        """Test interpolation of 2D data (nlevels, npoints)."""
        lon, lat = random_mesh_small
        n_levels = 5
        data = np.random.rand(n_levels, len(lon))

        interp = RegridInterpolator(lon, lat, resolution=5.0)
        result = interp(data)

        assert result.shape == (n_levels,) + interp.target_lon.shape

    def test_interpolate_3d(self, random_mesh_small):
        """Test interpolation of 3D data (ntime, nlevels, npoints)."""
        lon, lat = random_mesh_small
        n_time = 3
        n_levels = 5
        data = np.random.rand(n_time, n_levels, len(lon))

        interp = RegridInterpolator(lon, lat, resolution=5.0)
        result = interp(data)

        assert result.shape == (n_time, n_levels) + interp.target_lon.shape

    def test_influence_radius(self, random_mesh_small):
        """Test that influence radius affects valid mask."""
        lon, lat = random_mesh_small

        # Very small influence radius should have fewer valid points
        interp_small = RegridInterpolator(
            lon, lat, resolution=5.0, influence_radius=100_000
        )
        interp_large = RegridInterpolator(
            lon, lat, resolution=5.0, influence_radius=1_000_000
        )

        # Larger influence radius should have more valid points
        assert interp_large.valid_mask.sum() >= interp_small.valid_mask.sum()

    def test_fill_value(self, random_mesh_small, synthetic_data):
        """Test custom fill value for invalid points."""
        lon, lat = random_mesh_small
        data = synthetic_data

        interp = RegridInterpolator(lon, lat, resolution=5.0)
        result = interp(data, fill_value=-999.0)

        # Check that invalid points have fill value
        invalid_points = ~interp.valid_mask
        if invalid_points.any():
            np.testing.assert_allclose(result[invalid_points], -999.0)

    def test_reusability(self, random_mesh_small):
        """Test that interpolator can be reused for multiple variables."""
        lon, lat = random_mesh_small
        data1 = np.random.rand(len(lon))
        data2 = np.random.rand(len(lon))

        interp = RegridInterpolator(lon, lat, resolution=5.0)
        result1 = interp(data1)
        result2 = interp(data2)

        assert result1.shape == result2.shape
        # Results should be different
        assert not np.allclose(result1, result2)

    def test_custom_bounds(self, random_mesh_small, synthetic_data):
        """Test interpolation with custom bounds."""
        lon, lat = random_mesh_small
        data = synthetic_data

        interp = RegridInterpolator(
            lon, lat,
            resolution=2.0,
            lon_bounds=(-30, 30),
            lat_bounds=(30, 60)
        )

        result = interp(data)

        # Check bounds are respected
        assert interp.target_lon.min() >= -30
        assert interp.target_lon.max() <= 30
        assert interp.target_lat.min() >= 30
        assert interp.target_lat.max() <= 60


class TestRegridFunction:
    """Tests for regrid convenience function."""

    def test_basic_regrid(self, random_mesh_small, synthetic_data):
        """Test basic regrid function."""
        lon, lat = random_mesh_small
        data = synthetic_data

        result, interp = regrid(data, lon, lat, resolution=5.0)

        assert result.shape == interp.target_lon.shape
        assert isinstance(interp, RegridInterpolator)

    def test_regrid_returns_interpolator(self, random_mesh_small, synthetic_data):
        """Test that regrid returns usable interpolator."""
        lon, lat = random_mesh_small
        data = synthetic_data

        result1, interp = regrid(data, lon, lat, resolution=5.0)

        # Interpolator should be reusable
        data2 = np.random.rand(len(lon))
        result2 = interp(data2)

        assert result2.shape == result1.shape
