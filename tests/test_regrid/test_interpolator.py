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


class TestRegridInputFormats:
    """Tests for regrid function with various input formats."""

    def test_2d_data_2d_coords(self):
        """Test regrid with 2D data and 2D coordinates."""
        lon_1d = np.linspace(-180, 180, 36)
        lat_1d = np.linspace(-90, 90, 18)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        data_2d = np.sin(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(lon_2d))

        with pytest.warns(UserWarning, match="Raveling 2D arrays"):
            result, interp = regrid(data_2d, lon_2d, lat_2d, resolution=10.0)

        assert result.shape == interp.target_lon.shape

    def test_2d_data_1d_coords(self):
        """Test regrid with 2D data and 1D coordinates (meshgrid case)."""
        lon_1d = np.linspace(-180, 180, 36)
        lat_1d = np.linspace(-90, 90, 18)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        data_2d = np.sin(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(lon_2d))

        with pytest.warns(UserWarning, match="Creating meshgrid"):
            result, interp = regrid(data_2d, lon_1d, lat_1d, resolution=10.0)

        assert result.shape == interp.target_lon.shape

    def test_missing_coords_raises(self, synthetic_data):
        """Test that missing coordinates raises clear error."""
        data = synthetic_data

        with pytest.raises(ValueError, match="lon and lat coordinates are required"):
            regrid(data)

    def test_xarray_auto_coords(self):
        """Test automatic coordinate extraction from xarray."""
        xr = pytest.importorskip("xarray")

        lon_vals = np.linspace(-180, 180, 36)
        lat_vals = np.linspace(-90, 90, 18)
        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
        data_vals = np.sin(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(lon_2d))

        da = xr.DataArray(
            data_vals,
            coords={"lat": lat_vals, "lon": lon_vals},
            dims=["lat", "lon"],
        )

        # Should work without explicit lon/lat
        with pytest.warns(UserWarning):  # Will warn about meshgrid
            result, interp = regrid(da, resolution=10.0)

        assert result.shape == interp.target_lon.shape

    def test_xarray_no_recognized_coords_raises(self):
        """Test that xarray without recognized coords raises error."""
        xr = pytest.importorskip("xarray")

        data_vals = np.random.rand(18, 36)
        da = xr.DataArray(
            data_vals,
            coords={"dim0": np.arange(18), "dim1": np.arange(36)},
            dims=["dim0", "dim1"],
        )

        with pytest.raises(ValueError, match="lon and lat coordinates are required"):
            regrid(da, resolution=10.0)


class TestRegridMultiLevel:
    """Tests for regrid function with multi-level unstructured data."""

    def test_multilevel_unstructured(self, random_mesh_small):
        """Test regrid with multi-level unstructured data (nlevels, npoints)."""
        lon, lat = random_mesh_small
        n_levels = 42
        data = np.random.rand(n_levels, len(lon))

        # Should work without warning (unstructured coords used directly)
        result, interp = regrid(data, lon, lat, resolution=5.0)

        # Result should have shape (nlevels, nlat, nlon)
        assert result.shape == (n_levels,) + interp.target_lon.shape

    def test_multilevel_unstructured_like_fesom(self):
        """Test regrid with FESOM-like data shape (nlevels, npoints)."""
        # Simulate FESOM-like unstructured mesh
        n_points = 196608
        n_levels = 42
        rng = np.random.default_rng(42)

        lon = rng.uniform(-180, 180, n_points)
        lat = rng.uniform(-90, 90, n_points)
        data = rng.random((n_levels, n_points))

        # This should NOT create a meshgrid - coords are unstructured
        result, interp = regrid(data, lon, lat, resolution=5.0)

        # Result should have shape (nlevels, nlat, nlon)
        assert result.shape == (n_levels,) + interp.target_lon.shape

    def test_regular_grid_vs_unstructured_distinction(self):
        """Test that regular grid and unstructured cases are handled correctly."""
        # Regular grid case: lon/lat have DIFFERENT sizes
        lon_1d = np.linspace(-180, 180, 36)  # nlon = 36
        lat_1d = np.linspace(-90, 90, 18)    # nlat = 18
        data_regular = np.random.rand(18, 36)  # (nlat, nlon)

        with pytest.warns(UserWarning, match="Creating meshgrid"):
            result_regular, _ = regrid(data_regular, lon_1d, lat_1d, resolution=10.0)

        # Unstructured case: lon/lat have SAME size
        n_points = 648  # 18 * 36
        lon_unstruct = np.random.uniform(-180, 180, n_points)
        lat_unstruct = np.random.uniform(-90, 90, n_points)
        data_unstruct = np.random.rand(n_points)

        # Should not warn about meshgrid
        result_unstruct, _ = regrid(data_unstruct, lon_unstruct, lat_unstruct, resolution=10.0)

        # Both should produce 2D output
        assert result_regular.ndim == 2
        assert result_unstruct.ndim == 2

    def test_3d_regular_grid(self):
        """Test regrid with 3D regular grid data (nlevels, nlat, nlon)."""
        lon_1d = np.linspace(-180, 180, 36)
        lat_1d = np.linspace(-90, 90, 18)
        n_levels = 10
        data_3d = np.random.rand(n_levels, 18, 36)  # (nlevels, nlat, nlon)

        with pytest.warns(UserWarning, match="Creating meshgrid"):
            result, interp = regrid(data_3d, lon_1d, lat_1d, resolution=10.0)

        # Result should have shape (nlevels, nlat_target, nlon_target)
        assert result.shape[0] == n_levels
        assert result.shape[1:] == interp.target_lon.shape


class TestInterpolator2DCoords:
    """Tests for RegridInterpolator with 2D coordinates."""

    def test_2d_coords_raveled(self):
        """Test that 2D coordinates are properly raveled."""
        lon_1d = np.linspace(-180, 180, 36)
        lat_1d = np.linspace(-90, 90, 18)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        with pytest.warns(UserWarning, match="Raveling 2D"):
            interp = RegridInterpolator(lon_2d, lat_2d, resolution=10.0)

        # Check that source coordinates are 1D
        assert interp.source_lon.ndim == 1
        assert interp.source_lat.ndim == 1
        assert interp.source_lon.size == 36 * 18
