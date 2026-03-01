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

        with pytest.warns(UserWarning, match="Raveling 2D"):
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


class TestRegridAsXarray:
    """Tests for regrid() with as_xarray=True."""

    def test_returns_dataarray(self, random_mesh_small, synthetic_data):
        """as_xarray=True returns an xr.DataArray."""
        xr = pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        result, _ = regrid(synthetic_data, lon, lat, resolution=5.0, as_xarray=True)
        assert isinstance(result, xr.DataArray)

    def test_has_lat_lon_dims(self, random_mesh_small, synthetic_data):
        """Output DataArray has lat and lon as the last two dimensions."""
        pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        result, _ = regrid(synthetic_data, lon, lat, resolution=5.0, as_xarray=True)
        assert result.dims[-2:] == ("lat", "lon")

    def test_coordinate_values(self, random_mesh_small, synthetic_data):
        """lat/lon coordinate values match the interpolator's target grid."""
        pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        result, interp = regrid(synthetic_data, lon, lat, resolution=5.0, as_xarray=True)
        np.testing.assert_array_equal(result.lat.values, interp.target_lat[:, 0])
        np.testing.assert_array_equal(result.lon.values, interp.target_lon[0, :])

    def test_false_returns_ndarray(self, random_mesh_small, synthetic_data):
        """Default (as_xarray=False) still returns a plain ndarray."""
        lon, lat = random_mesh_small
        result, _ = regrid(synthetic_data, lon, lat, resolution=5.0)
        assert isinstance(result, np.ndarray)

    def test_leading_dims_numpy_input(self, random_mesh_small):
        """Plain numpy input with leading dims produces correct dim names."""
        pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        n_levels = 4
        data = np.random.rand(n_levels, len(lon))
        result, _ = regrid(data, lon, lat, resolution=5.0, as_xarray=True)
        assert result.dims == ("dim_0", "lat", "lon")
        assert result.shape[0] == n_levels

    def test_leading_dims_xarray_input(self, random_mesh_small):
        """xr.DataArray input: leading dim names and coords are preserved."""
        xr = pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        n_levels = 4
        depth = np.arange(n_levels, dtype=float)
        data_vals = np.random.rand(n_levels, len(lon))
        da = xr.DataArray(
            data_vals,
            dims=["depth", "npoints"],
            coords={"depth": depth, "lon": ("npoints", lon), "lat": ("npoints", lat)},
            name="temperature",
            attrs={"units": "degC"},
        )
        result, _ = regrid(da, resolution=5.0, as_xarray=True)
        assert result.dims == ("depth", "lat", "lon")
        np.testing.assert_array_equal(result.depth.values, depth)
        assert result.name == "temperature"
        assert result.attrs == {"units": "degC"}

    def test_var_name_and_attrs_from_xarray(self, random_mesh_small, synthetic_data):
        """Variable name and attributes are copied from an xr.DataArray input."""
        xr = pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        da = xr.DataArray(
            synthetic_data,
            dims=["npoints"],
            coords={"lon": ("npoints", lon), "lat": ("npoints", lat)},
            name="salinity",
            attrs={"units": "psu", "long_name": "Sea surface salinity"},
        )
        result, _ = regrid(da, resolution=5.0, as_xarray=True)
        assert result.name == "salinity"
        assert result.attrs == {"units": "psu", "long_name": "Sea surface salinity"}

    def test_plain_numpy_name_and_attrs(self, random_mesh_small, synthetic_data):
        """Plain numpy input produces name='data' and empty attrs."""
        pytest.importorskip("xarray")
        lon, lat = random_mesh_small
        result, _ = regrid(synthetic_data, lon, lat, resolution=5.0, as_xarray=True)
        assert result.name == "data"
        assert result.attrs == {}


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


class TestLinearInterpolation:
    """Tests for linear interpolation method."""

    def test_linear_basic(self):
        """Test basic linear interpolation."""
        # Create a regular source grid
        lon_1d = np.linspace(-10, 10, 20)
        lat_1d = np.linspace(-10, 10, 20)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="linear",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )
        result = interp(data)

        assert result.shape == interp.target_lon.shape
        # Should have some valid (finite) values
        assert np.isfinite(result).any()

    def test_linear_smoother_than_nearest(self):
        """Linear interpolation should produce smoother results than nearest."""
        lon_1d = np.linspace(-10, 10, 10)
        lat_1d = np.linspace(-10, 10, 10)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp_nn = RegridInterpolator(
            lon, lat, resolution=1.0, method="nearest",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )
        interp_lin = RegridInterpolator(
            lon, lat, resolution=1.0, method="linear",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )

        result_nn = interp_nn(data)
        result_lin = interp_lin(data)

        # Count unique values — nearest neighbor has fewer unique values
        # (many target cells map to same source point)
        valid_nn = result_nn[interp_nn.valid_mask]
        valid_lin = result_lin[np.isfinite(result_lin)]

        if len(valid_nn) > 0 and len(valid_lin) > 0:
            assert len(np.unique(valid_lin)) >= len(np.unique(valid_nn))

    def test_linear_nan_outside_hull(self):
        """Linear interpolation returns NaN outside convex hull of source."""
        # Small cluster of source points
        rng = np.random.default_rng(42)
        lon = rng.uniform(0, 10, 100)
        lat = rng.uniform(0, 10, 100)
        data = np.ones(100)

        interp = RegridInterpolator(
            lon, lat, resolution=5.0, method="linear",
            influence_radius=5_000_000,  # very large to not interfere
        )
        result = interp(data)

        # Points far from data region should be NaN
        far_mask = (interp.target_lon < -30) | (interp.target_lon > 40)
        assert np.isnan(result[far_mask]).all()

    def test_linear_exact_for_linear_field(self):
        """Linear interpolation reproduces a linear field exactly."""
        # Dense source grid covering the target region
        lon_1d = np.linspace(-10, 10, 21)
        lat_1d = np.linspace(-10, 10, 21)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        # Linear field: f(lon, lat) = 2*lon + 3*lat
        data = 2.0 * lon + 3.0 * lat

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="linear",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )
        result = interp(data)

        # At each valid target point, check the linear field is reproduced
        expected = 2.0 * interp.target_lon + 3.0 * interp.target_lat
        valid = np.isfinite(result)
        np.testing.assert_allclose(result[valid], expected[valid], atol=0.1)

    def test_linear_multidim(self):
        """Test linear interpolation with multi-dimensional data."""
        lon_1d = np.linspace(-10, 10, 15)
        lat_1d = np.linspace(-10, 10, 15)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        n_levels = 3
        data = np.random.default_rng(42).random((n_levels, len(lon)))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="linear",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )
        result = interp(data)

        assert result.shape == (n_levels,) + interp.target_lon.shape

    def test_linear_0_360_source_lon(self):
        """Linear interpolation works when source uses 0-360 and target -180-180."""
        # Simulate EN4-style data with 0-360 longitude convention
        lon_1d = np.linspace(0.5, 359.5, 360)
        lat_1d = np.linspace(-89.5, 89.5, 180)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        # Simple smooth field
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="linear",
            influence_radius=500_000,
        )
        result = interp(data)

        # The entire grid should be covered (no NaN gaps)
        assert np.isfinite(result[interp.valid_mask]).all()
        # Most of the domain should be valid
        assert interp.valid_mask.sum() > 0.9 * interp.valid_mask.size

    def test_linear_0_360_matches_pm180(self):
        """Linear results should be equivalent regardless of source lon convention."""
        lat_1d = np.linspace(-85, 85, 18)

        # Source in 0-360
        lon_0360 = np.linspace(5, 355, 36)
        lon2d_0360, lat2d_0360 = np.meshgrid(lon_0360, lat_1d)
        lon_flat_0360 = lon2d_0360.ravel()
        lat_flat = lat2d_0360.ravel()
        data = np.sin(np.deg2rad(lat_flat)) * np.cos(np.deg2rad(lon_flat_0360))

        # Source in -180-180 (same physical locations)
        lon_pm180 = np.where(lon_0360 > 180, lon_0360 - 360, lon_0360)
        lon2d_pm180, _ = np.meshgrid(lon_pm180, lat_1d)
        lon_flat_pm180 = lon2d_pm180.ravel()
        data_pm = np.sin(np.deg2rad(lat_flat)) * np.cos(np.deg2rad(lon_flat_pm180))

        interp_0360 = RegridInterpolator(
            lon_flat_0360, lat_flat, resolution=10.0, method="linear",
            lon_bounds=(-180, 180), lat_bounds=(-80, 80),
            influence_radius=2_000_000,
        )
        interp_pm180 = RegridInterpolator(
            lon_flat_pm180, lat_flat, resolution=10.0, method="linear",
            lon_bounds=(-180, 180), lat_bounds=(-80, 80),
            influence_radius=2_000_000,
        )

        result_0360 = interp_0360(data)
        result_pm180 = interp_pm180(data_pm)

        # Both should cover the same domain with similar values
        both_valid = np.isfinite(result_0360) & np.isfinite(result_pm180)
        assert both_valid.sum() > 0
        np.testing.assert_allclose(
            result_0360[both_valid], result_pm180[both_valid], atol=0.1,
        )

    def test_linear_with_regrid_function(self):
        """Test linear interpolation through regrid convenience function."""
        lon_1d = np.linspace(-10, 10, 20)
        lat_1d = np.linspace(-10, 10, 20)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat))

        result, interp = regrid(
            data, lon, lat, resolution=2.0, method="linear",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )

        assert result.shape == interp.target_lon.shape
        assert interp.method == "linear"

    def test_invalid_method_raises(self, random_mesh_small, synthetic_data):
        """Test that invalid method raises ValueError."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=5.0, method="nearest")
        # Forcibly set invalid method to test runtime error
        interp.method = "invalid"
        with pytest.raises(ValueError, match="Unknown method"):
            interp(synthetic_data)


class TestIDWInterpolation:
    """Tests for IDW (inverse distance weighting) interpolation method."""

    def test_idw_basic(self):
        """Test basic IDW interpolation produces valid output."""
        lon_1d = np.linspace(-10, 10, 20)
        lat_1d = np.linspace(-10, 10, 20)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="idw",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )
        result = interp(data)

        assert result.shape == interp.target_lon.shape
        assert np.isfinite(result[interp.valid_mask]).all()

    def test_idw_smoother_than_nearest(self):
        """IDW should produce more unique values than nearest neighbor."""
        lon_1d = np.linspace(-10, 10, 10)
        lat_1d = np.linspace(-10, 10, 10)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp_nn = RegridInterpolator(
            lon, lat, resolution=1.0, method="nearest",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )
        interp_idw = RegridInterpolator(
            lon, lat, resolution=1.0, method="idw",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )

        result_nn = interp_nn(data)
        result_idw = interp_idw(data)

        valid_nn = result_nn[interp_nn.valid_mask]
        valid_idw = result_idw[interp_idw.valid_mask]

        if len(valid_nn) > 0 and len(valid_idw) > 0:
            assert len(np.unique(valid_idw)) >= len(np.unique(valid_nn))

    def test_idw_exact_at_source(self):
        """IDW should reproduce values close to source at nearby locations."""
        # Dense source grid so IDW has good coverage
        lon_1d = np.linspace(-10, 10, 21)
        lat_1d = np.linspace(-10, 10, 21)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = 2.0 * lon + 3.0 * lat

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="idw",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )
        result = interp(data)

        # IDW should approximate the linear field reasonably well
        expected = 2.0 * interp.target_lon + 3.0 * interp.target_lat
        valid = interp.valid_mask & np.isfinite(result)
        np.testing.assert_allclose(result[valid], expected[valid], atol=2.0)

    def test_idw_multidim(self):
        """Test IDW interpolation with multi-dimensional data."""
        lon_1d = np.linspace(-10, 10, 15)
        lat_1d = np.linspace(-10, 10, 15)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        n_levels = 3
        data = np.random.default_rng(42).random((n_levels, len(lon)))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="idw",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )
        result = interp(data)

        assert result.shape == (n_levels,) + interp.target_lon.shape


class TestCubicInterpolation:
    """Tests for cubic (Clough-Tocher) interpolation method."""

    def test_cubic_basic(self):
        """Test basic cubic interpolation produces valid output."""
        lon_1d = np.linspace(-10, 10, 20)
        lat_1d = np.linspace(-10, 10, 20)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="cubic",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )
        result = interp(data)

        assert result.shape == interp.target_lon.shape
        assert np.isfinite(result).any()

    def test_cubic_smoother_than_linear(self):
        """Cubic interpolation should produce at least as many unique values as linear."""
        lon_1d = np.linspace(-10, 10, 10)
        lat_1d = np.linspace(-10, 10, 10)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp_lin = RegridInterpolator(
            lon, lat, resolution=1.0, method="linear",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )
        interp_cub = RegridInterpolator(
            lon, lat, resolution=1.0, method="cubic",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )

        result_lin = interp_lin(data)
        result_cub = interp_cub(data)

        valid_lin = result_lin[np.isfinite(result_lin)]
        valid_cub = result_cub[np.isfinite(result_cub)]

        if len(valid_lin) > 0 and len(valid_cub) > 0:
            assert len(np.unique(valid_cub)) >= len(np.unique(valid_lin))

    def test_cubic_exact_for_linear_field(self):
        """Cubic interpolation reproduces a linear field exactly."""
        lon_1d = np.linspace(-10, 10, 21)
        lat_1d = np.linspace(-10, 10, 21)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = 2.0 * lon + 3.0 * lat

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="cubic",
            lon_bounds=(-8, 8), lat_bounds=(-8, 8),
            influence_radius=500_000,
        )
        result = interp(data)

        expected = 2.0 * interp.target_lon + 3.0 * interp.target_lat
        valid = np.isfinite(result)
        np.testing.assert_allclose(result[valid], expected[valid], atol=0.1)

    def test_cubic_multidim(self):
        """Test cubic interpolation with multi-dimensional data."""
        lon_1d = np.linspace(-10, 10, 15)
        lat_1d = np.linspace(-10, 10, 15)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        n_levels = 3
        data = np.random.default_rng(42).random((n_levels, len(lon)))

        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="cubic",
            lon_bounds=(-10, 10), lat_bounds=(-10, 10),
        )
        result = interp(data)

        assert result.shape == (n_levels,) + interp.target_lon.shape


class TestNaNSourceData:
    """Tests for interpolation with NaN values in source data (e.g. land masking)."""

    def _make_en4_like_data(self):
        """Create EN4-like regular grid data with ~30% NaN (land)."""
        lon_1d = np.arange(0.5, 360, 1.0)
        lat_1d = np.arange(-89.5, 90, 1.0)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon = lon_2d.ravel()
        lat = lat_2d.ravel()
        data = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
        rng = np.random.default_rng(42)
        data[rng.random(len(data)) < 0.3] = np.nan
        return lon, lat, data

    def test_cubic_with_nan_source(self):
        """Cubic interpolation should produce valid output despite NaN in source."""
        lon, lat, data = self._make_en4_like_data()
        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="cubic",
        )
        result = interp(data)
        valid = np.isfinite(result)
        assert valid.sum() > 0.5 * result.size

    def test_cubic_nan_values_in_range(self):
        """Cubic with NaN source should not produce wild overshoots."""
        lon, lat, data = self._make_en4_like_data()
        interp = RegridInterpolator(
            lon, lat, resolution=2.0, method="cubic",
        )
        result = interp(data)
        valid = result[np.isfinite(result)]
        assert np.abs(valid).max() < 2.0
