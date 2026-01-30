"""Tests for nereus.core.grids."""

import numpy as np
import pytest

from nereus.core.grids import (
    create_regular_grid,
    extract_coordinates,
    grid_cell_area,
    prepare_coordinates,
    prepare_input_arrays,
)


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


class TestPrepareInputArrays:
    """Tests for prepare_input_arrays function."""

    def test_all_1d_same_size(self):
        """Test all 1D arrays of same size pass through unchanged."""
        data = np.random.rand(100)
        lon = np.random.uniform(-180, 180, 100)
        lat = np.random.uniform(-90, 90, 100)

        d, lo, la = prepare_input_arrays(data, lon, lat)

        np.testing.assert_array_equal(d, data)
        np.testing.assert_array_equal(lo, lon)
        np.testing.assert_array_equal(la, lat)

    def test_all_2d_same_shape(self):
        """Test all 2D arrays of same shape are raveled."""
        data = np.random.rand(10, 20)
        lon = np.random.uniform(-180, 180, (10, 20))
        lat = np.random.uniform(-90, 90, (10, 20))

        with pytest.warns(UserWarning, match="Raveling 2D arrays"):
            d, lo, la = prepare_input_arrays(data, lon, lat)

        assert d.ndim == 1
        assert d.size == 200
        assert lo.size == 200
        assert la.size == 200

    def test_1d_data_2d_coords(self):
        """Test 1D data with 2D lon/lat (pre-ravelled case)."""
        data = np.random.rand(200)
        lon = np.random.uniform(-180, 180, (10, 20))
        lat = np.random.uniform(-90, 90, (10, 20))

        with pytest.warns(UserWarning, match="Raveling 2D lon/lat"):
            d, lo, la = prepare_input_arrays(data, lon, lat)

        assert d.size == lo.size == la.size == 200

    def test_2d_data_1d_coords_meshgrid(self):
        """Test 2D data with 1D coords creates meshgrid."""
        data = np.random.rand(18, 36)  # (ny, nx)
        lon = np.linspace(-180, 180, 36)  # nx
        lat = np.linspace(-90, 90, 18)  # ny

        with pytest.warns(UserWarning, match="Creating meshgrid"):
            d, lo, la = prepare_input_arrays(data, lon, lat)

        assert d.size == 18 * 36
        assert lo.size == 18 * 36
        assert la.size == 18 * 36

    def test_1d_size_mismatch_raises(self):
        """Test that 1D arrays with different sizes raise error."""
        data = np.random.rand(100)
        lon = np.random.rand(200)
        lat = np.random.rand(100)

        with pytest.raises(ValueError, match="1D arrays must have the same size"):
            prepare_input_arrays(data, lon, lat)

    def test_2d_shape_mismatch_raises(self):
        """Test that 2D arrays with different shapes raise error."""
        data = np.random.rand(10, 20)
        lon = np.random.rand(10, 30)
        lat = np.random.rand(10, 20)

        with pytest.raises(ValueError, match="2D arrays must have the same shape"):
            prepare_input_arrays(data, lon, lat)

    def test_2d_data_wrong_1d_lon_size_raises(self):
        """Test 2D data with wrong 1D lon size raises error."""
        data = np.random.rand(10, 20)
        lon = np.random.rand(15)  # Should be 20
        lat = np.random.rand(10)

        with pytest.raises(ValueError, match="1D lon array size"):
            prepare_input_arrays(data, lon, lat)

    def test_2d_data_wrong_1d_lat_size_raises(self):
        """Test 2D data with wrong 1D lat size raises error."""
        data = np.random.rand(10, 20)
        lon = np.random.rand(20)
        lat = np.random.rand(15)  # Should be 10

        with pytest.raises(ValueError, match="1D lat array size"):
            prepare_input_arrays(data, lon, lat)

    def test_mixed_1d_2d_coords_raises(self):
        """Test that mixed 1D/2D lon/lat raises error."""
        data = np.random.rand(100)
        lon = np.random.rand(10, 10)  # 2D
        lat = np.random.rand(100)  # 1D

        with pytest.raises(ValueError, match="both be 1D or both be 2D"):
            prepare_input_arrays(data, lon, lat)


class TestPrepareCoordinates:
    """Tests for prepare_coordinates function."""

    def test_both_1d_same_size(self):
        """Test both 1D with same size pass through."""
        lon = np.random.uniform(-180, 180, 100)
        lat = np.random.uniform(-90, 90, 100)

        lo, la = prepare_coordinates(lon, lat)

        np.testing.assert_array_equal(lo, lon)
        np.testing.assert_array_equal(la, lat)

    def test_both_2d_same_shape(self):
        """Test both 2D with same shape are raveled."""
        lon = np.random.uniform(-180, 180, (10, 20))
        lat = np.random.uniform(-90, 90, (10, 20))

        with pytest.warns(UserWarning, match="Raveling 2D lon/lat"):
            lo, la = prepare_coordinates(lon, lat)

        assert lo.size == 200
        assert la.size == 200

    def test_both_1d_different_size_meshgrid(self):
        """Test 1D coords with different sizes creates meshgrid."""
        lon = np.linspace(-180, 180, 36)
        lat = np.linspace(-90, 90, 18)

        with pytest.warns(UserWarning, match="Creating meshgrid"):
            lo, la = prepare_coordinates(lon, lat)

        assert lo.size == 36 * 18
        assert la.size == 36 * 18

    def test_mixed_1d_2d_raises(self):
        """Test mixed 1D/2D raises error."""
        lon = np.random.rand(10, 10)
        lat = np.random.rand(100)

        with pytest.raises(ValueError, match="both be 1D or both be 2D"):
            prepare_coordinates(lon, lat)


class TestExtractCoordinates:
    """Tests for extract_coordinates function."""

    def test_non_xarray_returns_none(self):
        """Test that non-xarray input returns (None, None)."""
        data = np.random.rand(100)

        lon, lat = extract_coordinates(data)

        assert lon is None
        assert lat is None

    def test_xarray_with_lon_lat(self):
        """Test extraction from xarray with lon/lat coords."""
        xr = pytest.importorskip("xarray")

        lon_vals = np.linspace(-180, 180, 36)
        lat_vals = np.linspace(-90, 90, 18)
        data_vals = np.random.rand(18, 36)

        da = xr.DataArray(
            data_vals,
            coords={"lat": lat_vals, "lon": lon_vals},
            dims=["lat", "lon"],
        )

        lon, lat = extract_coordinates(da)

        assert lon is not None
        assert lat is not None
        np.testing.assert_array_equal(lon, lon_vals)
        np.testing.assert_array_equal(lat, lat_vals)

    def test_xarray_with_longitude_latitude(self):
        """Test extraction with longitude/latitude names."""
        xr = pytest.importorskip("xarray")

        lon_vals = np.linspace(-180, 180, 36)
        lat_vals = np.linspace(-90, 90, 18)
        data_vals = np.random.rand(18, 36)

        da = xr.DataArray(
            data_vals,
            coords={"latitude": lat_vals, "longitude": lon_vals},
            dims=["latitude", "longitude"],
        )

        lon, lat = extract_coordinates(da)

        assert lon is not None
        assert lat is not None
        np.testing.assert_array_equal(lon, lon_vals)
        np.testing.assert_array_equal(lat, lat_vals)

    def test_xarray_with_x_y(self):
        """Test extraction with x/y names."""
        xr = pytest.importorskip("xarray")

        x_vals = np.linspace(-180, 180, 36)
        y_vals = np.linspace(-90, 90, 18)
        data_vals = np.random.rand(18, 36)

        da = xr.DataArray(
            data_vals,
            coords={"y": y_vals, "x": x_vals},
            dims=["y", "x"],
        )

        lon, lat = extract_coordinates(da)

        assert lon is not None
        assert lat is not None
        np.testing.assert_array_equal(lon, x_vals)
        np.testing.assert_array_equal(lat, y_vals)

    def test_xarray_case_insensitive(self):
        """Test that coordinate names are matched case-insensitively."""
        xr = pytest.importorskip("xarray")

        lon_vals = np.linspace(-180, 180, 36)
        lat_vals = np.linspace(-90, 90, 18)
        data_vals = np.random.rand(18, 36)

        da = xr.DataArray(
            data_vals,
            coords={"LAT": lat_vals, "LON": lon_vals},
            dims=["LAT", "LON"],
        )

        lon, lat = extract_coordinates(da)

        assert lon is not None
        assert lat is not None

    def test_xarray_without_coords_returns_none(self):
        """Test that xarray without recognized coords returns None."""
        xr = pytest.importorskip("xarray")

        data_vals = np.random.rand(18, 36)
        da = xr.DataArray(
            data_vals,
            coords={"dim0": np.arange(18), "dim1": np.arange(36)},
            dims=["dim0", "dim1"],
        )

        lon, lat = extract_coordinates(da)

        assert lon is None
        assert lat is None

    def test_xarray_2d_coords(self):
        """Test extraction of 2D coordinates (curvilinear grids)."""
        xr = pytest.importorskip("xarray")

        # Create 2D coordinates (like a curvilinear grid)
        y_idx = np.arange(18)
        x_idx = np.arange(36)
        lon_2d = np.random.uniform(-180, 180, (18, 36))
        lat_2d = np.random.uniform(-90, 90, (18, 36))
        data_vals = np.random.rand(18, 36)

        da = xr.DataArray(
            data_vals,
            coords={
                "y": y_idx,
                "x": x_idx,
                "lon": (["y", "x"], lon_2d),
                "lat": (["y", "x"], lat_2d),
            },
            dims=["y", "x"],
        )

        lon, lat = extract_coordinates(da)

        assert lon is not None
        assert lat is not None
        np.testing.assert_array_equal(lon, lon_2d)
        np.testing.assert_array_equal(lat, lat_2d)
