"""Tests for core mesh utilities."""

import numpy as np
import pytest
import xarray as xr

from nereus.core.mesh import (
    DASK_THRESHOLD_POINTS,
    add_mesh_metadata,
    create_lonlat_mesh,
    ensure_lon_pm180,
    get_mesh_type,
    is_nereus_mesh,
    mesh_from_arrays,
    normalize_lon,
    should_use_dask,
    validate_mesh,
)


class TestNormalizeLon:
    """Tests for longitude normalization."""

    def test_normalize_pm180(self):
        """Test normalization to [-180, 180]."""
        lon = np.array([0, 90, 180, 270, 360, -90, -180])
        result = normalize_lon(lon, "pm180")

        expected = np.array([0, 90, -180, -90, 0, -90, -180])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_0360(self):
        """Test normalization to [0, 360]."""
        lon = np.array([0, 90, -90, -180, 180, 270])
        result = normalize_lon(lon, "0360")

        expected = np.array([0, 90, 270, 180, 180, 270])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_convention(self):
        """Test error for invalid convention."""
        with pytest.raises(ValueError):
            normalize_lon(np.array([0]), "invalid")


class TestShouldUseDask:
    """Tests for dask auto-detection."""

    def test_small_mesh_no_dask(self):
        """Test small mesh doesn't use dask."""
        assert should_use_dask(1000) is False
        assert should_use_dask(DASK_THRESHOLD_POINTS - 1) is False

    def test_large_mesh_uses_dask(self):
        """Test large mesh uses dask."""
        assert should_use_dask(DASK_THRESHOLD_POINTS + 1) is True
        assert should_use_dask(10_000_000) is True

    def test_explicit_override(self):
        """Test explicit dask setting overrides auto."""
        assert should_use_dask(100, use_dask=True) is True
        assert should_use_dask(10_000_000, use_dask=False) is False


class TestCreateLonLatMesh:
    """Tests for regular lon-lat mesh creation."""

    def test_basic_creation(self):
        """Test basic mesh creation."""
        mesh = create_lonlat_mesh(10.0)

        assert isinstance(mesh, xr.Dataset)
        assert "lon" in mesh
        assert "lat" in mesh
        assert "area" in mesh
        assert mesh.attrs["nereus_mesh_type"] == "lonlat"

    def test_resolution_tuple(self):
        """Test mesh with different lon/lat resolution."""
        mesh = create_lonlat_mesh((10.0, 5.0))

        # Check that we have more latitude points than with equal resolution
        mesh_equal = create_lonlat_mesh(10.0)
        assert mesh.attrs["nlat"] > mesh_equal.attrs["nlat"]

    def test_custom_bounds(self):
        """Test mesh with custom bounds."""
        mesh = create_lonlat_mesh(1.0, lon_bounds=(0, 10), lat_bounds=(0, 10))

        lon_vals = mesh["lon"].values
        lat_vals = mesh["lat"].values

        assert lon_vals.min() >= 0
        assert lon_vals.max() <= 10
        assert lat_vals.min() >= 0
        assert lat_vals.max() <= 10

    def test_area_positive(self):
        """Test all areas are positive."""
        mesh = create_lonlat_mesh(5.0)

        assert np.all(mesh["area"].values > 0)


class TestMeshFromArrays:
    """Tests for mesh creation from arrays."""

    def test_basic_creation(self):
        """Test basic mesh creation from arrays."""
        lon = np.array([0, 1, 2])
        lat = np.array([0, 0, 0])

        mesh = mesh_from_arrays(lon, lat)

        assert isinstance(mesh, xr.Dataset)
        assert mesh.sizes["npoints"] == 3
        np.testing.assert_array_equal(mesh["lon"].values, lon)

    def test_with_area(self):
        """Test mesh creation with provided area."""
        lon = np.array([0, 1, 2])
        lat = np.array([0, 0, 0])
        area = np.array([1e6, 1e6, 1e6])

        mesh = mesh_from_arrays(lon, lat, area=area)

        np.testing.assert_array_equal(mesh["area"].values, area)

    def test_2d_input_flattened(self):
        """Test 2D input is flattened."""
        lon = np.array([[0, 1], [2, 3]])
        lat = np.array([[0, 0], [1, 1]])

        mesh = mesh_from_arrays(lon, lat)

        assert mesh.sizes["npoints"] == 4

    def test_lon_normalized(self):
        """Test longitude is normalized."""
        lon = np.array([270, 180, 0])
        lat = np.array([0, 0, 0])

        mesh = mesh_from_arrays(lon, lat)

        assert mesh["lon"].values[0] == pytest.approx(-90)


class TestEnsureLonPM180:
    """Tests for dataset longitude normalization."""

    def test_normalizes_out_of_range(self):
        """Test out-of-range longitudes are normalized."""
        ds = xr.Dataset({
            "lon": (("x",), [270, 180, 0]),
            "lat": (("x",), [0, 0, 0]),
        })

        result = ensure_lon_pm180(ds)

        assert result["lon"].values[0] == pytest.approx(-90)

    def test_preserves_in_range(self):
        """Test in-range longitudes are unchanged."""
        ds = xr.Dataset({
            "lon": (("x",), [-90, 0, 90]),
            "lat": (("x",), [0, 0, 0]),
        })

        result = ensure_lon_pm180(ds)

        np.testing.assert_array_equal(result["lon"].values, ds["lon"].values)


class TestValidateMesh:
    """Tests for mesh validation."""

    def test_valid_mesh(self):
        """Test validation of valid mesh."""
        ds = xr.Dataset({
            "lon": (("npoints",), [0, 1]),
            "lat": (("npoints",), [0, 0]),
            "area": (("npoints",), [1e6, 1e6]),
        })

        errors = validate_mesh(ds)
        assert errors == []

    def test_missing_lon(self):
        """Test validation catches missing lon."""
        ds = xr.Dataset({
            "lat": (("npoints",), [0, 0]),
            "area": (("npoints",), [1e6, 1e6]),
        })

        errors = validate_mesh(ds)
        assert any("lon" in e for e in errors)

    def test_missing_area(self):
        """Test validation catches missing area."""
        ds = xr.Dataset({
            "lon": (("npoints",), [0, 1]),
            "lat": (("npoints",), [0, 0]),
        })

        errors = validate_mesh(ds)
        assert any("area" in e for e in errors)

    def test_lon_out_of_range(self):
        """Test validation catches out-of-range lon."""
        ds = xr.Dataset({
            "lon": (("npoints",), [200, 1]),
            "lat": (("npoints",), [0, 0]),
            "area": (("npoints",), [1e6, 1e6]),
        })

        errors = validate_mesh(ds)
        assert any("180" in e for e in errors)

    def test_negative_area(self):
        """Test validation catches negative area."""
        ds = xr.Dataset({
            "lon": (("npoints",), [0, 1]),
            "lat": (("npoints",), [0, 0]),
            "area": (("npoints",), [-1e6, 1e6]),
        })

        errors = validate_mesh(ds)
        assert any("non-positive" in e for e in errors)

    def test_strict_mode(self):
        """Test strict mode raises error."""
        ds = xr.Dataset({"x": [1, 2, 3]})

        with pytest.raises(ValueError):
            validate_mesh(ds, strict=True)


class TestMeshMetadata:
    """Tests for mesh metadata functions."""

    def test_add_metadata(self):
        """Test adding metadata to dataset."""
        ds = xr.Dataset({"x": [1, 2, 3]})

        result = add_mesh_metadata(ds, "fesom", "/path/to/mesh")

        assert result.attrs["nereus_mesh_type"] == "fesom"
        assert result.attrs["nereus_source_path"] == "/path/to/mesh"
        assert result.attrs["nereus_dask_backend"] is False

    def test_is_nereus_mesh(self):
        """Test nereus mesh detection."""
        ds = xr.Dataset({"x": [1, 2, 3]})

        assert is_nereus_mesh(ds) is False

        ds_with_meta = add_mesh_metadata(ds, "fesom")
        assert is_nereus_mesh(ds_with_meta) is True

    def test_get_mesh_type(self):
        """Test mesh type extraction."""
        ds = xr.Dataset({"x": [1, 2, 3]})

        assert get_mesh_type(ds) is None

        ds_with_meta = add_mesh_metadata(ds, "healpix")
        assert get_mesh_type(ds_with_meta) == "healpix"
