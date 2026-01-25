"""Tests for FESOM model support."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import nereus as nr
from nereus.models.fesom import load_mesh, open_dataset


@pytest.fixture
def simple_mesh_dir():
    """Create a simple temporary mesh directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_dir = Path(tmpdir)

        # Create simple nod2d.out file (4 nodes)
        with open(mesh_dir / "nod2d.out", "w") as f:
            f.write("4\n")  # Number of nodes
            f.write("1 0.0 0.0 0\n")  # Node 1: lon=0, lat=0
            f.write("2 1.0 0.0 0\n")  # Node 2: lon=1, lat=0
            f.write("3 0.5 1.0 0\n")  # Node 3: lon=0.5, lat=1
            f.write("4 1.5 1.0 0\n")  # Node 4: lon=1.5, lat=1

        # Create elem2d.out file (2 triangles)
        with open(mesh_dir / "elem2d.out", "w") as f:
            f.write("2\n")  # Number of elements
            f.write("1 2 3\n")  # Triangle 1
            f.write("2 4 3\n")  # Triangle 2

        # Create aux3d.out file (3 levels = 2 layers)
        with open(mesh_dir / "aux3d.out", "w") as f:
            f.write("3\n")  # Number of levels
            f.write("0.0\n")
            f.write("10.0\n")
            f.write("100.0\n")

        yield mesh_dir


@pytest.fixture
def minimal_mesh_dir():
    """Create a minimal mesh directory (only nod2d.out)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_dir = Path(tmpdir)

        # Create minimal nod2d.out file
        with open(mesh_dir / "nod2d.out", "w") as f:
            f.write("3\n")
            f.write("1 0.0 0.0 0\n")
            f.write("2 10.0 0.0 0\n")
            f.write("3 5.0 10.0 0\n")

        yield mesh_dir


class TestFesomMesh:
    """Tests for FESOM mesh loading (xr.Dataset-based)."""

    def test_load_basic_mesh(self, simple_mesh_dir):
        """Test loading a basic mesh returns xr.Dataset."""
        mesh = load_mesh(simple_mesh_dir)

        assert isinstance(mesh, xr.Dataset)
        assert mesh.sizes["npoints"] == 4
        assert "lon" in mesh
        assert "lat" in mesh
        assert "area" in mesh

    def test_node_coordinates(self, simple_mesh_dir):
        """Test node coordinates are loaded correctly."""
        mesh = load_mesh(simple_mesh_dir)

        np.testing.assert_array_almost_equal(mesh["lon"].values, [0.0, 1.0, 0.5, 1.5])
        np.testing.assert_array_almost_equal(mesh["lat"].values, [0.0, 0.0, 1.0, 1.0])

    def test_element_connectivity(self, simple_mesh_dir):
        """Test element connectivity is loaded correctly."""
        mesh = load_mesh(simple_mesh_dir)

        assert "triangles" in mesh
        assert mesh["triangles"].shape == (2, 3)
        # Should be 0-indexed
        np.testing.assert_array_equal(mesh["triangles"].values[0], [0, 1, 2])
        np.testing.assert_array_equal(mesh["triangles"].values[1], [1, 3, 2])

    def test_vertical_levels(self, simple_mesh_dir):
        """Test vertical level information."""
        mesh = load_mesh(simple_mesh_dir)

        assert "depth" in mesh
        assert mesh.sizes["depth_level"] == 2  # 3 interfaces = 2 layers
        np.testing.assert_array_almost_equal(mesh["depth"].values, [5.0, 55.0])

    def test_layer_thickness(self, simple_mesh_dir):
        """Test layer thickness computation."""
        mesh = load_mesh(simple_mesh_dir)

        assert "layer_thickness" in mesh
        thickness = mesh["layer_thickness"].values
        np.testing.assert_array_almost_equal(thickness, [10.0, 90.0])

    def test_depth_bounds(self, simple_mesh_dir):
        """Test depth bounds computation."""
        mesh = load_mesh(simple_mesh_dir)

        assert "depth_bounds" in mesh
        bounds = mesh["depth_bounds"].values
        np.testing.assert_array_almost_equal(bounds[0], [0.0, 10.0])
        np.testing.assert_array_almost_equal(bounds[1], [10.0, 100.0])

    def test_area_computed(self, simple_mesh_dir):
        """Test that area is computed from elements."""
        mesh = load_mesh(simple_mesh_dir)

        assert "area" in mesh
        assert len(mesh["area"]) == mesh.sizes["npoints"]
        assert np.all(mesh["area"].values > 0)  # All areas should be positive

    def test_element_centers(self, simple_mesh_dir):
        """Test element center coordinates."""
        mesh = load_mesh(simple_mesh_dir)

        assert "lon_tri" in mesh
        assert "lat_tri" in mesh
        assert mesh.sizes["nelem"] == 2

    def test_minimal_mesh(self, minimal_mesh_dir):
        """Test loading minimal mesh without elements or depth."""
        mesh = load_mesh(minimal_mesh_dir)

        assert mesh.sizes["npoints"] == 3
        assert "area" in mesh
        assert len(mesh["area"]) == 3

    def test_mesh_metadata(self, simple_mesh_dir):
        """Test nereus mesh metadata attributes."""
        mesh = load_mesh(simple_mesh_dir)

        assert mesh.attrs["nereus_mesh_type"] == "fesom"
        assert "nereus_mesh_version" in mesh.attrs
        assert mesh.attrs["nereus_dask_backend"] is False

    def test_mesh_not_found(self):
        """Test error when mesh files not found."""
        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/path")

    def test_use_dask_explicit(self, simple_mesh_dir):
        """Test explicit dask usage."""
        # Force dask even for small mesh
        mesh = load_mesh(simple_mesh_dir, use_dask=True)

        assert mesh.attrs["nereus_dask_backend"] is True

    def test_lon_normalization(self, minimal_mesh_dir):
        """Test longitude normalization to [-180, 180]."""
        # Create mesh with lon > 180
        with open(minimal_mesh_dir / "nod2d.out", "w") as f:
            f.write("2\n")
            f.write("1 270.0 0.0 0\n")  # 270 should become -90
            f.write("2 0.0 0.0 0\n")

        mesh = load_mesh(minimal_mesh_dir)
        assert mesh["lon"].values[0] == pytest.approx(-90.0)


class TestSpatialFunctions:
    """Tests for standalone spatial functions with FESOM mesh."""

    def test_find_nearest_single(self, simple_mesh_dir):
        """Test finding single nearest point."""
        mesh = load_mesh(simple_mesh_dir)
        lon = mesh["lon"].values
        lat = mesh["lat"].values

        # Find nearest to (0.1, 0.1) - should be node 0 at (0, 0)
        idx = nr.find_nearest(lon, lat, 0.1, 0.1)

        assert idx == 0  # Closest to first node

    def test_find_nearest_with_distance(self, simple_mesh_dir):
        """Test finding nearest with distance."""
        mesh = load_mesh(simple_mesh_dir)
        lon = mesh["lon"].values
        lat = mesh["lat"].values

        idx, dist = nr.find_nearest(lon, lat, 0.0, 0.0, return_distance=True)

        assert idx == 0
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_find_nearest_multiple(self, simple_mesh_dir):
        """Test finding multiple nearest points."""
        mesh = load_mesh(simple_mesh_dir)
        lon = mesh["lon"].values
        lat = mesh["lat"].values

        # Find 2 nearest to (0.5, 0.5)
        indices = nr.find_nearest(lon, lat, 0.5, 0.5, k=2)

        assert len(indices) == 2
        assert len(set(indices)) == 2  # Two different indices

    def test_find_nearest_array(self, simple_mesh_dir):
        """Test finding nearest for array of query points."""
        mesh = load_mesh(simple_mesh_dir)
        lon = mesh["lon"].values
        lat = mesh["lat"].values

        query_lon = np.array([0.0, 1.0])
        query_lat = np.array([0.0, 0.0])

        indices = nr.find_nearest(lon, lat, query_lon, query_lat)

        assert indices.shape == (2,)
        assert indices[0] == 0  # First query -> node 0
        assert indices[1] == 1  # Second query -> node 1

    def test_subset_by_bbox(self, simple_mesh_dir):
        """Test bounding box subsetting."""
        mesh = load_mesh(simple_mesh_dir)
        lon = mesh["lon"].values
        lat = mesh["lat"].values

        # Select only left half
        mask = nr.subset_by_bbox(lon, lat, 0.0, 0.75, -1.0, 2.0)

        assert mask.shape == (4,)
        assert mask.dtype == np.bool_
        # Nodes at lon=0.0, 0.5 should be selected (nodes 0 and 2)
        assert np.sum(mask) == 2

    def test_subset_by_bbox_empty(self, simple_mesh_dir):
        """Test bbox subsetting with no points inside."""
        mesh = load_mesh(simple_mesh_dir)
        lon = mesh["lon"].values
        lat = mesh["lat"].values

        mask = nr.subset_by_bbox(lon, lat, 100.0, 110.0, 0.0, 10.0)
        assert np.sum(mask) == 0


class TestFesomFunctions:
    """Tests for FESOM-specific functions."""

    def test_node_to_element(self, simple_mesh_dir):
        """Test node to element interpolation."""
        from nereus.models.fesom import node_to_element

        mesh = load_mesh(simple_mesh_dir)

        # Create node data
        node_data = np.array([1.0, 2.0, 3.0, 4.0])

        elem_data = node_to_element(node_data, mesh)

        assert elem_data.shape == (2,)  # 2 elements
        # Element 0: nodes [0, 1, 2] with values [1, 2, 3] -> mean = 2
        assert elem_data[0] == pytest.approx(2.0)
        # Element 1: nodes [1, 3, 2] with values [2, 4, 3] -> mean = 3
        assert elem_data[1] == pytest.approx(3.0)

    def test_element_to_node(self, simple_mesh_dir):
        """Test element to node interpolation."""
        from nereus.models.fesom import element_to_node

        mesh = load_mesh(simple_mesh_dir)

        # Create element data
        elem_data = np.array([1.0, 2.0])

        node_data = element_to_node(elem_data, mesh)

        assert node_data.shape == (4,)  # 4 nodes

    def test_compute_element_centers(self, simple_mesh_dir):
        """Test computing element centers."""
        from nereus.models.fesom import compute_element_centers

        # Create mesh without element centers
        mesh_dir = simple_mesh_dir
        mesh = load_mesh(mesh_dir)

        # Element centers should already be computed by load_mesh
        assert "lon_tri" in mesh
        assert "lat_tri" in mesh


class TestFesomDataset:
    """Tests for FESOM dataset handling."""

    def test_open_dataset_without_mesh(self):
        """Test error when opening dataset without mesh."""
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            with pytest.raises(ValueError, match="Either mesh or mesh_path"):
                open_dataset(f.name)


class TestMeshValidation:
    """Tests for mesh validation utilities."""

    def test_is_nereus_mesh(self, simple_mesh_dir):
        """Test nereus mesh detection."""
        from nereus.core.mesh import is_nereus_mesh

        mesh = load_mesh(simple_mesh_dir)
        assert is_nereus_mesh(mesh) is True

        # Regular dataset should not be nereus mesh
        ds = xr.Dataset({"x": [1, 2, 3]})
        assert is_nereus_mesh(ds) is False

    def test_validate_mesh(self, simple_mesh_dir):
        """Test mesh validation."""
        from nereus.core.mesh import validate_mesh

        mesh = load_mesh(simple_mesh_dir)
        errors = validate_mesh(mesh)

        assert errors == []  # No errors for valid mesh

    def test_validate_mesh_missing_vars(self):
        """Test validation catches missing variables."""
        from nereus.core.mesh import validate_mesh

        ds = xr.Dataset({"x": [1, 2, 3]})
        errors = validate_mesh(ds)

        assert len(errors) > 0
        assert any("lon" in e for e in errors)

    def test_validate_mesh_strict(self):
        """Test strict validation raises error."""
        from nereus.core.mesh import validate_mesh

        ds = xr.Dataset({"x": [1, 2, 3]})

        with pytest.raises(ValueError):
            validate_mesh(ds, strict=True)


class TestUniversalLoader:
    """Tests for universal mesh loader."""

    def test_load_mesh_fesom(self, simple_mesh_dir):
        """Test universal loader with FESOM mesh."""
        mesh = nr.load_mesh(simple_mesh_dir)

        assert isinstance(mesh, xr.Dataset)
        assert mesh.attrs["nereus_mesh_type"] == "fesom"

    def test_load_mesh_explicit_type(self, simple_mesh_dir):
        """Test universal loader with explicit type."""
        mesh = nr.load_mesh(simple_mesh_dir, mesh_type="fesom")

        assert mesh.attrs["nereus_mesh_type"] == "fesom"

    def test_load_mesh_unknown_type(self):
        """Test error for unknown mesh type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Could not auto-detect"):
                nr.load_mesh(tmpdir)


class TestMaskByDepth:
    """Tests for mask_by_depth function."""

    def test_mask_by_depth_2d(self):
        """Test masking 2D data (single level)."""
        from nereus.models.fesom import mask_by_depth

        # Create test data: 5 points
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Mask: points 2 and 4 are invalid (below bottom)
        mask = np.array([True, True, False, True, False])

        result = mask_by_depth(data, mask)

        assert result.shape == (5,)
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert np.isnan(result[2])
        assert result[3] == 4.0
        assert np.isnan(result[4])

    def test_mask_by_depth_3d(self):
        """Test masking 3D data (multiple levels)."""
        from nereus.models.fesom import mask_by_depth

        # Create test data: 3 levels, 4 points
        data = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Level 0
            [5.0, 6.0, 7.0, 8.0],  # Level 1
            [9.0, 10.0, 11.0, 12.0],  # Level 2
        ])
        # Mask: progressive bottom - point 3 ends at level 1, point 4 ends at level 0
        mask = np.array([
            [True, True, True, True],   # Level 0: all valid
            [True, True, True, False],  # Level 1: point 4 below bottom
            [True, True, False, False],  # Level 2: points 3,4 below bottom
        ])

        result = mask_by_depth(data, mask)

        assert result.shape == (3, 4)
        # Level 0: all valid
        np.testing.assert_array_equal(result[0, :], [1.0, 2.0, 3.0, 4.0])
        # Level 1: point 4 masked
        assert result[1, 0] == 5.0
        assert result[1, 1] == 6.0
        assert result[1, 2] == 7.0
        assert np.isnan(result[1, 3])
        # Level 2: points 3,4 masked
        assert result[2, 0] == 9.0
        assert result[2, 1] == 10.0
        assert np.isnan(result[2, 2])
        assert np.isnan(result[2, 3])

    def test_mask_by_depth_xarray(self):
        """Test masking works with xarray DataArrays."""
        from nereus.models.fesom import mask_by_depth

        data = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("npoints",)
        )
        mask = xr.DataArray(
            np.array([True, False, True]),
            dims=("npoints",)
        )

        result = mask_by_depth(data, mask)

        assert isinstance(result, np.ndarray)
        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert result[2] == 3.0

    def test_mask_by_depth_shape_mismatch(self):
        """Test error on shape mismatch."""
        from nereus.models.fesom import mask_by_depth

        data = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, False])  # Wrong size

        with pytest.raises(ValueError, match="does not match mask shape"):
            mask_by_depth(data, mask)

    def test_mask_by_depth_preserves_dtype(self):
        """Test that result is float64 to support NaN."""
        from nereus.models.fesom import mask_by_depth

        data = np.array([1, 2, 3], dtype=np.int32)
        mask = np.array([True, False, True])

        result = mask_by_depth(data, mask)

        assert result.dtype == np.float64
