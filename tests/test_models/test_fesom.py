"""Tests for FESOM model support."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nereus.models.fesom import FesomMesh, load_mesh


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
    """Tests for FesomMesh class."""

    def test_load_basic_mesh(self, simple_mesh_dir):
        """Test loading a basic mesh."""
        mesh = load_mesh(simple_mesh_dir)

        assert mesh.n2d == 4
        assert len(mesh.lon) == 4
        assert len(mesh.lat) == 4

    def test_node_coordinates(self, simple_mesh_dir):
        """Test node coordinates are loaded correctly."""
        mesh = load_mesh(simple_mesh_dir)

        np.testing.assert_array_almost_equal(mesh.lon, [0.0, 1.0, 0.5, 1.5])
        np.testing.assert_array_almost_equal(mesh.lat, [0.0, 0.0, 1.0, 1.0])

    def test_element_connectivity(self, simple_mesh_dir):
        """Test element connectivity is loaded correctly."""
        mesh = load_mesh(simple_mesh_dir)

        assert mesh.elem.shape == (2, 3)
        # Should be 0-indexed
        np.testing.assert_array_equal(mesh.elem[0], [0, 1, 2])
        np.testing.assert_array_equal(mesh.elem[1], [1, 3, 2])

    def test_vertical_levels(self, simple_mesh_dir):
        """Test vertical level information."""
        mesh = load_mesh(simple_mesh_dir)

        assert mesh.nlev == 2  # 3 interfaces = 2 layers
        np.testing.assert_array_almost_equal(mesh.depth_lev, [0.0, 10.0, 100.0])
        np.testing.assert_array_almost_equal(mesh.depth, [5.0, 55.0])

    def test_layer_thickness(self, simple_mesh_dir):
        """Test layer thickness computation."""
        mesh = load_mesh(simple_mesh_dir)

        thickness = mesh.layer_thickness
        np.testing.assert_array_almost_equal(thickness, [10.0, 90.0])

    def test_n3d(self, simple_mesh_dir):
        """Test total 3D node count."""
        mesh = load_mesh(simple_mesh_dir)

        assert mesh.n3d == mesh.n2d * mesh.nlev
        assert mesh.n3d == 4 * 2

    def test_area_computed(self, simple_mesh_dir):
        """Test that area is computed from elements."""
        mesh = load_mesh(simple_mesh_dir)

        assert len(mesh.area) == mesh.n2d
        assert np.all(mesh.area > 0)  # All areas should be positive

    def test_minimal_mesh(self, minimal_mesh_dir):
        """Test loading minimal mesh without elements or depth."""
        mesh = load_mesh(minimal_mesh_dir)

        assert mesh.n2d == 3
        assert len(mesh.area) == 3
        assert mesh.nlev == 1  # Default single level

    def test_repr(self, simple_mesh_dir):
        """Test string representation."""
        mesh = load_mesh(simple_mesh_dir)
        repr_str = repr(mesh)

        assert "FesomMesh" in repr_str
        assert "n2d=4" in repr_str
        assert "nlev=2" in repr_str

    def test_mesh_not_found(self):
        """Test error when mesh files not found."""
        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/path")


class TestFesomMeshBase:
    """Tests for MeshBase interface methods in FesomMesh."""

    def test_npoints(self, simple_mesh_dir):
        """Test npoints property from MeshBase."""
        mesh = load_mesh(simple_mesh_dir)
        assert mesh.npoints == mesh.n2d

    def test_find_nearest_single(self, simple_mesh_dir):
        """Test finding single nearest point."""
        mesh = load_mesh(simple_mesh_dir)

        # Find nearest to (0.1, 0.1) - should be node 0 at (0, 0)
        distances, indices = mesh.find_nearest(0.1, 0.1)

        assert indices.shape == (1,)
        assert indices[0] == 0  # Closest to first node

    def test_find_nearest_multiple(self, simple_mesh_dir):
        """Test finding multiple nearest points."""
        mesh = load_mesh(simple_mesh_dir)

        # Find 2 nearest to (0.5, 0.5)
        distances, indices = mesh.find_nearest(0.5, 0.5, k=2)

        assert indices.shape == (1, 2)
        assert len(set(indices[0])) == 2  # Two different indices

    def test_find_nearest_array(self, simple_mesh_dir):
        """Test finding nearest for array of query points."""
        mesh = load_mesh(simple_mesh_dir)

        query_lon = np.array([0.0, 1.0])
        query_lat = np.array([0.0, 0.0])

        distances, indices = mesh.find_nearest(query_lon, query_lat)

        assert indices.shape == (2,)
        assert indices[0] == 0  # First query -> node 0
        assert indices[1] == 1  # Second query -> node 1

    def test_subset_by_bbox(self, simple_mesh_dir):
        """Test bounding box subsetting."""
        mesh = load_mesh(simple_mesh_dir)

        # Select only left half
        mask = mesh.subset_by_bbox(0.0, 0.75, -1.0, 2.0)

        assert mask.shape == (4,)
        assert mask.dtype == np.bool_
        # Nodes at lon=0.0, 0.5 should be selected (nodes 0 and 2)
        assert np.sum(mask) == 2

    def test_subset_by_bbox_empty(self, simple_mesh_dir):
        """Test bbox subsetting with no points inside."""
        mesh = load_mesh(simple_mesh_dir)

        mask = mesh.subset_by_bbox(100.0, 110.0, 0.0, 10.0)
        assert np.sum(mask) == 0


class TestFesomDataset:
    """Tests for FESOM dataset handling."""

    def test_open_dataset_without_mesh(self):
        """Test error when opening dataset without mesh."""
        from nereus.models.fesom import open_dataset

        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            with pytest.raises(ValueError, match="Either mesh or mesh_path"):
                open_dataset(f.name)
