"""Tests for core spatial utilities."""

import numpy as np
import pytest

from nereus.core.spatial import (
    find_nearest,
    haversine_distance,
    points_in_polygon,
    subset_by_bbox,
)


class TestFindNearest:
    """Tests for find_nearest function."""

    @pytest.fixture
    def grid_coords(self):
        """Create a simple 3x3 grid."""
        lon = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
        lat = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        return lon, lat

    def test_single_query_single_neighbor(self, grid_coords):
        """Test finding single nearest point for single query."""
        lon, lat = grid_coords

        idx = find_nearest(lon, lat, 0.1, 0.1)

        assert idx == 0  # Closest to (0, 0)

    def test_exact_match(self, grid_coords):
        """Test exact coordinate match."""
        lon, lat = grid_coords

        idx = find_nearest(lon, lat, 1.0, 1.0)

        assert idx == 4  # Exact match at center

    def test_single_query_with_distance(self, grid_coords):
        """Test distance return."""
        lon, lat = grid_coords

        idx, dist = find_nearest(lon, lat, 0.0, 0.0, return_distance=True)

        assert idx == 0
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_multiple_neighbors(self, grid_coords):
        """Test finding multiple nearest neighbors."""
        lon, lat = grid_coords

        indices = find_nearest(lon, lat, 1.0, 1.0, k=5)

        assert len(indices) == 5
        assert 4 in indices  # Center point should be included

    def test_array_query(self, grid_coords):
        """Test array of query points."""
        lon, lat = grid_coords

        query_lon = np.array([0.0, 2.0])
        query_lat = np.array([0.0, 2.0])

        indices = find_nearest(lon, lat, query_lon, query_lat)

        assert indices.shape == (2,)
        assert indices[0] == 0  # (0, 0)
        assert indices[1] == 8  # (2, 2)

    def test_array_query_with_k(self, grid_coords):
        """Test array query with k > 1."""
        lon, lat = grid_coords

        query_lon = np.array([0.0, 2.0])
        query_lat = np.array([0.0, 0.0])

        indices = find_nearest(lon, lat, query_lon, query_lat, k=2)

        assert indices.shape == (2, 2)

    def test_array_query_with_distance(self, grid_coords):
        """Test array query with distance return."""
        lon, lat = grid_coords

        query_lon = np.array([0.0, 1.0])
        query_lat = np.array([0.0, 1.0])

        indices, distances = find_nearest(
            lon, lat, query_lon, query_lat, return_distance=True
        )

        assert indices.shape == (2,)
        assert distances.shape == (2,)
        # Exact matches should have zero distance
        assert distances[0] == pytest.approx(0.0, abs=1e-6)
        assert distances[1] == pytest.approx(0.0, abs=1e-6)


class TestSubsetByBbox:
    """Tests for subset_by_bbox function."""

    @pytest.fixture
    def grid_coords(self):
        """Create a simple 3x3 grid."""
        lon = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
        lat = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        return lon, lat

    def test_basic_bbox(self, grid_coords):
        """Test basic bounding box selection."""
        lon, lat = grid_coords

        mask = subset_by_bbox(lon, lat, 0, 1, 0, 1)

        # Should select points at (0,0), (1,0), (0,1), (1,1)
        assert mask.dtype == np.bool_
        assert np.sum(mask) == 4

    def test_single_point(self, grid_coords):
        """Test bbox containing single point."""
        lon, lat = grid_coords

        mask = subset_by_bbox(lon, lat, 0.5, 1.5, 0.5, 1.5)

        # Should select only center point (1, 1)
        assert np.sum(mask) == 1
        assert mask[4] == True  # Center point

    def test_empty_bbox(self, grid_coords):
        """Test bbox with no points."""
        lon, lat = grid_coords

        mask = subset_by_bbox(lon, lat, 10, 20, 10, 20)

        assert np.sum(mask) == 0

    def test_full_selection(self, grid_coords):
        """Test bbox containing all points."""
        lon, lat = grid_coords

        mask = subset_by_bbox(lon, lat, -1, 3, -1, 3)

        assert np.sum(mask) == 9


class TestPointsInPolygon:
    """Tests for points_in_polygon function."""

    @pytest.fixture
    def grid_coords(self):
        """Create a simple 3x3 grid."""
        lon = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
        lat = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        return lon, lat

    def test_square_polygon(self, grid_coords):
        """Test points inside square polygon."""
        lon, lat = grid_coords

        # Square polygon around center
        poly_lon = [0.5, 1.5, 1.5, 0.5, 0.5]
        poly_lat = [0.5, 0.5, 1.5, 1.5, 0.5]

        mask = points_in_polygon(lon, lat, poly_lon, poly_lat)

        # Only center point should be inside
        assert mask[4] == True
        assert np.sum(mask) == 1

    def test_triangle_polygon(self, grid_coords):
        """Test points inside triangle polygon."""
        lon, lat = grid_coords

        # Large triangle
        poly_lon = [-0.5, 2.5, 1.0, -0.5]
        poly_lat = [-0.5, -0.5, 2.5, -0.5]

        mask = points_in_polygon(lon, lat, poly_lon, poly_lat)

        # Multiple points should be inside
        assert np.sum(mask) > 0


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_zero_distance(self):
        """Test distance to same point."""
        d = haversine_distance(0, 0, 0, 0)
        assert d == pytest.approx(0.0)

    def test_quarter_earth(self):
        """Test quarter Earth circumference distance."""
        # From equator at 0 to pole
        d = haversine_distance(0, 0, 0, 90)

        # Quarter of Earth circumference ~10,000 km
        assert d == pytest.approx(10_000_000, rel=0.01)

    def test_equator_90_degrees(self):
        """Test 90 degrees along equator."""
        d = haversine_distance(0, 0, 90, 0)

        # Quarter of Earth circumference ~10,000 km
        assert d == pytest.approx(10_000_000, rel=0.01)

    def test_array_input(self):
        """Test array input."""
        lon1 = np.array([0, 0])
        lat1 = np.array([0, 0])
        lon2 = np.array([0, 90])
        lat2 = np.array([0, 0])

        d = haversine_distance(lon1, lat1, lon2, lat2)

        assert d.shape == (2,)
        assert d[0] == pytest.approx(0.0)
        assert d[1] == pytest.approx(10_000_000, rel=0.01)

    def test_known_distance(self):
        """Test known city-to-city distance."""
        # New York to London ~5570 km
        nyc_lon, nyc_lat = -74.0, 40.7
        london_lon, london_lat = -0.1, 51.5

        d = haversine_distance(nyc_lon, nyc_lat, london_lon, london_lat)

        assert d == pytest.approx(5_570_000, rel=0.02)
