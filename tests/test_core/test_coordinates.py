"""Tests for nereus.core.coordinates."""

import numpy as np
import pytest

from nereus.core.coordinates import (
    EARTH_RADIUS,
    cartesian_to_lonlat,
    chord_to_meters,
    great_circle_distance,
    great_circle_path,
    lonlat_to_cartesian,
    meters_to_chord,
)


class TestLonLatToCartesian:
    """Tests for lonlat_to_cartesian function."""

    def test_origin(self):
        """Test that (0, 0) maps to (1, 0, 0) on unit sphere."""
        x, y, z = lonlat_to_cartesian(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(x, [1.0], rtol=1e-10)
        np.testing.assert_allclose(y, [0.0], atol=1e-10)
        np.testing.assert_allclose(z, [0.0], atol=1e-10)

    def test_north_pole(self):
        """Test that (0, 90) maps to (0, 0, 1)."""
        x, y, z = lonlat_to_cartesian(np.array([0.0]), np.array([90.0]))
        np.testing.assert_allclose(x, [0.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0], atol=1e-10)
        np.testing.assert_allclose(z, [1.0], rtol=1e-10)

    def test_south_pole(self):
        """Test that (0, -90) maps to (0, 0, -1)."""
        x, y, z = lonlat_to_cartesian(np.array([0.0]), np.array([-90.0]))
        np.testing.assert_allclose(x, [0.0], atol=1e-10)
        np.testing.assert_allclose(y, [0.0], atol=1e-10)
        np.testing.assert_allclose(z, [-1.0], rtol=1e-10)

    def test_lon_90(self):
        """Test that (90, 0) maps to (0, 1, 0)."""
        x, y, z = lonlat_to_cartesian(np.array([90.0]), np.array([0.0]))
        np.testing.assert_allclose(x, [0.0], atol=1e-10)
        np.testing.assert_allclose(y, [1.0], rtol=1e-10)
        np.testing.assert_allclose(z, [0.0], atol=1e-10)

    def test_unit_length(self):
        """Test that all points are on the unit sphere."""
        rng = np.random.default_rng(42)
        lon = rng.uniform(-180, 180, 100)
        lat = rng.uniform(-90, 90, 100)

        x, y, z = lonlat_to_cartesian(lon, lat)
        lengths = np.sqrt(x**2 + y**2 + z**2)

        np.testing.assert_allclose(lengths, 1.0, rtol=1e-10)

    def test_custom_radius(self):
        """Test with custom radius."""
        radius = 6371000.0
        x, y, z = lonlat_to_cartesian(
            np.array([0.0]), np.array([0.0]), radius=radius
        )
        np.testing.assert_allclose(x, [radius], rtol=1e-10)


class TestCartesianToLonLat:
    """Tests for cartesian_to_lonlat function."""

    def test_roundtrip(self):
        """Test that conversion is reversible."""
        rng = np.random.default_rng(42)
        lon = rng.uniform(-180, 180, 100)
        lat = rng.uniform(-90, 90, 100)

        x, y, z = lonlat_to_cartesian(lon, lat)
        lon2, lat2 = cartesian_to_lonlat(x, y, z)

        np.testing.assert_allclose(lon, lon2, rtol=1e-10)
        np.testing.assert_allclose(lat, lat2, rtol=1e-10)


class TestMetersToChord:
    """Tests for meters_to_chord function."""

    def test_zero_distance(self):
        """Test that zero meters gives zero chord."""
        chord = meters_to_chord(0.0)
        assert chord == pytest.approx(0.0, abs=1e-10)

    def test_quarter_earth(self):
        """Test that 1/4 Earth circumference gives chord = sqrt(2)."""
        # 1/4 circumference = pi * R / 2
        quarter_circ = np.pi * EARTH_RADIUS / 2
        chord = meters_to_chord(quarter_circ)
        # For 90 degree arc, chord = 2 * sin(45) = sqrt(2)
        np.testing.assert_allclose(chord, np.sqrt(2), rtol=1e-10)

    def test_small_distance_approximation(self):
        """Test that for small distances, chord ≈ arc/R."""
        # 1 km is small compared to Earth radius
        meters = 1000.0
        chord = meters_to_chord(meters)
        approx = meters / EARTH_RADIUS
        # Should be very close for small distances
        np.testing.assert_allclose(chord, approx, rtol=1e-4)


class TestChordToMeters:
    """Tests for chord_to_meters function."""

    def test_roundtrip(self):
        """Test that conversion is reversible."""
        meters_orig = 500000.0
        chord = meters_to_chord(meters_orig)
        meters_back = chord_to_meters(chord)
        assert meters_back == pytest.approx(meters_orig, rel=1e-10)


class TestGreatCircleDistance:
    """Tests for great_circle_distance function."""

    def test_zero_distance(self):
        """Test distance between same point is zero."""
        dist = great_circle_distance(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0])
        )
        np.testing.assert_allclose(dist, [0.0], atol=1e-10)

    def test_quarter_earth(self):
        """Test distance from equator to pole."""
        dist = great_circle_distance(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([90.0])
        )
        expected = np.pi * EARTH_RADIUS / 2
        np.testing.assert_allclose(dist, [expected], rtol=1e-10)

    def test_half_earth(self):
        """Test distance between antipodal points."""
        dist = great_circle_distance(
            np.array([0.0]), np.array([0.0]),
            np.array([180.0]), np.array([0.0])
        )
        expected = np.pi * EARTH_RADIUS
        np.testing.assert_allclose(dist, [expected], rtol=1e-10)


class TestGreatCirclePath:
    """Tests for great_circle_path function."""

    def test_endpoints(self):
        """Test that path starts and ends at correct points."""
        start_lon, start_lat = -10.0, 40.0
        end_lon, end_lat = 20.0, 60.0

        lon, lat = great_circle_path(start_lon, start_lat, end_lon, end_lat, n_points=50)

        np.testing.assert_allclose(lon[0], start_lon, rtol=1e-10)
        np.testing.assert_allclose(lat[0], start_lat, rtol=1e-10)
        np.testing.assert_allclose(lon[-1], end_lon, rtol=1e-10)
        np.testing.assert_allclose(lat[-1], end_lat, rtol=1e-10)

    def test_n_points(self):
        """Test that correct number of points is returned."""
        lon, lat = great_circle_path(0, 0, 10, 10, n_points=100)
        assert len(lon) == 100
        assert len(lat) == 100
