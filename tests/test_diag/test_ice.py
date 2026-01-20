"""Tests for sea ice diagnostics."""

import numpy as np
import pytest

from nereus.diag.ice import ice_area, ice_extent, ice_volume


class TestIceArea:
    """Tests for ice_area function."""

    def test_ice_area_basic(self):
        """Test basic ice area calculation."""
        # 4 cells of equal area with varying concentration
        concentration = np.array([0.0, 0.5, 0.8, 1.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])  # 1 km^2 each

        result = ice_area(concentration, area)

        # Expected: 0*1e6 + 0.5*1e6 + 0.8*1e6 + 1.0*1e6 = 2.3e6
        assert result == pytest.approx(2.3e6)

    def test_ice_area_zero_concentration(self):
        """Test ice area with zero concentration."""
        concentration = np.array([0.0, 0.0, 0.0, 0.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_area(concentration, area)
        assert result == 0.0

    def test_ice_area_full_concentration(self):
        """Test ice area with full concentration."""
        concentration = np.array([1.0, 1.0, 1.0, 1.0])
        area = np.array([1e6, 2e6, 3e6, 4e6])

        result = ice_area(concentration, area)
        assert result == pytest.approx(10e6)

    def test_ice_area_with_mask(self):
        """Test ice area with spatial mask."""
        concentration = np.array([0.5, 0.5, 0.5, 0.5])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        mask = np.array([True, True, False, False])

        result = ice_area(concentration, area, mask=mask)
        assert result == pytest.approx(1e6)  # Only first two cells

    def test_ice_area_with_nan(self):
        """Test ice area handles NaN values."""
        concentration = np.array([0.5, np.nan, 0.5, 1.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_area(concentration, area)
        # NaN treated as 0: 0.5*1e6 + 0 + 0.5*1e6 + 1.0*1e6 = 2e6
        assert result == pytest.approx(2e6)

    def test_ice_area_multidimensional(self):
        """Test ice area with time dimension."""
        # (2 timesteps, 4 points)
        concentration = np.array([
            [0.0, 0.5, 0.8, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_area(concentration, area)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(2.3e6)
        assert result[1] == pytest.approx(2e6)

    def test_ice_area_clips_concentration(self):
        """Test that concentration values outside [0, 1] are clipped."""
        concentration = np.array([1.5, -0.5, 0.5, 0.5])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_area(concentration, area)
        # Clipped: 1.0*1e6 + 0*1e6 + 0.5*1e6 + 0.5*1e6 = 2e6
        assert result == pytest.approx(2e6)


class TestIceVolume:
    """Tests for ice_volume function."""

    def test_ice_volume_basic(self):
        """Test basic ice volume calculation."""
        thickness = np.array([0.0, 1.0, 2.0, 3.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_volume(thickness, area)
        # Volume where ice exists (thickness > 0): 1*1e6 + 2*1e6 + 3*1e6 = 6e6
        assert result == pytest.approx(6e6)

    def test_ice_volume_with_concentration(self):
        """Test ice volume with concentration."""
        thickness = np.array([2.0, 2.0, 2.0, 2.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        concentration = np.array([0.5, 0.5, 1.0, 0.0])

        result = ice_volume(thickness, area, concentration=concentration)
        # Volume: 2*0.5*1e6 + 2*0.5*1e6 + 2*1.0*1e6 + 0 = 4e6
        assert result == pytest.approx(4e6)

    def test_ice_volume_with_mask(self):
        """Test ice volume with mask."""
        thickness = np.array([1.0, 1.0, 1.0, 1.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        mask = np.array([True, True, False, False])

        result = ice_volume(thickness, area, mask=mask)
        assert result == pytest.approx(2e6)

    def test_ice_volume_zero_thickness(self):
        """Test ice volume with zero thickness."""
        thickness = np.array([0.0, 0.0, 0.0, 0.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_volume(thickness, area)
        assert result == 0.0

    def test_ice_volume_multidimensional(self):
        """Test ice volume with time dimension."""
        thickness = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0, 0.0],
        ])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_volume(thickness, area)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(4e6)
        assert result[1] == pytest.approx(4e6)


class TestIceExtent:
    """Tests for ice_extent function."""

    def test_ice_extent_default_threshold(self):
        """Test ice extent with default 15% threshold."""
        concentration = np.array([0.0, 0.10, 0.15, 0.20, 1.0])
        area = np.array([1e6, 1e6, 1e6, 1e6, 1e6])

        result = ice_extent(concentration, area)
        # Cells with conc >= 0.15: [0.15, 0.20, 1.0] = 3 cells
        assert result == pytest.approx(3e6)

    def test_ice_extent_custom_threshold(self):
        """Test ice extent with custom threshold."""
        concentration = np.array([0.0, 0.30, 0.50, 0.80, 1.0])
        area = np.array([1e6, 1e6, 1e6, 1e6, 1e6])

        result = ice_extent(concentration, area, threshold=0.5)
        # Cells with conc >= 0.5: [0.50, 0.80, 1.0] = 3 cells
        assert result == pytest.approx(3e6)

    def test_ice_extent_with_mask(self):
        """Test ice extent with mask."""
        concentration = np.array([1.0, 1.0, 1.0, 1.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        mask = np.array([True, True, False, False])

        result = ice_extent(concentration, area, mask=mask)
        assert result == pytest.approx(2e6)

    def test_ice_extent_zero(self):
        """Test ice extent when no ice above threshold."""
        concentration = np.array([0.0, 0.05, 0.10, 0.14])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_extent(concentration, area)
        assert result == 0.0

    def test_ice_extent_all_ice(self):
        """Test ice extent when all above threshold."""
        concentration = np.array([0.20, 0.50, 0.80, 1.0])
        area = np.array([1e6, 2e6, 3e6, 4e6])

        result = ice_extent(concentration, area)
        assert result == pytest.approx(10e6)

    def test_ice_extent_multidimensional(self):
        """Test ice extent with time dimension."""
        concentration = np.array([
            [0.0, 0.10, 0.20, 1.0],
            [1.0, 1.0, 0.10, 0.0],
        ])
        area = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_extent(concentration, area)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(2e6)  # [0.20, 1.0]
        assert result[1] == pytest.approx(2e6)  # [1.0, 1.0]
