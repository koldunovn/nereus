"""Tests for sea ice diagnostics."""

import numpy as np
import pytest

from nereus.diag.ice import (
    ice_area,
    ice_area_nh,
    ice_area_sh,
    ice_extent,
    ice_extent_nh,
    ice_extent_sh,
    ice_volume,
    ice_volume_nh,
    ice_volume_sh,
)
from nereus.core.types import is_dask_array


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


class TestDaskCompatibility:
    """Tests for dask array compatibility."""

    @pytest.fixture
    def dask_deps(self):
        """Import dask and xarray, skip if not available."""
        da = pytest.importorskip("dask.array")
        xr = pytest.importorskip("xarray")
        return da, xr

    def test_ice_area_dask_xarray(self, dask_deps):
        """Test ice_area with xarray DataArray backed by dask."""
        da, xr = dask_deps

        concentration_np = np.array([0.0, 0.5, 0.8, 1.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])

        # Create xarray with dask backend
        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=2),
            dims=["npoints"],
        )

        result = ice_area(concentration, area_np)

        # Should return lazy dask array
        assert is_dask_array(result)

        # Compute and check result
        computed = float(result.compute())
        assert computed == pytest.approx(2.3e6)

    def test_ice_area_dask_multidimensional(self, dask_deps):
        """Test ice_area with multidimensional dask array."""
        da, xr = dask_deps

        concentration_np = np.array([
            [0.0, 0.5, 0.8, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])

        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=(1, 4)),
            dims=["time", "npoints"],
        )

        result = ice_area(concentration, area_np)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed[0] == pytest.approx(2.3e6)
        assert computed[1] == pytest.approx(2e6)

    def test_ice_area_dask_with_mask(self, dask_deps):
        """Test ice_area with dask array and mask."""
        da, xr = dask_deps

        concentration_np = np.array([0.5, 0.5, 0.5, 0.5])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])
        mask_np = np.array([True, True, False, False])

        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=2),
            dims=["npoints"],
        )

        result = ice_area(concentration, area_np, mask=mask_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(1e6)

    def test_ice_volume_dask_effective_thickness(self, dask_deps):
        """Test ice_volume with dask array (effective thickness)."""
        da, xr = dask_deps

        thickness_np = np.array([0.0, 1.0, 2.0, 3.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])

        thickness = xr.DataArray(
            da.from_array(thickness_np, chunks=2),
            dims=["npoints"],
        )

        result = ice_volume(thickness, area_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(6e6)

    def test_ice_volume_dask_real_thickness(self, dask_deps):
        """Test ice_volume with dask array (real thickness with concentration)."""
        da, xr = dask_deps

        thickness_np = np.array([2.0, 2.0, 2.0, 2.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])
        concentration_np = np.array([0.5, 0.5, 1.0, 0.0])

        thickness = xr.DataArray(
            da.from_array(thickness_np, chunks=2),
            dims=["npoints"],
        )
        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=2),
            dims=["npoints"],
        )

        result = ice_volume(thickness, area_np, concentration=concentration)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(4e6)

    def test_ice_volume_dask_multidimensional(self, dask_deps):
        """Test ice_volume with multidimensional dask array."""
        da, xr = dask_deps

        thickness_np = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0, 0.0],
        ])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])

        thickness = xr.DataArray(
            da.from_array(thickness_np, chunks=(1, 4)),
            dims=["time", "npoints"],
        )

        result = ice_volume(thickness, area_np)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed[0] == pytest.approx(4e6)
        assert computed[1] == pytest.approx(4e6)

    def test_ice_extent_dask_xarray(self, dask_deps):
        """Test ice_extent with xarray DataArray backed by dask."""
        da, xr = dask_deps

        concentration_np = np.array([0.0, 0.10, 0.15, 0.20, 1.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6, 1e6])

        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=3),
            dims=["npoints"],
        )

        result = ice_extent(concentration, area_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(3e6)

    def test_ice_extent_dask_multidimensional(self, dask_deps):
        """Test ice_extent with multidimensional dask array."""
        da, xr = dask_deps

        concentration_np = np.array([
            [0.0, 0.10, 0.20, 1.0],
            [1.0, 1.0, 0.10, 0.0],
        ])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])

        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=(1, 4)),
            dims=["time", "npoints"],
        )

        result = ice_extent(concentration, area_np)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed[0] == pytest.approx(2e6)
        assert computed[1] == pytest.approx(2e6)

    def test_ice_extent_dask_with_mask(self, dask_deps):
        """Test ice_extent with dask array and mask."""
        da, xr = dask_deps

        concentration_np = np.array([1.0, 1.0, 1.0, 1.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])
        mask_np = np.array([True, True, False, False])

        concentration = xr.DataArray(
            da.from_array(concentration_np, chunks=2),
            dims=["npoints"],
        )

        result = ice_extent(concentration, area_np, mask=mask_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(2e6)

    def test_numpy_input_returns_eager(self, dask_deps):
        """Test that numpy input still returns eager (non-dask) result."""
        concentration_np = np.array([0.0, 0.5, 0.8, 1.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])

        result = ice_area(concentration_np, area_np)

        # Should return float, not dask array
        assert not is_dask_array(result)
        assert isinstance(result, float)
        assert result == pytest.approx(2.3e6)


class TestHemisphereHelpers:
    """Tests for hemisphere convenience functions."""

    def test_ice_area_nh(self):
        """Test Northern Hemisphere ice area."""
        # 4 cells: 2 in NH (lat > 0), 2 in SH (lat < 0)
        concentration = np.array([0.5, 0.5, 0.5, 0.5])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])

        result = ice_area_nh(concentration, area, lat)
        # Only NH cells: 0.5*1e6 + 0.5*1e6 = 1e6
        assert result == pytest.approx(1e6)

    def test_ice_area_sh(self):
        """Test Southern Hemisphere ice area."""
        concentration = np.array([0.5, 0.5, 0.5, 0.5])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])

        result = ice_area_sh(concentration, area, lat)
        # Only SH cells: 0.5*1e6 + 0.5*1e6 = 1e6
        assert result == pytest.approx(1e6)

    def test_ice_volume_nh(self):
        """Test Northern Hemisphere ice volume."""
        thickness = np.array([2.0, 2.0, 2.0, 2.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])

        result = ice_volume_nh(thickness, area, lat)
        # Only NH cells: 2*1e6 + 2*1e6 = 4e6
        assert result == pytest.approx(4e6)

    def test_ice_volume_sh_with_concentration(self):
        """Test Southern Hemisphere ice volume with concentration."""
        thickness = np.array([2.0, 2.0, 2.0, 2.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])
        concentration = np.array([0.5, 0.5, 0.5, 0.5])

        result = ice_volume_sh(thickness, area, lat, concentration)
        # Only SH cells with concentration: 2*0.5*1e6 + 2*0.5*1e6 = 2e6
        assert result == pytest.approx(2e6)

    def test_ice_extent_nh(self):
        """Test Northern Hemisphere ice extent."""
        concentration = np.array([0.5, 0.1, 0.5, 0.1])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])

        result = ice_extent_nh(concentration, area, lat)
        # Only NH cells with conc >= 0.15: first cell only = 1e6
        assert result == pytest.approx(1e6)

    def test_ice_extent_sh_custom_threshold(self):
        """Test Southern Hemisphere ice extent with custom threshold."""
        concentration = np.array([0.5, 0.3, 0.5, 0.3])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])

        result = ice_extent_sh(concentration, area, lat, threshold=0.4)
        # SH cells with conc >= 0.4: third cell only = 1e6
        assert result == pytest.approx(1e6)

    def test_hemisphere_helpers_multidimensional(self):
        """Test hemisphere helpers with time dimension."""
        concentration = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0, 0.0],
        ])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        lat = np.array([45.0, 60.0, -45.0, -60.0])

        result_nh = ice_area_nh(concentration, area, lat)
        result_sh = ice_area_sh(concentration, area, lat)

        assert result_nh.shape == (2,)
        assert result_sh.shape == (2,)
        # Time 0: NH = 0.5+0.5 = 1.0e6, SH = 0.5+0.5 = 1.0e6
        assert result_nh[0] == pytest.approx(1e6)
        assert result_sh[0] == pytest.approx(1e6)
        # Time 1: NH = 1.0+0.0 = 1.0e6, SH = 1.0+0.0 = 1.0e6
        assert result_nh[1] == pytest.approx(1e6)
        assert result_sh[1] == pytest.approx(1e6)
