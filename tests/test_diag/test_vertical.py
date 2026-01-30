"""Tests for vertical/ocean diagnostics."""

import numpy as np
import pytest

from nereus.core.types import is_dask_array
from nereus.diag.vertical import (
    RHO_SEAWATER,
    CP_SEAWATER,
    find_closest_depth,
    heat_content,
    interpolate_to_depth,
    surface_mean,
    volume_mean,
)


class TestSurfaceMean:
    """Tests for surface_mean function."""

    def test_surface_mean_basic(self):
        """Test basic area-weighted mean."""
        data = np.array([10.0, 20.0, 30.0])
        area = np.array([1e6, 1e6, 1e6])

        result = surface_mean(data, area)

        # All cells have same area, so mean is (10 + 20 + 30) / 3 = 20
        assert result == pytest.approx(20.0)

    def test_surface_mean_varying_area(self):
        """Test surface mean with varying cell areas."""
        data = np.array([10.0, 20.0])
        area = np.array([3e6, 1e6])  # First cell is 3x larger

        result = surface_mean(data, area)

        # Total area: 4e6
        # Weighted sum: 10*3e6 + 20*1e6 = 50e6
        # Mean: 50e6 / 4e6 = 12.5
        assert result == pytest.approx(12.5)

    def test_surface_mean_with_mask(self):
        """Test surface mean with spatial mask."""
        data = np.array([10.0, 20.0, 30.0, 40.0])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        mask = np.array([True, True, False, False])

        result = surface_mean(data, area, mask=mask)

        # Only first two cells: (10 + 20) / 2 = 15
        assert result == pytest.approx(15.0)

    def test_surface_mean_with_nan(self):
        """Test surface mean handles NaN values."""
        data = np.array([10.0, np.nan, 30.0])
        area = np.array([1e6, 1e6, 1e6])

        result = surface_mean(data, area)

        # NaN is excluded: (10 + 30) / 2 = 20
        assert result == pytest.approx(20.0)

    def test_surface_mean_multidimensional(self):
        """Test surface mean with time dimension."""
        data = np.array([
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ])
        area = np.array([1e6, 1e6, 1e6])

        result = surface_mean(data, area)

        assert result.shape == (2,)
        assert result[0] == pytest.approx(20.0)  # (10+20+30)/3
        assert result[1] == pytest.approx(50.0)  # (40+50+60)/3

    def test_surface_mean_all_masked(self):
        """Test surface mean when all points are masked."""
        data = np.array([10.0, 20.0, 30.0])
        area = np.array([1e6, 1e6, 1e6])
        mask = np.array([False, False, False])

        result = surface_mean(data, area, mask=mask)

        # No valid cells -> NaN
        assert np.isnan(result)

    def test_surface_mean_all_nan(self):
        """Test surface mean when all values are NaN."""
        data = np.array([np.nan, np.nan, np.nan])
        area = np.array([1e6, 1e6, 1e6])

        result = surface_mean(data, area)

        # No valid cells -> NaN
        assert np.isnan(result)


class TestVolumeMean:
    """Tests for volume_mean function."""

    def test_volume_mean_basic(self):
        """Test basic volume-weighted mean."""
        # 2 levels, 3 points
        data = np.array([
            [10.0, 10.0, 10.0],  # Level 0
            [20.0, 20.0, 20.0],  # Level 1
        ])
        area = np.array([1e6, 1e6, 1e6])
        thickness = np.array([10.0, 10.0])  # Uniform thickness

        result = volume_mean(data, area, thickness)

        # All cells have same volume, so mean is (10 + 20) / 2 = 15
        assert result == pytest.approx(15.0)

    def test_volume_mean_varying_thickness(self):
        """Test volume mean with varying layer thickness."""
        data = np.array([
            [10.0, 10.0],  # Level 0
            [20.0, 20.0],  # Level 1
        ])
        area = np.array([1e6, 1e6])
        # Level 0 is thicker, so more weight on 10.0
        thickness = np.array([30.0, 10.0])

        result = volume_mean(data, area, thickness)

        # Total volume: 30*2e6 + 10*2e6 = 80e6
        # Weighted sum: 10*(30*2e6) + 20*(10*2e6) = 600e6 + 400e6 = 1000e6
        # Mean: 1000e6 / 80e6 = 12.5
        assert result == pytest.approx(12.5)

    def test_volume_mean_with_depth_range(self):
        """Test volume mean with depth range."""
        data = np.array([
            [10.0, 10.0],  # Level 0 at 50m
            [20.0, 20.0],  # Level 1 at 150m
            [30.0, 30.0],  # Level 2 at 300m
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0, 200.0])
        depth = np.array([50.0, 150.0, 300.0])

        # Only include levels in depth range 0-200m
        result = volume_mean(data, area, thickness, depth, depth_max=200.0)

        # Only first two levels: mean of 10 and 20 with equal volumes
        assert result == pytest.approx(15.0)

    def test_volume_mean_with_mask(self):
        """Test volume mean with spatial mask."""
        data = np.array([
            [10.0, 100.0],  # Level 0
            [20.0, 200.0],  # Level 1
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([10.0, 10.0])
        mask = np.array([True, False])  # Only first point

        result = volume_mean(data, area, thickness, mask=mask)

        # Only first column: (10 + 20) / 2 = 15
        assert result == pytest.approx(15.0)

    def test_volume_mean_with_nan(self):
        """Test volume mean with NaN values."""
        data = np.array([
            [10.0, np.nan],
            [20.0, 20.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([10.0, 10.0])

        result = volume_mean(data, area, thickness)

        # NaN excluded: (10*1e6*10 + 20*1e6*10 + 20*1e6*10) / (3*1e7) = 50/3
        expected = (10 + 20 + 20) / 3
        assert result == pytest.approx(expected)

    def test_volume_mean_multidimensional(self):
        """Test volume mean with time dimension."""
        # (2 times, 2 levels, 2 points)
        data = np.array([
            [[10.0, 10.0], [20.0, 20.0]],  # Time 0
            [[30.0, 30.0], [40.0, 40.0]],  # Time 1
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([10.0, 10.0])

        result = volume_mean(data, area, thickness)

        assert result.shape == (2,)
        assert result[0] == pytest.approx(15.0)  # (10+20)/2
        assert result[1] == pytest.approx(35.0)  # (30+40)/2

    def test_volume_mean_2d_thickness(self):
        """Test volume mean with 2D thickness array."""
        data = np.array([
            [10.0, 20.0],  # Level 0
            [30.0, 40.0],  # Level 1
        ])
        area = np.array([1e6, 2e6])
        # Varying thickness per cell
        thickness = np.array([
            [10.0, 20.0],  # Level 0 thickness
            [30.0, 40.0],  # Level 1 thickness
        ])

        result = volume_mean(data, area, thickness)

        # Volumes: 10*1e6, 20*2e6, 30*1e6, 40*2e6 = 10e6, 40e6, 30e6, 80e6
        # Total: 160e6
        # Weighted sum: 10*10e6 + 20*40e6 + 30*30e6 + 40*80e6 = 100e6 + 800e6 + 900e6 + 3200e6 = 5000e6
        # Mean: 5000e6 / 160e6 = 31.25
        assert result == pytest.approx(31.25)

    def test_volume_mean_requires_depth_for_range(self):
        """Test that depth is required when using depth_min/depth_max."""
        data = np.array([[10.0, 10.0], [20.0, 20.0]])
        area = np.array([1e6, 1e6])
        thickness = np.array([10.0, 10.0])

        with pytest.raises(ValueError, match="depth array required"):
            volume_mean(data, area, thickness, depth_max=100.0)

    def test_volume_mean_2d_area(self):
        """Test volume mean with 2D depth-dependent area."""
        data = np.array([
            [10.0, 20.0],  # Level 0
            [30.0, 40.0],  # Level 1
        ])
        # Area varies with depth (e.g., unstructured mesh)
        area = np.array([
            [1e6, 2e6],  # Level 0 areas
            [0.5e6, 1e6],  # Level 1 areas (smaller at depth)
        ])
        thickness = np.array([10.0, 10.0])

        result = volume_mean(data, area, thickness)

        # Volumes: 10*1e6, 10*2e6, 10*0.5e6, 10*1e6 = 10e6, 20e6, 5e6, 10e6
        # Total: 45e6
        # Weighted sum: 10*10e6 + 20*20e6 + 30*5e6 + 40*10e6 = 100e6 + 400e6 + 150e6 + 400e6 = 1050e6
        # Mean: 1050e6 / 45e6 = 23.333...
        expected = 1050 / 45
        assert result == pytest.approx(expected)

    def test_volume_mean_2d_area_extra_level(self):
        """Test volume mean with 2D area having one extra level (levels vs layers)."""
        import warnings

        data = np.array([
            [10.0, 20.0],  # Layer 0
            [30.0, 40.0],  # Layer 1
        ])
        # Area has 3 levels but data has 2 layers
        area = np.array([
            [1e6, 2e6],  # Level 0
            [1e6, 2e6],  # Level 1
            [0.5e6, 1e6],  # Level 2 (should be ignored)
        ])
        thickness = np.array([10.0, 10.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = volume_mean(data, area, thickness)
            assert len(w) == 1
            assert "one more vertical level" in str(w[0].message)

        # Should use only first 2 levels of area
        # Volumes: 10*1e6, 10*2e6, 10*1e6, 10*2e6
        # Total: 60e6
        # Weighted sum: 10*10e6 + 20*20e6 + 30*10e6 + 40*20e6 = 1600e6
        # Mean: 1600e6 / 60e6 = 26.666...
        expected = 1600 / 60
        assert result == pytest.approx(expected)


class TestHeatContent:
    """Tests for heat_content function."""

    def test_heat_content_basic(self):
        """Test basic heat content calculation."""
        # Uniform 10°C ocean
        temperature = np.array([
            [10.0, 10.0],  # Level 0
            [10.0, 10.0],  # Level 1
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness)

        # Volume: 2*2*100*1e6 = 4e8 m^3
        # Heat: rho * cp * T * V = 1025 * 3985 * 10 * 4e8
        expected = RHO_SEAWATER * CP_SEAWATER * 10.0 * 4e8
        assert result == pytest.approx(expected, rel=1e-6)

    def test_heat_content_with_reference(self):
        """Test heat content with reference temperature."""
        temperature = np.array([
            [15.0, 15.0],
            [15.0, 15.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness, reference_temp=10.0)

        # Heat content relative to 10°C
        # Volume: 4e8 m^3, T_anomaly: 5°C
        expected = RHO_SEAWATER * CP_SEAWATER * 5.0 * 4e8
        assert result == pytest.approx(expected, rel=1e-6)

    def test_heat_content_with_depth_range(self):
        """Test heat content in depth range."""
        temperature = np.array([
            [20.0, 20.0],  # 50m
            [15.0, 15.0],  # 150m
            [5.0, 5.0],    # 500m
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0, 400.0])
        depth = np.array([50.0, 150.0, 500.0])

        # Only 0-200m
        result = heat_content(
            temperature, area, thickness, depth,
            depth_max=200.0
        )

        # Volume: 2*100*1e6 + 2*100*1e6 = 4e8 m^3
        # Heat: rho*cp*(20*2e8 + 15*2e8) = rho*cp*70e8
        expected = RHO_SEAWATER * CP_SEAWATER * (20 * 2e8 + 15 * 2e8)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_heat_content_with_mask(self):
        """Test heat content with spatial mask."""
        temperature = np.array([
            [10.0, 100.0],  # Second point excluded
            [10.0, 100.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])
        mask = np.array([True, False])

        result = heat_content(temperature, area, thickness, mask=mask)

        # Only first column: volume = 2*100*1e6 = 2e8
        expected = RHO_SEAWATER * CP_SEAWATER * 10.0 * 2e8
        assert result == pytest.approx(expected, rel=1e-6)

    def test_heat_content_custom_constants(self):
        """Test heat content with custom rho and cp."""
        temperature = np.array([[10.0, 10.0]])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0])

        result = heat_content(
            temperature, area, thickness,
            rho=1000.0, cp=4000.0
        )

        # Volume: 2e8 m^3
        expected = 1000.0 * 4000.0 * 10.0 * 2e8
        assert result == pytest.approx(expected)

    def test_heat_content_multidimensional(self):
        """Test heat content with time dimension."""
        temperature = np.array([
            [[10.0, 10.0], [10.0, 10.0]],  # Time 0
            [[20.0, 20.0], [20.0, 20.0]],  # Time 1
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness)

        volume = 4e8  # 2 levels * 2 points * 100m * 1e6 m^2
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 10.0 * volume)
        assert result[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 20.0 * volume)

    def test_heat_content_with_nan(self):
        """Test heat content handles NaN values."""
        temperature = np.array([
            [10.0, np.nan],
            [10.0, 10.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness)

        # NaN excluded, 3 valid cells with volume 1e8 each
        expected = RHO_SEAWATER * CP_SEAWATER * 10.0 * 3e8
        assert result == pytest.approx(expected, rel=1e-6)

    def test_heat_content_2d_area(self):
        """Test heat content with 2D depth-dependent area."""
        temperature = np.array([
            [10.0, 10.0],  # Level 0
            [10.0, 10.0],  # Level 1
        ])
        # Area varies with depth
        area = np.array([
            [1e6, 2e6],  # Level 0 areas
            [0.5e6, 1e6],  # Level 1 areas (smaller at depth)
        ])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness)

        # Volumes: 100*1e6 + 100*2e6 + 100*0.5e6 + 100*1e6 = 4.5e8 m^3
        expected = RHO_SEAWATER * CP_SEAWATER * 10.0 * 4.5e8
        assert result == pytest.approx(expected, rel=1e-6)

    def test_heat_content_map_basic(self):
        """Test heat content map output (J/m² at each point)."""
        temperature = np.array([
            [10.0, 20.0],  # Level 0
            [10.0, 20.0],  # Level 1
        ])
        area = np.array([1e6, 1e6])  # Not used for map output
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness, output="map")

        # Heat content per unit area: rho * cp * sum_z(T * thickness)
        # Point 0: rho * cp * (10*100 + 10*100) = rho * cp * 2000
        # Point 1: rho * cp * (20*100 + 20*100) = rho * cp * 4000
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)
        assert result[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 4000.0)

    def test_heat_content_map_with_depth_range(self):
        """Test heat content map with depth range."""
        temperature = np.array([
            [20.0, 30.0],  # 50m
            [15.0, 25.0],  # 150m
            [5.0, 10.0],   # 500m (excluded)
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0, 400.0])
        depth = np.array([50.0, 150.0, 500.0])

        result = heat_content(
            temperature, area, thickness, depth,
            depth_max=200.0, output="map"
        )

        # Only first two levels included
        # Point 0: rho * cp * (20*100 + 15*100) = rho * cp * 3500
        # Point 1: rho * cp * (30*100 + 25*100) = rho * cp * 5500
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 3500.0)
        assert result[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 5500.0)

    def test_heat_content_map_with_mask(self):
        """Test heat content map with spatial mask."""
        temperature = np.array([
            [10.0, 20.0],
            [10.0, 20.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])
        mask = np.array([True, False])

        result = heat_content(temperature, area, thickness, mask=mask, output="map")

        # Point 0: rho * cp * 2000 * 1 (mask=True)
        # Point 1: rho * cp * 4000 * 0 (mask=False)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)
        assert result[1] == pytest.approx(0.0)

    def test_heat_content_map_with_reference(self):
        """Test heat content map with reference temperature."""
        temperature = np.array([
            [15.0, 25.0],
            [15.0, 25.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(
            temperature, area, thickness,
            reference_temp=10.0, output="map"
        )

        # Anomaly: T - 10
        # Point 0: rho * cp * (5*100 + 5*100) = rho * cp * 1000
        # Point 1: rho * cp * (15*100 + 15*100) = rho * cp * 3000
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 1000.0)
        assert result[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 3000.0)

    def test_heat_content_map_multidimensional(self):
        """Test heat content map with time dimension."""
        temperature = np.array([
            [[10.0, 20.0], [10.0, 20.0]],  # Time 0
            [[30.0, 40.0], [30.0, 40.0]],  # Time 1
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness, output="map")

        # Shape: (time, npoints)
        assert result.shape == (2, 2)
        # Time 0, Point 0: rho * cp * 2000
        # Time 0, Point 1: rho * cp * 4000
        # Time 1, Point 0: rho * cp * 6000
        # Time 1, Point 1: rho * cp * 8000
        assert result[0, 0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)
        assert result[0, 1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 4000.0)
        assert result[1, 0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 6000.0)
        assert result[1, 1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 8000.0)

    def test_heat_content_map_with_nan(self):
        """Test heat content map handles NaN values."""
        temperature = np.array([
            [10.0, np.nan],
            [10.0, 20.0],
        ])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0, 100.0])

        result = heat_content(temperature, area, thickness, output="map")

        # Point 0: rho * cp * (10*100 + 10*100) = rho * cp * 2000
        # Point 1: NaN excluded, only second level: rho * cp * (20*100) = rho * cp * 2000
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)
        assert result[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)

    def test_heat_content_map_2d_thickness(self):
        """Test heat content map with 2D thickness array."""
        temperature = np.array([
            [10.0, 20.0],
            [10.0, 20.0],
        ])
        area = np.array([1e6, 1e6])
        # Varying thickness per cell
        thickness = np.array([
            [100.0, 200.0],  # Level 0
            [50.0, 100.0],   # Level 1
        ])

        result = heat_content(temperature, area, thickness, output="map")

        # Point 0: rho * cp * (10*100 + 10*50) = rho * cp * 1500
        # Point 1: rho * cp * (20*200 + 20*100) = rho * cp * 6000
        assert result.shape == (2,)
        assert result[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 1500.0)
        assert result[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 6000.0)

    def test_heat_content_invalid_output(self):
        """Test that invalid output parameter raises error."""
        temperature = np.array([[10.0, 10.0]])
        area = np.array([1e6, 1e6])
        thickness = np.array([100.0])

        with pytest.raises(ValueError, match="output must be 'total' or 'map'"):
            heat_content(temperature, area, thickness, output="invalid")


class TestDaskCompatibility:
    """Tests for dask array compatibility."""

    @pytest.fixture
    def dask_deps(self):
        """Import dask and xarray, skip if not available."""
        da = pytest.importorskip("dask.array")
        xr = pytest.importorskip("xarray")
        return da, xr

    def test_surface_mean_dask_basic(self, dask_deps):
        """Test surface_mean with dask array."""
        da, xr = dask_deps

        data_np = np.array([10.0, 20.0, 30.0])
        area_np = np.array([1e6, 1e6, 1e6])

        data = xr.DataArray(
            da.from_array(data_np, chunks=2),
            dims=["npoints"],
        )

        result = surface_mean(data, area_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(20.0)

    def test_surface_mean_dask_multidimensional(self, dask_deps):
        """Test surface_mean with multidimensional dask array."""
        da, xr = dask_deps

        data_np = np.array([
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ])
        area_np = np.array([1e6, 1e6, 1e6])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(1, 3)),
            dims=["time", "npoints"],
        )

        result = surface_mean(data, area_np)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed[0] == pytest.approx(20.0)
        assert computed[1] == pytest.approx(50.0)

    def test_surface_mean_dask_with_mask(self, dask_deps):
        """Test surface_mean with dask array and mask."""
        da, xr = dask_deps

        data_np = np.array([10.0, 20.0, 30.0, 40.0])
        area_np = np.array([1e6, 1e6, 1e6, 1e6])
        mask_np = np.array([True, True, False, False])

        data = xr.DataArray(
            da.from_array(data_np, chunks=2),
            dims=["npoints"],
        )

        result = surface_mean(data, area_np, mask=mask_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(15.0)

    def test_volume_mean_dask_basic(self, dask_deps):
        """Test volume_mean with dask array."""
        da, xr = dask_deps

        data_np = np.array([
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ])
        area_np = np.array([1e6, 1e6, 1e6])
        thickness_np = np.array([10.0, 10.0])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(1, 3)),
            dims=["level", "npoints"],
        )

        result = volume_mean(data, area_np, thickness_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(15.0)

    def test_volume_mean_dask_multidimensional(self, dask_deps):
        """Test volume_mean with multidimensional dask array."""
        da, xr = dask_deps

        data_np = np.array([
            [[10.0, 10.0], [20.0, 20.0]],
            [[30.0, 30.0], [40.0, 40.0]],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([10.0, 10.0])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(1, 2, 2)),
            dims=["time", "level", "npoints"],
        )

        result = volume_mean(data, area_np, thickness_np)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed[0] == pytest.approx(15.0)
        assert computed[1] == pytest.approx(35.0)

    def test_volume_mean_dask_with_depth_range(self, dask_deps):
        """Test volume_mean with dask array and depth range."""
        da, xr = dask_deps

        data_np = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([100.0, 100.0, 200.0])
        depth_np = np.array([50.0, 150.0, 300.0])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(1, 2)),
            dims=["level", "npoints"],
        )

        result = volume_mean(data, area_np, thickness_np, depth_np, depth_max=200.0)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(15.0)

    def test_volume_mean_dask_with_mask(self, dask_deps):
        """Test volume_mean with dask array and mask."""
        da, xr = dask_deps

        data_np = np.array([
            [10.0, 100.0],
            [20.0, 200.0],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([10.0, 10.0])
        mask_np = np.array([True, False])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(1, 2)),
            dims=["level", "npoints"],
        )

        result = volume_mean(data, area_np, thickness_np, mask=mask_np)

        assert is_dask_array(result)
        assert float(result.compute()) == pytest.approx(15.0)

    def test_heat_content_dask_basic(self, dask_deps):
        """Test heat_content with dask array."""
        da, xr = dask_deps

        temp_np = np.array([
            [10.0, 10.0],
            [10.0, 10.0],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([100.0, 100.0])

        temperature = xr.DataArray(
            da.from_array(temp_np, chunks=(1, 2)),
            dims=["level", "npoints"],
        )

        result = heat_content(temperature, area_np, thickness_np)

        assert is_dask_array(result)
        expected = RHO_SEAWATER * CP_SEAWATER * 10.0 * 4e8
        assert float(result.compute()) == pytest.approx(expected, rel=1e-6)

    def test_heat_content_dask_multidimensional(self, dask_deps):
        """Test heat_content with multidimensional dask array."""
        da, xr = dask_deps

        temp_np = np.array([
            [[10.0, 10.0], [10.0, 10.0]],
            [[20.0, 20.0], [20.0, 20.0]],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([100.0, 100.0])

        temperature = xr.DataArray(
            da.from_array(temp_np, chunks=(1, 2, 2)),
            dims=["time", "level", "npoints"],
        )

        result = heat_content(temperature, area_np, thickness_np)

        assert is_dask_array(result)
        computed = result.compute()
        volume = 4e8
        assert computed[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 10.0 * volume)
        assert computed[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 20.0 * volume)

    def test_heat_content_dask_with_reference(self, dask_deps):
        """Test heat_content with dask array and reference temperature."""
        da, xr = dask_deps

        temp_np = np.array([
            [15.0, 15.0],
            [15.0, 15.0],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([100.0, 100.0])

        temperature = xr.DataArray(
            da.from_array(temp_np, chunks=(1, 2)),
            dims=["level", "npoints"],
        )

        result = heat_content(temperature, area_np, thickness_np, reference_temp=10.0)

        assert is_dask_array(result)
        expected = RHO_SEAWATER * CP_SEAWATER * 5.0 * 4e8
        assert float(result.compute()) == pytest.approx(expected, rel=1e-6)

    def test_numpy_input_returns_eager(self, dask_deps):
        """Test that numpy input still returns eager (non-dask) result."""
        data_np = np.array([
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ])
        area_np = np.array([1e6, 1e6, 1e6])
        thickness_np = np.array([10.0, 10.0])

        result = volume_mean(data_np, area_np, thickness_np)

        assert not is_dask_array(result)
        assert isinstance(result, float)
        assert result == pytest.approx(15.0)

    def test_heat_content_map_dask_basic(self, dask_deps):
        """Test heat_content map output with dask array."""
        da, xr = dask_deps

        temp_np = np.array([
            [10.0, 20.0],
            [10.0, 20.0],
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([100.0, 100.0])

        temperature = xr.DataArray(
            da.from_array(temp_np, chunks=(1, 2)),
            dims=["level", "npoints"],
        )

        result = heat_content(temperature, area_np, thickness_np, output="map")

        assert is_dask_array(result)
        computed = result.compute()
        assert computed.shape == (2,)
        assert computed[0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)
        assert computed[1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 4000.0)

    def test_heat_content_map_dask_multidimensional(self, dask_deps):
        """Test heat_content map with multidimensional dask array."""
        da, xr = dask_deps

        temp_np = np.array([
            [[10.0, 20.0], [10.0, 20.0]],  # Time 0
            [[30.0, 40.0], [30.0, 40.0]],  # Time 1
        ])
        area_np = np.array([1e6, 1e6])
        thickness_np = np.array([100.0, 100.0])

        temperature = xr.DataArray(
            da.from_array(temp_np, chunks=(1, 2, 2)),
            dims=["time", "level", "npoints"],
        )

        result = heat_content(temperature, area_np, thickness_np, output="map")

        assert is_dask_array(result)
        computed = result.compute()
        assert computed.shape == (2, 2)
        assert computed[0, 0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 2000.0)
        assert computed[0, 1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 4000.0)
        assert computed[1, 0] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 6000.0)
        assert computed[1, 1] == pytest.approx(RHO_SEAWATER * CP_SEAWATER * 8000.0)


class TestFindClosestDepth:
    """Tests for find_closest_depth function."""

    def test_find_closest_depth_exact_match(self):
        """Test finding exact depth match."""
        depths = np.array([0, 10, 25, 50, 100, 200, 500, 1000])
        idx, val = find_closest_depth(depths, 100)
        assert idx == 4
        assert val == 100.0

    def test_find_closest_depth_between_levels(self):
        """Test finding closest depth when target is between levels."""
        depths = np.array([0, 10, 25, 50, 100, 200, 500, 1000])
        idx, val = find_closest_depth(depths, 75)
        # 75 is closer to 50 (diff=25) than to 100 (diff=25), argmin returns first
        # Actually 75-50=25, 100-75=25, so it's a tie. np.argmin returns first.
        assert idx == 3 or idx == 4  # Either 50 or 100
        assert val in [50.0, 100.0]

    def test_find_closest_depth_closer_to_lower(self):
        """Test finding closest when closer to lower level."""
        depths = np.array([0, 10, 25, 50, 100, 200, 500, 1000])
        idx, val = find_closest_depth(depths, 60)
        # 60 is closer to 50 (diff=10) than to 100 (diff=40)
        assert idx == 3
        assert val == 50.0

    def test_find_closest_depth_closer_to_upper(self):
        """Test finding closest when closer to upper level."""
        depths = np.array([0, 10, 25, 50, 100, 200, 500, 1000])
        idx, val = find_closest_depth(depths, 90)
        # 90 is closer to 100 (diff=10) than to 50 (diff=40)
        assert idx == 4
        assert val == 100.0

    def test_find_closest_depth_below_minimum(self):
        """Test finding closest when target is below minimum depth."""
        depths = np.array([10, 25, 50, 100])
        idx, val = find_closest_depth(depths, 5)
        assert idx == 0
        assert val == 10.0

    def test_find_closest_depth_above_maximum(self):
        """Test finding closest when target is above maximum depth."""
        depths = np.array([10, 25, 50, 100])
        idx, val = find_closest_depth(depths, 500)
        assert idx == 3
        assert val == 100.0

    def test_find_closest_depth_list_input(self):
        """Test with list input."""
        depths = [0, 10, 25, 50, 100]
        idx, val = find_closest_depth(depths, 30)
        assert idx == 2
        assert val == 25.0

    def test_find_closest_depth_single_depth(self):
        """Test with single depth level."""
        depths = np.array([50.0])
        idx, val = find_closest_depth(depths, 100)
        assert idx == 0
        assert val == 50.0


class TestInterpolateToDepth:
    """Tests for interpolate_to_depth function."""

    def test_interpolate_to_depth_exact_level(self):
        """Test interpolation to exact depth level."""
        # 3 levels, 2 points
        data = np.array([
            [10.0, 20.0],  # 0m
            [15.0, 25.0],  # 50m
            [20.0, 30.0],  # 100m
        ])
        depths = np.array([0, 50, 100])

        result = interpolate_to_depth(data, None, None, depths, 50)

        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(15.0)
        assert result[0, 1] == pytest.approx(25.0)

    def test_interpolate_to_depth_between_levels(self):
        """Test linear interpolation between levels."""
        data = np.array([
            [10.0, 20.0],  # 0m
            [20.0, 40.0],  # 100m
        ])
        depths = np.array([0, 100])

        result = interpolate_to_depth(data, None, None, depths, 50)

        # Linear interpolation: 10 + 0.5*(20-10) = 15, 20 + 0.5*(40-20) = 30
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(15.0)
        assert result[0, 1] == pytest.approx(30.0)

    def test_interpolate_to_depth_multiple_targets(self):
        """Test interpolation to multiple target depths."""
        data = np.array([
            [10.0, 20.0],  # 0m
            [20.0, 40.0],  # 100m
        ])
        depths = np.array([0, 100])

        result = interpolate_to_depth(data, None, None, depths, [25, 50, 75])

        assert result.shape == (3, 2)
        # At 25m: 10 + 0.25*10 = 12.5, 20 + 0.25*20 = 25
        assert result[0, 0] == pytest.approx(12.5)
        assert result[0, 1] == pytest.approx(25.0)
        # At 50m: 15, 30
        assert result[1, 0] == pytest.approx(15.0)
        assert result[1, 1] == pytest.approx(30.0)
        # At 75m: 10 + 0.75*10 = 17.5, 20 + 0.75*20 = 35
        assert result[2, 0] == pytest.approx(17.5)
        assert result[2, 1] == pytest.approx(35.0)

    def test_interpolate_to_depth_with_coordinates(self):
        """Test interpolation returns coordinates when provided."""
        data = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
        ])
        depths = np.array([0, 100])
        lon = np.array([-30.0, -20.0])
        lat = np.array([60.0, 65.0])

        result, lon_out, lat_out = interpolate_to_depth(data, lon, lat, depths, 50)

        assert result.shape == (1, 2)
        np.testing.assert_array_equal(lon_out, lon)
        np.testing.assert_array_equal(lat_out, lat)

    def test_interpolate_to_depth_time_dimension(self):
        """Test interpolation with time dimension."""
        # (2 times, 3 levels, 2 points)
        data = np.array([
            [[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]],  # Time 0
            [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]],  # Time 1
        ])
        depths = np.array([0, 50, 100])

        result = interpolate_to_depth(data, None, None, depths, [25, 75])

        assert result.shape == (2, 2, 2)  # (time, targets, points)
        # Time 0, depth 25: 12.5, 22.5
        assert result[0, 0, 0] == pytest.approx(12.5)
        assert result[0, 0, 1] == pytest.approx(22.5)

    def test_interpolate_to_depth_extrapolation_warning(self):
        """Test that extrapolation generates warning."""
        import warnings

        data = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
        ])
        depths = np.array([50, 100])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = interpolate_to_depth(data, None, None, depths, 25)
            assert len(w) == 1
            assert "Extrapolation" in str(w[0].message)

    def test_interpolate_to_depth_decreasing_depths(self):
        """Test interpolation with decreasing depth order."""
        data = np.array([
            [20.0, 30.0],  # 100m
            [15.0, 25.0],  # 50m
            [10.0, 20.0],  # 0m
        ])
        depths = np.array([100, 50, 0])  # Decreasing

        result = interpolate_to_depth(data, None, None, depths, 25)

        # At 25m (between 0 and 50): linear interp
        # 10 + 0.5*5 = 12.5, 20 + 0.5*5 = 22.5
        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(12.5)
        assert result[0, 1] == pytest.approx(22.5)

    def test_interpolate_to_depth_single_target_scalar(self):
        """Test interpolation with scalar target depth."""
        data = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
        ])
        depths = np.array([0, 100])

        result = interpolate_to_depth(data, None, None, depths, 50.0)

        assert result.shape == (1, 2)

    def test_interpolate_to_depth_shape_validation(self):
        """Test that mismatched depths raise error."""
        data = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
        ])
        depths = np.array([0, 50, 100])  # 3 levels but data has 2

        with pytest.raises(ValueError, match="levels"):
            interpolate_to_depth(data, None, None, depths, 50)


class TestInterpolateToDepthDask:
    """Tests for interpolate_to_depth with dask arrays."""

    @pytest.fixture
    def dask_deps(self):
        """Import dask and xarray, skip if not available."""
        da = pytest.importorskip("dask.array")
        xr = pytest.importorskip("xarray")
        return da, xr

    def test_interpolate_to_depth_dask_basic(self, dask_deps):
        """Test interpolation with dask array."""
        da, xr = dask_deps

        data_np = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 40.0, 60.0],
        ])
        depths = np.array([0, 100])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(2, 2)),
            dims=["level", "npoints"],
        )

        result = interpolate_to_depth(data, None, None, depths, 50)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed.shape == (1, 3)
        assert computed[0, 0] == pytest.approx(15.0)
        assert computed[0, 1] == pytest.approx(30.0)
        assert computed[0, 2] == pytest.approx(45.0)

    def test_interpolate_to_depth_dask_multiple_targets(self, dask_deps):
        """Test dask interpolation to multiple depths."""
        da, xr = dask_deps

        data_np = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
        ])
        depths = np.array([0, 100])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(2, 1)),
            dims=["level", "npoints"],
        )

        result = interpolate_to_depth(data, None, None, depths, [25, 50, 75])

        assert is_dask_array(result)
        computed = result.compute()
        assert computed.shape == (3, 2)
        assert computed[0, 0] == pytest.approx(12.5)
        assert computed[1, 0] == pytest.approx(15.0)
        assert computed[2, 0] == pytest.approx(17.5)

    def test_interpolate_to_depth_dask_with_time(self, dask_deps):
        """Test dask interpolation with time dimension."""
        da, xr = dask_deps

        data_np = np.array([
            [[10.0, 20.0], [20.0, 40.0]],
            [[100.0, 200.0], [200.0, 400.0]],
        ])
        depths = np.array([0, 100])

        data = xr.DataArray(
            da.from_array(data_np, chunks=(1, 2, 2)),
            dims=["time", "level", "npoints"],
        )

        result = interpolate_to_depth(data, None, None, depths, 50)

        assert is_dask_array(result)
        computed = result.compute()
        assert computed.shape == (2, 1, 2)
        assert computed[0, 0, 0] == pytest.approx(15.0)
        assert computed[1, 0, 0] == pytest.approx(150.0)
