"""Tests for vertical/ocean diagnostics."""

import numpy as np
import pytest

from nereus.core.types import is_dask_array
from nereus.diag.vertical import RHO_SEAWATER, CP_SEAWATER, heat_content, volume_mean


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
