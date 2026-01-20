"""Tests for Hovmoller diagram functions."""

import numpy as np
import pytest

from nereus.diag.hovmoller import hovmoller, plot_hovmoller


class TestHovmoller:
    """Tests for hovmoller function."""

    def test_hovmoller_depth_mode_basic(self):
        """Test basic time-depth Hovmoller computation."""
        # 3 times, 2 levels, 4 points
        data = np.array([
            [[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]],  # t=0
            [[11.0, 11.0, 11.0, 11.0], [21.0, 21.0, 21.0, 21.0]],  # t=1
            [[12.0, 12.0, 12.0, 12.0], [22.0, 22.0, 22.0, 22.0]],  # t=2
        ])
        area = np.array([1e6, 1e6, 1e6, 1e6])
        depth = np.array([50.0, 150.0])

        time_out, depth_out, result = hovmoller(
            data, area, depth=depth, mode="depth"
        )

        assert time_out.shape == (3,)
        assert depth_out.shape == (2,)
        assert result.shape == (3, 2)

        # Since all points have same value, mean equals that value
        assert result[0, 0] == pytest.approx(10.0)
        assert result[0, 1] == pytest.approx(20.0)
        assert result[1, 0] == pytest.approx(11.0)
        assert result[2, 1] == pytest.approx(22.0)

    def test_hovmoller_depth_mode_with_time(self):
        """Test Hovmoller with explicit time array."""
        data = np.array([
            [[10.0, 10.0], [20.0, 20.0]],
            [[11.0, 11.0], [21.0, 21.0]],
        ])
        area = np.array([1e6, 1e6])
        depth = np.array([50.0, 150.0])
        time = np.array([0.0, 1.0])

        time_out, depth_out, result = hovmoller(
            data, area, time=time, depth=depth, mode="depth"
        )

        np.testing.assert_array_equal(time_out, time)
        np.testing.assert_array_equal(depth_out, depth)

    def test_hovmoller_depth_mode_varying_values(self):
        """Test Hovmoller with spatially varying values."""
        # Area-weighted mean should favor larger cells
        data = np.array([
            [[10.0, 30.0], [20.0, 40.0]],  # t=0
        ])
        area = np.array([3e6, 1e6])  # First point has 3x area
        depth = np.array([50.0, 150.0])

        _, _, result = hovmoller(data, area, depth=depth, mode="depth")

        # Level 0: (10*3 + 30*1) / 4 = 15
        # Level 1: (20*3 + 40*1) / 4 = 25
        assert result[0, 0] == pytest.approx(15.0)
        assert result[0, 1] == pytest.approx(25.0)

    def test_hovmoller_depth_mode_with_nan(self):
        """Test Hovmoller handles NaN in depth mode."""
        data = np.array([
            [[10.0, np.nan], [20.0, 20.0]],
        ])
        area = np.array([1e6, 1e6])
        depth = np.array([50.0, 150.0])

        _, _, result = hovmoller(data, area, depth=depth, mode="depth")

        # Level 0: only first point is valid
        assert result[0, 0] == pytest.approx(10.0)
        # Level 1: mean of 20 and 20
        assert result[0, 1] == pytest.approx(20.0)

    def test_hovmoller_depth_mode_with_mask(self):
        """Test Hovmoller with spatial mask."""
        data = np.array([
            [[10.0, 100.0], [20.0, 200.0]],
        ])
        area = np.array([1e6, 1e6])
        depth = np.array([50.0, 150.0])
        mask = np.array([True, False])

        _, _, result = hovmoller(data, area, depth=depth, mask=mask, mode="depth")

        assert result[0, 0] == pytest.approx(10.0)
        assert result[0, 1] == pytest.approx(20.0)

    def test_hovmoller_latitude_mode_basic(self):
        """Test basic time-latitude Hovmoller computation."""
        # Create data with known latitudes
        # 2 times, 8 points covering -80 to +80 latitude
        lats = np.array([-75, -45, -15, 15, 45, 75, -60, 30])
        data = np.array([
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 15.0, 45.0],  # t=0
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 16.0, 46.0],  # t=1
        ])
        area = np.ones(8) * 1e6

        time_out, lat_out, result = hovmoller(
            data, area, lat=lats, mode="latitude"
        )

        assert time_out.shape == (2,)
        assert len(lat_out) == 36  # Default 5-degree bins from -90 to 90
        assert result.shape == (2, 36)

    def test_hovmoller_latitude_mode_custom_bins(self):
        """Test latitude Hovmoller with custom bins."""
        lats = np.array([-60, -30, 0, 30, 60])
        data = np.array([
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        area = np.ones(5) * 1e6
        lat_bins = np.array([-90, -45, 0, 45, 90])

        _, lat_out, result = hovmoller(
            data, area, lat=lats, lat_bins=lat_bins, mode="latitude"
        )

        assert len(lat_out) == 4  # 4 bins
        # Bin centers: -67.5, -22.5, 22.5, 67.5
        np.testing.assert_array_almost_equal(lat_out, [-67.5, -22.5, 22.5, 67.5])

    def test_hovmoller_latitude_mode_2d_input(self):
        """Test latitude Hovmoller with 2D input (ntime, npoints)."""
        lats = np.array([-60, 60])
        data = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ])
        area = np.ones(2) * 1e6

        time_out, _, result = hovmoller(data, area, lat=lats, mode="latitude")

        assert time_out.shape == (2,)
        assert result.shape[0] == 2

    def test_hovmoller_requires_depth_for_depth_mode(self):
        """Test that depth is required for mode='depth'."""
        data = np.array([[[10.0, 10.0], [20.0, 20.0]]])
        area = np.array([1e6, 1e6])

        with pytest.raises(ValueError, match="depth array required"):
            hovmoller(data, area, mode="depth")

    def test_hovmoller_requires_lat_for_latitude_mode(self):
        """Test that lat is required for mode='latitude'."""
        data = np.array([[10.0, 10.0]])
        area = np.array([1e6, 1e6])

        with pytest.raises(ValueError, match="lat array required"):
            hovmoller(data, area, mode="latitude")

    def test_hovmoller_invalid_mode(self):
        """Test error for invalid mode."""
        data = np.array([[10.0, 10.0]])
        area = np.array([1e6, 1e6])

        with pytest.raises(ValueError, match="Invalid mode"):
            hovmoller(data, area, mode="invalid")

    def test_hovmoller_single_timestep(self):
        """Test Hovmoller with single timestep (2D input for depth mode)."""
        data = np.array([
            [10.0, 10.0],  # Level 0
            [20.0, 20.0],  # Level 1
        ])
        area = np.array([1e6, 1e6])
        depth = np.array([50.0, 150.0])

        time_out, _, result = hovmoller(data, area, depth=depth, mode="depth")

        assert time_out.shape == (1,)
        assert result.shape == (1, 2)


class TestPlotHovmoller:
    """Tests for plot_hovmoller function."""

    def test_plot_hovmoller_basic(self):
        """Test basic Hovmoller plotting."""
        time = np.arange(10)
        y = np.array([0, 100, 200, 500, 1000])
        data = np.random.rand(10, 5)

        fig, ax = plot_hovmoller(time, y, data)

        assert fig is not None
        assert ax is not None

        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_hovmoller_depth_mode(self):
        """Test Hovmoller plot in depth mode."""
        time = np.arange(5)
        depth = np.array([0, 100, 500])
        data = np.random.rand(5, 3)

        fig, ax = plot_hovmoller(time, depth, data, mode="depth")

        # Y-axis should be inverted for depth
        assert ax.yaxis_inverted()

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_hovmoller_latitude_mode(self):
        """Test Hovmoller plot in latitude mode."""
        time = np.arange(5)
        lat = np.array([-60, -30, 0, 30, 60])
        data = np.random.rand(5, 5)

        fig, ax = plot_hovmoller(time, lat, data, mode="latitude")

        # Y-axis should not be inverted for latitude
        assert not ax.yaxis_inverted()

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_hovmoller_with_options(self):
        """Test Hovmoller plot with various options."""
        time = np.arange(5)
        y = np.array([0, 100, 500])
        data = np.random.rand(5, 3)

        fig, ax = plot_hovmoller(
            time, y, data,
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=True,
            colorbar_label="Temperature (°C)",
            title="Test Hovmoller",
        )

        assert ax.get_title() == "Test Hovmoller"

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_hovmoller_no_colorbar(self):
        """Test Hovmoller plot without colorbar."""
        time = np.arange(5)
        y = np.array([0, 100, 500])
        data = np.random.rand(5, 3)

        fig, ax = plot_hovmoller(time, y, data, colorbar=False)

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_hovmoller_explicit_invert(self):
        """Test explicit y-axis inversion control."""
        time = np.arange(5)
        y = np.array([0, 100, 500])
        data = np.random.rand(5, 3)

        # Force no inversion in depth mode
        fig, ax = plot_hovmoller(time, y, data, mode="depth", invert_y=False)
        assert not ax.yaxis_inverted()

        import matplotlib.pyplot as plt
        plt.close(fig)

        # Force inversion in latitude mode
        fig, ax = plot_hovmoller(time, y, data, mode="latitude", invert_y=True)
        assert ax.yaxis_inverted()

        plt.close(fig)

    def test_plot_hovmoller_existing_ax(self):
        """Test plotting on existing axes."""
        import matplotlib.pyplot as plt

        fig_orig, ax_orig = plt.subplots()

        time = np.arange(5)
        y = np.array([0, 100, 500])
        data = np.random.rand(5, 3)

        fig, ax = plot_hovmoller(time, y, data, ax=ax_orig)

        assert fig is fig_orig
        assert ax is ax_orig

        plt.close(fig)
