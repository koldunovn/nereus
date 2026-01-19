"""Tests for nereus.plotting.maps."""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nereus.plotting.maps import plot, plot_polar
from nereus.regrid.interpolator import RegridInterpolator


class TestPlot:
    """Tests for plot function."""

    def test_basic_plot(self, random_mesh_small, synthetic_data):
        """Test basic plotting."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, interp = plot(data, lon, lat, resolution=5.0)

        assert fig is not None
        assert ax is not None
        assert isinstance(interp, RegridInterpolator)

        plt.close(fig)

    def test_different_projections(self, random_mesh_small, synthetic_data):
        """Test plotting with different projections."""
        lon, lat = random_mesh_small
        data = synthetic_data

        projections = ["pc", "rob", "merc", "npstere", "spstere"]

        for proj in projections:
            fig, ax, interp = plot(data, lon, lat, projection=proj, resolution=10.0)
            plt.close(fig)

    def test_reuse_interpolator(self, random_mesh_small, synthetic_data):
        """Test reusing interpolator for multiple plots."""
        lon, lat = random_mesh_small
        data1 = synthetic_data
        data2 = np.random.rand(len(lon))

        fig1, ax1, interp = plot(data1, lon, lat, resolution=5.0)
        fig2, ax2, interp2 = plot(data2, lon, lat, interpolator=interp)

        # Should be the same interpolator
        assert interp is interp2

        plt.close(fig1)
        plt.close(fig2)

    def test_colorbar(self, random_mesh_small, synthetic_data):
        """Test colorbar options."""
        lon, lat = random_mesh_small
        data = synthetic_data

        # With colorbar
        fig1, ax1, _ = plot(data, lon, lat, colorbar=True, resolution=10.0)
        plt.close(fig1)

        # Without colorbar
        fig2, ax2, _ = plot(data, lon, lat, colorbar=False, resolution=10.0)
        plt.close(fig2)

    def test_colorbar_label(self, random_mesh_small, synthetic_data):
        """Test colorbar label."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, _ = plot(
            data, lon, lat,
            colorbar=True,
            colorbar_label="Temperature (°C)",
            resolution=10.0
        )
        plt.close(fig)

    def test_title(self, random_mesh_small, synthetic_data):
        """Test plot title."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, _ = plot(data, lon, lat, title="Test Title", resolution=10.0)

        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_colormap(self, random_mesh_small, synthetic_data):
        """Test different colormaps."""
        lon, lat = random_mesh_small
        data = synthetic_data

        for cmap in ["viridis", "coolwarm", "RdBu_r"]:
            fig, ax, _ = plot(data, lon, lat, cmap=cmap, resolution=10.0)
            plt.close(fig)

    def test_vmin_vmax(self, random_mesh_small, synthetic_data):
        """Test color scale limits."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, _ = plot(data, lon, lat, vmin=-0.5, vmax=0.5, resolution=10.0)
        plt.close(fig)

    def test_map_features(self, random_mesh_small, synthetic_data):
        """Test map feature options."""
        lon, lat = random_mesh_small
        data = synthetic_data

        # With all features
        fig1, ax1, _ = plot(
            data, lon, lat,
            coastlines=True,
            land=True,
            gridlines=True,
            resolution=10.0
        )
        plt.close(fig1)

        # Without features
        fig2, ax2, _ = plot(
            data, lon, lat,
            coastlines=False,
            land=False,
            gridlines=False,
            resolution=10.0
        )
        plt.close(fig2)

    def test_existing_axes(self, random_mesh_small, synthetic_data):
        """Test plotting on existing axes."""
        import cartopy.crs as ccrs

        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        fig2, ax2, _ = plot(data, lon, lat, ax=ax, resolution=10.0)

        # Should use provided axes
        assert ax2 is ax
        assert fig2 is fig

        plt.close(fig)

    def test_custom_figsize(self, random_mesh_small, synthetic_data):
        """Test custom figure size."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, _ = plot(data, lon, lat, figsize=(15, 8), resolution=10.0)

        np.testing.assert_allclose(fig.get_size_inches(), [15, 8])
        plt.close(fig)


class TestPlotPolar:
    """Tests for plot_polar function."""

    def test_north_polar(self, random_mesh_small, synthetic_data):
        """Test north polar plot."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, interp = plot_polar(
            data, lon, lat,
            hemisphere="north",
            lat_limit=50,
            resolution=5.0
        )

        plt.close(fig)

    def test_south_polar(self, random_mesh_small, synthetic_data):
        """Test south polar plot."""
        lon, lat = random_mesh_small
        data = synthetic_data

        fig, ax, interp = plot_polar(
            data, lon, lat,
            hemisphere="south",
            lat_limit=50,
            resolution=5.0
        )

        plt.close(fig)
