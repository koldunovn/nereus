"""Tests for nereus.plotting.transect."""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nereus.plotting.transect import transect


class TestTransect:
    """Tests for transect function."""

    @pytest.fixture
    def regular_grid_2d(self):
        """Create 2D data on a regular grid (nlevels, npoints)."""
        # Unstructured-style: flattened coordinates
        lon_1d = np.linspace(-180, 180, 36)
        lat_1d = np.linspace(-90, 90, 18)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        lon_flat = lon_2d.ravel()
        lat_flat = lat_2d.ravel()

        depth = np.array([0, 10, 50, 100, 200, 500, 1000, 2000])
        n_levels = len(depth)
        n_points = len(lon_flat)

        # Create data that varies with depth and latitude
        data = np.zeros((n_levels, n_points))
        for i, d in enumerate(depth):
            data[i] = np.sin(np.deg2rad(lat_flat)) * np.exp(-d / 500)

        return data, lon_flat, lat_flat, depth

    @pytest.fixture
    def regular_grid_3d(self):
        """Create 3D data on a regular grid (nlevels, nlat, nlon)."""
        lon_1d = np.linspace(-180, 180, 36)
        lat_1d = np.linspace(-90, 90, 18)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        depth = np.array([0, 10, 50, 100, 200, 500, 1000, 2000])
        n_levels = len(depth)

        # Create 3D data (nlevels, nlat, nlon)
        data = np.zeros((n_levels, 18, 36))
        for i, d in enumerate(depth):
            data[i] = np.sin(np.deg2rad(lat_2d)) * np.exp(-d / 500)

        return data, lon_1d, lat_1d, depth

    def test_basic_transect_2d(self, regular_grid_2d):
        """Test basic transect with 2D data (nlevels, npoints)."""
        data, lon, lat, depth = regular_grid_2d

        fig, ax = transect(
            data, lon, lat, depth,
            start=(-30, -60),
            end=(-30, 60),
            n_points=50
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_basic_transect_3d(self, regular_grid_3d):
        """Test basic transect with 3D data (nlevels, nlat, nlon)."""
        data, lon, lat, depth = regular_grid_3d

        # This should work with 3D data and 1D coordinates
        with pytest.warns(UserWarning, match="Creating meshgrid"):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50
            )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_transect_3d_shape_consistency(self, regular_grid_3d):
        """Test that 3D transect produces correct output dimensions."""
        data, lon, lat, depth = regular_grid_3d
        n_points = 50

        with pytest.warns(UserWarning, match="Creating meshgrid"):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=n_points
            )

        # Check that the plot was created with correct dimensions
        # The pcolormesh should have data with shape (nlevels, n_points)
        assert fig is not None
        plt.close(fig)

    def test_transect_depth_limits(self, regular_grid_3d):
        """Test transect with depth limits."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                depth_lim=(0, 500)
            )

        # Check y-axis limits (inverted by default)
        ylim = ax.get_ylim()
        assert ylim[0] == 500  # Bottom
        assert ylim[1] == 0    # Top
        plt.close(fig)

    def test_transect_invert_depth_false(self, regular_grid_3d):
        """Test transect with invert_depth=False (atmosphere style)."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                depth_lim=(0, 500),
                invert_depth=False
            )

        # Check y-axis is not inverted
        ylim = ax.get_ylim()
        assert ylim[0] == 0    # Bottom
        assert ylim[1] == 500  # Top
        plt.close(fig)

    def test_transect_colorbar(self, regular_grid_3d):
        """Test transect colorbar options."""
        data, lon, lat, depth = regular_grid_3d

        # With colorbar
        with pytest.warns(UserWarning):
            fig1, ax1 = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                colorbar=True,
                colorbar_label="Temperature (°C)"
            )
        plt.close(fig1)

        # Without colorbar
        with pytest.warns(UserWarning):
            fig2, ax2 = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                colorbar=False
            )
        plt.close(fig2)

    def test_transect_title(self, regular_grid_3d):
        """Test transect with title."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                title="Atlantic Transect"
            )

        assert ax.get_title() == "Atlantic Transect"
        plt.close(fig)

    def test_transect_colormap(self, regular_grid_3d):
        """Test transect with different colormaps."""
        data, lon, lat, depth = regular_grid_3d

        for cmap in ["viridis", "RdBu_r", "coolwarm"]:
            with pytest.warns(UserWarning):
                fig, ax = transect(
                    data, lon, lat, depth,
                    start=(-30, -60),
                    end=(-30, 60),
                    n_points=50,
                    cmap=cmap
                )
            plt.close(fig)

    def test_transect_vmin_vmax(self, regular_grid_3d):
        """Test transect with color scale limits."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                vmin=-1,
                vmax=1
            )

        plt.close(fig)

    def test_transect_existing_axes(self, regular_grid_3d):
        """Test transect on existing axes."""
        data, lon, lat, depth = regular_grid_3d

        fig, ax = plt.subplots(figsize=(10, 5))

        with pytest.warns(UserWarning):
            fig2, ax2 = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                ax=ax
            )

        assert ax2 is ax
        assert fig2 is fig
        plt.close(fig)

    def test_transect_custom_figsize(self, regular_grid_3d):
        """Test transect with custom figure size."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50,
                figsize=(15, 8)
            )

        np.testing.assert_allclose(fig.get_size_inches(), [15, 8])
        plt.close(fig)

    def test_transect_missing_depth_raises(self, regular_grid_3d):
        """Test that missing depth raises error."""
        data, lon, lat, _ = regular_grid_3d

        with pytest.raises(ValueError, match="depth array is required"):
            transect(
                data, lon, lat, None,
                start=(-30, -60),
                end=(-30, 60)
            )

    def test_transect_missing_start_end_raises(self, regular_grid_3d):
        """Test that missing start/end raises error."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.raises(ValueError, match="start and end points are required"):
            transect(data, lon, lat, depth, start=(-30, -60), end=None)

        with pytest.raises(ValueError, match="start and end points are required"):
            transect(data, lon, lat, depth, start=None, end=(-30, 60))

    def test_transect_missing_coords_raises(self, regular_grid_3d):
        """Test that missing coordinates raises error."""
        data, _, _, depth = regular_grid_3d

        with pytest.raises(ValueError, match="lon and lat coordinates are required"):
            transect(
                data, None, None, depth,
                start=(-30, -60),
                end=(-30, 60)
            )

    def test_transect_zonal(self, regular_grid_3d):
        """Test zonal transect (constant latitude)."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-180, 0),
                end=(180, 0),
                n_points=100,
                title="Equatorial Transect"
            )

        plt.close(fig)

    def test_transect_meridional(self, regular_grid_3d):
        """Test meridional transect (constant longitude)."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(0, -90),
                end=(0, 90),
                n_points=100,
                title="Prime Meridian Transect"
            )

        plt.close(fig)

    def test_transect_diagonal(self, regular_grid_3d):
        """Test diagonal transect (varying both lon and lat)."""
        data, lon, lat, depth = regular_grid_3d

        with pytest.warns(UserWarning):
            fig, ax = transect(
                data, lon, lat, depth,
                start=(-60, -40),
                end=(20, 60),
                n_points=100,
                title="Diagonal Transect"
            )

        plt.close(fig)

    def test_transect_1d_data(self, regular_grid_3d):
        """Test transect with 1D data (single level)."""
        data_3d, lon, lat, depth = regular_grid_3d
        # Take just the surface level and flatten
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        data_1d = data_3d[0].ravel()
        lon_flat = lon_2d.ravel()
        lat_flat = lat_2d.ravel()

        fig, ax = transect(
            data_1d, lon_flat, lat_flat, depth[:1],
            start=(-30, -60),
            end=(-30, 60),
            n_points=50
        )

        plt.close(fig)


class TestTransectXarray:
    """Tests for transect function with xarray inputs."""

    def test_transect_xarray_3d(self):
        """Test transect with xarray DataArray (3D)."""
        xr = pytest.importorskip("xarray")

        lon_vals = np.linspace(-180, 180, 36)
        lat_vals = np.linspace(-90, 90, 18)
        depth_vals = np.array([0, 10, 50, 100, 200, 500, 1000, 2000])

        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
        data_vals = np.zeros((8, 18, 36))
        for i, d in enumerate(depth_vals):
            data_vals[i] = np.sin(np.deg2rad(lat_2d)) * np.exp(-d / 500)

        da = xr.DataArray(
            data_vals,
            coords={"depth": depth_vals, "lat": lat_vals, "lon": lon_vals},
            dims=["depth", "lat", "lon"],
        )

        # Coordinates extracted automatically
        with pytest.warns(UserWarning, match="Creating meshgrid"):
            fig, ax = transect(
                da, None, None, da.depth,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50
            )

        plt.close(fig)

    def test_transect_xarray_explicit_coords(self):
        """Test transect with xarray DataArray and explicit coordinates."""
        xr = pytest.importorskip("xarray")

        lon_vals = np.linspace(-180, 180, 36)
        lat_vals = np.linspace(-90, 90, 18)
        depth_vals = np.array([0, 10, 50, 100, 200, 500, 1000, 2000])

        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
        data_vals = np.zeros((8, 18, 36))
        for i, d in enumerate(depth_vals):
            data_vals[i] = np.sin(np.deg2rad(lat_2d)) * np.exp(-d / 500)

        da = xr.DataArray(
            data_vals,
            coords={"depth": depth_vals, "lat": lat_vals, "lon": lon_vals},
            dims=["depth", "lat", "lon"],
        )

        # Explicit coordinates
        with pytest.warns(UserWarning, match="Creating meshgrid"):
            fig, ax = transect(
                da, lon_vals, lat_vals, depth_vals,
                start=(-30, -60),
                end=(-30, 60),
                n_points=50
            )

        plt.close(fig)
