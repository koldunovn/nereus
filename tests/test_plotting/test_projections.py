"""Tests for nereus.plotting.projections."""

import cartopy.crs as ccrs
import pytest

from nereus.plotting.projections import (
    get_data_bounds_for_projection,
    get_projection,
    is_global_projection,
    is_polar_projection,
)


class TestGetProjection:
    """Tests for get_projection function."""

    def test_platecarree(self):
        """Test PlateCarree aliases."""
        proj = get_projection("pc")
        assert isinstance(proj, ccrs.PlateCarree)

        proj = get_projection("platecarree")
        assert isinstance(proj, ccrs.PlateCarree)

    def test_robinson(self):
        """Test Robinson projection."""
        proj = get_projection("rob")
        assert isinstance(proj, ccrs.Robinson)

    def test_mercator(self):
        """Test Mercator projection."""
        proj = get_projection("merc")
        assert isinstance(proj, ccrs.Mercator)

    def test_north_polar(self):
        """Test North Polar Stereographic."""
        proj = get_projection("npstere")
        assert isinstance(proj, ccrs.NorthPolarStereo)

    def test_south_polar(self):
        """Test South Polar Stereographic."""
        proj = get_projection("spstere")
        assert isinstance(proj, ccrs.SouthPolarStereo)

    def test_orthographic(self):
        """Test Orthographic projection."""
        proj = get_projection("ortho")
        assert isinstance(proj, ccrs.Orthographic)

    def test_custom_kwargs(self):
        """Test passing custom kwargs."""
        proj = get_projection("ortho", central_longitude=45, central_latitude=30)
        assert isinstance(proj, ccrs.Orthographic)

    def test_pass_through_projection(self):
        """Test that existing projections are passed through."""
        orig = ccrs.Robinson()
        proj = get_projection(orig)
        assert proj is orig

    def test_invalid_projection(self):
        """Test that invalid projection raises error."""
        with pytest.raises(ValueError, match="Unknown projection"):
            get_projection("invalid_projection")

    def test_case_insensitive(self):
        """Test that projection names are case insensitive."""
        proj1 = get_projection("PC")
        proj2 = get_projection("pc")
        assert type(proj1) == type(proj2)


class TestIsGlobalProjection:
    """Tests for is_global_projection function."""

    def test_robinson_is_global(self):
        """Test that Robinson is global."""
        assert is_global_projection("rob") is True
        assert is_global_projection(ccrs.Robinson()) is True

    def test_mollweide_is_global(self):
        """Test that Mollweide is global."""
        assert is_global_projection("moll") is True

    def test_platecarree_not_global(self):
        """Test that PlateCarree is not global."""
        assert is_global_projection("pc") is False


class TestIsPolarProjection:
    """Tests for is_polar_projection function."""

    def test_north_polar(self):
        """Test North Polar detection."""
        assert is_polar_projection("npstere") == "north"
        assert is_polar_projection(ccrs.NorthPolarStereo()) == "north"

    def test_south_polar(self):
        """Test South Polar detection."""
        assert is_polar_projection("spstere") == "south"
        assert is_polar_projection(ccrs.SouthPolarStereo()) == "south"

    def test_non_polar(self):
        """Test non-polar projections."""
        assert is_polar_projection("pc") is None
        assert is_polar_projection("rob") is None


class TestGetDataBoundsForProjection:
    """Tests for get_data_bounds_for_projection function."""

    def test_north_polar_bounds(self):
        """Test bounds for north polar projection."""
        lon_bounds, lat_bounds = get_data_bounds_for_projection("npstere")

        assert lon_bounds == (-180.0, 180.0)
        assert lat_bounds[0] >= 0  # Northern hemisphere only
        assert lat_bounds[1] == 90.0

    def test_south_polar_bounds(self):
        """Test bounds for south polar projection."""
        lon_bounds, lat_bounds = get_data_bounds_for_projection("spstere")

        assert lon_bounds == (-180.0, 180.0)
        assert lat_bounds[0] == -90.0
        assert lat_bounds[1] <= 0  # Southern hemisphere only

    def test_custom_extent(self):
        """Test with custom extent."""
        lon_bounds, lat_bounds = get_data_bounds_for_projection(
            "pc", extent=(0, 30, 40, 60)
        )

        # Should have buffer
        assert lon_bounds[0] <= 0
        assert lon_bounds[1] >= 30
        assert lat_bounds[0] <= 40
        assert lat_bounds[1] >= 60
