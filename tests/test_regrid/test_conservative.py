"""Tests for the "conservative" method of nereus.regrid.interpolator."""

import numpy as np
import pytest

from nereus.regrid.conservative import build_voronoi_cells, geometry_spherical_area
from nereus.regrid.interpolator import RegridInterpolator


class TestConservativeInterpolation:
    """Tests for conservative (area-weighted overlap) remapping."""

    def test_conservative_basic(self, random_mesh_small, synthetic_data):
        """Basic conservative regridding onto a regular grid."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=10.0, method="conservative")

        assert interp.shape == interp.target_lon.shape

        result = interp(synthetic_data)
        assert result.shape == interp.target_lon.shape
        assert np.isfinite(result).any()

    def test_conservative_return_fraction(self, random_mesh_small, synthetic_data):
        """return_fraction=True yields a same-shaped coverage fraction in [0, 1]."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=10.0, method="conservative")

        result, fraction = interp(synthetic_data, return_fraction=True)

        assert fraction.shape == result.shape
        finite = np.isfinite(fraction)
        assert finite.any()
        assert (fraction[finite] >= 0.0).all()
        assert (fraction[finite] <= 1.0).all()
        # Every fully-covered cell must have a finite value
        assert np.isfinite(result[fraction > 0]).all()

    def test_conservative_mass_conservation(self, random_mesh_small):
        """Sum(source_value * source_area) ~= sum(target_value * covered_area).

        Uses a positive-definite field: for a near-zero-mean field, a tiny
        absolute error in the (nearly cancelling) area-weighted sum blows up
        the *relative* error, even though the absolute conservation error
        stays small -- so a positive-definite field is the meaningful check
        here.
        """
        lon, lat = random_mesh_small
        data = 10.0 + np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp = RegridInterpolator(lon, lat, resolution=5.0, method="conservative")
        regridded, fraction = interp(data, return_fraction=True)

        source_polys = build_voronoi_cells(lon, lat)
        source_area = np.array(
            [geometry_spherical_area(poly) for poly in source_polys]
        )
        source_integral = np.sum(data * source_area)

        covered_area = interp._target_area.reshape(regridded.shape) * fraction
        target_integral = np.nansum(regridded * covered_area)

        assert target_integral == pytest.approx(source_integral, rel=0.02)

    def test_conservative_arbitrary_target(self, random_mesh_small, synthetic_data):
        """Conservative remapping onto an arbitrary unstructured target."""
        lon, lat = random_mesh_small
        rng = np.random.default_rng(7)
        target_lon = rng.uniform(-180, 180, 200)
        target_lat = rng.uniform(-90, 90, 200)

        interp = RegridInterpolator(
            lon,
            lat,
            method="conservative",
            target_lon=target_lon,
            target_lat=target_lat,
        )

        assert interp.shape == (200,)

        result, fraction = interp(synthetic_data, return_fraction=True)
        assert result.shape == (200,)
        assert fraction.shape == (200,)

    def test_conservative_multidim(self, random_mesh_small, synthetic_3d_data):
        """Conservative regridding batches ND data (all levels) at once."""
        lon, lat = random_mesh_small
        data, _depths = synthetic_3d_data
        n_levels = data.shape[0]

        interp = RegridInterpolator(lon, lat, resolution=10.0, method="conservative")
        result = interp(data)

        assert result.shape == (n_levels,) + interp.target_lon.shape

        # Levels are independent linear scalings of the same source field,
        # so a level-doubled input must regrid to (approximately) double.
        doubled = interp(data[0] * 2)
        single = interp(data[0])
        finite = np.isfinite(single) & np.isfinite(doubled) & (single != 0)
        np.testing.assert_allclose(
            doubled[finite], single[finite] * 2, rtol=1e-8, atol=1e-10
        )

    def test_conservative_nan_source(self, random_mesh_small, synthetic_data):
        """NaN source points reduce valid_fraction and are excluded, not propagated."""
        lon, lat = random_mesh_small
        data_nan = synthetic_data.copy()
        data_nan[:300] = np.nan

        interp = RegridInterpolator(lon, lat, resolution=10.0, method="conservative")
        result_full, frac_full = interp(synthetic_data, return_fraction=True)
        result_nan, frac_nan = interp(data_nan, return_fraction=True)

        both_covered = (frac_full > 0) & (frac_nan > 0)
        assert both_covered.any()
        # Coverage can only drop (or stay the same) once source data goes missing.
        assert (frac_nan[both_covered] <= frac_full[both_covered] + 1e-9).all()
        # Cells whose only overlapping source data is NaN fall back to fill_value.
        newly_uncovered = (frac_full > 0) & (frac_nan == 0)
        if newly_uncovered.any():
            assert np.isnan(result_nan[newly_uncovered]).all()

    def test_conservative_return_fraction_requires_method(self, random_mesh_small):
        """return_fraction=True is rejected for non-conservative methods."""
        lon, lat = random_mesh_small
        interp = RegridInterpolator(lon, lat, resolution=10.0, method="nearest")

        with pytest.raises(ValueError, match="conservative"):
            interp(np.zeros(len(lon)), return_fraction=True)

    def test_conservative_target_points_require_method(self, random_mesh_small):
        """target_lon/target_lat are rejected for non-conservative methods."""
        lon, lat = random_mesh_small

        with pytest.raises(ValueError, match="conservative"):
            RegridInterpolator(
                lon,
                lat,
                method="nearest",
                target_lon=lon[:10],
                target_lat=lat[:10],
            )

    def test_conservative_target_points_must_be_paired(self, random_mesh_small):
        """Providing only one of target_lon/target_lat raises."""
        lon, lat = random_mesh_small

        with pytest.raises(ValueError, match="target_lon and target_lat"):
            RegridInterpolator(
                lon, lat, method="conservative", target_lon=lon[:10]
            )
