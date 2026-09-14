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


class TestPoleHandling:
    """Regression tests for duplicate Voronoi generators at the poles.

    Every longitude at lat=+-90 maps to the same Cartesian point, so a pole
    row pulled from a regular lat/lon source (a very common case, e.g. any
    reanalysis dataset used directly as unstructured "source" points) used
    to make scipy's SphericalVoronoi raise "Duplicate generators present".
    """

    @staticmethod
    def _regular_grid_with_poles(lon_step=10.0, lat_step=10.0):
        lon_1d = np.arange(-180, 180, lon_step)
        lat_1d = np.arange(-90, 90 + lat_step / 2, lat_step)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        return lon2d.ravel(), lat2d.ravel()

    def test_build_voronoi_cells_no_crash_with_pole_duplicates(self):
        """A pole row with many duplicate Cartesian points no longer raises."""
        lon, lat = self._regular_grid_with_poles()
        n_at_pole = int(np.sum(np.abs(lat) == 90.0))
        assert n_at_pole > 1  # sanity check the fixture actually has duplicates

        polys = build_voronoi_cells(lon, lat)

        assert len(polys) == len(lon)
        assert all(p is not None for p in polys)

    def test_sparse_mesh_pole_owning_cell_has_nonzero_area(self):
        """A cell that encloses a pole -- without any point sitting there --
        must still be a valid, non-degenerate polygon.

        Before this fix, the single point nearest a pole would get a
        self-intersecting planar polygon (its cell's boundary necessarily
        sweeps the full 360 degrees of longitude around it) that collapsed
        to an empty, zero-area geometry after the `buffer(0)` repair --
        silently dropping that source cell's contribution near the pole.
        """
        rng = np.random.default_rng(11)
        lon = rng.uniform(-180, 180, 60)
        lat = rng.uniform(-70, 70, 60)  # nobody within 20 degrees of either pole

        polys = build_voronoi_cells(lon, lat)
        areas = np.array([geometry_spherical_area(p) for p in polys])
        assert (areas > 0.0).all()

        from nereus.core.coordinates import lonlat_to_cartesian

        _, _, z = lonlat_to_cartesian(lon, lat)
        north_owner, south_owner = np.argmax(z), np.argmin(z)
        earth_area = 4 * np.pi * 6_371_000.0**2

        # The pole-owning cells should be a sizeable, sane fraction of the
        # globe (a cap out to ~20-90 degrees co-latitude), not a sliver.
        assert 0.001 < areas[north_owner] / earth_area < 0.5
        assert 0.001 < areas[south_owner] / earth_area < 0.5

    def test_pole_wedges_partition_area_without_inflation(self):
        """Duplicate pole points split one shared cell rather than each
        getting a full copy of it.

        Compares the total area of the 36 duplicated north-pole wedges to
        the area of the single (undeduplicated) Voronoi cell obtained by
        using just *one* point at the pole with the same surrounding mesh.
        """
        lon, lat = self._regular_grid_with_poles()
        polys = build_voronoi_cells(lon, lat)

        north_idx = np.nonzero(lat == 90.0)[0]
        wedge_areas = np.array([geometry_spherical_area(polys[i]) for i in north_idx])

        # Equally spaced duplicates -> equal-area wedges.
        np.testing.assert_allclose(wedge_areas, wedge_areas[0], rtol=1e-6)

        # Rebuild with only a single representative point at the pole.
        keep = np.ones(len(lon), dtype=bool)
        keep[north_idx[1:]] = False
        lon_single, lat_single = lon[keep], lat[keep]
        polys_single = build_voronoi_cells(lon_single, lat_single)
        single_pole_idx = np.nonzero(lat_single == 90.0)[0][0]
        reference_area = geometry_spherical_area(polys_single[single_pole_idx])

        # The chord-based area approximation isn't perfectly additive under
        # subdivision (see conservative.py), so allow a little slack here --
        # this is guarding against gross errors (e.g. an ~N-fold inflation
        # from copying the merged cell to every duplicate), not exactness.
        assert wedge_areas.sum() == pytest.approx(reference_area, rel=1e-3)

    def test_near_pole_points_are_not_merged(self):
        """Points close to, but not exactly at, a pole stay distinct cells."""
        rng = np.random.default_rng(5)
        lon = np.concatenate([[0.0, 90.0, 180.0, -90.0], rng.uniform(-180, 180, 50)])
        lat = np.concatenate([[89.9, 89.9, 89.9, 89.9], rng.uniform(-80, 80, 50)])

        polys = build_voronoi_cells(lon, lat)
        areas = np.array([geometry_spherical_area(p) for p in polys[:4]])

        # Distinct (non-duplicate) cells from an irregular surrounding mesh
        # should not all come out identical the way merged pole wedges do.
        assert not np.allclose(areas, areas[0], rtol=1e-3)

    def test_conservative_regrid_with_pole_duplicated_source(self):
        """End-to-end conservative regrid from a regular-grid-like source
        that includes duplicated pole rows (e.g. reanalysis data)."""
        lon, lat = self._regular_grid_with_poles(lon_step=5.0, lat_step=5.0)
        data = 10.0 + np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))

        interp = RegridInterpolator(lon, lat, resolution=5.0, method="conservative")
        regridded, fraction = interp(data, return_fraction=True)

        assert not np.isnan(regridded).any()

        source_polys = build_voronoi_cells(lon, lat)
        source_area = np.array(
            [geometry_spherical_area(poly) for poly in source_polys]
        )
        source_integral = np.sum(data * source_area)

        covered_area = interp._target_area.reshape(regridded.shape) * fraction
        target_integral = np.nansum(regridded * covered_area)

        assert target_integral == pytest.approx(source_integral, rel=0.02)

    def test_conservative_arbitrary_target_with_pole_duplicates(
        self, random_mesh_small, synthetic_data
    ):
        """Mesh-to-mesh conservative remapping onto a pole-duplicated target."""
        lon, lat = random_mesh_small
        target_lon, target_lat = self._regular_grid_with_poles(
            lon_step=20.0, lat_step=20.0
        )

        interp = RegridInterpolator(
            lon,
            lat,
            method="conservative",
            target_lon=target_lon,
            target_lat=target_lat,
        )
        result, fraction = interp(synthetic_data, return_fraction=True)

        assert result.shape == target_lon.shape
        assert not np.isnan(result).any()
