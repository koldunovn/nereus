"""Tests for IFS TCO model support."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nereus.models.ifs_tco import load_mesh


@pytest.fixture
def ifs_tco_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal grid and area files for IFS TCO tests."""
    grid_path = tmp_path / "tco_grid.nc"
    area_path = tmp_path / "tco_areas.nc"

    a1_lon = np.array([[0.0, 10.0]])
    a1_lat = np.array([[0.0, 5.0]])
    a2_lon = np.array([[190.0, 10.0, 200.0, -170.0]])
    a2_lat = np.array([[45.0, 0.0, -30.0, 60.0]])

    grid = xr.Dataset(
        {
            "A1.lon": (("y_A1", "x_A1"), a1_lon),
            "A1.lat": (("y_A1", "x_A1"), a1_lat),
            "A2.lon": (("y_A2", "x_A2"), a2_lon),
            "A2.lat": (("y_A2", "x_A2"), a2_lat),
        }
    )
    grid.to_netcdf(grid_path)

    area = xr.Dataset(
        {
            "A2.srf": (("y_A2", "x_A2"), np.array([[1.0, 2.0, 3.0, 4.0]])),
        }
    )
    area.to_netcdf(area_path)

    return grid_path, area_path


def test_load_mesh_selects_prefix_and_flattens(ifs_tco_files: tuple[Path, Path]) -> None:
    grid_path, area_path = ifs_tco_files

    mesh = load_mesh(grid_path, area_path)

    assert mesh.sizes["npoints"] == 4
    np.testing.assert_allclose(mesh["lon"].values, [-170.0, 10.0, -160.0, -170.0])
    np.testing.assert_allclose(mesh["lat"].values, [45.0, 0.0, -30.0, 60.0])
    np.testing.assert_allclose(mesh["area"].values, [1.0, 2.0, 3.0, 4.0])
    assert "A2.lon" in mesh
    assert "A2.srf" in mesh
    assert mesh.attrs["nereus_mesh_type"] == "ifs_tco"


def test_load_mesh_metadata(ifs_tco_files: tuple[Path, Path]) -> None:
    grid_path, area_path = ifs_tco_files

    mesh = load_mesh(grid_path, area_path)

    assert mesh.attrs["ifs_tco_prefix"] == "A2"
    assert mesh.attrs["ifs_tco_grid_file"] == str(grid_path)
    assert mesh.attrs["ifs_tco_area_file"] == str(area_path)
