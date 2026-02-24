"""Tests for MPAS-Ocean model support."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import nereus as nr
from nereus.models.mpas.mesh import load_mesh, open_dataset


# ---------------------------------------------------------------------------
# Helpers for creating synthetic MPAS NetCDF files
# ---------------------------------------------------------------------------

NCELLS = 20
NVERTLEVELS = 5


def _make_mpas_file(path, include_data=False):
    """Create a minimal MPAS-Ocean NetCDF file with mesh variables."""
    # Cell coordinates in radians
    lon_rad = np.linspace(0.01, 2 * np.pi - 0.01, NCELLS)
    lat_rad = np.linspace(-np.pi / 4, np.pi / 4, NCELLS)
    area = np.full(NCELLS, 1e9, dtype=np.float64)
    bottom_depth = np.linspace(100.0, 5000.0, NCELLS)
    max_level = np.linspace(2, NVERTLEVELS, NCELLS).astype(np.int32)
    ref_bottom_depth = np.array([10.0, 30.0, 60.0, 100.0, 200.0])

    ds = xr.Dataset(
        {
            "lonCell": (("nCells",), lon_rad, {
                "units": "radians",
                "long_name": "Longitude location of cell centers in radians.",
            }),
            "latCell": (("nCells",), lat_rad, {
                "units": "radians",
                "long_name": "Latitude location of cell centers in radians.",
            }),
            "areaCell": (("nCells",), area, {
                "units": "m^2",
                "long_name": "Area of each cell in the primary grid.",
            }),
            "bottomDepth": (("nCells",), bottom_depth, {
                "units": "m",
                "long_name": "Depth of the bottom of the ocean.",
            }),
            "maxLevelCell": (("nCells",), max_level, {
                "units": "unitless",
                "long_name": "Index to the last active ocean cell.",
            }),
            "refBottomDepth": (("nVertLevels",), ref_bottom_depth, {
                "units": "m",
                "long_name": "Reference depth of ocean for each vertical level.",
            }),
        },
        attrs={
            "on_a_sphere": "YES",
            "sphere_radius": 6371229.0,
            "model_name": "mpas",
            "core_name": "ocean",
            "source": "MPAS",
            "Conventions": "MPAS",
        },
    )

    if include_data:
        rng = np.random.default_rng(42)
        ds["temperature"] = xr.DataArray(
            rng.uniform(0, 30, (1, NCELLS, NVERTLEVELS)),
            dims=("Time", "nCells", "nVertLevels"),
            attrs={"units": "degC", "long_name": "temperature"},
        )
        ds["salinity"] = xr.DataArray(
            rng.uniform(33, 36, (1, NCELLS, NVERTLEVELS)),
            dims=("Time", "nCells", "nVertLevels"),
            attrs={"units": "PSU", "long_name": "salinity"},
        )
        ds["ssh"] = xr.DataArray(
            rng.uniform(-1, 1, (1, NCELLS)),
            dims=("Time", "nCells"),
            attrs={"units": "m", "long_name": "sea surface height"},
        )

    ds.to_netcdf(path)
    ds.close()


@pytest.fixture
def mpas_mesh_file(tmp_path):
    """Create a minimal MPAS mesh-only NetCDF file."""
    path = tmp_path / "mpas_mesh.nc"
    _make_mpas_file(path, include_data=False)
    return path


@pytest.fixture
def mpas_data_file(tmp_path):
    """Create an MPAS NetCDF file with mesh + data variables."""
    path = tmp_path / "mpas_data.nc"
    _make_mpas_file(path, include_data=True)
    return path


@pytest.fixture
def mpas_minimal_file(tmp_path):
    """Create an MPAS file with only the minimum required variables."""
    path = tmp_path / "mpas_minimal.nc"
    lon_rad = np.linspace(0.1, 6.0, 10)
    lat_rad = np.linspace(-1.0, 1.0, 10)
    area = np.full(10, 2e9, dtype=np.float64)

    ds = xr.Dataset({
        "lonCell": (("nCells",), lon_rad),
        "latCell": (("nCells",), lat_rad),
        "areaCell": (("nCells",), area),
    })
    ds.to_netcdf(path)
    ds.close()
    return path


# ---------------------------------------------------------------------------
# Tests: mesh.py - load_mesh
# ---------------------------------------------------------------------------

class TestLoadMesh:
    """Tests for MPAS mesh loading."""

    def test_load_basic_mesh(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        assert isinstance(mesh, xr.Dataset)
        assert mesh.sizes["npoints"] == NCELLS
        assert "lon" in mesh
        assert "lat" in mesh
        assert "area" in mesh

    def test_mesh_coordinates_degrees(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        # Should be converted from radians to degrees
        assert np.all(mesh["lon"].values >= -180)
        assert np.all(mesh["lon"].values <= 180)
        assert np.all(mesh["lat"].values >= -90)
        assert np.all(mesh["lat"].values <= 90)

    def test_mesh_area(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        np.testing.assert_allclose(mesh["area"].values, 1e9)

    def test_mesh_depth(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        assert "depth" in mesh
        assert mesh.sizes["depth_level"] == NVERTLEVELS
        assert np.all(mesh["depth"].values > 0)

    def test_mesh_depth_values(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        # refBottomDepth = [10, 30, 60, 100, 200]
        # layer_thickness = [10, 20, 30, 40, 100]
        # depth (mid) = [5, 20, 45, 80, 150]
        expected_depth = np.array([5.0, 20.0, 45.0, 80.0, 150.0])
        np.testing.assert_allclose(mesh["depth"].values, expected_depth)

    def test_mesh_layer_thickness(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        assert "layer_thickness" in mesh
        expected = np.array([10.0, 20.0, 30.0, 40.0, 100.0])
        np.testing.assert_allclose(mesh["layer_thickness"].values, expected)

    def test_mesh_bathymetry(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        assert "bathymetry" in mesh
        assert mesh["bathymetry"].shape == (NCELLS,)
        assert np.all(mesh["bathymetry"].values > 0)

    def test_mesh_max_level_cell(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        assert "maxLevelCell" in mesh
        assert mesh["maxLevelCell"].shape == (NCELLS,)

    def test_mesh_metadata(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file)
        assert mesh.attrs["nereus_mesh_type"] == "mpas"
        assert "nereus_mesh_version" in mesh.attrs
        assert mesh.attrs["nereus_dask_backend"] is False

    def test_mesh_dask(self, mpas_mesh_file):
        mesh = load_mesh(mpas_mesh_file, use_dask=True)
        assert mesh.attrs["nereus_dask_backend"] is True

    def test_minimal_file(self, mpas_minimal_file):
        """Test loading a file with only lon/lat/area (no depth, no bathy)."""
        mesh = load_mesh(mpas_minimal_file)
        assert mesh.sizes["npoints"] == 10
        assert "lon" in mesh
        assert "lat" in mesh
        assert "area" in mesh
        assert "depth" not in mesh
        assert "bathymetry" not in mesh
        assert "maxLevelCell" not in mesh

    def test_lon_normalization(self, tmp_path):
        """MPAS lon is [0, 2pi] radians -> should be [-180, 180] degrees."""
        path = tmp_path / "lon_test.nc"
        # lon near 2*pi -> ~360 degrees -> should become ~0 (or negative)
        lon_rad = np.array([0.01, np.pi, 2 * np.pi - 0.01])
        lat_rad = np.array([0.0, 0.0, 0.0])
        area = np.ones(3) * 1e9

        ds = xr.Dataset({
            "lonCell": (("nCells",), lon_rad),
            "latCell": (("nCells",), lat_rad),
            "areaCell": (("nCells",), area),
        })
        ds.to_netcdf(path)
        ds.close()

        mesh = load_mesh(path)
        assert np.all(mesh["lon"].values >= -180)
        assert np.all(mesh["lon"].values <= 180)


# ---------------------------------------------------------------------------
# Tests: open_dataset
# ---------------------------------------------------------------------------

class TestOpenDataset:
    """Tests for MPAS data loading with mesh coordinates."""

    def test_open_with_mesh(self, mpas_data_file):
        mesh = load_mesh(mpas_data_file)
        ds = open_dataset(mpas_data_file, mesh=mesh)
        assert "lon" in ds.coords
        assert "lat" in ds.coords
        assert "temperature" in ds

    def test_open_without_mesh(self, mpas_data_file):
        """When no mesh is given, load it from the data file itself."""
        ds = open_dataset(mpas_data_file)
        assert "lon" in ds.coords
        assert "lat" in ds.coords

    def test_open_with_mesh_path(self, mpas_data_file):
        ds = open_dataset(mpas_data_file, mesh_path=mpas_data_file)
        assert "lon" in ds.coords
        assert "lat" in ds.coords

    def test_depth_coord_attached(self, mpas_data_file):
        mesh = load_mesh(mpas_data_file)
        ds = open_dataset(mpas_data_file, mesh=mesh)
        assert "depth" in ds.coords

    def test_data_variables_present(self, mpas_data_file):
        ds = open_dataset(mpas_data_file)
        assert "temperature" in ds
        assert "salinity" in ds
        assert "ssh" in ds

    def test_data_dims(self, mpas_data_file):
        ds = open_dataset(mpas_data_file)
        assert ds["temperature"].dims == ("Time", "nCells", "nVertLevels")
        assert ds["ssh"].dims == ("Time", "nCells")


# ---------------------------------------------------------------------------
# Tests: universal loader auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetection:
    """Tests for MPAS auto-detection in universal loader."""

    def test_detect_mpas(self, mpas_mesh_file):
        from nereus.models import detect_mesh_type

        assert detect_mesh_type(mpas_mesh_file) == "mpas"

    def test_load_mesh_auto(self, mpas_mesh_file):
        mesh = nr.load_mesh(mpas_mesh_file)
        assert mesh.attrs["nereus_mesh_type"] == "mpas"

    def test_load_mesh_explicit(self, mpas_mesh_file):
        mesh = nr.load_mesh(mpas_mesh_file, mesh_type="mpas")
        assert mesh.attrs["nereus_mesh_type"] == "mpas"

    def test_non_mpas_not_detected(self, tmp_path):
        """A NetCDF file without MPAS variables should not be detected as MPAS."""
        from nereus.models import detect_mesh_type

        path = tmp_path / "not_mpas.nc"
        ds = xr.Dataset({"dummy": (("x",), [1, 2, 3])})
        ds.to_netcdf(path)
        ds.close()

        assert detect_mesh_type(path) != "mpas"
