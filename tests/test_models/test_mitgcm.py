"""Tests for MITgcm model support."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import nereus as nr
from nereus.models.mitgcm.io import (
    find_iterations,
    get_dtype_from_meta,
    get_shape_from_meta,
    parse_meta,
    read_mds,
)
from nereus.models.mitgcm.mesh import load_mesh
from nereus.models.mitgcm.data import open_dataset


# ---------------------------------------------------------------------------
# Helpers for writing synthetic MDS files
# ---------------------------------------------------------------------------

def _write_meta(path, text):
    """Write a .meta file."""
    Path(path).write_text(text)


def _write_data(path, array):
    """Write a .data file in big-endian float32."""
    arr = np.asarray(array, dtype=">f4")
    arr.tofile(str(path))


def _make_grid_meta(nx, ny):
    """Return meta text for a 2D grid file (Fortran order: nx, ny)."""
    return (
        f" nDims = [   2 ];\n"
        f" dimList = [\n"
        f"   {nx},    1,   {nx},\n"
        f"   {ny},    1,   {ny}\n"
        f" ];\n"
        f" dataprec = [ 'float32' ];\n"
        f" nrecords = [     1 ];\n"
    )


def _make_diag2d_meta(nx, ny, nflds, fld_names, iteration):
    """Return meta text for a 2D diagnostic file."""
    fld_str = " ".join(f"'{name:<8s}'" for name in fld_names)
    return (
        f" nDims = [   2 ];\n"
        f" dimList = [\n"
        f"   {nx},    1,   {nx},\n"
        f"   {ny},    1,   {ny}\n"
        f" ];\n"
        f" dataprec = [ 'float32' ];\n"
        f" nrecords = [     {nflds} ];\n"
        f" timeStepNumber = [   {iteration} ];\n"
        f" timeInterval = [  0.000000000000E+00  1.000000000000E+04 ];\n"
        f" missingValue = [ -9.99000000000000E+02 ];\n"
        f" nFlds = [   {nflds} ];\n"
        f" fldList = {{ {fld_str} }};\n"
    )


def _make_diag3d_meta(nx, ny, nz, nflds, fld_names, iteration):
    """Return meta text for a 3D diagnostic file."""
    fld_str = " ".join(f"'{name:<8s}'" for name in fld_names)
    return (
        f" nDims = [   3 ];\n"
        f" dimList = [\n"
        f"   {nx},    1,   {nx},\n"
        f"   {ny},    1,   {ny},\n"
        f"   {nz},    1,   {nz}\n"
        f" ];\n"
        f" dataprec = [ 'float32' ];\n"
        f" nrecords = [     {nflds} ];\n"
        f" timeStepNumber = [   {iteration} ];\n"
        f" timeInterval = [  0.000000000000E+00  1.000000000000E+04 ];\n"
        f" missingValue = [ -9.99000000000000E+02 ];\n"
        f" nFlds = [   {nflds} ];\n"
        f" fldList = {{ {fld_str} }};\n"
    )


def _make_rc_meta(nz):
    """Return meta text for RC (depth center) file."""
    return (
        f" nDims = [   3 ];\n"
        f" dimList = [\n"
        f"     1,    1,    1,\n"
        f"     1,    1,    1,\n"
        f"   {nz},    1,   {nz}\n"
        f" ];\n"
        f" dataprec = [ 'float32' ];\n"
        f" nrecords = [     1 ];\n"
    )


def _make_hfac_meta(nx, ny, nz):
    """Return meta text for hFacC (3D partial cell fraction) file."""
    return (
        f" nDims = [   3 ];\n"
        f" dimList = [\n"
        f"   {nx},    1,   {nx},\n"
        f"   {ny},    1,   {ny},\n"
        f"   {nz},    1,   {nz}\n"
        f" ];\n"
        f" dataprec = [ 'float32' ];\n"
        f" nrecords = [     1 ];\n"
        f" timeStepNumber = [     0 ];\n"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 4, 3, 5


@pytest.fixture
def grid_dir(tmp_path):
    """Create a minimal MITgcm grid directory (4x3, 5 depth levels)."""
    # XC / YC - cell center coordinates
    lon_2d = np.tile(np.linspace(-10, 10, NX), (NY, 1))  # (3, 4)
    lat_2d = np.tile(np.linspace(-5, 5, NY), (NX, 1)).T  # (3, 4)

    for prefix, arr in [("XC", lon_2d), ("YC", lat_2d)]:
        _write_meta(tmp_path / f"{prefix}.meta", _make_grid_meta(NX, NY))
        _write_data(tmp_path / f"{prefix}.data", arr)

    # RAC - cell areas
    area_2d = np.ones((NY, NX), dtype=np.float32) * 1e10
    _write_meta(tmp_path / "RAC.meta", _make_grid_meta(NX, NY))
    _write_data(tmp_path / "RAC.data", area_2d)

    # RC - depth centers (negative values, as in MITgcm convention)
    rc = -np.array([5, 15, 30, 55, 90], dtype=np.float32)
    _write_meta(tmp_path / "RC.meta", _make_rc_meta(NZ))
    _write_data(tmp_path / "RC.data", rc)

    # DRF - layer thicknesses
    drf = np.array([10, 10, 20, 30, 40], dtype=np.float32)
    _write_meta(tmp_path / "DRF.meta", _make_rc_meta(NZ))
    _write_data(tmp_path / "DRF.data", drf)

    # Depth - bathymetry (column 0 is land = 0)
    bathy = np.full((NY, NX), 100.0, dtype=np.float32)
    bathy[:, 0] = 0.0  # first column is land
    _write_meta(tmp_path / "Depth.meta", _make_grid_meta(NX, NY))
    _write_data(tmp_path / "Depth.data", bathy)

    return tmp_path


@pytest.fixture
def grid_dir_with_hfac(grid_dir):
    """Add hFacC to the grid directory. Column 0 is land at all levels,
    column 1 is ocean at surface but land below level 2."""
    hfac = np.ones((NZ, NY, NX), dtype=np.float32)
    hfac[:, :, 0] = 0.0           # column 0: full land
    hfac[2:, :, 1] = 0.0          # column 1: ocean levels 0-1, land below
    hfac[1, :, 1] = 0.5           # column 1, level 1: partial cell

    _write_meta(grid_dir / "hFacC.meta", _make_hfac_meta(NX, NY, NZ))
    _write_data(grid_dir / "hFacC.data", hfac)
    return grid_dir


@pytest.fixture
def diag_dir(grid_dir):
    """Add diagnostic files to grid directory (2 iterations, 2D + 3D)."""
    iters = [100, 200]
    flds_2d = ["ETAN", "SST"]
    flds_3d = ["THETA", "SALT"]

    for it in iters:
        # 2D diagnostics
        data_2d = np.stack(
            [np.full((NY, NX), float(i + 1) + it / 1000.0, dtype=np.float32)
             for i in range(len(flds_2d))]
        )
        _write_meta(
            grid_dir / f"diags2D.{it:010d}.meta",
            _make_diag2d_meta(NX, NY, len(flds_2d), flds_2d, it),
        )
        _write_data(grid_dir / f"diags2D.{it:010d}.data", data_2d)

        # 3D diagnostics
        data_3d = np.stack(
            [np.full((NZ, NY, NX), float(i + 1) + it / 1000.0, dtype=np.float32)
             for i in range(len(flds_3d))]
        )
        _write_meta(
            grid_dir / f"diags3D.{it:010d}.meta",
            _make_diag3d_meta(NX, NY, NZ, len(flds_3d), flds_3d, it),
        )
        _write_data(grid_dir / f"diags3D.{it:010d}.data", data_3d)

    return grid_dir


# ---------------------------------------------------------------------------
# Tests: io.py
# ---------------------------------------------------------------------------

class TestParseMeta:
    """Tests for MDS meta file parsing."""

    def test_parse_grid_meta(self, grid_dir):
        meta = parse_meta(grid_dir / "XC.meta")
        assert meta["nDims"] == 2
        assert meta["dimList"] == [[NX, 1, NX], [NY, 1, NY]]
        assert meta["dataprec"] == "float32"
        assert meta["nrecords"] == 1

    def test_parse_diagnostic_meta(self, diag_dir):
        meta = parse_meta(diag_dir / "diags2D.0000000100.meta")
        assert meta["nFlds"] == 2
        assert len(meta["fldList"]) == 2
        assert "ETAN" in meta["fldList"][0]
        assert meta["timeStepNumber"] == 100
        assert meta["missingValue"] == pytest.approx(-999.0)

    def test_parse_3d_meta(self, diag_dir):
        meta = parse_meta(diag_dir / "diags3D.0000000100.meta")
        assert meta["nDims"] == 3
        assert len(meta["dimList"]) == 3


class TestShapeAndDtype:
    """Tests for shape and dtype extraction."""

    def test_get_shape_2d(self):
        meta = {"dimList": [[4, 1, 4], [3, 1, 3]]}
        assert get_shape_from_meta(meta) == (3, 4)

    def test_get_shape_3d(self):
        meta = {"dimList": [[4, 1, 4], [3, 1, 3], [5, 1, 5]]}
        assert get_shape_from_meta(meta) == (5, 3, 4)

    def test_get_dtype_float32(self):
        meta = {"dataprec": "float32"}
        dt = get_dtype_from_meta(meta)
        assert dt == np.dtype(">f4")

    def test_get_dtype_float64(self):
        meta = {"dataprec": "float64"}
        dt = get_dtype_from_meta(meta)
        assert dt == np.dtype(">f8")

    def test_get_dtype_default(self):
        dt = get_dtype_from_meta({})
        assert dt == np.dtype(">f4")


class TestReadMds:
    """Tests for reading MDS file pairs."""

    def test_read_grid_file(self, grid_dir):
        meta, data = read_mds(grid_dir / "XC")
        assert data.shape == (NY, NX)
        assert data.dtype == np.dtype(">f4")

    def test_read_multi_field(self, diag_dir):
        meta, data = read_mds(diag_dir / "diags2D", iteration=100)
        assert data.shape == (2, NY, NX)  # 2 fields

    def test_read_3d_multi_field(self, diag_dir):
        meta, data = read_mds(diag_dir / "diags3D", iteration=100)
        assert data.shape == (2, NZ, NY, NX)  # 2 fields x 3D

    def test_read_with_iteration(self, diag_dir):
        meta, data = read_mds(diag_dir / "diags2D", iteration=200)
        assert meta["timeStepNumber"] == 200


class TestFindIterations:
    """Tests for iteration discovery."""

    def test_find_iterations(self, diag_dir):
        iters = find_iterations(diag_dir, "diags2D")
        assert iters == [100, 200]

    def test_find_iterations_3d(self, diag_dir):
        iters = find_iterations(diag_dir, "diags3D")
        assert iters == [100, 200]

    def test_find_iterations_nonexistent(self, grid_dir):
        iters = find_iterations(grid_dir, "nonexistent")
        assert iters == []


# ---------------------------------------------------------------------------
# Tests: mesh.py
# ---------------------------------------------------------------------------

class TestLoadMesh:
    """Tests for MITgcm mesh loading."""

    def test_load_basic_mesh(self, grid_dir):
        mesh = load_mesh(grid_dir)
        assert isinstance(mesh, xr.Dataset)
        assert mesh.sizes["npoints"] == NX * NY
        assert "lon" in mesh
        assert "lat" in mesh
        assert "area" in mesh

    def test_mesh_coordinates(self, grid_dir):
        mesh = load_mesh(grid_dir)
        # Check shape is flattened
        assert mesh["lon"].shape == (NX * NY,)
        assert mesh["lat"].shape == (NX * NY,)

    def test_mesh_area(self, grid_dir):
        mesh = load_mesh(grid_dir)
        # RAC was set to 1e10
        np.testing.assert_allclose(mesh["area"].values, 1e10)

    def test_mesh_depth(self, grid_dir):
        mesh = load_mesh(grid_dir)
        assert "depth" in mesh
        assert mesh.sizes["depth_level"] == NZ
        # Depths should be positive (abs of negative RC values)
        assert np.all(mesh["depth"].values > 0)

    def test_mesh_layer_thickness(self, grid_dir):
        mesh = load_mesh(grid_dir)
        assert "layer_thickness" in mesh
        np.testing.assert_allclose(
            mesh["layer_thickness"].values, [10, 10, 20, 30, 40]
        )

    def test_mesh_bathymetry(self, grid_dir):
        mesh = load_mesh(grid_dir)
        assert "bathymetry" in mesh
        # Column 0 is land (Depth=0), rest is ocean (Depth=100)
        bathy = mesh["bathymetry"].values
        assert bathy[0] == 0.0
        assert bathy[1] == 100.0

    def test_mesh_metadata(self, grid_dir):
        mesh = load_mesh(grid_dir)
        assert mesh.attrs["nereus_mesh_type"] == "mitgcm"
        assert "nereus_mesh_version" in mesh.attrs
        assert mesh.attrs["nx"] == NX
        assert mesh.attrs["ny"] == NY
        assert mesh.attrs["original_shape"] == (NY, NX)

    def test_mesh_lon_normalization(self, tmp_path):
        """Test lon values > 180 are normalized."""
        lon_2d = np.array([[200, 350], [200, 350]], dtype=np.float32)
        lat_2d = np.array([[10, 10], [20, 20]], dtype=np.float32)

        _write_meta(tmp_path / "XC.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "XC.data", lon_2d)
        _write_meta(tmp_path / "YC.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "YC.data", lat_2d)

        mesh = load_mesh(tmp_path)
        assert np.all(mesh["lon"].values >= -180)
        assert np.all(mesh["lon"].values <= 180)

    def test_mesh_without_rac(self, tmp_path):
        """Test area estimation when RAC.data is not present."""
        lon_2d = np.array([[0, 10], [0, 10]], dtype=np.float32)
        lat_2d = np.array([[0, 0], [10, 10]], dtype=np.float32)

        _write_meta(tmp_path / "XC.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "XC.data", lon_2d)
        _write_meta(tmp_path / "YC.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "YC.data", lat_2d)

        mesh = load_mesh(tmp_path)
        assert "area" in mesh
        assert np.all(mesh["area"].values > 0)

    def test_mesh_dask(self, grid_dir):
        mesh = load_mesh(grid_dir, use_dask=True)
        assert mesh.attrs["nereus_dask_backend"] is True

    def test_grid_type_G(self, tmp_path):
        """Test loading G-grid (corner points)."""
        lon_2d = np.array([[0, 10], [0, 10]], dtype=np.float32)
        lat_2d = np.array([[0, 0], [10, 10]], dtype=np.float32)

        _write_meta(tmp_path / "XG.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "XG.data", lon_2d)
        _write_meta(tmp_path / "YG.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "YG.data", lat_2d)

        mesh = load_mesh(tmp_path, grid_type="G")
        assert mesh.sizes["npoints"] == 4


# ---------------------------------------------------------------------------
# Tests: data.py
# ---------------------------------------------------------------------------

class TestOpenDataset:
    """Tests for MITgcm diagnostic data loading."""

    def test_load_2d_data(self, diag_dir):
        ds = open_dataset(diag_dir, prefix="diags2D")
        assert "ETAN" in ds
        assert "SST" in ds
        assert ds["ETAN"].dims == ("time", "npoints")
        assert ds.sizes["time"] == 2
        assert ds.sizes["npoints"] == NX * NY

    def test_load_3d_data(self, diag_dir):
        ds = open_dataset(diag_dir, prefix="diags3D")
        assert "THETA" in ds
        assert "SALT" in ds
        assert ds["THETA"].dims == ("time", "depth_level", "npoints")
        assert ds.sizes["depth_level"] == NZ

    def test_load_multiple_prefixes(self, diag_dir):
        ds = open_dataset(diag_dir, prefix=["diags2D", "diags3D"])
        # Should have both 2D and 3D fields
        assert "ETAN" in ds
        assert "THETA" in ds

    def test_specific_iterations(self, diag_dir):
        ds = open_dataset(diag_dir, prefix="diags2D", iters=[100])
        assert ds.sizes["time"] == 1

    def test_time_coordinate_seconds(self, diag_dir):
        ds = open_dataset(diag_dir, prefix="diags2D", delta_t=10.0)
        # iters = [100, 200], delta_t = 10 -> times = [1000, 2000]
        np.testing.assert_allclose(ds["time"].values, [1000.0, 2000.0])

    def test_time_coordinate_datetime(self, diag_dir):
        ds = open_dataset(
            diag_dir, prefix="diags2D", delta_t=10.0, ref_date="2000-1-1"
        )
        assert np.issubdtype(ds["time"].dtype, np.datetime64)

    def test_missing_value_replacement(self, tmp_path):
        """Test that missing values are replaced with NaN."""
        nx, ny = 3, 2
        data = np.array([[1.0, -999.0, 3.0], [4.0, 5.0, -999.0]], dtype=np.float32)

        _write_meta(
            tmp_path / "test.0000000001.meta",
            _make_diag2d_meta(nx, ny, 1, ["FLD"], 1),
        )
        _write_data(tmp_path / "test.0000000001.data", data)

        ds = open_dataset(tmp_path, prefix="test")
        vals = ds["FLD"].values[0]  # first time step
        assert np.isnan(vals[1])   # was -999
        assert np.isnan(vals[5])   # was -999
        assert vals[0] == pytest.approx(1.0)

    def test_attach_mesh_coords(self, diag_dir):
        mesh = load_mesh(diag_dir)
        ds = open_dataset(diag_dir, prefix="diags2D", mesh=mesh)
        assert "lon" in ds.coords
        assert "lat" in ds.coords

    def test_attach_mesh_depth(self, diag_dir):
        mesh = load_mesh(diag_dir)
        ds = open_dataset(diag_dir, prefix="diags3D", mesh=mesh)
        assert "depth" in ds.coords

    def test_no_iterations_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No iterations found"):
            open_dataset(tmp_path, prefix="nonexistent")


# ---------------------------------------------------------------------------
# Tests: universal loader auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetection:
    """Tests for MITgcm auto-detection in universal loader."""

    def test_detect_mitgcm(self, grid_dir):
        from nereus.models import detect_mesh_type

        assert detect_mesh_type(grid_dir) == "mitgcm"

    def test_load_mesh_auto(self, grid_dir):
        mesh = nr.load_mesh(grid_dir)
        assert mesh.attrs["nereus_mesh_type"] == "mitgcm"

    def test_load_mesh_explicit(self, grid_dir):
        mesh = nr.load_mesh(grid_dir, mesh_type="mitgcm")
        assert mesh.attrs["nereus_mesh_type"] == "mitgcm"


# ---------------------------------------------------------------------------
# Tests: land masking
# ---------------------------------------------------------------------------

class TestLandMaskWithHfac:
    """Tests for land masking using hFacC."""

    def test_mask_land_false_by_default(self, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac)
        assert "land_mask" not in mesh
        assert "hFacC" not in mesh

    def test_mask_land_loads_hfac(self, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        assert "land_mask" in mesh
        assert "hFacC" in mesh

    def test_land_mask_shape(self, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        assert mesh["land_mask"].shape == (NX * NY,)
        assert mesh["hFacC"].shape == (NZ, NX * NY)

    def test_land_mask_values(self, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        land = mesh["land_mask"].values
        # Column 0 (indices 0, 4, 8 in flattened 3x4) is land
        assert land[0] is np.True_
        assert land[NX] is np.True_
        # Column 1 is ocean at surface
        assert land[1] is np.False_

    def test_hfac_partial_cells(self, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        hfac = mesh["hFacC"].values
        # Column 1, level 1 should be 0.5 (partial cell)
        assert hfac[1, 1] == pytest.approx(0.5)
        # Column 1, levels 2+ should be 0 (land below)
        assert hfac[2, 1] == 0.0

    def test_land_mask_source_attr(self, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        assert mesh["land_mask"].attrs["source"] == "hFacC"


class TestLandMaskDepthFallback:
    """Tests for land masking falling back to Depth when hFacC is absent."""

    def test_fallback_to_depth(self, grid_dir):
        """grid_dir has Depth but no hFacC."""
        mesh = load_mesh(grid_dir, mask_land=True)
        assert "land_mask" in mesh
        assert "hFacC" not in mesh
        assert mesh["land_mask"].attrs["source"] == "Depth"

    def test_depth_mask_values(self, grid_dir):
        mesh = load_mesh(grid_dir, mask_land=True)
        land = mesh["land_mask"].values
        # Column 0 has Depth==0 (land), others have Depth==100
        assert land[0] is np.True_
        assert land[1] is np.False_

    def test_no_mask_files(self, tmp_path):
        """No hFacC, no Depth => mask_land silently does nothing."""
        lon_2d = np.array([[0, 10], [0, 10]], dtype=np.float32)
        lat_2d = np.array([[0, 0], [10, 10]], dtype=np.float32)

        _write_meta(tmp_path / "XC.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "XC.data", lon_2d)
        _write_meta(tmp_path / "YC.meta", _make_grid_meta(2, 2))
        _write_data(tmp_path / "YC.data", lat_2d)

        mesh = load_mesh(tmp_path, mask_land=True)
        assert "land_mask" not in mesh


class TestDataMasking:
    """Tests for mask_land in open_dataset."""

    def test_mask_land_2d(self, diag_dir, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        ds = open_dataset(diag_dir, prefix="diags2D", mesh=mesh, mask_land=True)
        vals = ds["ETAN"].isel(time=0).values
        # Column 0 is land => NaN
        assert np.isnan(vals[0])
        # Column 1 is ocean at surface => not NaN
        assert not np.isnan(vals[1])

    def test_mask_land_3d(self, diag_dir, grid_dir_with_hfac):
        mesh = load_mesh(grid_dir_with_hfac, mask_land=True)
        ds = open_dataset(diag_dir, prefix="diags3D", mesh=mesh, mask_land=True)
        vals = ds["THETA"].isel(time=0).values
        # Column 0 is full land at all levels
        assert np.isnan(vals[0, 0])
        assert np.isnan(vals[2, 0])
        # Column 1, level 0: ocean
        assert not np.isnan(vals[0, 1])
        # Column 1, level 2: land (hFac==0 below level 1)
        assert np.isnan(vals[2, 1])

    def test_mask_land_false_no_change(self, diag_dir):
        mesh = load_mesh(diag_dir)
        ds = open_dataset(diag_dir, prefix="diags2D", mesh=mesh, mask_land=False)
        # No NaN from masking (only from missingValue replacement)
        vals = ds["ETAN"].isel(time=0).values
        assert not np.any(np.isnan(vals))

    def test_mask_land_without_mesh_ignored(self, diag_dir):
        ds = open_dataset(diag_dir, prefix="diags2D", mask_land=True)
        # Should work without error, no masking applied
        vals = ds["ETAN"].isel(time=0).values
        assert not np.any(np.isnan(vals))
