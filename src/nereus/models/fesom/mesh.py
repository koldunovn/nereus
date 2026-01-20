"""FESOM2 mesh loading and handling.

This module provides functionality for loading and working with FESOM2 meshes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nereus.models._base import MeshBase

if TYPE_CHECKING:
    pass


@dataclass
class FesomMesh(MeshBase):
    """FESOM2 mesh class.

    This class represents a FESOM2 unstructured mesh with nodes and triangular
    elements. It provides access to node coordinates, areas, and depth information.

    Parameters
    ----------
    mesh_path : str or Path
        Path to the mesh directory containing mesh files.

    Attributes
    ----------
    lon : ndarray
        Longitude of mesh nodes in degrees.
    lat : ndarray
        Latitude of mesh nodes in degrees.
    area : ndarray
        Cluster area (area associated with each node) in m^2.
    n2d : int
        Number of 2D nodes.
    n3d : int
        Number of 3D nodes (total across all levels).
    nlev : int
        Number of vertical levels.
    depth : ndarray
        Depth of each vertical level in meters.
    depth_lev : ndarray
        Depth level boundaries in meters.
    elem : ndarray
        Element (triangle) connectivity, shape (n_elem, 3).

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh("/path/to/mesh")
    >>> print(f"Mesh has {mesh.n2d} nodes")
    >>> fig, ax, _ = nr.plot(data, mesh.lon, mesh.lat)
    """

    _mesh_path: Path = field(repr=False)

    # Node coordinates (2D)
    _lon: NDArray[np.float64] = field(init=False, repr=False)
    _lat: NDArray[np.float64] = field(init=False, repr=False)

    # Areas
    _area: NDArray[np.float64] = field(init=False, repr=False)  # 2D cluster area

    # Vertical structure
    _depth: NDArray[np.float64] = field(init=False, repr=False)  # Level centers
    _depth_lev: NDArray[np.float64] = field(init=False, repr=False)  # Level interfaces

    # Mesh topology
    _elem: NDArray[np.int32] = field(init=False, repr=False)  # Triangle connectivity

    # Dimensions
    _n2d: int = field(init=False, repr=False)
    _nlev: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load mesh data from files."""
        self._mesh_path = Path(self._mesh_path)
        self._load_mesh()

    def _load_mesh(self) -> None:
        """Load mesh files."""
        mesh_path = self._mesh_path

        # Load 2D node coordinates
        nod2d_file = mesh_path / "nod2d.out"
        if nod2d_file.exists():
            self._load_nod2d_out(nod2d_file)
        else:
            # Try netCDF format
            nc_file = mesh_path / "fesom.mesh.diag.nc"
            if nc_file.exists():
                self._load_from_netcdf(nc_file)
            else:
                raise FileNotFoundError(
                    f"Could not find mesh files in {mesh_path}. "
                    "Expected nod2d.out or fesom.mesh.diag.nc"
                )

        # Load element connectivity
        elem_file = mesh_path / "elem2d.out"
        if elem_file.exists():
            self._load_elem2d_out(elem_file)
        else:
            self._elem = np.array([], dtype=np.int32).reshape(0, 3)

        # Load cluster areas
        area_file = mesh_path / "mesh.diag.nc"
        if area_file.exists():
            self._load_area_from_diag(area_file)
        else:
            # Fall back to computing from element areas
            self._compute_cluster_area()

        # Load vertical levels
        depth_file = mesh_path / "aux3d.out"
        if depth_file.exists():
            self._load_aux3d_out(depth_file)
        else:
            # Try netCDF
            nc_file = mesh_path / "fesom.mesh.diag.nc"
            if nc_file.exists():
                self._load_depth_from_netcdf(nc_file)
            else:
                # Default single level
                self._depth = np.array([0.0])
                self._depth_lev = np.array([0.0, 10.0])
                self._nlev = 1

    def _load_nod2d_out(self, filepath: Path) -> None:
        """Load node coordinates from nod2d.out file."""
        with open(filepath) as f:
            n2d = int(f.readline().strip())
            data = np.loadtxt(f, usecols=(1, 2))

        self._n2d = n2d
        self._lon = data[:, 0].astype(np.float64)
        self._lat = data[:, 1].astype(np.float64)

    def _load_elem2d_out(self, filepath: Path) -> None:
        """Load element connectivity from elem2d.out file."""
        with open(filepath) as f:
            n_elem = int(f.readline().strip())
            data = np.loadtxt(f, dtype=np.int32)

        # FESOM uses 1-based indexing, convert to 0-based
        self._elem = data[:, :3] - 1

    def _load_aux3d_out(self, filepath: Path) -> None:
        """Load vertical level information from aux3d.out file."""
        with open(filepath) as f:
            nlev = int(f.readline().strip())
            depth_lev = np.array([float(f.readline().strip()) for _ in range(nlev)])

        self._nlev = nlev - 1  # Number of layers = levels - 1
        self._depth_lev = depth_lev

        # Compute level centers
        self._depth = 0.5 * (depth_lev[:-1] + depth_lev[1:])

    def _load_area_from_diag(self, filepath: Path) -> None:
        """Load cluster area from mesh.diag.nc file."""
        import netCDF4 as nc

        with nc.Dataset(filepath) as ds:
            if "cluster_area" in ds.variables:
                self._area = np.array(ds.variables["cluster_area"][:])
            elif "nod_area" in ds.variables:
                self._area = np.array(ds.variables["nod_area"][:])
            else:
                self._compute_cluster_area()

    def _load_from_netcdf(self, filepath: Path) -> None:
        """Load mesh from netCDF file."""
        import netCDF4 as nc

        with nc.Dataset(filepath) as ds:
            self._lon = np.array(ds.variables["lon"][:])
            self._lat = np.array(ds.variables["lat"][:])
            self._n2d = len(self._lon)

            if "area" in ds.variables:
                self._area = np.array(ds.variables["area"][:])
            elif "nod_area" in ds.variables:
                self._area = np.array(ds.variables["nod_area"][:])

    def _load_depth_from_netcdf(self, filepath: Path) -> None:
        """Load depth levels from netCDF file."""
        import netCDF4 as nc

        with nc.Dataset(filepath) as ds:
            if "depth" in ds.variables:
                self._depth = np.array(ds.variables["depth"][:])
                self._nlev = len(self._depth)
                # Estimate level interfaces
                self._depth_lev = np.zeros(self._nlev + 1)
                self._depth_lev[1:-1] = 0.5 * (self._depth[:-1] + self._depth[1:])
                self._depth_lev[-1] = 2 * self._depth[-1] - self._depth_lev[-2]

    def _compute_cluster_area(self) -> None:
        """Compute approximate cluster area from elements."""
        if len(self._elem) == 0:
            # Estimate from latitude (rough approximation)
            earth_area = 4 * np.pi * 6371000**2
            self._area = np.full(self._n2d, earth_area / self._n2d)
            return

        # Compute element areas and distribute to nodes
        self._area = np.zeros(self._n2d, dtype=np.float64)

        for tri in self._elem:
            # Get triangle vertices
            lon_tri = self._lon[tri]
            lat_tri = self._lat[tri]

            # Compute spherical area (approximate)
            area = self._compute_triangle_area(lon_tri, lat_tri)

            # Distribute 1/3 of area to each vertex
            self._area[tri] += area / 3

    def _compute_triangle_area(
        self, lon: NDArray[np.float64], lat: NDArray[np.float64]
    ) -> float:
        """Compute approximate area of spherical triangle."""
        R = 6371000.0  # Earth radius in meters

        # Convert to radians
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)

        # Use cross-product formula for spherical triangles (approximate)
        # Convert to Cartesian
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Edge vectors
        v1 = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
        v2 = np.array([x[2] - x[0], y[2] - y[0], z[2] - z[0]])

        # Cross product magnitude gives 2 * area on unit sphere
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross) * R**2

        return float(area)

    @property
    def lon(self) -> NDArray[np.float64]:
        """Longitude coordinates in degrees."""
        return self._lon

    @property
    def lat(self) -> NDArray[np.float64]:
        """Latitude coordinates in degrees."""
        return self._lat

    @property
    def area(self) -> NDArray[np.float64]:
        """Cluster area in square meters."""
        return self._area

    @property
    def n2d(self) -> int:
        """Number of 2D nodes."""
        return self._n2d

    @property
    def nlev(self) -> int:
        """Number of vertical levels."""
        return self._nlev

    @property
    def n3d(self) -> int:
        """Total number of 3D nodes."""
        return self._n2d * self._nlev

    @property
    def depth(self) -> NDArray[np.float64]:
        """Depth of level centers in meters."""
        return self._depth

    @property
    def depth_lev(self) -> NDArray[np.float64]:
        """Depth of level interfaces in meters."""
        return self._depth_lev

    @property
    def elem(self) -> NDArray[np.int32]:
        """Element (triangle) connectivity, shape (n_elem, 3)."""
        return self._elem

    @property
    def layer_thickness(self) -> NDArray[np.float64]:
        """Thickness of each layer in meters."""
        return np.diff(self._depth_lev)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FesomMesh(n2d={self._n2d}, nlev={self._nlev}, "
            f"n_elem={len(self._elem)}, path='{self._mesh_path}')"
        )


def load_mesh(mesh_path: str | os.PathLike) -> FesomMesh:
    """Load a FESOM2 mesh.

    Parameters
    ----------
    mesh_path : str or path-like
        Path to the mesh directory containing mesh files.

    Returns
    -------
    FesomMesh
        Loaded mesh object.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh("/work/meshes/core2")
    >>> print(mesh)
    FesomMesh(n2d=126859, nlev=47, n_elem=...)
    """
    return FesomMesh(_mesh_path=Path(mesh_path))


def open_dataset(
    data_path: str | os.PathLike,
    mesh: FesomMesh | None = None,
    mesh_path: str | os.PathLike | None = None,
) -> "xr.Dataset":
    """Open a FESOM2 data file with mesh information.

    Parameters
    ----------
    data_path : str or path-like
        Path to the data file (NetCDF).
    mesh : FesomMesh, optional
        Pre-loaded mesh. If not provided, mesh_path must be specified.
    mesh_path : str or path-like, optional
        Path to mesh directory. Ignored if mesh is provided.

    Returns
    -------
    xr.Dataset
        Dataset with mesh coordinates attached.

    Examples
    --------
    >>> ds = nr.fesom.open_dataset("temp.fesom.2010.nc", mesh=mesh)
    >>> ds = nr.fesom.open_dataset("temp.fesom.2010.nc", mesh_path="/meshes/core2")
    """
    import xarray as xr

    # Load mesh if not provided
    if mesh is None:
        if mesh_path is None:
            raise ValueError("Either mesh or mesh_path must be provided")
        mesh = load_mesh(mesh_path)

    # Open dataset
    ds = xr.open_dataset(data_path)

    # Add mesh coordinates
    n2d = mesh.n2d
    if "nod2" in ds.dims:
        ds = ds.assign_coords(
            lon=("nod2", mesh.lon),
            lat=("nod2", mesh.lat),
        )
    elif "nodes_2d" in ds.dims:
        ds = ds.assign_coords(
            lon=("nodes_2d", mesh.lon),
            lat=("nodes_2d", mesh.lat),
        )

    # Add depth coordinates if applicable
    if "nz" in ds.dims or "nz1" in ds.dims:
        depth_dim = "nz" if "nz" in ds.dims else "nz1"
        if len(ds[depth_dim]) == mesh.nlev:
            ds = ds.assign_coords(depth=(depth_dim, mesh.depth))

    return ds
