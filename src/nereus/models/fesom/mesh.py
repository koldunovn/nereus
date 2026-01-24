"""FESOM2 mesh loading.

This module provides functionality for loading FESOM2 meshes as xr.Dataset
objects with standardized variable names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from nereus.core.coordinates import EARTH_RADIUS, compute_element_centers
from nereus.core.mesh import (
    add_mesh_metadata,
    normalize_lon,
    should_use_dask,
    validate_mesh,
)

if TYPE_CHECKING:
    pass


def load_mesh(
    path: str | os.PathLike,
    *,
    use_dask: bool | None = None,
) -> xr.Dataset:
    """Load FESOM mesh from directory or NetCDF file.

    Parameters
    ----------
    path : str or Path
        Path to mesh directory (containing nod2d.out, etc.)
        or to fesom.mesh.diag.nc file.
    use_dask : bool, optional
        Whether to use dask arrays. If None, auto-detects based
        on mesh size (>1M points triggers dask).

    Returns
    -------
    xr.Dataset
        Standardized mesh dataset with:
        - lon, lat: Node coordinates (npoints,)
        - area: Node cluster area in m^2 (npoints,)
        - triangles: 0-indexed triangle connectivity (nelem, 3)
        - lon_tri, lat_tri: Element center coordinates (nelem,)
        - depth: Layer center depths in meters (nz,)
        - depth_bounds: Layer interfaces (nz, 2)
        - layer_thickness: Layer thickness in meters (nz,)
        Plus original FESOM variables with their native names.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh("/path/to/mesh")
    >>> print(f"Mesh has {mesh.sizes['npoints']} nodes")
    >>> area = mesh["area"]
    >>> lon = mesh["lon"].values
    """
    path = Path(path)

    if _is_netcdf(path):
        return _load_from_netcdf(path, use_dask=use_dask)
    else:
        return _load_from_ascii(path, use_dask=use_dask)


def _is_netcdf(path: Path) -> bool:
    """Check if path is a netCDF file."""
    if path.is_file():
        return path.suffix in (".nc", ".nc4")
    return False


def _load_from_netcdf(filepath: Path, use_dask: bool | None = None) -> xr.Dataset:
    """Load mesh from fesom.mesh.diag.nc file.

    Parameters
    ----------
    filepath : Path
        Path to netCDF file.
    use_dask : bool, optional
        Whether to use dask arrays.

    Returns
    -------
    xr.Dataset
        Standardized mesh dataset.
    """
    # Open with xarray to get dimensions
    with xr.open_dataset(filepath) as ds_orig:
        npoints = ds_orig.sizes.get("nod2", ds_orig.sizes.get("nod_n", len(ds_orig["lon"])))

    use_dask_actual = should_use_dask(npoints, use_dask)

    # Reopen with appropriate chunking
    if use_dask_actual:
        ds_orig = xr.open_dataset(filepath, chunks={})
    else:
        ds_orig = xr.open_dataset(filepath)

    # Build standardized dataset
    ds = xr.Dataset()

    # --- First, copy ALL original variables with dimension renaming ---
    # Map original dimensions to standardized/renamed dimensions
    # nod2 -> npoints (standardized name for node dimension)
    # nz stays as nz (original 48 levels for interfaces)
    dim_map = {
        "nod2": "npoints",
        "nod_n": "npoints",
    }

    # Copy all original variables
    for var_name in ds_orig.data_vars:
        var = ds_orig[var_name]
        # Rename dimensions according to map
        new_dims = tuple(dim_map.get(d, d) for d in var.dims)
        ds[var_name] = xr.DataArray(
            var.values,
            dims=new_dims,
            attrs=var.attrs,
        )

    # Copy original coordinates (with dimension renaming)
    for coord_name in ds_orig.coords:
        coord = ds_orig[coord_name]
        new_dims = tuple(dim_map.get(d, d) for d in coord.dims)
        ds.coords[coord_name] = xr.DataArray(
            coord.values,
            dims=new_dims if new_dims else (coord_name,),
            attrs=coord.attrs,
        )

    # --- Now add standardized variables ---

    # Standardized lon/lat (normalized)
    lon_data = ds_orig["lon"].values if not use_dask_actual else ds_orig["lon"].data
    lat_data = ds_orig["lat"].values if not use_dask_actual else ds_orig["lat"].data

    # Normalize longitude to [-180, 180]
    if not use_dask_actual:
        lon_data = normalize_lon(lon_data, "pm180")
    else:
        import dask.array as da
        lon_data = da.map_blocks(lambda x: normalize_lon(x, "pm180"), lon_data, dtype=np.float64)

    # Override lon with normalized version
    ds["lon"] = xr.DataArray(
        lon_data,
        dims=("npoints",),
        attrs={
            "units": "degrees_east",
            "long_name": "Longitude",
            "standard_name": "longitude",
        },
    )
    ds["lat"] = xr.DataArray(
        lat_data,
        dims=("npoints",),
        attrs={
            "units": "degrees_north",
            "long_name": "Latitude",
            "standard_name": "latitude",
        },
    )

    # --- Standardized area (surface level) ---
    area_var = None
    for name in ["nod_area", "cluster_area", "area"]:
        if name in ds_orig:
            area_var = name
            break

    if area_var:
        area_raw = ds_orig[area_var].values
        # nod_area may have shape (nz, nod2) - use surface level
        if area_raw.ndim == 2:
            area_data = area_raw[0, :]  # Surface level
        else:
            area_data = area_raw

        if use_dask_actual:
            import dask.array as da
            area_data = da.from_array(area_data, chunks=-1)

        ds["area"] = xr.DataArray(
            area_data,
            dims=("npoints",),
            attrs={
                "units": "m2",
                "long_name": "Node cluster area",
            },
        )
    else:
        area_data = None

    # --- Standardized triangles (0-indexed, shape nelem x 3) ---
    tri_var = None
    for name in ["face_nodes", "elem", "triangles"]:
        if name in ds_orig:
            tri_var = name
            break

    if tri_var:
        tri_data = ds_orig[tri_var].values
        # Convert from 1-indexed to 0-indexed if needed
        if tri_data.min() >= 1:
            tri_data = tri_data - 1

        # Ensure shape is (nelem, 3)
        if tri_data.shape[0] == 3 and tri_data.shape[1] != 3:
            tri_data = tri_data.T

        ds["triangles"] = xr.DataArray(
            tri_data,
            dims=("nelem", "three"),
            attrs={
                "long_name": "Triangle connectivity (0-indexed)",
                "cf_role": "face_node_connectivity",
                "start_index": 0,
            },
        )

        # Compute element centers
        lon_np = ds["lon"].values if not use_dask_actual else ds["lon"].compute().values
        lat_np = ds["lat"].values if not use_dask_actual else ds["lat"].compute().values
        lon_tri, lat_tri = compute_element_centers(lon_np, lat_np, tri_data)

        ds["lon_tri"] = xr.DataArray(
            lon_tri,
            dims=("nelem",),
            attrs={
                "units": "degrees_east",
                "long_name": "Element center longitude",
            },
        )
        ds["lat_tri"] = xr.DataArray(
            lat_tri,
            dims=("nelem",),
            attrs={
                "units": "degrees_north",
                "long_name": "Element center latitude",
            },
        )

        # Compute area from triangles if not available
        if area_data is None:
            area_data = _compute_cluster_area(lon_np, lat_np, tri_data)
            ds["area"] = xr.DataArray(
                area_data,
                dims=("npoints",),
                attrs={
                    "units": "m2",
                    "long_name": "Node cluster area (computed)",
                },
            )

    # --- Standardized depth levels ---
    # FESOM uses nz1 for layer centers (47 levels) and nz for layer interfaces (48 levels)
    # Original nz is kept as-is; standardized depth uses 'depth_level' dimension
    depth_centers = None
    depth_interfaces = None

    if "nz1" in ds_orig.coords:
        depth_centers = ds_orig["nz1"].values

    if "nz" in ds_orig.coords:
        depth_interfaces = ds_orig["nz"].values

    # Create standardized depth variables with 'depth_level' dimension (layer centers)
    if depth_centers is not None:
        ds["depth"] = xr.DataArray(
            depth_centers,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Depth of layer centers",
                "positive": "down",
            },
        )

        # Create depth_bounds and layer_thickness from interfaces if available
        if depth_interfaces is not None and len(depth_interfaces) > 1:
            depth_bounds = np.column_stack([
                depth_interfaces[:-1],
                depth_interfaces[1:],
            ])
            ds["depth_bounds"] = xr.DataArray(
                depth_bounds,
                dims=("depth_level", "nv"),
                attrs={
                    "units": "m",
                    "long_name": "Layer depth bounds",
                },
            )

            layer_thickness = np.diff(depth_interfaces)
            ds["layer_thickness"] = xr.DataArray(
                layer_thickness,
                dims=("depth_level",),
                attrs={
                    "units": "m",
                    "long_name": "Layer thickness",
                },
            )
    elif depth_interfaces is not None and len(depth_interfaces) > 1:
        # No layer centers, compute from interfaces
        depth_centers = (depth_interfaces[:-1] + depth_interfaces[1:]) / 2
        ds["depth"] = xr.DataArray(
            depth_centers,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Depth of layer centers",
                "positive": "down",
            },
        )

        depth_bounds = np.column_stack([
            depth_interfaces[:-1],
            depth_interfaces[1:],
        ])
        ds["depth_bounds"] = xr.DataArray(
            depth_bounds,
            dims=("depth_level", "nv"),
            attrs={
                "units": "m",
                "long_name": "Layer depth bounds",
            },
        )

        layer_thickness = np.diff(depth_interfaces)
        ds["layer_thickness"] = xr.DataArray(
            layer_thickness,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Layer thickness",
            },
        )

    # --- Global attributes ---
    ds.attrs.update(ds_orig.attrs)

    ds_orig.close()

    return add_mesh_metadata(ds, "fesom", filepath, use_dask=use_dask_actual)


def _load_from_ascii(mesh_dir: Path, use_dask: bool | None = None) -> xr.Dataset:
    """Load mesh from ASCII files.

    Expects mesh directory with:
    - nod2d.out: Node coordinates
    - elem2d.out: Triangle connectivity
    - aux3d.out: Vertical levels (optional)
    - mesh.diag.nc or fesom.mesh.diag.nc: Area data (optional)

    Parameters
    ----------
    mesh_dir : Path
        Path to mesh directory.
    use_dask : bool, optional
        Whether to use dask arrays.

    Returns
    -------
    xr.Dataset
        Standardized mesh dataset.
    """
    # --- Load node coordinates ---
    nod2d_file = mesh_dir / "nod2d.out"
    if not nod2d_file.exists():
        # Try netCDF fallback
        nc_file = mesh_dir / "fesom.mesh.diag.nc"
        if nc_file.exists():
            return _load_from_netcdf(nc_file, use_dask=use_dask)
        raise FileNotFoundError(
            f"Could not find mesh files in {mesh_dir}. "
            "Expected nod2d.out or fesom.mesh.diag.nc"
        )

    with open(nod2d_file) as f:
        n2d = int(f.readline().strip())
        data = np.loadtxt(f, usecols=(1, 2))

    lon_data = data[:, 0].astype(np.float64)
    lat_data = data[:, 1].astype(np.float64)

    # Normalize longitude
    lon_data = normalize_lon(lon_data, "pm180")

    use_dask_actual = should_use_dask(n2d, use_dask)

    if use_dask_actual:
        import dask.array as da
        lon_data = da.from_array(lon_data, chunks=-1)
        lat_data = da.from_array(lat_data, chunks=-1)

    ds = xr.Dataset()

    ds["lon"] = xr.DataArray(
        lon_data,
        dims=("npoints",),
        attrs={
            "units": "degrees_east",
            "long_name": "Longitude",
            "standard_name": "longitude",
        },
    )
    ds["lat"] = xr.DataArray(
        lat_data,
        dims=("npoints",),
        attrs={
            "units": "degrees_north",
            "long_name": "Latitude",
            "standard_name": "latitude",
        },
    )

    # --- Load triangles ---
    elem_file = mesh_dir / "elem2d.out"
    tri_data = None
    if elem_file.exists():
        with open(elem_file) as f:
            n_elem = int(f.readline().strip())
            elem_data = np.loadtxt(f, dtype=np.int32)

        # Convert from 1-indexed to 0-indexed
        tri_data = elem_data[:, :3] - 1

        ds["triangles"] = xr.DataArray(
            tri_data,
            dims=("nelem", "three"),
            attrs={
                "long_name": "Triangle connectivity (0-indexed)",
                "cf_role": "face_node_connectivity",
                "start_index": 0,
            },
        )

        # Compute element centers
        lon_np = lon_data if not use_dask_actual else lon_data.compute()
        lat_np = lat_data if not use_dask_actual else lat_data.compute()
        lon_tri, lat_tri = compute_element_centers(lon_np, lat_np, tri_data)

        ds["lon_tri"] = xr.DataArray(
            lon_tri,
            dims=("nelem",),
            attrs={
                "units": "degrees_east",
                "long_name": "Element center longitude",
            },
        )
        ds["lat_tri"] = xr.DataArray(
            lat_tri,
            dims=("nelem",),
            attrs={
                "units": "degrees_north",
                "long_name": "Element center latitude",
            },
        )

    # --- Load area ---
    area_data = None
    for nc_name in ["mesh.diag.nc", "fesom.mesh.diag.nc"]:
        nc_file = mesh_dir / nc_name
        if nc_file.exists():
            import netCDF4 as nc
            with nc.Dataset(nc_file) as ncds:
                for var_name in ["cluster_area", "nod_area", "area"]:
                    if var_name in ncds.variables:
                        area_raw = np.array(ncds.variables[var_name][:])
                        # nod_area may have shape (nz, nod2) - use surface level
                        if area_raw.ndim == 2:
                            area_data = area_raw[0, :]  # Surface level
                        else:
                            area_data = area_raw
                        break
            if area_data is not None:
                break

    if area_data is None and tri_data is not None:
        # Compute from triangles
        lon_np = ds["lon"].values if not use_dask_actual else ds["lon"].compute().values
        lat_np = ds["lat"].values if not use_dask_actual else ds["lat"].compute().values
        area_data = _compute_cluster_area(lon_np, lat_np, tri_data)

    if area_data is None:
        # Rough approximation
        earth_area = 4 * np.pi * EARTH_RADIUS**2
        area_data = np.full(n2d, earth_area / n2d)

    if use_dask_actual:
        import dask.array as da
        area_data = da.from_array(area_data, chunks=-1)

    ds["area"] = xr.DataArray(
        area_data,
        dims=("npoints",),
        attrs={
            "units": "m2",
            "long_name": "Node cluster area",
        },
    )

    # --- Load vertical levels ---
    aux3d_file = mesh_dir / "aux3d.out"
    if aux3d_file.exists():
        with open(aux3d_file) as f:
            nlev = int(f.readline().strip())
            depth_interfaces = np.array([float(f.readline().strip()) for _ in range(nlev)])

        # Layer centers
        depth_centers = 0.5 * (depth_interfaces[:-1] + depth_interfaces[1:])
        nlevels = len(depth_centers)

        ds["depth"] = xr.DataArray(
            depth_centers,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Depth of layer centers",
                "positive": "down",
            },
        )

        # Depth bounds
        depth_bounds = np.column_stack([
            depth_interfaces[:-1],
            depth_interfaces[1:],
        ])
        ds["depth_bounds"] = xr.DataArray(
            depth_bounds,
            dims=("depth_level", "nv"),
            attrs={
                "units": "m",
                "long_name": "Layer depth bounds",
            },
        )

        # Layer thickness
        layer_thickness = np.diff(depth_interfaces)
        ds["layer_thickness"] = xr.DataArray(
            layer_thickness,
            dims=("depth_level",),
            attrs={
                "units": "m",
                "long_name": "Layer thickness",
            },
        )

    return add_mesh_metadata(ds, "fesom", mesh_dir, use_dask=use_dask_actual)


def _compute_cluster_area(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
    triangles: NDArray[np.integer],
) -> NDArray[np.floating]:
    """Compute cluster area from triangles.

    Distributes 1/3 of each triangle's area to its vertices.

    Parameters
    ----------
    lon : array_like
        Node longitudes in degrees.
    lat : array_like
        Node latitudes in degrees.
    triangles : array_like
        Triangle connectivity (nelem, 3), 0-indexed.

    Returns
    -------
    ndarray
        Cluster area for each node in m^2.
    """
    n2d = len(lon)
    area = np.zeros(n2d, dtype=np.float64)

    for tri in triangles:
        tri_area = _compute_triangle_area(lon[tri], lat[tri])
        area[tri] += tri_area / 3

    return area


def _compute_triangle_area(
    lon: NDArray[np.floating],
    lat: NDArray[np.floating],
) -> float:
    """Compute approximate area of spherical triangle.

    Parameters
    ----------
    lon : array_like
        Longitude of 3 vertices in degrees.
    lat : array_like
        Latitude of 3 vertices in degrees.

    Returns
    -------
    float
        Triangle area in m^2.
    """
    # Convert to radians
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    # Convert to Cartesian on unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # Edge vectors
    v1 = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
    v2 = np.array([x[2] - x[0], y[2] - y[0], z[2] - z[0]])

    # Cross product magnitude gives 2 * area on unit sphere
    cross = np.cross(v1, v2)
    area = 0.5 * np.linalg.norm(cross) * EARTH_RADIUS**2

    return float(area)


def open_dataset(
    data_path: str | os.PathLike,
    mesh: xr.Dataset | None = None,
    mesh_path: str | os.PathLike | None = None,
) -> xr.Dataset:
    """Open a FESOM2 data file with mesh information.

    Parameters
    ----------
    data_path : str or path-like
        Path to the data file (NetCDF).
    mesh : xr.Dataset, optional
        Pre-loaded mesh dataset. If not provided, mesh_path must be specified.
    mesh_path : str or path-like, optional
        Path to mesh directory. Ignored if mesh is provided.

    Returns
    -------
    xr.Dataset
        Dataset with mesh coordinates attached.

    Examples
    --------
    >>> mesh = nr.fesom.load_mesh("/meshes/core2")
    >>> ds = nr.fesom.open_dataset("temp.fesom.2010.nc", mesh=mesh)
    >>> ds = nr.fesom.open_dataset("temp.fesom.2010.nc", mesh_path="/meshes/core2")
    """
    # Load mesh if not provided
    if mesh is None:
        if mesh_path is None:
            raise ValueError("Either mesh or mesh_path must be provided")
        mesh = load_mesh(mesh_path)

    # Open dataset
    ds = xr.open_dataset(data_path)

    # Get coordinate arrays
    lon_data = mesh["lon"].values
    lat_data = mesh["lat"].values

    # Add mesh coordinates based on dimension names
    if "nod2" in ds.dims:
        ds = ds.assign_coords(
            lon=("nod2", lon_data),
            lat=("nod2", lat_data),
        )
    elif "nodes_2d" in ds.dims:
        ds = ds.assign_coords(
            lon=("nodes_2d", lon_data),
            lat=("nodes_2d", lat_data),
        )
    elif "npoints" in ds.dims:
        ds = ds.assign_coords(
            lon=("npoints", lon_data),
            lat=("npoints", lat_data),
        )

    # Add depth coordinates if applicable
    if "depth" in mesh:
        depth_data = mesh["depth"].values
        nlev = len(depth_data)
        if "nz" in ds.dims and ds.sizes["nz"] == nlev:
            ds = ds.assign_coords(depth=("nz", depth_data))
        elif "nz1" in ds.dims and ds.sizes["nz1"] == nlev:
            ds = ds.assign_coords(depth=("nz1", depth_data))

    return ds
