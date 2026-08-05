"""Geometry helpers for area-conservative remapping.

This module builds the pieces needed for first-order conservative
regridding: source/target cell polygons and a sparse matrix of
area-weighted overlaps between them.

Source (and, for mesh-to-mesh remapping, target) cell polygons are derived
generically from a point cloud via a spherical Voronoi tessellation, so the
same code works for FESOM nodes, HEALPix pixels, or any other unstructured
mesh without needing mesh-specific connectivity. Overlap areas are computed
by intersecting polygons in the lon/lat plane (consistent with how the
``linear``/``cubic`` methods in :mod:`nereus.regrid.interpolator` already
triangulate in that plane) and then measuring the *spherical* area of the
resulting polygon, so weights remain physically meaningful even though the
polygon boundaries themselves are a planar approximation of true geodesics.

Known limitations: cells whose true (geodesic) boundary would enclose a
pole, or Voronoi cells built from a source point very close to a pole, are
not handled exactly -- the planar polygon can misrepresent such a cell's
true shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.spatial import SphericalVoronoi
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from nereus.core.coordinates import (
    EARTH_RADIUS,
    cartesian_to_lonlat,
    lonlat_to_cartesian,
    normalize_longitude,
)


def spherical_polygon_area(
    lon: NDArray[np.floating], lat: NDArray[np.floating]
) -> float:
    """Compute the spherical area enclosed by a lon/lat polygon.

    Fan-triangulates the polygon from its first vertex and sums the area
    of each spherical triangle (same unit-sphere cross-product approach
    used by ``models/fesom/mesh.py::_compute_triangle_area``).

    Parameters
    ----------
    lon, lat : array_like
        Polygon vertices in degrees. A closing vertex equal to the first
        one is tolerated but not required.

    Returns
    -------
    float
        Area in square meters. Returns 0.0 for degenerate polygons
        (fewer than 3 distinct vertices).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if len(lon) >= 2 and lon[0] == lon[-1] and lat[0] == lat[-1]:
        lon = lon[:-1]
        lat = lat[:-1]

    n = len(lon)
    if n < 3:
        return 0.0

    x, y, z = lonlat_to_cartesian(lon, lat)
    x0, y0, z0 = x[0], y[0], z[0]

    v1 = np.column_stack([x[1:-1] - x0, y[1:-1] - y0, z[1:-1] - z0])
    v2 = np.column_stack([x[2:] - x0, y[2:] - y0, z[2:] - z0])
    cross = np.cross(v1, v2)
    area = 0.5 * np.sum(np.linalg.norm(cross, axis=-1))

    return float(area * EARTH_RADIUS**2)


def geometry_spherical_area(geom: BaseGeometry) -> float:
    """Compute the spherical area of a shapely geometry.

    Handles ``Polygon``, ``MultiPolygon``, and ``GeometryCollection``
    (as can result from intersecting two polygons). Non-areal geometries
    (points, lines) contribute zero area.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Geometry, with coordinates in lon/lat degrees.

    Returns
    -------
    float
        Area in square meters.
    """
    if geom.is_empty:
        return 0.0
    if geom.geom_type == "Polygon":
        coords = np.asarray(geom.exterior.coords)
        return spherical_polygon_area(coords[:, 0], coords[:, 1])
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return sum(geometry_spherical_area(part) for part in geom.geoms)
    return 0.0


def build_voronoi_cells(
    lon: NDArray[np.floating], lat: NDArray[np.floating]
) -> list[BaseGeometry]:
    """Build one Voronoi cell polygon per input point.

    Points are tessellated on the unit sphere via
    :class:`scipy.spatial.SphericalVoronoi`, then each cell's vertices are
    converted back to lon/lat and normalized around that cell's own point
    (via :func:`nereus.core.coordinates.normalize_longitude`) so the
    resulting planar polygon doesn't spuriously wrap around the dateline.

    Parameters
    ----------
    lon, lat : array_like
        Point coordinates in degrees, shape ``(npoints,)``.

    Returns
    -------
    list of shapely geometry
        One polygon per input point, in the same order. Self-intersecting
        planar projections (rare, typically near poles) are repaired with
        ``buffer(0)`` and may become a ``MultiPolygon``.

    Raises
    ------
    ValueError
        If the spherical Voronoi tessellation fails, e.g. because the
        points contain exact duplicates.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    xyz = np.column_stack(lonlat_to_cartesian(lon, lat))

    try:
        sv = SphericalVoronoi(xyz, radius=1.0)
    except Exception as exc:  # scipy raises plain ValueError/QhullError
        raise ValueError(
            "Failed to build a spherical Voronoi tessellation from the "
            "source points for conservative remapping. This usually means "
            "the points contain exact duplicates or are otherwise "
            f"degenerate (e.g. all coplanar). Original error: {exc}"
        ) from exc
    sv.sort_vertices_of_regions()

    polygons: list[BaseGeometry] = []
    for i, region in enumerate(sv.regions):
        verts = sv.vertices[region]
        vlon, vlat = cartesian_to_lonlat(verts[:, 0], verts[:, 1], verts[:, 2])
        vlon = normalize_longitude(vlon, lon[i])
        poly = Polygon(np.column_stack([vlon, vlat]))
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons.append(poly)

    return polygons


def build_grid_box_cells(
    resolution: float | tuple[int, int],
    lon_bounds: tuple[float, float] = (-180.0, 180.0),
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
) -> list[Polygon]:
    """Build axis-aligned lon/lat box polygons for a regular target grid.

    Cell ordering matches
    :func:`nereus.core.grids.create_regular_grid`'s ``(nlat, nlon)``
    meshgrid, raveled in C order, so the returned list lines up with a
    raveled ``target_lon``/``target_lat`` grid of the same resolution and
    bounds.

    Parameters
    ----------
    resolution : float or tuple of int
        Grid resolution, same semantics as ``create_regular_grid``.
    lon_bounds, lat_bounds : tuple of float
        Grid bounds in degrees.

    Returns
    -------
    list of shapely.geometry.Polygon
        One box per target grid cell.
    """
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    if isinstance(resolution, (list, tuple)):
        nlon, nlat = resolution
    else:
        nlon = int((lon_max - lon_min) / resolution)
        nlat = int((lat_max - lat_min) / resolution)

    lon_edges = np.linspace(lon_min, lon_max, nlon + 1)
    lat_edges = np.linspace(lat_min, lat_max, nlat + 1)

    return [
        box(lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
        for j in range(nlat)
        for i in range(nlon)
    ]


def build_conservative_weights(
    source_polys: list[BaseGeometry],
    target_polys: list[BaseGeometry],
) -> csr_matrix:
    """Compute a sparse ``(n_target, n_source)`` area-overlap weight matrix.

    ``W[j, i]`` is the spherical area (m^2) of the overlap between source
    cell ``i`` and target cell ``j``. Source polygons are queried via an
    ``STRtree`` for speed, and are additionally tested with copies shifted
    by +/-360 degrees in longitude so that overlaps spanning the dateline
    (common for global unstructured ocean/atmosphere meshes) are still
    found regardless of which longitude branch a given cell happens to sit
    on.

    Parameters
    ----------
    source_polys : list of shapely geometry
        Source cell polygons, e.g. from :func:`build_voronoi_cells`.
    target_polys : list of shapely geometry
        Target cell polygons, e.g. from :func:`build_grid_box_cells` or
        :func:`build_voronoi_cells`.

    Returns
    -------
    scipy.sparse.csr_matrix
        Overlap-area weight matrix of shape ``(len(target_polys),
        len(source_polys))``.
    """
    n_source = len(source_polys)
    n_target = len(target_polys)

    padded = (
        list(source_polys)
        + [affinity.translate(p, xoff=-360.0) for p in source_polys]
        + [affinity.translate(p, xoff=360.0) for p in source_polys]
    )
    tree = STRtree(padded)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j, tpoly in enumerate(target_polys):
        if tpoly is None or tpoly.is_empty:
            continue
        for padded_idx in tree.query(tpoly):
            i = int(padded_idx) % n_source
            spoly = padded[int(padded_idx)]
            if spoly.is_empty or not spoly.is_valid or not spoly.intersects(tpoly):
                continue
            try:
                overlap = spoly.intersection(tpoly)
            except Exception:
                # Rare GEOS topology failures on degenerate polygons; skip
                # this source cell rather than aborting the whole regrid.
                continue
            area = geometry_spherical_area(overlap)
            if area <= 0.0:
                continue
            rows.append(j)
            cols.append(i)
            data.append(area)

    return csr_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(n_target, n_source),
    )
