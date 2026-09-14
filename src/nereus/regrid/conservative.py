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

Points sitting exactly at a pole are a special case: every longitude at
lat=+-90 maps to the same Cartesian point, so a pole row pulled from a
regular lat/lon source (e.g. reanalysis data enumerates every longitude at
lat=90) collapses to duplicate Voronoi generators. These are detected and
merged before tessellating, then the resulting single pole cell is split
back into one wedge per original point (see ``build_voronoi_cells``).

Known limitations: cells whose true (geodesic) boundary would enclose a
pole without any source point sitting there are not handled exactly -- the
planar polygon can misrepresent such a cell's true shape.
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


_POLE_LAT_EPS = 1e-6  # degrees; points this close to +-90 are treated as the pole


def _interp_boundary_lat(
    angle: float, b_angle: NDArray[np.floating], b_lat: NDArray[np.floating]
) -> float:
    """Circularly interpolate a Voronoi cell boundary's latitude at `angle`.

    ``b_angle`` must already be sorted ascending within ``[0, 360)``.
    """
    n = len(b_angle)
    j = int(np.searchsorted(b_angle, angle, side="right")) % n
    j_prev = (j - 1) % n
    a0, a1 = b_angle[j_prev], b_angle[j]
    span = (a1 - a0) % 360.0
    span = 360.0 if span == 0.0 else span
    t = ((angle - a0) % 360.0) / span
    return float(b_lat[j_prev] + t * (b_lat[j] - b_lat[j_prev]))


def _split_pole_cell(
    boundary_lon: NDArray[np.floating],
    boundary_lat: NDArray[np.floating],
    group_lon: NDArray[np.floating],
    pole_lat: float,
) -> list[BaseGeometry]:
    """Split a pole-enclosing Voronoi cell into one wedge per owning point.

    Used in two situations: (1) multiple points sit exactly at the same
    pole and so collapsed onto a single Voronoi generator (``group_lon``
    has more than one entry), or (2) a single point's cell happens to
    enclose a pole without sitting there itself (``group_lon`` has one
    entry) -- in the plain (lon, lat) plane, a cell containing a pole
    cannot be closed into a simple polygon without cutting it along a
    lat=+-90 edge, which is what this function does. ``group_lon`` records
    each owning point's own longitude -- a well-defined azimuth around the
    pole even where the points themselves coincide there (or don't reach
    it at all). Wedges are cut at the angles midway between each pair of
    (circularly) consecutive points, so the merged cell's area is
    partitioned exactly rather than being copied whole to every point.

    Parameters
    ----------
    boundary_lon, boundary_lat : array_like
        Vertices of the pole-enclosing Voronoi cell, in degrees.
    group_lon : array_like
        Longitude(s) of the point(s) owning this cell.
    pole_lat : float
        The true pole latitude this cell encloses (+90.0 or -90.0).

    Returns
    -------
    list of shapely geometry
        One wedge polygon per group member, in the same order as
        ``group_lon``.
    """
    n_group = len(group_lon)

    b_angle = np.mod(boundary_lon, 360.0)
    order = np.argsort(b_angle)
    b_angle = b_angle[order]
    b_lat = boundary_lat[order]

    g_order = np.argsort(np.mod(group_lon, 360.0))
    g_angle = np.mod(group_lon, 360.0)[g_order]

    span = np.mod(np.roll(g_angle, -1) - g_angle, 360.0)
    span = np.where(span == 0.0, 360.0, span)
    cut_after = np.mod(g_angle + span / 2.0, 360.0)
    cut_before = np.roll(cut_after, 1)

    sub_polys: list[BaseGeometry] = []
    for k in range(n_group):
        lo = float(cut_before[k])
        rel_span = (float(cut_after[k]) - lo) % 360.0
        rel_span = 360.0 if rel_span == 0.0 else rel_span
        hi = lo + rel_span  # unwrapped: hi - lo == rel_span exactly, never 0

        # Boundary points strictly inside this wedge, re-expressed on the
        # same unwrapped (lo, hi) scale as lo/hi themselves -- this is what
        # keeps the ring numerically monotonic (no dateline-style wrap)
        # regardless of how wide the wedge is, including the full-circle
        # (rel_span == 360) case where a single point owns the whole cap.
        b_rel = (b_angle - lo) % 360.0
        inside = (b_rel > 0.0) & (b_rel < rel_span)
        order_inside = np.argsort(b_rel[inside])
        arc_lon = lo + b_rel[inside][order_inside]
        arc_lat = b_lat[inside][order_inside]

        lat_lo = _interp_boundary_lat(lo % 360.0, b_angle, b_lat)
        lat_hi = _interp_boundary_lat(hi % 360.0, b_angle, b_lat)

        # Ring: down the left radial edge (pole -> boundary at lo), across
        # the boundary arc, up the right radial edge (boundary at hi ->
        # pole), then closed by the implicit top edge back from (hi,
        # pole_lat) to (lo, pole_lat) -- the standard equirectangular
        # representation of a wedge of a polar cap.
        ring_lon = np.concatenate([[lo, lo], arc_lon, [hi, hi]])
        ring_lat = np.concatenate([[pole_lat, lat_lo], arc_lat, [lat_hi, pole_lat]])

        poly = Polygon(np.column_stack([ring_lon, ring_lat]))
        if not poly.is_valid:
            poly = poly.buffer(0)
        sub_polys.append(poly)

    # sub_polys is ordered by ascending angle (g_order); map back to each
    # member's original position within the group.
    out: list[BaseGeometry] = [sub_polys[0]] * n_group
    for k, orig_pos in enumerate(g_order):
        out[orig_pos] = sub_polys[k]
    return out


def build_voronoi_cells(
    lon: NDArray[np.floating], lat: NDArray[np.floating]
) -> list[BaseGeometry]:
    """Build one Voronoi cell polygon per input point.

    Points are tessellated on the unit sphere via
    :class:`scipy.spatial.SphericalVoronoi`, then each cell's vertices are
    converted back to lon/lat and normalized around that cell's own point
    (via :func:`nereus.core.coordinates.normalize_longitude`) so the
    resulting planar polygon doesn't spuriously wrap around the dateline.

    Points sitting at a pole (within ``1e-6`` degrees of lat=+-90) are
    merged into a single Voronoi generator per pole -- every longitude
    maps to the same Cartesian point there, so ``SphericalVoronoi`` would
    otherwise reject them as duplicates. Separately, whichever single
    generator ends up nearest each pole (which, by the definition of a
    Voronoi tessellation, is always the one whose cell encloses that
    pole -- not necessarily one of the merged points above, e.g. for a
    sparse mesh with no point placed at the pole itself) is also handled
    specially, since such a cell cannot be closed into a simple polygon in
    the plain (lon, lat) plane without an explicit cut at lat=+-90. Both
    cases are handled by splitting/closing the cell via
    :func:`_split_pole_cell`, so pole-adjacent area is neither dropped
    (previously: an invalid, self-intersecting planar polygon collapsing
    to zero area) nor multiply counted (previously: a crash, or -- had it
    not crashed -- duplicated points each claiming the full merged area).

    Parameters
    ----------
    lon, lat : array_like
        Point coordinates in degrees, shape ``(npoints,)``.

    Returns
    -------
    list of shapely geometry
        One polygon per input point, in the same order. Self-intersecting
        planar projections (rare) are repaired with ``buffer(0)`` and may
        become a ``MultiPolygon``.

    Raises
    ------
    ValueError
        If the spherical Voronoi tessellation fails, e.g. because the
        points contain non-pole duplicates or are otherwise degenerate.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)

    at_north = lat >= 90.0 - _POLE_LAT_EPS
    at_south = lat <= -90.0 + _POLE_LAT_EPS

    groups: list[NDArray[np.intp]] = []
    generator_lon: list[float] = []
    generator_lat: list[float] = []

    for i in np.nonzero(~(at_north | at_south))[0]:
        groups.append(np.array([i]))
        generator_lon.append(float(lon[i]))
        generator_lat.append(float(lat[i]))

    for pole_mask, fixed_pole_lat in ((at_north, 90.0), (at_south, -90.0)):
        idx = np.nonzero(pole_mask)[0]
        if idx.size == 0:
            continue
        groups.append(idx)
        generator_lon.append(0.0)
        generator_lat.append(fixed_pole_lat)

    generator_lon_arr = np.array(generator_lon)
    generator_lat_arr = np.array(generator_lat)

    xyz = np.column_stack(lonlat_to_cartesian(generator_lon_arr, generator_lat_arr))

    try:
        sv = SphericalVoronoi(xyz, radius=1.0)
    except Exception as exc:  # scipy raises plain ValueError/QhullError
        raise ValueError(
            "Failed to build a spherical Voronoi tessellation from the "
            "source points for conservative remapping. This usually means "
            "the points contain (non-pole) duplicates or are otherwise "
            f"degenerate (e.g. all coplanar). Original error: {exc}"
        ) from exc
    sv.sort_vertices_of_regions()

    # The generator nearest each pole is, by definition, the one whose
    # Voronoi cell encloses it -- regardless of whether any point sits
    # exactly at the pole.
    north_owner = int(np.argmax(xyz[:, 2]))
    south_owner = int(np.argmin(xyz[:, 2]))

    polygons: list[BaseGeometry | None] = [None] * n
    for gidx, (g, region) in enumerate(zip(groups, sv.regions, strict=True)):
        verts = sv.vertices[region]
        vlon, vlat = cartesian_to_lonlat(verts[:, 0], verts[:, 1], verts[:, 2])

        if gidx == north_owner:
            pole_lat: float | None = 90.0
        elif gidx == south_owner:
            pole_lat = -90.0
        else:
            pole_lat = None

        if pole_lat is None and g.size == 1:
            i = g[0]
            vlon_i = normalize_longitude(vlon, lon[i])
            poly = Polygon(np.column_stack([vlon_i, vlat]))
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons[i] = poly
        else:
            # Either multiple points collapsed onto this generator (an
            # exact pole duplicate group), or this is the single closest
            # point to a pole its cell encloses -- both need the cell
            # split/closed via the pole edge rather than the plain
            # site-centered construction above.
            if pole_lat is None:
                pole_lat = float(lat[g[0]])
            wedges = _split_pole_cell(vlon, vlat, lon[g], pole_lat)
            for i, wedge in zip(g, wedges, strict=True):
                polygons[i] = wedge

    assert all(p is not None for p in polygons)
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
