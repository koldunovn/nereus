"""MITgcm diagnostic data loading.

Reads MITgcm MDS diagnostic output files and produces
``xr.Dataset`` objects with proper variable names, time coordinates,
and spatial dimensions matching the nereus mesh convention.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nereus.models.mitgcm.io import find_iterations, read_mds


def open_dataset(
    path: str | os.PathLike,
    prefix: str | list[str],
    iters: str | list[int] = "all",
    *,
    delta_t: float = 1.0,
    ref_date: str | None = None,
    mesh: xr.Dataset | None = None,
    mask_land: bool = False,
) -> xr.Dataset:
    """Load MITgcm diagnostic output as an xarray Dataset.

    Parameters
    ----------
    path : str or Path
        Directory containing MDS diagnostic files.
    prefix : str or list of str
        Diagnostic file prefix(es) (e.g., ``"diags2D"`` or
        ``["diags2D", "diags3D"]``).
    iters : "all" or list of int
        Iteration numbers to load. ``"all"`` discovers all available.
    delta_t : float
        Model timestep in seconds (used to compute time coordinate).
    ref_date : str, optional
        Reference date string (e.g., ``"1710-1-1"``). If given, time
        coordinate is ``datetime64``; otherwise seconds from start.
    mesh : xr.Dataset, optional
        Nereus mesh dataset. If provided, attaches ``lon``, ``lat``
        (and ``depth`` if available) as coordinates.
    mask_land : bool
        If True, replace land-point values with NaN. Uses ``hFacC``
        from mesh (3D per-level masking) or ``land_mask`` (2D surface
        masking). Requires ``mesh`` loaded with ``mask_land=True``.
        If mesh has no mask variables, this option is silently ignored.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(time, npoints)`` for 2D fields
        and ``(time, depth_level, npoints)`` for 3D fields.

    Examples
    --------
    >>> ds = nr.mitgcm.open_dataset(
    ...     "/path/to/run/", prefix="diags2D",
    ...     delta_t=3600, ref_date="1710-1-1",
    ... )
    """
    path = Path(path)

    if isinstance(prefix, str):
        prefix = [prefix]

    all_vars: dict[str, list[np.ndarray]] = {}
    time_values: list | None = None

    for pfx in prefix:
        # Discover iterations
        if iters == "all":
            iter_list = find_iterations(path, pfx)
        else:
            iter_list = list(iters)

        if not iter_list:
            continue

        # Build time coordinate from first prefix
        if time_values is None:
            time_seconds = np.array(iter_list, dtype=np.float64) * delta_t
            if ref_date is not None:
                ref = pd.Timestamp(ref_date)
                time_values = [
                    ref + pd.Timedelta(seconds=float(s)) for s in time_seconds
                ]
            else:
                time_values = time_seconds

        for it in iter_list:
            meta, data = read_mds(path / pfx, iteration=it)

            fld_list = meta.get("fldList", [pfx])
            nflds = meta.get("nFlds", 1)
            missing = meta.get("missingValue", None)

            for fi in range(nflds):
                name = fld_list[fi].strip() if fi < len(fld_list) else f"{pfx}_{fi}"

                if nflds > 1:
                    field = data[fi].astype(np.float64)
                else:
                    field = data.astype(np.float64)

                # Replace missing values with NaN
                if missing is not None:
                    field = np.where(np.isclose(field, missing), np.nan, field)

                # Flatten spatial dimensions to 1D
                if field.ndim == 2:
                    # 2D field (ny, nx) -> (npoints,)
                    field = field.ravel()
                elif field.ndim == 3:
                    # 3D field (nz, ny, nx) -> (nz, npoints)
                    nz = field.shape[0]
                    field = field.reshape(nz, -1)

                if name not in all_vars:
                    all_vars[name] = []
                all_vars[name].append(field)

    if time_values is None:
        raise ValueError(f"No iterations found for prefixes: {prefix}")

    # Build dataset
    data_vars: dict = {}
    for name, arrays in all_vars.items():
        stacked = np.stack(arrays, axis=0)

        if stacked.ndim == 2:
            # (time, npoints)
            data_vars[name] = (("time", "npoints"), stacked)
        elif stacked.ndim == 3:
            # (time, depth_level, npoints)
            data_vars[name] = (("time", "depth_level", "npoints"), stacked)

    coords: dict = {"time": time_values}

    if mesh is not None:
        coords["lon"] = (("npoints",), mesh["lon"].values)
        coords["lat"] = (("npoints",), mesh["lat"].values)
        if "depth" in mesh:
            coords["depth"] = (("depth_level",), mesh["depth"].values)

    ds = xr.Dataset(data_vars, coords=coords)

    # Apply land masking
    if mask_land and mesh is not None:
        if "hFacC" in mesh:
            # 3D masking: hFacC has shape (depth_level, npoints)
            hfac = mesh["hFacC"].values
            land_3d = hfac == 0  # True where land
            for name in list(ds.data_vars):
                if ds[name].dims == ("time", "depth_level", "npoints"):
                    ds[name] = ds[name].where(~land_3d)
                elif ds[name].dims == ("time", "npoints"):
                    # Use surface mask for 2D fields
                    ds[name] = ds[name].where(~land_3d[0])
        elif "land_mask" in mesh:
            # 2D-only masking
            land_2d = mesh["land_mask"].values  # True where land
            for name in list(ds.data_vars):
                if ds[name].dims[-1] == "npoints":
                    ds[name] = ds[name].where(~land_2d)

    return ds
