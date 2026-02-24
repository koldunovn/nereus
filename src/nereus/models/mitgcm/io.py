"""Low-level I/O for MITgcm MDS (Meta/Data Set) binary files.

MITgcm stores output as paired files:
- ``.meta``: ASCII metadata (dimensions, precision, field names, etc.)
- ``.data``: Raw binary data in big-endian format

This module provides utilities to read these file pairs without
depending on external packages like xmitgcm.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def parse_meta(meta_path: str | Path) -> dict:
    """Parse a MITgcm ``.meta`` file.

    Parameters
    ----------
    meta_path : str or Path
        Path to the ``.meta`` file.

    Returns
    -------
    dict
        Parsed metadata with keys like ``nDims``, ``dimList``,
        ``dataprec``, ``nrecords``, ``fldList``, ``timeStepNumber``,
        ``timeInterval``, ``missingValue``.
    """
    meta_path = Path(meta_path)
    text = meta_path.read_text()

    meta: dict = {}

    # nDims = [ 2 ];
    m = re.search(r"nDims\s*=\s*\[\s*(\d+)\s*\]", text)
    if m:
        meta["nDims"] = int(m.group(1))

    # dimList = [ 90, 1, 90, 1170, 1, 1170 ];
    m = re.search(r"dimList\s*=\s*\[([\s\d,]+)\]", text)
    if m:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        # Stored as triples: (size, start, end) for each dimension
        meta["dimList"] = [nums[i : i + 3] for i in range(0, len(nums), 3)]

    # dataprec = [ 'float32' ];
    m = re.search(r"dataprec\s*=\s*\[\s*'(\w+)'\s*\]", text)
    if m:
        meta["dataprec"] = m.group(1)

    # nrecords = [ 4 ];
    m = re.search(r"nrecords\s*=\s*\[\s*(\d+)\s*\]", text)
    if m:
        meta["nrecords"] = int(m.group(1))

    # timeStepNumber = [ 8760 ];
    m = re.search(r"timeStepNumber\s*=\s*\[\s*(\d+)\s*\]", text)
    if m:
        meta["timeStepNumber"] = int(m.group(1))

    # timeInterval = [ 0.000E+00 3.154E+07 ];
    m = re.search(r"timeInterval\s*=\s*\[([\s\dEe.+-]+)\]", text)
    if m:
        meta["timeInterval"] = [float(x) for x in m.group(1).split()]

    # missingValue = [ -9.99000000000000E+02 ];
    m = re.search(r"missingValue\s*=\s*\[([\s\dEe.+-]+)\]", text)
    if m:
        meta["missingValue"] = float(m.group(1).strip())

    # nFlds = [ 4 ];
    m = re.search(r"nFlds\s*=\s*\[\s*(\d+)\s*\]", text)
    if m:
        meta["nFlds"] = int(m.group(1))

    # fldList = { 'ETAN    ' 'MXLDEPTH' ... };
    m = re.search(r"fldList\s*=\s*\{([^}]+)\}", text)
    if m:
        meta["fldList"] = [s.strip() for s in re.findall(r"'([^']+)'", m.group(1))]

    return meta


def get_shape_from_meta(meta: dict) -> tuple[int, ...]:
    """Extract C-order array shape from parsed metadata.

    MITgcm stores dimensions in Fortran order (fastest-varying first).
    This function reverses them to C order.

    Parameters
    ----------
    meta : dict
        Parsed metadata from :func:`parse_meta`.

    Returns
    -------
    tuple of int
        Array shape in C order (e.g., ``(1170, 90)`` for a 2D field).
    """
    dim_list = meta["dimList"]
    # dimList entries are [size, start, end] in Fortran order
    # Reverse to get C order
    return tuple(d[0] for d in reversed(dim_list))


def get_dtype_from_meta(meta: dict) -> np.dtype:
    """Convert ``dataprec`` string to numpy dtype (big-endian).

    Parameters
    ----------
    meta : dict
        Parsed metadata from :func:`parse_meta`.

    Returns
    -------
    np.dtype
        Big-endian numpy dtype.
    """
    prec = meta.get("dataprec", "float32")
    type_map = {
        "float32": ">f4",
        "float64": ">f8",
    }
    return np.dtype(type_map.get(prec, ">f4"))


def read_mds(
    prefix_path: str | Path,
    iteration: int | None = None,
) -> tuple[dict, NDArray]:
    """Read a MITgcm MDS file pair (``.meta`` + ``.data``).

    Parameters
    ----------
    prefix_path : str or Path
        Path prefix without extension. For grid files like ``XC``,
        this is the full path without ``.meta``/``.data``.
        For time-varying files, include the iteration suffix
        or use the ``iteration`` parameter.
    iteration : int, optional
        If given, appended as ``.{iteration:010d}`` to the prefix.

    Returns
    -------
    meta : dict
        Parsed metadata.
    data : ndarray
        Data array. For multi-field files, shape is
        ``(nFlds, *spatial_shape)``.
    """
    prefix_path = Path(prefix_path)

    if iteration is not None:
        prefix_path = prefix_path.parent / f"{prefix_path.name}.{iteration:010d}"

    meta_file = Path(str(prefix_path) + ".meta")
    data_file = Path(str(prefix_path) + ".data")

    meta = parse_meta(meta_file)
    shape = get_shape_from_meta(meta)
    dtype = get_dtype_from_meta(meta)

    nflds = meta.get("nFlds", 1)
    nrecords = meta.get("nrecords", 1)

    raw = np.fromfile(data_file, dtype=dtype)

    if nflds > 1:
        data = raw.reshape((nrecords, *shape))
    else:
        data = raw.reshape(shape)

    return meta, data


def find_iterations(directory: str | Path, prefix: str) -> list[int]:
    """Discover available iteration numbers for a diagnostic prefix.

    Parameters
    ----------
    directory : str or Path
        Directory containing MDS files.
    prefix : str
        File prefix (e.g., ``"diags2D"``).

    Returns
    -------
    list of int
        Sorted list of iteration numbers.
    """
    directory = Path(directory)
    iters = []
    for meta_file in sorted(directory.glob(f"{prefix}.*.meta")):
        # Extract iteration number from filename like "diags2D.0000008760.meta"
        parts = meta_file.stem.split(".")
        if len(parts) >= 2:
            try:
                iters.append(int(parts[-1]))
            except ValueError:
                continue
    return sorted(iters)
