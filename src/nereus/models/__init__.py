"""Model-specific modules for nereus.

This module provides model-specific functionality for various climate models.

Available submodules:
- fesom: FESOM2 ocean model support
- healpix: HEALPix grid support
- nemo: NEMO ocean model support
- icono: ICON-Ocean model support (stub)
- icona: ICON-Atmosphere model support (stub)
- ifs: IFS model support (stub)

Universal loader:
- load_mesh: Auto-detect mesh type and load

Examples
--------
>>> import nereus as nr

# Model-specific loading
>>> mesh = nr.fesom.load_mesh("/path/to/mesh")
>>> mesh = nr.healpix.load_mesh(3145728)
>>> mesh = nr.nemo.load_mesh("/path/to/mesh_mask.nc")

# Universal loader with auto-detection
>>> mesh = nr.load_mesh("/path/to/mesh")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nereus.models import fesom, healpix, icona, icono, ifs, nemo

if TYPE_CHECKING:
    import xarray as xr


def detect_mesh_type(path: str | os.PathLike) -> str:
    """Auto-detect mesh type from path.

    Parameters
    ----------
    path : str or Path
        Path to mesh directory or file.

    Returns
    -------
    str
        Detected mesh type: "fesom", "nemo", or "unknown".

    Notes
    -----
    HEALPix meshes are generated, not loaded from files,
    so this function only detects file-based mesh types.
    """
    path = Path(path)

    # Check for FESOM indicators
    if path.is_dir():
        # Check for FESOM ASCII files
        if (path / "nod2d.out").exists() or (path / "elem2d.out").exists():
            return "fesom"
        # Check for FESOM netCDF
        if (path / "fesom.mesh.diag.nc").exists():
            return "fesom"

    if path.is_file():
        suffix = path.suffix.lower()
        name = path.name.lower()

        if suffix in (".nc", ".nc4"):
            # Try to detect from filename
            if "fesom" in name or "mesh.diag" in name:
                return "fesom"
            if "mesh_mask" in name or "coordinates" in name:
                return "nemo"

            # Try to peek at variables
            try:
                import xarray as xr

                with xr.open_dataset(path) as ds:
                    # FESOM indicators
                    if "face_nodes" in ds or "nod_area" in ds:
                        return "fesom"
                    # NEMO indicators
                    if "tmask" in ds or "glamt" in ds or "nav_lon" in ds:
                        return "nemo"
            except Exception:
                pass

    return "unknown"


def load_mesh(
    path: str | os.PathLike | int,
    *,
    mesh_type: Literal["fesom", "healpix", "nemo", "auto"] | None = None,
    use_dask: bool | None = None,
    **kwargs,
) -> "xr.Dataset":
    """Universal mesh loader with auto-detection.

    Parameters
    ----------
    path : str, Path, or int
        Path to mesh directory/file, or number of points for HEALPix.
    mesh_type : str, optional
        Mesh type: "fesom", "healpix", "nemo", or "auto".
        If None or "auto", attempts to auto-detect.
    use_dask : bool, optional
        Whether to use dask arrays. Auto-detects if None.
    **kwargs
        Additional arguments passed to model-specific loader.

    Returns
    -------
    xr.Dataset
        Standardized mesh dataset.

    Examples
    --------
    >>> # Auto-detect FESOM mesh
    >>> mesh = nr.load_mesh("/path/to/mesh")

    >>> # Explicit type
    >>> mesh = nr.load_mesh("/path/to/mesh_mask.nc", mesh_type="nemo")

    >>> # HEALPix from npoints
    >>> mesh = nr.load_mesh(3145728, mesh_type="healpix")
    """
    # Handle integer input (HEALPix)
    if isinstance(path, int):
        if mesh_type is None:
            mesh_type = "healpix"
        if mesh_type != "healpix":
            raise ValueError(f"Integer path only valid for HEALPix, got mesh_type={mesh_type}")
        return healpix.load_mesh(path, use_dask=use_dask, **kwargs)

    path = Path(path)

    # Auto-detect mesh type
    if mesh_type is None or mesh_type == "auto":
        mesh_type = detect_mesh_type(path)
        if mesh_type == "unknown":
            raise ValueError(
                f"Could not auto-detect mesh type for {path}. "
                "Please specify mesh_type explicitly."
            )

    # Load based on type
    if mesh_type == "fesom":
        return fesom.load_mesh(path, use_dask=use_dask, **kwargs)
    elif mesh_type == "nemo":
        return nemo.load_mesh(path, use_dask=use_dask, **kwargs)
    elif mesh_type == "healpix":
        raise ValueError("HEALPix meshes require npoints (int), not a path")
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")


__all__ = [
    "detect_mesh_type",
    "fesom",
    "healpix",
    "icona",
    "icono",
    "ifs",
    "load_mesh",
    "nemo",
]
