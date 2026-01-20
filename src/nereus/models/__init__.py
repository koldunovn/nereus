"""Model-specific modules for nereus.

This module provides model-specific functionality for various climate models.

Available submodules:
- fesom: FESOM2 ocean model support
- icono: ICON-Ocean model support (stub)
- icona: ICON-Atmosphere model support (stub)
- ifs: IFS model support (stub)
- healpix: HEALPix grid support (stub)
"""

from nereus.models import fesom, healpix, icona, icono, ifs

__all__ = [
    "fesom",
    "icono",
    "icona",
    "ifs",
    "healpix",
]
