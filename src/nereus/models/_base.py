"""Base classes and protocols for model meshes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from nereus.core.coordinates import lonlat_to_cartesian

if TYPE_CHECKING:
    pass


class MeshBase(ABC):
    """Abstract base class for model meshes.

    This class defines the interface that all model mesh implementations
    must follow.
    """

    @property
    @abstractmethod
    def lon(self) -> NDArray[np.floating]:
        """Longitude coordinates in degrees."""
        ...

    @property
    @abstractmethod
    def lat(self) -> NDArray[np.floating]:
        """Latitude coordinates in degrees."""
        ...

    @property
    @abstractmethod
    def area(self) -> NDArray[np.floating]:
        """Cell area in square meters."""
        ...

    @property
    def npoints(self) -> int:
        """Number of mesh points."""
        return len(self.lon)

    def _ensure_kdtree(self) -> None:
        """Ensure KDTree is built for spatial queries."""
        if not hasattr(self, "_kdtree") or self._kdtree is None:
            xyz = np.column_stack(lonlat_to_cartesian(self.lon, self.lat))
            self._kdtree = cKDTree(xyz)

    def find_nearest(
        self,
        lon: float | NDArray[np.floating],
        lat: float | NDArray[np.floating],
        k: int = 1,
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]]:
        """Find nearest mesh points.

        Parameters
        ----------
        lon : float or array_like
            Query longitude(s).
        lat : float or array_like
            Query latitude(s).
        k : int
            Number of nearest neighbors to find.

        Returns
        -------
        distances : ndarray
            Distances to nearest points (in chord units on unit sphere).
        indices : ndarray
            Indices of nearest mesh points.
        """
        self._ensure_kdtree()

        lon_arr = np.atleast_1d(lon)
        lat_arr = np.atleast_1d(lat)
        xyz = np.column_stack(lonlat_to_cartesian(lon_arr, lat_arr))

        distances, indices = self._kdtree.query(xyz, k=k)

        return distances, indices

    def subset_by_bbox(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> NDArray[np.bool_]:
        """Get mask for points within bounding box.

        Parameters
        ----------
        lon_min, lon_max : float
            Longitude bounds.
        lat_min, lat_max : float
            Latitude bounds.

        Returns
        -------
        mask : ndarray
            Boolean mask of points within bounds.
        """
        lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
        lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
        return lon_mask & lat_mask
