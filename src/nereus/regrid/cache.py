"""Caching for RegridInterpolator instances.

This module provides in-memory LRU caching with optional disk persistence
for RegridInterpolator objects.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from nereus.regrid.interpolator import RegridInterpolator

# Global cache instance
_cache: InterpolatorCache | None = None
_cache_lock = threading.Lock()


class InterpolatorCache:
    """In-memory LRU cache with optional disk persistence.

    This cache stores RegridInterpolator instances keyed by a hash of their
    source coordinates and parameters. It uses LRU (Least Recently Used)
    eviction policy.

    Parameters
    ----------
    max_memory_items : int
        Maximum number of interpolators to keep in memory.
    disk_path : str or Path, optional
        Directory for disk cache. If None, disk caching is disabled.

    Examples
    --------
    >>> cache = InterpolatorCache(max_memory_items=5)
    >>> interp = cache.get_or_create(lon, lat, resolution=1.0)
    >>> # Second call returns cached interpolator
    >>> interp2 = cache.get_or_create(lon, lat, resolution=1.0)
    >>> interp is interp2
    True
    """

    def __init__(
        self,
        max_memory_items: int = 10,
        disk_path: str | Path | None = None,
    ) -> None:
        self.max_memory_items = max_memory_items
        self.disk_path = Path(disk_path) if disk_path else None
        self._memory_cache: OrderedDict[str, "RegridInterpolator"] = OrderedDict()
        self._lock = threading.Lock()

        if self.disk_path:
            self.disk_path.mkdir(parents=True, exist_ok=True)

    def _compute_key(
        self,
        source_lon: np.ndarray,
        source_lat: np.ndarray,
        **kwargs: Any,
    ) -> str:
        """Compute cache key from coordinates and parameters."""
        # Create a stable hash from coordinates and params
        hasher = hashlib.sha256()

        # Hash coordinates (sample for large arrays)
        lon_flat = np.asarray(source_lon).ravel()
        lat_flat = np.asarray(source_lat).ravel()

        # Sample points for large arrays to speed up hashing
        n = len(lon_flat)
        if n > 1000:
            step = n // 1000
            lon_sample = lon_flat[::step]
            lat_sample = lat_flat[::step]
        else:
            lon_sample = lon_flat
            lat_sample = lat_flat

        hasher.update(lon_sample.tobytes())
        hasher.update(lat_sample.tobytes())
        hasher.update(str(n).encode())  # Include array size

        # Hash parameters
        for key in sorted(kwargs.keys()):
            hasher.update(f"{key}={kwargs[key]}".encode())

        return hasher.hexdigest()[:16]

    def get_or_create(
        self,
        source_lon: np.ndarray,
        source_lat: np.ndarray,
        **kwargs: Any,
    ) -> "RegridInterpolator":
        """Get cached interpolator or create new one.

        Parameters
        ----------
        source_lon : array_like
            Source grid longitude coordinates.
        source_lat : array_like
            Source grid latitude coordinates.
        **kwargs
            Additional parameters passed to RegridInterpolator.

        Returns
        -------
        RegridInterpolator
            Cached or newly created interpolator.
        """
        from nereus.regrid.interpolator import RegridInterpolator

        key = self._compute_key(source_lon, source_lat, **kwargs)

        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                # Move to end (most recently used)
                self._memory_cache.move_to_end(key)
                return self._memory_cache[key]

            # Check disk cache
            if self.disk_path:
                disk_file = self.disk_path / f"{key}.pkl"
                if disk_file.exists():
                    try:
                        with open(disk_file, "rb") as f:
                            interp = pickle.load(f)
                        self._add_to_memory_cache(key, interp)
                        return interp
                    except (pickle.PickleError, OSError):
                        # Corrupted cache file, ignore
                        pass

            # Create new interpolator
            interp = RegridInterpolator(
                source_lon=source_lon,
                source_lat=source_lat,
                **kwargs,
            )

            self._add_to_memory_cache(key, interp)

            # Save to disk
            if self.disk_path:
                self._save_to_disk(key, interp)

            return interp

    def _add_to_memory_cache(self, key: str, interp: "RegridInterpolator") -> None:
        """Add interpolator to memory cache with LRU eviction."""
        self._memory_cache[key] = interp
        self._memory_cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._memory_cache) > self.max_memory_items:
            self._memory_cache.popitem(last=False)

    def _save_to_disk(self, key: str, interp: "RegridInterpolator") -> None:
        """Save interpolator to disk cache."""
        if not self.disk_path:
            return

        disk_file = self.disk_path / f"{key}.pkl"
        try:
            with open(disk_file, "wb") as f:
                pickle.dump(interp, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OSError):
            # Failed to save, ignore
            pass

    def clear(self) -> None:
        """Clear all cached interpolators."""
        with self._lock:
            self._memory_cache.clear()

            if self.disk_path:
                for f in self.disk_path.glob("*.pkl"):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    def __len__(self) -> int:
        """Number of interpolators in memory cache."""
        return len(self._memory_cache)


def get_cache() -> InterpolatorCache:
    """Get the global interpolator cache.

    Returns
    -------
    InterpolatorCache
        The global cache instance.
    """
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = InterpolatorCache()
        return _cache


def set_cache_options(
    max_memory_items: int = 10,
    disk_path: str | Path | None = None,
) -> None:
    """Configure the global interpolator cache.

    Parameters
    ----------
    max_memory_items : int
        Maximum number of interpolators to keep in memory.
    disk_path : str or Path, optional
        Directory for disk cache. If None, disk caching is disabled.
    """
    global _cache
    with _cache_lock:
        _cache = InterpolatorCache(
            max_memory_items=max_memory_items,
            disk_path=disk_path,
        )


def clear_cache() -> None:
    """Clear the global interpolator cache."""
    cache = get_cache()
    cache.clear()
