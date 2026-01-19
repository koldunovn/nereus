"""Tests for nereus.regrid.cache."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nereus.regrid.cache import (
    InterpolatorCache,
    clear_cache,
    get_cache,
    set_cache_options,
)


class TestInterpolatorCache:
    """Tests for InterpolatorCache class."""

    def test_basic_caching(self, random_mesh_small):
        """Test that interpolators are cached."""
        lon, lat = random_mesh_small
        cache = InterpolatorCache(max_memory_items=5)

        interp1 = cache.get_or_create(lon, lat, resolution=5.0)
        interp2 = cache.get_or_create(lon, lat, resolution=5.0)

        # Should be the same object
        assert interp1 is interp2

    def test_different_params_different_cache(self, random_mesh_small):
        """Test that different parameters create different cache entries."""
        lon, lat = random_mesh_small
        cache = InterpolatorCache(max_memory_items=5)

        interp1 = cache.get_or_create(lon, lat, resolution=5.0)
        interp2 = cache.get_or_create(lon, lat, resolution=10.0)

        # Should be different objects
        assert interp1 is not interp2

    def test_lru_eviction(self, random_mesh_small):
        """Test LRU eviction when cache is full."""
        lon, lat = random_mesh_small
        cache = InterpolatorCache(max_memory_items=2)

        # Create 3 interpolators, should evict the first
        interp1 = cache.get_or_create(lon, lat, resolution=1.0)
        interp2 = cache.get_or_create(lon, lat, resolution=2.0)
        interp3 = cache.get_or_create(lon, lat, resolution=3.0)

        # Cache should only have 2 items
        assert len(cache) == 2

    def test_clear(self, random_mesh_small):
        """Test cache clearing."""
        lon, lat = random_mesh_small
        cache = InterpolatorCache(max_memory_items=5)

        cache.get_or_create(lon, lat, resolution=5.0)
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0

    def test_disk_cache(self, random_mesh_small):
        """Test disk caching."""
        lon, lat = random_mesh_small

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = InterpolatorCache(max_memory_items=5, disk_path=tmpdir)

            interp1 = cache.get_or_create(lon, lat, resolution=5.0)

            # Check that file was created
            disk_files = list(Path(tmpdir).glob("*.pkl"))
            assert len(disk_files) == 1

    def test_disk_cache_restore(self, random_mesh_small):
        """Test restoring from disk cache."""
        lon, lat = random_mesh_small

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache and populate
            cache1 = InterpolatorCache(max_memory_items=5, disk_path=tmpdir)
            interp1 = cache1.get_or_create(lon, lat, resolution=5.0)

            # Clear memory cache
            cache1._memory_cache.clear()
            assert len(cache1) == 0

            # Should restore from disk
            interp2 = cache1.get_or_create(lon, lat, resolution=5.0)

            # Should have same shape (new object loaded from disk)
            assert interp2.target_lon.shape == interp1.target_lon.shape


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_cache(self):
        """Test getting global cache."""
        cache = get_cache()
        assert isinstance(cache, InterpolatorCache)

    def test_set_cache_options(self):
        """Test setting cache options."""
        set_cache_options(max_memory_items=20)
        cache = get_cache()
        assert cache.max_memory_items == 20

        # Reset to default
        set_cache_options(max_memory_items=10)

    def test_clear_cache(self, random_mesh_small):
        """Test clearing global cache."""
        lon, lat = random_mesh_small
        cache = get_cache()
        cache.get_or_create(lon, lat, resolution=5.0)

        clear_cache()
        assert len(cache) == 0
