"""Region mask utilities for geographic data analysis.

This module provides functions for loading GeoJSON region definitions
and creating boolean masks for points within named regions.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np


def load_geojson(name: str) -> dict:
    """Load a GeoJSON file from the package data directory.

    Parameters
    ----------
    name : str
        Name of the GeoJSON file (without .geojson extension).
        Available files: MOCBasins, NinoRegions, oceanBasins

    Returns
    -------
    dict
        Parsed GeoJSON data as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the GeoJSON file does not exist.
    """
    try:
        files = resources.files("nereus.diag") / "data" / f"{name}.geojson"
        with files.open("r") as f:
            return json.load(f)
    except (TypeError, AttributeError):
        # Fallback for older Python versions
        data_dir = Path(__file__).parent / "data"
        filepath = data_dir / f"{name}.geojson"
        if not filepath.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")
        with open(filepath) as f:
            return json.load(f)


def list_available_regions(geojson_name: str = "MOCBasins") -> list[str]:
    """List available region names in a GeoJSON file.

    Parameters
    ----------
    geojson_name : str, optional
        Name of the GeoJSON file (without extension).
        Default is "MOCBasins".

    Returns
    -------
    list[str]
        List of region names defined in the GeoJSON file.
    """
    geojson = load_geojson(geojson_name)
    names = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name") or props.get("NAME") or props.get("Name")
        if name:
            names.append(name)
    return names


def get_region_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    region: str,
    geojson_name: str = "MOCBasins",
) -> np.ndarray:
    """Create a boolean mask for points within a named region.

    Parameters
    ----------
    lon : np.ndarray
        Array of longitude values.
    lat : np.ndarray
        Array of latitude values.
    region : str
        Name of the region to create a mask for.
    geojson_name : str, optional
        Name of the GeoJSON file (without extension).
        Default is "MOCBasins".

    Returns
    -------
    np.ndarray
        Boolean array where True indicates points inside the region.

    Raises
    ------
    ImportError
        If shapely is not installed.
    ValueError
        If the region is not found in the GeoJSON file.
    """
    try:
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError:
        raise ImportError(
            "shapely is required for region masks. "
            "Install with: pip install shapely>=2.0"
        )

    geojson = load_geojson(geojson_name)
    geometries = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name") or props.get("NAME") or props.get("Name")
        if name == region:
            geometries.append(shape(feature["geometry"]))

    if not geometries:
        available = list_available_regions(geojson_name)
        raise ValueError(
            f"Region '{region}' not found in {geojson_name}.geojson. "
            f"Available regions: {available}"
        )

    region_geom = unary_union(geometries)
    prepared_geom = prep(region_geom)
    mask = np.array([prepared_geom.contains(Point(x, y)) for x, y in zip(lon, lat)])
    return mask
