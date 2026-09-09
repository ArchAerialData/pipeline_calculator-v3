from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


M_PER_DEG_LAT = 111_320.0


def lonlat_array_to_ecef(points_lonlat, geod):
    """Ellipsoid-surface Cartesian coordinates for conservative neighbor search.

    The straight chord between surface points cannot exceed their surface
    geodesic distance. A radius search here therefore retains every geodesic
    neighbor, including at the dateline and across large latitude spans.
    Final acceptance must still use geod.inv, not the chord distance.
    """
    points = np.asarray(points_lonlat, dtype=float).reshape((-1, 2))
    if not np.isfinite(points).all():
        raise ValueError("Non-finite coordinates in overlap analysis")
    lon, lat = np.radians(points[:, 0]), np.radians(points[:, 1])
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    radius = float(geod.a) / np.sqrt(1 - float(geod.es) * sin_lat ** 2)
    return np.column_stack((radius * cos_lat * np.cos(lon),
                            radius * cos_lat * np.sin(lon),
                            radius * (1 - float(geod.es)) * sin_lat))


def compute_origin(points_lonlat: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Choose a stable origin (lon0, lat0) for local XY conversions.

    We use a circular mean for longitude to avoid dateline issues, and a simple
    arithmetic mean for latitude.
    """
    if not points_lonlat:
        return (0.0, 0.0)

    lons = [float(p[0]) for p in points_lonlat]
    lats = [float(p[1]) for p in points_lonlat]

    lat0 = float(sum(lats) / len(lats))

    # Circular mean for longitudes in degrees.
    lon_rads = [math.radians(l) for l in lons]
    x = sum(math.cos(r) for r in lon_rads) / len(lon_rads)
    y = sum(math.sin(r) for r in lon_rads) / len(lon_rads)
    lon0 = float(math.degrees(math.atan2(y, x)))

    # Normalize to [-180, 180).
    lon0 = (lon0 + 180.0) % 360.0 - 180.0
    return (lon0, lat0)


def _delta_lon_deg(lon: float, lon0: float) -> float:
    # Normalize to [-180, 180) so dateline-crossing datasets don't explode.
    return (float(lon) - float(lon0) + 180.0) % 360.0 - 180.0


def lonlat_to_xy(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Convert lon/lat degrees to local XY meters around (lon0, lat0).

    Uses an equirectangular approximation, which is sufficient for our small
    neighbor search radius (e.g. 15m). Exact distances are still validated by
    geodesic checks elsewhere.
    """
    lat0_rad = math.radians(float(lat0))
    m_per_deg_x = M_PER_DEG_LAT * math.cos(lat0_rad)
    m_per_deg_y = M_PER_DEG_LAT

    dx_deg = _delta_lon_deg(lon, lon0)
    dy_deg = float(lat) - float(lat0)
    return (dx_deg * m_per_deg_x, dy_deg * m_per_deg_y)


def lonlat_array_to_xy(
    points_lonlat: Sequence[tuple[float, float]], lon0: float, lat0: float
) -> np.ndarray:
    """Vectorized lon/lat degrees -> XY meters.

    Returns:
      np.ndarray shape (N, 2) with dtype float64
    """
    if not points_lonlat:
        return np.zeros((0, 2), dtype=float)

    lons = np.array([p[0] for p in points_lonlat], dtype=float)
    lats = np.array([p[1] for p in points_lonlat], dtype=float)

    # Normalize delta lon to [-180, 180) degrees.
    dlon = (lons - float(lon0) + 180.0) % 360.0 - 180.0
    dlat = lats - float(lat0)

    lat0_rad = math.radians(float(lat0))
    m_per_deg_x = M_PER_DEG_LAT * math.cos(lat0_rad)
    m_per_deg_y = M_PER_DEG_LAT

    x = dlon * m_per_deg_x
    y = dlat * m_per_deg_y
    return np.column_stack((x, y))

