from __future__ import annotations

import math


def bearing_orientation_diff(b1_deg: float, b2_deg: float) -> float:
    """Return the orientation difference between two bearings, in degrees.

    "Parallel" pipelines should be treated as parallel even if digitized in
    opposite directions (anti-parallel).

    Returns:
      A value in [0, 90], where:
      - 0 means same (or anti-parallel) orientation
      - 90 means perpendicular
    """
    try:
        b1 = float(b1_deg)
        b2 = float(b2_deg)
    except Exception:
        return float("inf")

    if math.isnan(b1) or math.isnan(b2):
        return float("inf")

    diff = abs(b1 - b2) % 360.0
    diff = min(diff, 360.0 - diff)  # in [0, 180]
    return min(diff, 180.0 - diff)  # in [0, 90]

