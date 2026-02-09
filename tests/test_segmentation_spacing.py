from __future__ import annotations

import math

from pyproj import Geod

from pipeline_calculator.core.segmentation import segment_pipeline


def _p(sorted_vals: list[float], q: float) -> float:
    assert 0.0 <= q <= 1.0
    if not sorted_vals:
        return float("nan")
    idx = int(math.floor(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def test_segment_pipeline_midpoints_have_reasonable_spacing_on_long_edge() -> None:
    geod = Geod(ellps="GRS80")

    # Build a single long (~10km) geodesic edge due north so the expected
    # segment midpoints should be approximately evenly spaced.
    start = (-100.0, 40.0)
    end_lon, end_lat, _ = geod.fwd(start[0], start[1], 0.0, 10_000.0)
    end = (float(end_lon), float(end_lat))

    seg_len = 5.0
    segments = segment_pipeline(geod, [start, end], seg_len)

    # Expected: floor(total_len / seg_len) full segments.
    _, _, total_m = geod.inv(start[0], start[1], end[0], end[1])
    expected = int(math.floor(abs(float(total_m)) / seg_len))
    assert len(segments) == expected

    mids = [s["midpoint"] for s in segments]
    dists = []
    for (lon1, lat1), (lon2, lat2) in zip(mids, mids[1:]):
        _, _, d = geod.inv(float(lon1), float(lat1), float(lon2), float(lat2))
        dists.append(abs(float(d)))

    dists.sort()
    assert dists, "expected at least one inter-midpoint distance"

    # The previous buggy implementation clustered points (median ~0) and then
    # produced large gaps. This test ensures that does not happen.
    p50 = _p(dists, 0.50)
    p99 = _p(dists, 0.99)

    assert p50 > 4.0
    assert p99 < 6.0

