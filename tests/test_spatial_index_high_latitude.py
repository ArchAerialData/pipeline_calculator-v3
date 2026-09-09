from __future__ import annotations

from pyproj import Geod

from pipeline_calculator.core.overlap import find_parallel_segments


def test_find_parallel_segments_high_latitude_does_not_miss_east_west_neighbors() -> None:
    """Regression test for the 'meters -> degrees' KDTree bug.

    At higher latitudes, a single conversion R/111000 underestimates the
    longitude search radius by cos(latitude), which can miss true neighbors.
    """
    geod = Geod(ellps="GRS80")

    # Two parallel lines at latitude 60 deg, offset ~12m east-west, length ~20m.
    lat0 = 60.0
    lon0 = -100.0

    start_a = (lon0, lat0)
    end_a_lon, end_a_lat, _ = geod.fwd(lon0, lat0, 0.0, 20.0)
    end_a = (float(end_a_lon), float(end_a_lat))

    start_b_lon, start_b_lat, _ = geod.fwd(lon0, lat0, 90.0, 12.0)
    start_b = (float(start_b_lon), float(start_b_lat))
    end_b_lon, end_b_lat, _ = geod.fwd(start_b[0], start_b[1], 0.0, 20.0)
    end_b = (float(end_b_lon), float(end_b_lat))

    pipelines = [
        {"name": "A", "coordinates": [start_a, end_a]},
        {"name": "B", "coordinates": [start_b, end_b]},
    ]

    parallel = find_parallel_segments(
        pipelines,
        geod,
        segment_length=5.0,
        detection_range=15.0,
        angular_tolerance=15.0,
    )

    assert (0, 1) in parallel
    assert parallel[(0, 1)], "expected at least one nearby parallel segment pair"
    assert all(float(p["distance"]) <= 15.0 + 1e-6 for p in parallel[(0, 1)])

