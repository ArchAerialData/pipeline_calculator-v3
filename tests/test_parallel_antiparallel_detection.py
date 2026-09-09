from __future__ import annotations

from pyproj import Geod

from pipeline_calculator.core.overlap import find_parallel_segments


def test_find_parallel_segments_detects_antiparallel_digitization() -> None:
    geod = Geod(ellps="GRS80")

    start = (-100.0, 40.0)
    end_lon, end_lat, _ = geod.fwd(start[0], start[1], 0.0, 50.0)
    end = (float(end_lon), float(end_lat))

    # Same geometry, opposite digitization direction.
    pipelines = [
        {"name": "A", "coordinates": [start, end]},
        {"name": "B", "coordinates": [end, start]},
    ]

    parallel = find_parallel_segments(
        pipelines,
        geod,
        segment_length=5.0,
        detection_range=5.0,
        angular_tolerance=10.0,
    )

    assert (0, 1) in parallel
    assert parallel[(0, 1)], "expected anti-parallel segments to be treated as parallel"

