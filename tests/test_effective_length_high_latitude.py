from __future__ import annotations

from pyproj import Geod

from pipeline_calculator.core.effective_length import compute_effective_length_by_clusters


def test_effective_length_clusters_detect_neighbors_high_latitude() -> None:
    geod = Geod(ellps="GRS80")

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

    per_pipe_totals = []
    for p in pipelines:
        (lon1, lat1), (lon2, lat2) = p["coordinates"]
        _, _, d = geod.inv(lon1, lat1, lon2, lat2)
        per_pipe_totals.append(abs(float(d)))

    eff = compute_effective_length_by_clusters(
        pipelines,
        per_pipe_totals,
        geod,
        segment_length=5.0,
        detection_range=15.0,
        angular_tolerance=15.0,
        min_parallel_length=10.0,
    )

    # Two identical-length, fully-overlapping pipelines should discount by ~2x.
    assert eff > 0.0
    assert abs(eff - per_pipe_totals[0]) / per_pipe_totals[0] < 0.05

