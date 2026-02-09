from __future__ import annotations

import pipeline_calculator_v3 as pc


def test_find_parallel_segments_and_overlap_results() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 25.0
    analyzer.min_parallel_length = 100.0
    analyzer.detection_range = 30.0
    analyzer.angular_tolerance = 10.0

    # Two roughly parallel lines ~11m apart, ~250m long.
    pipelines = [
        {"name": "A", "coordinates": [(-100.0, 40.0), (-99.997, 40.0)]},
        {"name": "B", "coordinates": [(-100.0, 40.00009), (-99.997, 40.00009)]},
    ]

    parallel = analyzer.find_parallel_segments(pipelines)
    assert (0, 1) in parallel
    assert len(parallel[(0, 1)]) > 0

    overlap = analyzer.calculate_overlap_results(pipelines, parallel)
    assert "bundled_sections" in overlap
    assert overlap["bundled_sections"], "expected at least one bundled section"

    section = overlap["bundled_sections"][0]
    assert section["pipeline_1"] in ("A", "B")
    assert section["pipeline_2"] in ("A", "B")
    assert float(section["bundled_length_meters"]) > 0.0

    # Clamp is expected: oriented width never exceeds 2 * detection range.
    assert float(section.get("oriented_width_m", 0.0)) <= 2.0 * analyzer.detection_range + 1e-6

    poly = section.get("corridor_polygon")
    assert isinstance(poly, list)
    assert len(poly) >= 5
    assert poly[0] == poly[-1]

