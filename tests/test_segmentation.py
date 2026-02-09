from __future__ import annotations

import math

import pipeline_calculator_v3 as pc


def test_segment_pipeline_count_and_indices() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 10.0

    coords = [(-100.0, 40.0), (-100.0, 40.001)]  # ~111 m northward
    _, _, distance_m = analyzer.geod.inv(coords[0][0], coords[0][1], coords[1][0], coords[1][1])
    expected = int(math.floor(abs(distance_m) / analyzer.segment_length))

    segments = analyzer.segment_pipeline(coords)

    assert len(segments) == expected
    assert [s["segment_index"] for s in segments] == list(range(len(segments)))
    assert all(abs(float(s["length"]) - analyzer.segment_length) < 1e-6 for s in segments)


def test_segment_pipeline_handles_short_line() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 1000.0
    segments = analyzer.segment_pipeline([(-100.0, 40.0), (-100.0, 40.0001)])
    assert segments == []

