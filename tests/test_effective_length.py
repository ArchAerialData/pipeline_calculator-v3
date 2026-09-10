from __future__ import annotations

import pipeline_calculator_v3 as pc


def test_effective_length_clusters_two_identical_pipelines() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 25.0
    analyzer.detection_range = 30.0
    analyzer.angular_tolerance = 10.0

    coords = [(-100.0, 40.0), (-99.997, 40.0)]
    pipelines = [
        {"objectid": "1", "name": "A", "coordinates": coords},
        {"objectid": "2", "name": "B", "coordinates": coords},
    ]

    pipeline_data, total_meters, _ = analyzer.calculate_pipeline_lengths(pipelines)
    per_pipe_totals = [d["Shape_Length"] for d in pipeline_data]

    eff = analyzer.compute_effective_length_by_clusters(pipelines, per_pipe_totals)

    # Total is double-counted across 2 identical pipelines; effective should be close to half.
    assert eff > 0.0
    assert eff < total_meters
    assert abs(eff - (total_meters / 2.0)) / total_meters < 0.15


def test_effective_length_clusters_non_overlapping_pipelines_no_discount() -> None:
    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 25.0
    analyzer.detection_range = 30.0
    analyzer.angular_tolerance = 10.0

    pipelines = [
        {"objectid": "1", "name": "A", "coordinates": [(-100.0, 40.0), (-99.997, 40.0)]},
        # Far away (~1km) so no clustering overlap.
        {"objectid": "2", "name": "B", "coordinates": [(-100.0, 40.01), (-99.997, 40.01)]},
    ]

    pipeline_data, total_meters, _ = analyzer.calculate_pipeline_lengths(pipelines)
    per_pipe_totals = [d["Shape_Length"] for d in pipeline_data]

    eff = analyzer.compute_effective_length_by_clusters(pipelines, per_pipe_totals)

    assert abs(eff - total_meters) / total_meters < 0.05


def test_effective_length_clusters_multipart_paths_without_gap_penalty() -> None:
    analyzer = pc.PipelineAnalyzer()
    # Each 25 m part must independently meet the configured bundling minimum.
    analyzer.min_parallel_length = 20.0
    analyzer.segment_length = 5.0
    analyzer.detection_range = 5.0
    analyzer.angular_tolerance = 10.0

    def line_from(start, length_m):
        end_lon, end_lat, _ = analyzer.geod.fwd(start[0], start[1], 0.0, length_m)
        return [start, (float(end_lon), float(end_lat))]

    paths = [
        line_from((-100.0, 40.0), 25.0),
        line_from((-99.0, 41.0), 25.0),
    ]
    pipelines = [
        {"objectid": "1", "name": "A", "coordinates": paths[0], "coordinate_paths": paths},
        {"objectid": "2", "name": "B", "coordinates": paths[0], "coordinate_paths": paths},
    ]

    pipeline_data, total_meters, _ = analyzer.calculate_pipeline_lengths(pipelines)
    per_pipe_totals = [d["Shape_Length"] for d in pipeline_data]

    eff = analyzer.compute_effective_length_by_clusters(pipelines, per_pipe_totals)

    assert eff > 0.0
    assert eff < total_meters
    assert abs(eff - (total_meters / 2.0)) / total_meters < 0.05

