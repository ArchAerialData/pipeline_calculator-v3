from __future__ import annotations

from pipeline_calculator.core.segmentation import segment_pipeline


def coordinate_paths_for_pipeline(pipeline):
    """Return valid coordinate paths for a pipeline with legacy fallback."""
    paths = []

    for path in pipeline.get("coordinate_paths") or []:
        coords = list(path or [])
        if len(coords) >= 2:
            paths.append(coords)

    if paths:
        return paths

    coords = list(pipeline.get("coordinates") or [])
    return [coords] if len(coords) >= 2 else []


def segment_pipeline_paths(geod, pipeline, segment_length):
    """Segment every coordinate path without connecting disjoint parts."""
    segments = []

    for path_index, coords in enumerate(coordinate_paths_for_pipeline(pipeline)):
        path_segments = segment_pipeline(geod, coords, segment_length)
        for path_segment_index, segment in enumerate(path_segments):
            segment_with_path = dict(segment)
            segment_with_path["path_index"] = path_index
            segment_with_path["path_segment_index"] = segment.get(
                "segment_index",
                path_segment_index,
            )
            segment_with_path["segment_index"] = len(segments)
            segments.append(segment_with_path)

    return segments
