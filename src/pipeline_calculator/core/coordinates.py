from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

from pipeline_calculator.core.segmentation import segment_pipeline


def coordinate_paths_for_pipeline(pipeline, *, context=None):
    """Return valid coordinate paths for a pipeline with legacy fallback."""
    paths = []

    for path_position, path in enumerate(pipeline.get('coordinate_paths') or []):
        if context is not None and path_position % 256 == 0:
            context.check()
        coords = list(path or [])
        if len(coords) >= 2:
            paths.append(coords)

    if paths:
        return paths

    coords = list(pipeline.get("coordinates") or [])
    return [coords] if len(coords) >= 2 else []


def segment_pipeline_paths(geod, pipeline, segment_length, *, context=None):
    """Segment every coordinate path without connecting disjoint parts."""
    segments = []

    for path_index, coords in enumerate(coordinate_paths_for_pipeline(pipeline, context=context)):
        if context is not None and path_index % 256 == 0:
            context.check()
        path_segments = segment_pipeline(geod, coords, segment_length, context=context)
        for path_segment_index, segment in enumerate(path_segments):
            if context is not None and path_segment_index % 256 == 0:
                context.check()
            segment_with_path = dict(segment)
            segment_with_path["path_index"] = path_index
            segment_with_path["path_segment_index"] = segment.get(
                "segment_index",
                path_segment_index,
            )
            segment_with_path["segment_index"] = len(segments)
            segments.append(segment_with_path)

    return segments
