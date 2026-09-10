from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

from pipeline_calculator.core.segmentation import MAX_ANALYSIS_SEGMENTS, segment_pipeline


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


def segment_pipeline_paths(geod, pipeline, segment_length, *, context=None, max_segments=None):
    """Segment every coordinate path without connecting disjoint parts."""
    segments = []
    limit = MAX_ANALYSIS_SEGMENTS if max_segments is None else min(max_segments, MAX_ANALYSIS_SEGMENTS)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError('Segment budget must be a nonnegative integer')

    for path_index, coords in enumerate(coordinate_paths_for_pipeline(pipeline, context=context)):
        if context is not None and path_index % 256 == 0:
            context.check()
        path_segments = segment_pipeline(geod, coords, segment_length, context=context,
                                         max_segments=limit-len(segments))
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
