from __future__ import annotations

from pipeline_calculator.core.bundling import qualifying_sections, savings_from_sections
from pipeline_calculator.core.constants import MIN_PARALLEL_LENGTH


def compute_effective_length_by_clusters(
    pipelines,
    per_pipeline_total_meters,
    geod,
    segment_length,
    detection_range,
    angular_tolerance,
    progress_callback=None,
    min_parallel_length=MIN_PARALLEL_LENGTH,
):
    """Compatibility helper using the same qualified coverage as analysis results."""
    from pipeline_calculator.core.overlap import find_parallel_segments

    groups = find_parallel_segments(pipelines, geod, segment_length, detection_range,
                                    angular_tolerance, progress_callback)
    sections = qualifying_sections(pipelines, groups, segment_length, min_parallel_length)
    total = float(sum(per_pipeline_total_meters))
    savings = savings_from_sections(pipelines, sections, segment_length)
    return max(0.0, min(total, total - savings))
