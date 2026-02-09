"""Core (refactor-in-progress)."""

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.constants import (
    ANGULAR_TOLERANCE,
    DEFAULT_DETECTION_RANGE,
    GAP_TOLERANCE,
    MIN_PARALLEL_LENGTH,
    SEGMENT_LENGTH,
)
from pipeline_calculator.core.effective_length import compute_effective_length_by_clusters
from pipeline_calculator.core.overlap import calculate_overlap_results, find_parallel_segments
from pipeline_calculator.core.segmentation import segment_pipeline

__all__ = [
    "PipelineAnalyzer",
    "DEFAULT_DETECTION_RANGE",
    "MIN_PARALLEL_LENGTH",
    "SEGMENT_LENGTH",
    "ANGULAR_TOLERANCE",
    "GAP_TOLERANCE",
    "segment_pipeline",
    "find_parallel_segments",
    "calculate_overlap_results",
    "compute_effective_length_by_clusters",
]
