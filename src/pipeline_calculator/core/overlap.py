from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

import math
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree

from pipeline_calculator.core.angles import bearing_orientation_diff
from pipeline_calculator.core.coordinates import segment_pipeline_paths
from pipeline_calculator.core.corridor_coverage import qualified_path_runs
from pipeline_calculator.core.corridor_buffer import (
    CorridorDisplayOptions, CorridorGeometryBudget, build_buffered_corridor,
)
from pipeline_calculator.core.spatial import lonlat_array_to_ecef
from pipeline_calculator.core.bundling import qualifying_sections, savings_from_sections
from pipeline_calculator.core.workload import check_density_workload
from pipeline_calculator.core.segmentation import MAX_ANALYSIS_SEGMENTS

MAX_CANDIDATE_CHECKS = 5_000_000
# Bound cheap index visits separately from unique cross-pipeline comparisons.
# A straight path alone contributes roughly seven hits per 5 m segment at
# the default radius, even when there is no other pipeline nearby.
MAX_NEIGHBOR_VISITS = 20_000_000


def find_parallel_segments(pipelines, geod, segment_length, detection_range, angular_tolerance, progress_callback=None, *, context=None):
    """Identify pipeline segments that run parallel within detection range.

    Mutates pipelines by adding a `segments` list on each pipeline dict.
    """
    if not math.isfinite(float(segment_length)) or segment_length <= 0:
        raise ValueError("Segment length must be a finite positive number")
    if not all(math.isfinite(float(v)) for v in (detection_range, angular_tolerance)):
        raise ValueError("Analysis parameters must be finite")
    if segment_length <= 0 or detection_range <= 0 or not 0 <= angular_tolerance <= 90:
        raise ValueError("Segment length and detection range must be positive; angle must be between 0 and 90")
    # Segment all pipelines
    segment_count = 0
    for p_idx, pipeline in enumerate(pipelines):
        if context is not None and p_idx % 256 == 0:
            context.check()
        if context is not None:
            context.report("Segmenting paths", segment_count, context.estimated_segments)
        if progress_callback:
            progress = 0.5 + (p_idx / max(len(pipelines), 1)) * 0.25  # 50-75% progress
            progress_callback(progress)
        pipeline["segments"] = segment_pipeline_paths(geod, pipeline, segment_length, context=context,
                                                       max_segments=MAX_ANALYSIS_SEGMENTS-segment_count,
                                                       progress_offset=segment_count,
                                                       progress_total=context.estimated_segments if context else None)
        segment_count += len(pipeline["segments"])
        if segment_count > MAX_ANALYSIS_SEGMENTS:
            raise ValueError("Analysis segment limit exceeded; split the dataset or increase segment length")

    if context is not None:
        context.report("Building spatial index", 0, segment_count)
    # Build spatial index
    all_segments = []
    segment_to_pipeline = {}

    for p_idx, pipeline in enumerate(pipelines):
        if context is not None and p_idx % 256 == 0:
            context.check()
        for segment_position, seg in enumerate(pipeline['segments']):
            if context is not None and segment_position % 256 == 0:
                context.report("Building spatial index", len(all_segments), segment_count)
            seg_idx = len(all_segments)
            all_segments.append(seg["midpoint"])
            segment_to_pipeline[seg_idx] = (p_idx, seg)

    if not all_segments:
        return {}

    try:
        xy = lonlat_array_to_ecef(all_segments, geod)
        tree = KDTree(xy)
    except AnalysisCancelled:
        raise
    except Exception as e:
        raise ValueError("Could not build overlap spatial index") from e

    radius = math.hypot(float(detection_range), float(segment_length)) + 1e-8
    check_density_workload(tree, xy, radius, context)
    if context is not None:
        context.report("Searching neighbors", 0, len(all_segments))
    parallel_groups = defaultdict(list)
    candidate_checks = 0
    neighbor_visits = 0

    for seg_idx, (p_idx, segment) in segment_to_pipeline.items():
        if context is not None and seg_idx % 256 == 0:
            context.report("Searching neighbors", seg_idx, len(all_segments))
        try:
            nearby_indices = tree.query_ball_point(
                xy[seg_idx], radius
            )
            neighbor_visits += len(nearby_indices)
            if neighbor_visits > MAX_NEIGHBOR_VISITS:
                raise ValueError("Neighbor-search limit exceeded; split the dataset or reduce detection range")

            for candidate_position, near_idx in enumerate(nearby_indices):
                if context is not None and candidate_position % 256 == 0:
                    context.check()
                # Each unordered pair needs one comparison. This also excludes
                # self hits before they consume the cross-pipeline budget.
                if near_idx <= seg_idx:
                    continue

                near_p_idx, near_segment = segment_to_pipeline[near_idx]
                if p_idx == near_p_idx:
                    continue
                candidate_checks += 1
                if candidate_checks > MAX_CANDIDATE_CHECKS:
                    raise ValueError("Neighbor-search limit exceeded; split the dataset or reduce detection range")

                if bearing_orientation_diff(segment["bearing"], near_segment["bearing"]) <= angular_tolerance:
                    lon1, lat1 = segment["midpoint"]
                    lon2, lat2 = near_segment["midpoint"]
                    azimuth, _, midpoint_distance = geod.inv(lon1, lat1, lon2, lat2)
                    delta1 = math.radians(azimuth - segment["bearing"])
                    # Back azimuth at the second midpoint handles meridian convergence.
                    reverse_azimuth, _, _ = geod.inv(lon2, lat2, lon1, lat1)
                    delta2 = math.radians(reverse_azimuth - near_segment["bearing"])
                    distance = max(abs(midpoint_distance * math.sin(delta1)),
                                   abs(midpoint_distance * math.sin(delta2)))
                    half_extent = (segment["length"] + near_segment["length"]) / 2
                    along = max(abs(midpoint_distance * math.cos(delta1)),
                                abs(midpoint_distance * math.cos(delta2)))
                    # Compare short, finite segment tangents instead of requiring
                    # their midpoint samples to line up. Touching endpoints alone
                    # do not constitute longitudinal overlap.
                    if distance <= detection_range and along < half_extent - 1e-8:
                        key = tuple(sorted([p_idx, near_p_idx]))
                        if key[0] == p_idx:
                            parallel_groups[key].append(
                                {
                                    "pipeline_1_segment": segment["segment_index"],
                                    "pipeline_2_segment": near_segment["segment_index"],
                                    "pipeline_1_path": segment.get("path_index", 0),
                                    "pipeline_2_path": near_segment.get("path_index", 0),
                                    "pipeline_1_path_segment": segment.get(
                                        "path_segment_index",
                                        segment["segment_index"],
                                    ),
                                    "pipeline_2_path_segment": near_segment.get(
                                        "path_segment_index",
                                        near_segment["segment_index"],
                                    ),
                                    "distance": distance,
                                    "midpoint_distance": midpoint_distance,
                                }
                            )
                        else:
                            parallel_groups[key].append(
                                {
                                    "pipeline_1_segment": near_segment["segment_index"],
                                    "pipeline_2_segment": segment["segment_index"],
                                    "pipeline_1_path": near_segment.get("path_index", 0),
                                    "pipeline_2_path": segment.get("path_index", 0),
                                    "pipeline_1_path_segment": near_segment.get(
                                        "path_segment_index",
                                        near_segment["segment_index"],
                                    ),
                                    "pipeline_2_path_segment": segment.get(
                                        "path_segment_index",
                                        segment["segment_index"],
                                    ),
                                    "distance": distance,
                                    "midpoint_distance": midpoint_distance,
                                }
                            )
        except AnalysisCancelled:
            raise
        except Exception as e:
            raise ValueError(f"Could not analyze segment {seg_idx}: {e}") from e

    return parallel_groups


def _build_section_corridor(pipelines, qualified, geod, segment_length,
                            options, budget, cache, scope, context):
    runs = qualified_path_runs(pipelines, qualified, segment_length, geod,
                               scope=scope, cache=cache, context=context, budget=budget)
    return build_buffered_corridor(runs, geod=geod, options=options, budget=budget, context=context)


def calculate_overlap_results(
    pipelines,
    parallel_groups,
    geod,
    survey_mile_m,
    segment_length,
    min_parallel_length,
    detection_range,
    angular_tolerance,
    progress_callback=None,
    *, context=None, corridor_options=None, corridor_budget=None, corridor_scope='Combined',
):
    """Calculate bundled lengths and overlap statistics."""
    if not math.isfinite(float(min_parallel_length)) or min_parallel_length <= 0:
        raise ValueError("Minimum parallel length must be finite and positive")
    results = {
        "bundled_sections": [],
        "pipeline_overlaps": {},
        "pipeline_overlaps_by_id": {},
        "total_bundled_length": 0,
        "effective_total_length": 0,
        "savings_meters": 0,
        "savings_miles": 0,
        "savings_percentage": 0,
        "parameter_impacts": {},
    }

    bundled_segments = defaultdict(set)
    sections = qualifying_sections(pipelines, parallel_groups, segment_length, min_parallel_length, context=context)
    # Finish every numerical section before attempting any optional map work.
    records = []
    for index, qualified in enumerate(sections):
        if context is not None:
            context.report('Recording bundled sections', index, len(sections))
        p1_idx, p2_idx = qualified['pair']
        length = qualified['length']
        record = {
            'pipeline_1': pipelines[p1_idx]['name'], 'pipeline_2': pipelines[p2_idx]['name'],
            'pipeline_1_id': pipelines[p1_idx].get('id', p1_idx),
            'pipeline_2_id': pipelines[p2_idx].get('id', p2_idx),
            'source_path_indices': list(qualified['paths']),
            'bundled_length_meters': length, 'bundled_length_miles': length / survey_mile_m,
            'average_separation': float(np.mean([s['distance'] for s in qualified['representatives']])),
            'segment_count': min(map(len, qualified['segment_ids'])),
        }
        bundled_segments[p1_idx].update(qualified['segment_ids'][0])
        bundled_segments[p2_idx].update(qualified['segment_ids'][1])
        records.append((record, qualified))
        results['bundled_sections'].append(record)
    results['bundled_sections'].sort(key=lambda s: s['bundled_length_miles'], reverse=True)

    for p_idx, pipeline in enumerate(pipelines):
        if context is not None and p_idx % 256 == 0:
            context.check()
        bundled_count = len(bundled_segments[p_idx])
        bundled_length = bundled_count * segment_length

        detail = {
            "bundled_segments": bundled_count,
            "bundled_length_meters": bundled_length,
            "bundled_length_miles": bundled_length / survey_mile_m,
        }
        results["pipeline_overlaps"][pipeline["name"]] = detail
        source_id = pipeline.get("id", p_idx)
        results["pipeline_overlaps_by_id"][str(source_id)] = dict(
            detail, source_id=source_id, source_name=pipeline["name"])

    total_bundled = sum(section["bundled_length_meters"] for section in results["bundled_sections"])
    results["total_bundled_length"] = total_bundled
    results["savings_meters"] = savings_from_sections(pipelines, sections, segment_length, context=context)

    measured_paths = {}
    corridor_options = corridor_options or CorridorDisplayOptions()
    corridor_budget = corridor_budget or CorridorGeometryBudget()
    for index, (record, qualified) in enumerate(records):
        if context is not None:
            context.report('Building corridor maps', index, len(records))
        try:
            record.update(_build_section_corridor(
                pipelines, qualified, geod, segment_length, corridor_options, corridor_budget,
                measured_paths, corridor_scope, context))
        except AnalysisCancelled:
            raise
        except Exception as exc:
            record.update(visualization_schema_version=1, visualization_kind='qualified_path_buffer',
                          visualization_status='omitted', visualization_polygons=[], corridor_polygon=[],
                          visualization_metadata={
                              'policy': 'qualified_path_buffer_v1', 'padding_m': corridor_options.padding_m,
                              'approximation_target_m': corridor_options.approximation_target_m,
                              'cap_style': 'round', 'join_style': 'round', 'source_runs': [],
                              'part_count': 0, 'hole_count': 0, 'vertex_count': 0, 'chart_count': 0,
                          }, diagnostics=[{
                'level': 'warning', 'code': getattr(exc, 'code', 'corridor_visualization_omitted'),
                'message': 'A corridor map is unavailable; mileage is unaffected.',
                'context': {'error': str(exc), 'error_type': type(exc).__name__},
            }])
    if context is not None:
        context.report('Building corridor maps', len(records), len(records))

    if progress_callback:
        progress_callback(1.0)

    return results
