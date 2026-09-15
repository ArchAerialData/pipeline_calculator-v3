from __future__ import annotations

from pipeline_calculator.core.execution import (
    AnalysisCancelled, CallbackExecutionContext, ExecutionContext, ScopedExecutionContext,
)
from pipeline_calculator.core.options import AnalysisOptions

import math

from pipeline_calculator.core.constants import (
    ANGULAR_TOLERANCE,
    DEFAULT_DETECTION_RANGE,
    GEOD_ELLPS,
    MIN_PARALLEL_LENGTH,
    SEGMENT_LENGTH,
    SURVEY_MILE_METERS,
)
from pipeline_calculator.core.coordinates import coordinate_paths_for_pipeline
from pipeline_calculator.core.effective_length import compute_effective_length_by_clusters
from pipeline_calculator.core.overlap import calculate_overlap_results, find_parallel_segments
from pipeline_calculator.core.segmentation import segment_pipeline
from pipeline_calculator.core.workload import check_segment_workload
from pipeline_calculator.core.overlap import MAX_ANALYSIS_SEGMENTS
from pipeline_calculator.parsers.kml_kmz import (
    extract_features_from_file,
    extract_features_from_file_with_diagnostics,
)


class PipelineAnalyzer:
    """Combined pipeline length and overlap analyzer (package implementation)."""

    def __init__(
        self,
        *,
        geod=None,
        survey_mile=SURVEY_MILE_METERS,
        detection_range=DEFAULT_DETECTION_RANGE,
        min_parallel_length=MIN_PARALLEL_LENGTH,
        segment_length=SEGMENT_LENGTH,
        angular_tolerance=ANGULAR_TOLERANCE,
    ):
        if geod is None:
            from pyproj import Geod

            geod = Geod(ellps=GEOD_ELLPS)

        self._context = None
        self.geod = geod
        self.survey_mile = float(survey_mile)
        self.detection_range = float(detection_range)
        self.min_parallel_length = float(min_parallel_length)
        self.segment_length = float(segment_length)
        self.angular_tolerance = float(angular_tolerance)

    def extract_features_from_file(self, file_path, progress_callback=None):
        return extract_features_from_file(file_path, progress_callback=progress_callback, context=self._context)

    def calculate_pipeline_lengths(self, pipelines):
        if self._context is not None:
            self._context.report("Calculating source lengths", 0, len(pipelines))
        pipeline_data = []
        total_length_meters = 0.0
        total_length_miles = 0.0
        self._estimated_segments = 0

        for pipeline_index, pipeline in enumerate(pipelines):
            if self._context is not None:
                self._context.checkpoint()
            if self._context is not None:
                self._context.report("Calculating source lengths", pipeline_index, len(pipelines))
            length_meters = 0.0
            paths = coordinate_paths_for_pipeline(pipeline, context=self._context)

            if not paths:
                continue

            for coords in paths:
                path_length = 0.0
                if self._context is not None:
                    self._context.checkpoint()
                for i in range(len(coords) - 1):
                    if self._context is not None:
                        self._context.checkpoint()
                    try:
                        lon1, lat1 = coords[i]
                        lon2, lat2 = coords[i + 1]
                        if (lon1 == lon2 and lat1 == lat2
                                and math.isfinite(lon1) and -90 <= lat1 <= 90):
                            continue
                        _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
                        if not math.isfinite(distance):
                            raise ValueError("Non-finite geodesic length")
                        length_meters += abs(distance)
                        path_length += abs(distance)
                    except AnalysisCancelled:
                        raise
                    except Exception as e:
                        raise ValueError(f"Could not calculate length for pipeline {pipeline.get('name', '')}") from e
                if path_length > 0 and math.isfinite(self.segment_length) and self.segment_length > 0:
                    # Only need an exact estimate within the supported budget.
                    # Avoid overflow for tiny finite steps without losing source
                    # mileage to an optional overlap-workload calculation.
                    ceiling = MAX_ANALYSIS_SEGMENTS + 1
                    if path_length + 1e-8 >= self.segment_length * ceiling:
                        self._estimated_segments = ceiling
                    else:
                        self._estimated_segments = min(ceiling, self._estimated_segments +
                            math.floor((path_length + 1e-8) / self.segment_length))

            length_miles = length_meters / self.survey_mile

            pipeline_data.append(
                {
                    "Placemark_ID": pipeline.get("placemark_id") or "N/A",
                    "OBJECTID": pipeline.get("objectid", "N/A"),
                    "Name": pipeline.get("name", ""),
                    "Shape_Length": length_meters,
                    "pipelinelength": length_miles,
                }
            )

            total_length_meters += length_meters
            total_length_miles += length_miles

        return pipeline_data, total_length_meters, total_length_miles

    def segment_pipeline(self, coordinates):
        return segment_pipeline(self.geod, coordinates, self.segment_length, context=self._context)

    def find_parallel_segments(self, pipelines, progress_callback=None):
        return find_parallel_segments(
            pipelines,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
            context=self._context,
        )

    def calculate_overlap_results(self, pipelines, parallel_groups, progress_callback=None):
        return calculate_overlap_results(
            pipelines,
            parallel_groups,
            geod=self.geod,
            survey_mile_m=self.survey_mile,
            segment_length=self.segment_length,
            min_parallel_length=self.min_parallel_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
            context=self._context,
        )

    def compute_effective_length_by_clusters(self, pipelines, per_pipeline_total_meters, progress_callback=None):
        return compute_effective_length_by_clusters(
            pipelines,
            per_pipeline_total_meters,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
            context=self._context,
            min_parallel_length=self.min_parallel_length,
        )

    def analyze_complete(self, file_path, progress_callback=None, *, context=None, options=None):
        """Parse once, retaining the ordinary result and optional state breakdown."""
        options = options or AnalysisOptions()
        if not isinstance(options, AnalysisOptions):
            raise TypeError('options must be AnalysisOptions')
        callback_context = None
        if options.state_breakdown and progress_callback is not None:
            callback_context = CallbackExecutionContext(
                context if context is not None else ExecutionContext(), progress_callback)
            context = callback_context
            # Every stage now reports through the job context. An individual
            # overlap pass must not publish its local 100% to the public API.
            progress_callback = None
        try:
            parse_context = (ScopedExecutionContext(context, 0, .03)
                             if context is not None and options.state_breakdown else context)
            parsed = extract_features_from_file_with_diagnostics(
                file_path, progress_callback=progress_callback, context=parse_context)
            combined_context = (ScopedExecutionContext(context, .03, .43, 'Combined')
                                if context is not None and options.state_breakdown else context)
            combined = self.analyze_features(
                parsed.pipelines, parsed.placemarks, diagnostics=parsed.diagnostics,
                parsed_kml_files=parsed.parsed_kml_files, progress_callback=progress_callback,
                context=combined_context)
            if options.state_breakdown:
                # These samples belong to the completed combined pass. State
                # runs produce their own; retaining both doubles peak memory.
                for pipeline in parsed.pipelines:
                    pipeline.pop('segments', None)
                from pipeline_calculator.core.state_analysis import build_state_breakdown
                combined['geography'] = build_state_breakdown(
                    self, parsed.pipelines, combined, context=context)
                from pipeline_calculator.core.corridor_geometry import prepare_scope_visualizations
                # Prepare Combined visuals after state analysis so their optional
                # display diagnostics are not inherited as state input errors.
                combined = prepare_scope_visualizations(combined, context=context)
            if callback_context is not None:
                # Optional geography failures still yield a finished, explicitly
                # incomplete result. Cancellation raises before completion.
                callback_context.report('Complete', 1, 1, fraction=1.0)
            return combined
        except AnalysisCancelled:
            raise
        except Exception as exc:
            raise ValueError(f'Analysis failed: {exc}') from exc

    def analyze_features(self, pipelines, placemarks=None, *, diagnostics=None,
                         parsed_kml_files=None, progress_callback=None, context=None):
        """Analyze normalized features without XML round trips or identity changes."""
        previous_context = self._context
        self._context = context
        diagnostics = list(diagnostics or [])
        try:
            pipeline_data, total_meters, total_miles = self.calculate_pipeline_lengths(pipelines)

            overlap_results = None
            if len(pipelines) >= 2:
                try:
                    check_segment_workload(self._estimated_segments, context)
                    if self._estimated_segments > MAX_ANALYSIS_SEGMENTS:
                        raise ValueError('Analysis segment limit exceeded; split the dataset or increase segment length')
                    parallel_groups = self.find_parallel_segments(pipelines, progress_callback)
                    overlap_results = self.calculate_overlap_results(pipelines, parallel_groups, progress_callback)
                    eff_total_m = total_meters - overlap_results["savings_meters"]

                    eff_total_m = max(0.0, min(float(total_meters), float(eff_total_m)))
                    total_savings = max(0.0, float(total_meters) - eff_total_m)

                    overlap_results["effective_total_meters"] = eff_total_m
                    overlap_results["effective_total_miles"] = eff_total_m / self.survey_mile
                    overlap_results["savings_meters"] = total_savings
                    overlap_results["savings_miles"] = total_savings / self.survey_mile
                    overlap_results["savings_percentage"] = (
                        (total_savings / total_meters * 100) if total_meters > 0 else 0
                    )
                    overlap_results["computation_method"] = "qualified_segment_coverage_v2"
                except AnalysisCancelled:
                    raise
                except Exception as e:
                    diagnostics.append({
                        "level": "error",
                        "code": "overlap_analysis_failed",
                        "message": "Overlap calculation failed; adjusted mileage and savings are unavailable.",
                        "context": {"error": str(e)},
                    })
                    overlap_results = None

            if context is not None:
                context.report("Finalizing results")
            return {
                "pipelines": pipeline_data,
                "placemarks": list(placemarks or []),
                "total_meters": total_meters,
                "total_miles": total_miles,
                "overlap_analysis": overlap_results,
                "analysis_complete": not any(d.get("level") == "error" for d in diagnostics),
                "diagnostics": diagnostics,
                "parsed_kml_files": list(parsed_kml_files or []),
                "analysis_parameters": {
                    "detection_range": self.detection_range,
                    "min_parallel_length": self.min_parallel_length,
                    "segment_length": self.segment_length,
                    "angular_tolerance": self.angular_tolerance,
                },
            }
        except AnalysisCancelled:
            raise
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

        finally:
            self._context = previous_context
