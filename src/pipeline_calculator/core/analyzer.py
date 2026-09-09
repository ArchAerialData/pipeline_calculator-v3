from __future__ import annotations

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

        self.geod = geod
        self.survey_mile = float(survey_mile)
        self.detection_range = float(detection_range)
        self.min_parallel_length = float(min_parallel_length)
        self.segment_length = float(segment_length)
        self.angular_tolerance = float(angular_tolerance)

    def extract_features_from_file(self, file_path, progress_callback=None):
        return extract_features_from_file(file_path, progress_callback=progress_callback)

    def calculate_pipeline_lengths(self, pipelines):
        pipeline_data = []
        total_length_meters = 0.0
        total_length_miles = 0.0

        for pipeline in pipelines:
            length_meters = 0.0
            paths = coordinate_paths_for_pipeline(pipeline)

            if not paths:
                continue

            for coords in paths:
                for i in range(len(coords) - 1):
                    try:
                        lon1, lat1 = coords[i]
                        lon2, lat2 = coords[i + 1]
                        _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
                        length_meters += abs(distance)
                    except Exception as e:
                        print(f"Warning: Error calculating distance for pipeline {pipeline.get('name', '')}: {str(e)}")
                        continue

            length_miles = length_meters / self.survey_mile

            pipeline_data.append(
                {
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
        return segment_pipeline(self.geod, coordinates, self.segment_length)

    def find_parallel_segments(self, pipelines, progress_callback=None):
        return find_parallel_segments(
            pipelines,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
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
        )

    def analyze_complete(self, file_path, progress_callback=None):
        """Complete analysis of KMZ/KML file."""
        try:
            parsed = extract_features_from_file_with_diagnostics(file_path, progress_callback=progress_callback)
            pipelines = parsed.pipelines
            placemarks = parsed.placemarks

            pipeline_data, total_meters, total_miles = self.calculate_pipeline_lengths(pipelines)

            overlap_results = None
            if len(pipelines) >= 2:
                try:
                    parallel_groups = self.find_parallel_segments(pipelines, progress_callback)
                    overlap_results = self.calculate_overlap_results(pipelines, parallel_groups, progress_callback)
                    per_pipe_totals = [d["Shape_Length"] for d in pipeline_data]
                    eff_total_m = self.compute_effective_length_by_clusters(pipelines, per_pipe_totals, progress_callback)

                    eff_total_m = max(0.0, min(float(total_meters), float(eff_total_m)))
                    total_savings = max(0.0, float(total_meters) - eff_total_m)

                    overlap_results["effective_total_meters"] = eff_total_m
                    overlap_results["effective_total_miles"] = eff_total_m / self.survey_mile
                    overlap_results["savings_meters"] = total_savings
                    overlap_results["savings_miles"] = total_savings / self.survey_mile
                    overlap_results["savings_percentage"] = (
                        (total_savings / total_meters * 100) if total_meters > 0 else 0
                    )
                    overlap_results["computation_method"] = "clustered_segments_v1"
                except Exception as e:
                    print(f"Warning: Overlap analysis failed: {str(e)}")
                    overlap_results = None

            return {
                "pipelines": pipeline_data,
                "placemarks": placemarks,
                "total_meters": total_meters,
                "total_miles": total_miles,
                "overlap_analysis": overlap_results,
                "diagnostics": parsed.diagnostics,
                "parsed_kml_files": parsed.parsed_kml_files,
                "analysis_parameters": {
                    "detection_range": self.detection_range,
                    "min_parallel_length": self.min_parallel_length,
                    "segment_length": self.segment_length,
                    "angular_tolerance": self.angular_tolerance,
                },
            }
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

