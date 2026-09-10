from __future__ import annotations

from pipeline_calculator.gui.state import AnalysisParameters


def test_analysis_parameters_from_strings_defaults_and_clamps() -> None:
    params, corrections = AnalysisParameters.from_strings(
        "",  # detection: default
        "not-a-number",  # min_parallel: default
        "-1",  # segment_length: valid parse but clamped
        "999",  # angular: valid parse but clamped
    )

    assert params.detection_range == 15
    assert params.min_parallel_length == 200
    assert params.segment_length == 1
    assert params.angular_tolerance == 90

    # Visible fields must match the parameters actually used, including clamps.
    assert corrections == {"detection_range": "15", "min_parallel_length": "200",
                           "segment_length": "1", "angular_tolerance": "90"}


def test_analysis_parameters_parses_commas() -> None:
    params, corrections = AnalysisParameters.from_strings("1,234.5", "10", "5", "15")
    assert corrections == {}
    assert params.detection_range == 1234.5

