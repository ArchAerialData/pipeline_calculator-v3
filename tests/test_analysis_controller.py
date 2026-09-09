from __future__ import annotations

from pathlib import Path

from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController, analyze_file
from pipeline_calculator.gui.state import AnalysisParameters


def _write_two_pipeline_kml(path: Path) -> None:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>A</name>
      <LineString>
        <coordinates>-100.0,40.0,0 -99.997,40.0,0</coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>B</name>
      <LineString>
        <coordinates>-100.0,40.00009,0 -99.997,40.00009,0</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    path.write_text(kml, encoding="utf-8")


def test_analyze_file_returns_schema(tmp_path: Path) -> None:
    kml_path = tmp_path / "parallel.kml"
    _write_two_pipeline_kml(kml_path)

    params = AnalysisParameters(detection_range=30.0, min_parallel_length=100.0, segment_length=25.0, angular_tolerance=10.0)
    results = analyze_file(str(kml_path), params)

    assert "pipelines" in results
    assert "placemarks" in results
    assert "total_miles" in results
    assert "overlap_analysis" in results
    assert results["analysis_parameters"]["detection_range"] == 30.0


def test_analysis_controller_job_completes(tmp_path: Path) -> None:
    kml_path = tmp_path / "parallel.kml"
    _write_two_pipeline_kml(kml_path)

    params = AnalysisParameters(detection_range=30.0, min_parallel_length=100.0, segment_length=25.0, angular_tolerance=10.0)
    job = AnalysisController().start(str(kml_path), params)

    assert job.done.wait(10), "analysis job did not complete in time"
    assert job.error is None
    assert job.result is not None
    assert len(job.result.get("pipelines", [])) == 2

