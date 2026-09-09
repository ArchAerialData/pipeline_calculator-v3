from __future__ import annotations

from pathlib import Path

from pipeline_calculator.core.analyzer import PipelineAnalyzer


def test_core_pipeline_analyzer_analyze_complete(tmp_path: Path) -> None:
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
    kml_path = tmp_path / "parallel.kml"
    kml_path.write_text(kml, encoding="utf-8")

    analyzer = PipelineAnalyzer(
        segment_length=25.0,
        min_parallel_length=100.0,
        detection_range=30.0,
        angular_tolerance=10.0,
    )
    res = analyzer.analyze_complete(str(kml_path))

    assert res["pipelines"]
    assert res["total_meters"] > 0.0
    assert res["overlap_analysis"] is not None

