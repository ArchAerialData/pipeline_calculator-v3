from __future__ import annotations

from pathlib import Path

import pipeline_calculator_v3 as pc


def _write_kml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_analyze_complete_two_parallel_pipelines(tmp_path: Path) -> None:
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
    kml_path = _write_kml(tmp_path, "parallel.kml", kml)

    analyzer = pc.PipelineAnalyzer()
    analyzer.segment_length = 25.0
    analyzer.min_parallel_length = 100.0
    analyzer.detection_range = 30.0
    analyzer.angular_tolerance = 10.0

    res = analyzer.analyze_complete(str(kml_path))

    assert res["pipelines"]
    assert res["total_meters"] > 0.0
    assert res["overlap_analysis"] is not None
    ov = res["overlap_analysis"]
    assert ov["effective_total_meters"] >= 0.0
    assert ov["savings_meters"] >= 0.0
    assert ov["savings_percentage"] >= 0.0


def test_analyze_complete_smoke_real_kmz() -> None:
    kmz_path = Path("test_data") / "Brazos_NGL and Delaware_Gas combined.kmz"
    if not kmz_path.exists():
        # Repo fixture missing; nothing to validate.
        return

    analyzer = pc.PipelineAnalyzer()
    # Coarsen to keep runtime reasonable while still exercising the code paths.
    analyzer.segment_length = 50.0
    analyzer.min_parallel_length = 200.0
    analyzer.detection_range = 30.0

    res = analyzer.analyze_complete(str(kmz_path))
    assert res["pipelines"]
    assert res["total_meters"] > 0.0

