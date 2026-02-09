from __future__ import annotations

import zipfile
from pathlib import Path

import pipeline_calculator_v3 as legacy
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz


def test_extract_features_from_kmz(tmp_path: Path) -> None:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Pipeline A</name>
      <LineString>
        <coordinates>-100.0,40.0,0 -100.001,40.0,0</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    kmz_path = tmp_path / "sample.kmz"
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)

    pipelines, placemarks = parse_kml_kmz(str(kmz_path))

    assert len(pipelines) == 1
    assert pipelines[0]["name"] == "Pipeline A"
    assert placemarks == []

    # Backwards-compatible analyzer wrapper.
    analyzer = legacy.PipelineAnalyzer()
    pipelines2, placemarks2 = analyzer.extract_features_from_file(str(kmz_path))
    assert pipelines2 == pipelines
    assert placemarks2 == placemarks
