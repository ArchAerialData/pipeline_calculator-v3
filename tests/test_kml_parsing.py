from __future__ import annotations

from pathlib import Path

import pytest

import pipeline_calculator_v3 as legacy
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz


def test_extract_features_from_kml(tmp_path: Path) -> None:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Pipeline A</name>
      <ExtendedData>
        <Data name="OBJECTID"><value>123</value></Data>
      </ExtendedData>
      <LineString>
        <coordinates>
          -100.0,40.0,0 -100.001,40.0,0
        </coordinates>
      </LineString>
    </Placemark>

    <Placemark>
      <name>Pipeline B</name>
      <ExtendedData>
        <SchemaData>
          <SimpleData name="OBJECTID">456</SimpleData>
        </SchemaData>
      </ExtendedData>
      <LineString>
        <coordinates>-100.0,40.0001,0 -100.001,40.0001,0</coordinates>
      </LineString>
    </Placemark>

    <Placemark>
      <name>POI 1</name>
      <ExtendedData>
        <Data name="OBJECTID"><value>POI-9</value></Data>
      </ExtendedData>
      <Point><coordinates>-100.0,40.0,0</coordinates></Point>
    </Placemark>

    <Placemark>
      <name>Invalid Feature</name>
      <Point><coordinates>999,999,0</coordinates></Point>
    </Placemark>
  </Document>
</kml>
"""
    kml_path = tmp_path / "sample.kml"
    kml_path.write_text(kml, encoding="utf-8")

    pipelines, placemarks = parse_kml_kmz(str(kml_path))

    assert len(pipelines) == 2
    assert {p["name"] for p in pipelines} == {"Pipeline A", "Pipeline B"}
    assert {p["objectid"] for p in pipelines} == {"123", "456"}
    assert all(len(p["coordinates"]) >= 2 for p in pipelines)

    assert len(placemarks) == 1
    assert placemarks[0]["Name"] == "POI 1"
    assert placemarks[0]["Placemark_ID"] == "POI-9"

    # Backwards-compatible analyzer wrapper.
    analyzer = legacy.PipelineAnalyzer()
    pipelines2, placemarks2 = analyzer.extract_features_from_file(str(kml_path))
    assert pipelines2 == pipelines
    assert placemarks2 == placemarks


def test_extract_features_from_invalid_file_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.kml"
    bad_path.write_text("not xml", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_kml_kmz(str(bad_path))
