from __future__ import annotations

from pathlib import Path

import pytest

import pipeline_calculator_v3 as legacy
from pipeline_calculator.export.xlsx import build_analysis_workbook


def test_build_analysis_workbook_has_expected_structure(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")

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

    analyzer = legacy.PipelineAnalyzer()
    analyzer.segment_length = 25.0
    analyzer.min_parallel_length = 100.0
    analyzer.detection_range = 30.0
    analyzer.angular_tolerance = 10.0
    results = analyzer.analyze_complete(str(kml_path))

    wb = build_analysis_workbook(results)

    assert wb.sheetnames == ["Pipeline Length Analysis", "Pipeline Overlap Analysis"]

    ws = wb["Pipeline Length Analysis"]
    assert [c.value for c in ws[1]] == [
        "Object ID (if available)",
        "Polyline Name (if available)",
        "Pipeline Lengths (US Survey)",
        "TOTAL MILEAGE",
    ]
    assert ws.cell(row=2, column=4).value == "=SUM(C2:C100000)"

    ws2 = wb["Pipeline Overlap Analysis"]
    assert [c.value for c in ws2[1]][:4] == [
        "Pipeline 1",
        "Pipeline 2",
        "Bundled Length (mi)",
        "TOTAL MILEAGE REMOVED",
    ]
    # CI/export expects a numeric savings value in D2.
    assert isinstance(ws2.cell(row=2, column=4).value, (int, float))
    assert ws2.cell(row=2, column=4).number_format == "0.000"


def test_build_analysis_workbook_header_fills() -> None:
    pytest.importorskip("openpyxl")

    # Minimal results, no overlap rows required for style validation.
    results = {
        "pipelines": [{"OBJECTID": "1", "Name": "A", "pipelinelength": 1.234}],
        "overlap_analysis": {"bundled_sections": [], "savings_miles": 0.0},
    }

    wb = build_analysis_workbook(results)
    ws = wb["Pipeline Length Analysis"]
    ws2 = wb["Pipeline Overlap Analysis"]

    # Header fills: gray by default, with column 3 yellow and column 4 green.
    def rgb(cell):
        return getattr(getattr(cell.fill, "fgColor", None), "rgb", None)

    assert rgb(ws.cell(row=1, column=1)) == "FFD9D9D9"
    assert rgb(ws.cell(row=1, column=3)) == "FFFFFF00"
    assert rgb(ws.cell(row=1, column=4)) == "FF00B050"

    assert rgb(ws2.cell(row=1, column=1)) == "FFD9D9D9"
    assert rgb(ws2.cell(row=1, column=3)) == "FFFFFF00"
    assert rgb(ws2.cell(row=1, column=4)) == "FF00B050"
