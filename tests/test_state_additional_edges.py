"""End-to-end coverage beyond the mainland fixture suite's controlled networks.

Expected physical lengths come from saved XML and the native resource, never
from the application's partition ledger. These checks deliberately exercise
package/report behavior as well as geometry accounting.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import load_workbook
from pyproj import Geod
import pytest
from shapely import from_wkb
from shapely.geometry import Point, box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.geography import BoundaryDataset
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.export.package import export_analysis_package


GEOD = Geod(ellps="GRS80")
NS = {"k": "http://www.opengis.net/kml/2.2"}
RESOURCE = Path(__file__).resolve().parents[1] / "src/pipeline_calculator/data/states_2025.zip"


def _kmz(path, sources):
    """Write one source placemark for each group of independent coordinate paths."""
    root = ET.Element("kml", xmlns=NS["k"])
    document = ET.SubElement(root, "Document")
    for paths in sources:
        placemark = ET.SubElement(document, "Placemark", id="repeated-id")
        ET.SubElement(placemark, "name").text = "Repeated pipeline name"
        multi = ET.SubElement(placemark, "MultiGeometry")
        for points in paths:
            line = ET.SubElement(multi, "LineString")
            ET.SubElement(line, "coordinates").text = " ".join(
                f"{lon!r},{lat!r},0" for lon, lat in points)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", ET.tostring(root, encoding="utf-8"))
    return path


def _map_lines(path):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("doc.kml"))
    paths = []
    for node in root.findall(".//k:LineString/k:coordinates", NS):
        paths.append([tuple(map(float, item.split(",")[:2])) for item in node.text.split()])
    return root, paths


def _physical_length(path):
    return math.fsum(abs(GEOD.inv(*a, *b)[2])
                     for points in _map_lines(path)[1]
                     for a, b in zip(points, points[1:]))


def _raw_boundary(code):
    with ZipFile(RESOURCE) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        state = next(entry for entry in manifest["states"] if entry["code"] == code)
        return from_wkb(archive.read(state["file"]))


def _only_exit_distance(points, native_polygon):
    """Independent point predicate/root bracket for a known single-exit path.

    Reading native WKB bypasses application boundary normalization and clipping.
    A sampled ownership certificate prevents bisection from hiding reentry here.
    """
    first, last = points
    bearing, _, length = GEOD.inv(*first, *last)

    def inside(distance):
        return native_polygon.covers(Point(GEOD.fwd(*first, bearing, distance)[:2]))

    probes = [inside(length * index / 200) for index in range(201)]
    assert probes[0] and not probes[-1]
    transitions = [index for index in range(1, len(probes)) if probes[index] != probes[index - 1]]
    assert len(transitions) == 1, "This independent bracket requires exactly one exit"
    lower, upper = 0.0, length
    for _ in range(60):
        middle = (lower + upper) / 2
        if inside(middle):
            lower = middle
        else:
            upper = middle
    assert upper - lower < 1e-6
    return (lower + upper) / 2


def test_real_shared_only_analysis_exports_one_physical_line_and_no_state_maps(tmp_path):
    # This native TX/NM meridian interval has certified shared membership in the
    # bundled dataset; neither state has any exclusive physical line to export.
    points = [(-103.064732, 32.744215), (-103.064732, 32.75427)]
    source = _kmz(tmp_path / "shared_only.kmz", [[points]])
    expected = _physical_length(source)
    result = PipelineAnalyzer().analyze_complete(source, options=AnalysisOptions(True))
    geography = result["geography"]
    assert geography["status"] == "complete", geography["diagnostics"]
    assert geography["crossing_count"] == 0
    assert geography["reconciliation"]["passed"]
    assert geography["reconciliation"]["shared_meters"] == pytest.approx(expected, abs=1e-6)
    assert {state["state_code"] for state in geography["states"]} == {"NM", "TX"}
    assert len(geography["fragments"]) == 1
    fragment = geography["fragments"][0]
    assert fragment["kind"] == "shared"
    for state in geography["states"]:
        assert state["analysis_complete"]
        assert state["interior_meters"] == 0
        assert state["shared_allocation_meters"] == pytest.approx(expected / 2, abs=1e-6)
        assert state["interior_savings_meters"] == 0
        assert state["adjusted_total_meters"] == state["total_meters"]
        assert state["shared_overlap_status"] == "not_calculated"
        assert state["pipelines"][0]["shared_border_allocations"] == [
            {"fragment_id": fragment["id"], "allocated_meters": state["shared_allocation_meters"]}]

    package = export_analysis_package(result, tmp_path, source.name, include_json=True)
    assert sorted(str(path.relative_to(package)).replace("\\", "/")
                  for path in package.rglob("*") if path.is_file()) == [
        "Combined/analysis.kmz", "analysis.json", "analysis.xlsx"]
    assert not (package / "States").exists()
    combined_map = package / "Combined/analysis.kmz"
    root, lines = _map_lines(combined_map)
    assert len(lines) == 1
    assert _physical_length(combined_map) == pytest.approx(expected, abs=1e-6)
    assert "Shared Borders" in [node.text for node in root.findall(".//k:Folder/k:name", NS)]
    saved = json.loads((package / "analysis.json").read_text(encoding="utf-8"))
    assert len(saved["geography"]["fragments"]) == 1
    assert all("fragments" not in state for state in saved["geography"]["states"])
    workbook = load_workbook(package / "analysis.xlsx")
    assert workbook["Shared Borders"]["G2"].value == "Not calculated"
    assert workbook["Shared Borders"]["H2"].value == "Combined/analysis.kmz — Shared Borders"
    for row in workbook["State Summary"].iter_rows(min_row=2, max_row=3, values_only=True):
        assert row[2] == 0
        assert row[3] == row[4] == row[6]
        assert row[5] == 0 and row[7] == "complete"
    workbook.close()


def test_real_alaska_hawaii_multipart_and_dateline_outside_coverage_roundtrip(tmp_path):
    alaska = [(-149.9003, 61.2181), GEOD.fwd(-149.9003, 61.2181, 90, 500)[:2]]
    hawaii = [(-157.8583, 21.3069), GEOD.fwd(-157.8583, 21.3069, 90, 400)[:2]]
    # The positive-longitude start is in Alaska; this geodesic exits its coastal
    # boundary before crossing the dateline. The far endpoint is outside coverage.
    aleutian = [(179.8, 51.9), (-179.8, 51.9)]
    source = _kmz(tmp_path / "islands_dateline.kmz", [[alaska, hawaii, aleutian]])
    original_paths = _map_lines(source)[1]
    native_alaska, native_hawaii = _raw_boundary("AK"), _raw_boundary("HI")
    for points, polygon in ((alaska, native_alaska), (hawaii, native_hawaii)):
        bearing, _, length = GEOD.inv(*points[0], *points[-1])
        assert all(polygon.covers(Point(GEOD.fwd(*points[0], bearing, length * i / 100)[:2]))
                   for i in range(101))
    alaska_exit = _only_exit_distance(aleutian, native_alaska)
    lengths = [abs(GEOD.inv(*points[0], *points[-1])[2]) for points in original_paths]
    expected = {"AK": lengths[0] + alaska_exit, "HI": lengths[1]}

    result = PipelineAnalyzer().analyze_complete(source, options=AnalysisOptions(True))
    geography = result["geography"]
    assert geography["status"] == "complete", geography["diagnostics"]
    assert geography["crossing_count"] == 0  # Disconnected states and a coverage exit.
    assert result["total_meters"] == pytest.approx(math.fsum(lengths), abs=1e-6)
    reconciliation = geography["reconciliation"]
    assert reconciliation["passed"]
    assert reconciliation["unresolved_meters"] == reconciliation["shared_meters"] == 0
    assert reconciliation["outside_meters"] == pytest.approx(lengths[2] - alaska_exit, abs=.01)
    assert {state["state_code"] for state in geography["states"]} == set(expected)
    for state in geography["states"]:
        assert state["analysis_complete"]
        assert len(state["pipelines"]) == 1
        assert state["pipelines"][0]["source_id"] == 0
        assert state["total_meters"] == pytest.approx(expected[state["state_code"]], abs=.01)
        assert state["interior_savings_meters"] == 0
    assert {fragment["path_index"] for fragment in geography["fragments"]} == {0, 1, 2}
    assert {fragment["source_id"] for fragment in geography["fragments"]} == {0}

    package = export_analysis_package(result, tmp_path, source.name, include_json=True)
    for map_path, expected_length in [
        (package / "Combined/analysis.kmz", math.fsum(lengths)),
        (package / "States/Alaska/analysis.kmz", expected["AK"]),
        (package / "States/Hawaii/analysis.kmz", expected["HI"]),
    ]:
        assert _physical_length(map_path) == pytest.approx(expected_length, abs=.01)
        assert PipelineAnalyzer().analyze_complete(map_path)["total_meters"] == pytest.approx(
            expected_length, abs=.01)
        _, map_paths = _map_lines(map_path)
        assert all(abs(a[0] - b[0]) <= 180 for path in map_paths for a, b in zip(path, path[1:]))
    assert len(_map_lines(package / "States/Alaska/analysis.kmz")[1]) == 2
    assert len(_map_lines(package / "States/Hawaii/analysis.kmz")[1]) == 1


@pytest.mark.parametrize("failed_scope", ["combined", "BB"])
def test_overlap_engine_failure_is_isolated_and_exported_as_unavailable(tmp_path, monkeypatch, failed_scope):
    import pipeline_calculator.core.geography as geography_module

    boundaries = BoundaryDataset({"AA": box(-1, -1, 0, 1), "BB": box(0, -1, 1, 1)},
                                 {"AA": "Alpha", "BB": "Beta"})
    monkeypatch.setattr(geography_module, "load_boundaries", lambda: boundaries)
    source = _kmz(tmp_path / "scope_failure.kmz", [
        [[(-.003, 0), (.003, 0)]], [[(-.003, .00008), (.003, .00008)]]])
    analyzer = PipelineAnalyzer()
    original = analyzer.find_parallel_segments
    calls = []

    def fail_one_engine_scope(pipelines, *args, **kwargs):
        longitudes = [point[0] for item in pipelines for path in item["coordinate_paths"] for point in path]
        # Classify by distant interior endpoints, not cut endpoints whose last
        # coordinate bit can straddle zero without measurable foreign mileage.
        scope = ("combined" if min(longitudes) < -.001 and max(longitudes) > .001
                 else ("AA" if min(longitudes) < -.001 else "BB"))
        calls.append(scope)
        if scope == failed_scope:
            raise RuntimeError("Injected overlap engine failure")
        return original(pipelines, *args, **kwargs)

    monkeypatch.setattr(analyzer, "find_parallel_segments", fail_one_engine_scope)
    result = analyzer.analyze_complete(source, options=AnalysisOptions(True))
    assert calls == ["combined", "AA", "BB"]
    geography = result["geography"]
    assert geography["reconciliation"]["passed"]
    assert result["total_meters"] == pytest.approx(_physical_length(source), abs=1e-6)
    assert result["analysis_complete"] is (failed_scope != "combined")
    assert geography["analysis_complete"] is (failed_scope == "combined")
    for state in geography["states"]:
        failed = state["state_code"] == failed_scope
        assert state["analysis_complete"] is not failed
        assert state["total_meters"] > 0
        if failed:
            assert state["overlap_analysis"] is None
            assert state["interior_savings_meters"] is state["adjusted_total_meters"] is None
            assert any(item["code"] == "overlap_analysis_failed" for item in state["diagnostics"])
        else:
            assert state["interior_savings_meters"] > 0
            assert state["adjusted_total_meters"] < state["total_meters"]
            assert not any(item["code"] == "overlap_analysis_failed" for item in state["diagnostics"])

    package = export_analysis_package(result, tmp_path, source.name, include_json=True)
    saved = json.loads((package / "analysis.json").read_text(encoding="utf-8"))
    assert saved["geography"]["analysis_complete"] is (failed_scope == "combined")
    workbook = load_workbook(package / "analysis.xlsx")
    combined_savings = workbook["Pipeline Overlap Analysis"]["D2"].value
    assert (combined_savings == "Unavailable") is (failed_scope == "combined")
    for row in workbook["State Summary"].iter_rows(min_row=2, max_row=3, values_only=True):
        if row[1] == failed_scope:
            assert row[5:8] == ("Unavailable", "Unavailable", "incomplete")
        else:
            assert row[5] > 0 and row[6] < row[4] and row[7] == "complete"
    workbook.close()
    assert _physical_length(package / "Combined/analysis.kmz") == pytest.approx(_physical_length(source), abs=.001)
    for state in geography["states"]:
        state_map = package / "States" / state["state_name"] / "analysis.kmz"
        assert _physical_length(state_map) == pytest.approx(state["interior_meters"], abs=.001)
