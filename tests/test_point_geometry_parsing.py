"""Pin inventory counts Point geometries independently of pipeline paths."""
from __future__ import annotations

import zipfile

import pytest
from pyproj import Geod

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics


NS = "http://www.opengis.net/kml/2.2"
LINE = "<LineString><coordinates>-100,40 -99.99,40</coordinates></LineString>"
PIN = "<Point><coordinates>-100,40,0</coordinates></Point>"
RING = ("<LinearRing><coordinates>-100,40 -99.99,40 -99.99,40.01 -100,40"
        "</coordinates></LinearRing>")
POLYGON = f"<Polygon><outerBoundaryIs>{RING}</outerBoundaryIs></Polygon>"


def document(body):
    return f'<kml xmlns="{NS}"><Document>{body}</Document></kml>'


def write_input(tmp_path, body, container="kml"):
    path = tmp_path / f"input.{container}"
    if container == "kmz":
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc.kml", document(body))
    else:
        path.write_text(document(body), encoding="utf-8")
    return path


@pytest.mark.parametrize("container", ["kml", "kmz"])
def test_nested_multipart_pins_count_individually_without_counting_shape_vertices(tmp_path, container):
    body = (
        '<Style id="icon"><IconStyle><Icon><href>pin.png</href></Icon></IconStyle></Style>'
        '<Placemark id="pipe"><name>Shared feature</name><ExtendedData>'
        '<Data name="OBJECTID"><value>shared-id</value></Data></ExtendedData>'
        '<styleUrl>#icon</styleUrl><MultiGeometry>' + LINE + POLYGON + PIN +
        '<MultiGeometry>' + PIN + PIN + '</MultiGeometry></MultiGeometry></Placemark>'
        '<Placemark><name>Standalone shapes</name><MultiGeometry>' + LINE + RING +
        '</MultiGeometry></Placemark>'
    )
    path = write_input(tmp_path, body, container)
    parsed = parse_kml_kmz_with_diagnostics(path)

    # Three co-located pins remain three existing Point geometries. Their shared
    # source name/OBJECTID is not replaced or used to deduplicate the inventory.
    assert parsed.placemarks == [
        {"Placemark_ID": "shared-id", "Name": "Shared feature", "Count": 1}
    ] * 3
    assert len(parsed.pipelines) == 2
    assert parsed.pipelines[0]["placemark_id"] == "pipe"
    assert all(pipeline["coordinate_paths"] == [[(-100.0, 40.0), (-99.99, 40.0)]]
               for pipeline in parsed.pipelines)
    assert not any(item["level"] != "info" for item in parsed.diagnostics)


def test_points_alongside_polygons_and_points_in_separate_features_are_all_counted(tmp_path):
    body = ('<Placemark><name>Boundary and pins</name><MultiGeometry>' +
            POLYGON + PIN + PIN + '</MultiGeometry></Placemark>' +
            '<Placemark><name>Another pin</name>' + PIN + '</Placemark>')
    result = PipelineAnalyzer().analyze_complete(write_input(tmp_path, body))

    assert result["analysis_complete"] is True
    assert result["pipelines"] == []
    assert result["total_meters"] == 0
    assert [row["Name"] for row in result["placemarks"]] == [
        "Boundary and pins", "Boundary and pins", "Another pin",
    ]
    assert [row["Placemark_ID"] for row in result["placemarks"]] == ["PM_1", "PM_2", "PM_3"]
    assert sum(row["Count"] for row in result["placemarks"]) == 3


@pytest.mark.parametrize("point,code,legacy_slot", [
    ("<Point/>", "missing_point_coordinate", False),
    ("<Point><coordinates>  </coordinates></Point>", "missing_point_coordinate", False),
    ("<Point><coordinates>-100,40 -99,40</coordinates></Point>", "ambiguous_point_coordinate", True),
    ("<Point><coordinates>-100,40 bad</coordinates></Point>", "ambiguous_point_coordinate", False),
    ("<Point><coordinates>999,40</coordinates></Point>", "invalid_coordinate", False),
    ("<Point><coordinates>NaN,40</coordinates></Point>", "invalid_coordinate", False),
    ("<Point><coordinates>-100,40,NaN</coordinates></Point>", "invalid_coordinate", True),
    ("<Point><coordinates>-100,40,bad</coordinates></Point>", "invalid_coordinate", True),
    ("<Point><coordinates>-100,40,0,5</coordinates></Point>", "invalid_coordinate", True),
])
def test_malformed_points_do_not_manufacture_pins_or_hide_later_valid_points(tmp_path, point, code, legacy_slot):
    body = ('<Placemark><MultiGeometry>' + point + PIN + '</MultiGeometry></Placemark>' +
            '<Placemark>' + LINE + '</Placemark>')
    path = write_input(tmp_path, body)
    parsed = parse_kml_kmz_with_diagnostics(path)
    result = PipelineAnalyzer().analyze_complete(path)

    assert len(parsed.placemarks) == 1
    assert parsed.placemarks[0]["Count"] == 1
    assert any(item["code"] == code and item["level"] == "error" for item in parsed.diagnostics)
    assert result["analysis_complete"] is False
    # Generated pipeline names are stable even where a malformed pin had been
    # accepted by the historical parser, or later valid pins were invisible.
    assert parsed.pipelines[0]["name"] == ("Item_2" if legacy_slot else "Item_1")


def test_invalid_pin_with_valid_pipeline_remains_an_actionable_error(tmp_path):
    body = '<Placemark><MultiGeometry>' + LINE + '<Point/>' + PIN + '</MultiGeometry></Placemark>'
    result = PipelineAnalyzer().analyze_complete(write_input(tmp_path, body))

    assert len(result["pipelines"]) == len(result["placemarks"]) == 1
    assert result["total_meters"] > 0
    assert result["analysis_complete"] is False
    assert any(item["code"] == "missing_point_coordinate" and item["level"] == "error"
               for item in result["diagnostics"])


def test_expanded_point_inventory_does_not_renumber_unnamed_pipelines_or_paths(tmp_path):
    body = ('<Placemark><MultiGeometry>' + PIN + PIN + '</MultiGeometry></Placemark>' +
            '<Placemark><MultiGeometry>' + LINE + LINE + PIN + '</MultiGeometry></Placemark>' +
            '<Placemark><MultiGeometry><Point/>' + PIN + '</MultiGeometry></Placemark>' +
            '<Placemark>' + LINE + '</Placemark>' +
            '<Placemark><Point><coordinates>-100,40 -99,40</coordinates></Point></Placemark>' +
            '<Placemark>' + LINE + '</Placemark>')
    parsed = parse_kml_kmz_with_diagnostics(write_input(tmp_path, body))

    assert [(pipeline["id"], pipeline["name"]) for pipeline in parsed.pipelines] == [
        (0, "Item_2"), (1, "Item_3"), (2, "Item_5"),
    ]
    assert [len(pipeline["coordinate_paths"]) for pipeline in parsed.pipelines] == [2, 1, 1]
    assert [row["Name"] for row in parsed.placemarks] == ["Item_1", "Item_1", "Item_2", "Item_3"]


def test_feature_naming_and_pin_counts_continue_across_linked_documents(tmp_path):
    root = write_input(tmp_path, '<Placemark><MultiGeometry>' + PIN + PIN +
                       '</MultiGeometry></Placemark><NetworkLink><Link><href>child.kml</href>'
                       '</Link></NetworkLink>')
    (tmp_path / "child.kml").write_text(document('<Placemark><MultiGeometry>' + LINE +
                                              PIN + '</MultiGeometry></Placemark>'), encoding="utf-8")
    parsed = parse_kml_kmz_with_diagnostics(root)

    assert len(parsed.placemarks) == 3
    assert [pipeline["name"] for pipeline in parsed.pipelines] == ["Item_2"]
    assert len(parsed.parsed_kml_files) == 2


@pytest.mark.parametrize("coordinates", [
    "-100,40 -99.99,40 -99.99,40.01 -100,40.01 -100,40",  # A closed pipeline.
    "-100,40.02 -100,40 -99.98,40 -99.98,40.01 -99.99,40.01 -99.99,39.99",  # An open crossing.
])
def test_closed_and_crossing_lines_remain_complete_mileage_not_pin_or_polygon_inventory(tmp_path, coordinates):
    line = f"<LineString><coordinates>{coordinates}</coordinates></LineString>"
    body = '<Placemark><MultiGeometry>' + line + PIN + POLYGON + '</MultiGeometry></Placemark>'
    parsed = parse_kml_kmz_with_diagnostics(write_input(tmp_path, body))
    expected_path = [tuple(map(float, token.split(","))) for token in coordinates.split()]
    expected_meters = Geod(ellps="GRS80").line_length(*zip(*expected_path))
    _, meters, _ = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)

    assert parsed.pipelines[0]["coordinate_paths"] == [expected_path]
    assert len(parsed.placemarks) == 1
    assert meters == pytest.approx(expected_meters, abs=1e-6)


def test_large_polygon_inventory_has_bounded_information_not_per_feature_warnings(tmp_path):
    body = ('<Placemark>' + LINE + '</Placemark>' +
            ('<Placemark>' + POLYGON + '</Placemark>') * 10001 +
            '<Placemark><MultiGeometry>' + LINE + POLYGON + '<Model/></MultiGeometry></Placemark>')
    result = PipelineAnalyzer().analyze_complete(write_input(tmp_path, body, "kmz"))

    assert result["analysis_complete"] is True
    assert len(result["pipelines"]) == 2
    assert result["placemarks"] == []
    assert len(result["diagnostics"]) == 3  # Primary selection and two exclusion categories.
    standalone = next(item for item in result["diagnostics"] if item["code"] == "unsupported_geometry")
    mixed = next(item for item in result["diagnostics"] if item["code"] == "ignored_non_centerline_geometry")
    assert standalone["context"]["feature_count"] == 10001
    assert standalone["context"]["geometry_types"] == ["LinearRing", "Polygon"]
    assert mixed["context"]["feature_count"] == 1
    assert mixed["context"]["geometry_types"] == ["LinearRing", "Model", "Polygon"]
    assert all(item["level"] == "info" for item in result["diagnostics"])
