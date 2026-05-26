from __future__ import annotations

import zipfile
from pathlib import Path

import pipeline_calculator_v3 as legacy
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz, parse_kml_kmz_with_diagnostics


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


def test_extract_features_from_kmz_multigeometry_linestring_paths(tmp_path: Path) -> None:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Multipart Pipeline</name>
      <MultiGeometry>
        <LineString>
          <coordinates>-100.0,40.0,0 -99.999,40.0,0</coordinates>
        </LineString>
        <LineString>
          <coordinates>-90.0,40.0,0 -89.999,40.0,0</coordinates>
        </LineString>
      </MultiGeometry>
    </Placemark>
  </Document>
</kml>
"""
    kmz_path = tmp_path / "multipart.kmz"
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)

    pipelines, placemarks = parse_kml_kmz(str(kmz_path))

    assert placemarks == []
    assert len(pipelines) == 1
    assert pipelines[0]["name"] == "Multipart Pipeline"
    assert len(pipelines[0]["coordinate_paths"]) == 2
    assert pipelines[0]["coordinates"] == pipelines[0]["coordinate_paths"][0]


def test_multipart_pipeline_length_excludes_gap_between_paths(tmp_path: Path) -> None:
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Multipart Pipeline</name>
      <MultiGeometry>
        <LineString>
          <coordinates>-100.0,40.0,0 -99.999,40.0,0</coordinates>
        </LineString>
        <LineString>
          <coordinates>-90.0,40.0,0 -89.999,40.0,0</coordinates>
        </LineString>
      </MultiGeometry>
    </Placemark>
  </Document>
</kml>
"""
    kmz_path = tmp_path / "multipart.kmz"
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)

    analyzer = PipelineAnalyzer()
    pipelines, _ = parse_kml_kmz(str(kmz_path))
    pipeline_data, total_meters, _ = analyzer.calculate_pipeline_lengths(pipelines)

    path_1 = pipelines[0]["coordinate_paths"][0]
    path_2 = pipelines[0]["coordinate_paths"][1]
    _, _, first_path_m = analyzer.geod.inv(path_1[0][0], path_1[0][1], path_1[1][0], path_1[1][1])
    _, _, second_path_m = analyzer.geod.inv(path_2[0][0], path_2[0][1], path_2[1][0], path_2[1][1])
    _, _, gap_m = analyzer.geod.inv(path_1[-1][0], path_1[-1][1], path_2[0][0], path_2[0][1])
    expected_meters = abs(first_path_m) + abs(second_path_m)

    assert len(pipeline_data) == 1
    assert abs(total_meters - expected_meters) < 1e-6
    assert total_meters < abs(gap_m)


def test_kmz_prefers_root_doc_kml_over_archive_order(tmp_path: Path) -> None:
    kmz_path = tmp_path / "multi.kmz"
    earlier = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Earlier</name><LineString><coordinates>-100,40,0 -99.999,40,0</coordinates></LineString></Placemark>
</Document></kml>
"""
    doc = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Doc</name><LineString><coordinates>-90,40,0 -89.999,40,0</coordinates></LineString></Placemark>
</Document></kml>
"""
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.kml", earlier)
        zf.writestr("doc.kml", doc)

    result = parse_kml_kmz_with_diagnostics(str(kmz_path))

    assert [p["name"] for p in result.pipelines] == ["Doc"]
    assert any(d["code"] == "unparsed_kml_file" for d in result.diagnostics)


def test_kmz_follows_local_network_link(tmp_path: Path) -> None:
    kmz_path = tmp_path / "linked.kmz"
    main = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <NetworkLink><name>Local Link</name><Link><href>files/linked.kml</href></Link></NetworkLink>
</Document></kml>
"""
    linked = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>LinkedLine</name><LineString><coordinates>-100,40,0 -99.999,40,0</coordinates></LineString></Placemark>
</Document></kml>
"""
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", main)
        zf.writestr("files/linked.kml", linked)

    result = parse_kml_kmz_with_diagnostics(str(kmz_path))

    assert [p["name"] for p in result.pipelines] == ["LinkedLine"]
    assert result.parsed_kml_files == ["doc.kml", "files/linked.kml"]


def test_remote_network_link_is_reported_not_fetched(tmp_path: Path) -> None:
    kmz_path = tmp_path / "remote.kmz"
    main = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <NetworkLink><name>Remote Link</name><Link><href>https://example.com/linked.kml</href></Link></NetworkLink>
</Document></kml>
"""
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", main)

    result = PipelineAnalyzer().analyze_complete(str(kmz_path))

    assert result["pipelines"] == []
    assert result["total_miles"] == 0.0
    assert any(d["code"] == "remote_network_link_skipped" for d in result["diagnostics"])


def test_gx_track_and_multitrack_are_pipeline_paths(tmp_path: Path) -> None:
    kml_path = tmp_path / "tracks.kml"
    kml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
    <Placemark>
      <name>Track A</name>
      <gx:Track>
        <gx:coord>-100 40 0</gx:coord>
        <gx:coord>-99.999 40 0</gx:coord>
      </gx:Track>
    </Placemark>
    <Placemark>
      <name>MultiTrack B</name>
      <gx:MultiTrack>
        <gx:Track>
          <gx:coord>-90 40 0</gx:coord>
          <gx:coord>-89.999 40 0</gx:coord>
        </gx:Track>
        <gx:Track>
          <gx:coord>-80 40 0</gx:coord>
          <gx:coord>-79.999 40 0</gx:coord>
        </gx:Track>
      </gx:MultiTrack>
    </Placemark>
  </Document>
</kml>
""",
        encoding="utf-8",
    )

    pipelines, _ = parse_kml_kmz(str(kml_path))

    assert [p["name"] for p in pipelines] == ["Track A", "MultiTrack B"]
    assert len(pipelines[0]["coordinate_paths"]) == 1
    assert len(pipelines[1]["coordinate_paths"]) == 2


def test_polygon_only_kmz_is_not_counted_as_pipeline_mileage(tmp_path: Path) -> None:
    kmz_path = tmp_path / "polygon.kmz"
    polygon = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Area</name><Polygon><outerBoundaryIs><LinearRing>
    <coordinates>-100,40,0 -99.999,40,0 -99.999,40.001,0 -100,40,0</coordinates>
  </LinearRing></outerBoundaryIs></Polygon></Placemark>
</Document></kml>
"""
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", polygon)

    result = PipelineAnalyzer().analyze_complete(str(kmz_path))

    assert result["pipelines"] == []
    assert result["total_miles"] == 0.0
    assert any(d["code"] == "unsupported_geometry" for d in result["diagnostics"])


def test_invalid_coordinates_are_reported_in_diagnostics(tmp_path: Path) -> None:
    kml_path = tmp_path / "invalid_coords.kml"
    kml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Bad Line</name><LineString><coordinates>-100,40,0 999,999,0 bad</coordinates></LineString></Placemark>
</Document></kml>
""",
        encoding="utf-8",
    )

    result = parse_kml_kmz_with_diagnostics(str(kml_path))

    assert result.pipelines == []
    assert any(d["code"] == "invalid_coordinate" for d in result.diagnostics)
    assert any(d["code"] == "short_linestring" for d in result.diagnostics)
