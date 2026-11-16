"""Original user KMZ: Unix ZIP metadata must not prevent ordinary import."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from pipeline_calculator.parsers.source import prepare_source
from pipeline_calculator.core.placemarks import summarize_placemarks


FIXTURE = Path(__file__).parent / "fixtures/geography/antero_midstream_data.kmz"


def test_original_antero_archive_preserves_every_source_path_without_repair():
    expected = json.loads(FIXTURE.with_suffix(".expected.json").read_text(encoding="utf-8"))
    original = FIXTURE.read_bytes()
    assert len(original) == expected["bytes"]
    assert hashlib.sha256(original).hexdigest() == expected["sha256"]
    with ZipFile(FIXTURE) as archive:
        assert len(archive.infolist()) == expected["archive_member_count"]
        assert all(entry.extra.startswith(bytes.fromhex(expected["unix_extra_hex"]))
                   for entry in archive.infolist())
        data = archive.read(expected["primary"])
    assert hashlib.sha256(data).hexdigest() == expected["kml_sha256"]
    root = ET.fromstring(data)
    counts = Counter(e.tag for e in root.iter())
    assert {key: counts[key] for key in expected["xml_counts"]} == expected["xml_counts"]

    # Independently read source geometry; do not derive the oracle from our parser.
    paths_by_feature = []
    empty = []
    for ordinal, placemark in enumerate(root.iter("Placemark"), 1):
        paths = [[tuple(map(float, value.split(",")[:2]))
                  for value in line.findtext("coordinates", "").split()]
                 for line in placemark.iter("LineString")]
        if paths:
            paths_by_feature.append(paths)
        elif not list(placemark.iter("Point")):
            empty.append({"feature_ordinal": ordinal, "name": placemark.findtext("name"),
                          "objectid": placemark.findtext('.//SimpleData[@name="OBJECTID"]')})
    assert empty == expected["metadata_only_features"]
    assert sum(len(path) for paths in paths_by_feature for path in paths) == expected["line_vertex_count"]

    session = prepare_source(FIXTURE)
    assert not session.requires_repair
    assert session.report["rules"] == []
    assert session.report["status"] == "not_needed"
    assert session.report["source_sha256"] == expected["sha256"]
    assert all(d["original_sha256"] == d["effective_sha256"] for d in session.report["documents"])
    parsed = session.fresh_parse()
    assert len(parsed.pipelines) == expected["pipeline_count"]
    assert len(parsed.placemarks) == expected["xml_counts"]["Point"]
    assert [p["coordinate_paths"] for p in parsed.pipelines] == paths_by_feature
    assert parsed.parsed_kml_files == [expected["primary"]]
    summary = summarize_placemarks(parsed.placemarks)
    assert summary['total'] == 7275
    assert summary['folder_count'] == 6
    assert [(g['name'], g['count']) for g in summary['groups']] == [
        ('Antero Waterline Facility', 4192), ('Antero Gas Pipeline Facility', 1906),
        ('Interchange Locations', 420), ('Launcher Receiver', 477),
        ('Antero Pads', 250), ('Compressor Stations', 30),
    ]
    assert all(g['folder_path'] == [g['name']] and g['source_kml'] == expected['primary']
               for g in summary['groups'])
    assert Counter(p['folder_name'] for p in parsed.pipelines) == {
        'Antero Gas Pipelines': 639, 'Antero Water Line': 321,
    }
    assert not any(d["level"] == "error" for d in parsed.diagnostics)
    warnings = [d for d in parsed.diagnostics if d["level"] == "warning"]
    assert [d["code"] for d in warnings] == ["no_supported_geometry"] * len(empty)
    assert [d["context"]["feature_name"] for d in warnings] == [e["name"] for e in empty]
    assert FIXTURE.read_bytes() == original
