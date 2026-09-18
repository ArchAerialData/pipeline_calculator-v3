"""Optional linked-source errors must remain explicit without weakening repair safety."""
from __future__ import annotations

import zipfile

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics
from pipeline_calculator.parsers.repair import RepairFailure
from pipeline_calculator.parsers.source import prepare_source


NS = "http://www.opengis.net/kml/2.2"
LINE = "<LineString><coordinates>-100,30 -100.01,30</coordinates></LineString>"
LINKS = ("<NetworkLink><Link><href>bad.kml</href></Link></NetworkLink>"
         "<NetworkLink><Link><href>healthy.kml</href></Link></NetworkLink>")


def document(name="Root", *, extra="", repairable=False):
    attribute = ' xsi:schemaLocation="schema.xsd"' if repairable else ""
    return (f'<kml xmlns="{NS}"><Document{attribute}><Placemark><name>{name}</name>'
            f'{LINE}</Placemark>{extra}</Document></kml>').encode()


def input_graph(tmp_path, container, bad, *, repairable=False):
    root = document(extra=LINKS, repairable=repairable)
    path = tmp_path / ("doc." + container)
    entries = {"doc.kml": root, "bad.kml": bad, "healthy.kml": document("Healthy")}
    if container == "kmz":
        with zipfile.ZipFile(path, "w") as archive:
            for name, contents in entries.items():
                archive.writestr(name, contents)
    else:
        for name, contents in entries.items():
            (tmp_path / name).write_bytes(contents)
    return path


@pytest.mark.parametrize("container", ["kml", "kmz"])
@pytest.mark.parametrize("bad", [
    b"<kml><broken>",
    b'<!DOCTYPE kml [<!ENTITY name "Untrusted">]>' + document("&name;"),
], ids=["xml-syntax", "unsupported-dtd"])
def test_optional_link_failure_preserves_available_paths_and_incomplete_status(tmp_path, container, bad):
    path = input_graph(tmp_path, container, bad)
    original = path.read_bytes()
    parsed = parse_kml_kmz_with_diagnostics(path)
    assert [p["name"] for p in parsed.pipelines] == ["Root", "Healthy"]
    assert all(p["coordinate_paths"] == [[(-100.0, 30.0), (-100.01, 30.0)]]
               for p in parsed.pipelines)
    failures = [d for d in parsed.diagnostics if d["code"] == "linked_kml_parse_error"]
    assert len(failures) == 1 and failures[0]["level"] == "error"
    assert failures[0]["context"]["source"].endswith("bad.kml")
    assert not any(name.endswith("bad.kml") for name in parsed.parsed_kml_files)
    session = prepare_source(path)
    assert not session.requires_repair and not session.can_save
    assert session.fresh_parse().pipelines == parsed.pipelines
    result = PipelineAnalyzer().analyze_complete(path)
    assert result["analysis_complete"] is False
    assert result["total_meters"] > 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("container", ["kml", "kmz"])
def test_repaired_root_never_certifies_a_failed_link_as_complete(tmp_path, container):
    path = input_graph(tmp_path, container, b"<!DOCTYPE kml>" + document(), repairable=True)
    original = path.read_bytes()
    with pytest.raises(RepairFailure) as caught:
        prepare_source(path)
    assert caught.value.category == "coverage"
    assert "unsupported_dtd" in {f["code"] for f in caught.value.findings}
    assert path.read_bytes() == original


@pytest.mark.parametrize("linked", [False, True], ids=["primary", "linked"])
@pytest.mark.parametrize("bad,category", [
    (b'<?xml version="1.0" encoding="unknown-encoding"?>' + document(), "policy"),
    (b"<kml>" + b"<Folder>" * 260 + b"</Folder>" * 260 + b"</kml>", "limit"),
], ids=["encoding-policy", "xml-complexity-limit"])
def test_policy_and_limit_failures_remain_fatal(tmp_path, linked, bad, category):
    if linked:
        path = input_graph(tmp_path, "kmz", bad)
    else:
        path = tmp_path / "doc.kml"
        path.write_bytes(bad)
    with pytest.raises(ValueError):
        parse_kml_kmz_with_diagnostics(path)
    with pytest.raises(RepairFailure) as caught:
        prepare_source(path)
    assert caught.value.category == category


@pytest.mark.parametrize("container", ["kml", "kmz"])
def test_ambiguous_linked_geometry_is_still_rejected(tmp_path, container):
    ambiguous = document().replace(
        b"</LineString>", b"<coordinates>-90,30 -90.01,30</coordinates></LineString>")
    path = input_graph(tmp_path, container, ambiguous)
    with pytest.raises(ValueError, match="complete pipeline geometry cannot be read safely"):
        parse_kml_kmz_with_diagnostics(path)
    with pytest.raises(RepairFailure) as caught:
        prepare_source(path)
    assert "ambiguous_coordinate_structure" in {f["code"] for f in caught.value.findings}


def test_primary_dtd_error_is_still_fatal(tmp_path):
    path = tmp_path / "doc.kml"
    path.write_bytes(b"<!DOCTYPE kml>" + document())
    with pytest.raises(ValueError, match="DTD/entity"):
        parse_kml_kmz_with_diagnostics(path)
    with pytest.raises(RepairFailure, match="DTD/entity"):
        prepare_source(path)
