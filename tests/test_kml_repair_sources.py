from __future__ import annotations

import json
from pathlib import Path
import stat
import zipfile

import pytest

from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.parsers import source
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics
from pipeline_calculator.parsers.repair import RepairFailure


NS = "http://www.opengis.net/kml/2.2"
LINE = '<Placemark id="line"><name>{}</name><LineString><coordinates>-100,30,7 -100.01,30,9</coordinates></LineString></Placemark>'


def kml(name="Pipeline", *, broken=False, extra=""):
    metadata = f' xsi:schemaLocation="{NS} schema.xsd"' if broken else ""
    return (f'<kml xmlns="{NS}"><Document{metadata}>{LINE.format(name)}{extra}</Document></kml>').encode()


def kmz(path, entries, *, comment=b""):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = comment
        for name, data in entries:
            archive.writestr(name, data)
    return path


def link(target):
    return f'<NetworkLink><Link><href>{target}</href></Link></NetworkLink>'


def test_valid_ordinary_input_keeps_parser_contract_and_fresh_baseline(tmp_path):
    path = tmp_path / "system.kml"
    path.write_bytes(kml().replace(f' xmlns="{NS}"'.encode(), b""))
    ordinary = parse_kml_kmz_with_diagnostics(path)
    session = source.prepare_source(path)
    assert not session.requires_repair
    assert session.fresh_parse() == ordinary
    first = session.fresh_parse()
    first.pipelines[0]["coordinate_paths"][0].clear()
    first.pipelines[0]["segments"] = ["analysis cache"]
    path.unlink()
    assert session.fresh_parse() == ordinary
    assert not session.can_save


def test_candidate_requires_approval_and_report_cannot_mutate_session(tmp_path):
    path = tmp_path / "broken.kml"
    original = kml(broken=True)
    path.write_bytes(original)
    session = source.prepare_source(path)
    assert session.requires_repair and not session.verified
    assert session.report["status"] == "eligible"
    with pytest.raises(RepairFailure, match="approved"):
        session.fresh_parse()
    report = session.report
    report["rules"].clear()
    session.approve()
    assert session.verified and session.can_save
    assert session.report["status"] == "verified"
    assert session.report["rules"] == ["missing_xsi_schema_namespace_v1"]
    assert session.fresh_parse().pipelines[0]["name"] == "Pipeline"
    assert path.read_bytes() == original
    assert str(tmp_path) not in json.dumps(session.report)


def test_linked_repair_uses_original_graph_order_and_ignores_unreachable_xml(tmp_path):
    path = kmz(tmp_path / "network.kmz", [
        ("doc.kml", kml("Root", extra=link("links/child.kml") + link("links/child.kml"))),
        ("links/child.kml", kml("Child", broken=True, extra=link("../doc.kml"))),
        ("unreachable.kml", b"<broken"), ("files/logo.png", b"image resource"),
    ])
    session = source.prepare_source(path).approve()
    parsed = session.fresh_parse()
    assert parsed.parsed_kml_files == ["doc.kml", "links/child.kml"]
    assert [p["name"] for p in parsed.pipelines] == ["Root", "Child"]
    assert [p["id"] for p in parsed.pipelines] == [0, 1]
    assert session.report["uninspected_documents"] == ["unreachable.kml"]
    assert [d["code"] for d in parsed.diagnostics] == ["selected_primary_kml", "unparsed_kml_file"]
    path.unlink()
    destination = tmp_path / "verified.kmz"
    receipt = session.save(destination)
    assert receipt["ordinary_reimport_verified"]
    with zipfile.ZipFile(destination) as archive:
        assert archive.read("unreachable.kml") == b"<broken"
        assert archive.read("files/logo.png") == b"image resource"
    assert parse_kml_kmz_with_diagnostics(destination).pipelines == parsed.pipelines


def test_ordinary_link_failure_remains_diagnostic_but_repaired_coverage_blocks(tmp_path):
    path = kmz(tmp_path / "network.kmz", [("doc.kml", kml(extra=link("child.kml"))), ("child.kml", b"<kml><broken>")])
    ordinary = source.prepare_source(path)
    assert not ordinary.requires_repair
    assert "linked_kml_parse_error" in {d["code"] for d in ordinary.fresh_parse().diagnostics}
    kmz(path, [("doc.kml", kml(broken=True, extra=link("child.kml"))), ("child.kml", b"<kml><broken>")])
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.category == "coverage"


@pytest.mark.parametrize("target", ["missing.kml", "https://example.invalid/client.kml", "../outside.kml", "child.kmz"])
def test_repair_never_continues_without_required_linked_geometry(tmp_path, target):
    path = kmz(tmp_path / "network.kmz", [("doc.kml", kml(broken=True, extra=link(target)))])
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.category == "coverage"
    assert "Please ask the client" in error.value.client_request()


def test_duplicate_feature_names_do_not_collapse_paths(tmp_path):
    path = kmz(tmp_path / "network.kmz", [
        ("doc.kml", kml("Same", broken=True, extra=link("child.kml"))),
        ("child.kml", kml("Same")),
    ])
    parsed = source.prepare_source(path).approve().fresh_parse()
    assert len(parsed.pipelines) == 2
    assert [p["id"] for p in parsed.pipelines] == [0, 1]
    assert [p["source_kml"] for p in parsed.pipelines] == ["doc.kml", "child.kml"]


def test_primary_selection_flip_disables_save_without_changing_session_geometry(tmp_path):
    parent = kml("Root", extra=link("child.kml")).decode()
    child = kml("Child", broken=True).decode()
    child = child.replace("</kml>", "<!--" + "x" * len(parent) + "--></kml>")
    parent = parent.replace("</kml>", "<!--" + "x" * (len(child) + 10 - len(parent) - 7) + "--></kml>")
    path = kmz(tmp_path / "network.kmz", [("root.kml", parent), ("child.kml", child)])
    session = source.prepare_source(path).approve()
    assert session.report["primary"] == "root.kml"
    assert [p["name"] for p in session.fresh_parse().pipelines] == ["Root", "Child"]
    assert not session.can_save
    assert "different primary" in session.save_unavailable_reason
    with pytest.raises(RepairFailure, match="different primary"):
        session.save(tmp_path / "repaired.kmz")


@pytest.mark.parametrize("actual_format,suffix", [("kml", ".kmz"), ("kmz", ".kml")])
def test_format_only_correction_is_byte_identical_and_reopens_normally(tmp_path, actual_format, suffix):
    path = tmp_path / ("wrong" + suffix)
    if actual_format == "kml":
        path.write_bytes(kml())
    else:
        kmz(path, [("doc.kml", kml())], comment=b"preserved ZIP comment")
    original = path.read_bytes()
    session = source.prepare_source(path).approve()
    assert session.report["rules"] == [source.FORMAT_RULE]
    assert session.report["edit_count"] == 0
    assert session.effective_format == actual_format
    output = tmp_path / ("repaired." + actual_format)
    session.save(output)
    assert output.read_bytes() == original
    assert not source.prepare_source(output).requires_repair
    assert len(parse_kml_kmz_with_diagnostics(output).pipelines) == 1


def test_multiple_rules_across_documents_are_one_transaction(tmp_path):
    root = b' \n<?xml version="1.0" encoding="UTF-8"?>' + kml("Root & Branch", extra=link("child.kml"))
    path = kmz(tmp_path / "wrong.kml", [("doc.kml", root), ("child.kml", kml("Child", broken=True))])
    session = source.prepare_source(path)
    assert set(session.report["rules"]) == {
        source.FORMAT_RULE, "leading_xml_whitespace_v1", "literal_metadata_ampersand_v1", "missing_xsi_schema_namespace_v1",
    }
    session.approve()
    assert [p["name"] for p in session.fresh_parse().pipelines] == ["Root & Branch", "Child"]
    output = tmp_path / "fixed.kmz"
    session.save(output)
    assert not source.prepare_source(output).requires_repair


@pytest.mark.parametrize("extra", [link("child.kml"), '<Style><IconStyle><Icon><href>image.png</href></Icon></IconStyle></Style>',
                                  '<styleUrl>styles.kml#normal</styleUrl>'])
def test_standalone_local_dependencies_disable_portable_save(tmp_path, extra):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True, extra=extra))
    (tmp_path / "child.kml").write_bytes(kml("Child"))
    session = source.prepare_source(path).approve()
    assert not session.can_save
    assert "Saving" in session.save_unavailable_reason
    path.rename(tmp_path / "moved.kml")
    assert session.fresh_parse().pipelines[0]["name"] == "Pipeline"


def test_misnamed_plaintext_with_relative_dependencies_is_refused(tmp_path):
    path = tmp_path / "wrong.kmz"
    path.write_bytes(kml(extra='<styleUrl>styles.kml#normal</styleUrl>'))
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.findings[0]["code"] == "format_mismatch_dependencies"


def test_repair_blocks_xml_base_that_changes_the_intended_linked_geometry(tmp_path):
    extra = '<Folder xml:base="sub/">' + link("child.kml") + '</Folder>'
    path = kmz(tmp_path / "base.kmz", [
        ("doc.kml", kml("Root", broken=True, extra=extra)),
        ("child.kml", kml("Wrong root child")),
        ("sub/child.kml", kml("Intended base child").replace(b"-100,30,7 -100.01,30,9", b"-101,31,7 -101.05,31,9")),
    ])
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.category == "coverage"
    assert "unsupported_xml_base" in {finding["code"] for finding in error.value.findings}
    request = error.value.client_request()
    assert "explicitly resolved local links" in request
    assert "do not simply remove xml:base" in request


@pytest.mark.parametrize("base", ["", "sub/", "https://example.invalid/"])
def test_xml_base_is_a_repaired_coverage_gate_only(tmp_path, base):
    path = tmp_path / "source.kml"
    extra = f'<Folder xml:base="{base}"/>'
    path.write_bytes(kml(extra=extra))
    assert not source.prepare_source(path).requires_repair
    path.write_bytes(kml(broken=True, extra=extra))
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert "unsupported_xml_base" in {finding["code"] for finding in error.value.findings}


@pytest.mark.parametrize("names", [("files/a.png", "FILES/A.png"), ("a", "a/b"), ("../evil", "safe"), ("C:/bad", "safe")])
def test_all_archive_members_receive_path_and_collision_checks(tmp_path, names):
    path = kmz(tmp_path / "bad.kmz", [("doc.kml", kml(broken=True)), (names[0], b"a"), (names[1], b"b")])
    with pytest.raises(RepairFailure):
        source.prepare_source(path)


def test_unused_resource_crc_corruption_blocks_repair(tmp_path):
    path = tmp_path / "bad.kmz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("doc.kml", kml(broken=True))
        archive.writestr("unused.bin", b"UNIQUE_CONTENT")
    data = path.read_bytes().replace(b"UNIQUE_CONTENT", b"ALTEREDCONTENT")
    path.write_bytes(data)
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.findings[0]["code"] == "invalid_archive"


def test_unused_symlink_member_is_not_extracted_or_accepted(tmp_path):
    path = tmp_path / "bad.kmz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("doc.kml", kml(broken=True))
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, b"../../target")
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.findings[0]["code"] == "special_archive_member"


@pytest.mark.parametrize("change", [lambda data: b"MZpreamble" + data, lambda data: data + b"trailer"])
def test_zip_polyglot_preamble_or_trailing_data_is_refused(tmp_path, change):
    path = kmz(tmp_path / "bad.kmz", [("doc.kml", kml(broken=True))])
    path.write_bytes(change(path.read_bytes()))
    with pytest.raises(RepairFailure):
        source.prepare_source(path)


def test_save_preserves_archive_metadata_and_never_overwrites(tmp_path):
    path = tmp_path / "broken.kmz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.comment = b"archive comment"
        entry = zipfile.ZipInfo("doc.kml", (2020, 3, 4, 5, 6, 8))
        entry.comment = b"entry comment"
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.create_system = 3
        entry.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(entry, kml(broken=True))
    original = path.read_bytes()
    session = source.prepare_source(path).approve()
    with pytest.raises(RepairFailure, match="alias"):
        session.save(path)
    output = tmp_path / "repaired.kmz"
    session.save(output)
    with zipfile.ZipFile(output) as archive:
        saved = archive.getinfo("doc.kml")
        assert saved.date_time == entry.date_time
        assert saved.comment == entry.comment
        assert saved.external_attr == entry.external_attr
        assert archive.comment == b"archive comment"
    sentinel = output.read_bytes()
    with pytest.raises(FileExistsError):
        session.save(output)
    assert output.read_bytes() == sentinel
    assert path.read_bytes() == original


def test_save_racing_collision_does_not_replace_other_file(tmp_path, monkeypatch):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True))
    session = source.prepare_source(path).approve()
    output = tmp_path / "repaired.kml"
    real_link = source.os.link
    def racing_link(src, dst):
        Path(dst).write_bytes(b"other user's file")
        return real_link(src, dst)
    monkeypatch.setattr(source.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        session.save(output)
    assert output.read_bytes() == b"other user's file"
    assert not list(tmp_path.glob(".repaired.*"))
    assert session.verified


def test_cancellation_before_save_cleans_stage_and_retains_source(tmp_path):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True))
    session = source.prepare_source(path).approve()
    context = ExecutionContext()
    context.cancel()
    output = tmp_path / "repaired.kml"
    with pytest.raises(AnalysisCancelled):
        session.save(output, context=context)
    assert not output.exists()
    assert session.verified and session.fresh_parse().pipelines


def test_cancellation_after_atomic_publish_reports_completed_save(tmp_path, monkeypatch):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True))
    session = source.prepare_source(path).approve()
    context = ExecutionContext()
    real_link = source.os.link
    def publish_then_cancel(src, dst):
        real_link(src, dst)
        context.cancel()
    monkeypatch.setattr(source.os, "link", publish_then_cancel)
    output = tmp_path / "repaired.kml"
    receipt = session.save(output, context=context)
    assert receipt["status"] == "saved" and output.exists()


def test_source_change_during_acquisition_is_operational_not_client_damage(tmp_path, monkeypatch):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True))
    real_inspect = source.inspect_document
    def inspect_then_change(data, source, *, context=None):
        result = real_inspect(data, source=source, context=context)
        Path(source).write_bytes(kml("Changed", broken=True))
        return result
    monkeypatch.setattr(source, "inspect_document", inspect_then_change)
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.category == "operation"
    assert "Please ask the client" not in error.value.client_request()


def test_unsupported_geometry_loss_and_invalid_coordinates_block_repaired_input(tmp_path):
    path = tmp_path / "bad.kml"
    for altered in [kml(broken=True).replace(b"LineString", b"linestring"),
                    kml(broken=True).replace(b"-100.01,30,9", b"-100.01,NaN,9")]:
        path.write_bytes(altered)
        with pytest.raises(RepairFailure):
            source.prepare_source(path)


def test_report_and_parser_are_independent_of_later_source_edits(tmp_path):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True))
    session = source.prepare_source(path).approve()
    expected = session.fresh_parse()
    expected_hash = session.report["source_sha256"]
    path.write_bytes(kml("Different client"))
    assert session.fresh_parse() == expected
    session.save(tmp_path / "verified.kml")
    assert session.report["source_sha256"] == expected_hash
    assert parse_kml_kmz_with_diagnostics(tmp_path / "verified.kml").pipelines[0]["name"] == "Pipeline"


def test_nul_in_original_zip_filename_cannot_be_silently_truncated(tmp_path):
    path = kmz(tmp_path / "bad.kmz", [("doc.kml", kml(broken=True)), ("resourceXsuffix", b"asset")])
    path.write_bytes(path.read_bytes().replace(b"resourceXsuffix", b"resource\x00suffix"))
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.findings[0]["code"] == "invalid_archive_path"


@pytest.mark.parametrize("encoding", ["UTF-16", "ISO-8859-1"])
def test_format_correction_does_not_bypass_repair_encoding_policy(tmp_path, encoding):
    path = tmp_path / "wrong.kmz"
    document = f'<?xml version="1.0" encoding="{encoding}"?>'.encode() + kml()
    if encoding == "UTF-16":
        document = document.decode().encode("utf-16")
    path.write_bytes(document)
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.findings[0]["code"] == "unsupported_repair_encoding"
    # The same valid ordinary file remains usable with its correct suffix.
    normal = tmp_path / "ordinary.kml"
    normal.write_bytes(document)
    assert not source.prepare_source(normal).requires_repair


def test_repaired_graph_rejects_non_utf8_otherwise_valid_linked_document(tmp_path):
    child = ('<?xml version="1.0" encoding="UTF-16"?>' + kml().decode()).encode("utf-16")
    path = kmz(tmp_path / "graph.kmz", [("doc.kml", kml(broken=True, extra=link("child.kml"))), ("child.kml", child)])
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.findings[0]["code"] == "unsupported_repair_encoding"


def test_complete_source_element_limit_is_aggregate_not_per_document(tmp_path, monkeypatch):
    path = kmz(tmp_path / "graph.kmz", [("doc.kml", kml(extra=link("child.kml"))), ("child.kml", kml())])
    monkeypatch.setattr(source, "MAX_SOURCE_ELEMENTS", 14)
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert error.value.category == "limit"
    assert error.value.findings[0]["code"] == "xml_complexity_limit"


def test_client_findings_use_relative_source_identity(tmp_path):
    path = tmp_path / "bad.kml"
    path.write_bytes(kml(broken=True).replace(b"-100.01,30,9", b"999,30,9"))
    with pytest.raises(RepairFailure) as error:
        source.prepare_source(path)
    assert str(tmp_path) not in error.value.client_request()
    assert "bad.kml" in error.value.client_request()
    assert "invalid_coordinate" in {f["code"] for f in error.value.findings}


def test_resource_limit_findings_explicitly_mark_truncation(monkeypatch):
    monkeypatch.setattr(source, "MAX_DIAGNOSTICS", 3)
    findings = source._bounded_findings([{"code": str(i)} for i in range(5)])
    assert len(findings) == 3
    assert findings[-1]["code"] == "inspection_findings_truncated"


def test_cancel_after_staging_removes_output_and_keeps_verified_source(tmp_path, monkeypatch):
    path = tmp_path / "source.kml"
    path.write_bytes(kml(broken=True))
    session = source.prepare_source(path).approve()
    def cancelled_staged_validation(path, *, context=None):
        raise AnalysisCancelled("cancelled while validating staged copy")
    monkeypatch.setattr(source, "prepare_source", cancelled_staged_validation)
    with pytest.raises(AnalysisCancelled):
        session.save(tmp_path / "repaired.kml")
    assert not list(tmp_path.glob(".repaired.*"))
    assert not (tmp_path / "repaired.kml").exists()
    assert session.verified
