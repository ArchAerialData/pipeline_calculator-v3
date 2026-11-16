"""Small boundary probes for import budgets; no large files or native allocations.

These establish that limits are enforced and cumulative. They do not certify
peak memory or performance at the production maximum workload.
"""
import base64
import json
import struct
from types import SimpleNamespace
import zipfile

import pytest
from shapely import to_wkb
from shapely.geometry import GeometryCollection, LineString

from pipeline_calculator.importing import archive, normalize, worker
from pipeline_calculator.importing.models import ImportFailure


LINE = LineString([(-100., 30.), (-100.001, 30.)])


def descriptor(name="first"):
    return {"id": f"example.gpkg::{name}", "dataset": "example.gpkg", "name": name,
            "format": "gpkg", "dimensions": []}


def record(index=0, *, geometry=LINE, attributes=None):
    return {"fid": str(index), "record_index": index, "attributes": attributes or {},
            "wkb": base64.b64encode(to_wkb(geometry)).decode() if geometry is not None else None}


def add(normalizer, records, *, name="first"):
    normalizer.add_layer(descriptor(name), [{"kind": "batch", "records": records}], "EPSG:4326")


def make_archive(path):
    payload = json.dumps({"type": "FeatureCollection", "features": []}).encode()
    metadata = b"unrelated metadata" * 32
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as stream:
        stream.writestr("pipes.geojson", payload)
        stream.writestr("notes.txt", metadata)
    return len(payload) + len(metadata)


def test_archive_expanded_byte_limit_accepts_boundary_then_refuses_one_byte_less(tmp_path, monkeypatch):
    source = tmp_path / "compressed.zip"
    expanded_bytes = make_archive(source)
    monkeypatch.setattr(archive, "MAX_EXPANDED_BYTES", expanded_bytes)
    captured = archive.capture(source)
    assert captured.datasets[0]["id"] == "pipes.geojson"
    assert "notes.txt" in captured.ignored
    monkeypatch.setattr(archive, "MAX_EXPANDED_BYTES", expanded_bytes - 1)
    with pytest.raises(ImportFailure) as exc:
        archive.capture(source)
    assert exc.value.code == "package_size_limit"


def test_archive_entry_limit_is_inclusive_and_precedes_reader_work(tmp_path, monkeypatch):
    source = tmp_path / "entries.zip"
    make_archive(source)
    monkeypatch.setattr(archive, "MAX_ENTRIES", 2)
    assert len(archive.capture(source).datasets) == 1
    monkeypatch.setattr(archive, "MAX_ENTRIES", 1)
    with pytest.raises(ImportFailure) as exc:
        archive.capture(source)
    assert exc.value.code == "archive_entry_limit"


def test_feature_limit_accumulates_across_selected_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(normalize, "MAX_FEATURES", 1)
    normalizer = normalize.Normalizer("digest")
    add(normalizer, [record()])
    assert len(normalizer.result.pipelines) == 1
    with pytest.raises(ImportFailure) as exc:
        add(normalizer, [record(1)], name="second")
    assert exc.value.code == "feature_limit"
    assert len(normalizer.result.pipelines) == 1


def test_vertex_limit_accumulates_across_batches_and_layers(monkeypatch):
    monkeypatch.setattr(normalize, "MAX_VERTICES", 2)
    normalizer = normalize.Normalizer("digest")
    add(normalizer, [record()])
    assert len(normalizer.result.pipelines) == 1
    with pytest.raises(ImportFailure) as exc:
        add(normalizer, [record(1)], name="second")
    assert exc.value.code == "vertex_limit"
    assert len(normalizer.result.pipelines) == 1


def test_attribute_budget_counts_utf8_bytes_and_accumulates(monkeypatch):
    attributes = {"Name": "é"}
    byte_count = len(b'{"Name": "\xc3\xa9"}')
    monkeypatch.setattr(normalize, "MAX_ATTRIBUTE_BYTES", byte_count)
    normalizer = normalize.Normalizer("digest")
    add(normalizer, [record(attributes=attributes)])
    assert normalizer.result.pipelines[0]["name"] == "é"
    with pytest.raises(ImportFailure) as exc:
        add(normalizer, [record(1, attributes=attributes)], name="second")
    assert exc.value.code == "attribute_limit"
    assert len(normalizer.result.pipelines) == 1


def test_attribute_budget_rejects_one_over_before_retaining_feature(monkeypatch):
    monkeypatch.setattr(normalize, "MAX_ATTRIBUTE_BYTES", len(b'{"Name": "\xc3\xa9"}') - 1)
    normalizer = normalize.Normalizer("digest")
    with pytest.raises(ImportFailure) as exc:
        add(normalizer, [record(attributes={"Name": "é"})])
    assert exc.value.code == "attribute_limit"
    assert normalizer.result.pipelines == []


def test_diagnostic_budget_fails_instead_of_truncating_omissions(monkeypatch):
    monkeypatch.setattr(normalize, "MAX_DIAGNOSTICS", 1)
    normalizer = normalize.Normalizer("digest")
    add(normalizer, [record(geometry=None)])
    assert [item["code"] for item in normalizer.result.diagnostics] == ["null_geometry"]
    with pytest.raises(ImportFailure) as exc:
        add(normalizer, [record(1, geometry=None)], name="second")
    assert exc.value.code == "diagnostic_limit"
    assert len(normalizer.result.diagnostics) == 1


def test_normalizer_collection_depth_boundary(monkeypatch):
    monkeypatch.setattr(normalize, "MAX_COLLECTION_DEPTH", 2)
    allowed = GeometryCollection([GeometryCollection([LINE])])
    normalizer = normalize.Normalizer("digest")
    add(normalizer, [record(geometry=allowed)])
    assert len(normalizer.result.pipelines) == 1
    with pytest.raises(ImportFailure) as exc:
        add(normalize.Normalizer("digest"), [record(geometry=GeometryCollection([allowed]))])
    assert exc.value.code == "geometry_depth_limit"


def gpkg_blob(wkb):
    return b"GP" + bytes([0, 1]) + struct.pack("<i", 4326) + wkb


def line_wkb(count=2):
    return struct.pack("<BII", 1, 2, count) + b"".join(
        struct.pack("<dd", -100. + index / 1000, 30.) for index in range(count))


def collection_wkb(*children):
    return struct.pack("<BII", 1, 7, len(children)) + b"".join(children)


def test_original_wkb_vertex_count_is_checked_before_geometry_decode(monkeypatch):
    monkeypatch.setattr(worker, "MAX_VERTICES", 2)
    assert worker._wkb_type_audit(gpkg_blob(line_wkb(2)), 4326) == (2, [])
    with pytest.raises(ImportFailure) as exc:
        worker._wkb_type_audit(gpkg_blob(line_wkb(3)), 4326)
    assert exc.value.code == "vertex_limit"


def test_original_wkb_vertex_limit_is_cumulative_inside_collections(monkeypatch):
    monkeypatch.setattr(worker, "MAX_VERTICES", 3)
    # Each child is individually within budget; their combined four vertices are not.
    with pytest.raises(ImportFailure) as exc:
        worker._wkb_type_audit(gpkg_blob(collection_wkb(line_wkb(), line_wkb())), 4326)
    assert exc.value.code == "vertex_limit"


def test_original_wkb_huge_declared_count_refuses_without_payload_allocation(monkeypatch):
    monkeypatch.setattr(worker, "MAX_VERTICES", 3)
    only_header = struct.pack("<BII", 1, 2, 0xFFFFFFFF)
    with pytest.raises(ImportFailure) as exc:
        worker._wkb_type_audit(gpkg_blob(only_header), 4326)
    assert exc.value.code == "vertex_limit"


def test_original_wkb_collection_depth_boundary(monkeypatch):
    monkeypatch.setattr(worker, "MAX_COLLECTION_DEPTH", 2)
    allowed = collection_wkb(collection_wkb(line_wkb()))
    assert worker._wkb_type_audit(gpkg_blob(allowed), 4326) == (2, [])
    with pytest.raises(ImportFailure):
        worker._wkb_type_audit(gpkg_blob(collection_wkb(allowed)), 4326)


def test_native_transfer_budget_counts_encoded_bytes_before_sending(monkeypatch):
    sent = []
    pipe = SimpleNamespace(send_bytes=sent.append)
    value = {"value": "é"}
    encoded = b'{"value": "\xc3\xa9"}'
    monkeypatch.setattr(worker, "MAX_IPC_BYTES", len(encoded))
    worker._send(pipe, value)
    assert sent == [encoded]
    monkeypatch.setattr(worker, "MAX_IPC_BYTES", len(encoded) - 1)
    with pytest.raises(ImportFailure) as exc:
        worker._send(pipe, value)
    assert exc.value.code == "batch_size_limit"
    assert sent == [encoded]
