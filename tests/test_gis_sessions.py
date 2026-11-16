import pytest
import json
import multiprocessing
import time
from shapely.geometry import LineString

from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.importing import prepare_import
from pipeline_calculator.importing.models import ImportFailure
from gis_helpers import LINE, OTHER_LINE, write_vector, write_geojson, paths


@pytest.mark.parametrize("suffix", [".shp", ".gpkg", ".geojson"])
def test_source_capture_survives_removal_before_first_parse(tmp_path, suffix):
    source = tmp_path / ("captured" + suffix)
    (write_geojson if suffix == ".geojson" else write_vector)(source, [LineString(LINE)])
    session = prepare_import(source)
    for component in tmp_path.iterdir():
        component.unlink()
    assert paths(session.fresh_parse()) == [LINE]


def test_every_analysis_receives_detached_mutable_records(tmp_path):
    source = write_geojson(tmp_path / "fresh.geojson", [LineString(LINE)])
    session = prepare_import(source)
    first = session.fresh_parse()
    first.pipelines[0]["coordinate_paths"][0].clear()
    first.pipelines[0]["segments"] = ["overlap cache"]
    first.diagnostics.append({"level": "error", "code": "consumer mutation"})
    second = session.fresh_parse()
    assert paths(second) == [LINE]
    assert "segments" not in second.pipelines[0]
    assert not any(d["code"] == "consumer mutation" for d in second.diagnostics)


def test_layer_selection_reuses_frozen_input_after_original_changes(tmp_path):
    source = write_vector(tmp_path / "choice.gpkg", [LineString(LINE)], layer="first")
    write_vector(source, [LineString(OTHER_LINE)], layer="second")
    session = prepare_import(source)
    source.write_bytes(b"source replaced after discovery")
    for layer, expected in [("first", LINE), ("second", OTHER_LINE)]:
        chosen = session.with_selection([next(item["id"] for item in session.layers if item["name"] == layer)])
        assert paths(chosen.fresh_parse()) == [expected]


def test_shapefile_sidecar_rewrite_cannot_change_captured_geometry_or_crs(tmp_path):
    source = write_vector(tmp_path / "captured.shp", [LineString(LINE)])
    session = prepare_import(source)
    source.with_suffix(".prj").write_text("EPSG:3857")
    source.with_suffix(".dbf").write_bytes(b"changed after capture")
    assert paths(session.fresh_parse()) == [LINE]


def test_shapefile_component_change_during_capture_is_rejected(tmp_path, monkeypatch):
    from pipeline_calculator.importing import archive
    source = write_vector(tmp_path / "race.shp", [LineString(LINE)])
    read = archive.read_stable
    changed = []
    def capture_then_change(path, context=None):
        result = read(path, context)
        if path.suffix.lower() == ".shp" and not changed:
            path.write_bytes(path.read_bytes() + b"changed during multi-component capture")
            changed.append(True)
        return result
    monkeypatch.setattr(archive, "read_stable", capture_then_change)
    with pytest.raises(ImportFailure) as exc:
        prepare_import(source)
    assert exc.value.code == "source_changed"


def test_cancelled_import_stops_without_preparing_source(tmp_path):
    source = write_geojson(tmp_path / "cancel.geojson", [LineString(LINE)])
    context = ExecutionContext()
    context.cancel()
    with pytest.raises(AnalysisCancelled):
        prepare_import(source, context=context)


def test_cancellation_before_fresh_parse_does_not_poison_snapshot(tmp_path):
    source = write_geojson(tmp_path / "cancel.geojson", [LineString(LINE)])
    session = prepare_import(source)
    context = ExecutionContext()
    context.cancel()
    with pytest.raises(AnalysisCancelled):
        session.fresh_parse(context=context)
    assert paths(session.fresh_parse()) == [LINE]


def _blocking_native_reader(pipe, path, driver, command, layer):
    """Spawnable stand-in for a native driver that does not yield Python checks."""
    pipe.send_bytes(json.dumps({"kind": "started"}).encode())
    time.sleep(30)


def test_cancellation_terminates_blocked_native_reader_within_two_seconds(tmp_path, monkeypatch):
    from pipeline_calculator.importing import worker
    before_children = {child.pid for child in multiprocessing.active_children()}
    monkeypatch.setattr(worker, "_entry", _blocking_native_reader)
    context = ExecutionContext()
    messages = worker.run_native(tmp_path / "unused.gpkg", "GPKG", "discover", context=context)
    assert next(messages) == {"kind": "started"}
    started = time.monotonic()
    context.cancel()
    with pytest.raises(AnalysisCancelled):
        next(messages)
    assert time.monotonic() - started < 2
    assert {child.pid for child in multiprocessing.active_children()} <= before_children
