import json
import sqlite3
import struct

import pytest
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon, GeometryCollection

from pipeline_calculator.importing import prepare_import
from pipeline_calculator.importing.models import ImportDecisionRequired, ImportFailure
from gis_helpers import LINE, OTHER_LINE, write_vector, write_geojson, paths, diagnostic_codes


@pytest.mark.parametrize("suffix", [".shp", ".gpkg", ".geojson", ".json"])
def test_lines_use_same_geometry_across_formats(tmp_path, suffix):
    path = tmp_path / ("pipes" + suffix)
    geometry = MultiLineString([LINE, OTHER_LINE])
    if suffix in {".json", ".geojson"}:
        write_geojson(path, [geometry], properties=[{"PLINE_ID": "client-pipe"}])
    else:
        write_vector(path, [geometry], fields={"PLINE_ID": ["client-pipe"]})
    parsed = prepare_import(path).fresh_parse()
    assert len(parsed.pipelines) == 1
    assert paths(parsed) == [LINE, OTHER_LINE]
    assert parsed.pipelines[0]["name"] == "client-pipe"
    assert not parsed.parsed_kml_files
    assert parsed.parsed_sources


def test_duplicate_identifiers_and_locations_do_not_merge_features(tmp_path):
    source = write_vector(tmp_path / "duplicates.gpkg", [LineString(LINE)] * 2,
                          fields={"Name": ["same", "same"], "PLINE_ID": ["duplicate", "duplicate"]})
    parsed = prepare_import(source).fresh_parse()
    assert len(parsed.pipelines) == 2
    assert [p["name"] for p in parsed.pipelines] == ["same", "same"]
    assert len({p["id"] for p in parsed.pipelines}) == 2
    assert len({p["source_feature_id"] for p in parsed.pipelines}) == 2


@pytest.mark.parametrize("suffix", [".shp", ".gpkg", ".geojson"])
@pytest.mark.parametrize("geometry,count", [(Point(-100, 30), 1),
    (MultiPoint([(-100, 30), (-100, 30), (-100.1, 30.1)]), 3)], ids=["point", "multipoint"])
def test_points_and_multipoints_across_formats(tmp_path, suffix, geometry, count):
    source = tmp_path / ("pins" + suffix)
    (write_geojson if suffix == ".geojson" else write_vector)(source, [geometry])
    parsed = prepare_import(source).fresh_parse()
    assert parsed.pipelines == []
    assert len(parsed.placemarks) == count
    assert all(pin["Count"] == 1 for pin in parsed.placemarks)
    assert tuple(parsed.placemarks[0]["coordinates"]) == (-100, 30)
    assert not any(d["level"] == "error" for d in parsed.diagnostics)


@pytest.mark.parametrize("suffix", [".gpkg", ".geojson"])
def test_mixed_collections_count_actual_pins_and_exclude_polygon_edges(tmp_path, suffix):
    geometry = GeometryCollection([
        MultiLineString([LINE, OTHER_LINE]), Point(-100, 30),
        GeometryCollection([MultiPoint([(-100, 30), (-100, 30)]),
                            Polygon([(-101, 30), (-101, 30.1), (-101.1, 30), (-101, 30)])]),
    ])
    path = tmp_path / ("mixed" + suffix)
    (write_vector if suffix == ".gpkg" else write_geojson)(path, [geometry])
    parsed = prepare_import(path).fresh_parse()
    assert len(parsed.pipelines) == 1 and paths(parsed) == [LINE, OTHER_LINE]
    assert len(parsed.placemarks) == 3
    assert all(pin["Count"] == 1 and tuple(pin["coordinates"]) == (-100, 30) for pin in parsed.placemarks)
    assert "polygon_excluded" in diagnostic_codes(parsed)
    assert not any(d["level"] == "error" for d in parsed.diagnostics)


def test_multilayer_geopackage_requires_choice_and_preserves_scope(tmp_path):
    source = tmp_path / "layers.gpkg"
    write_vector(source, [LineString(LINE)], layer="pipes")
    write_vector(source, [Point(-100, 30)], layer="sites")
    write_vector(source, [Polygon([(-101, 30), (-101, 30.1), (-101.1, 30), (-101, 30)])], layer="areas")
    session = prepare_import(source)
    with pytest.raises(ImportDecisionRequired) as exc:
        session.fresh_parse()
    assert exc.value.source_session is session
    layers = {layer["name"]: layer for layer in session.layers}
    assert {"pipes", "sites", "areas"}.issubset(layers)
    chosen = session.with_selection([layers[name]["id"] for name in ("pipes", "sites")])
    parsed = chosen.fresh_parse()
    assert len(parsed.pipelines) == 1 and len(parsed.placemarks) == 1
    assert {record["source_layer"] for record in parsed.pipelines + parsed.placemarks} == {"pipes", "sites"}
    assert not any("areas" == p["source_layer"] for p in parsed.pipelines)


@pytest.mark.parametrize("epsg", [32614, 2277])
def test_projected_crs_meters_and_us_feet_convert_to_longitude_latitude(tmp_path, epsg):
    longitude_latitude = [(-97.7431, 30.2672), (-97.7420, 30.2680)]
    forward = Transformer.from_crs(4326, epsg, always_xy=True)
    projected = [forward.transform(*xy) for xy in longitude_latitude]
    source = write_vector(tmp_path / "projected.gpkg", [LineString(projected)], crs=f"EPSG:{epsg}")
    parsed = prepare_import(source).fresh_parse()
    for actual, expected in zip(paths(parsed)[0], longitude_latitude):
        assert actual == pytest.approx(expected, abs=1e-8)
    assert parsed.import_metadata["coordinate_operations"]


def test_missing_shapefile_projection_requires_explicit_crs_assignment(tmp_path):
    source = write_vector(tmp_path / "missing.shp", [LineString(LINE)])
    source.with_suffix(".prj").unlink()
    session = prepare_import(source)
    with pytest.raises(ImportDecisionRequired) as exc:
        session.fresh_parse()
    assert exc.value.missing_crs == [session.layers[0]["id"]]
    chosen = session.with_selection([session.layers[0]["id"]],
                                    crs_overrides={session.layers[0]["id"]: "EPSG:4326"})
    assert paths(chosen.fresh_parse()) == [LINE]


def test_null_geometry_is_an_explicit_omission(tmp_path):
    source = write_geojson(tmp_path / "null.geojson", [LineString(LINE), None])
    parsed = prepare_import(source).fresh_parse()
    assert paths(parsed) == [LINE]
    assert any(d["code"] == "null_geometry" and d["level"] == "error" for d in parsed.diagnostics)


def test_z_does_not_change_horizontal_geometry(tmp_path):
    source = write_geojson(tmp_path / "elevation.geojson", [LineString([
        (*LINE[0], -200), (*LINE[1], 1200)])])
    parsed = prepare_import(source).fresh_parse()
    assert paths(parsed) == [LINE]
    assert parsed.import_metadata


def test_invalid_vertex_does_not_connect_good_vertices_across_it(tmp_path):
    source = write_geojson(tmp_path / "invalid.geojson", [LineString([
        LINE[0], (-100, 95), LINE[1]]), LineString(OTHER_LINE)])
    parsed = prepare_import(source).fresh_parse()
    assert paths(parsed) == [OTHER_LINE]
    assert any(d["level"] == "error" for d in parsed.diagnostics)


def test_geojson_legacy_crs_is_not_silently_assumed_wgs84(tmp_path):
    source = write_geojson(tmp_path / "legacy.geojson", [LineString(LINE)])
    payload = json.loads(source.read_text())
    payload["crs"] = {"type": "name", "properties": {"name": "EPSG:3857"}}
    source.write_text(json.dumps(payload))
    with pytest.raises(ImportFailure):
        prepare_import(source).fresh_parse()


@pytest.mark.parametrize("nested", [False, True], ids=["curve", "curve-in-collection"])
def test_gpkg_curve_in_generic_layer_is_rejected_before_linearization(tmp_path, nested):
    source = write_vector(tmp_path / "curve.gpkg", [LineString(LINE)], geometry_type="Unknown")
    # A standards-shaped GeoPackage binary: no envelope, LE SRS 4326, ISO WKB
    # CircularString with three vertices. No production decoder builds this fixture.
    circular_wkb = struct.pack("<BII", 1, 8, 3) + b"".join(
        struct.pack("<dd", *xy) for xy in [(-100, 30), (-100.001, 30.001), (-100.002, 30)])
    if nested:
        circular_wkb = struct.pack("<BII", 1, 7, 1) + circular_wkb
    blob = b"GP" + bytes([0, 1]) + struct.pack("<i", 4326) + circular_wkb
    with sqlite3.connect(source) as db:
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            db.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        db.execute("UPDATE gpkg_geometry_columns SET geometry_type_name='GEOMETRY'")
        db.execute("UPDATE pipes SET geom=?", (blob,))
    with pytest.raises(ImportFailure):
        prepare_import(source).fresh_parse()


def test_live_gpkg_sidecar_is_not_ignored(tmp_path):
    source = write_vector(tmp_path / "live.gpkg", [LineString(LINE)])
    source.with_name(source.name + "-wal").write_bytes(b"uncheckpointed transaction")
    with pytest.raises(ImportFailure):
        prepare_import(source).fresh_parse()


def test_gpkg_geometry_header_srs_must_match_its_layer(tmp_path):
    source = write_vector(tmp_path / "srs-mismatch.gpkg", [LineString(LINE)])
    with sqlite3.connect(source) as db:
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            db.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        geometry = bytearray(db.execute("SELECT geom FROM pipes").fetchone()[0])
        assert geometry[:2] == b"GP"
        endian = "<" if geometry[3] & 1 else ">"
        struct.pack_into(endian + "i", geometry, 4, 3857)
        db.execute("UPDATE pipes SET geom=?", (geometry,))
    with pytest.raises(ImportFailure):
        prepare_import(source).fresh_parse()


def test_nullable_int64_attributes_keep_original_precision(tmp_path):
    source = write_vector(tmp_path / "identifiers.gpkg", [LineString(LINE), LineString(OTHER_LINE)])
    identifier = 9007199254740993  # First integer that float64 cannot represent.
    with sqlite3.connect(source) as db:
        # The fixture edit uses stdlib SQLite without SpatiaLite. Remove the
        # writer's spatial-index triggers before an attribute-only update;
        # geometry bytes and their existing index remain unchanged.
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            db.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        db.execute("ALTER TABLE pipes ADD COLUMN client_number INTEGER")
        db.execute("UPDATE pipes SET client_number=? WHERE fid=1", (identifier,))
    parsed = prepare_import(source).fresh_parse()
    by_id = {p["source_feature_id"]: p for p in parsed.pipelines}
    assert by_id["1"]["attributes"]["client_number"] == identifier
    assert isinstance(by_id["1"]["attributes"]["client_number"], int)
    assert by_id["2"]["attributes"]["client_number"] is None


def test_deleted_shapefile_row_keeps_physical_ids_across_reader_batches(tmp_path):
    source = write_vector(tmp_path / "deleted.shp", [LineString(LINE)] * 260)
    dbf_path = source.with_suffix(".dbf")
    dbf = bytearray(dbf_path.read_bytes())
    header_bytes, record_bytes = struct.unpack_from("<HH", dbf, 8)
    dbf[header_bytes + 10 * record_bytes] = 0x2A
    dbf_path.write_bytes(dbf)
    parsed = prepare_import(source).fresh_parse()
    assert len(parsed.pipelines) == 259
    assert {int(p["source_feature_id"]) for p in parsed.pipelines} == set(range(260)) - {10}
    assert all(p["coordinate_paths"] == [LINE] for p in parsed.pipelines)
