import stat
import struct
import zipfile

import pytest
from shapely.geometry import LineString

from pipeline_calculator.importing import prepare_import
from pipeline_calculator.importing.models import ImportDecisionRequired, ImportFailure
from gis_helpers import LINE, OTHER_LINE, write_vector, write_geojson, zip_files, paths


def test_nested_unicode_shapefile_zip_needs_no_manual_extraction(tmp_path):
    folder = tmp_path / "source"
    source = write_vector(folder / "Pípés.SHP", [LineString(LINE)])
    archive = zip_files(tmp_path / "client package.ZIP", folder.iterdir(), prefix="GIS/客户 data/")
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr("__MACOSX/._irrelevant", b"metadata")
        zipped.writestr("notes.txt", "This is data, not an import instruction.")
    session = prepare_import(archive)
    assert paths(session.fresh_parse()) == [LINE]
    assert len(session.layers) == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["client package.ZIP", "source"]


def test_unrelated_json_beside_shapefile_is_not_an_extra_gis_layer(tmp_path):
    folder = tmp_path / "source"
    write_vector(folder / "pipes.shp", [LineString(LINE)])
    archive = zip_files(tmp_path / "metadata.zip", folder.iterdir())
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr("readme.json", '{"description":"export metadata, not GeoJSON"}')
    session = prepare_import(archive)
    assert len([layer for layer in session.layers if layer["supported"]]) == 1
    assert paths(session.fresh_parse()) == [LINE]


def test_zip_of_geopackage_discovers_layers_before_selection(tmp_path):
    source = write_vector(tmp_path / "source.gpkg", [LineString(LINE)], layer="first")
    write_vector(source, [LineString(OTHER_LINE)], layer="second")
    archive = zip_files(tmp_path / "source.zip", [source], prefix="nested/")
    session = prepare_import(archive)
    with pytest.raises(ImportDecisionRequired):
        session.fresh_parse()
    second = next(layer for layer in session.layers if layer["name"] == "second")
    assert paths(session.with_selection([second["id"]]).fresh_parse()) == [OTHER_LINE]


def test_same_basename_shapefiles_in_separate_folders_remain_distinct(tmp_path):
    first = write_vector(tmp_path / "a" / "pipes.shp", [LineString(LINE)])
    second = write_vector(tmp_path / "b" / "pipes.shp", [LineString(OTHER_LINE)])
    archive = tmp_path / "both.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for folder in (first.parent, second.parent):
            for component in folder.iterdir():
                zipped.write(component, folder.name + "/" + component.name)
    session = prepare_import(archive)
    assert len(session.layers) == 2
    with pytest.raises(ImportDecisionRequired):
        session.fresh_parse()
    parsed = session.with_selection([layer["id"] for layer in session.layers]).fresh_parse()
    assert len(parsed.pipelines) == 2
    assert len({p["id"] for p in parsed.pipelines}) == 2
    assert {tuple(path) for path in paths(parsed)} == {tuple(LINE), tuple(OTHER_LINE)}


@pytest.mark.parametrize("missing", [".shx", ".dbf"])
def test_missing_required_shapefile_sidecar_is_actionable(tmp_path, missing):
    source = write_vector(tmp_path / "pipes.shp", [LineString(LINE)])
    source.with_suffix(missing).unlink()
    with pytest.raises(ImportFailure) as exc:
        prepare_import(source).fresh_parse()
    assert missing in str(exc.value).lower()


@pytest.mark.parametrize("name", ["../escape.geojson", "/absolute.geojson", "C:/drive.geojson",
                                 "nested/../../escape.geojson"])
def test_archive_path_escape_is_rejected(tmp_path, name):
    archive = tmp_path / "escape.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, '{"type":"FeatureCollection","features":[]}')
    with pytest.raises(ImportFailure):
        prepare_import(archive).fresh_parse()
    assert not (tmp_path / "escape.geojson").exists()


@pytest.mark.parametrize("names", [("pipes.geojson", "PIPES.GEOJSON"),
                                  ("pípés.geojson", "pi\u0301pe\u0301s.geojson")])
def test_ambiguous_case_and_unicode_archive_names_are_rejected(tmp_path, names):
    archive = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in names:
            zipped.writestr(name, '{"type":"FeatureCollection","features":[]}')
    with pytest.raises(ImportFailure):
        prepare_import(archive).fresh_parse()


def test_archive_symlink_member_is_not_followed(tmp_path):
    archive = tmp_path / "symlink.zip"
    member = zipfile.ZipInfo("pipes.geojson")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member, "../../private.geojson")
    with pytest.raises(ImportFailure):
        prepare_import(archive).fresh_parse()


def test_corrupt_zip_crc_does_not_publish_partial_data(tmp_path):
    source = write_geojson(tmp_path / "pipes.geojson", [LineString(LINE)])
    archive = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.write(source, source.name)
    data = archive.read_bytes()
    assert b"FeatureCollection" in data
    archive.write_bytes(data.replace(b"FeatureCollection", b"Broken_Collection", 1))
    with pytest.raises(ImportFailure):
        prepare_import(archive).fresh_parse()


def test_unknown_json_is_not_a_vector_dataset(tmp_path):
    source = tmp_path / "settings.json"
    source.write_text('{"settings":{"name":"not GIS"}}')
    with pytest.raises(ImportFailure):
        prepare_import(source).fresh_parse()


def test_zip_expanded_budget_includes_unrelated_members(tmp_path, monkeypatch):
    from pipeline_calculator.importing import archive as archive_module
    source = write_geojson(tmp_path / "pipes.geojson", [LineString(LINE)])
    archive = zip_files(tmp_path / "bounded.zip", [source])
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("unrelated.txt", b"x" * 4096)
    monkeypatch.setattr(archive_module, "MAX_EXPANDED_BYTES", 1024)
    with pytest.raises(ImportFailure) as exc:
        prepare_import(archive).fresh_parse()
    assert exc.value.code == "package_size_limit"


def test_zip_entry_budget_precedes_dataset_discovery(tmp_path, monkeypatch):
    from pipeline_calculator.importing import archive as archive_module
    archive = tmp_path / "too-many.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for i in range(3):
            zipped.writestr(f"file{i}.txt", b"x")
    monkeypatch.setattr(archive_module, "MAX_ENTRIES", 2)
    with pytest.raises(ImportFailure) as exc:
        prepare_import(archive).fresh_parse()
    assert exc.value.code == "archive_entry_limit"


def test_encrypted_zip_flag_is_rejected_before_decoding(tmp_path):
    source = write_geojson(tmp_path / "pipes.geojson", [LineString(LINE)])
    archive = zip_files(tmp_path / "encrypted.zip", [source])
    data = bytearray(archive.read_bytes())
    struct.pack_into("<H", data, 6, 1)
    central = data.index(b"PK\x01\x02")
    struct.pack_into("<H", data, central + 8, 1)
    archive.write_bytes(data)
    with pytest.raises(ImportFailure) as exc:
        prepare_import(archive).fresh_parse()
    assert exc.value.code == "encrypted_archive"
