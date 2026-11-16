"""Portable SQLite capabilities and canonical Unicode shapefile identities."""
from pathlib import Path
import unicodedata

import pytest
from shapely.geometry import LineString

from pipeline_calculator.importing import prepare_import
from pipeline_calculator.importing import archive, worker
from pipeline_calculator.importing.models import ImportFailure
from gis_helpers import LINE, paths, write_vector


class _ConnectionWithoutExtensions:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return self


class _ConnectionWithExtensions(_ConnectionWithoutExtensions):
    def __init__(self):
        super().__init__()
        self.extension_decisions = []

    def enable_load_extension(self, enabled):
        self.extension_decisions.append(enabled)


@pytest.mark.parametrize('connection_type', [_ConnectionWithoutExtensions, _ConnectionWithExtensions])
def test_database_disables_extensions_only_when_the_sqlite_build_exposes_api(tmp_path, monkeypatch, connection_type):
    connection = connection_type()
    calls = []
    def connect(database, **kwargs):
        calls.append((database, kwargs))
        return connection
    monkeypatch.setattr(worker.sqlite3, 'connect', connect)
    assert worker._database(tmp_path / 'source.gpkg') is connection
    assert calls[0][0].endswith('?mode=ro&immutable=1') and calls[0][1] == {'uri': True}
    assert connection.statements == ['PRAGMA query_only=ON', 'PRAGMA trusted_schema=OFF']
    if isinstance(connection, _ConnectionWithExtensions):
        assert connection.extension_decisions == [False]


def test_loose_shapefile_pairs_canonical_unicode_equivalent_component_stems(tmp_path):
    nfc = 'caf\u00e9'
    nfd = unicodedata.normalize('NFD', nfc)
    source = write_vector(tmp_path / (nfc + '.shp'), [LineString(LINE)])
    for component in list(tmp_path.iterdir()):
        if component.suffix.lower() != '.shp':
            component.rename(tmp_path / (nfd + component.suffix))
    parsed = prepare_import(source).fresh_parse()
    assert paths(parsed) == [LINE]


def test_loose_shapefile_refuses_unicode_equivalent_duplicate_component_names(tmp_path, monkeypatch):
    # Simulate a case-sensitive directory inventory even on normalization-
    # insensitive filesystems, where these two names cannot coexist on disk.
    nfc = 'caf\u00e9'
    nfd = unicodedata.normalize('NFD', nfc)
    source = write_vector(tmp_path / (nfc + '.shp'), [LineString(LINE)])
    duplicates = [source, tmp_path / (nfc + '.shx'), tmp_path / (nfd + '.shx'), tmp_path / (nfc + '.dbf')]
    original_iterdir, original_is_file = Path.iterdir, Path.is_file
    monkeypatch.setattr(Path, 'iterdir', lambda path: iter(duplicates) if path == tmp_path else original_iterdir(path))
    monkeypatch.setattr(Path, 'is_file', lambda path: True if path in duplicates else original_is_file(path))
    with pytest.raises(ImportFailure) as error:
        archive.capture(source)
    assert error.value.code == 'ambiguous_sidecar'
