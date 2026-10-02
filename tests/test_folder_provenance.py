"""Physical KML folder ownership survives every source and mileage route."""
from __future__ import annotations

from copy import deepcopy
from zipfile import ZipFile

import pytest
from shapely.geometry import box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.geography import BoundaryDataset
from pipeline_calculator.core.state_analysis import build_state_breakdown
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics
from pipeline_calculator.parsers.repair import RepairFailure
from pipeline_calculator.parsers.source import prepare_source


NS = 'http://www.opengis.net/kml/2.2'
POINT = '<Point><coordinates>-0.001,0</coordinates></Point>'
LINE = '<LineString><coordinates>-0.001,0 0.001,0</coordinates></LineString>'


def document(body, *, damaged=False):
    repair = ' xsi:schemaLocation="kml.xsd"' if damaged else ''
    return f'<kml xmlns="{NS}"><Document{repair}><name>Document title</name>{body}</Document></kml>'


def feature(name, geometry=POINT):
    return f'<Placemark><name>{name}</name>{geometry}</Placemark>'


def write_source(tmp_path, body, extension, *, damaged=False, child=None):
    path = tmp_path / ('source.' + extension)
    if extension == 'kmz':
        with ZipFile(path, 'w') as archive:
            archive.writestr('doc.kml', document(body, damaged=damaged))
            if child is not None:
                archive.writestr('child.kml', document(child))
    else:
        path.write_text(document(body, damaged=damaged), encoding='utf-8')
        if child is not None:
            (tmp_path / 'child.kml').write_text(document(child), encoding='utf-8')
    return path


@pytest.mark.parametrize('extension', ['kml', 'kmz'])
def test_nested_unnamed_duplicate_folders_keep_physical_owner_and_point_multiplicity(tmp_path, extension):
    mixed = '<MultiGeometry>' + LINE + POINT + POINT + '</MultiGeometry>'
    body = (feature('Root') + '<Folder><name>  System  </name>' + feature('Parent') +
            '<Folder><name>  Facilities \n</name>' + feature('Mixed', mixed) + '</Folder>' +
            '<Folder><name>Facilities</name>' + feature('Same title') + '</Folder>' +
            '<Folder><name>   </name>' + feature('Unnamed') + '</Folder></Folder>')
    path = write_source(tmp_path, body, extension)
    parsed = parse_kml_kmz_with_diagnostics(path)
    assert [(row['Name'], row['folder_path'], row['folder_name'], row['folder_id'])
            for row in parsed.placemarks] == [
        ('Root', [], '', ''),
        ('Parent', ['System'], 'System', 'folder-1'),
        ('Mixed', ['System', 'Facilities'], 'Facilities', 'folder-2'),
        ('Mixed', ['System', 'Facilities'], 'Facilities', 'folder-2'),
        ('Same title', ['System', 'Facilities'], 'Facilities', 'folder-3'),
        ('Unnamed', ['System', 'Unnamed folder'], 'Unnamed folder', 'folder-4'),
    ]
    line = parsed.pipelines[0]
    assert (line['folder_path'], line['folder_name'], line['folder_id']) == (
        ['System', 'Facilities'], 'Facilities', 'folder-2')
    assert all(row['source_kml'] == ('doc.kml' if extension == 'kmz' else str(path.resolve()))
               for row in parsed.pipelines + parsed.placemarks)
    # The same physical owner does not imply shared mutable per-record metadata.
    parsed.placemarks[2]['folder_path'].clear()
    assert parsed.placemarks[3]['folder_path'] == line['folder_path'] == ['System', 'Facilities']


@pytest.mark.parametrize('extension', ['kml', 'kmz'])
@pytest.mark.parametrize('damaged', [False, True])
def test_linked_documents_keep_source_local_folder_ancestry_and_identity(tmp_path, extension, damaged):
    link = '<NetworkLink><Link><href>child.kml</href></Link></NetworkLink>'
    mixed = '<MultiGeometry>' + LINE + POINT + '</MultiGeometry>'
    body = '<Folder><name>Facilities</name>' + feature('Root pin', mixed) + link + link + '</Folder>'
    child = '<Folder><name>Facilities</name>' + feature('Child pin') + '</Folder>'
    path = write_source(tmp_path, body, extension, child=child, damaged=damaged)
    parsed = (prepare_source(path).approve().fresh_parse() if damaged
              else parse_kml_kmz_with_diagnostics(path))
    assert len(parsed.placemarks) == 2
    assert [row['folder_path'] for row in parsed.placemarks] == [['Facilities'], ['Facilities']]
    assert [row['folder_id'] for row in parsed.placemarks] == ['folder-1', 'folder-1']
    assert len({(row['source_kml'], row['folder_id']) for row in parsed.placemarks}) == 2
    assert [row['source_kml'] for row in parsed.placemarks] == parsed.parsed_kml_files


@pytest.mark.parametrize('extension', ['kml', 'kmz'])
def test_repaired_snapshot_save_and_reimport_preserve_folder_membership(tmp_path, extension):
    body = '<Folder><name> System </name><Folder><name>Facilities</name>'
    body += feature('Mixed', '<MultiGeometry>' + LINE + POINT + POINT + '</MultiGeometry>')
    body += '</Folder><Folder><name>Facilities</name>' + feature('Sibling') + '</Folder></Folder>'
    path = write_source(tmp_path, body, extension, damaged=True)
    session = prepare_source(path).approve()
    before = session.fresh_parse()
    assert session.report['coverage']['point_count'] == 3
    assert [row['folder_id'] for row in before.placemarks] == ['folder-2', 'folder-2', 'folder-3']
    assert all(row['folder_path'] == ['System', 'Facilities']
               for row in before.pipelines + before.placemarks)
    output = tmp_path / ('repaired.' + extension)
    assert session.save(output)['ordinary_reimport_verified']
    after = parse_kml_kmz_with_diagnostics(output)
    expected = deepcopy(before)
    if extension == 'kml':
        for row in expected.pipelines + expected.placemarks:
            row['source_kml'] = str(output.resolve())
    assert after.pipelines == expected.pipelines
    assert after.placemarks == expected.placemarks


@pytest.mark.parametrize('collection', ['pipelines', 'placemarks'])
@pytest.mark.parametrize('field,wrong_value', [
    ('folder_path', ['Wrong parent', 'Facilities']),
    ('folder_name', 'Wrong owner'),
    ('folder_id', 'folder-2'),
    ('source_kml', 'wrong-document.kml'),
])
def test_repair_verification_rejects_changed_folder_or_document_attribution(
        tmp_path, monkeypatch, collection, field, wrong_value):
    from pipeline_calculator.parsers import kml_kmz as parser
    body = '<Folder><name>Facilities</name>'
    body += feature('Mixed', '<MultiGeometry>' + LINE + POINT + '</MultiGeometry>') + '</Folder>'
    path = write_source(tmp_path, body, 'kmz', damaged=True)
    ordinary = parser._parse_kml_bytes

    def corrupted(data, state, **kwargs):
        links = ordinary(data, state, **kwargs)
        getattr(state, collection)[0][field] = wrong_value
        return links

    monkeypatch.setattr(parser, '_parse_kml_bytes', corrupted)
    with pytest.raises(RepairFailure) as failure:
        prepare_source(path)
    assert 'projection_mismatch' in {item['code'] for item in failure.value.findings}


def test_pipeline_folder_metadata_survives_combined_state_and_fragment_results(tmp_path):
    body = '<Folder><name>System</name><Folder><name>Gas lines</name>' + feature('Crossing', LINE)
    body += '</Folder></Folder>'
    parsed = parse_kml_kmz_with_diagnostics(write_source(tmp_path, body, 'kml'))
    source = parsed.pipelines[0]
    analyzer = PipelineAnalyzer()
    combined = analyzer.analyze_parsed(parsed)
    states = BoundaryDataset({'AA': box(-1, -1, 0, 1), 'BB': box(0, -1, 1, 1)},
                             {'AA': 'West', 'BB': 'East'})
    geography = build_state_breakdown(analyzer, parsed.pipelines, combined, boundaries=states)
    assert geography['status'] == 'complete', geography['diagnostics']
    assert len(geography['states']) == 2
    rows = combined['pipelines'] + geography['fragments']
    rows += [row for state in geography['states'] for row in state['pipelines']]
    for row in rows:
        assert {key: row[key] for key in ('folder_name', 'folder_path', 'folder_id', 'source_kml')} == {
            'folder_name': 'Gas lines', 'folder_path': ['System', 'Gas lines'],
            'folder_id': 'folder-2', 'source_kml': str((tmp_path / 'source.kml').resolve()),
        }
        assert row['folder_path'] is not source['folder_path']


def test_unavailable_state_geometry_keeps_folder_metadata_on_unresolved_fragments(tmp_path, monkeypatch):
    import pipeline_calculator.core.geography as geography_module
    body = '<Folder><name>Gas lines</name>' + feature('Crossing', LINE) + '</Folder>'
    parsed = parse_kml_kmz_with_diagnostics(write_source(tmp_path, body, 'kml'))
    analyzer = PipelineAnalyzer()
    combined = analyzer.analyze_parsed(parsed)

    def unavailable():
        raise OSError('boundary resource unavailable')

    monkeypatch.setattr(geography_module, 'load_boundaries', unavailable)
    geography = build_state_breakdown(analyzer, parsed.pipelines, combined)
    assert geography['status'] == 'unavailable'
    assert len(geography['fragments']) == 1
    fragment = geography['fragments'][0]
    assert fragment['folder_path'] == ['Gas lines']
    assert fragment['folder_name'] == 'Gas lines'
    assert fragment['folder_id'] == 'folder-1'
    assert fragment['source_kml'] == parsed.pipelines[0]['source_kml']
