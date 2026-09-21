"""Point pins survive the same immutable source and repair routes as mileage."""
from __future__ import annotations

import zipfile

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics
from pipeline_calculator.parsers.repair import RepairFailure, inspect_geometry
from pipeline_calculator.parsers.source import prepare_source


NS = "http://www.opengis.net/kml/2.2"
LINE = '<LineString><coordinates>-100,40,0 -99.999,40,0</coordinates></LineString>'
POINT = '<Point><coordinates>-100,40,10</coordinates></Point>'
RING = '<LinearRing><coordinates>-100,40 -99.999,40 -99.999,40.001 -100,40</coordinates></LinearRing>'
POLYGON = '<Polygon><outerBoundaryIs>' + RING + '</outerBoundaryIs></Polygon>'
BODY = (
    '<Placemark><MultiGeometry>' + LINE + POINT +
    '<MultiGeometry>' + POINT + POLYGON + '</MultiGeometry></MultiGeometry></Placemark>'
    '<Placemark><visibility>0</visibility><MultiGeometry>' + POINT + POINT + '</MultiGeometry></Placemark>'
    '<Placemark>' + LINE + '</Placemark>'
)


def document(body=BODY, *, repaired=False):
    metadata = ' xsi:schemaLocation="kml.xsd"' if repaired else ''
    return f'<kml xmlns="{NS}"><Document{metadata}>{body}</Document></kml>'.encode()


def write_source(path, content):
    if path.suffix == '.kmz':
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('doc.kml', content)
    else:
        path.write_bytes(content)
    return path


def assert_pins(parsed):
    assert parsed.placemarks == [
        {'Placemark_ID': 'PM_1', 'Name': 'Item_1', 'Count': 1},
        {'Placemark_ID': 'PM_2', 'Name': 'Item_1', 'Count': 1},
        {'Placemark_ID': 'PM_3', 'Name': 'Item_2', 'Count': 1},
        {'Placemark_ID': 'PM_4', 'Name': 'Item_2', 'Count': 1},
    ]
    # Extra pins must not renumber the source pipelines' generated names/IDs.
    assert [(p['id'], p['name']) for p in parsed.pipelines] == [(0, 'Item_1'), (1, 'Item_3')]


@pytest.mark.parametrize('suffix', ['.kml', '.kmz'])
@pytest.mark.parametrize('repaired', [False, True])
def test_point_pins_survive_snapshot_retry_and_repaired_save(tmp_path, suffix, repaired):
    path = write_source(tmp_path / ('source' + suffix), document(repaired=repaired))
    original_bytes = path.read_bytes()
    session = prepare_source(path)
    assert session.requires_repair is repaired
    if repaired:
        assert session.report['coverage']['point_count'] == 4
        session.approve()
    parsed = session.fresh_parse()
    assert_pins(parsed)
    result = PipelineAnalyzer().analyze_parsed(parsed)
    assert result['analysis_complete']
    assert sum(p['Count'] for p in result['placemarks']) == 4
    assert path.read_bytes() == original_bytes

    # Neither client changes nor mutations of a previous result affect retries.
    parsed.placemarks.clear()
    path.unlink()
    retry = session.fresh_parse()
    assert_pins(retry)
    repeated = PipelineAnalyzer().analyze_parsed(retry)
    assert repeated['total_meters'] == result['total_meters']
    assert repeated['overlap_analysis'] == result['overlap_analysis']

    if repaired:
        output = tmp_path / ('repaired' + suffix)
        receipt = session.save(output)
        assert receipt['ordinary_reimport_verified']
        reopened = parse_kml_kmz_with_diagnostics(output)
        assert_pins(reopened)
        assert prepare_source(output).fresh_parse().placemarks == retry.placemarks


def test_independent_repair_inventory_has_all_pins_not_line_or_polygon_vertices():
    inventory = inspect_geometry(document())
    assert not inventory['findings']
    assert len(inventory['pipelines']) == 2
    assert len(inventory['points']) == 4
    assert [feature['kind'] for feature in inventory['features']] == [
        'pipeline', 'point', 'point', 'point', 'point', 'pipeline',
    ]


def test_linked_documents_count_once_and_keep_names_across_repair(tmp_path):
    link = '<NetworkLink><Link><href>child.kml</href></Link></NetworkLink>'
    path = tmp_path / 'linked.kmz'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('doc.kml', document(BODY + link + link, repaired=True))
        archive.writestr('child.kml', document())
        archive.writestr('unused.kml', document('<Placemark>' + POINT + '</Placemark>'))
    session = prepare_source(path).approve()
    parsed = session.fresh_parse()
    assert parsed.parsed_kml_files == ['doc.kml', 'child.kml']
    assert len(parsed.placemarks) == 8
    assert [p['name'] for p in parsed.pipelines] == ['Item_1', 'Item_3', 'Item_4', 'Item_6']
    assert [p['Name'] for p in parsed.placemarks] == [
        'Item_1', 'Item_1', 'Item_2', 'Item_2', 'Item_4', 'Item_4', 'Item_5', 'Item_5',
    ]
    assert 'unparsed_kml_file' in {d['code'] for d in parsed.diagnostics}
    session.save(tmp_path / 'linked_repaired.kmz')


@pytest.mark.parametrize('point,code', [
    ('<Point/>', 'missing_point_coordinate'),
    ('<Point><coordinates>-100,40 -99,40</coordinates></Point>', 'ambiguous_point_coordinate'),
    ('<Point><coordinates>-100,40,nan</coordinates></Point>', 'invalid_coordinate'),
])
def test_bad_pin_beside_line_remains_incomplete_and_repair_does_not_hide_it(tmp_path, point, code):
    body = '<Placemark><MultiGeometry>' + LINE + point + POINT + '</MultiGeometry></Placemark>'
    path = write_source(tmp_path / 'source.kmz', document(body))
    result = PipelineAnalyzer().analyze_parsed(prepare_source(path).fresh_parse())
    assert len(result['pipelines']) == 1
    assert len(result['placemarks']) == 1
    assert result['analysis_complete'] is False
    assert code in {d['code'] for d in result['diagnostics']}
    write_source(path, document(body, repaired=True))
    with pytest.raises(RepairFailure) as error:
        prepare_source(path)
    assert code in {finding['code'] for finding in error.value.findings}


def test_desktop_worker_handles_many_polygon_exclusions_with_pins(tmp_path):
    body = '<Placemark><MultiGeometry>' + LINE + POINT + POINT + '</MultiGeometry></Placemark>'
    body += ('<Placemark>' + POLYGON + '</Placemark>') * 10_001
    path = write_source(tmp_path / 'many.kmz', document(body))
    job = AnalysisController().start(str(path), AnalysisParameters())
    assert job.done.wait(30), 'Desktop analysis did not complete'
    job._thread.join(5)
    assert job.state == 'completed', job.error
    assert job.result['analysis_complete']
    assert len(job.result['placemarks']) == 2
    assert len(job.result['pipelines']) == 1
    assert len(job.result['diagnostics']) < 10
    assert all(d['level'] == 'info' for d in job.result['diagnostics'])
