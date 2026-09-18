"""Valid XML must not conceal omitted, duplicated or misresolved pipeline paths."""
from __future__ import annotations

import zipfile

import pytest

from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics
from pipeline_calculator.parsers.repair import RepairFailure
from pipeline_calculator.parsers.source import prepare_source


NS = 'http://www.opengis.net/kml/2.2'
GX = 'http://www.google.com/kml/ext/2.2'
LINE = '<LineString><coordinates>-100,30 -100.01,30</coordinates></LineString>'


def document(body, namespace=NS):
    declaration = f' xmlns="{namespace}"' if namespace else ''
    return f'<kml{declaration} xmlns:gx="{GX}"><Document>{body}</Document></kml>'


def input_file(tmp_path, body, container):
    path = tmp_path / ('input.' + container)
    xml = document(body)
    if container == 'kml':
        path.write_text(xml, encoding='utf-8')
    else:
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('doc.kml', xml)
    return path


@pytest.mark.parametrize('container', ['kml', 'kmz'])
@pytest.mark.parametrize('body,code', [
    ('<Placemark><LineString><coordinates>-100,30 -100.01,30</coordinates>'
     '<coordinates>-100,30 -110,30</coordinates></LineString></Placemark>',
     'ambiguous_coordinate_structure'),
    ('<Placemark><LineString><coordinates>-100,30 -100.01,30<more/> -110,30'
     '</coordinates></LineString></Placemark>', 'unknown_coordinate_structure'),
    ('<Placemark><Placemark>' + LINE + '</Placemark></Placemark>', 'nested_placemark'),
    ('<Placemark>' + LINE + '</Placemark><Placemark><linestring><coordinates>'
     '-100,30 -110,30</coordinates></linestring></Placemark>', 'geometry_case_ambiguity'),
    ('<Placemark><LineString xmlns="urn:foreign"><coordinates>-100,30 -110,30'
     '</coordinates></LineString></Placemark>', 'geometry_namespace_ambiguity'),
    ('<Placemark><gx:Track><gx:coord>-100 30 0</gx:coord><gx:coord>-100.01 30 0'
     '<more/> -110 30 0</gx:coord></gx:Track></Placemark>', 'unknown_coordinate_structure'),
    ('<NetworkLink><Link><href>a.kml</href><href>b.kml</href></Link></NetworkLink>',
     'ambiguous_network_link'),
    ('<NetworkLinkControl><Update><targetHref>other.kml</targetHref><Change>'
     '<Placemark targetId="old">' + LINE + '</Placemark></Change></Update>'
     '</NetworkLinkControl>', 'unsupported_update_geometry'),
])
def test_ambiguous_geometry_blocks_both_input_paths(tmp_path, container, body, code):
    path = input_file(tmp_path, body, container)
    original = path.read_bytes()
    with pytest.raises(RepairFailure) as caught:
        prepare_source(path)
    assert code in {item['code'] for item in caught.value.findings}
    assert 'without deleting or reconnecting geometry' in caught.value.client_request()
    with pytest.raises(ValueError, match='complete pipeline geometry cannot be read safely'):
        parse_kml_kmz_with_diagnostics(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('namespace', [NS, '', 'http://earth.google.com/kml/2.0',
                                     'http://earth.google.com/kml/2.1',
                                     'http://earth.google.com/kml/2.2'])
def test_ordinary_supported_containers_namespaces_and_noncenterlines_unchanged(tmp_path, namespace):
    body = ('<Folder><Placemark><name>Pipe</name><MultiGeometry>' + LINE + LINE +
            '</MultiGeometry></Placemark><Placemark><Point><coordinates>-100,30'
            '</coordinates></Point></Placemark><Placemark><Polygon><outerBoundaryIs>'
            '<LinearRing><coordinates>-100,30 -100.01,30 -100.01,30.01 -100,30'
            '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Folder>')
    path = tmp_path / 'supported.kml'
    path.write_text(document(body, namespace), encoding='utf-8')
    ordinary = parse_kml_kmz_with_diagnostics(path)
    session = prepare_source(path)
    assert not session.requires_repair
    assert session.fresh_parse() == ordinary
    assert len(ordinary.pipelines) == 1
    assert len(ordinary.pipelines[0]['coordinate_paths']) == 2
    assert len(ordinary.placemarks) == 1
    assert [item['code'] for item in ordinary.diagnostics] == ['unsupported_geometry']


@pytest.mark.parametrize('container', ['kml', 'kmz'])
def test_xml_base_cannot_silently_select_wrong_existing_child(tmp_path, container):
    body = ('<Folder xml:base="sub/"><NetworkLink><Link><href>child.kml</href>'
            '</Link></NetworkLink></Folder>')
    path = input_file(tmp_path, body, container)
    wrong = document('<Placemark><name>Wrong child</name>' + LINE + '</Placemark>')
    intended = document('<Placemark><name>Intended child</name>' +
                        LINE.replace('-100.01,30', '-110,30') + '</Placemark>')
    if container == 'kmz':
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr('child.kml', wrong)
            archive.writestr('sub/child.kml', intended)
    else:
        (tmp_path / 'child.kml').write_text(wrong, encoding='utf-8')
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub/child.kml').write_text(intended, encoding='utf-8')
    with pytest.raises(RepairFailure) as caught:
        prepare_source(path)
    assert 'unsupported_xml_base' in {item['code'] for item in caught.value.findings}
    with pytest.raises(ValueError, match='complete pipeline geometry cannot be read safely'):
        parse_kml_kmz_with_diagnostics(path)


def test_worker_does_not_publish_partial_or_duplicate_result(tmp_path):
    path = input_file(tmp_path, '<Placemark><Placemark>' + LINE + '</Placemark></Placemark>', 'kmz')
    job = AnalysisController().start(str(path), AnalysisParameters())
    assert job.done.wait(10)
    job._thread.join(5)
    assert job.state == 'failed' and job.result is None and job.source_session is None
    assert isinstance(job.error, RepairFailure)
    assert 'nested_placemark' in {item['code'] for item in job.error.findings}
