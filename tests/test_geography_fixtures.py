"""Real user archives, checked against independently prepared XML/length oracles."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.coordinates import coordinate_paths_for_pipeline
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.export.package import export_analysis_package
from pipeline_calculator.parsers.kml_kmz import extract_features_from_file_with_diagnostics


FIXTURES = Path(__file__).parent / 'fixtures' / 'geography'
POLYGONS = 'adamas_ng_pipeline_row'
CENTERLINES = POLYGONS + '_centerlines'


def expectations(name):
    return json.loads((FIXTURES / f'{name}.expected.json').read_text(encoding='utf-8'))


def xml_documents(path):
    with ZipFile(path) as archive:
        return [ET.fromstring(archive.read(name)) for name in archive.namelist()
                if name.lower().endswith('.kml')]


@pytest.mark.parametrize('name', [POLYGONS, CENTERLINES])
def test_fixture_integrity_and_independent_xml_types(name):
    expected = expectations(name)
    path = FIXTURES / expected['fixture']
    assert path.stat().st_size == expected['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
    documents = xml_documents(path)
    counts = Counter(element.tag.rsplit('}', 1)[-1]
                     for document in documents for element in document.iter())
    xml_expected = expected.get('xml_expectations', expected.get('xml_counts'))
    assert {key: counts[key] for key in xml_expected} == xml_expected
    if name == CENTERLINES:
        ids = Counter(element.attrib.get('id') for document in documents
                      for element in document.iter() if element.tag.rsplit('}', 1)[-1] == 'Placemark')
        assert ids == expected['geometry_characteristics']['duplicate_xml_ids']


@pytest.mark.parametrize('enabled', [False, True])
def test_polygon_only_archive_is_incomplete_and_contributes_no_pipeline_mileage(enabled):
    expected = expectations(POLYGONS)['parser_expectations']
    result = PipelineAnalyzer().analyze_complete(
        FIXTURES / f'{POLYGONS}.kmz', options=AnalysisOptions(enabled))
    assert result['pipelines'] == result['placemarks'] == []
    assert result['total_meters'] == result['total_miles'] == 0
    assert result['analysis_complete'] is False
    assert Counter(d['code'] for d in result['diagnostics']) == expected['diagnostic_counts']
    unsupported = next(d for d in result['diagnostics'] if d['code'] == 'no_supported_features')
    assert unsupported['level'] == 'error' and 'No supported pipeline' in unsupported['message']
    if enabled:
        assert result['geography']['status'] == 'incomplete'
        assert result['geography']['states'] == result['geography']['fragments'] == []
    else:
        assert 'geography' not in result


@pytest.fixture(scope='module')
def centerlines():
    path = FIXTURES / f'{CENTERLINES}.kmz'
    expected = expectations(CENTERLINES)
    count = expected['parser_expectations']['source_pipelines']
    assert len(expected['source_expectations']) == count
    assert [source['source_index'] for source in expected['source_expectations']] == list(range(count))
    return path, expected, extract_features_from_file_with_diagnostics(path)


def test_centerline_parser_preserves_repeated_ids_and_disconnected_paths(centerlines):
    path, expected, parsed = centerlines
    parser = expected['parser_expectations']
    assert len(parsed.pipelines) == parser['source_pipelines']
    assert len({p['id'] for p in parsed.pipelines}) == parser['unique_source_ids']
    assert {p['placemark_id'] for p in parsed.pipelines} == {'ID_00000'}
    assert not any(d['level'] in ('error', 'warning') for d in parsed.diagnostics)
    paths = [coordinate_paths_for_pipeline(p) for p in parsed.pipelines]
    assert sum(map(len, paths)) == parser['coordinate_paths']
    assert sum(len(points) for paths_ in paths for points in paths_) == parser['vertices']
    assert sum(len(paths_) > 1 for paths_ in paths) == expected['geometry_characteristics']['multipath_features']
    assert sum(points[0] == points[-1] for paths_ in paths for points in paths_) == expected['geometry_characteristics']['closed_paths']
    # Compare every original path and vertex with XML, independently of the app's
    # normalized records. This detects dropped paths as well as inserted bridges.
    xml_features = [element for document in xml_documents(path) for element in document.iter()
                    if element.tag.rsplit('}', 1)[-1] == 'Placemark']
    for pipeline, actual_paths, element, source in zip(parsed.pipelines, paths, xml_features, expected['source_expectations']):
        xml_paths = []
        for line in element.iter():
            if line.tag.rsplit('}', 1)[-1] == 'LineString':
                coords = next(c for c in line if c.tag.rsplit('}', 1)[-1] == 'coordinates')
                xml_paths.append([tuple(map(float, token.split(',')[:2])) for token in coords.text.split()])
        assert actual_paths == xml_paths
        assert pipeline['name'] == source['name']
        assert len(actual_paths) == source['path_count']
    rows, meters, _ = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)
    for row, source in zip(rows, expected['source_expectations']):
        assert row['Shape_Length'] == pytest.approx(source['original_meters'], rel=0, abs=.001)
    assert meters == pytest.approx(expected['geography_expectations']['original_meters'], rel=0, abs=.001)


def test_centerline_state_attribution_and_package_roundtrips(centerlines, tmp_path):
    path, expected, parsed = centerlines
    result = PipelineAnalyzer().analyze_complete(path, options=AnalysisOptions(True))
    geo = result['geography']
    target = expected['geography_expectations']
    assert result['analysis_complete'] and geo['status'] == 'complete', geo['diagnostics']
    assert geo['crossing_count'] == 0
    assert geo['reconciliation']['passed']
    for key in ('shared_meters', 'outside_meters', 'unresolved_meters'):
        assert geo['reconciliation'][key] == target[key] == 0
    assert result['total_meters'] == pytest.approx(target['original_meters'], rel=0, abs=.001)
    assert [state['state_code'] for state in geo['states']] == target['state_codes']
    grouped = defaultdict(list)
    for fragment in geo['fragments']:
        assert fragment['kind'] == 'state'
        grouped[fragment['source_id'], fragment['path_index']].append(fragment)
    assert len(grouped) == expected['parser_expectations']['coordinate_paths']
    for source, pipeline in zip(expected['source_expectations'], parsed.pipelines):
        source_fragments = []
        for index, points in enumerate(coordinate_paths_for_pipeline(pipeline)):
            fragments = grouped[pipeline['id'], index]
            # Every complete path is in an interior; no clipping or bridge is
            # needed for this fixture. Retain the exact original coordinate path.
            assert len(fragments) == 1
            assert fragments[0]['coordinates'] == [list(point) for point in points]
            assert fragments[0]['state_codes'] == [source['state_code']]
            source_fragments.extend(fragments)
        assert math.fsum(f['length_meters'] for f in source_fragments) == pytest.approx(
            source['original_meters'], rel=0, abs=.001)
    for state, expected_state in zip(geo['states'], target['states']):
        assert state['analysis_complete']
        assert len(state['pipelines']) == expected_state['source_pipelines']
        assert state['interior_meters'] == state['total_meters']
        assert state['total_meters'] == pytest.approx(expected_state['original_meters'], rel=0, abs=.001)
        assert sum(len(parts) for parts in grouped.values()
                   if parts[0]['state_codes'] == [state['state_code']]) == expected_state['coordinate_paths']
    assert math.fsum(s['total_meters'] for s in geo['states']) == pytest.approx(result['total_meters'], rel=0, abs=.001)
    package = export_analysis_package(result, tmp_path, path, include_maps=True, include_json=True)
    assert (package / 'analysis.xlsx').is_file()
    serialized = json.loads((package / 'analysis.json').read_text(encoding='utf-8'))
    assert serialized['geography']['status'] == 'complete'
    maps = [(package / 'Combined' / 'analysis.kmz', result['total_meters'])] + [
        (package / 'States' / s['state_name'] / 'analysis.kmz', s['interior_meters']) for s in geo['states']]
    assert set(package.rglob('*.kmz')) == {entry[0] for entry in maps}
    for map_path, meters in maps:
        imported = extract_features_from_file_with_diagnostics(map_path)
        _, actual, _ = PipelineAnalyzer().calculate_pipeline_lengths(imported.pipelines)
        assert actual == pytest.approx(meters, rel=0, abs=.001)
