"""Corridor decisions agree across analysis, previews and atomic report packages."""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from openpyxl import load_workbook
import pytest
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon, box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.corridor_geometry import normalize_polygons, prepare_corridor
from pipeline_calculator.core.execution import AnalysisCancelled
from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml
from pipeline_calculator.export.geography_kmz import build_geography_kml
from pipeline_calculator.export.package import export_analysis_package
from test_geography_export import geography_result


NS = {'k': 'http://www.opengis.net/kml/2.2'}


def read_map(path):
    with ZipFile(path) as archive:
        return ET.fromstring(archive.read('doc.kml'))


def polygons(root):
    def coordinates(element):
        return [tuple(map(float, p.split(',')[:2])) for p in element.text.split()]
    return [Polygon(coordinates(p.find('k:outerBoundaryIs/k:LinearRing/k:coordinates', NS)),
                    [coordinates(h) for h in p.findall('k:innerBoundaryIs/k:LinearRing/k:coordinates', NS)])
            for p in root.findall('.//k:Polygon', NS)]


@pytest.mark.parametrize('reverse', [False, True])
def test_default_dateline_analysis_exports_valid_maps_and_reimports(tmp_path, reverse):
    paths = [[(179.997, 52+i*.0001), (-179.997, 52+i*.0001)] for i in range(2)]
    if reverse:
        paths = [list(reversed(path)) for path in paths]
    source = tmp_path / 'antimeridian.kml'
    source.write_text('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>' + ''.join(
        f'<Placemark><name>{i}</name><LineString><coordinates>' +
        ' '.join(f'{lon},{lat}' for lon, lat in path) +
        '</coordinates></LineString></Placemark>' for i, path in enumerate(paths)) +
        '</Document></kml>', encoding='utf-8')
    analyzer = PipelineAnalyzer()
    result = analyzer.analyze_complete(source, options=AnalysisOptions(state_breakdown=True))
    assert result['analysis_complete'] and result['geography']['analysis_complete']
    assert result['total_meters'] == pytest.approx(824.1352745413919, abs=1e-6)
    assert result['geography']['reconciliation']['passed']
    before = deepcopy(result)
    output = export_analysis_package(result, tmp_path, source, include_json=True)
    assert result == before
    assert (output / 'analysis.xlsx').is_file() and (output / 'analysis.json').is_file()
    root = read_map(output / 'Combined' / 'analysis.kmz')
    shapes = polygons(root)
    # Paths are ~11 m apart: their two 5 m buffers stay separate, with each
    # split once at the dateline. Do not auto-widen to connect the gap.
    assert len(shapes) == 4 and all(shape.is_valid for shape in shapes)
    for points in root.findall('.//k:coordinates', NS):
        xy = [tuple(map(float, p.split(',')[:2])) for p in points.text.split()]
        assert all(abs(a[0]-b[0]) <= 180 for a, b in zip(xy, xy[1:]))
    reimported = analyzer.analyze_complete(output / 'Combined' / 'analysis.kmz')
    assert reimported['total_meters'] == pytest.approx(result['total_meters'], abs=.001)
    section = result['overlap_analysis']['bundled_sections'][0]
    preview_shapes = polygons(ET.fromstring(build_overlap_corridor_kml(section, 1)))
    assert len(preview_shapes) == 4
    # Canonical previews preserve the package's full-precision polygon vertices.
    for preview, exported in zip(preview_shapes, shapes):
        assert preview.equals_exact(exported, 0)


@pytest.mark.parametrize('representation', ['bad_preferred', 'bbox'])
def test_state_fallback_is_clipped_and_holes_remain_in_preview_and_map(geography_result, representation):
    state = Polygon([(-1, -1), (0, -1), (0, 1), (-1, 1)],
                    [[(-.4, -.1), (-.2, -.1), (-.2, .1), (-.4, .1)]])
    dataset = BoundaryDataset({'TX': state})
    section = {'pipeline_1': 'A', 'pipeline_2': 'B'}
    if representation == 'bad_preferred':
        section.update(corridor_polygon=[(-.8, -.5), (.5, .5), (-.8, .5), (.5, -.5), (-.8, -.5)],
                       oriented_polygon=[(-.8, -.5), (.5, -.5), (.5, .5), (-.8, .5)])
    else:
        section['bbox'] = dict(min_lon=-.8, max_lon=.5, min_lat=-.5, max_lat=.5)
    clipped = clip_state_corridor(section, 'TX', dataset, PipelineAnalyzer().geod)
    assert clipped['visualization_status'] == 'ready'
    assert any(d['code'] == 'corridor_geometry_fallback' for d in clipped['diagnostics'])
    assert len(clipped['clipped_polygons']) == 1
    assert len(clipped['clipped_polygons'][0]['holes']) == 1
    geography_result['geography']['states'][0]['overlap_analysis']['bundled_sections'] = [clipped]
    for xml in (build_overlap_corridor_kml(clipped, 1), build_geography_kml(geography_result, 'TX')):
        shapes = polygons(ET.fromstring(xml))
        assert len(shapes) == 1 and len(shapes[0].interiors) == 1
        assert state.covers(shapes[0])


def test_injected_alaska_dateline_boundary_preserves_multipart_and_holes(geography_result):
    dataset = BoundaryDataset({'AK': Polygon([(179, 50), (-179, 50), (-179, 54), (179, 54)],
                             [[(179.2, 51), (179.4, 51), (179.4, 52), (179.2, 52)]])})
    section = {'corridor_polygon': [(179.1, 50.5), (-179.1, 50.5), (-179.1, 53), (179.1, 53)]}
    clipped = clip_state_corridor(section, 'AK', dataset, PipelineAnalyzer().geod)
    assert clipped['visualization_status'] == 'ready'
    shapes = polygons(ET.fromstring(build_overlap_corridor_kml(clipped, 1)))
    assert len(shapes) == 2 and sum(len(p.interiors) for p in shapes) == 1
    assert all(dataset.geometries['AK'].covers(p) for p in shapes)
    assert all(p.bounds[2]-p.bounds[0] < 1 for p in shapes)


def test_hole_crossing_dateline_remains_excluded_and_multiple_world_turns_are_rejected():
    normalized = normalize_polygons([{
        'outer': [(179, 50), (-179, 50), (-179, 54), (179, 54)],
        'holes': [[(179.7, 51), (-179.7, 51), (-179.7, 52), (179.7, 52)]],
    }])
    shapes = [Polygon(p['outer'], p['holes']) for p in normalized]
    assert len(shapes) == 2 and all(p.is_valid for p in shapes)
    assert sum(p.area for p in shapes) == pytest.approx(7.4)
    assert not any(p.covers(Point(lon, 51.5)) for p in shapes for lon in (179.9, -179.9))
    with pytest.raises(ValueError, match='winds around the globe'):
        normalize_polygons([{'outer': [(0, 0), (170, 0), (-20, 0), (150, 0),
                                      (150, 1), (-20, 1), (170, 1), (0, 1)]}])


@pytest.mark.parametrize('section', [
    {},
    {'corridor_polygon': [(0, 0), (1, 1), (0, 1), (1, 0)]},
    {'corridor_polygon': [(2, 0), (3, 0), (3, 1), (2, 1)]},
    {'clipped_polygons': [], 'bbox': dict(min_lon=-1, max_lon=0, min_lat=0, max_lat=1)},
])
def test_unusable_or_outside_state_shapes_stay_omitted(section):
    dataset = BoundaryDataset({'TX': box(-1, -1, 0, 1)})
    clipped = clip_state_corridor(section, 'TX', dataset, PipelineAnalyzer().geod)
    assert clipped['clipped_polygons'] == [] and clipped['visualization_status'] == 'omitted'
    assert any(d['code'] == 'state_corridor_omitted' for d in clipped['diagnostics'])
    assert prepare_corridor(clipped, require_clipped=True)['visualization_polygons'] == []


def test_package_preflight_discloses_one_omission_without_mutating_results(geography_result, tmp_path):
    valid = {'pipeline_1': 'A', 'pipeline_2': 'B',
             'corridor_polygon': [(-100, 35), (-99.999, 35), (-99.999, 35.001), (-100, 35.001)]}
    invalid = {'pipeline_1': 'missing', 'pipeline_2': 'geometry', 'bundled_length_miles': .02}
    geography_result['overlap_analysis']['bundled_sections'] = [valid, invalid]
    before = deepcopy(geography_result)
    output = export_analysis_package(geography_result, tmp_path, include_json=True)
    assert geography_result == before
    exported = json.loads((output / 'analysis.json').read_text(encoding='utf-8'))
    assert exported['total_meters'] == before['total_meters']
    assert exported['overlap_analysis']['savings_miles'] == before['overlap_analysis']['savings_miles']
    assert [s['visualization_status'] for s in exported['overlap_analysis']['bundled_sections']] == ['ready', 'omitted']
    omitted = [d for d in exported['diagnostics'] if d['code'] == 'corridor_visualization_omitted']
    assert len(omitted) == 1 and omitted[0]['context']['section_index'] == 1
    workbook = load_workbook(output / 'analysis.xlsx')
    assert 'Diagnostics' in workbook
    assert any(cell.value == 'corridor_visualization_omitted' for row in workbook['Diagnostics'] for cell in row)
    root = read_map(output / 'Combined' / 'analysis.kmz')
    assert len(polygons(root)) == 1
    assert '1 corridor visualization(s) omitted' in root.find('k:Document/k:description', NS).text


def test_combined_omission_is_available_to_ui_before_export(tmp_path, monkeypatch):
    original = PipelineAnalyzer.analyze_features
    def inject(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        for section in (result.get('overlap_analysis') or {}).get('bundled_sections', []):
            for key in ('corridor_polygon', 'oriented_polygon', 'bbox'):
                section.pop(key, None)
            section['visualization_polygons'] = []
        return result
    monkeypatch.setattr(PipelineAnalyzer, 'analyze_features', inject)
    source = tmp_path / 'pipelines.kml'
    source.write_text('<kml><Document>' + ''.join(
        f'<Placemark><name>{i}</name><LineString><coordinates>-100,{31+i*.00005} -99.996,{31+i*.00005}</coordinates></LineString></Placemark>'
        for i in range(2)) + '</Document></kml>', encoding='utf-8')
    result = PipelineAnalyzer().analyze_complete(source, options=AnalysisOptions(state_breakdown=True))
    assert result['analysis_complete']
    assert result['overlap_analysis']['bundled_sections'][0]['visualization_status'] == 'omitted'
    assert any(d['code'] == 'corridor_visualization_omitted' for d in result['diagnostics'])


def test_geos_failures_become_omissions_and_cancellation_propagates(monkeypatch):
    from pipeline_calculator.core import corridor_geometry
    data = {'corridor_polygon': [(0, 0), (.01, 0), (.01, .01), (0, .01)]}
    def fail(*args, **kwargs):
        raise GEOSException('Injected geometry engine failure')
    monkeypatch.setattr(corridor_geometry, 'normalize_polygons', fail)
    assert prepare_corridor(data)['visualization_status'] == 'omitted'
    class Cancelled:
        def check(self):
            raise AnalysisCancelled()
    with pytest.raises(AnalysisCancelled):
        prepare_corridor(data, context=Cancelled())
    with pytest.raises(AnalysisCancelled):
        normalize_polygons([{'outer': data['corridor_polygon']}], context=Cancelled())


@pytest.mark.parametrize('dateline', [False, True])
def test_nonround_clipped_edges_remain_contained_after_serialization(geography_result, dateline):
    if dateline:
        outline = [(179.12345678901234, 51.12345678901234), (-179.8765432109876, 51.24681357913579),
                   (-179.9876543210987, 52.54321098765432), (179.23456789012345, 52.65432109876543)]
        corridor = [(179, 51), (-179.8, 51), (-179.8, 53), (179, 53)]
    else:
        outline = [(-100.12345678901234, 31.12345678901234), (-99.8765432109876, 31.24681357913579),
                   (-99.9876543210987, 32.54321098765432), (-100.23456789012345, 32.65432109876543)]
        corridor = [(-100.5, 31), (-99.5, 31), (-99.5, 33), (-100.5, 33)]
    dataset = BoundaryDataset({'TX': Polygon(outline)})
    clipped = clip_state_corridor({'corridor_polygon': corridor}, 'TX', dataset, PipelineAnalyzer().geod)
    assert clipped['visualization_status'] == 'ready'
    revalidated = prepare_corridor(clipped, require_clipped=True)
    assert revalidated['visualization_polygons'] == clipped['clipped_polygons']
    geography_result['geography']['states'][0]['overlap_analysis']['bundled_sections'] = [clipped]
    for xml in (build_overlap_corridor_kml(clipped, 1), build_geography_kml(geography_result, 'TX')):
        for part in polygons(ET.fromstring(xml)):
            assert dataset.geometries['TX'].covers(part)


@pytest.mark.parametrize('middle', [180, -180])
def test_map_split_handles_an_existing_dateline_vertex(geography_result, middle, tmp_path):
    fragment = geography_result['geography']['fragments'][0]
    fragment['coordinates'] = [(179.999, 52), (middle, 52.000001), (-179.999, 52)]
    geography_result['geography']['fragments'] = [fragment]
    for reverse in (False, True):
        if reverse:
            fragment['coordinates'].reverse()
        xml = build_geography_kml(geography_result)
        root = ET.fromstring(xml)
        for node in root.findall('.//k:LineString/k:coordinates', NS):
            points = [tuple(map(float, p.split(',')[:2])) for p in node.text.split()]
            assert all(abs(a[0]-b[0]) <= 180 for a, b in zip(points, points[1:]))
        source = tmp_path / 'map.kml'
        source.write_text(xml, encoding='utf-8')
        geod = PipelineAnalyzer().geod
        expected = sum(geod.inv(*a, *b)[2] for a, b in zip(fragment['coordinates'], fragment['coordinates'][1:]))
        assert PipelineAnalyzer().analyze_complete(source)['total_meters'] == pytest.approx(expected, abs=.001)
