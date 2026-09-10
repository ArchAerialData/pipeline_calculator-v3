"""Numerical/fixture regressions for the September 2026 calculation audit."""
from __future__ import annotations

import zipfile

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.export.xlsx import build_analysis_workbook
from pipeline_calculator.parsers.kml_kmz import parse_kml_kmz_with_diagnostics


def line(analyzer, start, length, bearing=0):
    return [start, analyzer.geod.fwd(*start, bearing, length)[:2]]


def parallel_lines(analyzer, length=300, separation=2, start=(-100, 40), bearing=0):
    other = analyzer.geod.fwd(*start, bearing + 90, separation)[:2]
    return [
        {"name": "A", "coordinates": line(analyzer, start, length, bearing)},
        {"name": "B", "coordinates": line(analyzer, other, length, bearing)},
    ]


def summarize(analyzer, pipelines):
    data, total, _ = analyzer.calculate_pipeline_lengths(pipelines)
    groups = analyzer.find_parallel_segments(pipelines)
    overlap = analyzer.calculate_overlap_results(pipelines, groups)
    effective = analyzer.compute_effective_length_by_clusters(pipelines, [d["Shape_Length"] for d in data])
    assert total - effective == pytest.approx(overlap["savings_meters"], abs=1e-6)
    return overlap, total, effective


def kml(body):
    return '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"><Document>' + body + '</Document></kml>'


def feature(coords, name="Pipe"):
    return f'<Placemark><name>{name}</name><LineString><coordinates>{coords}</coordinates></LineString></Placemark>'


def write_pipelines(tmp_path, pipelines):
    text = kml(''.join(feature(' '.join(f'{lon},{lat},0' for lon, lat in p['coordinates']), p['name']) for p in pipelines))
    path = tmp_path / 'pipelines.kmz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('doc.kml', text)
    return path


@pytest.mark.parametrize('radius', [5, 8, 15, 30])
@pytest.mark.parametrize('reverse', [False, True])
def test_unique_bundled_length_independent_of_neighbor_count(radius, reverse):
    analyzer = PipelineAnalyzer(detection_range=radius)
    pipelines = parallel_lines(analyzer)
    if reverse:
        pipelines[1]['coordinates'].reverse()
    overlap, total, effective = summarize(analyzer, pipelines)
    assert len(overlap['bundled_sections']) == 1
    assert overlap['total_bundled_length'] == pytest.approx(300, abs=5)
    assert total == pytest.approx(600, abs=1e-5)
    assert effective == pytest.approx(300, abs=5)


def test_below_minimum_kmz_has_no_savings(tmp_path):
    analyzer = PipelineAnalyzer(detection_range=5, min_parallel_length=200)
    path = write_pipelines(tmp_path, parallel_lines(analyzer, length=100))
    result = analyzer.analyze_complete(path)
    overlap = result['overlap_analysis']
    assert overlap['bundled_sections'] == []
    assert overlap['savings_meters'] == 0
    assert overlap['effective_total_meters'] == result['total_meters']
    assert build_analysis_workbook(result)['Pipeline Overlap Analysis']['D2'].value == 0


def test_separate_short_paths_do_not_combine_to_meet_minimum():
    analyzer = PipelineAnalyzer(min_parallel_length=150)
    first = parallel_lines(analyzer, length=100)
    second = parallel_lines(analyzer, length=100, start=(-99, 40))
    for p1, p2 in zip(first, second):
        p1['coordinate_paths'] = [p1['coordinates'], p2['coordinates']]
    overlap, total, effective = summarize(analyzer, first)
    assert overlap['bundled_sections'] == []
    assert effective == total


@pytest.mark.parametrize('start,bearing', [((-100, 0), 90), ((179.999, 60), 90), ((-100, 80), 0)])
def test_spatial_search_preserves_just_inside_neighbors(start, bearing):
    analyzer = PipelineAnalyzer(detection_range=15)
    inside = parallel_lines(analyzer, separation=14.99, start=start, bearing=bearing)
    overlap, _, _ = summarize(analyzer, inside)
    assert overlap['total_bundled_length'] == pytest.approx(300, abs=5)
    outside = parallel_lines(analyzer, separation=15.1, start=start, bearing=bearing)
    overlap, total, effective = summarize(analyzer, outside)
    assert overlap['bundled_sections'] == []
    assert effective == total


def test_three_identical_pipelines_discount_only_twice_and_order_is_irrelevant():
    analyzer = PipelineAnalyzer()
    coords = line(analyzer, (-100, 40), 300)
    pipelines = [{'name': name, 'coordinates': coords} for name in ['A', 'B', 'C']]
    overlap, _, effective = summarize(analyzer, pipelines)
    assert len(overlap['bundled_sections']) == 3  # pairwise reports, not additive savings
    assert effective == pytest.approx(300, abs=5)
    overlap2, _, effective2 = summarize(analyzer, pipelines[::-1])
    assert effective2 == pytest.approx(effective)
    assert overlap2['savings_meters'] == pytest.approx(600, abs=5)


@pytest.mark.parametrize('href', ['layers/../pipes/data.kml', './pipes/data.kml', 'layers/%2e%2e/pipes/data.kml'])
def test_kmz_internal_paths_are_normalized(tmp_path, href):
    path = tmp_path / 'linked.kmz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('doc.kml', kml(f'<NetworkLink><Link><href>{href}</href></Link></NetworkLink>'))
        archive.writestr('pipes/data.kml', kml(feature('-100,40 -100,40.001')))
    result = parse_kml_kmz_with_diagnostics(path)
    assert len(result.pipelines) == 1
    assert result.parsed_kml_files == ['doc.kml', 'pipes/data.kml']


def test_nested_kmz_parent_link_and_cycle_are_bounded(tmp_path):
    path = tmp_path / 'nested.kmz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('doc.kml', kml('<NetworkLink><Link><href>layers/index.kml</href></Link></NetworkLink>'))
        archive.writestr('layers/index.kml', kml('<NetworkLink><Link><href>../pipes/data.kml</href></Link></NetworkLink>'))
        archive.writestr('pipes/data.kml', kml(feature('-100,40 -100,40.001') + '<NetworkLink><Link><href>../doc.kml</href></Link></NetworkLink>'))
    assert len(parse_kml_kmz_with_diagnostics(path).pipelines) == 1


@pytest.mark.parametrize('href', ['../pipes/data.kml', 'a/../../pipes/data.kml', '/pipes/data.kml', 'file:///pipes/data.kml'])
def test_kmz_paths_cannot_leave_archive(tmp_path, href):
    path = tmp_path / 'escape.kmz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('doc.kml', kml(f'<NetworkLink><Link><href>{href}</href></Link></NetworkLink>'))
        archive.writestr('pipes/data.kml', kml(feature('-100,40 -100,40.001')))
    result = parse_kml_kmz_with_diagnostics(path)
    assert result.pipelines == []
    assert any(d['code'] == 'unsupported_network_link_target' for d in result.diagnostics)


@pytest.mark.parametrize('track', [False, True])
def test_invalid_interior_coordinate_rejects_only_affected_geometry(tmp_path, track):
    bad_geometry = ('<gx:Track><gx:coord>-100 40 0</gx:coord><gx:coord>bad</gx:coord>'
                    '<gx:coord>-99 40 0</gx:coord></gx:Track>' if track else
                    '<LineString><coordinates>-100,40 bad -99,40</coordinates></LineString>')
    path = tmp_path / 'invalid.kml'
    path.write_text(kml('<Placemark><MultiGeometry>' + bad_geometry +
                        '<LineString><coordinates>-100,40 -100,40.001</coordinates></LineString>'
                        '</MultiGeometry></Placemark>'))
    result = PipelineAnalyzer().analyze_complete(path)
    assert 100 < result['total_meters'] < 120  # no invented 85 km bridge
    assert result['analysis_complete'] is False
    assert any(d['level'] == 'error' for d in result['diagnostics'])
    workbook = build_analysis_workbook(result)
    assert 'INCOMPLETE' in workbook['Pipeline Length Analysis']['D1'].value


@pytest.mark.parametrize('failure_stage', ['find_parallel_segments', 'calculate_overlap_results'])
def test_overlap_failure_is_explicit_in_result_and_export(tmp_path, monkeypatch, failure_stage):
    analyzer = PipelineAnalyzer()
    path = write_pipelines(tmp_path, parallel_lines(analyzer))
    def fail(*args, **kwargs):
        raise RuntimeError('injected calculation failure')
    monkeypatch.setattr(analyzer, failure_stage, fail)
    result = analyzer.analyze_complete(path)
    assert result['total_meters'] == pytest.approx(600, abs=1e-5)
    assert result['analysis_complete'] is False
    assert result['overlap_analysis'] is None
    assert any(d['code'] == 'overlap_analysis_failed' for d in result['diagnostics'])
    workbook = build_analysis_workbook(result)
    assert workbook['Pipeline Overlap Analysis']['D2'].value == 'Unavailable'
    assert 'Diagnostics' in workbook.sheetnames


def test_spatial_index_failure_propagates(tmp_path, monkeypatch):
    from pipeline_calculator.core import overlap
    analyzer = PipelineAnalyzer()
    path = write_pipelines(tmp_path, parallel_lines(analyzer))
    def fail(*args, **kwargs):
        raise RuntimeError('index unavailable')
    monkeypatch.setattr(overlap, 'KDTree', fail)
    result = analyzer.analyze_complete(path)
    assert result['analysis_complete'] is False
    assert result['overlap_analysis'] is None


def test_unequal_lengths_do_not_discount_more_than_common_coverage():
    analyzer = PipelineAnalyzer(min_parallel_length=50)
    pipelines = parallel_lines(analyzer)
    pipelines[1]['coordinates'] = line(analyzer, pipelines[1]['coordinates'][0], 100)
    overlap, total, effective = summarize(analyzer, pipelines)
    assert overlap['total_bundled_length'] == pytest.approx(100, abs=5)
    assert total - effective == pytest.approx(100, abs=5)


def test_corridor_preserves_full_section_extent():
    analyzer = PipelineAnalyzer()
    pipelines = parallel_lines(analyzer)
    overlap, _, _ = summarize(analyzer, pipelines)
    ring = overlap['bundled_sections'][0]['corridor_polygon']
    start, end = pipelines[0]['coordinates']
    assert min(analyzer.geod.inv(*start, *point)[2] for point in ring) < 20
    assert min(analyzer.geod.inv(*end, *point)[2] for point in ring) < 20


def test_incomplete_notice_includes_actionable_error(monkeypatch):
    from pipeline_calculator.gui.tabs import summary_tab
    captured = []
    class Label:
        def __init__(self, parent, **kwargs):
            captured.append(kwargs['text'])
        def pack(self, **kwargs):
            pass
    monkeypatch.setattr(summary_tab, 'WrappedLabel', Label)
    summary_tab.add_status_notice(None, {'diagnostics': [{
        'level': 'error', 'message': 'Adjusted mileage and savings are unavailable.'
    }]})
    assert 'Analysis incomplete' in captured[0]
    assert 'savings are unavailable' in captured[0]


@pytest.mark.parametrize('length', [0, -1, float('nan'), float('inf')])
def test_invalid_segment_length_cannot_silently_disable_overlap(length):
    analyzer = PipelineAnalyzer(segment_length=length)
    with pytest.raises(ValueError, match='finite positive'):
        analyzer.find_parallel_segments(parallel_lines(analyzer))


def test_failed_geodesic_calculation_does_not_return_partial_mileage():
    class BrokenGeod:
        def inv(self, *args):
            return 0, 0, float('nan')
    analyzer = PipelineAnalyzer(geod=BrokenGeod())
    with pytest.raises(ValueError, match='Could not calculate length'):
        analyzer.calculate_pipeline_lengths([{'name': 'Bad', 'coordinates': [(0, 0), (1, 1)]}])
