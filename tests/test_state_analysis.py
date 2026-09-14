"""Integration evidence for state accounting, isolated overlap and job progress."""
from dataclasses import FrozenInstanceError
import json

import pytest
from pyproj import Geod
from shapely.geometry import box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext, ScopedExecutionContext
from pipeline_calculator.core.geography import BoundaryDataset
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.core.state_analysis import build_state_breakdown


GEOD = Geod(ellps='GRS80')


@pytest.fixture
def states():
    return BoundaryDataset({'AA': box(-1, -1, 0, 1), 'BB': box(0, -1, 1, 1)},
                           {'AA': 'West state', 'BB': 'East state'})


def pipeline(index, points, name='Repeated name'):
    return {'id': index, 'name': name, 'placemark_id': 'duplicate', 'objectid': 'duplicate',
            'coordinates': points, 'coordinate_paths': [points], 'source_kml': 'test.kml'}


def result_for(pipelines, states, context=None, analyzer=None):
    analyzer = analyzer or PipelineAnalyzer()
    combined = analyzer.analyze_features(pipelines)
    combined['geography'] = build_state_breakdown(analyzer, pipelines, combined,
                                                 boundaries=states, context=context)
    return combined


def parallel_crossings(length=300):
    paths = []
    for index, latitude in enumerate((0.0, 0.00009)):
        a = GEOD.fwd(0, latitude, 270, length/2)[:2]
        b = GEOD.fwd(0, latitude, 90, length/2)[:2]
        paths.append(pipeline(index, [a, b]))
    return paths


def test_clipped_state_overlap_is_independent_and_original_conserves(states):
    result = result_for(parallel_crossings(), states)
    geography = result['geography']
    assert geography['status'] == 'complete', geography['diagnostics']
    assert result['total_meters'] == pytest.approx(600, abs=.001)
    assert result['overlap_analysis']['savings_meters'] == pytest.approx(300, abs=5)
    assert geography['reconciliation']['passed']
    assert sum(s['total_meters'] for s in geography['states']) == pytest.approx(600, abs=.001)
    for state in geography['states']:
        assert state['total_meters'] == pytest.approx(300, abs=.001)
        assert state['interior_savings_meters'] == 0
        assert state['adjusted_total_meters'] == pytest.approx(300, abs=.001)
        assert len(state['pipelines']) == 2
        assert {p['source_id'] for p in state['pipelines']} == {0, 1}
        assert set(state['overlap_analysis']['pipeline_overlaps_by_id']) == {'0', '1'}
    json.dumps(result, allow_nan=False)  # No opaque GEOS/numpy serialization.


def test_shared_border_accounted_once_without_fictitious_state_geometry(states):
    source = pipeline(0, [(0, -.002), (0, .002)])
    result = result_for([source], states)
    geo = result['geography']
    assert geo['status'] == 'complete', geo['diagnostics']
    assert all(f['kind'] == 'shared' for f in geo['fragments'])
    assert sum(f['length_meters'] for f in geo['fragments']) == pytest.approx(result['total_meters'])
    for state in geo['states']:
        assert state['interior_meters'] == 0
        assert state['shared_allocation_meters'] == pytest.approx(result['total_meters']/2)
        assert state['total_meters'] == state['adjusted_total_meters']
        assert state['shared_overlap_status'] == 'not_calculated'
        assert state['overlap_analysis']['bundled_sections'] == []


def test_single_state_reuses_combined_analysis_without_modifying_it(states, monkeypatch):
    analyzer = PipelineAnalyzer()
    sources = [pipeline(0, [(-.005, 0), (-.004, 0)])]
    combined = analyzer.analyze_features(sources)
    snapshot = json.dumps(combined, sort_keys=True)
    monkeypatch.setattr(analyzer, 'analyze_features', lambda *a, **k: pytest.fail('unneeded second analysis'))
    geography = build_state_breakdown(analyzer, sources, combined, boundaries=states)
    assert geography['status'] == 'complete'
    assert len(geography['states']) == 1
    assert geography['states'][0]['total_meters'] == combined['total_meters']
    assert json.dumps(combined, sort_keys=True) == snapshot


def test_missing_boundaries_does_not_destroy_combined_result(tmp_path, monkeypatch):
    import pipeline_calculator.core.geography as geography
    path = tmp_path / 'pipeline.kml'
    path.write_text('<kml><Document><Placemark><LineString><coordinates>'
                    '-100,40 -99.999,40</coordinates></LineString></Placemark></Document></kml>')
    def fail():
        raise OSError('boundary resource unavailable')
    monkeypatch.setattr(geography, 'load_boundaries', fail)
    analyzer = PipelineAnalyzer()
    ordinary = analyzer.analyze_complete(path)
    assert 'geography' not in ordinary
    result = analyzer.analyze_complete(path, options=AnalysisOptions(True))
    assert result['total_meters'] == ordinary['total_meters']
    assert result['analysis_complete']
    assert result['geography']['status'] == 'unavailable'
    assert result['geography']['reconciliation']['unresolved_meters'] == result['total_meters']
    assert result['geography']['states'] == []


def test_state_mode_parses_once_and_returns_json_data(tmp_path, monkeypatch, states):
    from pipeline_calculator.core import analyzer as module
    import pipeline_calculator.core.geography as geography
    path = tmp_path / 'crossing.kml'
    path.write_text('<kml><Document><Placemark id="x"><LineString><coordinates>'
                    '-0.005,0 0.005,0</coordinates></LineString></Placemark></Document></kml>')
    calls = []
    parse = module.extract_features_from_file_with_diagnostics
    def counted(*args, **kwargs):
        calls.append(args[0])
        return parse(*args, **kwargs)
    monkeypatch.setattr(module, 'extract_features_from_file_with_diagnostics', counted)
    monkeypatch.setattr(geography, 'load_boundaries', lambda: states)
    result = PipelineAnalyzer().analyze_complete(path, options=AnalysisOptions(True))
    assert len(calls) == 1
    assert result['geography']['status'] == 'complete', result['geography']['diagnostics']
    assert len(result['geography']['states']) == 2
    json.loads(json.dumps(result, allow_nan=False))


def test_outside_coverage_stays_in_original_reconciliation(states):
    result = result_for([pipeline(0, [(.99, 0), (1.01, 0)])], states)
    geo = result['geography']
    rec = geo['reconciliation']
    assert rec['passed'] and rec['outside_meters'] > 0
    assert rec['attributed_state_meters'] + rec['outside_meters'] == pytest.approx(result['total_meters'])
    assert geo['status'] == 'complete'
    assert any(d['code'] == 'outside_state_coverage' for d in geo['diagnostics'])


def test_state_failure_retains_original_mileage_and_marks_adjusted_unavailable(states, monkeypatch):
    analyzer = PipelineAnalyzer()
    sources = parallel_crossings()
    combined = analyzer.analyze_features(sources)
    original = analyzer.analyze_features
    def fail_east(inputs, *args, **kwargs):
        if inputs and inputs[0]['coordinates'][-1][0] > 0:
            raise RuntimeError('injected state calculation failure')
        return original(inputs, *args, **kwargs)
    monkeypatch.setattr(analyzer, 'analyze_features', fail_east)
    geo = build_state_breakdown(analyzer, sources, combined, boundaries=states)
    east = next(s for s in geo['states'] if s['state_code'] == 'BB')
    assert east['total_meters'] == pytest.approx(300, abs=.001)
    assert east['adjusted_total_meters'] is None and east['interior_savings_meters'] is None
    assert not east['analysis_complete'] and geo['status'] == 'incomplete'
    assert geo['reconciliation']['passed']


def test_job_progress_maps_child_completions_and_cancellation(states):
    context = ExecutionContext()
    first = ScopedExecutionContext(context, 0, .4, 'Combined')
    first.report('Searching neighbors', 5, 10)
    first.finish()
    assert context.snapshot().fraction == .4
    result = result_for(parallel_crossings(), states, context=context)
    assert result['geography']['status'] == 'complete'
    assert context.snapshot().fraction == .99
    context.finish()
    assert context.snapshot().fraction == 1
    cancelled = ExecutionContext()
    cancelled.cancel()
    with pytest.raises(AnalysisCancelled):
        result_for(parallel_crossings(), states, context=cancelled)


def test_options_are_strict_and_immutable():
    with pytest.raises(TypeError):
        AnalysisOptions('false')
    with pytest.raises(FrozenInstanceError):
        AnalysisOptions().state_breakdown = True
