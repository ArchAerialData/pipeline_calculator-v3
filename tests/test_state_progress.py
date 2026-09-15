"""Public callback progress and cancellation across the entire geography job."""
import pytest
from shapely.geometry import box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.core.geography import BoundaryDataset
from pipeline_calculator.core.options import AnalysisOptions
import pipeline_calculator.core.geography as geography
import pipeline_calculator.core.state_analysis as state_analysis


@pytest.fixture
def crossing(tmp_path, monkeypatch):
    path = tmp_path / 'crossing.kml'
    path.write_text('<kml>' + ''.join(
        f'<Placemark><name>{i}</name><LineString><coordinates>'
        f'-0.003,{i * .00009} 0.003,{i * .00009}'
        '</coordinates></LineString></Placemark>' for i in range(2)) + '</kml>')
    boundaries = BoundaryDataset(
        {'AA': box(-1, -1, 0, 1), 'BB': box(0, -1, 1, 1)},
        {'AA': 'A State', 'BB': 'B State'})
    monkeypatch.setattr(geography, 'load_boundaries', lambda: boundaries)
    return path


@pytest.mark.parametrize('with_context', [False, True])
def test_public_callback_tracks_geography_until_snapshot_is_complete(crossing, monkeypatch, with_context):
    events, at_partition = [], []
    built = []
    context = ExecutionContext() if with_context else None
    real_partition = geography.partition_pipelines
    real_build = state_analysis.build_state_breakdown

    def partition(*args, **kwargs):
        at_partition.extend(events)
        return real_partition(*args, **kwargs)

    def build(*args, **kwargs):
        result = real_build(*args, **kwargs)
        built.append(result)
        return result

    def callback(fraction):
        assert isinstance(fraction, float)
        if fraction == 1:
            assert built and built[0]['reconciliation']['passed']
        if context is not None:
            assert fraction == context.snapshot().fraction
        events.append(fraction)

    monkeypatch.setattr(geography, 'partition_pipelines', partition)
    monkeypatch.setattr(state_analysis, 'build_state_breakdown', build)
    result = PipelineAnalyzer().analyze_complete(
        crossing, callback, context=context, options=AnalysisOptions(True))
    assert result['geography']['status'] == 'complete'
    assert at_partition and max(at_partition) < .66
    assert events == sorted(events)
    assert all(0 <= value < 1 for value in events[:-1])
    assert events[-1] == 1
    assert any(.66 < value < .98 for value in events)
    assert .99 in events


@pytest.mark.parametrize('stage', [
    'Loading state boundaries', 'Splitting geometry', 'Analyzing A State',
    'Analyzing B State', 'Finalizing state results',
])
@pytest.mark.parametrize('with_callback', [False, True])
def test_cancel_at_each_geography_stage_publishes_no_completion_and_allows_retry(
        crossing, stage, with_callback):
    class CancelAtStage(ExecutionContext):
        def report(self, current, *args, **kwargs):
            if current.startswith(stage):
                self.cancel()
            return super().report(current, *args, **kwargs)

    analyzer = PipelineAnalyzer()
    events = []
    with pytest.raises(AnalysisCancelled):
        analyzer.analyze_complete(
            crossing, events.append if with_callback else None,
            context=CancelAtStage(), options=AnalysisOptions(True))
    assert 1.0 not in events
    retry = analyzer.analyze_complete(crossing, events.append, options=AnalysisOptions(True))
    assert retry['analysis_complete'] and retry['geography']['status'] == 'complete'
    assert events[-1] == 1


@pytest.mark.parametrize('failure', ['boundaries', 'state'])
def test_callback_completes_explicit_unavailable_or_incomplete_result(crossing, monkeypatch, failure):
    analyzer = PipelineAnalyzer()
    if failure == 'boundaries':
        def missing():
            raise OSError('Resource unavailable')
        monkeypatch.setattr(geography, 'load_boundaries', missing)
    else:
        real_analyze = analyzer.analyze_features
        def fail_state(*args, **kwargs):
            if getattr(kwargs.get('context'), 'label', '').startswith('Analyzing B State'):
                raise ValueError('Injected state failure')
            return real_analyze(*args, **kwargs)
        monkeypatch.setattr(analyzer, 'analyze_features', fail_state)
    events = []
    result = analyzer.analyze_complete(crossing, events.append, options=AnalysisOptions(True))
    assert result['analysis_complete']
    assert result['geography']['status'] == ('unavailable' if failure == 'boundaries' else 'incomplete')
    assert events[-1] == 1 and 1 not in events[:-1]
    assert events == sorted(events)


def test_mode_off_retains_legacy_callback_contract(crossing):
    events = []
    result = PipelineAnalyzer().analyze_complete(crossing, events.append)
    assert 'geography' not in result
    assert events == [.5, .625, 1.0]
