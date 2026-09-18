"""Later corridor scopes retain usable capacity without removing work limits."""
from copy import deepcopy

import pytest
from shapely.geometry import box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor
from pipeline_calculator.core.state_analysis import build_state_breakdown


def parallel_inputs():
    return [
        {'id': 0, 'name': 'West to east A', 'coordinates': [(-100, 40), (-99.995, 40)]},
        {'id': 1, 'name': 'West to east B', 'coordinates': [(-100, 40.00004), (-99.995, 40.00004)]},
    ]


def test_combined_and_state_maps_can_finish_after_old_job_allowance():
    # Account for earlier sections without making CI construct millions of
    # coordinates. Everything after this checkpoint uses the real map builders.
    budget = CorridorGeometryBudget()
    budget.charge(2_000_000)
    analyzer = PipelineAnalyzer(segment_length=10, min_parallel_length=100)
    inputs = parallel_inputs()
    combined = analyzer.analyze_features(inputs, corridor_budget=budget)
    sections = combined['overlap_analysis']['bundled_sections']
    assert sections and all(s['visualization_status'] == 'ready' for s in sections)
    combined_work = budget.work_vertices
    combined_retained = budget.retained_vertices
    assert combined_work > 2_000_000

    boundaries = BoundaryDataset({'AA': box(-101, 39, -99.9975, 41),
                                  'BB': box(-99.9975, 39, -99, 41)})
    geography = build_state_breakdown(analyzer, inputs, combined, boundaries=boundaries,
                                      corridor_budget=budget)
    assert geography['status'] == 'complete'
    assert geography['reconciliation']['passed']
    assert {state['state_code'] for state in geography['states']} == {'AA', 'BB'}
    for state in geography['states']:
        sections = state['overlap_analysis']['bundled_sections']
        assert state['analysis_complete'] and sections
        assert all(s['visualization_status'] == 'ready' and s['clipped_polygons'] for s in sections)
    assert combined_work < budget.work_vertices < 10_000_000
    assert budget.retained_vertices > combined_retained


def test_new_job_allowance_still_omits_maps_without_changing_mileage():
    analyzer = PipelineAnalyzer(segment_length=10, min_parallel_length=100)
    budget = CorridorGeometryBudget()
    expected = analyzer.analyze_features(parallel_inputs(), corridor_budget=budget)
    snapshot = deepcopy(expected)
    retained = budget.retained_vertices
    assert retained > 0
    # Leave enough room to begin work, but not to finish its buffer. Prior
    # successful sections and numerical results must survive this exhaustion.
    budget.charge(10_000_000 - budget.work_vertices - 100)
    actual = analyzer.analyze_features(parallel_inputs(), corridor_budget=budget)
    assert actual['analysis_complete'] is True
    for key in ('pipelines', 'total_meters', 'total_miles'):
        assert actual[key] == expected[key]
    old, new = expected['overlap_analysis'], actual['overlap_analysis']
    for key in old.keys() - {'bundled_sections'}:
        assert new[key] == old[key]
    assert len(new['bundled_sections']) == len(old['bundled_sections']) > 0
    for before, after in zip(old['bundled_sections'], new['bundled_sections']):
        for key in ('pipeline_1_id', 'pipeline_2_id', 'source_path_indices',
                    'bundled_length_meters', 'bundled_length_miles',
                    'average_separation', 'segment_count'):
            assert after[key] == before[key]
        assert after['visualization_status'] == 'omitted'
        assert after['visualization_polygons'] == after['corridor_polygon'] == []
        assert any(d['code'] == 'corridor_buffer_limit' for d in after['diagnostics'])
    assert 9_999_900 <= budget.work_vertices <= 10_000_000
    assert budget.retained_vertices == retained
    assert expected == snapshot
    with pytest.raises(ValueError, match='job construction budget exceeded'):
        budget.preflight(10_000_001 - budget.work_vertices)


@pytest.mark.parametrize('limit', ['max_work_vertices', 'max_native_vertices',
                                 'max_boundary_queries', 'max_retained_vertices',
                                 'max_parts', 'max_rings'])
def test_state_clipping_limits_have_a_structured_reason_and_keep_mileage(limit):
    polygon = box(-.1, -.1, .1, .1)
    section = {'bundled_length_meters': 200, 'visualization_schema_version': 1,
               'visualization_status': 'ready', 'visualization_polygons': [
                   {'outer': list(polygon.exterior.coords), 'holes': []}]}
    result = clip_state_corridor(section, 'AA', BoundaryDataset({'AA': box(-1, -1, 1, 1)}),
                                 PipelineAnalyzer().geod, budget=CorridorGeometryBudget(**{limit: 0}))
    assert result['visualization_status'] == 'omitted'
    assert result['clipped_polygons'] == []
    assert result['bundled_length_meters'] == section['bundled_length_meters']
    diagnostic = next(d for d in result['diagnostics'] if d['code'] == 'state_corridor_omitted')
    assert diagnostic['context']['reason_code'] == 'corridor_buffer_limit'
    assert diagnostic['context']['state_code'] == 'AA'


def test_state_clipping_invalid_geometry_is_not_mislabeled_as_a_work_limit():
    section = {'visualization_schema_version': 1, 'visualization_status': 'ready',
               'visualization_polygons': [{'outer': [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]],
                                           'holes': []}]}
    result = clip_state_corridor(section, 'AA', BoundaryDataset({'AA': box(-2, -2, 2, 2)}),
                                 PipelineAnalyzer().geod)
    assert result['visualization_status'] == 'omitted'
    diagnostic = next(d for d in result['diagnostics'] if d['code'] == 'state_corridor_omitted')
    assert diagnostic['context']['reason_code'] == 'state_corridor_geometry_invalid'
