"""Display failures must never erase qualified numerical results."""
from copy import deepcopy

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.core import overlap


def parallel_inputs():
    return [
        {'id': 0, 'name': 'A', 'coordinates': [(-100, 40), (-99.995, 40)]},
        {'id': 1, 'name': 'B', 'coordinates': [(-100, 40.00004), (-99.995, 40.00004)]},
    ]


def test_optional_geometry_failure_preserves_accounting(monkeypatch):
    analyzer = PipelineAnalyzer(segment_length=10, min_parallel_length=100)
    expected = analyzer.analyze_features(parallel_inputs())

    def fail(*args, **kwargs):
        raise RuntimeError('Injected renderer failure')

    monkeypatch.setattr(overlap, '_build_section_corridor', fail)
    actual = analyzer.analyze_features(parallel_inputs())
    assert actual['analysis_complete'] is True
    for key in ('pipelines', 'total_meters', 'total_miles'):
        assert actual[key] == expected[key]
    old, new = expected['overlap_analysis'], actual['overlap_analysis']
    assert new is not None
    assert new['savings_meters'] > 0
    for key in old.keys() - {'bundled_sections'}:
        assert new[key] == old[key]
    assert len(old['bundled_sections']) == len(new['bundled_sections'])
    for before, after in zip(old['bundled_sections'], new['bundled_sections']):
        for key in ('pipeline_1_id', 'pipeline_2_id', 'source_path_indices',
                    'bundled_length_meters', 'bundled_length_miles',
                    'average_separation', 'segment_count'):
            assert before[key] == after[key]
        assert after['visualization_status'] == 'omitted'
        assert after['visualization_polygons'] == []
        assert after['diagnostics'][0]['context']['error_type'] == 'RuntimeError'


def test_optional_geometry_cancellation_aborts(monkeypatch):
    def cancel(*args, **kwargs):
        raise AnalysisCancelled('Injected cancellation')

    monkeypatch.setattr(overlap, '_build_section_corridor', cancel)
    with pytest.raises(AnalysisCancelled):
        PipelineAnalyzer().analyze_features(parallel_inputs())


def test_map_work_follows_savings_and_advances_progress():
    class RecordingContext(ExecutionContext):
        def __init__(self):
            super().__init__()
            self.events = []

        def report(self, stage, completed=0, total=None, **kwargs):
            super().report(stage, completed, total, **kwargs)
            self.events.append((stage, self.progress.update(stage, completed, total)))

    context = RecordingContext()
    PipelineAnalyzer().analyze_features(deepcopy(parallel_inputs()), context=context)
    names = [name for name, _ in context.events]
    assert names.index('Calculating savings') < names.index('Building corridor maps')
    fractions = [fraction for _, fraction in context.events]
    assert fractions == sorted(fractions)
    map_values = [f for name, f in context.events if name == 'Building corridor maps']
    assert min(map_values) < max(map_values) < 1


@pytest.mark.parametrize('canonical', [
    {'visualization_schema_version': 2},
    {'visualization_schema_version': True},
    {'visualization_schema_version': 1},
    {'visualization_status': 'omitted'},
    {'visualization_status': 'unknown'},
    {'visualization_polygons': []},
    {'clipped_polygons': []},
])
def test_canonical_failure_never_revives_legacy_rectangle(canonical):
    from pipeline_calculator.core.corridor_geometry import prepare_corridor
    section = dict(bbox={'min_lon': -100, 'max_lon': -99.99,
                         'min_lat': 40, 'max_lat': 40.01}, **canonical)
    prepared = prepare_corridor(section)
    assert prepared['visualization_status'] == 'omitted'
    assert prepared['visualization_polygons'] == []


def test_explicit_omission_overrides_stale_nonempty_polygons():
    from pipeline_calculator.core.corridor_geometry import prepare_corridor
    section = {'visualization_status': 'omitted', 'visualization_schema_version': 1,
               'visualization_polygons': [{'outer': [[0, 0], [1, 0], [1, 1], [0, 0]], 'holes': []}]}
    assert prepare_corridor(section)['visualization_status'] == 'omitted'


def test_local_state_mask_preserves_holes_and_multiple_islands(monkeypatch):
    from shapely.geometry import MultiPolygon, Polygon, box
    from shapely.ops import unary_union
    from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor
    from pipeline_calculator.core.geography import corridors
    import math

    # Thousands of irrelevant remote vertices must not be overlaid on this map.
    island = Polygon([(10 + math.cos(i*math.tau/5000), 10 + math.sin(i*math.tau/5000))
                      for i in range(5000)])
    local = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)],
                    [[(-.2, -.2), (.2, -.2), (.2, .2), (-.2, .2)]])
    boundary = MultiPolygon([local, island, box(1.1, -.5, 1.3, .5)])
    dataset = BoundaryDataset({'AA': boundary})
    source = box(-.8, -.8, 1.2, .8)
    prepared = {'visualization_schema_version': 1, 'visualization_status': 'ready',
                'visualization_polygons': [{'outer': list(source.exterior.coords), 'holes': []}]}
    result = clip_state_corridor(prepared, 'AA', dataset, PipelineAnalyzer().geod)
    assert result['visualization_status'] == 'ready', result['diagnostics']
    shapes = [Polygon(p['outer'], p['holes']) for p in result['clipped_polygons']]
    assert len(shapes) == 2 and sum(len(p.interiors) for p in shapes) == 1
    actual = unary_union(shapes)
    assert actual.symmetric_difference(source.intersection(boundary)).is_empty
    assert actual.difference(boundary).is_empty


def test_state_clip_budget_failure_preserves_numeric_section():
    from shapely.geometry import box
    from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
    from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor

    source = box(-.1, -.1, .1, .1)
    section = {'bundled_length_meters': 200, 'visualization_schema_version': 1,
               'visualization_status': 'ready', 'visualization_polygons': [
                   {'outer': list(source.exterior.coords), 'holes': []}]}
    result = clip_state_corridor(section, 'AA', BoundaryDataset({'AA': box(-1, -1, 0, 1)}),
                                 PipelineAnalyzer().geod, budget=CorridorGeometryBudget(max_work_vertices=0))
    assert result['bundled_length_meters'] == 200
    assert result['visualization_status'] == 'omitted' and result['clipped_polygons'] == []
    assert 'visualization_polygons' not in result
    assert section['visualization_status'] == 'ready'


def test_state_clip_honors_injected_native_limits():
    from shapely.geometry import box
    from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
    from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor
    source = box(-.1, -.1, .1, .1)
    section = {'visualization_schema_version': 1, 'visualization_status': 'ready',
               'visualization_polygons': [{'outer': list(source.exterior.coords), 'holes': []}]}
    result = clip_state_corridor(section, 'AA', BoundaryDataset({'AA': box(-1, -1, 0, 1)}),
        PipelineAnalyzer().geod, budget=CorridorGeometryBudget(max_native_vertices=0))
    assert result['visualization_status'] == 'omitted'
    with pytest.raises(ValueError, match='at least two operands'):
        CorridorGeometryBudget(max_native_operands=0)


def test_local_mask_does_not_change_original_boundary_edge_slopes():
    """Clipping to a query rectangle first used to omit39 of these120 maps."""
    import math
    import random
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor

    outer = [(-100+math.cos(i*math.tau/12000), 40+math.sin(i*math.tau/12000)) for i in range(12000)]
    hole = [(-100+.4*math.cos(i*math.tau/6000), 40+.4*math.sin(i*math.tau/6000)) for i in range(6000)]
    dataset = BoundaryDataset({'AA': Polygon(outer, [hole])})
    boundary = dataset.geometries['AA']
    rng = random.Random(1701)
    for radius in (1, .4):
        for _ in range(60):
            angle = rng.random()*math.tau
            x, y = -100+radius*math.cos(angle), 40+radius*math.sin(angle)
            source = box(x-.001, y-.001, x+.001, y+.001)
            section = {'visualization_schema_version': 1, 'visualization_status': 'ready',
                       'visualization_polygons': [{'outer': list(source.exterior.coords), 'holes': []}]}
            result = clip_state_corridor(section, 'AA', dataset, PipelineAnalyzer().geod)
            assert result['visualization_status'] == 'ready', result['diagnostics']
            actual = unary_union([Polygon(p['outer'], p['holes']) for p in result['clipped_polygons']])
            assert actual.difference(boundary).is_empty
            assert actual.symmetric_difference(source.intersection(boundary)).is_empty


def test_failed_normalization_clears_every_canonical_map_representation(monkeypatch):
    from pipeline_calculator.core import corridor_geometry
    from pipeline_calculator.export.corridor_metadata import validate_corridor_results

    def fail(*args, **kwargs):
        raise RuntimeError('Injected normalization failure')

    monkeypatch.setattr(corridor_geometry, 'normalize_polygons', fail)
    geometry = [{'outer': [[0, 0], [1, 0], [1, 1], [0, 0]], 'holes': []}]
    result = corridor_geometry.prepare_corridor({
        'visualization_schema_version': 1, 'visualization_status': 'ready',
        'visualization_polygons': geometry, 'clipped_polygons': geometry,
        'corridor_polygon': geometry[0]['outer'],
        'visualization_metadata': {'part_count': 1, 'hole_count': 0, 'vertex_count': 4},
    })
    assert result['visualization_status'] == 'omitted'
    assert result['visualization_polygons'] == result['clipped_polygons'] == result['corridor_polygon'] == []
    assert result['visualization_metadata']['part_count'] == 0
    validate_corridor_results({'overlap_analysis': {'bundled_sections': [result]}})


def test_state_clip_part_and_ring_caps_are_checked_before_publication():
    from shapely.geometry import MultiPolygon, box
    from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
    from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor
    source = box(-1, -1, 2, 1)
    islands = MultiPolygon([box(-.9, -.9, -.1, .9), box(.1, -.9, .9, .9)])
    section = {'visualization_schema_version': 1, 'visualization_status': 'ready',
               'visualization_polygons': [{'outer': list(source.exterior.coords), 'holes': []}]}
    for kwargs in ({'max_parts': 1}, {'max_rings': 1}):
        result = clip_state_corridor(section, 'AA', BoundaryDataset({'AA': islands}),
            PipelineAnalyzer().geod, budget=CorridorGeometryBudget(**kwargs))
        assert result['visualization_status'] == 'omitted'
        assert not result['clipped_polygons']


def test_state_clip_native_preflight_counts_ring_closure_coordinates():
    from shapely.geometry import Polygon, box
    from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
    from pipeline_calculator.core.geography.corridors import _bounded_intersection
    triangle = Polygon([(0, 0), (.1, 0), (0, .1)])
    with pytest.raises(ValueError, match='input exceeds'):
        _bounded_intersection(triangle, box(-1, -1, 1, 1), None,
                              CorridorGeometryBudget(max_native_vertices=8))


def test_state_clipping_transfers_owned_retained_output_without_double_charge():
    from shapely.geometry import box
    from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
    from pipeline_calculator.core.geography import BoundaryDataset, clip_state_corridor
    budget = CorridorGeometryBudget()
    analyzer = PipelineAnalyzer()
    result = analyzer.analyze_features(parallel_inputs(), corridor_budget=budget)
    section = result['overlap_analysis']['bundled_sections'][0]
    assert section['visualization_status'] == 'ready'
    count = budget.retained_vertices
    budget.max_retained_vertices = count
    dataset = BoundaryDataset({'AA': box(-101, 39, -99, 41)})
    clipped = clip_state_corridor(section, 'AA', dataset, analyzer.geod,
                                 budget=budget, replace_retained=True)
    assert clipped['visualization_status'] == 'ready', clipped['diagnostics']
    assert budget.retained_vertices == count
    # A separately retained Combined map still occupies its own output budget.
    extra = clip_state_corridor(section, 'AA', dataset, analyzer.geod, budget=budget)
    assert extra['visualization_status'] == 'omitted'
    assert budget.retained_vertices == count


def test_single_state_reuse_keeps_generating_scope_when_zero_path_is_removed():
    from shapely.geometry import box
    from pipeline_calculator.core.geography import BoundaryDataset
    from pipeline_calculator.core.state_analysis import build_state_breakdown
    analyzer = PipelineAnalyzer()
    sources = parallel_inputs()
    for pipeline in sources:
        path = pipeline['coordinates']
        pipeline['coordinate_paths'] = [[path[0], path[0]], path]
    combined = analyzer.analyze_features(sources)
    geography = build_state_breakdown(analyzer, sources, combined,
        boundaries=BoundaryDataset({'AA': box(-101, 39, -99, 41)}))
    state = geography['states'][0]
    section = state['overlap_analysis']['bundled_sections'][0]
    assert section['visualization_status'] == 'ready'
    assert section['visualization_metadata']['state_code'] == 'AA'
    assert {run['scope'] for run in section['visualization_metadata']['source_runs']} == {'Combined'}
    assert {run['scope_path_index'] for run in section['visualization_metadata']['source_runs']} == {1}
    assert {f['path_index'] for f in geography['fragments']} == {1}
