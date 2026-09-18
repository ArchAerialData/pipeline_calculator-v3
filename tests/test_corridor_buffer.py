"""Independent capsule, distance, topology, projection and budget checks."""
from dataclasses import replace
import json
import math

import pytest
from pyproj import Geod
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from pipeline_calculator.core.corridor_buffer import (
    CorridorDisplayOptions, CorridorGeometryBudget, build_buffered_corridor,
)
from pipeline_calculator.core.corridor_coverage import QualifiedPathRun
from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext


GEOD = Geod(ellps='GRS80')


def geographic(points, origin=(-100, 40)):
    return tuple(GEOD.fwd(*origin, math.degrees(math.atan2(x, y)), math.hypot(x, y))[:2]
                 for x, y in points)


def run(points, index=0):
    length = sum(GEOD.inv(*a, *b)[2] for a, b in zip(points, points[1:]))
    return QualifiedPathRun('Combined', index, index, 0, 0, 0, 0., length, tuple(points))


def projected(result, origin=(-100, 40)):
    def ring(points):
        result = []
        for point in points:
            az, _, distance = GEOD.inv(*origin, *point)
            result.append((distance * math.sin(math.radians(az)), distance * math.cos(math.radians(az))))
        return result
    return unary_union([Polygon(ring(p['outer']), [ring(h) for h in p['holes']])
                        for p in result['visualization_polygons']])


def ready(runs, **kwargs):
    result = build_buffered_corridor(runs, geod=GEOD, **kwargs)
    assert result['visualization_status'] == 'ready', result['diagnostics']
    assert result['visualization_metadata']['padding_m'] == 5
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize('origin', [(-100, 40), (-155, 20), (-150, 68), (30, 0), (179.999, 55), (-179.999, 55)])
def test_geographic_capsule_matches_analytic_neighborhood(origin):
    points = geographic([(0, 0), (0, 300)], origin)
    result = ready([run(points)])
    shape = projected(result, origin)
    assert shape.is_valid
    # The analytic area is independent of the production buffering machinery.
    assert shape.area == pytest.approx(2 * 5 * 300 + math.pi * 25, abs=.2)
    assert shape.bounds == pytest.approx((-5, -5, 5, 305), abs=.008)
    assert shape.covers(LineString([(0, 0), (0, 300)]))
    for point in shape.exterior.coords:
        assert LineString([(0, 0), (0, 300)]).distance(Point(point)) == pytest.approx(5, abs=.01)


@pytest.mark.parametrize('points', [
    [(0, 0), (0, 300), (300, 300)],
    [(0, 0), (0, 300), (300, 300), (300, 0)],
    [(0, 0), (0, 150), (150, 150), (150, 300), (300, 300)],
    [(0, 0), (0, 300), (0, 0), (300, 0)],
    [(0, 0), (150, 150), (0, 150), (150, 0)],
])
def test_bends_and_backtracking_preserve_support_without_hull(points):
    result = ready([run(geographic(points))])
    shape = projected(result)
    source = LineString(points)
    # Separate denser resolution and inner/outer radius: test both under-buffering
    # and excess fill rather than merely looking for some valid polygon.
    assert source.buffer(4.95, quad_segs=128).difference(shape).is_empty
    assert shape.difference(source.buffer(5.05, quad_segs=128)).is_empty
    assert shape.covers(source)


def test_closed_loop_keeps_hole_and_does_not_publish_partial_legacy_ring():
    points = geographic([(0, 0), (300, 0), (300, 300), (0, 300), (0, 0)])
    result = ready([run(points)])
    assert result['visualization_metadata']['hole_count'] == 1
    assert not result['corridor_polygon']
    assert not projected(result).covers(Point(150, 150))


@pytest.mark.parametrize('separation,parts', [(9.9, 1), (10.1, 2), (15, 2)])
def test_fixed_radius_does_not_widen_to_connect_paths(separation, parts):
    result = ready([run(geographic([(0, 0), (0, 300)])),
                    run(geographic([(separation, 0), (separation, 300)]), 1)])
    assert result['visualization_metadata']['part_count'] == parts
    if parts > 1:
        assert not result['corridor_polygon']
        assert not projected(result).covers(Point(separation / 2, 150))


def test_duplicate_reversed_support_does_not_double_area():
    points = geographic([(0, 0), (0, 300), (300, 300)])
    single = projected(ready([run(points)]))
    repeated = projected(ready([run(points), run(points[::-1], 1)]))
    assert repeated.symmetric_difference(single).area < .01


def test_dense_redundant_geodesic_vertices_do_not_change_the_neighborhood():
    sparse = geographic([(0, 0), (0, 1000)])
    dense = geographic([(0, station) for station in range(0, 1001, 5)])
    a, b = projected(ready([run(sparse)])), projected(ready([run(dense)]))
    assert a.hausdorff_distance(b) < .01
    assert a.symmetric_difference(b).area < .03


def test_duplicate_source_vertices_and_span_endpoints_do_not_create_zero_length_chunks():
    points = geographic([(0, 0), (0, 300), (300, 300)])
    duplicate = (points[0], points[0], points[1], points[1], points[2], points[2])
    result = ready([run(duplicate)])
    assert projected(result).symmetric_difference(projected(ready([run(points)]))).area < .01


def test_oblique_sparse_geodesic_edges_have_independent_distance_oracle():
    from scipy.optimize import minimize_scalar
    start = (179.98, 68)
    bearing, length = 67., 32_000.
    end = GEOD.fwd(*start, bearing, length)[:2]
    result = ready([run((start, end))])
    # The reference minimizes actual ellipsoid distance along the ORIGINAL
    # geodesic, independently of AEQD, geographic chord transforms and buffering.
    for polygon in result['visualization_polygons']:
        ring = polygon['outer']
        for index in range(0, len(ring) - 1, max(1, len(ring) // 15)):
            a, b = ring[index:index + 2]
            # Dateline split edges run inside the buffer and are not outer
            # distance contours; all must still remain within radius + error.
            point = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            def distance(station):
                return abs(GEOD.inv(*point, *GEOD.fwd(*start, bearing, station)[:2])[2])
            optimum = minimize_scalar(distance, bounds=(0, length), method='bounded', options={'xatol': 1e-6})
            measured = min(distance(0), distance(length), optimum.fun)
            assert measured <= 5.05
            if abs(abs(point[0]) - 180) > 1e-8:
                assert measured >= 4.95


def test_sparse_long_geodesic_chart_seams_match_alternative_partition():
    points = geographic([(0, 0), (0, 100_000)])
    first = ready([run(points)])
    alternative = ready([run(points)], options=replace(CorridorDisplayOptions(), max_chunk_m=15_000))
    a, b = projected(first), projected(alternative)
    assert first['visualization_metadata']['chart_count'] == 5
    assert a.hausdorff_distance(b) < .02
    assert a.covers(LineString([(0, 0), (0, 100_000)]))


def test_dateline_polygons_have_canonical_longitude_and_no_world_spanning_edges():
    result = ready([run(geographic([(-300, 0), (300, 0)], (180, 55)))])
    assert result['visualization_metadata']['part_count'] == 2
    for polygon in result['visualization_polygons']:
        for ring in [polygon['outer'], *polygon['holes']]:
            assert all(-180 <= p[0] <= 180 for p in ring)
            assert all(abs(a[0] - b[0]) < 1 for a, b in zip(ring, ring[1:]))


@pytest.mark.parametrize('setting', ['max_work_vertices', 'max_charts', 'max_retained_vertices', 'max_section_points',
                                    'max_section_charts', 'max_native_vertices'])
def test_budget_exhaustion_omits_whole_section_with_no_fallback(setting):
    budget = CorridorGeometryBudget(**{setting: 0})
    result = build_buffered_corridor([run(geographic([(0, 0), (0, 300)]))], geod=GEOD, budget=budget)
    assert result['visualization_status'] == 'omitted'
    assert result['visualization_polygons'] == result['corridor_polygon'] == []
    assert result['diagnostics'][0]['code'] == 'corridor_buffer_limit'
    assert budget.retained_vertices == 0


def test_shared_job_budget_is_not_reset_between_sections():
    budget = CorridorGeometryBudget(max_charts=1)
    data = [run(geographic([(0, 0), (0, 300)]))]
    assert ready(data, budget=budget)
    retained = budget.retained_vertices
    second = build_buffered_corridor(data, geod=GEOD, budget=budget)
    assert second['visualization_status'] == 'omitted'
    assert budget.charts == 1 and budget.retained_vertices == retained


def test_retained_output_replacement_releases_only_owned_vertices():
    budget = CorridorGeometryBudget()
    budget.retain(100)
    budget.release_retained(40)
    assert budget.retained_vertices == 60
    budget.release_retained(0)
    with pytest.raises(ValueError, match='more corridor vertices'):
        budget.release_retained(61)
    assert budget.retained_vertices == 60
    budget.release_retained(60)
    assert budget.retained_vertices == 0


@pytest.mark.parametrize('invalid', [-1, 1.5, float('nan'), '1', None, True])
def test_release_rejects_invalid_counts(invalid):
    with pytest.raises(ValueError):
        CorridorGeometryBudget().release_retained(invalid)


def test_static_boundary_predicate_budget_is_separate_and_preflighted():
    budget = CorridorGeometryBudget(max_work_vertices=1, max_boundary_vertices=10_000, max_boundary_queries=2)
    budget.boundary_query(10_000)
    budget.boundary_query(8_000)
    assert budget.boundary_queries == 2 and budget.work_vertices == 0
    with pytest.raises(ValueError, match='predicate count'):
        budget.boundary_query(5)
    assert budget.boundary_queries == 2
    with pytest.raises(ValueError, match='predicate vertex'):
        budget.boundary_query(10_001)
    assert budget.boundary_queries == 2


@pytest.mark.parametrize('invalid', [0, -1, 1.5, float('nan'), '1', None, True])
def test_boundary_predicate_requires_positive_integer_vertex_count(invalid):
    budget = CorridorGeometryBudget()
    with pytest.raises(ValueError):
        budget.boundary_query(invalid)
    assert budget.boundary_queries == 0


def test_cancellation_propagates_even_during_projection():
    class Cancel(ExecutionContext):
        def __init__(self):
            super().__init__()
            self.checks = 0
        def check(self):
            self.checks += 1
            if self.checks == 70:
                self.cancel()
            super().check()
    with pytest.raises(AnalysisCancelled):
        build_buffered_corridor([run(geographic([(0, 0), (0, 300)]))], geod=GEOD, context=Cancel())


def test_native_failure_is_a_visible_omission_and_never_mutates_source(monkeypatch):
    from pipeline_calculator.core import corridor_buffer
    data = run(geographic([(0, 0), (0, 300)]))
    original = data.coordinates
    def fail(*args, **kwargs):
        raise RuntimeError('injected buffer failure')
    monkeypatch.setattr(corridor_buffer, '_buffer', fail)
    result = build_buffered_corridor([data], geod=GEOD)
    assert result['visualization_status'] == 'omitted'
    assert 'injected buffer failure' in result['diagnostics'][0]['context']['error']
    assert data.coordinates == original


def test_projection_domain_failure_is_explicit():
    result = build_buffered_corridor([run(((0, 90), (0, 89.99)))], geod=GEOD)
    assert result['visualization_status'] == 'omitted'


@pytest.mark.parametrize('mutation', ['thin', 'hull'])
def test_two_sided_runtime_oracle_rejects_plausible_but_incorrect_footprints(monkeypatch, mutation):
    from pipeline_calculator.core import corridor_buffer
    original = corridor_buffer._buffer
    calls = 0
    def broken(line, radius, quadrants, work):
        nonlocal calls
        calls += 1
        if calls == 1:
            result = original(line, .1 if mutation == 'thin' else radius, quadrants, work)
            return result.convex_hull if mutation == 'hull' else result
        return original(line, radius, quadrants, work)
    monkeypatch.setattr(corridor_buffer, '_buffer', broken)
    points = geographic([(0, 0), (0, 300), (300, 300), (300, 0)])
    result = build_buffered_corridor([run(points)], geod=GEOD)
    assert result['visualization_status'] == 'omitted'
    assert result['diagnostics'][0]['code'] == 'corridor_coverage_failed'


def test_self_overlap_complexity_is_rejected_before_native_buffer(monkeypatch):
    from pipeline_calculator.core.corridor_buffer import _buffer, _SectionBudget
    points = [(0, 0), (100, 100), (0, 100), (100, 0)] * 30
    called = False
    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('Native buffer must not receive this unbounded expansion')
    monkeypatch.setattr(LineString, 'buffer', forbidden)
    with pytest.raises(ValueError, match='expansion'):
        _buffer(LineString(points), 5, 18, _SectionBudget(CorridorGeometryBudget(), None))
    assert not called


def test_output_option_values_are_checked_without_touching_mileage():
    with pytest.raises(ValueError):
        replace(CorridorDisplayOptions(), padding_m=-1)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), None, '5', True, 10])
def test_invalid_padding_cannot_escape_into_json_metadata(invalid):
    with pytest.raises(ValueError):
        replace(CorridorDisplayOptions(), padding_m=invalid)
