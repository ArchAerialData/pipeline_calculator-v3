"""Geometric extent regressions, independent of original mileage calculations."""
import math

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.export.geometry_validation import prepare_geometry, validated_ring
from scripts.validation.common import geographic
from scripts.validation.common import gallery_specs
from scripts.validation.geometry import inside, local_points


def bent_pair():
    path = [(0, 0), (0, 150), (150, 150)]
    return [{'name': str(i), 'coordinates': [geographic((x + 2*i, y)) for x, y in path]}
            for i in range(2)]


def test_fallback_rectangle_contains_all_qualified_midpoints():
    analyzer = PipelineAnalyzer(min_parallel_length=10)
    pipes = bent_pair()
    matches = analyzer.find_parallel_segments(pipes)
    result = analyzer.calculate_overlap_results(pipes, matches)
    for section in result['bundled_sections']:
        origin = (section['center_lon'], section['center_lat'])
        ring = local_points(section['oriented_polygon'], origin)
        # Both source lines qualify throughout this right angle.
        for pipe in pipes:
            for point in local_points([s['midpoint'] for s in pipe['segments']], origin):
                assert inside(point, ring), point


def test_straight_outline_covers_sampled_segment_ends():
    analyzer = PipelineAnalyzer(min_parallel_length=10)
    pipes = [{'name': str(i), 'coordinates': [geographic((i*2, y)) for y in (0, 300)]}
             for i in range(2)]
    result = analyzer.calculate_overlap_results(pipes, analyzer.find_parallel_segments(pipes))
    section = result['bundled_sections'][0]
    origin = (section['center_lon'], section['center_lat'])
    ring = local_points(prepare_geometry(section)[0], origin)
    for pipe in pipes:
        assert all(inside(point, ring) for point in local_points(pipe['coordinates'], origin))


def test_unused_malformed_bbox_does_not_break_valid_preferred_outline():
    data = {'corridor_polygon': [(0, 0), (.001, 0), (.001, .001), (0, .001)],
            'bbox': {'min_lon': 0}}
    assert prepare_geometry(data)[2] == 'sampled_curve'


def test_large_self_crossing_ring_is_not_exported_unchecked():
    # Unequal lobes preserve nonzero area so area-only checks do not catch this.
    vertices = [(0, 0), (.003, .002), (0, .002), (.002, 0), (0, 0)]
    ring = [(a[0]+(b[0]-a[0])*i/150, a[1]+(b[1]-a[1])*i/150)
            for a, b in zip(vertices, vertices[1:]) for i in range(150)]
    with pytest.raises(ValueError, match='crosses|budget'):
        validated_ring(ring)


def test_large_simple_ring_can_still_follow_a_curve():
    ring = [(.001*math.cos(i*2*math.pi/600), .001*math.sin(i*2*math.pi/600))
            for i in range(600)]
    assert validated_ring(ring)[1] is True


def test_excessive_topology_work_uses_disclosed_rectangle(monkeypatch):
    from pipeline_calculator.export import geometry_validation
    monkeypatch.setattr(geometry_validation, 'TOPOLOGY_PAIR_BUDGET', 10)
    ring = [(.001*math.cos(i*2*math.pi/600), .001*math.sin(i*2*math.pi/600))
            for i in range(600)]
    data = {'corridor_polygon': ring,
            'oriented_polygon': [(-.002, -.002), (.002, -.002), (.002, .002), (-.002, .002)]}
    _, _, kind, reason = prepare_geometry(data)
    assert kind == 'oriented_rectangle' and 'work budget' in reason


@pytest.mark.parametrize('spec', gallery_specs(), ids=lambda spec: spec['id'])
def test_group_decisions_do_not_depend_on_input_order(spec):
    analyzer = PipelineAnalyzer(min_parallel_length=spec['minimum'])
    pipes = []
    for index, parts in enumerate(spec['paths']):
        paths = [[geographic(p, spec.get('origin', (-100, 40)), spec.get('bearing', 0))
                  for p in part] for part in parts]
        pipes.append({'name': str(index), 'coordinate_paths': paths})
    def summarize(items):
        _, source_meters, _ = analyzer.calculate_pipeline_lengths(items)
        result = analyzer.calculate_overlap_results(items, analyzer.find_parallel_segments(items))
        assert 0 <= result['savings_meters'] <= source_meters
        sections = sorted(s['bundled_length_meters'] for s in result['bundled_sections'])
        return source_meters, result['savings_meters'], sections
    before = summarize(pipes)
    after = summarize(list(reversed(pipes)))
    assert after[0] == pytest.approx(before[0], abs=1e-6)
    assert after[1] == pytest.approx(before[1], abs=1e-6)
    assert after[2] == pytest.approx(before[2], abs=1e-6)
