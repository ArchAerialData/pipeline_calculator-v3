"""Adversarial checks for the display-only numerical and map evidence tools."""
from copy import deepcopy
import json

import pytest
from shapely.geometry import LineString, Polygon

from scripts.validation.corridor_numeric import compare, differences, generate, numerical
from scripts.validation.corridor_audit import (
    GEOD, consecutive_ranges, document_polygons, inspect_document, path_span,
)


def observation():
    row = {'id': 'ordinary', 'sha256': 'frozen-input', 'parameters': {}, 'state_breakdown': False,
           'result': {'total_meters': 400., 'analysis_complete': True, 'diagnostics': [],
                      'overlap_analysis': {'savings_meters': 200., 'bundled_sections': [
                          {'pipeline_1_id': 1, 'pipeline_2_id': 2, 'source_path_indices': [0, 0],
                           'segment_count': 40, 'bundled_length_meters': 200., 'average_separation': 2.0,
                           'corridor_polygon': [[0, 0], [1, 1]]}]}},
           'qualified_membership': [{'sections': [{'sample_membership': [[[0, 0], [0, 1]], [[0, 0], [0, 1]]]}]}]}
    return {'status': 'complete', 'cases': [row], 'environment': {'python': 'fixed', 'packages': {}, 'boundary_sha256': 'fixed'}}


def test_numeric_comparator_excludes_only_enumerated_display_fields_and_codes():
    before = observation(); after = deepcopy(before)
    section = after['cases'][0]['result']['overlap_analysis']['bundled_sections'][0]
    section.update(visualization_schema_version=1, visualization_metadata={'padding_m': 5},
                   visualization_polygons=[], diagnostics=[{'code': 'corridor_visualization_omitted', 'level': 'warning'}])
    after['cases'][0]['result']['diagnostics'].append({'code': 'corridor_buffer_limit', 'level': 'warning'})
    assert compare(before, after)['passed']
    after['cases'][0]['result']['diagnostics'].append({'code': 'state_geometry_unresolved', 'level': 'error'})
    assert not compare(before, after)['passed']
    section['diagnostics'].append({'code': 'corridor_unknown_future_failure', 'level': 'warning'})
    assert 'corridor_unknown_future_failure' in json.dumps(numerical(section))


@pytest.mark.parametrize('field,value', [('segment_count', 39), ('pipeline_2_id', 3),
                                       ('source_path_indices', [0, 1]), ('average_separation', 2.000000000001)])
def test_numeric_comparator_rejects_section_changes_even_when_grand_totals_match(field, value):
    before = observation(); after = deepcopy(before)
    after['cases'][0]['result']['overlap_analysis']['bundled_sections'][0][field] = value
    assert not compare(before, after)['passed']


def test_numeric_comparator_checks_qualified_membership_and_unrelated_failures():
    before = observation(); after = deepcopy(before)
    after['cases'][0]['qualified_membership'][0]['sections'][0]['sample_membership'][0][0] = [0, 9]
    assert not compare(before, after)['passed']
    after = deepcopy(before)
    after['cases'][0]['result']['analysis_complete'] = False
    assert not compare(before, after)['passed']
    after['status'] = 'running'
    with pytest.raises(ValueError, match='completed'):
        compare(before, after)


def test_generator_is_deterministic_and_failure_cases_are_explicit(tmp_path):
    first = generate(tmp_path)
    bytes_before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    second = generate(tmp_path)
    assert first == second
    assert bytes_before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert {'ordinary_numeric_failure', 'state_numeric_failure', 'legacy_display_failure'} <= {c['id'] for c in first['cases']}


def kml(polygons):
    def ring(points):
        return '<LinearRing><coordinates>' + ' '.join(f'{x!r},{y!r},0' for x, y in points) + '</coordinates></LinearRing>'
    pieces = []
    for polygon in polygons:
        pieces.append('<Polygon><outerBoundaryIs>' + ring(polygon.exterior.coords) + '</outerBoundaryIs>' +
                      ''.join('<innerBoundaryIs>' + ring(hole.coords) + '</innerBoundaryIs>' for hole in polygon.interiors) + '</Polygon>')
    return '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><MultiGeometry>' + ''.join(pieces) + '</MultiGeometry></Placemark></Document></kml>'


def geographic_polygon(shape, origin=(-100, 40)):
    import math
    def point(xy):
        return GEOD.fwd(*origin, math.degrees(math.atan2(xy[0], xy[1])), math.hypot(*xy))[:2]
    return Polygon([point(p) for p in shape.exterior.coords], [[point(p) for p in ring.coords] for ring in shape.interiors])


def test_complete_polygon_parser_preserves_parts_holes_and_rejects_bad_second_part():
    first = Polygon([(0, 0), (0, .01), (.01, .01), (.01, 0)],
                    [[(.002, .002), (.002, .008), (.008, .008), (.008, .002)]])
    second = Polygon([(.02, 0), (.02, .01), (.03, .01), (.03, 0)])
    text = kml([first, second])
    parsed = document_polygons(text)
    assert len(parsed) == 2 and len(parsed[0]['holes']) == 1
    with pytest.raises(ValueError, match='Malformed'):
        document_polygons(text.replace('0.03,0.01,0', 'nan,0.01,0'))


def test_radius_audit_rejects_broad_rectangle_and_missing_second_piece():
    origin = (-100, 40)
    end = GEOD.fwd(*origin, 0, 100)[0:2]
    source = [[origin, end]]
    capsule = geographic_polygon(LineString([(0, 0), (0, 100)]).buffer(5, quad_segs=128))
    report, _, _ = inspect_document(kml([capsule]), source, padding_m=5)
    assert report['passed']
    rectangle = geographic_polygon(Polygon([(-50, -50), (50, -50), (50, 150), (-50, 150)]))
    report, _, _ = inspect_document(kml([rectangle]), source, padding_m=5)
    assert report['source_coverage_passed'] and not report['outer_radius_passed']
    partner = [GEOD.fwd(*point, 90, 20)[:2] for point in source[0]]
    report, _, _ = inspect_document(kml([capsule]), [*source, partner], padding_m=5)
    assert not report['source_coverage_passed'] and not report['inner_radius_passed']


def test_independent_run_slicing_preserves_gaps_and_native_bends():
    assert consecutive_ranges([4, 1, 0, 4, 9]) == [[0, 2], [4, 5], [9, 10]]
    a = (-100, 40); b = GEOD.fwd(*a, 0, 10)[:2]; c = GEOD.fwd(*b, 90, 10)[:2]
    span = path_span([a, a, b, c], 5, 15)
    assert b in span and a not in span and c not in span
    with pytest.raises(ValueError, match='exceeds'):
        path_span([a, b, c], 5, 21)
