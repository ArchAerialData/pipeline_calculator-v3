"""Fast analytic and adversarial checks for the independent sampled contract.

These run on both the reference and production core. Coordinates are constructed
with GeographicLib, separately from their pyproj measurement. Integer answers
come from counted samples or a hand-specified eligibility graph, not frozen
application output. The full KMZ/state/export comparison is a separate command.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import itertools
from pathlib import Path
import sys

import pytest

for dependency in ('geographiclib', 'matplotlib', 'psutil'):
    pytest.importorskip(dependency, reason='Install the KMZ suite requirements to run its optional audit tests')

from geographiclib.geodesic import Geodesic

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.reference import sampled
from pipeline_kmz_regression_suite.validation.application_contract import (
    compare_sampled_contract, expected_section_records, section_records,
)
from pipeline_calculator.core.bundling import qualifying_sections, savings_from_sections
from pipeline_calculator.core.coordinates import segment_pipeline_paths
from pipeline_calculator.core.overlap import calculate_overlap_results, find_parallel_segments

SECOND_GEOD = Geodesic(6378137.0, 1.0 / 298.257222101)
ORIGIN = (-100.0, 31.0)


def point(start, bearing, meters):
    result = SECOND_GEOD.Direct(start[1], start[0], bearing, meters)
    return result['lon2'], result['lat2']


def meridians(offsets, length=302.3):
    sources = []
    for index, offset in enumerate(offsets):
        longitude = point(ORIGIN, 90, offset)[0]
        start = longitude, ORIGIN[1]
        sources.append({'key': chr(65 + index), 'name': 'Deliberately repeated name',
                        'paths': [[start, point(start, 0, length)]]})
    return sources


def application(sources):
    pipelines = [{'id': source['key'], 'name': source['name'],
                  'coordinate_paths': deepcopy(source['paths'])} for source in sources]
    matches = find_parallel_segments(pipelines, sampled.GEOD, 5.0, 15.0, 15.0)
    sections = qualifying_sections(pipelines, matches, 5.0, 200.0)
    return pipelines, matches, sections


def results(sources):
    expected = sampled.analyze(sources)
    pipelines, matches, sections = application(sources)
    records = section_records(pipelines, sections, {s['key']: s['key'] for s in sources})
    assert records == expected_section_records(expected)
    actual_savings = savings_from_sections(pipelines, sections, 5.0)
    assert actual_savings == expected['savings_meters']
    assert sum(len(p['segments']) for p in pipelines) == expected['sample_count']
    return expected, pipelines, matches, sections


@pytest.mark.parametrize('offsets,pair_count,savings', [
    ([0, 8], 1, 300.0),
    ([0, 6, 13], 3, 600.0),
    ([0, 7, 18], 2, 300.0),
])
def test_hand_counted_pair_trio_and_nonclique(offsets, pair_count, savings):
    sources = meridians(offsets)
    expected, _, _, _ = results(sources)
    assert expected['qualifying_section_count'] == pair_count
    assert expected['savings_meters'] == savings
    assert expected['qualifying_pairwise_meters'] == pair_count * 300.0
    assert all(s['sample_count'] == 60 for s in expected['source_coverage'].values())
    # This proves the Q=300 formula is applicable: each pair covers exactly 60
    # diagonal cells, not 119 phase-shifted matches or an assumed nominal length.
    assert all(s['coverage_ranges'] == [[[0, 60]], [[0, 60]]]
               and s['eligible_match_count'] == 60 for s in expected['sections'])
    if len(offsets) == 3:
        assert expected['savings_meters'] != expected['qualifying_pairwise_meters']
    for ordered in itertools.permutations(sources):
        reordered, _, _, _ = results(list(ordered))
        assert reordered == expected


@pytest.mark.parametrize('length,count,savings', [(197.3, 39, 0.0), (202.3, 40, 200.0)])
def test_minimum_is_40_full_samples_not_nominal_length(length, count, savings):
    expected, _, _, _ = results(meridians([0, 8], length))
    assert expected['sample_count'] == 2 * count
    assert expected['savings_meters'] == savings
    assert expected['qualifying_section_count'] == int(count == 40)
    assert all(abs(row['unsampled_tail_meters'] - 2.3) < 1e-6
               for row in expected['source_coverage'].values())


@pytest.mark.parametrize('reverse', [False, True])
def test_nonaligned_and_opposite_direction_samples(reverse):
    sources = meridians([0, 8])
    if reverse:
        # The 2.3 m trailing remainder moves to the other end on reversal.
        sources[1]['paths'][0].reverse()
    else:
        sources[1]['paths'][0] = [point(p, 0, 1.7) for p in sources[1]['paths'][0]]
    expected, _, _, _ = results(sources)
    assert expected['savings_meters'] == 300.0
    assert expected['sections'][0]['eligible_match_count'] == 119
    assert expected['sections'][0]['coverage_ranges'] == [[[0, 60]], [[0, 60]]]


def test_disconnected_paths_restart_samples_and_cannot_qualify_together():
    sources = meridians([0, 8], 123.0)
    for source in sources:
        start = point(source['paths'][0][0], 0, 500.0)
        source['paths'].append([start, point(start, 0, 123.0)])
    expected, _, _, _ = results(sources)
    assert expected['savings_meters'] == 0.0
    assert [s['length_meters'] for s in expected['sections']] == [120.0, 120.0]
    assert expected['qualifying_section_count'] == 0
    assert all(s['path_sample_counts'] == [24, 24]
               and abs(s['unsampled_tail_meters'] - 6.0) < 1e-6
               for s in expected['source_coverage'].values())


@pytest.mark.parametrize('missing_axis', [0, 1])
def test_a_missing_grid_row_or_column_splits_qualification(missing_axis):
    # Each side has 25 matches (125 m). The absent row/column separates them;
    # adding their 250 m would falsely qualify. Geometry cannot repair this gap.
    cells = [(i, i) for i in range(25)]
    cells += [(i + (missing_axis == 0), i + (missing_axis == 1))
              for i in range(25, 50)]
    components = list(sampled._components({cell: cell for cell in cells}))
    assert sorted(map(len, components)) == [25, 25]
    pipelines = [{'segments': [{'length': 5.0, 'path_index': 0,
                               'path_segment_index': i} for i in range(51)]}
                 for _ in range(2)]
    matches = {(0, 1): [{'pipeline_1_segment': i, 'pipeline_2_segment': j,
                         'distance': 8.0} for i, j in cells]}
    assert qualifying_sections(pipelines, matches, 5.0, 200.0) == []
    assert len(cells) * 5.0 > 200.0


@pytest.mark.parametrize('lateral,phase,eligible', [
    (14.8, 4.5, True),  # hypotenuse exceeds 15 m, yet finite tangents overlap
    (15.2, 0.0, False),
    (0.0, 5.0, False),  # exact endpoint touch is not positive overlap
    (8.0, 5.2, False),
])
def test_finite_tangents_not_midpoint_proximity(lateral, phase, eligible):
    sources = meridians([0, lateral], 5.3)
    sources[1]['paths'][0] = [point(p, 0, phase) for p in sources[1]['paths'][0]]
    left, _ = sampled._sample_path(sources[0], 0, 0)
    right, _ = sampled._sample_path(sources[1], 0, 0)
    matches = sampled._matches(left, right, Counter())
    _, actual, _ = application(sources)
    assert len(matches) == int(eligible)
    assert sum(map(len, actual.values())) == int(eligible)
    if eligible:
        assert matches[0][2] > 15.0


@pytest.mark.parametrize('angle,eligible', [(14.9, True), (15.1, False),
                                                   (180.0, True), (90.0, False)])
def test_orientation_boundary_and_opposite_digitization(angle, eligible):
    # Every path has one sample centered at ORIGIN. Position is immaterial;
    # only the orientation predicate can distinguish these candidates.
    sources = []
    for key, bearing in [('A', 0.0), ('B', angle)]:
        start = point(ORIGIN, bearing + 180.0, 2.5)
        end = point(ORIGIN, bearing, 2.8)
        sources.append({'key': key, 'name': key, 'paths': [[start, end]]})
    left, _ = sampled._sample_path(sources[0], 0, 0)
    right, _ = sampled._sample_path(sources[1], 0, 0)
    matches = sampled._matches(left, right, Counter())
    _, actual, _ = application(sources)
    assert len(matches) == int(eligible)
    assert sum(map(len, actual.values())) == int(eligible)


def test_bend_sample_uses_endpoint_chord_and_vertex_does_not_reset_chainage():
    turn = point(ORIGIN, 90.0, 2.0)
    boundary = point(turn, 0.0, 3.0)
    finish = point(turn, 0.0, 6.3)
    source = {'key': 'bend', 'name': 'bend', 'paths': [[ORIGIN, turn, finish]]}
    reference, original = sampled._sample_path(source, 0, 0)
    production = segment_pipeline_paths(sampled.GEOD, {'coordinate_paths': source['paths']}, 5.0)
    assert len(reference) == len(production) == 1
    assert abs(original - 8.3) < 1e-6
    chord = SECOND_GEOD.Inverse(ORIGIN[1], ORIGIN[0], boundary[1], boundary[0])
    midpoint = point(ORIGIN, chord['azi1'], chord['s12'] / 2)
    for actual_midpoint, actual_bearing in [
        ((reference[0].lon, reference[0].lat), reference[0].bearing),
        (production[0]['midpoint'], production[0]['bearing']),
    ]:
        error = SECOND_GEOD.Inverse(*actual_midpoint[::-1], *midpoint[::-1])['s12']
        assert error < 1e-6
        assert abs(actual_bearing - chord['azi1']) < 1e-6
        assert 33.0 < actual_bearing < 34.0


def test_retrace_within_one_source_never_creates_cross_source_savings():
    source = meridians([0])[0]
    source['paths'][0].append(source['paths'][0][0])
    expected, _, matches, _ = results([source])
    assert expected['sample_count'] == 120
    assert expected['savings_meters'] == 0.0
    assert expected['sections'] == []
    assert not matches


def test_unique_source_constraint_prevents_overcount_from_multiple_matches():
    # Both A samples match B and C, and B-C match. A clique without the
    # disjoint-membership and compatibility rules would incorrectly become
    # one transitive four-node group, saving 15 m instead of the correct 10 m.
    nodes = {}
    for key, index, longitude in [('A', 0, 0.0), ('A', 1, 0.1),
                                  ('B', 0, 0.2), ('C', 0, 0.3)]:
        node = sampled.Sample(key, 0, index, index, longitude, 0.0, 0.0, key, '')
        nodes[node.identity] = node
    a0, a1, b, c = [('A', 0, 0), ('A', 0, 1), ('B', 0, 0), ('C', 0, 0)]
    edges = {tuple(sorted(pair)): distance for pair, distance in [
        ((a0, b), 1.0), ((b, c), 2.0), ((a0, c), 3.0),
        ((a1, b), 4.0), ((a1, c), 5.0),
    ]}
    savings, proof, _ = sampled._group_savings(nodes, edges)
    assert savings == 10.0
    assert proof['group_size_counts'] == {'1': 1, '3': 1}
    assert proof['rejected_shared_source_merges'] > 0
    pipelines = [{'name': key, 'segments': [
        {'midpoint': (n.lon, n.lat), 'bearing': n.bearing, 'length': 5.0}
        for n in nodes.values() if n.source == key]} for key in ['A', 'B', 'C']]
    index = {'A': 0, 'B': 1, 'C': 2}
    sections = [{'pair': (index[a[0]], index[b[0]]), 'matches': [
        {'pipeline_1_segment': a[2], 'pipeline_2_segment': b[2], 'distance': distance}]}
        for (a, b), distance in edges.items()]
    assert savings_from_sections(pipelines, sections, 5.0) == 10.0


@pytest.mark.parametrize('mutation', ['shift_ranges', 'drop_match', 'duplicate_section',
                                    'wrong_public_source', 'wrong_public_coverage'])
def test_same_savings_does_not_hide_changed_sample_evidence(mutation):
    # Corrupt only section evidence while leaving the otherwise plausible
    # reported 300 m savings and pairwise coverage lengths untouched.
    sources = meridians([0, 8])
    expected, pipelines, matches, sections = results(sources)
    public = calculate_overlap_results(pipelines, matches, sampled.GEOD,
                                       sampled.SURVEY_MILE_METERS, 5.0, 200.0, 15.0, 15.0)
    keys = {s['key']: s['key'] for s in sources}
    failures = []

    def check(label, actual, target, tolerance=0):
        passed = abs(actual-target) <= tolerance if tolerance else actual == target
        if not passed:
            failures.append(label)

    compare_sampled_contract(check, 'control', pipelines, sections, public, expected, keys)
    assert not failures
    damaged = deepcopy(sections)
    if mutation == 'shift_ranges':
        # Move the first member's covered samples to a different path-local
        # range without changing any count, length, or pair identity.
        for segment in pipelines[0]['segments']:
            segment['path_segment_index'] += 1
    elif mutation == 'drop_match':
        damaged[0]['matches'].pop()
    elif mutation == 'duplicate_section':
        damaged.append(deepcopy(damaged[0]))
    elif mutation == 'wrong_public_source':
        public['bundled_sections'][0]['pipeline_1_id'] = public['bundled_sections'][0]['pipeline_2_id']
    else:
        public['pipeline_overlaps_by_id']['A']['bundled_segments'] -= 1
    assert public['savings_meters'] == expected['savings_meters'] == 300.0
    compare_sampled_contract(check, 'control', pipelines, damaged, public, expected, keys)
    assert failures, f'{mutation} must not pass merely because savings are unchanged'
    assert any('sample' in label or 'section' in label for label in failures)
