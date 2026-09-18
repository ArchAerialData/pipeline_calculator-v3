"""Frozen independent KMZ goldens run in ordinary CI without oracle dependencies.

The dedicated suite reconstructs the references and checks detailed sample/export
geometry. These integration checks keep every delivered fixture in the standard
application regression gate, including environments without GeographicLib.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.parsers.kml_kmz import extract_features_from_file_with_diagnostics


SUITE = Path(__file__).parent / 'fixtures/geography/pipeline_kmz_regression_suite'
CORE = (
    '01_three_state_disconnected_networks',
    '02_three_state_transverse_crossings',
    '03_parallel_corridors_crossing_borders',
    '04_shared_border_and_near_border',
    'variants/01_three_state_disconnected_networks__source_order',
    'variants/04_shared_border_and_near_border__redundant_vertices',
    'variants/04_shared_border_and_near_border__reversed',
)


def close(actual, expected, tolerance=.001):
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=0, abs=tolerance)


@pytest.fixture(scope='module')
def boundary_digest():
    resource = Path(__file__).parents[1] / 'src/pipeline_calculator/data/states_2025.zip'
    return hashlib.sha256(resource.read_bytes()).hexdigest()


@pytest.mark.parametrize('stem', CORE, ids=lambda stem: Path(stem).name)
def test_independent_kmz_goldens_in_both_analysis_modes(stem, boundary_digest):
    path = SUITE / 'fixtures' / f'{stem}.kmz'
    expected = json.loads((SUITE / 'expected' / f'{Path(stem).name}.expected.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['archive']['sha256']
    assert boundary_digest == expected['boundary_sha256']
    assert expected['reference_status'] == 'reference-verified'
    xml_to_key = dict(zip(expected['archive']['xml_ids'], expected['archive']['xml_source_order']))
    parsed = extract_features_from_file_with_diagnostics(path)
    assert len(parsed.pipelines) == expected['archive']['source_count']
    assert sum(len(p['coordinate_paths']) for p in parsed.pipelines) == expected['archive']['path_count']
    assert sum(len(c) for p in parsed.pipelines for c in p['coordinate_paths']) == expected['archive']['vertex_count']
    id_to_key = {p['id']: xml_to_key[p['placemark_id']] for p in parsed.pipelines}
    assert len(id_to_key) == len(parsed.pipelines)
    source_targets = {s['key']: s for s in expected['geometry']['sources']}

    analyzer = PipelineAnalyzer()
    ordinary = analyzer.analyze_complete(path, options=AnalysisOptions(False))
    result = analyzer.analyze_complete(path, options=AnalysisOptions(True))
    assert 'geography' not in ordinary
    target = expected['analyses']['Combined']
    for observed in (ordinary, result):
        assert observed['analysis_complete'], observed['diagnostics']
        assert Counter(d['code'] for d in observed['diagnostics']) == expected['expected_application_status']['combined_diagnostic_counts']
        close(observed['total_meters'], target['original_meters'])
        close(observed['overlap_analysis']['savings_meters'], target['savings_meters'], 1e-7)
        close(observed['overlap_analysis']['effective_total_meters'], target['adjusted_meters'])
        assert len(observed['overlap_analysis']['bundled_sections']) == target['qualifying_section_count']
        assert len(observed['pipelines']) == len(source_targets)
        for row in observed['pipelines']:
            close(row['Shape_Length'], source_targets[xml_to_key[row['Placemark_ID']]]['original_meters'])

    geography = result['geography']
    assert geography['status'] == 'complete', geography['diagnostics']
    assert geography['analysis_complete']
    assert geography['reconciliation']['passed']
    assert Counter(d['code'] for d in geography['diagnostics']) == expected['expected_application_status']['geography_diagnostic_counts']
    assert geography['crossing_count'] == expected['geometry']['crossing_count']
    assert geography['reconciliation']['outside_meters'] == expected['geometry']['outside_meters'] == 0
    assert geography['reconciliation']['unresolved_meters'] == expected['geometry']['unresolved_meters'] == 0

    # Compare ownership per original source/path so mistakes cannot cancel in totals.
    actual_intervals, target_intervals = defaultdict(list), defaultdict(list)
    for fragment in geography['fragments']:
        identity = (id_to_key[fragment['source_id']], fragment['path_index'],
                    'interior' if fragment['kind'] == 'state' else fragment['kind'],
                    tuple(fragment['state_codes']))
        actual_intervals[identity].append(fragment['length_meters'])
        assert fragment['length_meters'] > 0
    for fragment in expected['geometry']['intervals']:
        identity = (fragment['source_key'], fragment['path_index'], fragment['kind'], tuple(fragment['state_codes']))
        target_intervals[identity].append(fragment['length_meters'])
    assert actual_intervals.keys() == target_intervals.keys()
    for identity, lengths in target_intervals.items():
        close(math.fsum(actual_intervals[identity]), math.fsum(lengths))

    states = {state['state_code']: state for state in geography['states']}
    assert states.keys() == expected['geometry']['states'].keys()
    for code, state in states.items():
        accounting = expected['geometry']['states'][code]
        state_target = expected['analyses'][code]
        tolerance = max(.001, accounting['length_error_bound_meters'])
        assert state['analysis_complete'], state['diagnostics']
        close(state['interior_meters'], accounting['interior_meters'], tolerance)
        close(state['shared_allocation_meters'], accounting['shared_allocation_meters'], tolerance)
        close(state['total_meters'], accounting['attributed_original_meters'], tolerance)
        close(state['interior_savings_meters'], state_target['savings_meters'], 1e-7)
        close(state['adjusted_total_meters'], state['total_meters'] - state_target['savings_meters'], 1e-7)
        assert len(state['overlap_analysis']['bundled_sections']) == state_target['qualifying_section_count']
        if accounting['shared_allocation_meters']:
            assert state['shared_overlap_status'] == 'not_calculated'
        rows = {id_to_key[row['source_id']]: row for row in state['pipelines']}
        assert len(rows) == len(state['pipelines'])
        assert rows.keys() == {key for key, source in source_targets.items() if code in source['states']}
        for key, row in rows.items():
            source_target = source_targets[key]['states'][code]
            close(row['Shape_Length'], source_target['attributed_original_meters'])
            close(row['interior_meters'], source_target['interior_meters'])
            close(row['shared_allocation_meters'], source_target['shared_allocation_meters'])
    close(math.fsum(state['total_meters'] for state in states.values()), result['total_meters'])
