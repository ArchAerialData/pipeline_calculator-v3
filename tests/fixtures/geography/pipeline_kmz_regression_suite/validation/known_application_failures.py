"""Narrow baseline defect classification, never a change to independent goldens.

The normal application comparison still exits unsuccessfully on these failures.
The audit runner may distinguish these two documented defects from new failures.
Every unrelated assertion stays mandatory, including on affected sources.
"""
from __future__ import annotations

import json

CLASSIFICATION_SCHEMA_VERSION = 1
BASELINE_COMMIT = '71da499d5756648ae395660f0a241ea00edbea4f'

KNOWN_ARCHIVES = {
    '02_three_state_transverse_crossings.kmz': 'f8d3e18f20a5dfd5d744d02e1b7fcfdd4ace00787ab80762a488d648ce908592',
    '04_shared_border_and_near_border.kmz': '606242cbd38dd4bf7dc0462f40c0cb2d2761173017a22e7218f19d836112f8fe',
    'variants/04_shared_border_and_near_border__redundant_vertices.kmz': 'd8ef3481933429d473cd7b9fba3d5d190d7dee7f490ba7483d804d00b10aa308',
}


def _same(actual, expected):
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _additional(actual, expected, extra):
    return _same(actual, sorted(expected + [extra]))


def classify_fixture(row):
    """Classify an already measured result; no report values are edited."""
    if not row['mismatches']:
        return {'status': 'passed', 'defect_id': None, 'unexpected_assertions': []}
    rules = {}
    fixture = row['fixture']
    signature = row.get('sha256') == KNOWN_ARCHIVES.get(fixture)
    spans = row.get('partition_observation', [])
    defect = None
    if fixture.startswith('02_'):
        defect = 'baseline-endpoint-touch-one-ulp-crossing'
        source = [s for s in spans if s['key'] == '02_endpoint_touch']
        signature = signature and len(source) == 2 and all(s['path_index'] == 0 for s in source)
        if signature:
            a, b = sorted(source, key=lambda s: s['start_m'])
            signature = (a['kind'] == b['kind'] == 'state' and a['state_codes'] == ['NM'] and
                         b['state_codes'] == ['TX'] and 0 < b['length_meters'] <= 1e-8 and
                         abs(b['start_m']-363.25) <= 1e-6 and abs(b['end_m']-363.25) <= 1e-6)
        rules = {
            'crossing events': lambda a, e: a == 10 and e == 9,
            'Partition/02_endpoint_touch/path0: interval count': lambda a, e: a == 2 and e == 1,
            'Combined export/02_endpoint_touch/path0: interval count': lambda a, e: a == 2 and e == 1,
            'TX: source sample identities': lambda a, e: _additional(a, e, '02_endpoint_touch'),
            'TX: attributed source identities': lambda a, e: _additional(a, e, '02_endpoint_touch'),
            'TX export: interval source/path identities': lambda a, e: _additional(a, e, ['02_endpoint_touch', 0]),
            'TX export/02_endpoint_touch/path0: interval count': lambda a, e: a == 1 and e == 0,
        }
    elif '04_shared_border_and_near_border' in fixture:
        defect = 'baseline-shared-junction-roundoff-unresolved'
        source = [s for s in spans if s['key'] == '04_shared_qualification_route']
        tiny = [s for s in source if s['kind'] == 'unresolved']
        signature = (signature and len(source) == 4 and len(tiny) == 1 and
                     tiny[0]['path_index'] == 0 and tiny[0]['state_codes'] == ['NM', 'TX'] and
                     0 < tiny[0]['length_meters'] <= 1e-12 and
                     abs(tiny[0]['start_m']-64) <= 1e-6 and abs(tiny[0]['end_m']-64) <= 1e-6)
        rules = {
            'Geography status': lambda a, e: a == 'incomplete' and e == 'complete',
            'Geography diagnostics': lambda a, e: a == {'ambiguous_state_coverage': 1, 'state_geometry_unresolved': 1} and e == {},
            'unresolved_meters': lambda a, e: 0 < a <= 1e-12 and e == 0,
            'Partition/04_shared_qualification_route/path0: interval count': lambda a, e: a == 4 and e == 3,
            'Combined export/04_shared_qualification_route/path0: interval count': lambda a, e: a == 4 and e == 3,
            'Geography analysis complete': lambda a, e: a is False and e is True,
            'NM: complete': lambda a, e: a is False and e is True,
            'TX: complete': lambda a, e: a is False and e is True,
            'NM: diagnostics': lambda a, e: e == {'selected_primary_kml': 1} and a == {**e, 'state_partition_incomplete': 1},
            'TX: diagnostics': lambda a, e: e == {'selected_primary_kml': 1} and a == {**e, 'state_partition_incomplete': 1},
        }
    unexpected = []
    for failure in row['mismatches']:
        rule = rules.get(failure['assertion'])
        actual, expected = json.loads(json.dumps([failure['actual'], failure['expected']]))
        if not signature or rule is None or not rule(actual, expected):
            unexpected.append(failure['assertion'])
    return {'status': 'unexpected_failure' if unexpected else 'known_failure',
            'defect_id': defect if signature else None, 'unexpected_assertions': unexpected}
