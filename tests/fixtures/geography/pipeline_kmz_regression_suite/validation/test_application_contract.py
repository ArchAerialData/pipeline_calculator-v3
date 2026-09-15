"""Adversarial tests of the audit checker, using deliberately wrong observations."""
from copy import deepcopy
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

for dependency in ('geographiclib', 'matplotlib', 'psutil'):
    pytest.importorskip(dependency, reason='Install the KMZ suite requirements to run its optional audit tests')

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon, box

from pipeline_kmz_regression_suite.validation.application_contract import (
    GEOD, compare_interval_ledger, geometry_span_error, polygon_containment, compare_export_corridors,
)
from pipeline_kmz_regression_suite.validation.known_application_failures import (
    KNOWN_ARCHIVES, classify_fixture,
)


def recorder():
    checks = []
    def check(label, actual, expected, tolerance=0):
        passed = (type(actual) in (float, int) and math.isfinite(actual) and abs(actual-expected) <= tolerance
                  if type(expected) in (float, int) else actual == expected)
        checks.append((label, passed))
    return checks, check


class IntervalAndExportCheckerTests(unittest.TestCase):
    def setUp(self):
        self.a = (-103.0, 33.0)
        point = GEOD.Direct(self.a[1], self.a[0], 90, 302.3)
        self.b = (point['lon2'], point['lat2'])
        self.gold = {'key': 'route', 'path_index': 0, 'kind': 'interior', 'state_codes': ['TX'],
                     'start_m': 0.0, 'end_m': 302.3, 'length_meters': 302.3,
                     'cut_error_bound_meters': 2e-6, 'coordinates': [self.a, self.b]}
        self.observed = {**self.gold, 'source_id': 7, 'kind': 'state'}
        self.originals = {'route': {'paths': [[self.a, self.b]]}}

    def check_rows(self, observed, gold=None):
        checks, check = recorder()
        compare_interval_ledger(check, 'test', observed, gold or [self.gold], {7: 'route'}, originals=self.originals)
        return checks

    def test_correct_fragment_passes_and_export_identity_is_independent(self):
        self.assertTrue(all(passed for _, passed in self.check_rows([self.observed])))
        checks, check = recorder()
        exported = {**self.observed, 'paths': [self.observed['coordinates']]}
        compare_interval_ledger(check, 'export', [exported], [self.gold], {7: 'route'}, export=True, originals=self.originals)
        self.assertTrue(all(passed for _, passed in checks))
        exported['source_id'] = 8
        checks, check = recorder()
        compare_interval_ledger(check, 'export', [exported], [self.gold], {7: 'route'}, export=True)
        self.assertTrue(any(not passed for _, passed in checks))

    def test_correct_length_wrong_owner_fails(self):
        bad = {**self.observed, 'state_codes': ['NM']}
        self.assertTrue(any('ownership' in label and not passed for label, passed in self.check_rows([bad])))

    def test_missing_or_malformed_fragment_geometry_fails_explicitly(self):
        for coordinates in (None, [], [self.a], 'coordinates', [self.a, (float('nan'), 33)],
                            [self.a, (181, 33)], [self.a, (True, 33)], [self.a, (-103,)],
                            [self.a, ('-103', 33)]):
            with self.subTest(coordinates=coordinates):
                bad = {**self.observed, 'coordinates': coordinates}
                checks = self.check_rows([bad])
                self.assertTrue(any('valid coordinates' in label and not passed for label, passed in checks))
        bad = dict(self.observed)
        del bad['coordinates']
        self.assertTrue(any('valid coordinates' in label and not passed for label, passed in self.check_rows([bad])))

    def test_missing_or_malformed_export_path_fails_explicitly(self):
        for paths in (None, [], [None], [[self.a]], ['coordinates'], [[self.a, (float('inf'), 33)]]):
            with self.subTest(paths=paths):
                checks, check = recorder()
                compare_interval_ledger(check, 'export', [{**self.observed, 'paths': paths}],
                                        [self.gold], {7: 'route'}, export=True, originals=self.originals)
                self.assertTrue(any('valid coordinates' in label and not passed for label, passed in checks))

    def test_centimeter_sideways_shift_cannot_use_cut_location_allowance(self):
        shifted = []
        for lon, lat in self.observed['coordinates']:
            p = GEOD.Direct(lat, lon, 0, .01)
            shifted.append((p['lon2'], p['lat2']))
        bad = {**self.observed, 'coordinates': shifted}
        self.assertTrue(any('original geodesic' in label and not passed for label, passed in self.check_rows([bad])))

    def test_known_extra_interval_does_not_skip_other_geometry_checks(self):
        bad = {**self.observed, 'state_codes': ['NM']}
        extra = {**self.observed, 'start_m': 302.3, 'end_m': 302.300000001,
                 'length_meters': 1e-9, 'coordinates': [self.b, self.b]}
        checks = self.check_rows([bad, extra])
        self.assertTrue(any('interval count' in label and not passed for label, passed in checks))
        self.assertTrue(any('ownership' in label and not passed for label, passed in checks))

    def test_same_endpoints_and_same_length_do_not_hide_wrong_bend(self):
        # Symmetric detours have equal length but opposite bends.
        north = (-103.0, 33.001)
        south = (-103.0, 32.999)
        west, east = (-103.001, 33.0), (-102.999, 33.0)
        self.assertGreater(geometry_span_error([west, north, east], [west, south, east]), 200)

    def test_polygon_containment_holes_and_tiny_foreign_shapes(self):
        boundary = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)], holes=[[(.8, .8), (1.2, .8), (1.2, 1.2), (.8, 1.2)]])
        self.assertTrue(polygon_containment([box(.1, .1, .5, .5)], boundary)[0]['passed'])
        self.assertFalse(polygon_containment([box(.9, .9, 1.1, 1.1)], boundary)[0]['passed'])
        self.assertFalse(polygon_containment([box(3, 3, 3+1e-10, 3+1e-10)], boundary)[0]['passed'])
        self.assertFalse(polygon_containment([box(1.999999, .1, 2.000001, .5)], boundary)[0]['passed'])

    def test_polygon_sub_ulp_boundary_residue_uses_local_bound(self):
        boundary = box(-104, 32, -103, 33)
        polygon = box(-103.001, 32.1, -103 + 1e-13, 32.2)
        row = polygon_containment([polygon], boundary)[0]
        self.assertTrue(row['passed'])
        self.assertLess(row['coordinate_tolerance_degrees'], 1e-11)

    def test_missing_or_fabricated_corridors_fail_despite_correct_line_totals(self):
        expected = {'sections': [dict(qualified=True, source_keys=['A', 'B'], path_indices=[0, 0],
                                     coverage_ranges=[[[0, 60]], [[0, 60]]], coverage_sample_counts=[60, 60],
                                     eligible_match_count=60, length_meters=300.0)]}
        sources = [dict(key=key, paths=[[self.a, self.b]]) for key in ('A', 'B')]
        metadata = {'pipeline_1_id': '1', 'pipeline_2_id': '2'}
        for corridors in ([], [dict(metadata=metadata, polygons=[box(-100, 30, -99, 31)])]):
            checks, check = recorder()
            compare_export_corridors(check, 'Combined', corridors, expected, sources, {1: 'A', 2: 'B'})
            self.assertTrue(any(not passed for _, passed in checks))

    def corridor_case(self):
        expected = {'sections': [dict(qualified=True, source_keys=['A', 'B'], path_indices=[0, 0],
                                     coverage_ranges=[[[0, 60]], [[0, 60]]], coverage_sample_counts=[60, 60],
                                     eligible_match_count=60, length_meters=300.0)]}
        sources = [dict(key=key, paths=[[self.a, self.b]]) for key in ('A', 'B')]
        metadata = {'pipeline_1_id': '1', 'pipeline_2_id': '2'}
        return expected, sources, metadata

    def corridor_checks(self, polygons, *, expected=None, sources=None):
        target, inputs, metadata = self.corridor_case()
        checks, check = recorder()
        compare_export_corridors(check, 'Combined', [dict(metadata=metadata, polygons=polygons)],
                                 expected or target, sources or inputs, {1: 'A', 2: 'B'})
        return checks

    def test_whole_state_corridor_fails_even_when_samples_and_state_containment_pass(self):
        state = box(-104, 32, -102, 34)
        self.assertTrue(polygon_containment([state], state)[0]['passed'])
        checks = self.corridor_checks([state])
        self.assertTrue(any('cover qualified path samples' in label and passed for label, passed in checks))
        self.assertTrue(any('independent section bounds' in label and not passed for label, passed in checks))

    def test_remote_island_appended_to_valid_corridor_fails(self):
        body = LineString([self.a, self.b]).buffer(.0001)
        island = box(-103.1, 33.1, -103.099, 33.101)
        state = box(-104, 32, -102, 34)
        self.assertTrue(all(row['passed'] for row in polygon_containment([body, island], state)))
        checks = self.corridor_checks([body, island])
        self.assertTrue(any('cover qualified path samples' in label and passed for label, passed in checks))
        self.assertTrue(any('independent section bounds' in label and not passed for label, passed in checks))

    def test_curve_rotated_rectangle_and_geographic_box_fallbacks_remain_allowed(self):
        line = LineString([self.a, self.b])
        west, south, east, north = line.bounds
        shapes = [line.buffer(.0001),
                  rotate(box(west-.00015, south-.0006, east+.00015, north+.0006), 8),
                  box(west-.001, south-.001, east+.001, north+.001)]
        for shape in shapes:
            with self.subTest(shape=shape.wkt):
                self.assertTrue(all(passed for _, passed in self.corridor_checks([shape])))

    def test_bent_section_rectangle_fallback_is_bounded_by_its_own_complete_span(self):
        expected, sources, _ = self.corridor_case()
        corner = GEOD.Direct(self.a[1], self.a[0], 90, 150)
        bend = (corner['lon2'], corner['lat2'])
        tip = GEOD.Direct(bend[1], bend[0], 0, 152.3)
        path = [self.a, bend, (tip['lon2'], tip['lat2'])]
        for source in sources:
            source['paths'] = [path]
        # A permitted rectangle covers the empty inside of a bend: a strict
        # narrow path buffer would incorrectly reject this documented fallback.
        west, south, east, north = LineString(path).bounds
        shape = box(west-.001, south-.001, east+.001, north+.001)
        self.assertTrue(all(passed for _, passed in self.corridor_checks([shape], expected=expected, sources=sources)))

    def test_clipped_multipart_polygons_and_holes_remain_allowed(self):
        body = LineString([self.a, self.b]).buffer(.0001)
        west, south, east, north = body.bounds
        # Split between sampled points, as a narrow boundary hole can do. Keep
        # the independent sample locations covered on both resulting pieces.
        cut = GEOD.Direct(self.a[1], self.a[0], 90, 150)
        slit = box(cut['lon2']-1e-7, south-1, cut['lon2']+1e-7, north+1)
        pieces = list(body.difference(slit).geoms)
        hole = box(self.a[0]+.0003, north-.00004, self.a[0]+.00032, north-.00002)
        pieces = [piece.difference(hole) for piece in pieces]
        self.assertTrue(any(len(p.interiors) for p in pieces))
        self.assertTrue(all(passed for _, passed in self.corridor_checks(pieces)))

    def test_separate_sections_cannot_borrow_each_others_geometry(self):
        expected, sources, metadata = self.corridor_case()
        second = deepcopy(expected['sections'][0])
        second['path_indices'] = [1, 1]
        expected['sections'].append(second)
        for source in sources:
            source['paths'].append([(lon+.1, lat) for lon, lat in source['paths'][0]])
        first_body = LineString([self.a, self.b]).buffer(.0001)
        second_body = translate(first_body, xoff=.1)
        cases = [([dict(metadata=metadata, polygons=[first_body]),
                   dict(metadata=metadata, polygons=[second_body])], True),
                 ([dict(metadata=metadata, polygons=[first_body, second_body]) for _ in range(2)], False)]
        for corridors, should_pass in cases:
            checks, check = recorder()
            compare_export_corridors(check, 'Combined', corridors, expected, sources, {1: 'A', 2: 'B'})
            self.assertEqual(all(passed for _, passed in checks), should_pass)
            if not should_pass:
                # Old union-by-pair coverage alone accepted this duplicated map.
                self.assertTrue(any('cover qualified path samples' in label and passed for label, passed in checks))
                self.assertTrue(any('one-to-one' in label and not passed for label, passed in checks))


class KnownDefectClassificationTests(unittest.TestCase):
    def baseline(self):
        name = '02_three_state_transverse_crossings.kmz'
        return {'fixture': name, 'sha256': KNOWN_ARCHIVES[name],
                'partition_observation': [
                    dict(key='02_endpoint_touch', path_index=0, kind='state', state_codes=['NM'],
                         start_m=0, end_m=363.249999999, length_meters=363.249999999),
                    dict(key='02_endpoint_touch', path_index=0, kind='state', state_codes=['TX'],
                         start_m=363.249999999, end_m=363.25, length_meters=1e-9)],
                'mismatches': [dict(assertion='crossing events', actual=10, expected=9),
                               dict(assertion='Partition/02_endpoint_touch/path0: interval count', actual=2, expected=1)]}

    def test_known_failure_remains_explicit_and_fixed_result_passes(self):
        row = self.baseline()
        self.assertEqual(classify_fixture(row)['status'], 'known_failure')
        row['mismatches'] = []
        self.assertEqual(classify_fixture(row)['status'], 'passed')

    def test_new_failure_cannot_hide_inside_affected_fixture(self):
        for alteration in ('new_assertion', 'wrong_count', 'large_fragment', 'different_source', 'different_archive'):
            with self.subTest(alteration=alteration):
                row = deepcopy(self.baseline())
                if alteration == 'new_assertion':
                    row['mismatches'].append(dict(assertion='Combined savings', actual=5, expected=10))
                elif alteration == 'wrong_count':
                    row['mismatches'][0]['actual'] = 11
                elif alteration == 'large_fragment':
                    row['partition_observation'][1]['length_meters'] = .01
                elif alteration == 'different_source':
                    row['partition_observation'][1]['key'] = 'other'
                else:
                    row['sha256'] = 'modified'
                self.assertEqual(classify_fixture(row)['status'], 'unexpected_failure')


class ApplicationReceiptTests(unittest.TestCase):
    def test_missing_inventory_replaces_old_full_success(self):
        from pipeline_kmz_regression_suite.validation import compare_application as application
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / 'validation/application_comparison.json'
            application.dump(report_path, {'status': 'complete', 'passed': True})
            with patch.object(application, 'SUITE', root), patch.object(sys, 'argv', ['compare_application.py']):
                with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                    application.main()
            report = application.read_json(report_path)
            self.assertEqual(report['status'], 'failed')
            self.assertIs(report['passed'], False)

    def test_focused_preflight_failure_keeps_full_receipt_and_replaces_focused(self):
        from pipeline_kmz_regression_suite.validation import compare_application as application
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full = root / 'validation/application_comparison.json'
            focused = root / 'validation/focused_application_comparison.json'
            original = {'status': 'complete', 'passed': True, 'marker': 'full'}
            application.dump(full, original)
            application.dump(focused, {'status': 'complete', 'passed': True})
            with patch.object(application, 'SUITE', root), patch.object(sys, 'argv', ['compare_application.py', '--only', 'typo']), \
                    patch.object(application, 'select_fixtures', side_effect=ValueError('No fixture matches typo')):
                with self.assertRaisesRegex(ValueError, 'No fixture matches'):
                    application.main()
            self.assertEqual(application.read_json(full), original)
            self.assertIs(application.read_json(focused)['passed'], False)
            self.assertEqual(application.read_json(focused)['status'], 'failed')

    def test_nonfinite_expected_json_is_rejected_before_application_runs(self):
        from pipeline_kmz_regression_suite.validation import compare_application as application
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'expected').mkdir()
            archive = root / 'fixtures/fake.kmz'
            (root / 'expected/fake.expected.json').write_text('{"bad": NaN}', encoding='utf-8')
            with patch.object(application, 'SUITE', root), patch.object(sys, 'argv', ['compare_application.py']), \
                    patch.object(application, 'select_fixtures', return_value=[archive]), patch.object(application, 'compare') as analyze:
                with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
                    application.main()
            analyze.assert_not_called()
            report = application.read_json(root / 'validation/application_comparison.json')
            self.assertEqual(report['status'], 'failed')

    def test_boundary_initialization_failure_is_failed_receipt(self):
        from pipeline_kmz_regression_suite.validation import compare_application as application
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'expected').mkdir()
            (root / 'fixtures').mkdir()
            archive = root / 'fixtures/fake.kmz'
            archive.write_bytes(b'archive')
            application.dump(root / 'expected/fake.expected.json', {'fixture': 'fake.kmz', 'archive': {'sha256': application.digest(archive)}})
            with patch.object(application, 'SUITE', root), patch.object(sys, 'argv', ['compare_application.py']), \
                    patch.object(application, 'select_fixtures', return_value=[archive]), \
                    patch.object(application, 'GeometryReference', side_effect=ValueError('invalid boundary')):
                with self.assertRaisesRegex(ValueError, 'invalid boundary'):
                    application.main()
            report = application.read_json(root / 'validation/application_comparison.json')
            self.assertEqual(report['status'], 'failed')
            self.assertIs(report['passed'], False)


if __name__ == '__main__':
    unittest.main()
