"""Adversarial tests of the validation harness, independent of the goldens."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

for dependency in ('geographiclib', 'matplotlib', 'psutil'):
    pytest.importorskip(dependency, reason='Install the KMZ suite requirements to run its optional audit tests')

from pipeline_kmz_regression_suite import suite
from pipeline_kmz_regression_suite.validation.run_stress import workload_valid
from pipeline_kmz_regression_suite.validation import run_audit


class ValidationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='kmz-harness-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def populate(self):
        for name in suite.CORE_FILES:
            path = self.root / 'fixtures' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            for suffix in suite.LEDGER_SUFFIXES:
                path = self.root / 'expected' / f'{Path(name).stem}.{suffix}'
                path.parent.mkdir(exist_ok=True)
                path.write_text('{}', encoding='utf-8')
        for name in suite.PREVIEW_FILES:
            path = self.root / 'previews' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\x89PNG\r\n\x1a\n')
        path = self.root / 'validation/design_manifest.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text('{}', encoding='utf-8')

    def test_empty_archive_inventory_is_not_success(self):
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            suite.select_fixtures(self.root)

    def test_unknown_selection_is_not_zero_test_success(self):
        self.populate()
        with self.assertRaisesRegex(ValueError, 'No fixture matches'):
            suite.select_fixtures(self.root, 'misspelled_case')

    def test_focused_selection_has_real_inputs(self):
        self.populate()
        self.assertEqual(2, len(suite.select_fixtures(self.root, '01_')))

    def test_omitted_archive_is_not_hidden_by_glob(self):
        self.populate()
        (self.root / 'fixtures' / suite.CORE_FILES[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'missing='):
            suite.select_fixtures(self.root)

    def test_extra_archive_requires_review(self):
        self.populate()
        (self.root / 'fixtures/unreviewed.kmz').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError, 'unexpected='):
            suite.select_fixtures(self.root)

    def test_nested_stress_name_cannot_hide_an_unreviewed_archive(self):
        self.populate()
        path = self.root / 'fixtures/variants/stress/unreviewed.kmz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'unreviewed')
        with self.assertRaisesRegex(ValueError, 'unexpected='):
            suite.select_fixtures(self.root)

    def test_nested_stress_archive_is_not_silently_ignored(self):
        path = self.root / 'fixtures/stress/nested/unreviewed.kmz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'unreviewed')
        with self.assertRaisesRegex(ValueError, 'Unexpected stress archive'):
            suite.validate_stress_reference(self.root)

    def test_missing_and_orphaned_expectations_fail(self):
        self.populate()
        path = self.root / 'expected' / f'{suite.MAIN_STEMS[0]}.expected.json'
        path.rename(path.with_name('orphan.expected.json'))
        with self.assertRaisesRegex(ValueError, 'Expectation inventory mismatch'):
            suite.select_fixtures(self.root)

    def test_missing_csv_is_required_evidence(self):
        self.populate()
        (self.root / 'expected' / f'{suite.MAIN_STEMS[0]}.overlaps.csv').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required ledger'):
            suite.select_fixtures(self.root)

    def test_nonfinite_expected_or_actual_numbers_fail(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=bad):
                self.assertTrue(suite.compare_values({'meters': 100.0}, {'meters': bad}))
                self.assertTrue(suite.compare_values({'meters': bad}, {'meters': 100.0}))

    def test_nonstandard_nan_json_is_rejected(self):
        path = self.root / 'bad.json'
        path.write_text('{"meters": NaN}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
            suite.read_json(path)

    def test_boolean_is_not_a_numeric_count(self):
        self.assertTrue(suite.compare_values(1, True))
        self.assertTrue(suite.compare_values(False, 0))
        self.assertTrue(suite.compare_values(40, 40.0))
        self.assertFalse(suite.compare_values(True, True))

    def test_one_sample_error_cannot_hide_in_float_tolerance(self):
        self.assertTrue(suite.compare_values({'savings_meters': 300.0}, {'savings_meters': 305.0}))
        self.assertFalse(suite.compare_values({'savings_meters': 300.0}, {'savings_meters': 300.0 + 1e-9}))

    def test_changed_csv_invalidates_pass_receipt(self):
        self.populate()
        report = self.receipt()
        suite.verify_report_freshness(report, self.root)
        path = self.root / 'expected' / f'{suite.MAIN_STEMS[0]}.intervals.csv'
        path.write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'stale'):
            suite.verify_report_freshness(report, self.root)

    def receipt(self):
        return {'status': 'complete', 'passed': True, 'selection': 'all_core', 'errors': [],
                'fixtures': [{'fixture': name, 'reference_status': 'reference-verified', 'errors': []}
                             for name in suite.CORE_FILES],
                'artifact_sha256': suite.artifact_fingerprints(self.root),
                'independent_module_audit': suite.audit_independence(),
                'validator_sha256': suite.digest(Path(suite.__file__))}

    def test_focused_or_empty_receipt_cannot_certify_full_suite(self):
        self.populate()
        for change in ({'selection': '01_'}, {'fixtures': []}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'every completed core fixture'):
                suite.verify_report_freshness({**self.receipt(), **change}, self.root)

    def test_changed_validator_invalidates_receipt(self):
        self.populate()
        with self.assertRaisesRegex(ValueError, 'accounting code changed'):
            suite.verify_report_freshness({**self.receipt(), 'validator_sha256': 'old-code'}, self.root)

    def test_truncated_png_is_not_a_valid_preview(self):
        self.populate()
        with self.assertRaises(OSError):
            suite.validate_previews(self.root)

    def test_failed_receipt_never_becomes_ready(self):
        with self.assertRaisesRegex(ValueError, 'No complete passing'):
            suite.verify_report_freshness({'passed': True, 'status': 'running'}, self.root)

    def test_later_unrelated_commit_does_not_block_regeneration(self):
        with patch.object(suite.subprocess, 'check_output', side_effect=['new-unrelated-commit\n', '']):
            result = suite.contract_provenance(self.root)
        self.assertTrue(result['baseline_contract_unchanged'])
        self.assertEqual('new-unrelated-commit', result['working_commit'])

    def test_dirty_contract_files_are_detected_even_with_same_head(self):
        with patch.object(suite.subprocess, 'check_output', side_effect=[suite.BASELINE + '\n', 'src/pipeline_calculator/core/overlap.py\n']):
            result = suite.contract_provenance(self.root)
        self.assertFalse(result['baseline_contract_unchanged'])

    def test_validation_error_replaces_old_success_receipt(self):
        suite.dump(self.root / 'validation/reference_report.json', {'status': 'complete', 'passed': True})
        with patch.object(suite, 'SUITE', self.root), patch.object(sys, 'argv', ['suite.py', 'validate']), \
                patch.object(suite, 'run_references', side_effect=ValueError('deliberate bad fixture')):
            with self.assertRaises(SystemExit):
                suite.main()
        report = suite.read_json(self.root / 'validation/reference_report.json')
        self.assertFalse(report['passed'])
        self.assertEqual('failed', report['status'])
        self.assertIn('deliberate bad fixture', report['errors'])

    def test_failed_publication_restores_previous_files(self):
        staged = self.root / 'staged'
        target = self.root / 'published'
        staged.mkdir()
        target.mkdir()
        for name in ('a.json', 'b.json'):
            (staged / name).write_text('new', encoding='utf-8')
            (target / name).write_text('old', encoding='utf-8')
        replace = Path.replace
        calls = []

        def fail_second(path, destination):
            calls.append(destination)
            if len(calls) == 2:
                raise OSError('simulated disk failure')
            return replace(path, destination)

        with patch.object(Path, 'replace', fail_second):
            with self.assertRaisesRegex(OSError, 'simulated disk'):
                suite.publish_generated(staged, target)
        self.assertEqual(['old', 'old'], [(target / name).read_text() for name in ('a.json', 'b.json')])
        self.assertFalse(list(target.glob('*.tmp')))

    def test_optimized_python_cannot_skip_validator_assertions(self):
        completed = subprocess.run([sys.executable, '-O', str(Path(suite.__file__)), 'validate'],
                                   capture_output=True, text=True)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn('requires assertions', completed.stderr)

    def test_empty_or_wrong_stress_measurement_is_not_a_pass(self):
        limits = {'segment_count': 100, 'neighbor_visits': 500, 'candidate_checks': 200}
        self.assertFalse(workload_valid([], 60, limits))
        self.assertFalse(workload_valid([{}], 60, limits))
        self.assertFalse(workload_valid([{'segment_count': 59, 'neighbor_visits': 100, 'candidate_checks': 70}], 60, limits))
        self.assertTrue(workload_valid([{'segment_count': 60, 'neighbor_visits': 100, 'candidate_checks': 70}], 60, limits))

    def test_audit_rejects_incomplete_application_inventory(self):
        report = {'status': 'complete', 'export_checks_enabled': True, 'selection': None,
                  'fixtures': [], 'passed': True}
        with self.assertRaisesRegex(ValueError, 'exact core inventory'):
            run_audit.classify_application(report)

    def test_audit_requires_exports_and_full_selection(self):
        for change in ({'export_checks_enabled': False}, {'selection': '01_'}):
            report = {'status': 'complete', 'export_checks_enabled': True, 'selection': None, **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'full application run'):
                run_audit.classify_application(report)

    def test_audit_rejects_stale_application_assets(self):
        report = {'status': 'complete', 'export_checks_enabled': True, 'selection': None,
                  'fixtures': [{'fixture': name, 'sha256': 'old'} for name in suite.CORE_FILES]}
        with patch.object(run_audit, 'digest', return_value='current'):
            with self.assertRaisesRegex(ValueError, 'stale'):
                run_audit.classify_application(report)

    def test_audit_rejects_empty_assertions_despite_green_fixture(self):
        report = {'status': 'complete', 'export_checks_enabled': True, 'selection': None,
                  'fixtures': [{'fixture': name, 'sha256': 'same', 'expectation_sha256': 'same',
                                'checks': [], 'mismatches': [], 'passed': True} for name in suite.CORE_FILES]}
        with patch.object(run_audit, 'digest', return_value='same'):
            with self.assertRaisesRegex(ValueError, 'empty application checks'):
                run_audit.classify_application(report)

    def test_comparison_startup_error_cannot_reuse_known_failure_receipt(self):
        path = self.root / 'validation/application_comparison.json'
        suite.dump(path, {'status': 'complete', 'passed': False,
                          'fixtures': [{'fixture': 'old-measured-known-defect'}]})
        with patch.object(run_audit, 'SUITE', self.root), \
                patch.object(run_audit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(ValueError, 'complete, full application run'):
                run_audit.run_application_comparison()
        self.assertEqual('running', suite.read_json(path)['status'])
        self.assertEqual([], suite.read_json(path)['fixtures'])


if __name__ == '__main__':
    unittest.main()
