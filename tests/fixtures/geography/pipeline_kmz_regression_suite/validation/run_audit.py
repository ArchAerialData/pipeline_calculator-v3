"""Reconstruct references, run adversarial controls, then classify fresh app results.

A passing audit means no unexpected regression; it does not mean the application
has no known defects. The ordinary comparison continues to fail on those defects.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.suite import (
    REPO, CORE_FILES, artifact_fingerprints, digest, dump, read_json,
    select_fixtures, verify_report_freshness,
)
from pipeline_kmz_regression_suite.validation.known_application_failures import (
    BASELINE_COMMIT, CLASSIFICATION_SCHEMA_VERSION, classify_fixture,
)

TEST_MODULES = ('test_geometry_reference.py', 'test_sampled_contract.py',
                'test_application_contract.py', 'test_suite_validation.py', 'test_stress_contract.py')


def classify_application(report):
    if report.get('status') != 'complete' or not report.get('export_checks_enabled') or report.get('selection'):
        raise ValueError('Audit needs a complete, full application run with exports enabled')
    rows = report.get('fixtures', [])
    if Counter(row['fixture'] for row in rows) != Counter(CORE_FILES):
        raise ValueError('Application report does not cover the exact core inventory')
    classifications = []
    for row in rows:
        fixture = SUITE / 'fixtures' / row['fixture']
        expectation = SUITE / 'expected' / f'{fixture.stem}.expected.json'
        if row['sha256'] != digest(fixture) or row.get('expectation_sha256') != digest(expectation):
            raise ValueError('Application report is stale: ' + row['fixture'])
        checks = row.get('checks', [])
        failed = [check for check in checks if check.get('passed') is not True]
        if not checks or failed != row['mismatches'] or row['passed'] != (not failed):
            raise ValueError('Inconsistent or empty application checks: ' + row['fixture'])
        expected = read_json(expectation)
        expected_scopes = {'Combined'} | {code for code, state in expected['geometry']['states'].items()
                                        if state['interior_meters'] > 0}
        if Counter(e['scope'] for e in row.get('exports', [])) != Counter(expected_scopes):
            raise ValueError('Application report is missing export scopes: ' + row['fixture'])
        classifications.append({'fixture': row['fixture'], 'check_count': len(checks),
                                'mismatch_count': len(failed), **classify_fixture(row)})
    if report['passed'] != all(row['passed'] for row in rows):
        raise ValueError('Application report has an inconsistent overall status')
    return classifications


def run_tests():
    # JUnit proves every requested module collected tests, without retaining a
    # machine-specific temporary XML document in the delivered asset package.
    with tempfile.TemporaryDirectory(prefix='kmz-audit-tests-') as tmp:
        xml_path = Path(tmp) / 'results.xml'
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                                 *(str(SUITE / 'validation' / name) for name in TEST_MODULES),
                                 f'--junitxml={xml_path}'], cwd=REPO)
        if not xml_path.exists():
            raise ValueError('Test runner did not write collection evidence')
        cases = ET.parse(xml_path).findall('.//testcase')
        counts = Counter()
        for case in cases:
            classname = case.get('classname', '').split('.')
            matches = [name for name in TEST_MODULES if Path(name).stem in classname]
            if len(matches) != 1:
                raise ValueError('Unexpected audit test collection: ' + case.get('classname', ''))
            counts[matches[0]] += 1
        if result.returncode or any(case.find(tag) is not None for case in cases
                                    for tag in ('failure', 'error', 'skipped')):
            raise ValueError('An adversarial test failed, errored or was skipped')
        if set(counts) != set(TEST_MODULES):
            raise ValueError('An audit test module collected no tests')
        return {'passed': True, 'test_count': len(cases), 'counts_by_module': dict(sorted(counts.items()))}


def run_application_comparison():
    path = SUITE / 'validation/application_comparison.json'
    # A startup/import error exits 1, just like a measured known defect. Invalidate
    # old evidence before launch so those two outcomes cannot be confused.
    dump(path, {'status': 'running', 'passed': False, 'fixtures': [], 'started_by': 'audit'})
    result = subprocess.run([sys.executable, str(SUITE / 'validation/compare_application.py')], cwd=REPO)
    application = read_json(path)
    classified = classify_application(application)
    if result.returncode != (0 if application['passed'] else 1):
        raise ValueError('Unexpected application comparison exit status')
    return application, classified


def main():
    tick = time.perf_counter()
    report_path = SUITE / 'validation/audit/audit_report.json'
    report = {'schema_version': '1.0.0', 'status': 'running', 'passed': False,
              'meaning': 'No unexpected regressions; known application defects are reported separately.',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'classification_schema_version': CLASSIFICATION_SCHEMA_VERSION,
              'known_defect_baseline_commit': BASELINE_COMMIT}
    dump(report_path, report)
    try:
        select_fixtures()
        before = artifact_fingerprints()
        application_before = {path.relative_to(REPO).as_posix(): digest(path)
                              for path in sorted((REPO / 'src').rglob('*.py'))}
        report['application_source_sha256'] = application_before
        report['adversarial_tests'] = run_tests()
        subprocess.run([sys.executable, str(SUITE / 'suite.py'), 'validate'], cwd=REPO, check=True)
        reference = read_json(SUITE / 'validation/reference_report.json')
        verify_report_freshness(reference)
        report['reference_verification'] = {
            'passed': True, 'fixture_count': len(reference['fixtures']),
            'intent_assertion_count': sum(row['coverage_assertion_count'] for row in reference['fixtures']),
            'hand_worked_control_count': len(reference['hand_worked_checks']),
            'stress_reference': reference['stress_reference'],
        }
        if reference['stress_reference']['status'] == 'reference-verified':
            stress_path = SUITE / 'validation/stress_report.json'
            dump(stress_path, {'status': 'running', 'passed': False, 'started_by': 'audit'})
            subprocess.run([sys.executable, str(SUITE / 'validation/run_stress.py'), '--validate-existing'],
                           cwd=REPO, check=True)
            stress = read_json(stress_path)
            if (stress.get('status') != 'complete' or not stress.get('passed') or
                    stress.get('sha256') != digest(SUITE / 'fixtures/stress/stress_branching_network.kmz') or
                    stress.get('expectation_sha256') != digest(SUITE / 'expected/stress/stress_branching_network.expected.json') or
                    not stress.get('checks') or stress.get('failures') or
                    any(row.get('passed') is not True for row in stress['checks'])):
                raise ValueError('Stress measurement is missing, stale or failed')
            report['stress_application'] = {
                'passed': True, 'source_count': stress['source_count'],
                'sample_count': stress['independent_sample_count'], 'check_count': len(stress['checks']),
                'application_runtime_seconds': stress['application_runtime_seconds'],
                'report_sha256': digest(stress_path),
            }
        else:
            report['stress_application'] = {'status': 'not_generated', 'passed': True}
        application, classified = run_application_comparison()
        unexpected = [row for row in classified if row['status'] == 'unexpected_failure']
        report['application'] = {
            'fully_passing': application['passed'], 'fixtures': classified,
            'check_count': sum(row['check_count'] for row in classified),
            'known_defect_ids': sorted({row['defect_id'] for row in classified if row['status'] == 'known_failure'}),
            'known_failure_fixture_count': sum(row['status'] == 'known_failure' for row in classified),
            'unexpected_failure_fixture_count': len(unexpected),
        }
        report['artifacts_unchanged'] = before == artifact_fingerprints()
        report['application_sources_unchanged'] = application_before == {
            path.relative_to(REPO).as_posix(): digest(path) for path in sorted((REPO / 'src').rglob('*.py'))}
        report['implementation_sha256'] = {path.relative_to(SUITE).as_posix(): digest(path)
                                           for path in sorted(SUITE.rglob('*.py'))}
        report['evidence_sha256'] = {name: digest(SUITE / 'validation' / name)
                                     for name in ('reference_report.json', 'application_comparison.json')}
        report.update(status='complete', passed=not unexpected and report['artifacts_unchanged']
                      and report['application_sources_unchanged'], elapsed_seconds=time.perf_counter()-tick)
        subprocess.run([sys.executable, str(SUITE / 'validation/render_reports.py')], cwd=REPO, check=True)
        dump(report_path, report)
        if not report['passed']:
            raise ValueError('Unexpected regression or file mutation; inspect the audit report')
        print(f"Audit passed: {report['adversarial_tests']['test_count']} adversarial tests, "
              f"{report['application']['check_count']} application checks; "
              f"{report['stress_application'].get('check_count', 0)} stress checks; "
              f"{len(report['application']['known_defect_ids'])} known production defects remain.")
    except Exception as error:
        report.update(status='failed', passed=False, error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.perf_counter()-tick)
        dump(report_path, report)
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
