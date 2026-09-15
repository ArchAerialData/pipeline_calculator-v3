"""Validate or generate the bounded branching-overlap stress profile offline.

--validate-existing remeasures frozen expectations without replacing artifacts.
Generation is staged and independently verified before publication. Exact sample
and partition checks run after, and outside, the timed production analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.suite import (
    REPO, BOUNDARIES, accounting_checks, contract_provenance, digest, dump,
    environment, publish_generated, read_json, validate_stress_reference, xml_inventory,
)
from pipeline_kmz_regression_suite.reference.geometry import read_kmz, GeometryReference
from pipeline_kmz_regression_suite.reference.sampled import analyze
from pipeline_kmz_regression_suite.validation.application_contract import (
    compare_interval_ledger, compare_sampled_contract,
)
import psutil

FIXTURE = Path('fixtures/stress/stress_branching_network.kmz')
EXPECTATION = Path('expected/stress/stress_branching_network.expected.json')
REPORT = Path('validation/stress_report.json')


def workload_valid(counters, sample_count, limits):
    """A missing profiler result must never masquerade as a measured pass."""
    return bool(counters) and all(
        type(c.get('segment_count')) is int and c['segment_count'] == sample_count and
        all(type(c.get(name)) is int and 0 < c[name] <= limits[name]
            for name in ('neighbor_visits', 'candidate_checks')) and
        c['segment_count'] <= limits['segment_count'] for c in counters)


def validate_stress_design(reference, sources):
    """Prove this profile's single-state and positive-overlap premises."""
    groups = reference['groups']
    geometric, sampled = reference['geometry'], reference['sampled']
    if type(groups) is not int or not 2 <= groups <= 64 or len(sources) != groups * 4:
        raise ValueError('Stress requires 2..64 groups with exactly four sources per group')
    accounting_checks(geometric)
    if (set(geometric['states']) != {'TX'} or geometric['crossing_count'] != 0 or
            any(geometric[name] != 0.0 for name in ('outside_meters', 'shared_meters', 'unresolved_meters')) or
            any(row['kind'] != 'interior' or row['state_codes'] != ['TX']
                for row in geometric['intervals'])):
        raise ValueError('Stress geometry must be exclusively inside TX with no crossings or unowned mileage')
    # Equal totals alone cannot establish identical sampling phases.
    def canonical_paths(rows):
        return {s['key']: [list(map(tuple, path)) for path in s['paths']] for s in rows}

    original_paths = canonical_paths(sources)
    state_paths = canonical_paths(geometric['state_inputs']['TX'])
    if state_paths != original_paths:
        raise ValueError('Stress TX paths differ from complete paths; state sampling needs a separate expectation')
    motifs = Counter(s['motif'] for s in sources)
    if (len(motifs) != groups or any(count != 4 for count in motifs.values()) or
            sampled['status'] != 'complete' or sampled['diagnostics'] or
            sampled['source_count'] != len(sources) or sampled['sample_count'] <= 0 or
            sampled['qualifying_section_count'] <= 0 or sampled['savings_meters'] <= 0 or
            any(sampled['motif_savings'].get(motif, 0) <= 0 for motif in motifs)):
        raise ValueError('Stress must exercise positive measured overlap in every four-source group')
    return {'passed': True, 'exclusive_state': 'TX', 'group_count': groups,
            'state_inputs_equal_complete_paths': True, 'positive_savings_in_every_group': True}


def build_reference(path, groups):
    """Create independent data only; this function imports no application code."""
    sources = read_kmz(path)
    reference = {'schema_version': 'stress/1.0.0', 'profile': 'branching-overlap-stress',
                 'groups': groups, 'archive': xml_inventory(path, sources),
                 'geometry': GeometryReference(BOUNDARIES).analyze(sources),
                 'sampled': analyze(sources)}
    validate_stress_design(reference, sources)
    return reference


def prepare_reference(groups=None, *, validate_existing=False, root=SUITE):
    """Return an independently verified pair; publish new files only on success."""
    root = Path(root)
    provenance = contract_provenance()
    if validate_existing:
        verification = validate_stress_reference(root)
        if verification.get('status') != 'reference-verified' or verification.get('passed') is not True:
            raise ValueError('--validate-existing requires a complete frozen stress KMZ and expectation')
        reference = read_json(root / EXPECTATION)
        if groups is not None and groups != reference['groups']:
            raise ValueError('--groups does not match the existing frozen stress profile')
        sources = read_kmz(root / FIXTURE)
        design = validate_stress_design(reference, sources)
    else:
        groups = 24 if groups is None else groups
        if type(groups) is not int or not 2 <= groups <= 64:
            raise ValueError('Successful stress profile supports 2..64 groups')
        if not provenance['baseline_contract_unchanged']:
            raise ValueError('Cannot generate stress expectations against a changed calculation contract: ' +
                             ', '.join(provenance['changed_contract_files']))
        # Failed generation/reference checks leave the delivered pair intact.
        with tempfile.TemporaryDirectory(prefix='pipeline-kmz-stress-', dir=root.parent) as temporary:
            staged = Path(temporary)
            subprocess.run([sys.executable, str(SUITE / 'generator/generate.py'), '--profile', 'stress',
                            '--stress-groups', str(groups), '--output', str(staged)], check=True)
            reference = build_reference(staged / FIXTURE, groups)
            dump(staged / EXPECTATION, reference)
            verification = validate_stress_reference(staged)
            if verification.get('status') != 'reference-verified' or verification.get('passed') is not True:
                raise ValueError('Staged stress reference was not independently verified')
            publish_generated(staged, root)
        sources = read_kmz(root / FIXTURE)
        design = validate_stress_design(reference, sources)
    return reference, sources, {'reference_verification': verification, 'reference_design': design,
                                'contract_provenance': provenance}


def _safe_value(value):
    """Retain valid JSON failure receipts even if broken code produces NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return value


def _equal(actual, target, tolerance=0):
    if type(target) is bool or type(actual) is bool:
        return type(actual) is type(target) and actual == target
    if isinstance(target, (int, float)):
        return (isinstance(actual, (int, float)) and math.isfinite(actual) and
                math.isfinite(target) and abs(actual - target) <= tolerance)
    if isinstance(target, dict):
        return (isinstance(actual, dict) and actual.keys() == target.keys() and
                all(_equal(actual[key], target[key], tolerance) for key in target))
    if isinstance(target, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(target) and
                all(_equal(a, b, tolerance) for a, b in zip(actual, target)))
    return actual == target


def compare_result(result, path, reference, sources, report):
    """Retain exact Combined/TX coverage evidence after the timed app run."""
    sys.path.insert(0, str(REPO / 'src'))
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core.bundling import qualifying_sections
    from pipeline_calculator.parsers.kml_kmz import extract_features_from_file_with_diagnostics

    checks = report.setdefault('checks', [])
    failures = report.setdefault('failures', [])

    def check(label, actual, target, tolerance=0):
        row = {'assertion': label, 'actual': _safe_value(actual), 'expected': _safe_value(target),
               'tolerance': tolerance, 'passed': _equal(actual, target, tolerance)}
        checks.append(row)
        if not row['passed']:
            failures.append(row)

    sampled, geometry = reference['sampled'], reference['geometry']
    geography = result.get('geography', {})
    public = result.get('overlap_analysis') or {}
    check('Combined analysis complete', result.get('analysis_complete'), True)
    check('Geography analysis complete', geography.get('analysis_complete'), True)
    check('Geography status', geography.get('status'), 'complete')
    check('Geography reconciliation', geography.get('reconciliation', {}).get('passed'), True)
    check('Combined original', result.get('total_meters'), geometry['original_meters'], .001)
    check('Combined savings', public.get('savings_meters'), sampled['savings_meters'], 1e-7)
    check('Combined adjusted', public.get('effective_total_meters'), sampled['adjusted_meters'], .001)
    check('Combined diagnostics', dict(Counter(d['code'] for d in result.get('diagnostics', []))),
          {'selected_primary_kml': 1})
    check('Geography diagnostics', geography.get('diagnostics'), [])
    check('Crossing events', geography.get('crossing_count'), 0)
    for name in ('outside_meters', 'unresolved_meters', 'shared_meters'):
        check(name, geography.get('reconciliation', {}).get(name), 0.0)
    check('Frozen default analysis profile', result.get('analysis_parameters'),
          {'detection_range': 15.0, 'min_parallel_length': 200.0,
           'segment_length': 5.0, 'angular_tolerance': 15.0})
    parsed = extract_features_from_file_with_diagnostics(path)
    xml_to_key = dict(zip(reference['archive']['xml_ids'], reference['archive']['xml_source_order']))
    id_to_key = {p['id']: xml_to_key.get(p['placemark_id'], '<unknown>') for p in parsed.pipelines}
    originals = {s['key']: s for s in sources}
    check('Unique source identities', len(id_to_key), len(originals))
    check('Public source identities', sorted(xml_to_key.get(p['Placemark_ID'], '<unknown>') for p in result['pipelines']),
          sorted(originals))
    for pipeline in parsed.pipelines:
        key = id_to_key[pipeline['id']]
        check(f'{key}: parser paths and vertices', pipeline['coordinate_paths'], originals.get(key, {}).get('paths'))
    compare_interval_ledger(check, 'Partition', geography.get('fragments', []), geometry['intervals'],
                            id_to_key, originals=originals)

    def scope(label, inputs, overlap):
        analyzer = PipelineAnalyzer()
        matches = analyzer.find_parallel_segments(inputs)
        sections = qualifying_sections(inputs, matches, analyzer.segment_length, analyzer.min_parallel_length)
        compare_sampled_contract(check, label, inputs, sections, overlap, sampled, id_to_key)

    scope('Combined', parsed.pipelines, public)
    states = geography.get('states', [])
    check('Represented states', [s['state_code'] for s in states], ['TX'])
    for state in states:
        code = state['state_code']
        check(f'{code}: analysis complete', state.get('analysis_complete'), True)
        check(f'{code}: interior', state.get('interior_meters'), geometry['original_meters'], .001)
        check(f'{code}: shared allocation', state.get('shared_allocation_meters'), 0.0)
        check(f'{code}: attributed original', state.get('total_meters'), geometry['original_meters'], .001)
        check(f'{code}: savings', state.get('interior_savings_meters'), sampled['savings_meters'], 1e-7)
        check(f'{code}: adjusted', state.get('adjusted_total_meters'), sampled['adjusted_meters'], .001)
        check(f'{code}: diagnostics', dict(Counter(d['code'] for d in state.get('diagnostics', []))),
              {'selected_primary_kml': 1})
        inputs = []
        for pipeline in parsed.pipelines:
            paths = [fragment['coordinates'] for fragment in geography.get('fragments', [])
                     if fragment['source_id'] == pipeline['id'] and fragment['kind'] == 'state' and
                     fragment['state_codes'] == [code]]
            if paths:
                inputs.append(dict(pipeline, coordinates=paths[0], coordinate_paths=paths))
        scope(code, inputs, state.get('overlap_analysis'))
    report['state_analysis_checked'] = bool(states) and not any(
        row['assertion'].startswith('TX') or row['assertion'] == 'Represented states' for row in failures)
    report['sample_contract_checked'] = not any('sample' in row['assertion'] or 'section' in row['assertion']
                                               for row in failures)


def run_profile(groups=None, *, validate_existing=False, root=SUITE, report=None):
    root = Path(root)
    report = {} if report is None else report
    tick = time.perf_counter()
    reference, sources, verification = prepare_reference(groups, validate_existing=validate_existing, root=root)
    report.update(verification, reference_runtime_seconds=time.perf_counter()-tick)
    path = root / FIXTURE
    geometric, sampled = reference['geometry'], reference['sampled']
    sys.path.insert(0, str(REPO / 'src'))
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core.options import AnalysisOptions
    from pipeline_calculator.core import overlap
    counters = []

    def capture(frame, event, arg):
        if event == 'return' and frame.f_code is overlap.find_parallel_segments.__code__:
            data = frame.f_locals
            counters.append({name: data.get(name) for name in ('segment_count', 'neighbor_visits', 'candidate_checks')})

    report.update(environment=environment(), fixture=FIXTURE.as_posix(), sha256=digest(path),
                  expectation_sha256=digest(root / EXPECTATION), groups=reference['groups'],
                  source_count=len(sources), actual_overlap_pass_work=counters)
    tick = time.perf_counter()
    previous_profiler = sys.getprofile()
    sys.setprofile(capture)
    try:
        result = PipelineAnalyzer().analyze_complete(path, options=AnalysisOptions(True))
    finally:
        sys.setprofile(previous_profiler)
        report['application_runtime_seconds'] = time.perf_counter()-tick
    limits = {'segment_count': overlap.MAX_ANALYSIS_SEGMENTS,
              'neighbor_visits': overlap.MAX_NEIGHBOR_VISITS, 'candidate_checks': overlap.MAX_CANDIDATE_CHECKS}
    workload_passed = workload_valid(counters, sampled['sample_count'], limits)
    public = result.get('overlap_analysis') or {}
    report.update(
        measurement_note='Application time includes sys.setprofile return-counter instrumentation and map preparation; excludes independent reference validation and post-run sample checks. Memory is process-wide.',
        memory_bytes=psutil.Process().memory_info()._asdict(), physical_memory_bytes=psutil.virtual_memory().total,
        logical_cpu_count=psutil.cpu_count(), physical_cpu_count=psutil.cpu_count(logical=False),
        limits={'segments': limits['segment_count'], 'neighbor_visits': limits['neighbor_visits'],
                'candidate_checks': limits['candidate_checks']},
        independent_original_meters=geometric['original_meters'], independent_sample_count=sampled['sample_count'],
        independent_qualifying_section_count=sampled['qualifying_section_count'],
        independent_savings_meters=sampled['savings_meters'], application_original_meters=result.get('total_meters'),
        application_savings_meters=public.get('savings_meters'),
        application_complete=result.get('analysis_complete') and result.get('geography', {}).get('analysis_complete'),
        workload_measurement_checked=workload_passed,
        diagnostics=result.get('diagnostics', []) + result.get('geography', {}).get('diagnostics', []))
    tick = time.perf_counter()
    compare_result(result, path, reference, sources, report)
    report['sample_and_partition_verification_seconds'] = time.perf_counter()-tick
    workload_check = {'assertion': 'Measured neighbor and sample work within limits', 'passed': workload_passed,
                      'actual': counters, 'expected': limits}
    report['checks'].append(workload_check)
    if not workload_passed:
        report['failures'].append(workload_check)
    report.update(passed=not report['failures'], status='complete')
    dump(root / REPORT, _safe_value(report))
    print(f'Stress passed={report["passed"]}: {len(sources)} sources, {sampled["sample_count"]} samples, '
          f'{report["application_runtime_seconds"]:.2f}s application; {len(report["checks"])} checks')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', type=int, default=None, help='Generate 2..64 groups (default 24); with --validate-existing, require this group count.')
    parser.add_argument('--validate-existing', action='store_true', help='Remeasure the frozen stress pair without replacing either artifact.')
    args = parser.parse_args(argv)
    if args.groups is not None and not 2 <= args.groups <= 64:
        parser.error('Successful stress profile supports 2..64 groups; limit rejection is a separate test.')
    report = {'status': 'running', 'passed': False, 'groups': args.groups,
              'mode': 'validate-existing' if args.validate_existing else 'generate-and-validate'}
    dump(SUITE / REPORT, report)
    try:
        run_profile(args.groups, validate_existing=args.validate_existing, root=SUITE, report=report)
        if not report['passed']:
            raise ValueError('Stress comparison failed; see validation/stress_report.json')
    except Exception as error:
        # Preserve completed checks, workload measurements and hashes.
        report.update(status='failed', passed=False, error=str(error), error_type=type(error).__name__)
        dump(SUITE / REPORT, _safe_value(report))
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()

