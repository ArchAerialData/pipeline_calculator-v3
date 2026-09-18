"""Offline reference creation and verification, deliberately independent of the app.

Application observation scripts live in validation/. Expectations are fixed before
those comparisons and are never changed to match application observations.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET

if not __debug__:
    raise RuntimeError('The fixture validator requires assertions; run Python without -O or PYTHONOPTIMIZE.')

SUITE = Path(__file__).resolve().parent
if str(SUITE.parent) not in sys.path:
    sys.path.insert(0, str(SUITE.parent))
REPO = SUITE.parents[3]
BOUNDARIES = REPO / 'src/pipeline_calculator/data/states_2025.zip'
BASELINE = '71da499d5756648ae395660f0a241ea00edbea4f'
BOUNDARY_SHA256 = '55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539'
MILE = 1609.347218694
PROFILE = {'id': 'default-grs80-5m-v1', 'ellipsoid': 'GRS80',
           'survey_mile_meters': MILE, 'segment_length_meters': 5.0,
           'detection_range_meters': 15.0, 'minimum_parallel_length_meters': 200.0,
           'angular_tolerance_degrees': 15.0}
CONTRACT_FILES = ['constants.py', 'segmentation.py', 'coordinates.py', 'angles.py',
                  'overlap.py', 'bundling.py', 'analyzer.py', 'state_analysis.py',
                  'geography/boundaries.py', 'geography/partition.py']
MAIN_STEMS = ('01_three_state_disconnected_networks', '02_three_state_transverse_crossings',
              '03_parallel_corridors_crossing_borders', '04_shared_border_and_near_border')
CORE_FILES = tuple(f'{name}.kmz' for name in MAIN_STEMS) + (
    f'variants/{MAIN_STEMS[0]}__source_order.kmz',
    f'variants/{MAIN_STEMS[3]}__redundant_vertices.kmz',
    f'variants/{MAIN_STEMS[3]}__reversed.kmz')
LEDGER_SUFFIXES = ('expected.json', 'intervals.csv', 'overlaps.csv', 'crossings.csv')
PREVIEW_FILES = tuple(f'{name}.png' for name in MAIN_STEMS) + (
    'details/01_control_details.png', 'details/03_control_details.png', 'details/04_control_details.png')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                     dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number in {path}: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def select_fixtures(root=SUITE, pattern=None, *, require_expectations=True):
    """Use a fixed, reviewed inventory; an empty glob can never certify a suite."""
    root = Path(root)
    actual = {p.relative_to(root / 'fixtures').as_posix()
              for p in (root / 'fixtures').rglob('*.kmz') if p.relative_to(root / 'fixtures').parts[0] != 'stress'}
    required = set(CORE_FILES)
    if actual != required:
        raise ValueError(f'Core archive inventory mismatch: missing={sorted(required-actual)}, unexpected={sorted(actual-required)}')
    if require_expectations:
        actual_expected = {p.relative_to(root / 'expected').as_posix() for p in (root / 'expected').rglob('*.expected.json')
                           if p.relative_to(root / 'expected').parts[0] != 'stress'}
        expected_names = {Path(name).stem + '.expected.json' for name in required}
        if actual_expected != expected_names:
            raise ValueError(f'Expectation inventory mismatch: missing={sorted(expected_names-actual_expected)}, unexpected={sorted(actual_expected-expected_names)}')
        for name in required:
            for suffix in LEDGER_SUFFIXES:
                path = root / 'expected' / f'{Path(name).stem}.{suffix}'
                if not path.is_file():
                    raise ValueError(f'Missing required ledger: {path.name}')
    selected = [root / 'fixtures' / name for name in CORE_FILES
                if pattern is None or Path(name).stem.startswith(pattern)]
    if not selected:
        raise ValueError(f'No fixture matches --only {pattern!r}')
    return selected


def contract_provenance(repository=REPO):
    """Permit unrelated later commits, but never silently change the contract."""
    files = [f'src/pipeline_calculator/core/{name}' for name in CONTRACT_FILES]
    files += ['src/pipeline_calculator/parsers/kml_kmz.py']
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASELINE, '--', *files],
                                     cwd=repository, text=True).splitlines()
    return {'contract_baseline_commit': BASELINE, 'working_commit': current,
            'changed_contract_files': changed, 'baseline_contract_unchanged': not changed}


def artifact_fingerprints(root=SUITE):
    paths = [Path('fixtures') / name for name in CORE_FILES]
    paths += [Path('expected') / f'{Path(name).stem}.{suffix}'
              for name in CORE_FILES for suffix in LEDGER_SUFFIXES]
    paths += [Path('validation/design_manifest.json')]
    paths += [Path('previews') / name for name in PREVIEW_FILES]
    for path in (Path('fixtures/stress/stress_branching_network.kmz'),
                 Path('expected/stress/stress_branching_network.expected.json')):
        if (Path(root) / path).exists():
            paths.append(path)
    return {path.as_posix(): digest(Path(root) / path) for path in sorted(paths)}


def verify_report_freshness(report, root=SUITE):
    """Receipts are tied to actual archives and ledgers, not just a PASS word."""
    if report.get('status') != 'complete' or not report.get('passed'):
        raise ValueError('No complete passing reference receipt is available')
    rows = report.get('fixtures', [])
    if (report.get('selection') != 'all_core' or report.get('errors') or
            Counter(row.get('fixture') for row in rows) != Counter(CORE_FILES) or
            any(row.get('reference_status') != 'reference-verified' or row.get('errors') for row in rows)):
        raise ValueError('Reference receipt does not cover every completed core fixture')
    if report.get('artifact_sha256') != artifact_fingerprints(root):
        raise ValueError('Reference receipt is stale: an archive, expectation, ledger or design changed')
    if report.get('independent_module_audit') != audit_independence():
        raise ValueError('Reference receipt is stale: generator, reference or intent-check code changed')
    if report.get('validator_sha256') != digest(SUITE / 'suite.py'):
        raise ValueError('Reference receipt is stale: orchestration or accounting code changed')


def validate_previews(root):
    from PIL import Image
    for name in PREVIEW_FILES:
        preview = Path(root) / 'previews' / name
        with Image.open(preview) as picture:
            if picture.format != 'PNG' or min(picture.size) <= 0:
                raise ValueError(f'Invalid preview PNG: {name}')
            picture.verify()
        with Image.open(preview) as picture:
            picture.load()


def environment():
    return {'python': sys.version, 'platform': platform.platform(),
            'processor': platform.processor(),
            'dependencies': {p: importlib.metadata.version(p) for p in
                ('numpy', 'scipy', 'shapely', 'pyproj', 'geographiclib', 'matplotlib', 'Pillow', 'psutil', 'pytest')}}


def audit_independence():
    """Mechanical backstop to the separately documented algorithm provenance."""
    audited = []
    for folder in ('reference', 'generator', 'validation'):
        candidates = sorted((SUITE / folder).glob('*.py'))
        if folder == 'validation':
            candidates = [p for p in candidates if p.name == 'coverage_checks.py']
        for path in candidates:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                assert not any('pipeline_calculator' in module for module in modules), path
            audited.append({'file': path.relative_to(SUITE).as_posix(), 'sha256': digest(path)})
    return audited


def accounting_checks(geo):
    intervals = geo['intervals']
    original = geo['combined_original_meters']
    for source in geo['sources']:
        rows = [i for i in intervals if i['key'] == source['key']]
        actual = math.fsum(i['length_meters'] for i in rows)
        assert abs(actual - source['original_meters']) <= max(.001, source['original_meters'] * 1e-10)
        for path in source['paths']:
            parts = sorted([i for i in rows if i['path_index'] == path['path_index']], key=lambda i: i['start_m'])
            if not parts:
                assert path['original_meters'] == 0
                continue
            assert abs(parts[0]['start_m']) < 1e-8
            assert abs(parts[-1]['end_m'] - path['original_meters']) < .001
            for a, b in zip(parts, parts[1:]):
                assert abs(a['end_m'] - b['start_m']) < 1e-8
    for row in intervals:
        assert row['length_meters'] > 0
        assert row['kind'] in ('interior', 'shared', 'outside', 'unresolved')
        assert row['cut_error_bound_meters'] <= .01
    outside = math.fsum(i['length_meters'] for i in intervals if i['kind'] == 'outside')
    unresolved = math.fsum(i['length_meters'] for i in intervals if i['kind'] == 'unresolved')
    assert outside == unresolved == 0, 'Main fixtures require fully resolved state coverage'
    attributed = math.fsum(s['attributed_original_meters'] for s in geo['states'].values())
    assert abs(attributed + outside + unresolved - original) <= max(.001, original * 1e-10)
    for state in geo['states'].values():
        assert abs(state['interior_meters'] + state['shared_allocation_meters'] - state['attributed_original_meters']) < 1e-8


def xml_inventory(path, sources):
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ['doc.kml'], 'Exactly one doc.kml is required'
        root = ET.fromstring(z.read('doc.kml'))
        info = z.getinfo('doc.kml')
    counts = {}
    for node in root.iter():
        name = node.tag.rsplit('}', 1)[-1]
        counts[name] = counts.get(name, 0) + 1
    assert not any(counts.get(t, 0) for t in ('Polygon', 'NetworkLink', 'Point', 'Track'))
    ids = [n.attrib['id'] for n in root.iter() if n.tag.endswith('}Placemark')]
    assert len(ids) == len(set(ids)) == len(sources)
    keys = [s['key'] for s in sources]
    assert len(keys) == len(set(keys))
    return {'sha256': digest(path), 'bytes': path.stat().st_size,
            'xml_counts': counts, 'xml_source_order': keys, 'xml_ids': ids,
            'source_count': len(sources), 'path_count': sum(len(s['paths']) for s in sources),
            'vertex_count': sum(len(p) for s in sources for p in s['paths']),
            'repeated_display_names': {k: n for k, n in Counter(s['name'] for s in sources).items() if n > 1},
            'repeated_objectids': {k: n for k, n in Counter(s.get('objectid', '') for s in sources).items() if n > 1},
            'zip_entry': {'name': info.filename, 'date_time': list(info.date_time),
                          'compression': info.compress_type, 'external_attr': info.external_attr}}


def state_inputs(sources, geometry, code):
    lookup = geometry['interior_fragments'].get(code, {})
    result = []
    for source in sources:
        fragments = lookup.get(source['key'], [])
        if fragments:
            result.append({**source, 'paths': [f['coordinates'] for f in fragments],
                           'fragment_references': [{k: v for k, v in f.items() if k != 'coordinates'}
                                                   for f in fragments]})
    return result


def expectation(path, reference, *, root=SUITE):
    from pipeline_kmz_regression_suite.reference.geometry import read_kmz
    from pipeline_kmz_regression_suite.reference.sampled import analyze
    sources = read_kmz(path)
    inventory = xml_inventory(path, sources)
    geo = reference.analyze(sources)
    accounting_checks(geo)
    combined = analyze(sources)
    original = geo['combined_original_meters']
    assert abs(combined['original_meters'] - original) < max(.001, original * 1e-10)
    analyses = {'Combined': combined}
    for code, accounting in sorted(geo['states'].items()):
        inputs = state_inputs(sources, geo, code)
        result = analyze(inputs)
        interior = accounting['interior_meters']
        assert abs(result['original_meters'] - interior) < max(.001, interior * 1e-10)
        attributed = accounting['attributed_original_meters']
        result.update(interior_meters=interior,
                      shared_allocation_meters=accounting['shared_allocation_meters'],
                      attributed_original_meters=attributed,
                      attributed_original_survey_miles=attributed / MILE,
                      adjusted_meters=attributed - result['savings_meters'],
                      adjusted_survey_miles=(attributed - result['savings_meters']) / MILE,
                      fragment_references={s['key']: s['fragment_references'] for s in inputs})
        analyses[code] = result
    value = {'schema_version': '1.0.0', 'fixture': path.relative_to(Path(root) / 'fixtures').as_posix(),
            'baseline_commit': BASELINE, 'boundary_sha256': BOUNDARY_SHA256,
            'analysis_profile': PROFILE, 'archive': inventory,
            'geometry': geo, 'analyses': analyses,
            'expected_application_status': {'analysis_complete': True, 'geography_status': 'complete',
                'unresolved_meters': 0.0, 'outside_meters': 0.0,
                'combined_diagnostic_counts': {'selected_primary_kml': 1},
                'geography_diagnostic_counts': {}, 'unexpected_warning_or_error_is_failure': True},
            'reference_status': 'reference-verified'}
    from pipeline_kmz_regression_suite.validation.coverage_checks import check_fixture
    value['coverage_checks'] = check_fixture(value, root=root)
    failed = [c for c in value['coverage_checks'] if not c['passed']]
    if failed:
        raise ValueError('Fixture intent failed; no golden written: ' + json.dumps(failed, indent=2))
    return value


def compare_values(expected, actual, path='$'):
    """Deep reproducibility comparison; sample counts/savings have no 5m slack."""
    errors = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            errors.append(f'{path}: key sets differ')
        for key in sorted(expected.keys() & actual.keys()):
            errors.extend(compare_values(expected[key], actual[key], f'{path}.{key}'))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f'{path}: list length {len(expected)} != {len(actual)}')
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                errors.extend(compare_values(a, b, f'{path}[{i}]'))
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            errors.append(f'{path}: Boolean/type mismatch {expected!r} != {actual!r}')
    elif isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        tolerance = 1e-7 if isinstance(expected, float) else 0
        if not math.isfinite(expected) or not math.isfinite(actual):
            errors.append(f'{path}: nonfinite numbers are invalid')
        elif isinstance(expected, int) and not isinstance(actual, int):
            errors.append(f'{path}: count/type mismatch {expected!r} != {actual!r}')
        elif abs(expected - actual) > tolerance:
            errors.append(f'{path}: {expected} != {actual} (tolerance {tolerance})')
    elif expected != actual:
        errors.append(f'{path}: {expected!r} != {actual!r}')
    return errors


def csv_contents(value):
    intervals = value['geometry']['intervals']
    columns = ['id', 'key', 'path_index', 'start_m', 'end_m', 'kind', 'state_codes',
               'length_meters', 'start_error_bound_meters', 'end_error_bound_meters', 'cut_error_bound_meters']
    interval_text, overlap_text, crossing_text = io.StringIO(newline=''), io.StringIO(newline=''), io.StringIO(newline='')
    writer = csv.DictWriter(interval_text, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for row in intervals:
        writer.writerow({**row, 'state_codes': '|'.join(row['state_codes'])})
    columns = ['scope', 'source_1', 'source_2', 'path_1', 'path_2', 'qualified', 'length_meters',
               'sample_count_1', 'sample_count_2', 'ranges_1', 'ranges_2', 'eligible_match_count',
               'minimum_transverse_meters', 'maximum_transverse_meters', 'maximum_longitudinal_meters']
    writer = csv.DictWriter(overlap_text, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for scope, result in sorted(value['analyses'].items()):
        for s in result['sections']:
            writer.writerow({**s, 'scope': scope, 'source_1': s['source_keys'][0], 'source_2': s['source_keys'][1],
                'path_1': s['path_indices'][0], 'path_2': s['path_indices'][1],
                'sample_count_1': s['coverage_sample_counts'][0], 'sample_count_2': s['coverage_sample_counts'][1],
                'ranges_1': json.dumps(s['coverage_ranges'][0]), 'ranges_2': json.dumps(s['coverage_ranges'][1])})
    columns = ['source_key', 'path_index', 'chainage_meters', 'longitude', 'latitude',
               'from_states', 'to_states', 'angle_degrees', 'via_shared_interval', 'cut_error_bound_meters']
    writer = csv.DictWriter(crossing_text, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for row in value['geometry']['crossings']:
        writer.writerow({**row, 'longitude': row['coordinate'][0], 'latitude': row['coordinate'][1],
                         'from_states': '|'.join(row['from_states']), 'to_states': '|'.join(row['to_states'])})
    return {'intervals.csv': interval_text.getvalue(), 'overlaps.csv': overlap_text.getvalue(),
            'crossings.csv': crossing_text.getvalue()}


def deterministic_archive_check(root=SUITE):
    with tempfile.TemporaryDirectory(prefix='pipeline-kmz-determinism-') as tmp:
        subprocess.run([sys.executable, str(SUITE / 'generator/generate.py'), '--output', tmp, '--no-previews'],
                       check=True, capture_output=True, text=True)
        checks = []
        generated = select_fixtures(Path(tmp), require_expectations=False)
        for path in generated:
            relative = path.relative_to(tmp)
            actual, expected = digest(path), digest(Path(root) / relative)
            checks.append({'file': relative.as_posix(), 'regenerated_sha256': actual,
                           'saved_sha256': expected, 'passed': actual == expected})
        assert all(c['passed'] for c in checks), 'Generator no longer recreates saved archives'
        return checks


def variant_checks(expectations):
    by_name = {Path(e['fixture']).stem: e for e in expectations}
    result = []
    for name, variant in by_name.items():
        if '__' not in name:
            continue
        base_name, mode = name.split('__', 1)
        if base_name not in by_name:
            continue
        base = by_name[base_name]
        a = {s['key']: s for s in base['geometry']['sources']}
        b = {s['key']: s for s in variant['geometry']['sources']}
        assert a.keys() == b.keys()
        errors = []
        max_delta = 0.0
        for key in a:
            delta = abs(a[key]['original_meters'] - b[key]['original_meters'])
            max_delta = max(max_delta, delta)
            if delta > .001:
                errors.append(f'{key}: original changed {delta} m')
            if a[key]['states'].keys() != b[key]['states'].keys():
                errors.append(f'{key}: represented states changed')
            for code in a[key]['states'].keys() & b[key]['states'].keys():
                for field in ('interior_meters', 'shared_allocation_meters', 'attributed_original_meters'):
                    delta = abs(a[key]['states'][code][field] - b[key]['states'][code][field])
                    max_delta = max(max_delta, delta)
                    if delta > .001:
                        errors.append(f'{key}/{code}: {field} changed {delta} m')
        saving_deltas = {scope: variant['analyses'][scope]['savings_meters'] - base['analyses'][scope]['savings_meters']
                         for scope in base['analyses']}
        if mode == 'source_order' and any(saving_deltas.values()):
            errors.append('Source-order permutation changed savings')
        result.append({'fixture': variant['fixture'], 'base_fixture': base['fixture'], 'variation': mode,
                       'maximum_source_length_or_allocation_difference_meters': max_delta,
                       'savings_difference_meters_by_scope': saving_deltas,
                       'savings_invariance_required': mode == 'source_order',
                       'passed': not errors, 'errors': errors})
    assert all(r['passed'] for r in result), result
    return result


def validate_stress_reference(root=SUITE):
    from pipeline_kmz_regression_suite.reference.geometry import read_kmz, GeometryReference
    from pipeline_kmz_regression_suite.reference.sampled import analyze
    root = Path(root)
    fixture = root / 'fixtures/stress/stress_branching_network.kmz'
    expected_path = root / 'expected/stress/stress_branching_network.expected.json'
    files = set((root / 'fixtures/stress').rglob('*.kmz'))
    if files - {fixture}:
        raise ValueError('Unexpected stress archive; every delivered stress input needs an explicit reference')
    if set((root / 'expected/stress').rglob('*.expected.json')) - {expected_path}:
        raise ValueError('Unexpected stress expectation without a reviewed input')
    if not fixture.exists() and not expected_path.exists():
        return {'status': 'not_generated', 'passed': True}
    if not fixture.exists() or not expected_path.exists():
        raise ValueError('Stress input and independent expectation must both be present')
    expected = read_json(expected_path)
    sources = read_kmz(fixture)
    if type(expected['groups']) is not int or not 2 <= expected['groups'] <= 64 or len(sources) != expected['groups'] * 4:
        raise ValueError('Invalid or inconsistent stress group/source count')
    actual = {'schema_version': 'stress/1.0.0', 'profile': 'branching-overlap-stress',
              'groups': expected['groups'], 'archive': xml_inventory(fixture, sources),
              'geometry': GeometryReference(BOUNDARIES).analyze(sources), 'sampled': analyze(sources)}
    errors = compare_values(expected, actual)
    if errors:
        raise ValueError('Stress expectation changed: ' + '\n'.join(errors[:10]))
    with tempfile.TemporaryDirectory(prefix='pipeline-kmz-stress-determinism-') as tmp:
        subprocess.run([sys.executable, str(SUITE / 'generator/generate.py'), '--output', tmp,
                        '--profile', 'stress', '--stress-groups', str(expected['groups'])],
                       check=True, capture_output=True, text=True)
        if digest(Path(tmp) / fixture.relative_to(root)) != digest(fixture):
            raise ValueError('Generator no longer recreates saved stress archive')
    return {'status': 'reference-verified', 'passed': True, 'sha256': digest(fixture),
            'expectation_sha256': digest(expected_path), 'source_count': len(sources),
            'sample_count': actual['sampled']['sample_count'], 'deterministic_archive_checked': True}


def run_references(root, *, write_expected=False, pattern=None):
    root = Path(root)
    from pipeline_kmz_regression_suite.reference.geometry import GeometryReference
    from pipeline_kmz_regression_suite.reference.sampled import hand_worked_checks
    reference = GeometryReference(BOUNDARIES)
    started = time.perf_counter()
    checks = hand_worked_checks()
    paths = select_fixtures(root, pattern, require_expectations=not write_expected)
    validate_previews(root)
    report = {'schema_version': '1.0.0', 'environment': environment(), 'baseline_commit': BASELINE,
              'contract_provenance': contract_provenance(), 'status': 'running', 'passed': False,
              'selection': pattern or 'all_core',
              'boundary_sha256': BOUNDARY_SHA256, 'hand_worked_checks': checks,
              'independent_module_audit': audit_independence(),
              'validator_sha256': digest(SUITE / 'suite.py'),
              'contract_source_sha256': {p: digest(REPO / 'src/pipeline_calculator/core' / p) for p in CONTRACT_FILES},
              'application_comparison': 'separate validation/application_comparison.json', 'fixtures': []}
    failures, expectations = [], []
    for path in paths:
        print(f'{"regenerate" if write_expected else "validate"}: {path.relative_to(root)}', flush=True)
        tick = time.perf_counter()
        value = expectation(path, reference, root=root)
        expectations.append(value)
        destination = root / 'expected' / f'{path.stem}.expected.json'
        errors = []
        if write_expected:
            dump(destination, value)
            for suffix, content in csv_contents(value).items():
                (root / 'expected' / f'{path.stem}.{suffix}').write_text(content, encoding='utf-8', newline='')
        else:
            errors = compare_values(read_json(destination), value)
            for suffix, content in csv_contents(value).items():
                if (root / 'expected' / f'{path.stem}.{suffix}').read_text(encoding='utf-8') != content:
                    errors.append(f'{path.stem}.{suffix}: ledger differs from independent reconstruction')
            failures.extend(errors)
        report['fixtures'].append({'fixture': value['fixture'], 'sha256': value['archive']['sha256'],
            'reference_status': 'incomplete' if errors else value['reference_status'],
            'source_count': value['archive']['source_count'],
            'sample_count': value['analyses']['Combined']['sample_count'],
            'original_meters': value['geometry']['combined_original_meters'],
            'coverage_assertion_count': len(value['coverage_checks']),
            'runtime_seconds': time.perf_counter() - tick, 'errors': errors})
    report['variant_checks'] = variant_checks(expectations)
    report['deterministic_archive_checks'] = deterministic_archive_check(root)
    report['stress_reference'] = validate_stress_reference(root) if pattern is None else {'status': 'not_selected'}
    report.update(runtime_seconds=time.perf_counter() - started, passed=not failures,
                  status='failed' if failures else 'complete', errors=failures,
                  artifact_sha256=artifact_fingerprints(root))
    if failures:
        raise ValueError('\n'.join(failures[:40]))
    return report


def publish_generated(staged, destination=SUITE):
    """Publish only after all new references pass; roll back ordinary I/O failures."""
    staged, destination = Path(staged), Path(destination)
    updates = [p for p in staged.rglob('*') if p.is_file()]
    backups, applied = {}, []
    try:
        for source in updates:
            target = destination / source.relative_to(staged)
            backups[target] = target.read_bytes() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
                candidate = Path(stream.name)
                stream.write(source.read_bytes())
            try:
                candidate.replace(target)
                applied.append(target)
            finally:
                candidate.unlink(missing_ok=True)
    except BaseException:
        for target in reversed(applied):
            if backups[target] is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(backups[target])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['regenerate', 'validate', 'compare', 'audit'])
    parser.add_argument('--only', help='Fixture stem prefix for a focused validation/comparison')
    args = parser.parse_args()
    if args.command in ('compare', 'audit'):
        script = 'compare_application.py' if args.command == 'compare' else 'run_audit.py'
        command = [sys.executable, str(SUITE / 'validation' / script)]
        if args.only:
            if args.command == 'audit':
                parser.error('--only is for fixture validation or comparison, not the adversarial audit')
            command += ['--only', args.only]
        raise SystemExit(subprocess.run(command).returncode)
    if args.command == 'regenerate' and args.only:
        parser.error('Regenerate all core assets together; use --only with validate for focused checks')
    report_path = SUITE / 'validation' / ('focused_reference_report.json' if args.only else 'reference_report.json')
    receipt = {'schema_version': '1.0.0', 'status': 'running', 'passed': False,
               'selection': args.only or 'all_core', 'command': args.command, 'errors': []}
    dump(report_path, receipt)
    try:
        if digest(BOUNDARIES) != BOUNDARY_SHA256:
            raise ValueError('Boundary resource checksum mismatch; no expectations created')
        if args.command == 'regenerate':
            provenance = contract_provenance()
            if not provenance['baseline_contract_unchanged']:
                raise ValueError('Contract source changed from the frozen baseline: ' + ', '.join(provenance['changed_contract_files']))
            # Existing optional assets must also be sound before publishing new core files.
            stress_reference = validate_stress_reference()
            # Neither archives nor goldens are overwritten until every case passes.
            with tempfile.TemporaryDirectory(prefix='pipeline-kmz-regenerate-') as tmp:
                staged = Path(tmp)
                subprocess.run([sys.executable, str(SUITE / 'generator/generate.py'), '--output', tmp], check=True)
                with zipfile.ZipFile(BOUNDARIES) as z:
                    dump(staged / 'validation/boundary_manifest.json', json.loads(z.read('manifest.json')))
                receipt = run_references(staged, write_expected=True)
                publish_generated(staged)
            receipt['stress_reference'] = stress_reference
            receipt['artifact_sha256'] = artifact_fingerprints()
        else:
            receipt = run_references(SUITE, pattern=args.only)
        dump(report_path, receipt)
        if not args.only:
            subprocess.run([sys.executable, str(SUITE / 'validation/render_reports.py')], check=True)
        print(f'Verified {len(receipt["fixtures"])} core archives in {receipt["runtime_seconds"]:.1f}s', flush=True)
    except Exception as error:
        receipt.update(status='failed', passed=False, errors=[str(error)])
        dump(report_path, receipt)
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
