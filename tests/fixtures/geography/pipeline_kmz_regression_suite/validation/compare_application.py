"""Compare already frozen independent expectations with the application.

This file is intentionally outside reference/. It never creates or updates a
golden. Map exports are temporary; only their measured assertions are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE.parent))
from pipeline_kmz_regression_suite.suite import REPO, BOUNDARIES, MILE, digest, dump, environment, select_fixtures, read_json
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))
from scripts.validation.corridor_numeric import VISUAL_DIAGNOSTICS
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.bundling import qualifying_sections
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.export.geography_kmz import write_geography_kmz
from pipeline_calculator.parsers.kml_kmz import extract_features_from_file_with_diagnostics
from pipeline_kmz_regression_suite.reference.geometry import GeometryReference, read_kmz
from pyproj import Geod
from shapely import from_wkb
from shapely.geometry import Polygon
from pipeline_kmz_regression_suite.validation.application_contract import compare_interval_ledger, compare_sampled_contract, polygon_containment, compare_export_corridors
import psutil

NS = {'k': 'http://www.opengis.net/kml/2.2'}
GEOD = Geod(ellps='GRS80')


def exported_geometry(path, *, include_corridors=False):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('doc.kml'))
    sources, polygons, ids, corridors = [], [], [], []
    for i, feature in enumerate(root.findall('.//k:Placemark', NS)):
        data = {n.attrib['name']: n.findtext('k:value', namespaces=NS)
                for n in feature.findall('k:ExtendedData/k:Data', NS)}
        paths = [[tuple(map(float, token.split(',')[:2])) for token in line.find('k:coordinates', NS).text.split()]
                 for line in feature.findall('.//k:LineString', NS)]
        if paths:
            ids.append(data.get('id'))
            sources.append({'key': f'export-{i}', 'name': feature.findtext('k:name', namespaces=NS) or '',
                            'paths': paths, 'xml_order': i, 'motif': 'export', **data})
        areas = []
        for poly in feature.findall('.//k:Polygon', NS):
            rings = []
            for ring in poly.findall('.//k:LinearRing', NS):
                rings.append([tuple(map(float, token.split(',')[:2]))
                              for token in ring.findtext('k:coordinates', namespaces=NS).split()])
            areas.append(Polygon(rings[0], rings[1:]))
        polygons.extend(areas)
        if areas:
            corridors.append({'metadata': data, 'polygons': areas})
    meters = math.fsum(abs(GEOD.inv(*a, *b)[2]) for s in sources for p in s['paths'] for a, b in zip(p, p[1:]))
    result = sources, polygons, ids, meters
    return (*result, corridors) if include_corridors else result


def compare(path, expected, reference, boundaries, export_check=True):
    tick = time.perf_counter()
    failures, checks = [], []

    def check(label, actual, target, tolerance=0):
        passed = (type(actual) is bool and actual is target if type(target) is bool else
                  isinstance(actual, (int, float)) and type(actual) is not bool and
                  math.isfinite(actual) and abs(actual - target) <= tolerance
                  if isinstance(target, (int, float)) else actual == target)
        row = {'assertion': label, 'actual': actual, 'expected': target, 'tolerance': tolerance, 'passed': passed}
        checks.append(row)
        if not passed:
            failures.append(row)

    result = PipelineAnalyzer().analyze_complete(path, options=AnalysisOptions(True))
    geo = result.get('geography', {})
    check('Combined analysis complete', result['analysis_complete'], True)
    check('Geography status', geo.get('status'), 'complete')
    check('crossing events', geo.get('crossing_count'), expected['geometry']['crossing_count'])
    check('Combined diagnostics', dict(Counter(d['code'] for d in result['diagnostics'] if d['code'] not in VISUAL_DIAGNOSTICS)),
          expected['expected_application_status']['combined_diagnostic_counts'])
    check('Geography diagnostics', dict(Counter(d['code'] for d in geo.get('diagnostics', []) if d['code'] not in VISUAL_DIAGNOSTICS)),
          expected['expected_application_status']['geography_diagnostic_counts'])
    check('source identity count', len(result['pipelines']), expected['archive']['source_count'])
    check('represented states', sorted(s['state_code'] for s in geo.get('states', [])), sorted(expected['geometry']['states']))
    check('Combined original', result['total_meters'], expected['analyses']['Combined']['original_meters'], .001)
    overlap = result.get('overlap_analysis')
    if overlap is None:
        check('Combined overlap present', False, True)
    else:
        check('Combined savings', overlap['savings_meters'], expected['analyses']['Combined']['savings_meters'], 1e-7)
    for name in ('outside_meters', 'unresolved_meters'):
        check(name, geo.get('reconciliation', {}).get(name, -1), 0.0)
    source_targets = {s['key']: s for s in expected['geometry']['sources']}
    xml_id_to_key = dict(zip(expected['archive']['xml_ids'], expected['archive']['xml_source_order']))
    originals = {s['key']: s for s in read_kmz(path)}
    parsed_input = extract_features_from_file_with_diagnostics(path)
    id_to_key = {pipeline['id']: xml_id_to_key[pipeline['placemark_id']] for pipeline in parsed_input.pipelines}
    check('Unique application source identities', len({s['id'] for s in parsed_input.pipelines}), len(originals))
    check('Public source identities', sorted(xml_id_to_key.get(row['Placemark_ID'], '<unknown>') for row in result['pipelines']),
          sorted(originals))
    check('Frozen default analysis profile', result['analysis_parameters'],
          {'detection_range': 15.0, 'min_parallel_length': 200.0, 'segment_length': 5.0, 'angular_tolerance': 15.0})
    for pipeline in parsed_input.pipelines:
        key = xml_id_to_key[pipeline['placemark_id']]
        check(f'{key}: parser preserves every path and vertex',
              pipeline['coordinate_paths'] == originals[key]['paths'], True)
    for row in result['pipelines']:
        key = xml_id_to_key[row['Placemark_ID']]
        check(f'{key}: original', row['Shape_Length'], source_targets[key]['original_meters'], .001)
    compare_interval_ledger(check, 'Partition', geo.get('fragments', []), expected['geometry']['intervals'],
                            id_to_key, originals=originals)
    check('Geography analysis complete', geo.get('analysis_complete'), True)
    check('Geography reconciliation passed', geo.get('reconciliation', {}).get('passed'), True)
    check('Combined adjusted', (overlap or {}).get('effective_total_meters'),
          expected['analyses']['Combined']['adjusted_meters'], .001)

    def sampled_scope(label, inputs, observed):
        analyzer = PipelineAnalyzer()
        groups = analyzer.find_parallel_segments(inputs)
        sections = qualifying_sections(inputs, groups, analyzer.segment_length, analyzer.min_parallel_length)
        compare_sampled_contract(check, label, inputs, sections, observed,
                                 expected['analyses'][label], id_to_key)

    sampled_scope('Combined', parsed_input.pipelines, overlap)
    for state in geo.get('states', []):
        code = state['state_code']
        target = expected['analyses'][code]
        accounting = expected['geometry']['states'][code]
        # Each uncertain interval endpoint contributes separately to a state bound.
        intervals = [f for f in expected['geometry']['intervals'] if code in f['state_codes']]
        endpoint_bound = math.fsum(2 * f.get('cut_error_bound_meters', .01) for f in intervals)
        tolerance = max(.001, endpoint_bound)
        check(f'{code}: complete', state['analysis_complete'], True)
        check(f'{code}: diagnostics', dict(Counter(d['code'] for d in state.get('diagnostics', []) if d['code'] not in VISUAL_DIAGNOSTICS)),
              expected['expected_application_status']['combined_diagnostic_counts'])
        check(f'{code}: interior', state['interior_meters'], accounting['interior_meters'], tolerance)
        check(f'{code}: shared allocation', state['shared_allocation_meters'], accounting['shared_allocation_meters'], tolerance)
        check(f'{code}: attributed original', state['total_meters'], accounting['attributed_original_meters'], tolerance)
        check(f'{code}: savings', state.get('interior_savings_meters', -1), target['savings_meters'], 1e-7)
        check(f'{code}: adjusted', state.get('adjusted_total_meters'),
              accounting['attributed_original_meters']-target['savings_meters'], tolerance)
        inputs = []
        for pipeline in parsed_input.pipelines:
            paths = [f['coordinates'] for f in geo.get('fragments', [])
                     if f['source_id'] == pipeline['id'] and f['kind'] == 'state' and f['state_codes'] == [code]]
            if paths:
                inputs.append(dict(pipeline, coordinates=paths[0], coordinate_paths=paths))
        sampled_scope(code, inputs, state.get('overlap_analysis'))
        expected_source_keys = sorted(key for key, source in source_targets.items() if code in source['states'])
        check(f'{code}: attributed source identities',
              sorted(xml_id_to_key[row['Placemark_ID']] for row in state['pipelines']), expected_source_keys)
        for row in state['pipelines']:
            key = xml_id_to_key[row['Placemark_ID']]
            source_state = source_targets[key]['states'].get(code, {
                'length_error_bound_meters': 0.0, 'attributed_original_meters': 0.0,
                'interior_meters': 0.0, 'shared_allocation_meters': 0.0})
            source_bound = max(.001, source_state['length_error_bound_meters'])
            check(f'{code}/{key}: attributed original', row['Shape_Length'], source_state['attributed_original_meters'], source_bound)
            check(f'{code}/{key}: interior', row['interior_meters'], source_state['interior_meters'], source_bound)
            check(f'{code}/{key}: shared allocation', row['shared_allocation_meters'], source_state['shared_allocation_meters'], source_bound)
            actual_shared = []
            fragments_by_id = {f['id']: f for f in geo.get('fragments', [])}
            for allocation in row.get('shared_border_allocations', []):
                fragment = fragments_by_id.get(allocation['fragment_id'])
                valid = fragment is not None and fragment['kind'] == 'shared' and code in fragment['state_codes'] and fragment['source_id'] == row['source_id']
                check(f'{code}/{key}: shared reference {allocation["fragment_id"]} identity', valid, True)
                if valid:
                    check(f'{code}/{key}: shared reference {allocation["fragment_id"]} equal allocation',
                          allocation['allocated_meters'], fragment['length_meters']/len(fragment['state_codes']), 1e-8)
                actual_shared.append(allocation['fragment_id'])
            check(f'{code}/{key}: each shared fragment allocated once', sorted(actual_shared),
                  sorted(f['id'] for f in geo.get('fragments', []) if f['source_id'] == row['source_id'] and
                         f['kind'] == 'shared' and code in f['state_codes']))

    exports = []
    if export_check:
        with tempfile.TemporaryDirectory(prefix='pipeline-kmz-export-') as temp:
            scopes = [None] + [s['state_code'] for s in geo.get('states', []) if s['interior_meters'] > 0]
            for code in scopes:
                label = code or 'Combined'
                target = expected['analyses']['Combined']['original_meters'] if code is None else expected['analyses'][code]['interior_meters']
                destination = Path(temp) / f'{label}.kmz'
                write_geography_kmz(result, destination, code)
                sources, polygons, ids, meters, corridors = exported_geometry(destination, include_corridors=True)
                scoped_sources = list(originals.values()) if code is None else expected['geometry']['state_inputs'][code]
                scope = result if code is None else next(s for s in geo['states'] if s['state_code'] == code)
                sections = (scope.get('overlap_analysis') or {}).get('bundled_sections', [])
                check(f'{label}: every expected corridor map is ready',
                      all(s.get('visualization_status') == 'ready' for s in sections), True)
                check(f'{label}: fixed-radius map policy',
                      all(s.get('visualization_metadata', {}).get('policy') == 'qualified_path_buffer_v1' and
                          s.get('visualization_metadata', {}).get('padding_m') == 5.0 for s in sections), True)
                compare_export_corridors(check, label, corridors, expected['analyses'][label], scoped_sources, id_to_key,
                                         padding_m=5.0, clip_boundary=boundaries[code] if code else None)
                endpoint_bound = math.fsum(2*f['cut_error_bound_meters'] for f in expected['geometry']['intervals']
                                          if code is None or (f['kind'] == 'interior' and f['state_codes'] == [code]))
                check(f'{label}: exported line mileage', meters, target, max(.001, endpoint_bound))
                check(f'{label}: unique exported fragment ids', len(ids), len(set(ids)))
                required_ids = {f['id'] for f in geo.get('fragments', [])
                                if code is None or (f['kind'] == 'state' and f['state_codes'] == [code])}
                check(f'{label}: fragment identities', sorted(ids), sorted(required_ids))
                expected_intervals = [f for f in expected['geometry']['intervals'] if code is None or
                                      (f['kind'] == 'interior' and f['state_codes'] == [code])]
                compare_interval_ledger(check, f'{label} export', sources, expected_intervals, id_to_key,
                                        export=True, originals=originals)
                parsed = extract_features_from_file_with_diagnostics(destination)
                _, reimport_meters, _ = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)
                check(f'{label}: application reimport line mileage excludes polygons', reimport_meters, meters, .001)
                polygon_checks = []
                if code:
                    polygon_checks = polygon_containment(polygons, boundaries[code])
                    check(f'{label}: polygons contained within canonical state and numeric boundary strip',
                          all(row['passed'] for row in polygon_checks), True)
                exports.append({'scope': label, 'line_meters': meters, 'source_fragment_count': len(sources),
                                'omitted_map_count': sum(s.get('visualization_status') == 'omitted' for s in sections),
                                'polygon_count': len(polygons), 'polygon_containment': polygon_checks,
                                'line_verification': 'Independent stable source/path interval spans and original geodesic membership'})
    process = psutil.Process()
    return {'fixture': path.relative_to(SUITE / 'fixtures').as_posix(), 'sha256': digest(path),
            'expectation_sha256': digest(SUITE / 'expected' / f'{path.stem}.expected.json'),
            'passed': not failures, 'runtime_seconds': time.perf_counter() - tick,
            'process_memory_bytes': process.memory_info()._asdict(),
            'diagnostics': result.get('diagnostics', []), 'geography_diagnostics': geo.get('diagnostics', []),
            'state_diagnostics': {s['state_code']: s.get('diagnostics', []) for s in geo.get('states', [])},
            'crossing_count_observed': geo.get('crossing_count'),
            'partition_observation': [{**{k: f[k] for k in ('path_index', 'start_m', 'end_m', 'kind', 'state_codes', 'length_meters')},
                                       'key': id_to_key.get(f['source_id'], '<unknown>')} for f in geo.get('fragments', [])],
            'checks': checks, 'mismatches': failures, 'exports': exports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only')
    parser.add_argument('--skip-exports', action='store_true')
    args = parser.parse_args()
    report_path = SUITE / 'validation' / ('focused_application_comparison.json' if args.only else 'application_comparison.json')
    report = {'status': 'running', 'passed': False, 'selection': args.only,
              'export_checks_enabled': not args.skip_exports, 'fixtures': []}
    dump(report_path, report)
    try:
        paths = select_fixtures(SUITE, args.only)
        frozen = []
        for path in paths:
            expected = read_json(SUITE / 'expected' / f'{path.stem}.expected.json')
            if expected['fixture'] != path.relative_to(SUITE / 'fixtures').as_posix():
                raise ValueError(f'Expectation points to the wrong archive: {path.name}')
            if digest(path) != expected['archive']['sha256']:
                raise ValueError(f'Archive changed since expectations: {path}')
            frozen.append((path, expected))
        reference = GeometryReference(BOUNDARIES)
        with zipfile.ZipFile(BOUNDARIES) as z:
            manifest = json.loads(z.read('manifest.json'))
            boundaries = {s['code']: from_wkb(z.read(s['file'])) for s in manifest['states']}
        report.update({'environment': environment(),
                  'application_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  'application_source_sha256': {
                      source.relative_to(REPO).as_posix(): digest(source)
                      for source in sorted((REPO / 'src').rglob('*.py'))},
                  'expectations_fixed_before_application_run': True})
        for path, expected in frozen:
            print(f'Application comparison: {path.name}', flush=True)
            row = compare(path, expected, reference, boundaries, not args.skip_exports)
            report['fixtures'].append(row)
            print(f'  passed={row["passed"]}; mismatches={len(row["mismatches"])}; seconds={row["runtime_seconds"]:.1f}', flush=True)
            dump(report_path, report)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        dump(report_path, report)
        raise
    report['passed'] = all(f['passed'] for f in report['fixtures'])
    report['status'] = 'complete'
    dump(report_path, report)
    if not report['passed']:
        raise SystemExit('Application mismatches recorded separately; expectations unchanged')


if __name__ == '__main__':
    main()
