"""Freeze/compare corridor-display numerical contracts and measured performance.

Run capture against an immutable baseline checkout before editing production.
Inputs are deterministic and shared between baseline/candidate. Never generate
independent mileage goldens from a candidate run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
VISUAL_FIELDS = frozenset({
    'bbox', 'center_lon', 'center_lat', 'oriented_polygon', 'corridor_polygon',
    'oriented_width_m', 'corridor_geometry_kind', 'corridor_approximation',
    'visualization_schema_version', 'visualization_kind', 'visualization_status',
    'visualization_polygons', 'visualization_metadata', 'visualization_approximation',
    'clipped_polygons',
})
VISUAL_DIAGNOSTICS = frozenset({
    'corridor_geometry_fallback', 'corridor_visualization_omitted',
    'state_corridor_omitted', 'state_corridor_unavailable',
    'corridor_buffer_limit', 'corridor_projection_unavailable', 'corridor_geometry_invalid', 'corridor_coverage_failed',
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def numerical(value, key=None):
    """Explicit visual exclusions; every unrecognized field/diagnostic survives."""
    if key == 'parsed_kml_files':
        return [Path(path).name for path in value]
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if name in VISUAL_FIELDS:
                continue
            if name == 'diagnostics':
                item = [d for d in item if d.get('code') not in VISUAL_DIAGNOSTICS]
                if not item:
                    continue  # legacy sections had no empty diagnostic field
            result[name] = numerical(item, name)
        return result
    if isinstance(value, (list, tuple)):
        return [numerical(item) for item in value]
    return value


def differences(actual, expected, path='$'):
    """Exact values, identities and membership; no floating tolerance blanket."""
    if type(actual) is not type(expected):
        return [{'path': path, 'actual': actual, 'expected': expected}]
    if isinstance(actual, dict):
        rows = []
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                rows.append({'path': f'{path}.{key}', 'actual': actual.get(key),
                             'expected': expected.get(key), 'missing_key': True})
            else:
                rows.extend(differences(actual[key], expected[key], f'{path}.{key}'))
        return rows
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return [{'path': path + '.length', 'actual': len(actual), 'expected': len(expected)}]
        return [row for i, (a, b) in enumerate(zip(actual, expected))
                for row in differences(a, b, f'{path}[{i}]')]
    if isinstance(actual, float) and not math.isfinite(actual):
        return [{'path': path, 'actual': repr(actual), 'expected': expected}]
    return [] if actual == expected else [{'path': path, 'actual': actual, 'expected': expected}]


def environment(source_root):
    names = ('numpy', 'scipy', 'pyproj', 'shapely', 'geographiclib', 'psutil')
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'cpu': platform.processor(), 'packages': packages,
            'source_root': str(source_root.resolve()),
            'source_files': {p.relative_to(source_root).as_posix(): digest(p)
                             for p in sorted((source_root / 'src').rglob('*.py'))},
            'boundary_sha256': digest(source_root / 'src/pipeline_calculator/data/states_2025.zip'),
            'validation_script_sha256': digest(__file__)}


def generate(output):
    from pyproj import Geod
    geod = Geod(ellps='GRS80')
    output.mkdir(parents=True, exist_ok=True)
    shapes = {
        'straight': [(0, 0), (0, 302.3)],
        'L': [(0, 0), (0, 153.2), (180.4, 153.2)],
        'U': [(0, 0), (0, 200.2), (150.1, 200.2), (150.1, 0)],
        'S': [(0, 0), (0, 110), (70, 190), (0, 270), (0, 400.3)],
        'loop': [(0, 0), (0, 200), (200, 200), (200, 0), (0, 0)],
        'multipart': [[(0, 0), (0, 123.2)], [(0, 600), (0, 823.1)]],
        'separate': [(0, 0), (0, 302.3)],
        'dateline': [(0, 0), (400, 0)],
    }
    cases = []
    for name, shape in shapes.items():
        paths = shape if name == 'multipart' else [shape]
        origin = (179.999, 60) if name == 'dateline' else (-100, 40)
        separation = 12 if name == 'separate' else 2
        placemarks = []
        for index, offset in enumerate((0, separation)):
            geometries = []
            for path in paths:
                coordinates = []
                for x, y in path:
                    x += offset
                    lon, lat, _ = geod.fwd(*origin, math.degrees(math.atan2(x, y)), math.hypot(x, y))
                    coordinates.append(f'{lon!r},{lat!r},0')
                geometries.append('<LineString><coordinates>' + ' '.join(coordinates) + '</coordinates></LineString>')
            placemarks.append(f'<Placemark id="{name}_{index}"><name>{escape(str(index))}</name><MultiGeometry>' +
                              ''.join(geometries) + '</MultiGeometry></Placemark>')
        path = output / f'{name}.kml'
        path.write_text('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>' + ''.join(placemarks) +
                        '</Document></kml>', encoding='utf-8')
        cases.append({'id': 'ordinary_' + name, 'path': str(path.resolve()), 'sha256': digest(path),
                      'parameters': {'min_parallel_length': 10}, 'state_breakdown': False})
    suite = ROOT / 'tests/fixtures/geography/pipeline_kmz_regression_suite/fixtures'
    for path in sorted(suite.rglob('*.kmz')):
        if 'stress' in path.parts:
            continue
        for state in (False, True):
            cases.append({'id': ('state_' if state else 'ordinary_') + path.stem,
                          'path': str(path), 'sha256': digest(path), 'parameters': {}, 'state_breakdown': state})
    straight = next(c for c in cases if c['id'] == 'ordinary_straight')
    for name, state in (('ordinary_numeric_failure', False), ('state_numeric_failure', True)):
        cases.append({**straight, 'id': name, 'parameters': {'segment_length': 1e-7}, 'state_breakdown': state})
    cases.append({**straight, 'id': 'legacy_display_failure', 'inject_display_failure': True,
                  'comparison_policy': 'legacy_visual_failure_recovery', 'recovery_reference_case': 'ordinary_straight'})
    manifest = {'schema_version': 1, 'generator_version': 1, 'cases': cases}
    save(output / 'cases.json', manifest)
    return manifest


def qualified_record(pipelines, sections):
    """Capture real qualified membership before display code touches the result."""
    records = []
    for section in sections:
        records.append({
            'sources': [pipelines[p].get('id', p) for p in section['pair']],
            'names': [pipelines[p].get('name', '') for p in section['pair']],
            'paths': list(section['paths']), 'lengths': list(section['lengths']), 'length': section['length'],
            'sample_membership': [sorted((pipelines[p]['segments'][i]['path_index'],
                                          pipelines[p]['segments'][i]['path_segment_index']) for i in ids)
                                  for p, ids in zip(section['pair'], section['segment_ids'])],
            'eligible_matches': sorted((m['pipeline_1_segment'], m['pipeline_2_segment'])
                                       for m in section['matches']),
        })
    return {'sources': [{'id': p.get('id', i), 'name': p.get('name', ''),
                          'coordinate_paths_sha256': hashlib.sha256(json.dumps(p.get('coordinate_paths', [p.get('coordinates')]),
                               separators=(',', ':'), allow_nan=False).encode()).hexdigest()} for i, p in enumerate(pipelines)],
            'sections': records}


def analyze_case(case, source_root, *, membership=True):
    sys.path.insert(0, str(source_root / 'src'))
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core.options import AnalysisOptions
    from pipeline_calculator.core import overlap
    if not Path(overlap.__file__).resolve().is_relative_to(source_root.resolve()):
        raise RuntimeError('Wrong application imported; capture each source revision in a fresh process')
    captured, visual_times = [], []
    original = overlap.qualifying_sections

    def record(pipelines, *args, **kwargs):
        sections = original(pipelines, *args, **kwargs)
        captured.append(qualified_record(pipelines, sections))
        return sections

    with ExitStack() as stack:
        if membership:
            stack.enter_context(patch.object(overlap, 'qualifying_sections', record))
        if case.get('inject_display_failure'):
            target = 'build_buffered_corridor' if hasattr(overlap, 'build_buffered_corridor') else 'compute_origin'
            stack.enter_context(patch.object(overlap, target, side_effect=RuntimeError('validation injected display failure')))
        elif hasattr(overlap, 'build_buffered_corridor'):
            builder = overlap.build_buffered_corridor
            def timed(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return builder(*args, **kwargs)
                finally:
                    visual_times.append(time.perf_counter() - start)
            stack.enter_context(patch.object(overlap, 'build_buffered_corridor', timed))
        start = time.perf_counter()
        result = PipelineAnalyzer(**case['parameters']).analyze_complete(case['path'], options=AnalysisOptions(case['state_breakdown']))
        elapsed = time.perf_counter() - start
    return {'result': result, 'qualified_membership': captured, 'analysis_seconds': elapsed,
            'visual_builder_seconds': sum(visual_times) if visual_times else None}


def capture(manifest, source_root, output):
    report = {'schema_version': 1, 'status': 'running', 'environment': environment(source_root),
              'visual_fields_excluded': sorted(VISUAL_FIELDS), 'visual_diagnostics_excluded': sorted(VISUAL_DIAGNOSTICS), 'cases': []}
    save(output, report)
    for case in manifest['cases']:
        if digest(case['path']) != case['sha256']:
            raise ValueError('Input changed: ' + case['id'])
        print('Capturing ' + case['id'], flush=True)
        observed = analyze_case(case, source_root)
        report['cases'].append({**case, **observed})
        save(output, report)
    report['status'] = 'complete'
    report['environment_after'] = environment(source_root)
    if report['environment']['source_files'] != report['environment_after']['source_files']:
        raise ValueError('Application sources changed during capture')
    save(output, report)
    return report


def compare(before, after):
    if before.get('status') != 'complete' or after.get('status') != 'complete':
        raise ValueError('Numeric comparison requires two completed captures')
    rows = []
    old = {row['id']: row for row in before['cases']}
    new = {row['id']: row for row in after['cases']}
    if len(old) != len(before['cases']) or len(new) != len(after['cases']) or set(old) != set(new) or not old:
        raise ValueError('Capture inventories differ, contain duplicates, or are empty')
    for name, base in old.items():
        candidate = new[name]
        errors = differences(candidate['sha256'], base['sha256'], '$.input_sha256')
        errors += differences(candidate['parameters'], base['parameters'], '$.parameters')
        errors += differences(candidate['state_breakdown'], base['state_breakdown'], '$.state_breakdown')
        target = base
        if base.get('comparison_policy') == 'legacy_visual_failure_recovery':
            if base['result']['analysis_complete'] or base['result']['overlap_analysis'] is not None:
                raise ValueError('Legacy display failure was not reproduced in the baseline')
            target = old[base['recovery_reference_case']]
            diagnostics = candidate['result'].get('diagnostics', [])
            if not any(d.get('code') in VISUAL_DIAGNOSTICS for d in diagnostics):
                errors.append({'path': '$.intentional_recovery.visual_diagnostic', 'actual': diagnostics, 'expected': 'explicit map-failure diagnostic'})
        errors += differences(numerical(candidate['result']), numerical(target['result']))
        errors += differences(numerical(candidate['qualified_membership']), numerical(target['qualified_membership']), '$.qualified_membership')
        rows.append({'id': name, 'passed': not errors, 'comparison_policy': base.get('comparison_policy', 'exact_numeric_parity'),
                     'differences': errors})
    for key in ('python', 'packages', 'boundary_sha256'):
        if before['environment'][key] != after['environment'][key]:
            raise ValueError('Baseline/candidate environment mismatch: ' + key)
    return {'schema_version': 1, 'passed': all(row['passed'] for row in rows),
            'baseline_environment': before['environment'], 'candidate_environment': after['environment'],
            'visual_fields_excluded': sorted(VISUAL_FIELDS), 'visual_diagnostics_excluded': sorted(VISUAL_DIAGNOSTICS), 'cases': rows}


def performance_cases():
    suite = ROOT / 'tests/fixtures/geography/pipeline_kmz_regression_suite/fixtures'
    paths = [('ordinary_corridors', suite / '03_parallel_corridors_crossing_borders.kmz', False),
             ('state_corridors', suite / '03_parallel_corridors_crossing_borders.kmz', True),
             ('state_disconnected', suite / '01_three_state_disconnected_networks.kmz', True),
             ('state_stress', suite / 'stress/stress_branching_network.kmz', True),
             ('state_centerlines', ROOT / 'tests/fixtures/geography/adamas_ng_pipeline_row_centerlines.kmz', True)]
    return [{'id': name, 'path': str(path), 'sha256': digest(path), 'parameters': {},
             'state_breakdown': state} for name, path, state in paths]


def benchmark_worker(case, source_root, output):
    import psutil
    process = psutil.Process()
    observed = analyze_case(case, source_root, membership=False)
    memory = process.memory_info()
    if os.name == 'nt':
        peak = memory.peak_wset
        method = 'Windows process peak working set, including imports'
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        method = 'getrusage process peak resident set, including imports'
    numeric = numerical(observed['result'])
    save(output, {'id': case['id'], 'input_sha256': digest(case['path']),
                  'analysis_seconds': observed['analysis_seconds'], 'visual_builder_seconds': observed['visual_builder_seconds'],
                  'peak_memory_bytes': peak, 'memory_method': method,
                  'numeric_sha256': hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  'analysis_complete': observed['result']['analysis_complete'],
                  'numeric': numeric})


def benchmark(source_root, output, runs=3, only=None):
    if runs < 3:
        raise ValueError('Performance evidence requires at least three measured runs')
    cases = [case for case in performance_cases() if not only or case['id'] in only]
    if not cases:
        raise ValueError('No performance cases selected')
    report = {'schema_version': 1, 'status': 'running', 'environment': environment(source_root),
              'runs_per_case': runs, 'cases': []}
    save(output / 'performance.json', report)
    for case in cases:
        case_path = output / (case['id'] + '-case.json')
        save(case_path, case)
        measurements = []
        for run in range(1, runs + 1):
            run_path = output / f'{case["id"]}-{run}.json'
            print(f'Benchmark {case["id"]} {run}/{runs}', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '_benchmark-worker',
                            '--case', str(case_path), '--source-root', str(source_root), '--output', str(run_path)],
                           check=True, env={**os.environ, 'PROJ_NETWORK': 'OFF'})
            measurements.append(json.loads(run_path.read_text()))
        report['cases'].append({'id': case['id'], 'input_sha256': case['sha256'], 'runs': measurements,
                                'median_seconds': statistics.median(row['analysis_seconds'] for row in measurements),
                                'peak_memory_bytes': max(row['peak_memory_bytes'] for row in measurements),
                                'numeric_stable': len({row['numeric_sha256'] for row in measurements}) == 1})
        save(output / 'performance.json', report)
    report['status'] = 'complete'
    report['environment_after'] = environment(source_root)
    if report['environment']['source_files'] != report['environment_after']['source_files']:
        raise ValueError('Application sources changed during benchmark')
    save(output / 'performance.json', report)


def compare_performance(before, after):
    if before.get('status') != 'complete' or after.get('status') != 'complete':
        raise ValueError('Performance comparison requires two completed captures')
    rows = []
    old = {row['id']: row for row in before['cases']}
    new = {row['id']: row for row in after['cases']}
    if not old or set(old) != set(new):
        raise ValueError('Performance inventories differ or are empty')
    for key in ('python', 'packages', 'boundary_sha256'):
        if before['environment'][key] != after['environment'][key]:
            raise ValueError('Performance environments differ: ' + key)
    for name, base in old.items():
        candidate = new[name]
        if base['input_sha256'] != candidate['input_sha256']:
            raise ValueError('Performance inputs differ: ' + name)
        numeric = not differences(candidate['runs'][0]['numeric'], base['runs'][0]['numeric'])
        ceiling = base['median_seconds'] + max(1.0, base['median_seconds'] * .25)
        memory_ceiling = base['peak_memory_bytes'] * 1.25
        rows.append({'id': name, 'baseline_median_seconds': base['median_seconds'],
                     'candidate_median_seconds': candidate['median_seconds'], 'time_ceiling_seconds': ceiling,
                     'baseline_peak_memory_bytes': base['peak_memory_bytes'], 'candidate_peak_memory_bytes': candidate['peak_memory_bytes'],
                     'memory_ceiling_bytes': memory_ceiling, 'numeric_equal': numeric,
                     'time_passed': candidate['median_seconds'] <= ceiling,
                     'memory_passed': candidate['peak_memory_bytes'] <= memory_ceiling,
                     'three_runs_each': min(len(base['runs']), len(candidate['runs'])) >= 3,
                     'numeric_stable': base['numeric_stable'] and candidate['numeric_stable']})
    return {'passed': all(all(row[k] for k in ('numeric_equal', 'time_passed', 'memory_passed', 'three_runs_each', 'numeric_stable'))
                          for row in rows), 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate_parser = sub.add_parser('generate'); generate_parser.add_argument('--output', type=Path, required=True)
    cap = sub.add_parser('capture'); cap.add_argument('--inputs', type=Path, required=True)
    cap.add_argument('--source-root', type=Path, default=ROOT); cap.add_argument('--output', type=Path, required=True)
    cmp = sub.add_parser('compare'); cmp.add_argument('--before', type=Path, required=True)
    cmp.add_argument('--after', type=Path, required=True); cmp.add_argument('--output', type=Path, required=True)
    bench = sub.add_parser('benchmark'); bench.add_argument('--source-root', type=Path, default=ROOT)
    bench.add_argument('--output', type=Path, required=True); bench.add_argument('--runs', type=int, default=3)
    bench.add_argument('--only', nargs='+')
    worker = sub.add_parser('_benchmark-worker'); worker.add_argument('--source-root', type=Path, required=True)
    worker.add_argument('--case', type=Path, required=True); worker.add_argument('--output', type=Path, required=True)
    perf = sub.add_parser('compare-performance'); perf.add_argument('--before', type=Path, required=True)
    perf.add_argument('--after', type=Path, required=True); perf.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args.output)
    elif args.command == 'capture':
        capture(json.loads(args.inputs.read_text()), args.source_root.resolve(), args.output)
    elif args.command == 'compare':
        report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        save(args.output, report)
        print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']),
                          'failing': [row['id'] for row in report['cases'] if not row['passed']]}))
        return int(not report['passed'])
    elif args.command == 'benchmark':
        benchmark(args.source_root.resolve(), args.output.resolve(), args.runs, args.only)
    elif args.command == '_benchmark-worker':
        benchmark_worker(json.loads(args.case.read_text()), args.source_root.resolve(), args.output)
    elif args.command == 'compare-performance':
        report = compare_performance(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        save(args.output, report)
        print(json.dumps(report))
        return int(not report['passed'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
