"""Opt-in real-dataset audit; writes local evidence and never opens a viewer.

Uses a separate dense-source Shapely reference for complete polygon checks.
Production calculations and KML generation do not import this audit module.
"""
from pathlib import Path
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from pyproj import Geod

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.bundling import qualifying_sections
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.gui.controllers import analysis_controller
from pipeline_calculator.parsers.kml_kmz import extract_features_from_file_with_diagnostics
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml
from pipeline_calculator.export.xlsx import build_analysis_workbook

GEOD = Geod(ellps='GRS80')
NS = {'k': 'http://www.opengis.net/kml/2.2'}
CASES = {
    'default': {},
    'range10': {'detection_range': 10},
    'range25': {'detection_range': 25},
    'minimum100': {'min_parallel_length': 100},
    'minimum500': {'min_parallel_length': 500},
    'angle5': {'angular_tolerance': 5},
    'angle30': {'angular_tolerance': 30},
    'step4': {'segment_length': 4},
    'step10': {'segment_length': 10},
    'step20': {'segment_length': 20},
    'step50': {'segment_length': 50},
    'strict': {'detection_range': 8, 'min_parallel_length': 500, 'segment_length': 10, 'angular_tolerance': 5},
    'broad': {'detection_range': 30, 'min_parallel_length': 100, 'segment_length': 10, 'angular_tolerance': 30},
    'step2-limit': {'segment_length': 2},
}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def inventory(path):
    """Walk raw XML independently, preserving each individual coordinate list."""
    raw_paths, tags, members = [], Counter(), []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            members.append({'name': entry.filename, 'bytes': entry.file_size})
            if not entry.filename.lower().endswith('.kml'):
                continue
            document = ET.fromstring(archive.read(entry))
            tags.update(e.tag.rsplit('}', 1)[-1] for e in document.iter())
            for line in document.findall('.//k:LineString', NS):
                raw_paths.append([tuple(map(float, token.split(',')[:2]))
                                  for token in line.find('k:coordinates', NS).text.split()])
    parsed = extract_features_from_file_with_diagnostics(path)
    loaded_paths = [coords for pipeline in parsed.pipelines for coords in pipeline['coordinate_paths']]
    lengths = [abs(GEOD.line_length(*zip(*coords))) for coords in raw_paths]
    assert raw_paths == loaded_paths, 'Parser omitted/reordered/changed source coordinates'
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'members': members,
            'tags': dict(tags), 'raw_paths': len(raw_paths), 'loaded_paths': len(loaded_paths),
            'pipeline_records': len(parsed.pipelines), 'coordinates': sum(map(len, raw_paths)),
            'raw_geodesic_meters': math.fsum(lengths), 'coordinate_lists_exact': True,
            'duplicate_names': [name for name, n in Counter(p['name'] for p in parsed.pipelines).items() if n > 1],
            'diagnostics': parsed.diagnostics}


def stations(coords, values):
    """Independent chainage interpolation on original vertices, including bends."""
    points = np.asarray(coords, dtype=float)
    az, _, length = GEOD.inv(points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1])
    ends = np.cumsum(length)
    starts = np.r_[0, ends[:-1]]
    indices = np.minimum(np.searchsorted(ends, values, side='right'), len(length)-1)
    lon, lat, _ = GEOD.fwd(points[indices, 0], points[indices, 1], np.asarray(az)[indices], values-starts[indices])
    return np.column_stack((lon, lat))


def project(points, center):
    points = np.asarray(points, dtype=float)
    lon, lat = np.full(len(points), center[0]), np.full(len(points), center[1])
    az, _, distance = GEOD.inv(lon, lat, points[:, 0], points[:, 1])
    az = np.radians(az)
    return np.column_stack((distance*np.sin(az), distance*np.cos(az)))


class CapturedAnalyzer(PipelineAnalyzer):
    def find_parallel_segments(self, pipelines, progress_callback=None):
        self.audit_pipelines = pipelines
        groups = super().find_parallel_segments(pipelines, progress_callback)
        self.audit_groups = groups
        return groups

    def calculate_overlap_results(self, pipelines, groups, progress_callback=None, **kwargs):
        self.audit_sections = qualifying_sections(pipelines, groups, self.segment_length, self.min_parallel_length)
        return super().calculate_overlap_results(pipelines, groups, progress_callback, **kwargs)


def inspect_corridors(analyzer, result, output):
    from scripts.validation.corridor_audit import inspect_document, section_runs
    qualified = sorted(analyzer.audit_sections, key=lambda s: s['length'], reverse=True)
    exported = result['overlap_analysis']['bundled_sections']
    assert len(qualified) == len(exported)
    rows = []
    for index, (source, section) in enumerate(zip(qualified, exported), 1):
        if section.get('visualization_status') == 'omitted':
            try:
                build_overlap_corridor_kml(section, index)
            except ValueError:
                rows.append({'index': index, 'status': 'omitted', 'diagnostics': section.get('diagnostics', []),
                             'passed': bool(section.get('diagnostics')), 'part_count': 0, 'hole_count': 0,
                             'uncovered_source_length_m': 0, 'outside_points': 0, 'max_outside_m': 0})
                continue
            raise AssertionError('An omitted section unexpectedly exported a map')
        kml = build_overlap_corridor_kml(section, index)
        (output/f'corridor-{index:03}.kml').write_text(kml, encoding='utf-8')
        doc = ET.fromstring(kml)
        paths, ranges = section_runs(analyzer.audit_pipelines, source, analyzer.segment_length)
        metadata = section.get('visualization_metadata', {})
        padding = metadata.get('padding_m') if metadata.get('policy') == 'qualified_path_buffer_v1' else None
        verified, _, _ = inspect_document(kml, paths, padding_m=padding)
        desc = doc.find('.//k:description', NS).text
        row = {'index': index, 'pair': [section['pipeline_1'], section['pipeline_2']],
               'length_m': section['bundled_length_meters'], 'separation_m': section['average_separation'],
               'ranges': ranges, 'description': desc, 'status': 'ready',
               'outside_points': int(not verified['source_coverage_passed']),
               'max_outside_m': 0 if verified['source_coverage_passed'] else None,
               'geometry_policy': metadata.get('policy', 'legacy'), **verified}
        rows.append(row)
    return rows


def run_case(path, name, output, reference=False):
    started = time.monotonic()
    values = AnalysisParameters(**CASES[name]).as_dict()
    params, corrections = AnalysisParameters.from_strings(**{k: str(v) for k, v in values.items()})
    assert not corrections and params.as_dict() == values
    analyzer = CapturedAnalyzer(**values)
    with patch.object(analysis_controller, 'PipelineAnalyzer', return_value=analyzer) as factory:
        result = analysis_controller.analyze_file(str(path), params)
        assert factory.call_args.kwargs == values
    assert result['analysis_parameters'] == values
    output.mkdir(parents=True, exist_ok=True)
    save(output/'results.json', result)
    build_analysis_workbook(result).save(output/'results.xlsx')
    overlap = result['overlap_analysis']
    row = {'case': name, 'parameters': values, 'complete': result['analysis_complete'],
           'diagnostics': result['diagnostics'], 'source_miles': result['total_miles'],
           'source_meters': result['total_meters'], 'ui_backend_parameters_exact': True}
    if overlap is not None:
        if reference:
            from audit_reference import check_samples
            row['independent_sample_check'] = check_samples(analyzer)
            assert abs(row['independent_sample_check']['independent_clique_savings_meters']-overlap['savings_meters']) < 1e-6
        assert abs(result['total_meters']-overlap['effective_total_meters']-overlap['savings_meters']) < 1e-6
        assert abs(sum(s['bundled_length_meters'] for s in overlap['bundled_sections'])-overlap['total_bundled_length']) < 1e-6
        corridors = inspect_corridors(analyzer, result, output)
        save(output/'corridor-checks.json', corridors)
        row.update(removed_miles=overlap['savings_miles'], adjusted_miles=overlap['effective_total_miles'],
                   pairwise_miles=overlap['total_bundled_length']/analyzer.survey_mile,
                   sections=len(corridors), invalid_polygons=0,
                   clipped_corridors=sum(c['outside_points'] > 0 or c['uncovered_source_length_m'] > .01 for c in corridors),
                   uncovered_source_length_m=sum(c['uncovered_source_length_m'] for c in corridors),
                   max_outside_m=max((c['max_outside_m'] or 0 for c in corridors), default=0),
                   geometry_checks_passed=all(c['passed'] for c in corridors),
                   polygon_parts=sum(c['part_count'] for c in corridors),
                   polygon_holes=sum(c['hole_count'] for c in corridors),
                   omitted_maps=sum(c['status'] == 'omitted' for c in corridors),
                   legacy_maps=sum(c.get('geometry_policy') == 'legacy' for c in corridors),
                   match_pairs=sum(map(len, analyzer.audit_groups.values())),
                   segment_count=sum(len(p['segments']) for p in analyzer.audit_pipelines))
    row['elapsed_seconds'] = time.monotonic()-started
    save(output/'audit.json', row)
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--geometry-deps', type=Path)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    if args.geometry_deps:
        sys.path.insert(0, str(args.geometry_deps.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    source = inventory(args.input)
    save(args.output/'input-inventory.json', source)
    rows = []
    for name in args.cases:
        rows.append(run_case(args.input, name, args.output/name, args.reference))
        assert abs(rows[-1]['source_meters']-source['raw_geodesic_meters']) < 1e-6
        save(args.output/'matrix.json', rows)
        gc.collect()
    failed = [row['case'] for row in rows if row.get('clipped_corridors', 0) or not row.get('geometry_checks_passed', True) or
              (not row['complete'] and row['case'] != 'step2-limit')]
    if failed:
        raise SystemExit('Audit failed: '+', '.join(failed))


if __name__ == '__main__':
    main()
