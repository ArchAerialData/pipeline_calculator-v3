"""Opt-in real-dataset audit; writes local evidence and never opens a viewer.

Requires Shapely as an independent validation-only geometry engine. Production
calculations and KML generation do not import this module or depend on Shapely.
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

    def calculate_overlap_results(self, pipelines, groups, progress_callback=None):
        self.audit_sections = qualifying_sections(pipelines, groups, self.segment_length, self.min_parallel_length)
        return super().calculate_overlap_results(pipelines, groups, progress_callback)


def inspect_corridors(analyzer, result, output):
    import shapely
    qualified = sorted(analyzer.audit_sections, key=lambda s: s['length'], reverse=True)
    exported = result['overlap_analysis']['bundled_sections']
    assert len(qualified) == len(exported)
    rows = []
    for index, (source, section) in enumerate(zip(qualified, exported), 1):
        kml = build_overlap_corridor_kml(section, index)
        (output/f'corridor-{index:03}.kml').write_text(kml, encoding='utf-8')
        doc = ET.fromstring(kml)
        coords = [tuple(map(float, token.split(',')[:2]))
                  for token in doc.find('.//k:Polygon//k:coordinates', NS).text.split()]
        center = (section['center_lon'], section['center_lat'])
        polygon = shapely.Polygon(project(coords, center))
        assert polygon.is_valid and polygon.area > 0 and coords[0] == coords[-1]
        # 2 cm accommodates serialized 7-decimal rounding and projection noise.
        boundary = shapely.buffer(polygon, .02)
        shapely.prepare(boundary)
        outside = []
        uncovered_length = 0.0
        test_count = 0
        ranges = []
        for pipeline_index, path_index, ids in zip(source['pair'], source['paths'], source['segment_ids']):
            pipeline = analyzer.audit_pipelines[pipeline_index]
            path_ids = sorted(pipeline['segments'][i]['path_segment_index'] for i in ids)
            values = np.unique(np.array([j+f for j in path_ids for f in (0, .5, 1)])*analyzer.segment_length)
            path = pipeline['coordinate_paths'][path_index]
            raw = np.asarray(path)
            _, _, edge_lengths = GEOD.inv(raw[:-1, 0], raw[:-1, 1], raw[1:, 0], raw[1:, 1])
            chainage = np.r_[0, np.cumsum(edge_lengths)]
            values = np.unique(np.r_[values, chainage[(chainage > values[0]) & (chainage < values[-1])]])
            points = stations(path, values)
            xy = project(points, center)
            # Test continuous source paths too, not only selected sample points.
            uncovered_length += shapely.difference(shapely.LineString(xy), boundary).length
            covered = shapely.contains_xy(boundary, xy[:, 0], xy[:, 1])
            test_count += len(xy)
            if not np.all(covered):
                outside.extend(shapely.distance(polygon, shapely.points(xy[~covered])).tolist())
            ranges.append({'pipeline_index': pipeline_index, 'path_index': path_index,
                           'start_m': float(values[0]), 'end_m': float(values[-1]), 'segments': len(ids)})
        desc = doc.find('.//k:description', NS).text
        row = {'index': index, 'pair': [section['pipeline_1'], section['pipeline_2']],
               'length_m': section['bundled_length_meters'], 'separation_m': section['average_separation'],
               'ranges': ranges, 'checked_points': test_count, 'outside_points': len(outside),
               'uncovered_source_length_m': uncovered_length,
               'max_outside_m': max(outside, default=0), 'valid_polygon': True,
               'vertices': len(coords), 'area_m2': polygon.area, 'description': desc}
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
                   max_outside_m=max((c['max_outside_m'] for c in corridors), default=0),
                   fallback_rectangles=sum('Geometry: sampled_curve.' not in c['description'] for c in corridors),
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
    failed = [row['case'] for row in rows if row.get('clipped_corridors', 0) or
              (not row['complete'] and row['case'] != 'step2-limit')]
    if failed:
        raise SystemExit('Audit failed: '+', '.join(failed))


if __name__ == '__main__':
    main()
