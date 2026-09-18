"""Compare the hash-pinned customer's stored mileage with every supplied path.

Investigation only: no geometry correction, production measurement change, or
customer source write. Independent XML enumeration is compared to the actual
repair session and analyzer. MILES metadata is never used as an analysis input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers.source import prepare_source

EXPECTED_SHA256 = '92bea274a0e588e2611e4a46b96b8a8f48d3b26f9089533a1b9dfbbd89f1eb4c'
NS = {'k': 'http://www.opengis.net/kml/2.2'}
SURVEY_MILE_METERS = 5280 * 1200 / 3937


def inspect(source: Path):
    before = source.read_bytes()
    assert hashlib.sha256(before).hexdigest() == EXPECTED_SHA256
    with zipfile.ZipFile(source) as archive:
        assert archive.namelist() == ['doc.kml']
        assert archive.testzip() is None
        raw = archive.read('doc.kml')
    # Independent, explicitly pinned patch, not production repair implementation.
    offset = 211
    insertion = b' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
    assert raw[offset:offset + 1] == b'>' and b'xmlns:xsi' not in raw
    effective = raw[:offset] + insertion + raw[offset:]
    assert effective[:offset] + effective[offset + len(insertion):] == raw
    root = ET.fromstring(effective)
    tags = Counter(element.tag.rsplit('}', 1)[-1] for element in root.iter())
    pms = root.findall('.//k:Placemark', NS)
    session = prepare_source(source)
    assert session.requires_repair
    session.approve()
    parsed = session.fresh_parse()
    app_rows, app_meters, app_miles = PipelineAnalyzer().calculate_pipeline_lengths(parsed.pipelines)
    assert len(pms) == len(parsed.pipelines) == len(app_rows) == 42
    geod = Geod(ellps='GRS80')
    rows, direct_paths = [], []
    observed_altitudes = set()
    max_meter_disagreement = 0.0
    for index, (pm, pipeline, app_row) in enumerate(zip(pms, parsed.pipelines, app_rows), 1):
        description = pm.findtext('k:description', namespaces=NS)
        # This exact table structure is verified in the pinned source; this is
        # not a general HTML metadata extractor or repair rule.
        pairs = re.findall(r'<td>\s*([^<>]+?)\s*</td>\s*<td>\s*([^<>]*?)\s*</td>', description, re.S)
        assert sum(key == 'MILES' for key, _ in pairs) == 1
        fields = dict(pairs)
        stored = Decimal(fields['MILES'])
        assert stored.is_finite() and stored >= 0
        elements = pm.findall('.//k:LineString/k:coordinates', NS)
        paths3d = [[tuple(map(float, token.split(','))) for token in element.text.split()]
                   for element in elements]
        assert all(len(path) >= 2 for path in paths3d)
        assert all(len(vertex) == 3 and all(map(math.isfinite, vertex))
                   for path in paths3d for vertex in path)
        observed_altitudes.update(vertex[2] for path in paths3d for vertex in path)
        paths = [[vertex[:2] for vertex in path] for path in paths3d]
        assert paths == pipeline['coordinate_paths']
        assert pm.findtext('k:name', namespaces=NS) == pipeline['name'] == app_row['Name']
        direct_paths.extend(paths)
        path_meters = [geod.line_length([v[0] for v in path], [v[1] for v in path]) for path in paths]
        meters = math.fsum(path_meters)
        disagreement = abs(meters - app_row['Shape_Length'])
        max_meter_disagreement = max(max_meter_disagreement, disagreement)
        assert disagreement < 0.000001
        vertices = [vertex for path in paths for vertex in path]
        rows.append({
            'record_number': index, 'name': pipeline['name'], 'placemark_id': pipeline['placemark_id'],
            'stored_MILES': str(stored), 'stored_FEET': fields.get('FEET'),
            'calculated_US_survey_miles': app_row['pipelinelength'],
            'stored_minus_calculated_miles': float(stored) - app_row['pipelinelength'],
            'direct_GRS80_meters': meters,
            'direct_GRS80_US_survey_miles': meters / SURVEY_MILE_METERS,
            'path_count': len(paths), 'vertex_count': len(vertices),
            'path_meters': path_meters,
            'longitude_range': [min(v[0] for v in vertices), max(v[0] for v in vertices)],
            'latitude_range': [min(v[1] for v in vertices), max(v[1] for v in vertices)],
            'coordinate_sequence_matches_application': True,
        })
    # All geometry locations, not only the records successfully matched above.
    coordinate_elements = root.findall('.//k:coordinates', NS)
    all_paths = [[tuple(map(float, token.split(',')[:2])) for token in element.text.split()]
                 for element in coordinate_elements]
    assert all_paths == direct_paths
    assert tags['LineString'] == tags['coordinates'] == len(direct_paths) == 44
    assert not any(tags[tag] for tag in ('Point', 'Polygon', 'LinearRing', 'Track', 'MultiTrack', 'NetworkLink'))
    stored_total = sum((Decimal(row['stored_MILES']) for row in rows), Decimal(0))
    # Investigation grouping only, not a proposed application tolerance.
    outliers = [row for row in rows if abs(row['stored_minus_calculated_miles']) > .1]
    remaining = [row for row in rows if row not in outliers]
    gap = float(stored_total) - app_miles
    outlier_gap = math.fsum(row['stored_minus_calculated_miles'] for row in outliers)
    assert source.read_bytes() == before
    return {
        'sample': source.name, 'sha256': EXPECTED_SHA256,
        'original_unchanged': True, 'only_patch': '54-byte missing xsi namespace insertion at byte211',
        'coverage': {'placemarks': len(pms), 'LineStrings': len(direct_paths),
                     'coordinate_blocks': len(coordinate_elements),
                     'vertices': sum(len(path) for path in direct_paths),
                     'altitudes': sorted(observed_altitudes),
                     'all_source_coordinate_sequences_match_application': True,
                     'duplicate_ids_matched_by_document_order_not_deduplicated': True,
                     'maximum_per_record_direct_vs_app_disagreement_meters': max_meter_disagreement},
        'totals': {'stored_MILES_decimal': str(stored_total), 'app_meters': app_meters,
                   'app_US_survey_miles': app_miles, 'stored_minus_calculated_miles': gap,
                   'gap_percent_of_stored': gap / float(stored_total) * 100},
        'outlier_grouping_threshold_miles': .1,
        'outlier_record_numbers': [row['record_number'] for row in outliers],
        'outlier_gap_miles': outlier_gap, 'outlier_share_of_net_gap_percent': outlier_gap / gap * 100,
        'remaining_records': len(remaining),
        'remaining_stored_miles': str(sum((Decimal(row['stored_MILES']) for row in remaining), Decimal(0))),
        'remaining_calculated_miles': math.fsum(row['calculated_US_survey_miles'] for row in remaining),
        'remaining_maximum_absolute_gap_miles': max(abs(row['stored_minus_calculated_miles']) for row in remaining),
        'remaining_maximum_relative_gap_percent': max(abs(row['stored_minus_calculated_miles']) / float(row['stored_MILES']) * 100 for row in remaining),
        'records': rows,
        'limitations': 'Stored attributes cannot establish whether the client exported its full routes. No overlap deductions or state allocations are applied in this original-length comparison.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ('record_number', 'name', 'stored_MILES', 'calculated_US_survey_miles',
               'stored_minus_calculated_miles', 'path_count', 'vertex_count')
    with args.csv.open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['records'])
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))
