"""Pure comparison helpers: application observations never become expectations.

The source keys, interval spans and sampled sets all come from the independently
frozen JSON. These helpers deliberately contain no application calculation code.
"""
from __future__ import annotations

from collections import defaultdict
import bisect
import json
import math

from geographiclib.geodesic import Geodesic
from shapely.geometry import Point, box
from shapely.ops import unary_union

GEOD = Geodesic(6378137.0, 1 / 298.257222101)
# Public clipping acceptance target in the fixture request, not a mileage budget.
CUT_TARGET_METERS = .01


def runs(indices):
    result = []
    for index in sorted(set(indices)):
        if result and result[-1][1] == index:
            result[-1][1] = index + 1
        else:
            result.append([index, index + 1])
    return result


SECTION_FIELDS = ('source_keys', 'path_indices', 'coverage_ranges',
                  'coverage_sample_counts', 'eligible_match_count', 'length_meters')


def expected_section_records(expected):
    return sorted(({key: section[key] for key in SECTION_FIELDS}
                   for section in expected['sections'] if section['qualified']),
                  key=lambda row: json.dumps(row, sort_keys=True))


def section_records(pipelines, sections, id_to_key):
    result = []
    for section in sections:
        sides = []
        for pipe, path, indices in zip(section['pair'], section['paths'], section['segment_ids']):
            pipeline = pipelines[pipe]
            local = {pipeline['segments'][index]['path_segment_index'] for index in indices}
            sides.append((id_to_key[pipeline['id']], path, runs(local), len(local)))
        sides.sort(key=lambda side: side[0])
        result.append(dict(source_keys=[s[0] for s in sides], path_indices=[s[1] for s in sides],
                           coverage_ranges=[s[2] for s in sides], coverage_sample_counts=[s[3] for s in sides],
                           eligible_match_count=len(section['matches']), length_meters=section['length']))
    return sorted(result, key=lambda row: json.dumps(row, sort_keys=True))


def compare_sampled_contract(check, label, pipelines, sections, public_overlap, expected, id_to_key):
    """Check exact sample coverage and public attribution, not just equal savings."""
    actual = section_records(pipelines, sections, id_to_key)
    targets = expected_section_records(expected)
    check(f'{label}: qualifying sample sections', actual, targets)
    check(f'{label}: source sample identities', sorted(id_to_key[p['id']] for p in pipelines),
          sorted(expected['source_coverage']))
    covered = defaultdict(set)
    for row in actual:
        for key, path, ranges in zip(row['source_keys'], row['path_indices'], row['coverage_ranges']):
            covered[key].update((path, sample) for start, end in ranges for sample in range(start, end))
    for pipeline in pipelines:
        key = id_to_key[pipeline['id']]
        target = expected['source_coverage'].get(key)
        if target is None:
            continue
        counts = [sum(s['path_index'] == path for s in pipeline['segments'])
                  for path in range(len(pipeline['coordinate_paths']))]
        check(f'{label}/{key}: path sample counts', counts, target['path_sample_counts'])
        coverage = {str(path): runs(i for p, i in covered[key] if p == path)
                    for path in range(len(counts))}
        check(f'{label}/{key}: unique sample coverage', coverage, target['qualifying_coverage_ranges_by_path'])
        public = (public_overlap or {}).get('pipeline_overlaps_by_id', {}).get(str(pipeline['id']))
        # Empty and singleton scopes legitimately omit the overlap detail table.
        if len(pipelines) >= 2:
            check(f'{label}/{key}: public covered samples', None if public is None else public['bundled_segments'],
                  target['qualifying_covered_sample_count'])
            check(f'{label}/{key}: public covered meters', None if public is None else public['bundled_length_meters'],
                  target['qualifying_covered_meters'], 1e-7)
    # Public sections retain source IDs, clipped-path IDs, lengths and counts.
    public_rows = []
    for row in (public_overlap or {}).get('bundled_sections', []):
        sides = sorted((id_to_key[row[f'pipeline_{i+1}_id']], row['source_path_indices'][i]) for i in range(2))
        public_rows.append((sides, row['segment_count'], row['bundled_length_meters']))
    expected_rows = [(list(zip(r['source_keys'], r['path_indices'])), min(r['coverage_sample_counts']), r['length_meters'])
                     for r in targets]
    check(f'{label}: public section source/path attribution', sorted(public_rows), sorted(expected_rows))
    check(f'{label}: pairwise qualifying meters', (public_overlap or {}).get('total_bundled_length', 0),
          expected['qualifying_pairwise_meters'], 1e-7)


def _distance(a, b):
    return GEOD.Inverse(a[1], a[0], b[1], b[0])['s12']


class MeasuredLine:
    """GeographicLib path sampling, independent of production path helpers."""
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.chainage = [0.0]
        self.edges = []
        for a, b in zip(coordinates, coordinates[1:]):
            inverse = GEOD.Inverse(a[1], a[0], b[1], b[0])
            self.chainage.append(self.chainage[-1] + inverse['s12'])
            self.edges.append(GEOD.Line(a[1], a[0], inverse['azi1']))

    def point(self, station):
        station = max(0.0, min(self.chainage[-1], station))
        if station == self.chainage[-1]:
            return self.coordinates[-1]
        index = min(len(self.edges)-1, bisect.bisect_right(self.chainage, station)-1)
        point = self.edges[index].Position(station-self.chainage[index])
        return point['lon2'], point['lat2']


def geometry_span_error(actual_coordinates, expected_coordinates):
    """Compare endpoints, every vertex and each intervening geodesic midpoint.

    Sampling at both paths' vertices catches bent-path shortcuts and invented
    detours, even when endpoints and aggregate distance have been preserved.
    """
    actual, expected = MeasuredLine(actual_coordinates), MeasuredLine(expected_coordinates)
    stations = sorted(set(actual.chainage + expected.chainage))
    stations += [(a+b)/2 for a, b in zip(stations, stations[1:])]
    error = max((_distance(actual.point(s), expected.point(s)) for s in stations), default=0.0)
    return max(error, abs(actual.chainage[-1]-expected.chainage[-1]))


def compare_interval_ledger(check, label, fragments, expected_intervals, id_to_key, *, export=False, originals=None):
    """Compare disjoint ordered spans per stable source/path and their geometry."""
    actual, expected = defaultdict(list), defaultdict(list)
    for row in fragments:
        key = id_to_key.get(int(row['source_id']), f"unknown:{row['source_id']}")
        actual[key, int(row['path_index'])].append(row)
    for row in expected_intervals:
        expected[row['key'], row['path_index']].append(row)
    check(f'{label}: interval source/path identities', sorted(actual), sorted(expected))
    for key in sorted(set(actual) | set(expected)):
        observed = sorted(actual[key], key=lambda row: float(row['start_m']))
        target = sorted(expected[key], key=lambda row: row['start_m'])
        prefix = f'{label}/{key[0]}/path{key[1]}'
        check(f'{prefix}: interval count', len(observed), len(target))
        expected_length = math.fsum(row['length_meters'] for row in target)
        check(f'{prefix}: interval mileage conservation', math.fsum(float(row['length_meters']) for row in observed),
              expected_length, max(.001, expected_length*1e-10))
        check(f'{prefix}: positive finite interval lengths',
              all(math.isfinite(float(row['length_meters'])) and float(row['length_meters']) > 0 for row in observed), True)
        check(f'{prefix}: interval length matches chainage',
              all(abs(float(row['length_meters']) - (float(row['end_m'])-float(row['start_m']))) <= 1e-8 for row in observed), True)
        if not export:
            check(f'{prefix}: intervals have no gaps or double counting',
                  all(abs(float(a['end_m'])-float(b['start_m'])) <= 1e-8 for a, b in zip(observed, observed[1:])), True)
        # Continue checking existing spans after a count failure. Otherwise a
        # known extra tiny interval could conceal a new corruption on this path.
        candidates = sorted((abs(float(row['start_m'])-gold['start_m']) +
                             abs(float(row['end_m'])-gold['end_m']), a, b)
                            for a, row in enumerate(observed) for b, gold in enumerate(target))
        matched_actual, matched_expected, pairs = set(), set(), []
        for _, a, b in candidates:
            if a not in matched_actual and b not in matched_expected:
                matched_actual.add(a)
                matched_expected.add(b)
                pairs.append((b, observed[a], target[b]))
        for index, row, gold in sorted(pairs, key=lambda pair: pair[0]):
            name = f'{prefix}/interval{index}'
            if not export:
                check(f'{name}: ownership', ('interior' if row['kind'] == 'state' else row['kind'], row['state_codes']),
                      (gold['kind'], gold['state_codes']))
            # Endpoints must meet the requested 1 cm location criterion. Length
            # conservation remains a separate stricter 1 mm check elsewhere.
            bound = CUT_TARGET_METERS + gold['cut_error_bound_meters']
            for field in ('start_m', 'end_m'):
                check(f'{name}: {field}', float(row[field]), gold[field], bound)
            check(f'{name}: measured length', float(row['length_meters']), gold['length_meters'], 2*bound)
            coordinates = row.get('coordinates')
            if export:
                paths = row['paths']
                check(f'{name}: independent path count', len(paths), 1)
                coordinates = paths[0] if len(paths) == 1 else None
            if coordinates is not None:
                check(f'{name}: follows independent source span', geometry_span_error(coordinates, gold['coordinates']), 0.0, 2*bound)
                if originals is not None and key[0] in originals:
                    source = MeasuredLine(originals[key[0]]['paths'][key[1]])
                    measured = MeasuredLine(coordinates)
                    # Cut locations may vary along the source within 1 cm, but
                    # that does not permit moving a near-border line sideways.
                    stations = measured.chainage + [(a+b)/2 for a, b in zip(measured.chainage, measured.chainage[1:])]
                    error = max(_distance(measured.point(s), source.point(float(row['start_m'])+s)) for s in stations)
                    check(f'{name}: lies on original geodesic', error, 0.0, 1e-5)


def polygon_containment(polygons, boundary):
    """Return a local, float-resolution containment test with an explicit bound.

    GEOS intersections can leave sub-ULP slivers. Allow only a strip 64 floating
    coordinate ULPs wide along the *local canonical boundary*, plus the matching
    area bound P*epsilon + pi*epsilon^2. Holes remain boundaries. No meter-scale
    boundary buffer, percentage-of-state-area tolerance or polygon-only vertex
    test is used. Typical width at longitude -103 is < 0.1 micrometer.
    """
    rows = []
    for polygon in polygons:
        values = polygon.bounds
        epsilon = 64 * max(math.ulp(value) for value in values)
        outside = polygon.difference(boundary)
        minx, miny, maxx, maxy = values
        local = boundary.boundary.intersection(box(minx-epsilon, miny-epsilon, maxx+epsilon, maxy+epsilon))
        bound = local.length * epsilon + math.pi*epsilon**2
        # A microscopic foreign polygon far from the border must still fail.
        residual = outside.difference(local.buffer(epsilon)).area if not outside.is_empty else 0.0
        rows.append({'outside_area_degrees2': outside.area, 'area_bound_degrees2': bound,
                     'coordinate_tolerance_degrees': epsilon, 'outside_numerical_strip_area_degrees2': residual,
                     'valid': polygon.is_valid, 'passed': polygon.is_valid and outside.area <= bound and residual == 0})
    return rows


def compare_export_corridors(check, label, corridors, expected, sources, id_to_key):
    """Each expected section needs an exported area covering its sampled paths."""
    from collections import Counter
    actual_pairs = []
    by_pair = defaultdict(list)
    for corridor in corridors:
        pair = tuple(sorted(id_to_key.get(int(corridor['metadata'].get(f'pipeline_{i}_id', -1)), '<unknown>')
                            for i in (1, 2)))
        actual_pairs.append(pair)
        by_pair[pair].extend(corridor['polygons'])
    sections = expected_section_records(expected)
    check(f'{label}: exported corridor section identities', sorted(Counter(actual_pairs).items()),
          sorted(Counter(tuple(s['source_keys']) for s in sections).items()))
    inputs = {source['key']: source for source in sources}
    lines, unions = {}, {pair: unary_union(polygons) for pair, polygons in by_pair.items()}
    missing = 0
    for section in sections:
        shape = unions.get(tuple(section['source_keys']))
        if shape is None:
            missing += sum(section['coverage_sample_counts'])
            continue
        epsilon = 64 * max(math.ulp(value) for value in shape.bounds)
        shape = shape.buffer(epsilon)
        for key, path, ranges in zip(section['source_keys'], section['path_indices'], section['coverage_ranges']):
            line_key = key, path
            if line_key not in lines:
                lines[line_key] = MeasuredLine(inputs[key]['paths'][path])
            for start, end in ranges:
                for sample in range(start, end):
                    midpoint = lines[line_key].point((sample+.5)*5.0)
                    missing += not shape.covers(Point(midpoint))
    check(f'{label}: exported corridors cover qualified path samples', missing, 0)
