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
from shapely.affinity import translate
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


def _valid_coordinate_path(coordinates):
    return (isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2 and
            all(isinstance(point, (list, tuple)) and len(point) == 2 and
                all(type(value) in (float, int) and math.isfinite(value) for value in point) and
                -180 <= point[0] <= 180 and -90 <= point[1] <= 90 for point in coordinates))


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
                paths = row.get('paths')
                path_count = len(paths) if isinstance(paths, (list, tuple)) else None
                check(f'{name}: independent path count', path_count, 1)
                coordinates = paths[0] if path_count == 1 else None
            valid_coordinates = _valid_coordinate_path(coordinates)
            check(f'{name}: source span has valid coordinates', valid_coordinates, True)
            if valid_coordinates:
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


def _section_corridor_evidence(section, inputs, lines, profile):
    """Bound every supported visualization using independent source geometry.

    This is a conservative extent certificate, not a golden corridor polygon.
    A curve may use 6-half-width miters and endpoint caps; a rotated rectangle
    may enclose a bent section; the last-resort geographic box adds .001 degree.
    None licenses geometry arbitrarily far from the *individual* section.
    Derive one enclosing box for all three choices without constructing any of
    the production candidates or reading the exported shape's claimed bounds.
    """
    sample = profile.get('segment_meters', 5.0)
    detection = profile.get('detection_meters', 15.0)
    support, midpoints = [], []
    for key, path, ranges in zip(section['source_keys'], section['path_indices'], section['coverage_ranges']):
        identity = key, path
        if identity not in lines:
            lines[identity] = MeasuredLine(inputs[key]['paths'][path])
        line = lines[identity]
        for start, end in ranges:
            lower, upper = start * sample, end * sample
            support.extend(line.point(station) for station in line.chainage if lower < station < upper)
            support.extend(line.point(index * sample) for index in range(start, end + 1))
            for index in range(start, end):
                a, b = line.point(index * sample), line.point((index + 1) * sample)
                inverse = GEOD.Inverse(a[1], a[0], b[1], b[0])
                chord = GEOD.Direct(a[1], a[0], inverse['azi1'], inverse['s12'] / 2)
                support.append((chord['lon2'], chord['lat2']))
                midpoints.append(line.point((index + .5) * sample))
    anchor = support[0][0]
    support = [(lon + 360 * round((anchor - lon) / 360), lat) for lon, lat in support]
    midpoints = [(lon + 360 * round((anchor - lon) / 360), lat) for lon, lat in midpoints]
    west, east = min(p[0] for p in support), max(p[0] for p in support)
    south, north = min(p[1] for p in support), max(p[1] for p in support)
    center = ((west + east) / 2, (south + north) / 2)
    # 5 cm covers the 1 cm cut criterion plus 7-decimal visualization rounding.
    # This margin affects the extent guard only, never state containment/mileage.
    radius = max(_distance(center, point) for point in support) + .05
    midpoint_radius = radius + math.hypot(detection, sample) / 2
    pad = max(1.5 * sample, 5.0)
    # Representative pair midpoints lie within midpoint_radius. Endpoint caps
    # extend by at most R + midpoint_radius + 1; miters by 6 * half_width.
    extent = max(midpoint_radius + max(pad, radius + midpoint_radius + 1) + detection,
                 midpoint_radius + 6 * detection,
                 math.sqrt(2) * max(radius + pad, midpoint_radius + detection))
    # Minimum GRS80 curvature radius is greater than 6,330,000 m. These angular
    # bounds therefore enclose the entire radial disk, including high latitudes.
    latitude_pad = math.degrees(extent / 6_330_000.0)
    latitude_limit = abs(center[1]) + latitude_pad
    if latitude_limit >= 85:
        raise ValueError('Corridor extent certification exceeds the reference nonpolar domain')
    longitude_pad = latitude_pad / math.cos(math.radians(latitude_limit))
    epsilon = 1e-7  # one visualization coordinate unit, including box rounding
    envelope = box(min(west - .001, center[0] - longitude_pad) - epsilon,
                   min(south - .001, center[1] - latitude_pad) - epsilon,
                   max(east + .001, center[0] + longitude_pad) + epsilon,
                   max(north + .001, center[1] + latitude_pad) + epsilon)
    return envelope, midpoints, anchor


def _distinct_section_assignment(candidates, section_count):
    """Bipartite augmenting paths: every map must match a distinct section."""
    assigned = {}

    def assign(corridor, visited):
        for section in candidates[corridor]:
            if section in visited:
                continue
            visited.add(section)
            if section not in assigned or assign(assigned[section], visited):
                assigned[section] = corridor
                return True
        return False

    return (len(candidates) == section_count and
            all(assign(corridor, set()) for corridor in range(len(candidates))))


def compare_export_corridors(check, label, corridors, expected, sources, id_to_key):
    """Each section needs its own bounded exported area covering its samples.

    A union across every section of a source pair is insufficient: a duplicated
    map could cover another section, or append a remote island, yet pass totals.
    Match whole corridor placemarks one-to-one to independent section evidence.
    Multipart clipped polygons and holes remain supported.
    """
    from collections import Counter
    actual_pairs = []
    by_pair = defaultdict(list)
    for corridor in corridors:
        pair = tuple(sorted(id_to_key.get(int(corridor['metadata'].get(f'pipeline_{i}_id', -1)), '<unknown>')
                            for i in (1, 2)))
        actual_pairs.append(pair)
        by_pair[pair].append(corridor['polygons'])
    sections = expected_section_records(expected)
    check(f'{label}: exported corridor section identities', sorted(Counter(actual_pairs).items()),
          sorted(Counter(tuple(s['source_keys']) for s in sections).items()))
    inputs = {source['key']: source for source in sources}
    lines = {}
    evidence = defaultdict(list)
    for section in sections:
        evidence[tuple(section['source_keys'])].append(
            _section_corridor_evidence(section, inputs, lines, expected.get('profile', {})))
    valid = all(polygons and all(not p.is_empty and p.is_valid and
                                all(math.isfinite(value) for value in p.bounds) for p in polygons)
                for maps in by_pair.values() for polygons in maps)
    check(f'{label}: exported corridor polygons valid', valid, True)
    missing = 0
    bounded, matched = True, True
    for pair in sorted(set(by_pair) | set(evidence)):
        candidates = []
        targets = evidence[pair]
        best_missing = [len(midpoints) for _, midpoints, _ in targets]
        for polygons in by_pair[pair]:
            eligible, has_bound = [], False
            if not polygons or any(p.is_empty or not p.is_valid for p in polygons):
                candidates.append(eligible)
                bounded = False
                continue
            for index, (envelope, midpoints, anchor) in enumerate(targets):
                # Map exports split dateline components; integer shifts preserve
                # their native coordinates while comparing one local section.
                shifted = [translate(p, xoff=360 * round((anchor - p.centroid.x) / 360)) for p in polygons]
                contained = all(envelope.covers(p) for p in shifted)
                has_bound |= contained
                shape = unary_union(shifted)
                epsilon = 64 * max(math.ulp(value) for value in shape.bounds)
                shape = shape.buffer(epsilon)
                absent = sum(not shape.covers(Point(point)) for point in midpoints)
                best_missing[index] = min(best_missing[index], absent)
                if contained and not absent:
                    eligible.append(index)
            bounded &= has_bound
            candidates.append(eligible)
        missing += sum(best_missing)
        matched &= _distinct_section_assignment(candidates, len(targets))
    check(f'{label}: exported corridors cover qualified path samples', missing, 0)
    check(f'{label}: exported corridors stay within independent section bounds', bounded, True)
    check(f'{label}: exported corridors match independent sections one-to-one', matched, True)
