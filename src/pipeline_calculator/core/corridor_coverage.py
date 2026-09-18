"""Original-path extents and bounded containment checks for corridor outlines."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import heapq
import math

from pipeline_calculator.core.segmentation import sampling_endpoint_allowance

MAX_COVERAGE_INSPECTIONS = 1_000_000


class QualifiedRunLimitError(ValueError):
    code = 'corridor_buffer_limit'


@dataclass(frozen=True)
class QualifiedPathRun:
    """A consecutive qualified interval; coordinates never connect source paths."""

    scope: str
    source_id: str | int
    source_index: int
    scope_path_index: int
    first_segment: int
    last_segment: int
    start_m: float
    end_m: float
    coordinates: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if (not isinstance(self.scope, str) or not isinstance(self.source_id, (str, int))
                or isinstance(self.source_id, bool)):
            raise ValueError('Qualified run scope and source ID must be JSON-native identities')
        for value in (self.source_index, self.scope_path_index, self.first_segment, self.last_segment):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError('Qualified run indices must be nonnegative integers')
        if self.last_segment < self.first_segment:
            raise ValueError('Qualified run segment range is reversed')
        if (not all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
                    for value in (self.start_m, self.end_m)) or not 0 <= self.start_m < self.end_m):
            raise ValueError('Qualified run chainage must be finite and increasing')
        points = tuple(tuple(float(value) for value in point) for point in self.coordinates)
        if len(points) < 2 or any(len(point) != 2 or not all(math.isfinite(v) for v in point) for point in points):
            raise ValueError('Qualified run requires finite two-dimensional source coordinates')
        object.__setattr__(self, 'coordinates', points)

    def provenance(self):
        return dict(scope=self.scope, source_id=self.source_id,
                    source_index=self.source_index, scope_path_index=self.scope_path_index,
                    first_segment=self.first_segment, last_segment=self.last_segment,
                    start_m=self.start_m, end_m=self.end_m)


def qualified_path_runs(pipelines, qualified, segment_length, geod, *,
                        scope='Combined', cache=None, context=None, max_points=100_000,
                        budget=None):
    """Extract all qualified samples, retaining bends but excluding sampled tails.

    ``cache`` belongs to one scope/pipeline collection, never to a worker globally.
    Index validation precedes MeasuredPath's legacy endpoint clamping.
    """
    step = float(segment_length)
    if not math.isfinite(step) or step <= 0:
        raise ValueError('Qualified run step must be finite and positive')
    pair, paths, ids_by_path = (qualified[key] for key in ('pair', 'paths', 'segment_ids'))
    if not (len(pair) == len(paths) == len(ids_by_path) == 2):
        raise ValueError('A qualified section must identify two paths')
    measured = {} if cache is None else cache
    result = []
    retained_points = 0
    if budget is not None:
        max_points = min(max_points, budget.max_section_points)
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 0:
        raise ValueError('Qualified extraction limit must be a nonnegative integer')

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    for pipeline_index, path_index, segment_ids in zip(pair, paths, ids_by_path):
        if context is not None:
            context.check()
        if not integer(pipeline_index) or pipeline_index >= len(pipelines) or not integer(path_index):
            raise ValueError('Qualified run references an invalid source/path')
        pipeline = pipelines[pipeline_index]
        segments = pipeline.get('segments', [])
        local_ids = set()
        for position, index in enumerate(segment_ids):
            if context is not None and position % 256 == 0:
                context.check()
            if position >= max_points:
                raise QualifiedRunLimitError('Qualified segment index count exceeds the extraction budget')
            if not integer(index) or index >= len(segments):
                raise ValueError('Qualified run references an invalid segment')
            segment = segments[index]
            local_index = segment.get('path_segment_index', index)
            if segment.get('path_index', 0) != path_index or not integer(local_index):
                raise ValueError('Qualified run segment belongs to another path')
            if not math.isclose(float(segment.get('length', step)), step, rel_tol=0, abs_tol=1e-9):
                raise ValueError('Qualified run segment has an inconsistent sample length')
            local_ids.add(local_index)
        if not local_ids:
            raise ValueError('Qualified run has no contributing samples')
        key = (pipeline_index, path_index)
        if key not in measured:
            # Match the existing path selection contract without copying every
            # unrelated path before the extraction budget can be checked.
            selected, valid_index = None, 0
            for position, candidate in enumerate(pipeline.get('coordinate_paths') or []):
                if context is not None and position % 256 == 0:
                    context.check()
                if position >= max_points:
                    raise QualifiedRunLimitError('Qualified path lookup exceeds its extraction budget')
                if candidate is not None and len(candidate) >= 2:
                    if valid_index == path_index:
                        selected = candidate
                        break
                    valid_index += 1
            if selected is None and valid_index == 0 and path_index == 0:
                selected = pipeline.get('coordinates') or []
            if selected is None or len(selected) < 2:
                raise ValueError('Qualified run path is missing')
            if len(selected) > max_points:
                raise QualifiedRunLimitError('Qualified path chainage cache exceeds its extraction budget')
            if budget is not None:
                budget.charge(len(selected))
            coordinates = []
            for position, point in enumerate(selected):
                if context is not None and position % 256 == 0:
                    context.check()
                lon, lat = map(float, point)
                if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
                    raise ValueError('Qualified run has invalid coordinates')
                coordinates.append((lon, lat))
            measured[key] = MeasuredPath(geod, coordinates, context=context)
        path = measured[key]
        if not math.isfinite(path.chainage[-1]) or path.chainage[-1] <= 0:
            raise ValueError('Qualified run has no positive measured extent')
        ordered = sorted(local_ids)
        ranges = []
        first = last = ordered[0]
        for index in ordered[1:]:
            if index != last + 1:
                ranges.append((first, last))
                first = index
            last = index
        ranges.append((first, last))
        source_id = pipeline.get('id', pipeline_index)
        if not isinstance(source_id, (str, int)) or isinstance(source_id, bool):
            raise ValueError('Qualified source ID must be a string or integer')
        for first, last in ranges:
            start, end = first * step, (last + 1) * step
            # Accept the same bounded terminal roundoff as sampling, then
            # retain the actual source endpoint rather than extending it.
            if not 0 <= start < end or end > path.chainage[-1] + sampling_endpoint_allowance(step):
                raise ValueError('Qualified interval exceeds its original path')
            end = min(end, path.chainage[-1])
            needed = 2 + max(0, bisect_left(path.chainage, end) - bisect_right(path.chainage, start))
            if retained_points + needed > max_points:
                raise QualifiedRunLimitError('Qualified run coordinates exceed the extraction budget')
            retained_points += needed
            points = tuple((float(p[0]), float(p[1])) for p in path.span(start, end))
            if not end > start:
                raise ValueError('Qualified run has no positive measured extent')
            result.append(QualifiedPathRun(str(scope), source_id, pipeline_index, path_index,
                                           first, last, float(start), float(end), points))
    return tuple(result)


class MeasuredPath:
    """Cache geodesic chainage once, then retain vertices inside each section."""
    def __init__(self, geod, coordinates, *, context=None):
        self.geod, self.coordinates = geod, coordinates
        self.chainage, self.bearings = [0.0], []
        for index, (a, b) in enumerate(zip(coordinates, coordinates[1:])):
            if context is not None and index % 256 == 0:
                context.check()
            bearing, _, distance = geod.inv(*a, *b)
            self.chainage.append(self.chainage[-1]+abs(distance))
            self.bearings.append(bearing)

    def point(self, station):
        station = max(0.0, min(self.chainage[-1], station))
        if station == self.chainage[-1]:
            return self.coordinates[-1]
        edge = min(bisect_right(self.chainage, station)-1, len(self.bearings)-1)
        return self.geod.fwd(*self.coordinates[edge], self.bearings[edge], station-self.chainage[edge])[:2]

    def span(self, start, end):
        first, last = bisect_right(self.chainage, start), bisect_left(self.chainage, end)
        return [self.point(start), *self.coordinates[first:last], self.point(end)]


def covers_paths(ring, paths, *, context=None):
    """Require every path vertex inside and no source edge crossing the outline.

    Sweep edge bounds along the longer axis. If geometry exhausts the work
    budget, return False so the caller uses its enclosing rectangle instead.
    This checks containment, not polygon topology (validated during export).
    """
    points = [point for path in paths for point in path]
    if not points or len(ring) < 4:
        return False
    swap = max(p[1] for p in ring)-min(p[1] for p in ring) > max(p[0] for p in ring)-min(p[0] for p in ring)
    def convert(p):
        return (p[1], p[0]) if swap else p
    ring = [convert(p) for p in ring]
    paths = [[convert(p) for p in path] for path in paths]
    edges = [(a, b) for a, b in zip(ring, ring[1:])]
    bounds = sorted((min(a[0], b[0]), max(a[0], b[0]), i) for i, (a, b) in enumerate(edges))
    active, endings, cursor, inspections = {}, [], 0, 0
    eps = 1e-8

    def cross(a, b, p):
        return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])

    for x, y in sorted(p for path in paths for p in path):
        if context is not None:
            context.checkpoint()
        while cursor < len(bounds) and bounds[cursor][0] <= x+eps:
            _, xmax, i = bounds[cursor]
            active[i] = edges[i]
            heapq.heappush(endings, (xmax, i))
            cursor += 1
        while endings and endings[0][0] < x-eps:
            _, i = heapq.heappop(endings)
            active.pop(i, None)
        inside, boundary = False, False
        for a, b in active.values():
            inspections += 1
            if inspections > MAX_COVERAGE_INSPECTIONS:
                return False
            if abs(cross(a, b, (x, y))) <= eps and min(a[1], b[1])-eps <= y <= max(a[1], b[1])+eps:
                boundary = True
                break
            if (a[0] > x) != (b[0] > x):
                crossing = a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0])
                if y < crossing:
                    inside = not inside
        if not inside and not boundary:
            return False

    # Interior vertices alone are insufficient: a straight source edge could
    # cross a concavity between them. Reject every proper boundary crossing.
    source_edges = sorted((min(a[0], b[0]), max(a[0], b[0]), a, b)
                          for path in paths for a, b in zip(path, path[1:]))
    active, endings, cursor = {}, [], 0
    for xmin, xmax, a, b in source_edges:
        if context is not None:
            context.checkpoint()
        while cursor < len(bounds) and bounds[cursor][0] <= xmax+eps:
            _, end, i = bounds[cursor]
            active[i] = edges[i]
            heapq.heappush(endings, (end, i))
            cursor += 1
        while endings and endings[0][0] < xmin-eps:
            _, i = heapq.heappop(endings)
            active.pop(i, None)
        for c, d in active.values():
            inspections += 1
            if inspections > MAX_COVERAGE_INSPECTIONS:
                return False
            if min(c[0], d[0]) > xmax+eps or max(c[1], d[1]) < min(a[1], b[1])-eps or min(c[1], d[1]) > max(a[1], b[1])+eps:
                continue
            if cross(a, b, c)*cross(a, b, d) < -eps and cross(c, d, a)*cross(c, d, b) < -eps:
                return False
    return True
