"""Original-path extents and bounded containment checks for corridor outlines."""
from bisect import bisect_left, bisect_right
import heapq

MAX_COVERAGE_INSPECTIONS = 1_000_000


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
