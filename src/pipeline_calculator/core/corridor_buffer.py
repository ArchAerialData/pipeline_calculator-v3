"""Bounded, metric buffers of qualified geodesic paths, independent of mileage.

Every native operation is preceded by an input/output-expansion preflight. A
failure omits the whole section, never a component, and never invents a rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from shapely import get_num_coordinates
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from pipeline_calculator.core.corridor_coverage import MeasuredPath
from pipeline_calculator.core.execution import AnalysisCancelled


class _MapFailure(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CorridorDisplayOptions:
    padding_m: float = 5.0
    approximation_target_m: float = 0.05
    max_chunk_m: float = 20_000.0
    max_chart_radius_m: float = 25_000.0
    source_step_m: float = 100.0
    source_error_m: float = 0.005
    output_error_m: float = 0.005
    arc_error_m: float = 0.005
    max_depth: int = 20

    def __post_init__(self):
        for key in ('padding_m', 'approximation_target_m', 'max_chunk_m', 'max_chart_radius_m',
                    'source_step_m', 'source_error_m', 'output_error_m', 'arc_error_m'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be a finite positive number')
        if self.padding_m != 5 or self.approximation_target_m != .05:
            raise ValueError('This corridor policy uses 5 m padding and a 0.05 m approximation target')
        if (self.max_chunk_m > 20_000 or self.max_chart_radius_m > 25_000 or self.source_step_m > 100
                or max(self.source_error_m, self.output_error_m, self.arc_error_m) > .005
                or not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool) or not 0 <= self.max_depth <= 20):
            raise ValueError('Unsupported corridor display precision/domain options')


@dataclass
class CorridorGeometryBudget:
    """Caller-owned cumulative budget shared by Combined and every state.

    A fresh _SectionBudget is created automatically per build call. Counters are
    never reset by the builder. Failed work is charged; omitted output is not
    retained. Preflight reservations are checked before allocation, actual work
    is charged after the operation, and unused reservation is released.
    """
    max_work_vertices: int = 2_000_000
    max_charts: int = 10_000
    max_retained_vertices: int = 1_000_000
    max_section_points: int = 100_000
    max_section_charts: int = 1_024
    max_parts: int = 4_096
    max_rings: int = 8_192
    max_native_vertices: int = 4_096
    max_native_operands: int = 32
    # Full immutable state geometry is reused for read-only certification only.
    # New geometry construction must continue to use the much smaller native cap.
    max_boundary_vertices: int = 2_000_000
    max_boundary_queries: int = 10_000
    work_vertices: int = 0
    charts: int = 0
    retained_vertices: int = 0
    boundary_queries: int = 0

    def __post_init__(self):
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in vars(self).values()):
            raise ValueError('Corridor work limits and counters must be nonnegative integers')
        if self.max_native_operands < 2:
            raise ValueError('Corridor union batches require at least two operands')

    def preflight(self, vertices):
        if vertices < 0 or self.work_vertices + vertices > self.max_work_vertices:
            raise _MapFailure('corridor_buffer_limit', 'Corridor job construction budget exceeded')

    def charge(self, vertices):
        self.preflight(vertices)
        self.work_vertices += vertices

    def retain(self, vertices):
        if self.retained_vertices + vertices > self.max_retained_vertices:
            raise _MapFailure('corridor_buffer_limit', 'Corridor job output budget exceeded')
        self.retained_vertices += vertices

    def release_retained(self, vertices):
        """Release a replaced/omitted shape, never permitting counter underflow."""
        if not isinstance(vertices, int) or isinstance(vertices, bool) or vertices < 0:
            raise ValueError('Released corridor vertex count must be a nonnegative integer')
        if vertices > self.retained_vertices:
            raise ValueError('Cannot release more corridor vertices than are retained')
        self.retained_vertices -= vertices

    def boundary_query(self, vertices):
        """Preflight one read-only predicate against an immutable state operand.

        This separate bound is for covers/final strict difference certification,
        not a way to submit a large newly constructed corridor to an overlay.
        Caller checks cancellation immediately before and after the native call.
        """
        if not isinstance(vertices, int) or isinstance(vertices, bool) or vertices <= 0:
            raise ValueError('Boundary operand vertex count must be a positive integer')
        if vertices > self.max_boundary_vertices:
            raise _MapFailure('corridor_buffer_limit', 'Immutable state boundary exceeds its predicate vertex limit')
        if self.boundary_queries >= self.max_boundary_queries:
            raise _MapFailure('corridor_buffer_limit', 'Corridor boundary predicate count exceeded')
        self.boundary_queries += 1


class _SectionBudget:
    def __init__(self, job, context):
        self.job, self.context = job, context
        self.points, self.charts = 0, 0

    def check(self):
        if self.context is not None:
            self.context.check()

    def charge(self, count, *, construction=False):
        self.check()
        self.job.charge(count)
        if construction:
            if self.points + count > self.job.max_section_points:
                raise _MapFailure('corridor_buffer_limit', 'Corridor section construction point limit exceeded')
            self.points += count

    def chart(self):
        self.check()
        if self.charts >= self.job.max_section_charts or self.job.charts >= self.job.max_charts:
            raise _MapFailure('corridor_buffer_limit', 'Corridor chart limit exceeded')
        self.charts += 1
        self.job.charts += 1

    def native(self, shapes, *, expansion=None):
        self.check()
        count = sum(int(get_num_coordinates(shape)) for shape in shapes)
        if len(shapes) > self.job.max_native_operands or count > self.job.max_native_vertices:
            raise _MapFailure('corridor_buffer_limit', 'Corridor native operation input limit exceeded')
        amount = count if expansion is None else max(count, expansion)
        if amount > self.job.max_section_points:
            raise _MapFailure('corridor_buffer_limit', 'Corridor intermediate output expansion exceeds its limit')
        self.job.preflight(amount + count)
        self.charge(count)

    def output(self, shape, estimate):
        self.check()
        count = int(get_num_coordinates(shape))
        if count > estimate or count > self.job.max_section_points:
            raise _MapFailure('corridor_buffer_limit', 'Corridor native output exceeded its reserved capacity')
        self.charge(count)
        return shape


class _Chart:
    def __init__(self, origin, geod, options, work):
        self.origin, self.geod, self.options, self.work = origin, geod, options, work

    def forward(self, point):
        self.work.charge(1)
        azimuth, _, distance = self.geod.inv(*self.origin, *point)
        if not math.isfinite(distance) or distance + self.options.padding_m > self.options.max_chart_radius_m:
            raise _MapFailure('corridor_projection_unavailable', 'Corridor exceeds the local projection domain')
        angle = math.radians(azimuth)
        return distance * math.sin(angle), distance * math.cos(angle)

    def inverse(self, point):
        self.work.charge(1)
        distance = math.hypot(*point)
        if not math.isfinite(distance) or distance > self.options.max_chart_radius_m:
            raise _MapFailure('corridor_projection_unavailable', 'Corridor padding exceeds the projection domain')
        lon, lat, _ = self.geod.fwd(*self.origin, math.degrees(math.atan2(*point)), distance)
        if not math.isfinite(lon) or not math.isfinite(lat) or abs(lat) >= 89.999999:
            raise _MapFailure('corridor_projection_unavailable', 'Polar corridor display is unavailable')
        lon += 360 * round((self.origin[0] - lon) / 360)
        return lon, lat


def _parts(shape):
    if shape.geom_type == 'Polygon':
        yield shape
    elif hasattr(shape, 'geoms'):
        for item in shape.geoms:
            yield from _parts(item)


def _edges(shapes):
    for shape in shapes:
        if shape.geom_type == 'LineString':
            rings = [shape.coords]
        else:
            rings = [ring.coords for polygon in _parts(shape)
                     for ring in [polygon.exterior, *polygon.interiors]]
        for ring in rings:
            for a, b in zip(ring, list(ring)[1:]):
                yield a, b


def _overlay_bound(shapes, work):
    """Count possible segment intersections through bounded envelope queries.

    Every actual intersection belongs to this superset. GEOS noding adds at most
    two edge vertices per pair plus original coordinates and closure vertices.
    """
    work.native(shapes)
    edges = list(_edges(shapes))
    work.charge(len(edges))
    segments = [LineString(edge) for edge in edges]
    tree = STRtree(segments)
    count = 0
    for index, segment in enumerate(segments):
        work.check()
        candidates = tree.query(segment)
        work.charge(len(candidates))
        count += sum(int(other) > index for other in candidates)
        if len(edges) * 2 + count * 4 + 16 > work.job.max_section_points:
            raise _MapFailure('corridor_buffer_limit', 'Corridor overlay complexity exceeds its limit')
    estimate = len(edges) * 2 + count * 4 + 16
    work.job.preflight(estimate)
    return estimate


def _buffer(line, radius, quadrants, work):
    count = int(get_num_coordinates(line))
    estimate = (count + 2) * (4 * quadrants + 8)
    work.native([line], expansion=estimate)
    # A round buffer is bounded by parallel source-edge sides and circular
    # vertex/end-cap arcs. Count possible *nonlocal* primitive intersections
    # before GEOS can node a self-crossing/backtracking stroke. Local incident
    # joins are already included in the per-vertex reserve above. A line can cut
    # a convex circle polygon at most twice; two circle polygons can contribute
    # at most twice their combined quadrant-edge count. Envelope hits overcount
    # actual intersections, deliberately including all near-coincident cases.
    coordinates = list(line.coords)
    primitives, owners, kinds = [], [], []
    for index, (x, y) in enumerate(coordinates):
        primitives.append(box(x - radius, y - radius, x + radius, y + radius))
        owners.append({index})
        kinds.append('circle')
    for index, (a, b) in enumerate(zip(coordinates, coordinates[1:])):
        distance = math.dist(a, b)
        if distance == 0:
            continue
        nx, ny = -(b[1] - a[1]) / distance * radius, (b[0] - a[0]) / distance * radius
        for sign in (-1, 1):
            left, right = (a[0] + sign * nx, a[1] + sign * ny), (b[0] + sign * nx, b[1] + sign * ny)
            primitives.append(LineString([left, right]))
            owners.append({index, index + 1})
            kinds.append('side')
    work.charge(sum(int(get_num_coordinates(p)) for p in primitives))
    tree = STRtree(primitives)
    for index, primitive in enumerate(primitives):
        work.check()
        candidates = tree.query(primitive)
        work.charge(len(candidates))
        for other in candidates:
            other = int(other)
            if other <= index or owners[index] & owners[other]:
                continue
            if kinds[index] == kinds[other] == 'circle':
                separation = math.dist(coordinates[index], coordinates[other])
                intersections = _circle_intersection_bound(separation, radius, quadrants)
            else:
                intersections = 2
            estimate += 4 * intersections
            if estimate > work.job.max_section_points:
                raise _MapFailure('corridor_buffer_limit', 'Corridor self-overlap expansion exceeds its limit')
    work.native([line], expansion=estimate)
    result = line.buffer(radius, quad_segs=quadrants, cap_style='round', join_style='round')
    work.output(result, estimate)
    work.native([result])
    if result.is_empty or not result.is_valid:
        raise _MapFailure('corridor_geometry_invalid', 'Corridor buffer is empty or invalid')
    return result


def _circle_intersection_bound(distance, radius, quadrants):
    """Bound possible polygonal circle intersections using their radial annuli.

    Circle chords lie between r*cos(pi/(2*q)) and r (conservative for arbitrary
    round-join phase). Law of cosines bounds the only angular windows in which
    two such boundaries can intersect. A straight edge meets a convex polygon
    at most twice; extra endpoint edges cover both window endpoints and phases.
    Near-coincident circles retain the full conservative combinatorial bound.
    """
    if distance > 2 * radius:
        return 0
    if distance <= 1e-10:
        return 8 * quadrants
    lower = radius * math.cos(math.pi / (2 * quadrants))
    values = []
    for opposite in (lower, radius):
        candidates = [lower, radius]
        critical_squared = distance * distance - opposite * opposite
        if critical_squared > 0 and lower <= math.sqrt(critical_squared) <= radius:
            candidates.append(math.sqrt(critical_squared))
        values.extend((u * u + distance * distance - opposite * opposite) / (2 * u * distance)
                      for u in candidates)
    low, high = max(-1., min(values)), min(1., max(values))
    if low > high:
        return 0
    width = math.acos(low) - math.acos(high)
    return min(8 * quadrants, 4 * (math.ceil(width / (math.pi / (2 * quadrants))) + 4))


def _geodesic_point(geod, a, b, fraction):
    if fraction == 0:
        return a
    if fraction == 1:
        return b
    bearing, _, distance = geod.inv(*a, *b)
    return geod.fwd(*a, bearing, distance * fraction)[:2]


def _dense_source(points, chart, *, reference=False):
    """Subdivide source geodesics, never longitude/latitude chords."""
    work, options = chart.work, chart.options
    step = options.source_step_m / (2 if reference else 1)
    tolerance = options.source_error_m / (2 if reference else 1)
    output = []

    def append(point):
        work.charge(1, construction=True)
        output.append(point)

    def subdivide(a, b, xy_a, xy_b, depth):
        work.check()
        _, _, distance = chart.geod.inv(*a, *b)
        if not math.isfinite(distance):
            raise _MapFailure('corridor_projection_unavailable', 'Non-finite source geodesic')
        probes = [_geodesic_point(chart.geod, a, b, fraction) for fraction in (.25, .5, .75)]
        errors = [math.dist(chart.forward(probe),
                           (xy_a[0] + fraction * (xy_b[0] - xy_a[0]),
                            xy_a[1] + fraction * (xy_b[1] - xy_a[1])))
                  for fraction, probe in zip((.25, .5, .75), probes)]
        if distance > step or max(errors) > tolerance:
            if depth >= options.max_depth:
                raise _MapFailure('corridor_buffer_limit', 'Corridor source subdivision depth exceeded')
            midpoint = probes[1]
            xy_mid = chart.forward(midpoint)
            subdivide(a, midpoint, xy_a, xy_mid, depth + 1)
            subdivide(midpoint, b, xy_mid, xy_b, depth + 1)
        else:
            append(xy_b)

    append(chart.forward(points[0]))
    for a, b in zip(points, points[1:]):
        if a != b:
            subdivide(a, b, chart.forward(a), chart.forward(b), 0)
    if len(output) < 2:
        raise _MapFailure('corridor_geometry_invalid', 'Qualified source run has no line extent')
    return output


def _geographic_ring(ring, chart):
    result = []
    work, options = chart.work, chart.options

    def append(point):
        work.charge(1, construction=True)
        result.append(point)

    def subdivide(a, b, ga, gb, depth):
        errors = []
        for fraction in (.25, .5, .75):
            xy = (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
            linear = (ga[0] + fraction * (gb[0] - ga[0]), ga[1] + fraction * (gb[1] - ga[1]))
            errors.append(math.dist(chart.forward(linear), xy))
        if math.dist(a, b) > options.source_step_m or max(errors) > options.output_error_m:
            if depth >= options.max_depth:
                raise _MapFailure('corridor_buffer_limit', 'Corridor output subdivision depth exceeded')
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            gm = chart.inverse(mid)
            subdivide(a, mid, ga, gm, depth + 1)
            subdivide(mid, b, gm, gb, depth + 1)
        else:
            append(gb)

    coordinates = list(ring.coords)
    first = chart.inverse(coordinates[0])
    append(first)
    for a, b in zip(coordinates, coordinates[1:]):
        subdivide(a, b, chart.inverse(a), chart.inverse(b), 0)
    result[-1] = result[0]
    return result


def _check_chart(chart):
    """Audit local radial roundtrips and transverse scale on this ellipsoid."""
    options, geod = chart.options, chart.geod
    # For a <=25 km AEQD chart the largest scale departure is transverse.
    # Test the domain boundary in every octant and both offset directions.
    radius = options.max_chart_radius_m - options.padding_m - .001
    for bearing in range(0, 360, 45):
        center = geod.fwd(*chart.origin, bearing, radius)[:2]
        xy = chart.forward(center)
        reconstructed = chart.inverse(xy)
        if abs(geod.inv(*center, *reconstructed)[2]) > 1e-5:
            raise _MapFailure('corridor_projection_unavailable', 'Corridor projection roundtrip check failed')
        for side in (-90, 90):
            offset = geod.fwd(*center, bearing + side, options.padding_m)[:2]
            # Offset is permitted up to the padding boundary itself.
            az, _, distance = geod.inv(*chart.origin, *offset)
            offset_xy = (distance * math.sin(math.radians(az)), distance * math.cos(math.radians(az)))
            if abs(math.dist(xy, offset_xy) - options.padding_m) > .01:
                raise _MapFailure('corridor_projection_unavailable', 'Corridor projection scale check failed')


def _validate_neighborhood(candidate, reference_line, options, work):
    """Two-sided verification after geographic conversion, using finer support.

    Tests additionally compare against independent geodesic distance/capsule
    oracles. The runtime reference deliberately differs in source and arc spacing.
    """
    quadrants = max(32, math.ceil(math.pi / (4 * math.acos(1 - min(.0025, options.arc_error_m / 2) / options.padding_m))))
    inner = _buffer(reference_line, options.padding_m - .03, quadrants, work)
    outer = _buffer(reference_line, options.padding_m + .03, quadrants, work)
    for left, right in ((inner, candidate), (candidate, outer)):
        estimate = _overlay_bound([left, right], work)
        work.check()
        difference = left.difference(right)
        work.output(difference, estimate)
        if not difference.is_empty:
            raise _MapFailure('corridor_coverage_failed', 'Corridor failed its two-sided neighborhood check')
    work.native([candidate, reference_line])
    if not candidate.covers(reference_line):
        raise _MapFailure('corridor_coverage_failed', 'Corridor does not contain every qualified source interval')


def _chunk_polygons(points, geod, options, work):
    measured = MeasuredPath(geod, points, context=work.context)
    total = measured.chainage[-1]
    if not math.isfinite(total) or total <= 0:
        raise _MapFailure('corridor_geometry_invalid', 'Qualified source run has invalid extent')
    pieces = max(1, math.ceil(total / options.max_chunk_m))
    if pieces + work.charts > work.job.max_section_charts:
        raise _MapFailure('corridor_buffer_limit', 'Corridor chart count would exceed its limit')
    for index in range(pieces):
        work.chart()
        start, end = total * index / pieces, total * (index + 1) / pieces
        source = measured.span(start, end)
        chart = _Chart(measured.point((start + end) / 2), geod, options, work)
        _check_chart(chart)
        xy = _dense_source(source, chart)
        quadrants = math.ceil(math.pi / (4 * math.acos(1 - options.arc_error_m / options.padding_m)))
        if quadrants > 256:
            raise _MapFailure('corridor_buffer_limit', 'Corridor arc resolution exceeds its limit')
        shape = _buffer(LineString(xy), options.padding_m, quadrants, work)
        geographic = []
        for polygon in _parts(shape):
            outer = _geographic_ring(polygon.exterior, chart)
            holes = [_geographic_ring(ring, chart) for ring in polygon.interiors]
            converted = Polygon(outer, holes)
            back = Polygon([chart.forward(p) for p in outer],
                           [[chart.forward(p) for p in ring] for ring in holes])
            work.native([converted, back])
            if not converted.is_valid or not back.is_valid:
                raise _MapFailure('corridor_geometry_invalid', 'Corridor conversion produced invalid topology')
            geographic.append((converted, back))
        # All parts are necessary for the coverage check (loops/backtracking may
        # produce more than one polygon). No largest-part selection is permitted.
        back_shapes = [item[1] for item in geographic]
        if len(back_shapes) == 1:
            back = back_shapes[0]
        else:
            estimate = _overlay_bound(back_shapes, work)
            back = work.output(unary_union(back_shapes), estimate)
        reference = LineString(_dense_source(source, chart, reference=True))
        _validate_neighborhood(back, reference, options, work)
        yield from (item[0] for item in geographic)


def _construction_paths(points, geod, work):
    """Canonical direction and reversal cuts preserve the exact generating set.

    Splitting at an about-face avoids microscopic duplicated buffer edges caused
    by oppositely interpolated versions of the same geodesic. Both halves retain
    the turning vertex, so their round-buffer union equals the unsplit support.
    """
    cleaned = []
    for point in points:
        work.check()
        point = tuple(point)
        if not cleaned or (point != cleaned[-1] and geod.inv(*cleaned[-1], *point)[2] != 0):
            cleaned.append(point)
    points = tuple(cleaned)
    if len(points) < 2:
        raise _MapFailure('corridor_geometry_invalid', 'Qualified source run has no positive extent')
    points = min(points, points[::-1])
    start = 0
    for index in range(1, len(points) - 1):
        work.check()
        a, b, c = points[index - 1:index + 2]
        incoming = geod.inv(*a, *b)[1] + 180
        outgoing = geod.inv(*b, *c)[0]
        turn = abs((outgoing - incoming + 180) % 360 - 180)
        if turn > 179.999999:
            yield points[start:index + 1]
            start = index
    yield points[start:]


def _dateline_parts(polygons, work):
    from shapely.affinity import translate

    result = []
    for polygon in polygons:
        west, _, east, _ = polygon.bounds
        if east - west > 180:
            raise _MapFailure('corridor_projection_unavailable', 'Corridor longitude is ambiguous')
        for zone in range(math.floor((west + 180) / 360), math.floor((east + 180) / 360) + 1):
            work.check()
            boundary = box(-180 + zone * 360, -90, 180 + zone * 360, 90)
            work.native([polygon, boundary])
            if boundary.covers(polygon):
                clipped = polygon
            else:
                estimate = _overlay_bound([polygon, boundary], work)
                clipped = work.output(polygon.intersection(boundary), estimate)
            for part in _parts(clipped):
                if part.area <= 0:
                    continue
                shifted = translate(part, xoff=-zone * 360) if zone else part
                work.native([shifted])
                if not shifted.is_valid:
                    raise _MapFailure('corridor_geometry_invalid', 'Corridor dateline splitting failed')
                result.append(shifted)
                if len(result) > work.job.max_parts:
                    raise _MapFailure('corridor_buffer_limit', 'Corridor part count exceeded')
    return result


def _union_components(shapes, work):
    """Union only overlapping envelope components, in bounded native batches."""
    if not shapes:
        raise _MapFailure('corridor_geometry_invalid', 'Corridor has no polygons')
    shapes = sorted(shapes, key=lambda item: (*item.bounds, item.wkb))
    tree = STRtree(shapes)
    parents = list(range(len(shapes)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, polygon in enumerate(shapes):
        work.check()
        candidates = tree.query(polygon)
        work.charge(len(candidates))
        for other in candidates:
            if int(other) > index:
                parents[root(int(other))] = root(index)
    groups = {}
    for index in range(len(shapes)):
        groups.setdefault(root(index), []).append(shapes[index])
    result = []
    for group in groups.values():
        while len(group) > 1:
            batch = group[:work.job.max_native_operands]
            estimate = _overlay_bound(batch, work)
            merged = work.output(unary_union(batch), estimate)
            work.native([merged])
            if not merged.is_valid:
                raise _MapFailure('corridor_geometry_invalid', 'Corridor union produced invalid topology')
            group = [merged, *group[len(batch):]]
        result.extend(_parts(group[0]))
    return result


def _canonical_polygon(polygon):
    from shapely.geometry.polygon import orient

    polygon = orient(polygon, sign=1.0)

    def ring(coordinates):
        points = [tuple(map(float, p)) for p in coordinates[:-1]]
        if not all(math.isfinite(v) for p in points for v in p):
            raise _MapFailure('corridor_geometry_invalid', 'Non-finite corridor output')
        offset = min(range(len(points)), key=points.__getitem__)
        points = points[offset:] + points[:offset]
        return [list(p) for p in [*points, points[0]]]

    return dict(outer=ring(list(polygon.exterior.coords)),
                holes=sorted((ring(list(hole.coords)) for hole in polygon.interiors)))


def build_buffered_corridor(runs, *, geod, options=None, budget=None, context=None):
    """Return canonical ready/omitted polygons; never mutate runs or statistics."""
    options = options or CorridorDisplayOptions()
    budget = budget or CorridorGeometryBudget()
    work = _SectionBudget(budget, context)
    result = dict(visualization_schema_version=1, visualization_kind='qualified_path_buffer',
                  visualization_status='omitted', visualization_polygons=[], corridor_polygon=[],
                  visualization_approximation='Approximate overlap area with 5 m padding around qualifying paths. Padding does not affect mileage.',
                  diagnostics=[])
    metadata = dict(policy='qualified_path_buffer_v1', padding_m=float(options.padding_m),
                    cap_style='round', join_style='round', approximation_target_m=float(options.approximation_target_m),
                    source_runs=[], part_count=0, hole_count=0, vertex_count=0, chart_count=0)
    result['visualization_metadata'] = metadata
    try:
        work.check()
        if not (math.isfinite(options.padding_m) and .1 <= options.padding_m <= 1000
                and options.approximation_target_m == .05 and 0 < options.max_chunk_m <= 20_000
                and 0 < options.max_chart_radius_m <= 25_000 and 0 < options.source_step_m <= 100
                and 0 < options.source_error_m <= .005 and 0 < options.output_error_m <= .005
                and 0 < options.arc_error_m <= .005 and 0 <= options.max_depth <= 20):
            raise _MapFailure('corridor_projection_unavailable', 'Unsupported corridor display precision/domain options')
        if not runs:
            raise _MapFailure('corridor_geometry_invalid', 'Corridor has no qualified source runs')
        source_points = sum(len(run.coordinates) for run in runs)
        work.charge(source_points, construction=True)
        metadata['source_runs'] = [run.provenance() for run in runs]
        polygons = []
        for run in runs:
            for point in run.coordinates:
                if len(point) != 2 or not all(math.isfinite(value) for value in point) or not (-180 <= point[0] <= 180 and -90 < point[1] < 90):
                    raise _MapFailure('corridor_geometry_invalid', 'Invalid qualified source coordinates')
            for coordinates in _construction_paths(run.coordinates, geod, work):
                polygons.extend(_chunk_polygons(coordinates, geod, options, work))
            if len(polygons) > budget.max_parts:
                raise _MapFailure('corridor_buffer_limit', 'Corridor intermediate part count exceeded')
        polygons = _union_components(_dateline_parts(polygons, work), work)
        if len(polygons) > budget.max_parts or sum(1 + len(p.interiors) for p in polygons) > budget.max_rings:
            raise _MapFailure('corridor_buffer_limit', 'Corridor output part/ring count exceeded')
        canonical = sorted((_canonical_polygon(p) for p in polygons), key=lambda p: p['outer'])
        count = sum(len(p['outer']) + sum(map(len, p['holes'])) for p in canonical)
        if count > budget.max_section_points:
            raise _MapFailure('corridor_buffer_limit', 'Corridor section output point limit exceeded')
        budget.retain(count)
        result.update(visualization_status='ready', visualization_polygons=canonical)
        if len(canonical) == 1 and not canonical[0]['holes']:
            result['corridor_polygon'] = canonical[0]['outer']
        metadata.update(part_count=len(canonical), hole_count=sum(len(p['holes']) for p in canonical), vertex_count=count)
    except AnalysisCancelled:
        raise
    except Exception as error:
        result['diagnostics'].append(dict(level='warning', code=getattr(error, 'code', 'corridor_geometry_invalid'),
                                         message='A corridor map could not be generated; mileage is unaffected.',
                                         context=dict(stage='buffer', error=str(error), chart_count=work.charts,
                                                      processed_vertices=budget.work_vertices)))
    metadata['chart_count'] = work.charts
    return result
